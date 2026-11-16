//! `pipeline.py:MultiViewPipeline` and the multi-view branch of `api.py:predict`
//! (`animalpose predict --views ... --cameras cams.json`).
//!
//! Synchronised cameras: per-view tracking with the single-view [`Pipeline`], cross-view identity
//! fusion, robust triangulation, then optional 3D temporal smoothing and 3D behavior. The run
//! writes what Python writes: `animals3d.npz` (`animal{g}_{name}` arrays) and one
//! `view{v}_tracks.csv` per view.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};

use super::camera::{load_cameras, Camera};
use super::dense::Vec3;
use super::fusion::{fuse_identities, triangulate_identities, Animal3d, Identities, ViewTrack};
use super::npz::{write_npz, Member};
use super::triangulation::nan_to_num;
use crate::algos::background::median_background;
use crate::algos::image::{GrayF, Image};
use crate::cli::predict::{Pipeline, PipelineCfg, Progress, VideoOut};
use crate::cli::smooth::smooth_sequence;
use crate::data::config::Config;
use crate::data::images::{read_frames_with_checkpoint, u8_from_gray};
use crate::data::sessions::{NpyArray, NpyData};
use crate::models::Model;
use crate::tensor::{Ctx, Graph};

/// One fused identity: [`Animal3d`] plus what the 3D models add.
#[derive(Clone, Debug, PartialEq)]
pub struct Animal {
    /// `keypoints3d`, `reproj_error`, `n_views`.
    pub tri: Animal3d,
    /// `smoothed3d (T, K, 3)` from the 3D temporal model.
    pub smoothed3d: Option<Vec<f64>>,
    /// `confidence3d (T, K)`.
    pub confidence3d: Option<Vec<f64>>,
    /// `imputed (T, K)`.
    pub imputed: Option<Vec<f64>>,
    /// `behavior_probs (T, C)` from the 3D behavior model.
    pub behavior_probs: Option<Vec<f64>>,
    /// `behavior (T,)`, the argmax of those.
    pub behavior: Option<Vec<i64>>,
}

/// `MultiViewPipeline.run`'s dict: `identities`, `animals`, `per_view`.
#[derive(Clone, Debug)]
pub struct MultiViewResult {
    /// `(view, track_id) -> global id`.
    pub identities: Identities,
    /// Global id -> 3D animal.
    pub animals: BTreeMap<usize, Animal>,
    /// The single-view tracking results.
    pub per_view: Vec<VideoOut>,
}

/// `MultiViewPipeline(pipeline, cameras, temporal3d, behavior3d, fusion_max_cost, reproj_thresh,
/// max_animals, resolve_left_right)`.
pub struct MultiViewPipeline {
    /// One camera per view.
    pub cameras: Vec<Camera>,
    /// The optional 3D temporal model (`dims: 3`).
    pub temporal3d: Option<Model>,
    /// The optional 3D behavior model (`dims: 3`) and its classes.
    pub behavior3d: Option<(Model, Vec<String>)>,
    /// `fusion_max_cost` (px).
    pub fusion_max_cost: f64,
    /// `reproj_thresh` (px).
    pub reproj_thresh: f64,
    /// `max_animals`.
    pub max_animals: Option<usize>,
    /// `resolve_left_right`.
    pub resolve_left_right: bool,
    /// `pipe.cfg.kp_conf_thresh`.
    pub kp_conf_thresh: f64,
    /// `pipe.cfg.temporal_window`.
    pub temporal_window: usize,
    /// `smooth_sequence`'s `hop`.
    pub temporal_hop: usize,
}

impl MultiViewPipeline {
    /// The defaults of the Python constructor, reading `kp_conf_thresh` and `temporal_window`
    /// from the single-view pipeline's config.
    pub fn new(cameras: Vec<Camera>, cfg: &PipelineCfg) -> Self {
        Self {
            cameras,
            temporal3d: None,
            behavior3d: None,
            fusion_max_cost: 15.0,
            reproj_thresh: 10.0,
            max_animals: None,
            resolve_left_right: false,
            kp_conf_thresh: cfg.kp_conf_thresh,
            temporal_window: cfg.temporal_window,
            temporal_hop: cfg.temporal_hop,
        }
    }

    /// `MultiViewPipeline.run(views)`: `pipe.track` per view, then [`Self::run_tracks`].
    ///
    /// As `PosePipeline.track` does when it is handed no background, a median background is taken
    /// from each view's own frames when the config needs one.
    pub fn run(&self, pipe: &Pipeline, views: &[Vec<GrayF>]) -> Result<MultiViewResult> {
        self.run_with_progress(pipe, views, &mut |_| Ok(()))
    }

    fn run_with_progress(
        &self,
        pipe: &Pipeline,
        views: &[Vec<GrayF>],
        progress: &mut impl FnMut(Progress) -> Result<()>,
    ) -> Result<MultiViewResult> {
        ensure!(
            views.len() == self.cameras.len(),
            "{} views but {} cameras",
            views.len(),
            self.cameras.len()
        );
        let mut per_view = Vec::with_capacity(views.len());
        let total = views.iter().map(Vec::len).sum();
        let mut completed = 0;
        for frames in views {
            progress(Progress::Tracking {
                done: completed,
                total,
            })?;
            let background = pipe.cfg.needs_background().then(|| {
                let samples: Vec<Image> = frames.iter().map(u8_from_gray).collect();
                median_background(&samples, pipe.cfg.background_frames)
            });
            per_view.push(pipe.track_with_progress(
                frames.iter().map(|f| Ok(crate::data::images::to_bgr8(f))),
                background.as_ref(),
                |done| {
                    progress(Progress::Tracking {
                        done: completed + done,
                        total,
                    })
                },
            )?);
            completed += frames.len();
        }
        progress(Progress::Refining)?;
        self.run_tracks(per_view)
    }

    /// Everything after per-view tracking: fusion, triangulation and the 3D models.
    pub fn run_tracks(&self, per_view: Vec<VideoOut>) -> Result<MultiViewResult> {
        ensure!(
            per_view.len() == self.cameras.len(),
            "{} views but {} cameras",
            per_view.len(),
            self.cameras.len()
        );
        let n_frames = per_view.iter().map(|r| r.n_frames).max().unwrap_or(0);
        let mut vtracks = Vec::new();
        for (v, r) in per_view.iter().enumerate() {
            for (tid, tr) in &r.tracks {
                let k = tr.conf.len() / tr.frames.len().max(1);
                vtracks.push(ViewTrack {
                    view: v,
                    track_id: *tid,
                    frames: tr.frames.clone(),
                    keypoints: tr
                        .keypoints
                        .chunks(2 * k.max(1))
                        .map(|row| row.chunks_exact(2).map(|p| [p[0], p[1]]).collect())
                        .collect(),
                    conf: tr.conf.chunks(k.max(1)).map(<[f64]>::to_vec).collect(),
                });
            }
        }
        let identities = fuse_identities(
            &vtracks,
            &self.cameras,
            self.fusion_max_cost,
            self.max_animals,
        );
        let tri = if vtracks.is_empty() {
            BTreeMap::new()
        } else {
            triangulate_identities(
                &vtracks,
                &self.cameras,
                &identities,
                n_frames,
                self.kp_conf_thresh,
                self.reproj_thresh,
                self.resolve_left_right,
            )
        };
        let mut animals = BTreeMap::new();
        for (gid, a) in tri {
            let animal = self.animal_3d(a)?;
            animals.insert(gid, animal);
        }
        Ok(MultiViewResult {
            identities,
            animals,
            per_view,
        })
    }

    /// The 3D temporal model and the 3D behavior model over one triangulated identity.
    fn animal_3d(&self, tri: Animal3d) -> Result<Animal> {
        let t = tri.keypoints3d.len();
        let k = tri.keypoints3d.first().map_or(0, Vec::len);
        let x: Vec<f64> = tri
            .keypoints3d
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let obs: Vec<f64> = tri
            .keypoints3d
            .iter()
            .flatten()
            .map(|p| f64::from(p.iter().all(|v| v.is_finite())))
            .collect();
        let mut animal = Animal {
            tri,
            smoothed3d: None,
            confidence3d: None,
            imputed: None,
            behavior_probs: None,
            behavior: None,
        };
        if let Some(model) = &self.temporal3d {
            if obs.iter().any(|o| *o > 0.0) && t > 0 {
                let conf: Vec<f64> = animal
                    .tri
                    .reproj_error
                    .iter()
                    .flatten()
                    .zip(&obs)
                    .map(|(e, o)| {
                        if *o > 0.0 {
                            let e = if e.is_nan() { 1e3 } else { nan_to_num(*e) };
                            // `.astype(np.float32)`
                            f64::from((1.0 / (1.0 + e / self.reproj_thresh)) as f32)
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let coords: Vec<f64> = x.iter().map(|v| nan_to_num(*v)).collect();
                let s = smooth_sequence(
                    model,
                    &coords,
                    &conf,
                    &obs,
                    t,
                    self.temporal_window,
                    self.temporal_hop,
                )?;
                animal.smoothed3d = Some(s.coords);
                animal.confidence3d = Some(s.confidence);
                animal.imputed = Some(s.imputed);
            }
        }
        if let Some((model, classes)) = &self.behavior3d {
            if t > 0 {
                let (pts, mask): (Vec<f64>, Vec<f64>) = match &animal.smoothed3d {
                    Some(s) => (s.clone(), vec![1.0; t * k]),
                    None => (x.iter().map(|v| nan_to_num(*v)).collect(), obs.clone()),
                };
                let mut g = Graph::new();
                let input = g.from_data(pts.iter().map(|v| *v as f32).collect(), &[1, t, k, 3]);
                let mask_node = g.from_data(mask.iter().map(|v| *v as f32).collect(), &[1, t, k]);
                let mut ctx = Ctx::eval();
                let out = model.forward_named(&mut ctx, &mut g, &[input, mask_node])?;
                let logits_node = out
                    .iter()
                    .find(|(name, _)| *name == "logits")
                    .map(|(_, n)| *n)
                    .context("the behavior model returned no `logits`")?;
                let logits = g.to_vec(logits_node);
                let c = classes.len().max(1);
                ensure!(
                    logits.len() == t * c,
                    "the behavior model returned {} logits for {t} frames x {c} classes",
                    logits.len()
                );
                let mut probs = vec![0.0; t * c];
                let mut best = vec![0i64; t];
                for f in 0..t {
                    let row = &logits[f * c..(f + 1) * c];
                    let m = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
                    let e: Vec<f64> = row.iter().map(|v| f64::from(*v - m).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    let mut arg = 0;
                    for (j, ej) in e.iter().enumerate() {
                        probs[f * c + j] = ej / sum;
                        if probs[f * c + j] > probs[f * c + arg] {
                            arg = j;
                        }
                    }
                    best[f] = arg as i64;
                }
                animal.behavior_probs = Some(probs);
                animal.behavior = Some(best);
            }
        }
        Ok(animal)
    }
}

impl MultiViewResult {
    /// `np.savez_compressed(out / "animals3d.npz", **{f"animal{g}_{k}": v ...})`, in the dict's
    /// order: `keypoints3d`, `reproj_error`, `n_views`, `smoothed3d`, `confidence3d`, `imputed`,
    /// `behavior_probs`, `behavior`.
    pub fn to_npz(&self, path: &Path) -> Result<()> {
        let mut members = Vec::new();
        for (g, a) in &self.animals {
            let t = a.tri.keypoints3d.len();
            let k = a.tri.keypoints3d.first().map_or(0, Vec::len);
            let x: Vec<f64> = a
                .tri
                .keypoints3d
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect();
            members.push((
                format!("animal{g}_keypoints3d"),
                Member::F64(vec![t, k, 3], x),
            ));
            members.push((
                format!("animal{g}_reproj_error"),
                Member::F64(
                    vec![t, k],
                    a.tri.reproj_error.iter().flatten().copied().collect(),
                ),
            ));
            members.push((
                format!("animal{g}_n_views"),
                Member::Npy(NpyArray {
                    shape: vec![t, k],
                    data: NpyData::Int(a.tri.n_views.iter().flatten().copied().collect()),
                }),
            ));
            let f32s = |shape: Vec<usize>, v: &[f64]| {
                Member::Npy(NpyArray {
                    shape,
                    data: NpyData::Float(v.iter().map(|x| *x as f32).collect()),
                })
            };
            if let Some(s) = &a.smoothed3d {
                members.push((format!("animal{g}_smoothed3d"), f32s(vec![t, k, 3], s)));
            }
            if let Some(c) = &a.confidence3d {
                members.push((format!("animal{g}_confidence3d"), f32s(vec![t, k], c)));
            }
            if let Some(i) = &a.imputed {
                members.push((format!("animal{g}_imputed"), f32s(vec![t, k], i)));
            }
            if let Some(p) = &a.behavior_probs {
                let c = p.len() / t.max(1);
                members.push((format!("animal{g}_behavior_probs"), f32s(vec![t, c], p)));
            }
            if let Some(b) = &a.behavior {
                members.push((
                    format!("animal{g}_behavior"),
                    Member::Npy(NpyArray {
                        shape: vec![t],
                        data: NpyData::Int(b.clone()),
                    }),
                ));
            }
        }
        write_npz(path, &members)
    }

    /// The files `api.predict` writes for a multi-view run: `animals3d.npz` and
    /// `view{v}_tracks.csv`.
    pub fn write(&self, out: &Path) -> Result<()> {
        self.to_npz(&out.join("animals3d.npz"))?;
        for (v, r) in self.per_view.iter().enumerate() {
            r.to_csv(&out.join(format!("view{v}_tracks.csv")))?;
        }
        Ok(())
    }

    /// One identity's keypoints as `(T, K, 3)` points.
    pub fn keypoints3d(&self, gid: usize) -> Option<&Vec<Vec<Vec3>>> {
        self.animals.get(&gid).map(|a| &a.tri.keypoints3d)
    }
}

/// `animalpose predict --views ... --cameras cams.json`: the multi-view branch of `api.predict`.
pub fn run_cli(
    config: &str,
    views: &[String],
    cameras: &str,
    out: &str,
    max_frames: Option<usize>,
    overrides: &[String],
) -> Result<()> {
    run_cli_with_progress(config, views, cameras, out, max_frames, overrides, |_| {
        Ok(())
    })
}

pub(crate) fn run_cli_with_progress(
    config: &str,
    views: &[String],
    cameras: &str,
    out: &str,
    max_frames: Option<usize>,
    overrides: &[String],
    mut progress: impl FnMut(Progress) -> Result<()>,
) -> Result<()> {
    progress(Progress::Preparing)?;
    let cfg = Config::load(config, overrides)?;
    let out_dir = PathBuf::from(out);
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("cannot create {}", out_dir.display()))?;
    let pipe = Pipeline::from_config(&cfg)?;
    let cams = load_cameras(cameras)?;
    ensure!(
        cams.len() == views.len(),
        "{cameras} holds {} cameras for {} views",
        cams.len(),
        views.len()
    );
    let mut frames = Vec::with_capacity(views.len());
    for v in views {
        progress(Progress::Preparing)?;
        let (f, _fps) =
            read_frames_with_checkpoint(v, max_frames, 1, || progress(Progress::Preparing))?;
        frames.push(f);
    }
    let mv = MultiViewPipeline::new(cams, &pipe.cfg);
    let res = mv.run_with_progress(&pipe, &frames, &mut progress)?;
    progress(Progress::WritingTracks)?;
    res.write(&out_dir)?;
    println!(
        "{} animals fused across {} views -> {}",
        res.animals.len(),
        views.len(),
        out_dir.display()
    );
    Ok(())
}
