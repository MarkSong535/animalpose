//! `animalpose predict`: detection, pose, tracking, temporal refinement and behavior over a folder
//! of frames — the port of `src/animalpose/pipeline.py:PosePipeline` plus the writers of the Python
//! command of the same name.
//!
//! `// differs from Python:`
//! * Videos decode and `overlay.mp4` encodes through the `ffmpeg` executable
//!   ([`crate::data::video`]) instead of OpenCV's bundled codecs; frames are streamed, never held
//!   in memory all at once, as in Python.
//! * Crops go through the pose network one at a time instead of `batch_crops` at a time. Inference
//!   runs in eval mode, where batch norm uses its running statistics and dropout is off, so a batch
//!   makes no difference to the numbers.
//! * Crop warping samples with exact bilinear weights and zero-fills every source pixel outside the
//!   frame, where `cv2.warpAffine` uses fixed-point weights and re-normalises a pixel that lies
//!   partly outside; crop border pixels can differ by up to one grey level.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::algos::background::foreground_fraction;
use crate::algos::blob::BlobProposer;
use crate::algos::decode::{
    detector_decode, pose_decode, pose_decode_pairs, HeatMaps, DEFAULT_PAIR_MIN_PEAK,
    DEFAULT_PAIR_RADIUS,
};
use crate::algos::geometry::{
    apply_affine, box_xyxy_to_cs, crop_affine, invert_affine, unletterbox_boxes, Box4,
};
use crate::algos::image::{letterbox, Bgr, GrayF};
use crate::algos::kinematics::{body_kinematics, Kinematics};
use crate::algos::nms::merge_candidates;
use crate::algos::postprocess::{adapt_prior, median_smooth};
use crate::algos::tracker::{Detection, MultiAnimalTracker, TrackerConfig};
use crate::data::config::Config;
use crate::data::frames::{timing_report, Source};
use crate::data::images::{to_bgr8, to_tensor};
use crate::data::sessions::{write_npz, NpyArray, NpyData};
use crate::data::synthetic::{canonical_mask, BILATERAL_PAIRS, CANONICAL_NAMES, NUM_KEYPOINTS};
use crate::data::video::VideoWriter;
use crate::models::config::ModelConfig;
use crate::models::{Model, ModelKind};
use crate::tensor::{Ctx, Graph, NodeId};
use anyhow::{bail, Context, Result};

use crate::cli::catalog::Catalog;
use crate::cli::pairs::resolve_pair_identity;
use crate::cli::smooth::smooth_sequence;
use crate::cli::train::repo_path;

/// `pipeline.py:PipelineConfig` with the same defaults and the same config keys.
#[derive(Clone, Debug)]
pub struct PipelineCfg {
    /// Network input size `(H, W)`.
    pub det_size: (u32, u32),
    /// Detector score a box must reach.
    pub det_thresh: f64,
    /// Boxes kept per frame.
    pub max_det: usize,
    /// Side of the square pose crop in pixels.
    pub crop_size: u32,
    /// `box_xyxy_to_cs` padding of that crop.
    pub crop_padding: f64,
    /// Keypoint confidence a point needs to count as observed.
    pub kp_conf_thresh: f64,
    /// Keep bilateral pairs on distinct heat-map peaks.
    pub pair_decoding: bool,
    /// Viterbi left/right identity per track.
    pub pair_identity: bool,
    /// Paw motion that costs as much as one unit of pair margin.
    pub pair_motion_frac: f64,
    /// Drop boxes that match the median background.
    pub background_filter: bool,
    /// Frames spread over the recording used for that background.
    pub background_frames: usize,
    /// Grey levels that count as "different from background".
    pub foreground_diff: f32,
    /// Minimum share of box pixels that must differ.
    pub min_foreground: f64,
    /// Candidate sources: `network` and/or `blob`.
    pub detector_candidates: Vec<String>,
    /// Score given to blob boxes.
    pub blob_score: f64,
    /// Grey levels for the blob difference; `None` reuses `foreground_diff`.
    pub blob_diff: Option<f32>,
    /// Morphological opening that removes the tail from a blob.
    pub blob_open_px: u32,
    /// Restrict blob search to the bright floor of the background.
    pub blob_arena_roi: bool,
    /// Plausible blob area in px^2; `None` learns it.
    pub blob_area: Option<(f64, f64)>,
    /// Blobs that calibrate that area; 0 keeps `blob_area` as-is.
    pub blob_calibrate: usize,
    /// IoU at which a candidate is dropped as a duplicate.
    pub merge_iou: f64,
    /// Resample the track onto this frame rate before classification.
    pub behavior_fps: Option<f64>,
    /// Keypoint confidence the classifier's mask requires.
    pub behavior_conf: f64,
    /// Restrict the classifier to one dataset's keypoints.
    pub behavior_keypoints: Option<String>,
    /// Top-k behaviors written to `behavior.csv`.
    pub behavior_top_k: usize,
    /// Moving-median window over the classifier's probabilities; 1 disables it.
    pub behavior_smooth: usize,
    /// Re-estimate the class prior on this recording.
    pub behavior_prior_adapt: bool,
    /// How far a class prior may move.
    pub behavior_prior_max_ratio: f64,
    /// `model`, `color` or `none`.
    pub embedder: String,
    /// Temporal window in frames.
    pub temporal_window: usize,
    /// Hop between the temporal windows; `smooth_sequence`'s `hop: int = 32` default, which
    /// `refine` never overrides.
    pub temporal_hop: usize,
}

impl Default for PipelineCfg {
    fn default() -> Self {
        Self {
            det_size: (256, 320),
            det_thresh: 0.3,
            max_det: 20,
            crop_size: 128,
            crop_padding: 1.25,
            kp_conf_thresh: 0.3,
            pair_decoding: true,
            pair_identity: true,
            pair_motion_frac: 0.1,
            background_filter: false,
            background_frames: 64,
            foreground_diff: 25.0,
            min_foreground: 0.08,
            detector_candidates: vec!["network".to_string()],
            blob_score: 0.9,
            blob_diff: None,
            blob_open_px: 5,
            blob_arena_roi: true,
            blob_area: None,
            blob_calibrate: 64,
            merge_iou: 0.5,
            behavior_fps: None,
            behavior_conf: 0.6,
            behavior_keypoints: None,
            behavior_top_k: 3,
            behavior_smooth: 5,
            behavior_prior_adapt: true,
            behavior_prior_max_ratio: 4.0,
            embedder: "model".to_string(),
            temporal_window: 64,
            temporal_hop: 32,
        }
    }
}

impl PipelineCfg {
    /// The `pipeline:` block of a config over the defaults above.
    pub fn from_config(cfg: &Config) -> Self {
        let mut out = Self::default();
        let size = cfg.f64_list_at("pipeline.det_size", &[256.0, 320.0]);
        if size.len() == 2 {
            out.det_size = (size[0] as u32, size[1] as u32);
        }
        out.det_thresh = cfg.f64_at("pipeline.det_thresh", out.det_thresh);
        out.max_det = cfg.usize_at("pipeline.max_det", out.max_det);
        out.crop_size = cfg.usize_at("pipeline.crop_size", out.crop_size as usize) as u32;
        out.crop_padding = cfg.f64_at("pipeline.crop_padding", out.crop_padding);
        out.kp_conf_thresh = cfg.f64_at("pipeline.kp_conf_thresh", out.kp_conf_thresh);
        out.pair_decoding = cfg.bool_at("pipeline.pair_decoding", out.pair_decoding);
        out.pair_identity = cfg.bool_at("pipeline.pair_identity", out.pair_identity);
        out.pair_motion_frac = cfg.f64_at("pipeline.pair_motion_frac", out.pair_motion_frac);
        out.background_filter = cfg.bool_at("pipeline.background_filter", out.background_filter);
        out.background_frames = cfg.usize_at("pipeline.background_frames", out.background_frames);
        out.foreground_diff =
            cfg.f64_at("pipeline.foreground_diff", f64::from(out.foreground_diff)) as f32;
        out.min_foreground = cfg.f64_at("pipeline.min_foreground", out.min_foreground);
        let candidates = cfg.str_list_at("pipeline.detector_candidates");
        if !candidates.is_empty() {
            out.detector_candidates = candidates;
        }
        out.blob_score = cfg.f64_at("pipeline.blob_score", out.blob_score);
        out.blob_diff = cfg
            .get("pipeline.blob_diff")
            .filter(|v| !v.is_null())
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        out.blob_open_px = cfg.usize_at("pipeline.blob_open_px", out.blob_open_px as usize) as u32;
        out.blob_arena_roi = cfg.bool_at("pipeline.blob_arena_roi", out.blob_arena_roi);
        let area = cfg.f64_list_at("pipeline.blob_area", &[]);
        out.blob_area = if area.len() == 2 {
            Some((area[0], area[1]))
        } else {
            None
        };
        out.blob_calibrate = cfg.usize_at("pipeline.blob_calibrate", out.blob_calibrate);
        out.merge_iou = cfg.f64_at("pipeline.merge_iou", out.merge_iou);
        out.behavior_fps = cfg
            .get("pipeline.behavior_fps")
            .filter(|v| !v.is_null())
            .and_then(|v| v.as_f64());
        out.behavior_conf = cfg.f64_at("pipeline.behavior_conf", out.behavior_conf);
        out.behavior_keypoints = cfg
            .get("pipeline.behavior_keypoints")
            .filter(|v| !v.is_null())
            .and_then(|v| v.as_str())
            .map(str::to_string);
        out.behavior_top_k = cfg.usize_at("pipeline.behavior_top_k", out.behavior_top_k);
        out.behavior_smooth = cfg.usize_at("pipeline.behavior_smooth", out.behavior_smooth);
        out.behavior_prior_adapt =
            cfg.bool_at("pipeline.behavior_prior_adapt", out.behavior_prior_adapt);
        out.behavior_prior_max_ratio = cfg.f64_at(
            "pipeline.behavior_prior_max_ratio",
            out.behavior_prior_max_ratio,
        );
        out.embedder = cfg.str_at("pipeline.embedder", &out.embedder);
        out.temporal_window = cfg.usize_at("pipeline.temporal_window", out.temporal_window);
        out
    }

    /// `track`'s `need_bg`: a background filter or blob candidates need a background image.
    pub fn needs_background(&self) -> bool {
        self.background_filter || self.detector_candidates.iter().any(|c| c == "blob")
    }
}

/// A checkpoint: its model config, the network it describes and the heat-map stride.
struct Net {
    model: Model,
    cfg: ModelConfig,
    /// The heat-map stride, 1 for a network that has none.
    stride: usize,
}

/// The pipeline: four networks and the configuration that wires them together.
pub struct Pipeline {
    /// The `pipeline:` block these networks run under.
    pub cfg: PipelineCfg,
    tracker_cfg: TrackerConfig,
    detector: Net,
    pose: Net,
    temporal: Option<Model>,
    behavior: Option<(Model, Vec<String>)>,
    catalog: Catalog,
    #[cfg_attr(not(all(feature = "mps", target_os = "macos")), allow(dead_code))]
    gpu: GpuCache,
}

/// One animal in one live frame: its track id, box, keypoints and their confidences.
#[derive(Clone, Debug)]
pub struct LiveAnimal {
    pub id: i64,
    pub bbox: Box4,
    pub keypoints: Vec<[f64; 2]>,
    pub conf: Vec<f64>,
}

/// `PosePipeline.track` one frame at a time, for live cameras: detection, pose and association with
/// identities that persist across frames. The temporal and behavior passes need a whole track and
/// run when a recording is analyzed.
pub struct LiveTracker {
    pipe: Pipeline,
    tracker: MultiAnimalTracker,
}

impl LiveTracker {
    /// A fresh tracker over `pipe`'s networks.
    pub fn new(pipe: Pipeline) -> Self {
        let tracker = MultiAnimalTracker::new(pipe.tracker_cfg.clone());
        Self { pipe, tracker }
    }

    /// Detects, estimates and associates the animals of one frame.
    pub fn step(&mut self, frame: &Bgr) -> Result<Vec<LiveAnimal>> {
        let boxes = self.pipe.detect(frame, frame.width, frame.height)?;
        let detections = self.pipe.frame_detections_bgr(frame, &boxes.0, &boxes.1)?;
        Ok(self
            .tracker
            .update(&detections)
            .into_iter()
            .map(|(id, i)| {
                let d = &detections[i];
                LiveAnimal {
                    id,
                    bbox: d.bbox,
                    keypoints: d.keypoints.clone().unwrap_or_default(),
                    conf: d.kp_conf.clone().unwrap_or_default(),
                }
            })
            .collect())
    }
}

/// A CPU forward pass: the tape, its input node and the named outputs.
type Recorded = (Graph, NodeId, Vec<(&'static str, NodeId)>);

/// Captured GPU forward passes, one per network and input shape (the `mps` feature): the first
/// frame runs on the CPU and is recorded, later frames replay the recording on the GPU.
#[derive(Default)]
struct GpuCache {
    #[cfg(all(feature = "mps", target_os = "macos"))]
    #[allow(clippy::type_complexity)]
    nets: std::sync::Mutex<
        BTreeMap<
            (usize, Vec<usize>),
            Option<(crate::tensor::mpsgraph::GpuForward, Vec<&'static str>)>,
        >,
    >,
}

/// One track's per-frame output, `pipeline.py:TrackResult`.
#[derive(Clone, Debug, Default)]
pub struct TrackOut {
    /// The tracker's identity.
    pub id: i64,
    /// Frame indices the track covers, contiguous.
    pub frames: Vec<usize>,
    /// Boxes `(T, 4)` xyxy, NaN where the tracker had no observation.
    pub boxes: Vec<f64>,
    /// Keypoints `(T, K, 2)` as the network decoded them, NaN where unmatched.
    pub keypoints: Vec<f64>,
    /// Keypoint confidences `(T, K)`.
    pub conf: Vec<f64>,
    /// Per-frame left/right evidence `(T, pairs.len())`.
    pub pair_margin: Option<Vec<f64>>,
    /// The pairs the temporal pass relabelled `(T, pairs.len())`.
    pub pair_swapped: Option<Vec<bool>>,
    /// Temporal-model coordinates `(T, K, 2)`.
    pub smoothed: Option<Vec<f64>>,
    /// Temporal-model confidence `(T, K)`.
    pub smoothed_conf: Option<Vec<f64>>,
    /// `1 - observed` `(T, K)`: 1 where the temporal model filled a gap.
    pub imputed: Option<Vec<f64>>,
    /// Per-frame catalogue index.
    pub behavior: Option<Vec<i64>>,
    /// Classifier probabilities `(T, C)`.
    pub behavior_probs: Option<Vec<f64>>,
    /// Catalogue probabilities `(T, len(CATALOG))`.
    pub behavior_full: Option<Vec<f64>>,
}

impl TrackOut {
    /// `TrackResult.best`: the smoothed keypoints when there are any.
    fn best(&self) -> &[f64] {
        self.smoothed.as_deref().unwrap_or(&self.keypoints)
    }

    /// The confidence the writers print.
    fn best_conf(&self) -> &[f64] {
        self.smoothed_conf.as_deref().unwrap_or(&self.conf)
    }

    /// Frames this track spans; `refine`'s `min_len` test uses it.
    fn span(&self) -> usize {
        self.frames.len()
    }
}

/// The whole recording, `pipeline.py:VideoResult`.
#[derive(Clone, Debug, Default)]
pub struct VideoOut {
    /// Tracks by their id, in the order the tracker created them.
    pub tracks: BTreeMap<i64, TrackOut>,
    /// Frames read.
    pub n_frames: usize,
    /// The behavior class names: the classifier's classes, then the catalogue's after `refine`.
    pub behaviors: Vec<String>,
    /// Presentation time per frame in seconds.
    pub times: Vec<f64>,
    /// `catalog.describe()` output: which behaviors this run models.
    pub behavior_info: Option<serde_json::Value>,
}

/// `animalpose predict` for a video file or a folder of frames, writing `overlay.mp4` too.
pub fn run(
    config: &str,
    frames: &str,
    out: &str,
    max_frames: Option<usize>,
    fps: Option<f64>,
    overrides: &[String],
) -> Result<()> {
    run_with(config, frames, out, max_frames, fps, overrides, true)
}

/// `api.py:predict` for one recording: stream the frames through the pipeline, then write
/// `tracks.npz`, `tracks.csv`, `timing.json`, `behavior.csv` + `behavior_catalog.json` (when a
/// classifier ran) and, with `overlay`, `overlay.mp4`.
pub fn run_with(
    config: &str,
    frames: &str,
    out: &str,
    max_frames: Option<usize>,
    fps: Option<f64>,
    overrides: &[String],
    overlay: bool,
) -> Result<()> {
    let cfg = Config::load(config, overrides)?;
    let pipeline = Pipeline::from_config(&cfg)?;
    let out_dir = PathBuf::from(out);
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("cannot create {}", out_dir.display()))?;

    let source = Source::open(frames)?;
    let times = source.timestamps(max_frames)?;
    let timing = timing_report(&times);
    std::fs::write(
        out_dir.join("timing.json"),
        serde_json::to_string_pretty(&timing)?,
    )?;
    if timing["variable"].as_bool() == Some(true) {
        println!("variable frame rate / dropped frames: {timing}; using real timestamps");
    }
    let fps = fps
        .or_else(|| timing["mean_fps"].as_f64())
        .or_else(|| source.fps())
        .unwrap_or(30.0);
    println!(
        "{} ({}): {} frames at {fps:.3} fps",
        source.path().display(),
        source.kind(),
        source.n_frames().min(max_frames.unwrap_or(usize::MAX))
    );
    let background = if pipeline.cfg.needs_background() {
        Some(source.sampled_background(64, max_frames)?)
    } else {
        None
    };
    let result = pipeline.run_stream(
        source.iter(max_frames)?,
        background.as_ref(),
        fps,
        Some(&times),
    )?;

    result.to_npz(&out_dir.join("tracks.npz"))?;
    result.to_csv(&out_dir.join("tracks.csv"))?;
    if result.behavior_info.is_some() {
        result.to_behavior_csv(&out_dir.join("behavior.csv"), pipeline.cfg.behavior_top_k)?;
        let info = result
            .behavior_info
            .clone()
            .unwrap_or_else(|| serde_json::json!([]));
        std::fs::write(
            out_dir.join("behavior_catalog.json"),
            serde_json::to_string_pretty(&info)?,
        )?;
    }
    println!(
        "{} frames, {} tracks -> {}",
        result.n_frames,
        result.tracks.len(),
        out_dir.display()
    );
    if overlay {
        if crate::data::video::ffmpeg_available() {
            let mut writer = VideoWriter::new(out_dir.join("overlay.mp4"), fps);
            for (t, frame) in source.iter(max_frames)?.enumerate() {
                writer.write(&crate::cli::overlay::draw_frame(&frame?, &result, t))?;
            }
            let written = writer.finish()?;
            println!("overlay.mp4: {written} frames");
        } else {
            println!("overlay.mp4 skipped: ffmpeg is not installed");
        }
    }
    Ok(())
}

impl Pipeline {
    /// `PosePipeline.from_checkpoints`: every checkpoint named by `checkpoints:`.
    pub fn from_config(cfg: &Config) -> Result<Self> {
        let pipeline_cfg = PipelineCfg::from_config(cfg);
        // `device:` as in Python: `cpu` keeps everything on the CPU, `auto` / `mps` use the GPU when
        // this build has one (the `mps` feature).
        crate::tensor::set_gpu(cfg.str_at("device", "auto") != "cpu");
        let detector = load_net(cfg, "checkpoints.detector", ModelKind::Detector)?;
        let pose = load_net(cfg, "checkpoints.pose", ModelKind::Pose)?;
        let temporal = match cfg.get("checkpoints.temporal").and_then(|v| v.as_str()) {
            Some(_) => Some(load_net(cfg, "checkpoints.temporal", ModelKind::Temporal)?.model),
            None => None,
        };
        let behavior = match cfg.get("checkpoints.behavior").and_then(|v| v.as_str()) {
            Some(_) => {
                let net = load_net(cfg, "checkpoints.behavior", ModelKind::Behavior)?;
                // `BehaviorTCN.classes`, which is what `VideoResult.behaviors` holds before `refine`.
                Some((net.model, net.cfg.classes()))
            }
            None => None,
        };
        match pipeline_cfg.embedder.as_str() {
            "model" | "color" | "none" => {}
            other => bail!("pipeline.embedder must be model, color or none, got {other:?}"),
        }
        Ok(Self {
            cfg: pipeline_cfg,
            tracker_cfg: tracker_config(cfg),
            detector,
            pose,
            temporal,
            behavior,
            catalog: Catalog::load(),
            gpu: GpuCache::default(),
        })
    }

    /// One eval-mode forward pass of the detector (`which = 0`) or the pose network (`1`), on the
    /// GPU when [`crate::tensor::gpu_active`], else on the CPU; the named outputs, row-major.
    fn run_net(
        &self,
        which: usize,
        input: Vec<f32>,
        shape: &[usize],
    ) -> Result<Vec<(&'static str, Vec<f32>)>> {
        let net = if which == 0 {
            &self.detector
        } else {
            &self.pose
        };
        let cpu = |input: Vec<f32>| -> Result<Recorded> {
            let mut g = Graph::new();
            let x = g.from_data(input, shape);
            let out = net.model.forward_named(&mut Ctx::eval(), &mut g, &[x])?;
            Ok((g, x, out))
        };
        #[cfg(all(feature = "mps", target_os = "macos"))]
        if crate::tensor::gpu_active() {
            let mut nets = self
                .gpu
                .nets
                .lock()
                .expect("the GPU cache lock is never poisoned");
            let key = (which, shape.to_vec());
            if let Some(entry) = nets.get_mut(&key) {
                if let Some((gpu, names)) = entry {
                    let values = gpu.run(&input)?;
                    return Ok(names.iter().copied().zip(values).collect());
                }
            } else {
                // First frame at this shape: run on the CPU, then record it for the GPU.
                let (g, x, out) = cpu(input)?;
                let ids: Vec<NodeId> = out.iter().map(|o| o.1).collect();
                let names: Vec<&'static str> = out.iter().map(|o| o.0).collect();
                let captured = match crate::tensor::mpsgraph::GpuForward::capture(&g, x, &ids) {
                    Ok(gpu) => Some((gpu, names)),
                    Err(err) => {
                        eprintln!("GPU capture failed, staying on the CPU: {err:#}");
                        None
                    }
                };
                nets.insert(key, captured);
                return Ok(out.iter().map(|(n, id)| (*n, g.to_vec(*id))).collect());
            }
        }
        let (g, _, out) = cpu(input)?;
        Ok(out.iter().map(|(n, id)| (*n, g.to_vec(*id))).collect())
    }

    /// `PosePipeline.run` = `track` followed by `refine`, over grey frames held in memory.
    pub fn run(&self, frames: &[GrayF], background: Option<&GrayF>, fps: f64) -> Result<VideoOut> {
        self.run_stream(frames.iter().map(|f| Ok(to_bgr8(f))), background, fps, None)
    }

    /// `PosePipeline.run` over streamed 8-bit BGR frames; `times` are the frames' presentation
    /// times (variable frame rates), `None` for a constant `fps`.
    pub fn run_stream(
        &self,
        frames: impl IntoIterator<Item = Result<Bgr>>,
        background: Option<&GrayF>,
        fps: f64,
        times: Option<&[f64]>,
    ) -> Result<VideoOut> {
        let mut result = self.track(frames, background)?;
        self.refine_at(&mut result, fps, times)?;
        Ok(result)
    }

    /// `PosePipeline.track`: detect, estimate and associate over the frames.
    pub(crate) fn track(
        &self,
        frames: impl IntoIterator<Item = Result<Bgr>>,
        background: Option<&GrayF>,
    ) -> Result<VideoOut> {
        let mut tracker = MultiAnimalTracker::new(self.tracker_cfg.clone());
        let mut per_track: BTreeMap<i64, BTreeMap<usize, Detection>> = BTreeMap::new();
        let mut proposer = background.map(|bg| {
            let mut proposer = BlobProposer::new(bg.clone())
                .with_diff(self.cfg.blob_diff.unwrap_or(self.cfg.foreground_diff))
                .with_score(self.cfg.blob_score as f32)
                .with_open_px(self.cfg.blob_open_px)
                .with_calibrate(self.cfg.blob_calibrate);
            if let Some(area) = self.cfg.blob_area {
                proposer = proposer.with_area(area);
            }
            if self.cfg.blob_arena_roi {
                proposer = proposer.with_roi(crate::algos::background::arena_roi(bg, 15));
            }
            proposer
        });

        let mut n_frames = 0;
        for (t, bgr) in frames.into_iter().enumerate() {
            let bgr = bgr?;
            n_frames += 1;
            let mut boxes = self.detect(&bgr, bgr.width, bgr.height)?;
            if let Some(proposer) = proposer.as_mut() {
                let (extra, extra_scores) = proposer.propose(&bgr);
                let extra_scores: Vec<f64> = extra_scores.iter().map(|s| f64::from(*s)).collect();
                boxes = merge_candidates(
                    &boxes.0,
                    &boxes.1,
                    &extra,
                    &extra_scores,
                    self.cfg.merge_iou,
                    self.cfg.max_det,
                );
            }
            let mut detections = self.frame_detections_bgr(&bgr, &boxes.0, &boxes.1)?;
            if let Some(bg) = background {
                detections.retain(|d| {
                    foreground_fraction(&bgr, bg, d.bbox, self.cfg.foreground_diff)
                        >= self.cfg.min_foreground
                });
            }
            for (tid, index) in tracker.update(&detections) {
                per_track
                    .entry(tid)
                    .or_default()
                    .insert(t, detections[index].clone());
            }
        }

        let mut tracks = BTreeMap::new();
        let pairs = BILATERAL_PAIRS.len();
        for (tid, observed) in per_track {
            let first = *observed.keys().next().expect("a track has a frame");
            let last = *observed.keys().next_back().expect("a track has a frame");
            let frames: Vec<usize> = (first..=last).collect();
            let span = frames.len();
            let mut track = TrackOut {
                id: tid,
                frames: frames.clone(),
                boxes: vec![f64::NAN; span * 4],
                keypoints: vec![f64::NAN; span * NUM_KEYPOINTS * 2],
                conf: vec![0.0; span * NUM_KEYPOINTS],
                pair_margin: (self.cfg.pair_decoding && self.cfg.pair_identity)
                    .then(|| vec![0.0; span * pairs]),
                ..TrackOut::default()
            };
            let mut have_margin = false;
            for (i, frame) in frames.iter().enumerate() {
                if let Some(det) = observed.get(frame) {
                    for d in 0..4 {
                        track.boxes[i * 4 + d] = det.bbox[d];
                    }
                    if let Some(kps) = &det.keypoints {
                        for (k, p) in kps.iter().enumerate() {
                            track.keypoints[(i * NUM_KEYPOINTS + k) * 2] = p[0];
                            track.keypoints[(i * NUM_KEYPOINTS + k) * 2 + 1] = p[1];
                        }
                    }
                    if let Some(conf) = &det.kp_conf {
                        for (k, c) in conf.iter().enumerate() {
                            track.conf[i * NUM_KEYPOINTS + k] = *c;
                        }
                    }
                    if let (Some(margin), Some(row)) = (&mut track.pair_margin, &det.pair_margin) {
                        have_margin = true;
                        for (p, m) in row.iter().enumerate() {
                            margin[i * pairs + p] = *m;
                        }
                    }
                }
            }
            // `track`: `resolve_pair_identity` over the whole track when the network produced margins.
            if self.cfg.pair_decoding && self.cfg.pair_identity && have_margin {
                let margin = track.pair_margin.clone().unwrap_or_default();
                let size = pair_sizes(&track.boxes);
                let mut flat: Vec<[f64; 2]> = track
                    .keypoints
                    .chunks_exact(2)
                    .map(|c| [c[0], c[1]])
                    .collect();
                let swapped = resolve_pair_identity(
                    &mut flat,
                    &margin,
                    &BILATERAL_PAIRS,
                    &size,
                    self.cfg.pair_motion_frac,
                    NUM_KEYPOINTS,
                );
                track.keypoints = flat.iter().flat_map(|p| [p[0], p[1]]).collect();
                track.pair_swapped = Some(swapped);
            }
            tracks.insert(tid, track);
        }
        Ok(VideoOut {
            tracks,
            n_frames,
            behaviors: self
                .behavior
                .as_ref()
                .map(|(_, classes)| classes.clone())
                .unwrap_or_default(),
            ..VideoOut::default()
        })
    }

    /// `data/datasets.py:letterbox` + `to_tensor` for one frame: the `(3, H, W)` planar RGB tensor
    /// in `[0, 1]` the detector is handed, plus the scale and pad `detect` undoes again.
    pub fn detector_input(&self, image: &Bgr) -> Result<(Vec<f32>, f64, [f64; 2])> {
        let (padded, scale, pad) =
            letterbox(image, self.cfg.det_size).context("the detector input is empty")?;
        Ok((to_tensor(&padded)?, scale, pad))
    }

    /// `PosePipeline.detect`: letterbox, run the detector, un-map the boxes.
    pub fn detect(&self, image: &Bgr, width: u32, height: u32) -> Result<(Vec<Box4>, Vec<f64>)> {
        let (tensor, scale, pad) = self.detector_input(image)?;
        let (h, w) = (
            usize::try_from(self.cfg.det_size.0)?,
            usize::try_from(self.cfg.det_size.1)?,
        );
        let out = self.run_net(0, tensor, &[1, 3, h, w])?;
        let heat = named(&out, "heat")?;
        let size = named(&out, "size")?;
        let offset = named(&out, "offset")?;
        let decoded = detector_decode(
            heat,
            size,
            offset,
            w / self.detector.stride,
            h / self.detector.stride,
            f64::from(self.detector.stride as u32),
            self.cfg.max_det,
        );
        // `keep = scores >= min(det_thresh, tracker.low_thresh)`; the pad and scale are then undone.
        let floor = self.cfg.det_thresh.min(self.tracker_cfg.low_thresh);
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        for (box4, score) in decoded.boxes.iter().zip(&decoded.scores) {
            if f64::from(*score) >= floor {
                boxes.push(*box4);
                scores.push(f64::from(*score));
            }
        }
        Ok((
            unletterbox_boxes(&boxes, scale, pad, (width, height)),
            scores,
        ))
    }

    /// `PosePipeline.frame_detections`: pose for every box, as the tracker wants it.
    fn frame_detections_bgr(
        &self,
        frame: &Bgr,
        boxes: &[Box4],
        scores: &[f64],
    ) -> Result<Vec<Detection>> {
        let mut out = Vec::with_capacity(boxes.len());
        for (i, box4) in boxes.iter().enumerate() {
            let pose = self.estimate_bgr(frame, *box4)?;
            let mut det =
                Detection::new(*box4, scores[i]).with_keypoints(pose.keypoints, pose.conf.clone());
            match self.cfg.embedder.as_str() {
                "model" => det = det.with_embedding(&pose.embedding),
                "color" => det = det.with_embedding(&color_histogram(frame, *box4, (12, 6))),
                _ => {}
            }
            if !pose.margin.is_empty() {
                det = det.with_pair_margin(pose.margin.clone());
            }
            out.push(det);
        }
        Ok(out)
    }

    /// `geometry.py:crop_affine` + `data/datasets.py:to_tensor` for one box: the `(3, S, S)` crop the
    /// pose net is handed, plus the matrix mapping image pixels into that crop.
    pub fn pose_input(&self, frame: &GrayF, box4: Box4) -> Result<(Vec<f32>, [[f64; 3]; 2])> {
        self.pose_input_bgr(&to_bgr8(frame), box4)
    }

    /// [`Self::pose_input`] on an 8-bit BGR frame: `cv2.warpAffine` of every channel.
    pub fn pose_input_bgr(&self, frame: &Bgr, box4: Box4) -> Result<(Vec<f32>, [[f64; 3]; 2])> {
        let (center, side) = box_xyxy_to_cs(box4, 1.0, self.cfg.crop_padding);
        let matrix = crop_affine(center, side, self.cfg.crop_size, 0.0);
        let crop = warp_affine_bgr(frame, &matrix, self.cfg.crop_size);
        Ok((to_tensor(&crop)?, matrix))
    }

    /// `PosePipeline.estimate` for one box: crop, run the pose net, decode, un-warp.
    pub fn estimate(&self, frame: &GrayF, box4: Box4) -> Result<Estimate> {
        self.estimate_bgr(&to_bgr8(frame), box4)
    }

    /// [`Self::estimate`] on an 8-bit BGR frame.
    pub fn estimate_bgr(&self, frame: &Bgr, box4: Box4) -> Result<Estimate> {
        let (tensor, matrix) = self.pose_input_bgr(frame, box4)?;
        let size = usize::try_from(self.cfg.crop_size)?;
        let out = self.run_net(1, tensor, &[1, 3, size, size])?;
        let heat = named(&out, "heat")?;
        let offset = named(&out, "offset")?;
        let embedding = named(&out, "embedding")?.to_vec();
        let maps = HeatMaps::new(
            heat,
            offset,
            NUM_KEYPOINTS,
            size / self.pose.stride,
            size / self.pose.stride,
        )?;
        let decoded = if self.cfg.pair_decoding {
            pose_decode_pairs(
                &maps,
                f64::from(self.pose.stride as u32),
                &BILATERAL_PAIRS,
                DEFAULT_PAIR_RADIUS,
                DEFAULT_PAIR_MIN_PEAK,
            )
        } else {
            pose_decode(&maps, f64::from(self.pose.stride as u32))
        };
        // The crop matrix maps image pixels to crop pixels; its inverse takes the keypoints back.
        let back = invert_affine(matrix);
        Ok(Estimate {
            keypoints: decoded
                .keypoints
                .iter()
                .map(|p| apply_affine(back, *p))
                .collect(),
            conf: decoded.conf,
            margin: decoded.margin,
            embedding,
        })
    }

    /// `PosePipeline.refine`: temporal smoothing, then the classifier and the catalogue.
    /// `PosePipeline.refine(result, fps=fps, times=times)`: unknown or missing times continue at
    /// the constant rate `fps` after the last known one.
    fn refine_at(&self, result: &mut VideoOut, fps: f64, times: Option<&[f64]>) -> Result<()> {
        let n = result
            .tracks
            .values()
            .map(|t| t.frames.last().copied().unwrap_or(0) + 1)
            .max()
            .unwrap_or(0)
            .max(result.n_frames);
        let known: &[f64] = times.unwrap_or(&[]);
        result.times = if known.len() >= n {
            known[..n].to_vec()
        } else {
            let last = known.last().copied().unwrap_or(-1.0 / fps);
            known
                .iter()
                .copied()
                .chain((1..=n - known.len()).map(|i| last + i as f64 / fps))
                .collect()
        };

        let span: BTreeMap<i64, (usize, usize)> = result
            .tracks
            .iter()
            .map(|(id, t)| (*id, (t.frames[0], t.frames.len())))
            .collect();
        let mut kin: BTreeMap<i64, Kinematics> = BTreeMap::new();
        let ids: Vec<i64> = result.tracks.keys().copied().collect();
        for tid in ids {
            let frames = result.tracks[&tid].span();
            if frames < MIN_TRACK_FRAMES {
                continue;
            }
            let track = &result.tracks[&tid];
            let observed: Vec<f64> = (0..frames)
                .flat_map(|t| {
                    (0..NUM_KEYPOINTS).map(move |k| {
                        let at = (t * NUM_KEYPOINTS + k) * 2;
                        let seen = track.conf[t * NUM_KEYPOINTS + k] >= self.cfg.kp_conf_thresh
                            && track.keypoints[at].is_finite()
                            && track.keypoints[at + 1].is_finite();
                        f64::from(seen)
                    })
                })
                .collect();
            if let Some(temporal) = &self.temporal {
                let smoothed = smooth_sequence(
                    temporal,
                    &track.keypoints,
                    &track.conf,
                    &observed,
                    frames,
                    self.cfg.temporal_window,
                    self.cfg.temporal_hop,
                )?;
                let track = result.tracks.get_mut(&tid).expect("the track exists");
                track.smoothed = Some(smoothed.coords);
                track.smoothed_conf = Some(smoothed.confidence);
                track.imputed = Some(smoothed.imputed);
            }
            if self.behavior.is_none() {
                continue;
            }
            let times: Vec<f64> = result.tracks[&tid]
                .frames
                .iter()
                .map(|f| result.times[*f])
                .collect();
            let probs = self.behavior_probs(&result.tracks[&tid], &times)?;
            let track = result.tracks.get_mut(&tid).expect("the track exists");
            track.behavior_probs = Some(probs);
            let (best, conf) = (track.best().to_vec(), track.best_conf().to_vec());
            kin.insert(
                tid,
                body_kinematics(&best, &conf, frames, NUM_KEYPOINTS, fps, Some(&times)),
            );
        }

        let Some((_, classes)) = self.behavior.as_ref() else {
            return Ok(());
        };
        if kin.is_empty() {
            return Ok(());
        }
        let width = self.catalog.names.len();
        // `refine` overwrites `modelled` for every track, so `describe` reports the last one.
        let mut modelled = Vec::new();
        let ids: Vec<i64> = result.tracks.keys().copied().collect();
        for tid in ids {
            if !kin.contains_key(&tid) {
                continue;
            }
            let (first, frames) = span[&tid];
            let partners: Vec<Kinematics> = kin
                .iter()
                .filter(|(other, _)| **other != tid)
                .map(|(other, other_kin)| {
                    let (other_first, other_len) = span[other];
                    align_kinematics(other_kin, other_first, other_len, first, frames)
                })
                .collect();
            let partner_refs: Vec<&Kinematics> = partners.iter().collect();
            let track = result.tracks.get(&tid).expect("the track exists");
            let probs = track
                .behavior_probs
                .clone()
                .unwrap_or_else(|| vec![0.0; frames * classes.len()]);
            let (full, next) =
                self.catalog
                    .full_probabilities(classes, &probs, frames, &kin[&tid], &partner_refs);
            modelled = next;
            let behavior: Vec<i64> = (0..frames)
                .map(|t| argmax_first(&full[t * width..(t + 1) * width]) as i64)
                .collect();
            let track = result.tracks.get_mut(&tid).expect("the track exists");
            track.behavior_full = Some(full);
            track.behavior = Some(behavior);
        }
        result.behaviors = self.catalog.names.clone();
        result.behavior_info = Some(self.catalog.describe(&modelled));
        Ok(())
    }

    /// `PosePipeline._behavior_probs`' input for one track: the `(coords, mask)` arrays the
    /// classifier is handed (NaNs cleared, `behavior_fps` resampling applied) and the time grid
    /// they sit on, which every frame is mapped back onto.
    pub fn behavior_input(
        &self,
        track: &TrackOut,
        times: &[f64],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f64>)> {
        let frames = track.span();
        let conf = track.best_conf();
        let pts = track.best();
        let mut mask: Vec<f64> = (0..frames)
            .flat_map(|t| {
                (0..NUM_KEYPOINTS).map(move |k| {
                    let at = (t * NUM_KEYPOINTS + k) * 2;
                    let seen = nan_to_num(conf[t * NUM_KEYPOINTS + k]) >= self.cfg.behavior_conf
                        && pts[at].is_finite()
                        && pts[at + 1].is_finite();
                    f64::from(seen)
                })
            })
            .collect();
        if let Some(mapping) = &self.cfg.behavior_keypoints {
            let keep = canonical_mask(mapping)?;
            for (i, m) in mask.iter_mut().enumerate() {
                *m *= f64::from(keep[i % NUM_KEYPOINTS]);
            }
        }
        // `behavior_fps` resamples the track onto an even time grid, so the classifier's sense of
        // time does not change with the recording's frame rate or with dropped frames.
        let (grid, x, m) = match self.cfg.behavior_fps {
            Some(rate) => resample(pts, &mask, times, rate, frames)?,
            None => (times.to_vec(), pts.to_vec(), mask.clone()),
        };
        let x = x.iter().map(|v| nan_to_num(*v) as f32).collect();
        let m = m.iter().map(|v| *v as f32).collect();
        Ok((x, m, grid))
    }

    /// `PosePipeline._behavior_probs`: classifier probabilities per frame.
    fn behavior_probs(&self, track: &TrackOut, times: &[f64]) -> Result<Vec<f64>> {
        let Some(model) = self.behavior.as_ref().map(|(m, _)| m) else {
            return Ok(Vec::new());
        };
        let classes = self.behavior.as_ref().expect("checked").1.len();
        let frames = track.span();
        let (x, m, grid) = self.behavior_input(track, times)?;
        let steps = grid.len();
        let mut g = Graph::new();
        let input = g.from_data(x, &[1, steps, NUM_KEYPOINTS, 2]);
        let mask_node = g.from_data(m, &[1, steps, NUM_KEYPOINTS]);
        let mut ctx = Ctx::eval();
        let out = model.forward_named(&mut ctx, &mut g, &[input, mask_node])?;
        let logits = g.to_vec(output(&out, "logits")?);
        let mut probs = softmax_rows(&logits, steps, classes);
        if self.cfg.behavior_smooth > 1 {
            probs = median_smooth(
                &probs,
                steps,
                classes,
                i64::try_from(self.cfg.behavior_smooth)?,
                true,
            );
        }
        if self.cfg.behavior_prior_adapt {
            // `adapt_prior(probs, max_ratio = ...)`: the defaults of
            // `behavior/postprocess.py:adapt_prior` are `iters = 25` and `tol = 1e-4`, and
            // `prior_source = None`, i.e. the mean of these very predictions.
            let (post, _) = adapt_prior(
                &probs,
                steps,
                classes,
                None,
                25,
                self.cfg.behavior_prior_max_ratio,
                1e-4,
            );
            probs = post;
        }
        // Each frame takes the nearest grid sample back.
        let mut out_probs = vec![0.0f64; frames * classes];
        for (t, time) in times.iter().enumerate() {
            let at = nearest(&grid, *time);
            for c in 0..classes {
                out_probs[t * classes + c] = probs[at * classes + c];
            }
        }
        Ok(out_probs)
    }
}

/// One crop's decoded pose, before the tracker wraps it.
#[derive(Clone, Debug)]
pub struct Estimate {
    /// Keypoint positions in image pixels `(K, 2)`.
    pub keypoints: Vec<[f64; 2]>,
    /// Keypoint confidences `(K,)`.
    pub conf: Vec<f64>,
    /// Per-pair left/right evidence `(pairs.len(),)`.
    pub margin: Vec<f64>,
    /// The pose network's identity embedding.
    pub embedding: Vec<f32>,
}

/// `refine`'s `min_len`: a shorter track gets no temporal pass and no behavior.
const MIN_TRACK_FRAMES: usize = 5;

/// The index of the first maximum of a row, `np.argmax`'s tie-break.
fn argmax_first(row: &[f64]) -> usize {
    let mut best = 0;
    let mut value = row[0];
    for (i, v) in row.iter().enumerate().skip(1) {
        if *v > value {
            value = *v;
            best = i;
        }
    }
    best
}

/// `np.searchsorted` plus the nearest-neighbour correction of `_behavior_probs`.
fn nearest(grid: &[f64], time: f64) -> usize {
    let at = grid.partition_point(|v| *v < time).min(grid.len() - 1);
    let prev = at.saturating_sub(1);
    if (grid[prev] - time).abs() < (grid[at] - time).abs() {
        prev
    } else {
        at
    }
}

/// `_behavior_probs`' `behavior_fps` branch: `np.arange` grid, `np.interp` per keypoint.
fn resample(
    pts: &[f64],
    mask: &[f64],
    times: &[f64],
    rate: f64,
    frames: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let step = 1.0 / rate;
    anyhow::ensure!(
        step.is_finite() && step > 0.0,
        "behavior_fps must be positive"
    );
    let count = ((times[frames - 1] - times[0] + 0.5 * step) / step)
        .floor()
        .max(0.0) as usize
        + 1;
    let grid: Vec<f64> = (0..count).map(|i| times[0] + i as f64 * step).collect();
    let near: Vec<usize> = grid.iter().map(|t| nearest(times, *t)).collect();
    let mut x = vec![f64::NAN; count * NUM_KEYPOINTS * 2];
    for k in 0..NUM_KEYPOINTS {
        let observed: Vec<usize> = (0..frames)
            .filter(|t| mask[t * NUM_KEYPOINTS + k] > 0.0)
            .collect();
        if observed.len() < 2 {
            continue;
        }
        let xs: Vec<f64> = observed.iter().map(|t| times[*t]).collect();
        for d in 0..2 {
            let ys: Vec<f64> = observed
                .iter()
                .map(|t| pts[(*t * NUM_KEYPOINTS + k) * 2 + d])
                .collect();
            for (i, time) in grid.iter().enumerate() {
                x[(i * NUM_KEYPOINTS + k) * 2 + d] = interp(&xs, &ys, *time);
            }
        }
    }
    let m: Vec<f64> = near
        .iter()
        .flat_map(|t| (0..NUM_KEYPOINTS).map(move |k| mask[t * NUM_KEYPOINTS + k]))
        .collect();
    Ok((grid, x, m))
}

/// `np.interp` on an increasing `xs`.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let hi = xs.partition_point(|v| *v <= x).max(1).min(xs.len() - 1);
    let lo = hi - 1;
    let w = (x - xs[lo]) / (xs[hi] - xs[lo]).max(1e-12);
    ys[lo] + w * (ys[hi] - ys[lo])
}

/// Row-wise softmax of `(rows, classes)` logits.
fn softmax_rows(logits: &[f32], rows: usize, classes: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * classes];
    for r in 0..rows {
        let row = &logits[r * classes..(r + 1) * classes];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = row
            .iter()
            .map(|v| (f64::from(*v) - f64::from(max)).exp())
            .collect();
        let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
        for c in 0..classes {
            out[r * classes + c] = exps[c] / sum;
        }
    }
    out
}

/// `PosePipeline._align`: another track's kinematics on these frames, NaN where it is absent.
fn align_kinematics(
    other: &Kinematics,
    other_first: usize,
    other_len: usize,
    first: usize,
    frames: usize,
) -> Kinematics {
    let at = |pos: usize| pos >= other_first && pos - other_first < other_len;
    let pick = |values: &[[f64; 2]]| -> Vec<[f64; 2]> {
        (0..frames)
            .map(|i| {
                let pos = first + i;
                if at(pos) {
                    values[pos - other_first]
                } else {
                    [f64::NAN; 2]
                }
            })
            .collect()
    };
    let pick1 = |values: &[f64]| -> Vec<f64> {
        (0..frames)
            .map(|i| {
                let pos = first + i;
                if at(pos) {
                    values[pos - other_first]
                } else {
                    f64::NAN
                }
            })
            .collect()
    };
    Kinematics {
        centre: pick(&other.centre),
        axis: pick(&other.axis),
        length: other.length,
        speed: pick1(&other.speed),
        velocity: pick(&other.velocity),
        nose: pick(&other.nose),
        tail: pick(&other.tail),
        body: pick(&other.body),
    }
}

/// The `tracker:` block over `TrackerConfig`'s defaults.
fn tracker_config(cfg: &Config) -> TrackerConfig {
    let mut out = TrackerConfig::default();
    out.high_thresh = cfg.f64_at("tracker.high_thresh", out.high_thresh);
    out.low_thresh = cfg.f64_at("tracker.low_thresh", out.low_thresh);
    out.w_iou = cfg.f64_at("tracker.w_iou", out.w_iou);
    out.w_oks = cfg.f64_at("tracker.w_oks", out.w_oks);
    out.w_app = cfg.f64_at("tracker.w_app", out.w_app);
    out.match_thresh = cfg.f64_at("tracker.match_thresh", out.match_thresh);
    out.low_match_thresh = cfg.f64_at("tracker.low_match_thresh", out.low_match_thresh);
    out.reid_thresh = cfg.f64_at("tracker.reid_thresh", out.reid_thresh);
    out.gate_mahalanobis = cfg.f64_at("tracker.gate_mahalanobis", out.gate_mahalanobis);
    out.n_init = cfg.usize_at("tracker.n_init", out.n_init);
    out.max_lost = cfg.usize_at("tracker.max_lost", out.max_lost);
    if cfg.get("tracker.max_animals").is_some_and(|v| !v.is_null()) {
        out.max_animals = cfg.get("tracker.max_animals").and_then(|v| v.as_usize());
    }
    out.ema = cfg.f64_at("tracker.ema", out.ema);
    out.gallery_size = cfg.usize_at("tracker.gallery_size", out.gallery_size);
    out.kp_conf_thresh = cfg.f64_at("tracker.kp_conf_thresh", out.kp_conf_thresh);
    out
}

/// Loads one checkpoint and checks the network kind it holds.
fn load_net(cfg: &Config, key: &str, kind: ModelKind) -> Result<Net> {
    let path = cfg
        .get(key)
        .and_then(|v| v.as_str())
        .with_context(|| format!("`{key}` is missing from the config"))?;
    let file = repo_path(path);
    let (got_cfg, got_kind, weights) =
        crate::models::load(&file).with_context(|| format!("cannot read {path}"))?;
    if got_kind != kind {
        bail!("{path} holds a {got_kind} network, the config wants a {kind} one");
    }
    let model = Model::build(kind, &got_cfg, &weights)
        .with_context(|| format!("cannot build the {kind} network of {path}"))?;
    let stride = match &model {
        Model::Detector(m) => m.stride(),
        Model::Pose(m) => m.stride(),
        _ => 1,
    };
    Ok(Net {
        model,
        cfg: got_cfg,
        stride,
    })
}

/// The output named `name` of [`Pipeline::run_net`].
fn named<'a>(out: &'a [(&'static str, Vec<f32>)], name: &str) -> Result<&'a [f32]> {
    out.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| v.as_slice())
        .with_context(|| format!("the network has no {name:?} output"))
}

/// The output node named `name`.
fn output(out: &[(&'static str, NodeId)], name: &str) -> Result<NodeId> {
    out.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, node)| *node)
        .with_context(|| format!("the network returned no `{name}` output"))
}

/// `np.nan_to_num` for one scalar.
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

/// `np.nan_to_num(size, nan = 1.0)` of `pipeline.py:PosePipeline.track`: the per-frame animal size
/// `resolve_pair_identity` normalises its paw-jump cost by, from the track's `(span, 4)` boxes.
///
/// `np.fmax` keeps the other operand where one side is NaN, and a frame the tracker never saw has no
/// box at all — its cost is then measured against a one-pixel animal, not against zero.
fn pair_sizes(boxes: &[f64]) -> Vec<f64> {
    boxes
        .chunks_exact(4)
        .map(|box4| {
            let (w, h) = (box4[2] - box4[0], box4[3] - box4[1]);
            let size = if w.is_nan() {
                h
            } else if h.is_nan() {
                w
            } else {
                w.max(h)
            };
            if size.is_nan() {
                1.0
            } else {
                nan_to_num(size)
            }
        })
        .collect()
}

/// `geometry.py:warp`: an affine crop sampled with bilinear weights, zeros outside.
pub fn warp_affine_bgr(image: &Bgr, matrix: &[[f64; 3]; 2], out_size: u32) -> Bgr {
    let (w, h) = (image.width, image.height);
    let channel = |c: usize| GrayF {
        width: w,
        height: h,
        pixels: image
            .pixels
            .iter()
            .skip(c)
            .step_by(3)
            .map(|v| f32::from(*v))
            .collect(),
    };
    let warped: Vec<GrayF> = (0..3)
        .map(|c| warp_affine(&channel(c), matrix, out_size))
        .collect();
    let n = (out_size * out_size) as usize;
    let mut pixels = Vec::with_capacity(n * 3);
    for i in 0..n {
        for plane in &warped {
            pixels.push(plane.pixels[i].round().clamp(0.0, 255.0) as u8);
        }
    }
    Bgr {
        width: out_size,
        height: out_size,
        pixels,
    }
}

/// `tracking/tracker.py:ColorHistogramEmbedder`: the normalised hue-saturation histogram of the
/// box centre (the middle 60 % of each side) plus its mean brightness, L2-normalised.
///
/// `// differs from Python:` OpenCV's 8-bit `BGR2HSV` uses lookup-table divisions; the float
/// formula here can move a pixel sitting exactly on a bin edge into the neighbouring bin.
pub fn color_histogram(image: &Bgr, box4: Box4, bins: (usize, usize)) -> Vec<f32> {
    let (w, h) = (f64::from(image.width), f64::from(image.height));
    let [x0, y0, x1, y1] = box4;
    let (cx, cy, bw, bh) = (
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        (x1 - x0) * 0.3,
        (y1 - y0) * 0.3,
    );
    let xa = (cx - bw).max(0.0) as usize;
    let xb = (cx + bw + 1.0).min(w).max(0.0) as usize;
    let ya = (cy - bh).max(0.0) as usize;
    let yb = (cy + bh + 1.0).min(h).max(0.0) as usize;
    let dims = bins.0 * bins.1 + 1;
    if xa >= xb || ya >= yb {
        return vec![(1.0 / (dims as f64).sqrt()) as f32; dims];
    }
    let mut hist = vec![0.0f64; bins.0 * bins.1];
    let mut v_sum = 0.0;
    let mut count = 0.0;
    for y in ya..yb {
        for x in xa..xb {
            let at = (y * image.width as usize + x) * 3;
            let (b, g, r) = (
                f64::from(image.pixels[at]),
                f64::from(image.pixels[at + 1]),
                f64::from(image.pixels[at + 2]),
            );
            let v = r.max(g).max(b);
            let diff = v - r.min(g).min(b);
            let s = if v > 0.0 {
                (255.0 * diff / v).round()
            } else {
                0.0
            };
            let mut hue = if diff == 0.0 {
                0.0
            } else if v == r {
                60.0 * (g - b) / diff
            } else if v == g {
                120.0 + 60.0 * (b - r) / diff
            } else {
                240.0 + 60.0 * (r - g) / diff
            };
            if hue < 0.0 {
                hue += 360.0;
            }
            let hue = (hue / 2.0).round() % 180.0;
            let hb = ((hue * bins.0 as f64 / 180.0) as usize).min(bins.0 - 1);
            let sb = ((s * bins.1 as f64 / 256.0) as usize).min(bins.1 - 1);
            hist[hb * bins.1 + sb] += 1.0;
            v_sum += v;
            count += 1.0;
        }
    }
    let total = hist.iter().sum::<f64>().max(1.0);
    let mut out: Vec<f64> = hist.iter().map(|c| c / total).collect();
    out.push(v_sum / count / 255.0);
    let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-9;
    out.iter().map(|v| (v / norm) as f32).collect()
}

/// `cv2.warpAffine` of a grey image into an `out_size` square crop.
pub fn warp_affine(image: &GrayF, matrix: &[[f64; 3]; 2], out_size: u32) -> GrayF {
    let back = invert_affine(*matrix);
    let mut pixels = vec![0.0f32; (out_size * out_size) as usize];
    for y in 0..out_size {
        for x in 0..out_size {
            let sx = back[0][0] * f64::from(x) + back[0][1] * f64::from(y) + back[0][2];
            let sy = back[1][0] * f64::from(x) + back[1][1] * f64::from(y) + back[1][2];
            pixels[(y * out_size + x) as usize] = sample(image, sx, sy);
        }
    }
    GrayF {
        width: out_size,
        height: out_size,
        pixels,
    }
}

/// One bilinear sample, `cv2.warpAffine(..., borderMode=BORDER_CONSTANT)`'s rule: each of the four
/// taps carries its weight, and a tap outside the image contributes the border value (zero) instead
/// of clamping to the edge pixel. A source coordinate a whole pixel outside is therefore zero, and
/// the last half pixel of each edge fades out towards it, exactly as OpenCV's `warpAffine` does.
fn sample(image: &GrayF, x: f64, y: f64) -> f32 {
    if !x.is_finite() || !y.is_finite() {
        return 0.0;
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let tap = |value: f64| -> (i64, f64) {
        let floor = value.floor().clamp(-9.0e18, 9.0e18) as i64;
        (floor, value - floor as f64)
    };
    let (x0, wx) = tap(x);
    let (y0, wy) = tap(y);
    let (width, height) = (
        usize::try_from(image.width).unwrap_or(0),
        usize::try_from(image.height).unwrap_or(0),
    );
    let at = |column: i64, row: i64| -> f64 {
        let (Ok(cx), Ok(cy)) = (usize::try_from(column), usize::try_from(row)) else {
            return 0.0;
        };
        if cx < width && cy < height {
            f64::from(image.pixels[cy * width + cx])
        } else {
            0.0
        }
    };
    let top = at(x0, y0) * (1.0 - wx) + at(x0 + 1, y0) * wx;
    let bottom = at(x0, y0 + 1) * (1.0 - wx) + at(x0 + 1, y0 + 1) * wx;
    (top * (1.0 - wy) + bottom * wy) as f32
}

impl VideoOut {
    /// `VideoResult.time_of`: the printed timestamp, empty when the frame has none.
    fn time_of(&self, frame: usize) -> String {
        match self.times.get(frame) {
            Some(t) => format!("{t:.4}"),
            None => String::new(),
        }
    }

    /// `VideoResult.to_csv`: one row per `(frame, track, keypoint)`.
    pub fn to_csv(&self, path: &Path) -> Result<()> {
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("cannot write {}", path.display()))?,
        );
        write!(
            file,
            "frame,time_s,track,keypoint,x,y,confidence,imputed,behavior\r\n"
        )?;
        for (tid, track) in &self.tracks {
            let best = track.best();
            let conf = track.best_conf();
            for (i, frame) in track.frames.iter().enumerate() {
                let behavior = match (&track.behavior, self.behaviors.is_empty()) {
                    (Some(behavior), false) => self
                        .behaviors
                        .get(usize::try_from(behavior[i]).unwrap_or(0))
                        .cloned()
                        .unwrap_or_default(),
                    _ => String::new(),
                };
                let time = self.time_of(*frame);
                for (k, name) in CANONICAL_NAMES.iter().enumerate() {
                    let at = (i * NUM_KEYPOINTS + k) * 2;
                    let imputed = match &track.imputed {
                        Some(imputed) => imputed[i * NUM_KEYPOINTS + k] as i64,
                        None => i64::from(!best[at].is_finite()),
                    };
                    write!(
                        file,
                        "{frame},{time},{tid},{name},{},{},{},{},{}\r\n",
                        fixed(best[at], 2),
                        fixed(best[at + 1], 2),
                        fixed(nan_to_num(conf[i * NUM_KEYPOINTS + k]), 3),
                        imputed,
                        behavior,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// `VideoResult.to_behavior_csv`: one row per `(frame, track)` with the top-k behaviors.
    pub fn to_behavior_csv(&self, path: &Path, k: usize) -> Result<()> {
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(path)
                .with_context(|| format!("cannot write {}", path.display()))?,
        );
        let mut header = vec![
            "frame".to_string(),
            "time_s".to_string(),
            "track".to_string(),
        ];
        for i in 0..k {
            header.push(format!("top{}", i + 1));
            header.push(format!("top{}_p", i + 1));
        }
        for name in &self.behaviors {
            header.push(format!("p_{name}"));
        }
        write!(file, "{}\r\n", header.join(","))?;
        let catalog = Catalog::load();
        for (tid, track) in &self.tracks {
            let Some(full) = &track.behavior_full else {
                continue;
            };
            let (idx, probs) = catalog.top_k(full, track.span(), k);
            for (i, frame) in track.frames.iter().enumerate() {
                let time = self.time_of(*frame);
                let mut row = vec![frame.to_string(), time, tid.to_string()];
                for j in 0..k {
                    let at = i * k + j;
                    row.push(
                        self.behaviors
                            .get(idx[at])
                            .cloned()
                            .unwrap_or_else(|| "other".to_string()),
                    );
                    row.push(fixed(probs[at], 3));
                }
                for c in 0..self.behaviors.len() {
                    row.push(fixed(full[i * self.behaviors.len() + c], 4));
                }
                write!(file, "{}\r\n", row.join(","))?;
            }
        }
        Ok(())
    }

    /// `VideoResult.to_npz`: the per-track arrays of `np.savez_compressed`.
    pub fn to_npz(&self, path: &Path) -> Result<()> {
        let mut entries: Vec<(String, NpyArray)> = Vec::new();
        let mut push = |name: String, array: NpyArray| entries.push((name, array));
        let flat = |values: &[f64], frames: usize, width: usize| NpyArray {
            shape: vec![frames, width],
            data: NpyData::Float(values.iter().map(|v| *v as f32).collect()),
        };
        let flat3 = |values: &[f64], frames: usize, keypoints: usize| NpyArray {
            shape: vec![frames, keypoints, 2],
            data: NpyData::Float(values.iter().map(|v| *v as f32).collect()),
        };
        for (tid, track) in &self.tracks {
            let prefix = format!("track{tid}");
            let frames = track.span();
            push(
                format!("{prefix}_frames"),
                NpyArray {
                    shape: vec![frames],
                    data: NpyData::Int(track.frames.iter().map(|f| *f as i64).collect()),
                },
            );
            push(format!("{prefix}_boxes"), flat(&track.boxes, frames, 4));
            push(
                format!("{prefix}_keypoints"),
                flat3(&track.keypoints, frames, NUM_KEYPOINTS),
            );
            push(
                format!("{prefix}_conf"),
                flat(&track.conf, frames, NUM_KEYPOINTS),
            );
            if let Some(values) = &track.smoothed {
                push(
                    format!("{prefix}_smoothed"),
                    flat3(values, frames, NUM_KEYPOINTS),
                );
            }
            for (name, values) in [
                ("smoothed_conf", &track.smoothed_conf),
                ("imputed", &track.imputed),
                ("behavior_probs", &track.behavior_probs),
                ("behavior_full", &track.behavior_full),
            ] {
                if let Some(values) = values {
                    let width = values.len() / frames.max(1);
                    push(format!("{prefix}_{name}"), flat(values, frames, width));
                }
            }
            if let Some(behavior) = &track.behavior {
                push(
                    format!("{prefix}_behavior"),
                    NpyArray {
                        shape: vec![behavior.len()],
                        data: NpyData::Int(behavior.clone()),
                    },
                );
            }
        }
        if !self.times.is_empty() {
            push(
                "times".to_string(),
                NpyArray {
                    shape: vec![self.times.len()],
                    data: NpyData::Float(self.times.iter().map(|v| *v as f32).collect()),
                },
            );
        }
        push(
            "keypoint_names".to_string(),
            NpyArray {
                shape: vec![CANONICAL_NAMES.len()],
                data: NpyData::Text(CANONICAL_NAMES.iter().map(|s| (*s).to_string()).collect()),
            },
        );
        push(
            "behaviors".to_string(),
            NpyArray {
                shape: vec![self.behaviors.len()],
                data: NpyData::Text(self.behaviors.clone()),
            },
        );
        let arrays: Vec<(&str, &NpyArray)> = entries
            .iter()
            .map(|(name, array)| (name.as_str(), array))
            .collect();
        write_npz(path, &arrays)
    }
}

/// `f"{value:.prec$}"`: numpy's spelling of the special values included.
fn fixed(value: f64, prec: usize) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value == f64::INFINITY {
        return "inf".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-inf".to_string();
    }
    let text = format!("{value:.prec$}");
    // Rust prints "-0.00" for a negative zero exactly as Python does, so nothing else is needed.
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("animalpose-predict-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the temporary directory is writable");
        dir
    }

    fn repo_file(rel: &str) -> PathBuf {
        crate::data::repo_root()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(rel)
    }

    /// `configs/pipeline.yaml`'s `pipeline:` and `tracker:` blocks.
    #[test]
    fn pipeline_config_matches_python() {
        let cfg = Config::load(repo_file("configs/pipeline.yaml"), &[]).expect("the config loads");
        let pc = PipelineCfg::from_config(&cfg);
        assert_eq!(pc.det_size, (256, 320));
        assert!((pc.det_thresh - 0.3).abs() < 1e-12);
        assert_eq!(pc.crop_size, 192);
        assert!((pc.kp_conf_thresh - 0.3).abs() < 1e-12);
        assert!(pc.pair_decoding && pc.pair_identity);
        assert!((pc.pair_motion_frac - 0.1).abs() < 1e-12);
        assert_eq!(pc.detector_candidates, ["network".to_string()]);
        assert_eq!(pc.embedder, "model");
        assert_eq!(pc.temporal_window, 64);

        let tc = tracker_config(&cfg);
        assert!((tc.high_thresh - 0.5).abs() < 1e-12);
        assert!((tc.low_thresh - 0.2).abs() < 1e-12);
        assert_eq!(tc.max_animals, None);
        assert_eq!(tc.max_lost, 90);
    }

    /// The defaults of `PipelineConfig` in `pipeline.py`.
    #[test]
    fn defaults_match_the_python_dataclass() {
        let pc = PipelineCfg::default();
        assert_eq!(pc.det_size, (256, 320));
        assert!((pc.det_thresh - 0.3).abs() < 1e-12);
        assert_eq!(pc.max_det, 20);
        assert_eq!(pc.crop_size, 128);
        assert!((pc.crop_padding - 1.25).abs() < 1e-12);
        assert!((pc.foreground_diff - 25.0).abs() < 1e-12);
        assert!((pc.min_foreground - 0.08).abs() < 1e-12);
        assert_eq!(pc.blob_calibrate, 64);
        assert!((pc.merge_iou - 0.5).abs() < 1e-12);
        assert!((pc.behavior_conf - 0.6).abs() < 1e-12);
        assert_eq!(pc.behavior_top_k, 3);
        assert_eq!(pc.behavior_smooth, 5);
        assert!(pc.behavior_prior_adapt);
        assert!((pc.behavior_prior_max_ratio - 4.0).abs() < 1e-12);
        assert_eq!(pc.temporal_window, 64);
        assert_eq!(pc.temporal_hop, 32);
        assert_eq!(pc.detector_candidates, ["network".to_string()]);
    }

    /// The per-frame animal size `resolve_pair_identity` is handed.
    #[test]
    fn pair_sizes_follow_np_fmax_and_nan_to_num() {
        // `np.nan_to_num(np.fmax(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), nan = 1.0)` over these rows:
        // 40, no box at all, 50, and a row whose width is NaN while its height is not.
        let boxes = vec![
            0.0,
            0.0,
            40.0,
            20.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            10.0,
            10.0,
            60.0,
            40.0,
            0.0,
            0.0,
            f64::NAN,
            30.0,
        ];
        let size = pair_sizes(&boxes);
        assert_eq!(size.len(), 4);
        assert!(
            (size[0] - 40.0).abs() < 1e-12 && (size[2] - 50.0).abs() < 1e-12,
            "{size:?}"
        );
        // An unobserved frame costs against a one-pixel animal (`nan = 1.0`), never against zero.
        assert!((size[1] - 1.0).abs() < 1e-12, "{size:?}");
        assert!((size[3] - 30.0).abs() < 1e-12, "{size:?}");
    }

    fn two_frame_track() -> TrackOut {
        let frames = vec![0, 1];
        let mut keypoints = vec![f64::NAN; 2 * NUM_KEYPOINTS * 2];
        keypoints[0] = 10.0;
        keypoints[1] = 20.0;
        TrackOut {
            id: 7,
            frames,
            boxes: vec![1.0, 2.0, 11.0, 12.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            keypoints,
            conf: vec![0.75; 2 * NUM_KEYPOINTS],
            imputed: Some(vec![0.0; 2 * NUM_KEYPOINTS]),
            ..TrackOut::default()
        }
    }

    /// `VideoResult.to_csv`: CRLF rows, `nan` for an absent point, an empty behavior column.
    #[test]
    fn csv_rows_match_the_python_format() {
        let dir = temp_dir("csv");
        let mut video = VideoOut::default();
        video.tracks.insert(7, two_frame_track());
        video.n_frames = 2;
        video.times = vec![0.0, 1.0 / 30.0];
        let path = dir.join("tracks.csv");
        video.to_csv(&path).expect("the csv is written");
        let text = std::fs::read_to_string(&path).expect("the csv is readable");
        let lines: Vec<&str> = text.split("\r\n").collect();
        assert_eq!(
            lines[0],
            "frame,time_s,track,keypoint,x,y,confidence,imputed,behavior"
        );
        assert_eq!(lines[1], "0,0.0000,7,nose,10.00,20.00,0.750,0,");
        assert_eq!(lines[26], "1,0.0333,7,nose,nan,nan,0.750,0,");
        assert_eq!(lines.len(), 52, "a header, 50 rows and the trailing CRLF");
        assert_eq!(lines[51], "");
    }

    /// `VideoResult.to_npz`: the array names and shapes `VideoResult.to_npz` produces.
    #[test]
    fn npz_arrays_have_the_python_shapes() {
        let dir = temp_dir("npz");
        let mut video = VideoOut::default();
        video.tracks.insert(7, two_frame_track());
        video.n_frames = 2;
        video.times = vec![0.0, 1.0 / 30.0];
        video.behaviors = vec!["rest".to_string(), "walk".to_string()];
        let path = dir.join("tracks.npz");
        video.to_npz(&path).expect("the npz is written");
        let arrays = crate::data::sessions::read_npz(&path).expect("the npz is readable");
        let mut names: Vec<String> = arrays.keys().cloned().collect();
        names.sort();
        assert_eq!(
            names,
            [
                "behaviors",
                "keypoint_names",
                "times",
                "track7_boxes",
                "track7_conf",
                "track7_frames",
                "track7_imputed",
                "track7_keypoints",
            ]
        );
        assert_eq!(arrays["track7_keypoints"].shape, vec![2, NUM_KEYPOINTS, 2]);
        assert_eq!(arrays["track7_boxes"].shape, vec![2, 4]);
        assert_eq!(arrays["track7_conf"].shape, vec![2, NUM_KEYPOINTS]);
        assert_eq!(arrays["track7_imputed"].shape, vec![2, NUM_KEYPOINTS]);
        assert_eq!(arrays["times"].shape, vec![2]);
        match &arrays["track7_keypoints"].data {
            NpyData::Float(v) => assert!((f64::from(v[0]) - 10.0).abs() < 1e-6),
            other => panic!("keypoints must be float, got {other:?}"),
        }
    }

    /// `geometry.py:warp` of a `crop_affine` matrix: the box centre lands in the crop centre.
    #[test]
    fn crop_warp_centers_the_box() {
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<f32> = (0..width * height).map(|i| (i % width) as f32).collect();
        let image = GrayF::from_pixels(width, height, pixels).expect("the image is sized");
        let (center, side) = box_xyxy_to_cs([4.0, 4.0, 12.0, 12.0], 1.0, 1.0);
        let matrix = crop_affine(center, side, 8, 0.0);
        let crop = warp_affine(&image, &matrix, 8);
        // `crop_affine` puts the box centre at the centre of the crop, i.e. at (3.5, 3.5) of an 8x8.
        let source = apply_affine(matrix, center);
        assert!((source[0] - 3.5).abs() < 1e-9 && (source[1] - 3.5).abs() < 1e-9);
        // A linear ramp stays exact under bilinear sampling: source x = 3.5 + (x - 3.5) * 7 / 8.
        for x in 0..8u32 {
            let want = center[0] + (f64::from(x) - 3.5) * side / f64::from(8);
            let got = f64::from(crop.pixels[usize::try_from(3 * 8 + x).unwrap()]);
            assert!(
                (got - want).abs() < 1e-5,
                "crop column {x}: {got} vs {want}"
            );
        }
    }

    /// `geometry.py:warp` asks OpenCV for `BORDER_CONSTANT`, which does not clamp: a tap outside the
    /// image contributes the border value (zero) with its own weight, so a crop that reaches a hair
    /// past the border *fades* towards it and only turns fully zero a whole pixel past it. The
    /// numbers below are what `cv2.warpAffine` returns for a flat 200 image.
    #[test]
    fn crop_warp_fades_at_the_border_like_opencv() {
        let image = GrayF::from_pixels(8, 8, vec![200.0f32; 64]).expect("the image is sized");
        let (center, side) = box_xyxy_to_cs([0.0, 0.0, 2.0, 2.0], 1.0, 1.25);
        let crop = warp_affine(&image, &crop_affine(center, side, 8, 0.0), 8);
        // The corner sample lands at (-0.0937, -0.0937): 200 * 0.90625 * 0.90625.
        assert!(
            (f64::from(crop.pixels[0]) - 164.257_812_5).abs() < 1e-4,
            "the corner is {}",
            crop.pixels[0]
        );
        // A sample that is past the border in one axis only: 200 * 0.90625, and 200 once both taps
        // of both axes are inside.
        assert!((f64::from(crop.pixels[1]) - 181.25).abs() < 1e-4);
        assert!((f64::from(crop.pixels[3 * 8 + 3]) - 200.0).abs() < 1e-4);
        // A crop that reaches a whole pixel past the border gets the border value there.
        let (center, side) = box_xyxy_to_cs([-6.0, -6.0, 2.0, 2.0], 1.0, 1.25);
        let crop = warp_affine(&image, &crop_affine(center, side, 8, 0.0), 8);
        assert!(
            crop.pixels[..6].iter().all(|v| *v == 0.0),
            "the far border is zero"
        );
        // The last sample of that row sits at x = -0.125: 200 * 0.875.
        assert!((f64::from(crop.pixels[7 * 8 + 5]) - 175.0).abs() < 1e-4);
    }

    #[test]
    fn special_values_are_printed_like_numpy() {
        assert_eq!(fixed(f64::NAN, 2), "nan");
        assert_eq!(fixed(f64::INFINITY, 2), "inf");
        assert_eq!(fixed(f64::NEG_INFINITY, 2), "-inf");
        assert_eq!(fixed(372.596, 2), "372.60");
        assert_eq!(fixed(0.0004, 4), "0.0004");
    }

    #[test]
    fn nearest_grid_sample_and_interp() {
        let grid = [0.0, 0.2, 0.4];
        assert_eq!(nearest(&grid, 0.0), 0);
        assert_eq!(nearest(&grid, 0.09), 0);
        assert_eq!(nearest(&grid, 0.11), 1);
        assert_eq!(nearest(&grid, 0.14), 1);
        assert_eq!(nearest(&grid, 9.0), 2);
        assert!((interp(&[0.0, 1.0], &[0.0, 10.0], 0.25) - 2.5).abs() < 1e-12);
        assert!((interp(&[0.0, 1.0], &[0.0, 10.0], -1.0)).abs() < 1e-12);
        assert!((interp(&[0.0, 1.0], &[0.0, 10.0], 5.0) - 10.0).abs() < 1e-12);
    }

    /// `behavior_fps` resampling keeps the observed keypoints and fills the gaps by interpolation.
    #[test]
    fn resample_interpolates_onto_an_even_grid() {
        let frames = 3;
        let mut pts = vec![f64::NAN; frames * NUM_KEYPOINTS * 2];
        let mut mask = vec![0.0; frames * NUM_KEYPOINTS];
        for t in [0, 2] {
            pts[(t * NUM_KEYPOINTS) * 2] = t as f64;
            pts[(t * NUM_KEYPOINTS) * 2 + 1] = 2.0 * t as f64;
            mask[t * NUM_KEYPOINTS] = 1.0;
        }
        let times: Vec<f64> = (0..frames).map(|t| t as f64 / 30.0).collect();
        let (grid, x, m) = resample(&pts, &mask, &times, 20.0, frames).expect("resampling works");
        assert!(grid.len() >= 2);
        assert!(x.iter().step_by(NUM_KEYPOINTS * 2).all(|v| v.is_finite()));
        assert!(m.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let logits = [1.0f32, 2.0, 3.0, -1.0, 0.0, 0.5];
        let probs = softmax_rows(&logits, 2, 3);
        for r in 0..2 {
            let sum: f64 = probs[r * 3..(r + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row {r} sums to {sum}");
        }
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax_first(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    /// A `.pt` checkpoint is refused with the export hint instead of failing to parse.
    #[test]
    fn torch_checkpoints_load_directly() {
        let pt = repo_file("weights/detector_general.pt");
        match std::fs::read(&pt) {
            Ok(bytes) if !crate::models::torch_pt::is_lfs_pointer(&bytes) => {}
            _ => return eprintln!("SKIP: {} is absent or an lfs pointer", pt.display()),
        }
        let cfg = Config::parse(
            &format!("checkpoints:\n  detector: {}\n", pt.display()),
            &[],
        )
        .expect("parses");
        let net = load_net(&cfg, "checkpoints.detector", ModelKind::Detector)
            .expect("the general detector loads from its .pt");
        assert!(matches!(net.model, Model::Detector(_)));
        let err = match load_net(&cfg, "checkpoints.detector", ModelKind::Pose) {
            Ok(_) => panic!("a detector checkpoint is not a pose network"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("detector network"), "{err}");
    }
}
