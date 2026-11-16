//! End-to-end multi-view runs on simulated rigs.
//!
//! * The stage after per-view tracking (`MultiViewPipeline.run` minus `pipe.track`): simulated
//!   per-view tracks, with a fragmented track and a mirrored side view, go through fusion,
//!   left/right resolution, triangulation, the 3D temporal model and the 3D behavior model, and
//!   the outputs `animalpose predict --views` writes.
//! * `animalpose data calibrate` on landmarks of a simulated rig, with `--refine`.
//! * `animalpose predict --views` through the command line with the tiny checkpoints (ignored by
//!   default, like `predict_e2e`: run with `--release -- --ignored`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use animalpose::cli::predict::{PipelineCfg, TrackOut, VideoOut};
use animalpose::data::sessions::{read_npz, NpyData};
use animalpose::data::synthetic::{self, camera_from_view, FLIP_INDEX, NUM_KEYPOINTS as K};
use animalpose::models::config::ModelConfig;
use animalpose::models::weights::Weights;
use animalpose::models::{Model, ModelKind};
use animalpose::multiview::camera::{dumps_cameras, load_cameras, Camera};
use animalpose::multiview::dense::Vec3;
use animalpose::multiview::npz::{write_npz, Member};
use animalpose::multiview::pipeline::MultiViewPipeline;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("animalpose-mv-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

const ARENA: f64 = 100.0;

/// `make_cameras(ARENA, ("top", "side", "angled"), size=(640, 480))`.
fn rig() -> Vec<Camera> {
    [
        ("top", 90.0, -90.0),
        ("side", 8.0, -90.0),
        ("angled", 45.0, -135.0),
    ]
    .iter()
    .map(|(name, el, az)| {
        Camera::from(&camera_from_view(
            ARENA,
            *el,
            *az,
            (640, 480),
            1.6,
            name,
            0.9,
        ))
    })
    .collect()
}

/// Closed-form 3D keypoints of animal `a` at frame `t` (cm).
fn pose(a: usize, t: usize) -> Vec<Vec3> {
    let t = t as f64;
    let body = synthetic::body_pose(
        &[0.2, 0.1, 1.0, 0.0, 0.0, 0.0],
        0.4 * t,
        0.3 * t,
        0.0,
        0.2 * t,
    );
    let h = 0.05 * t + std::f64::consts::PI * a as f64;
    let (c, s) = (h.cos(), h.sin());
    let (px, py) = (
        25.0 * (0.03 * t + 2.0 * a as f64).cos(),
        25.0 * (0.03 * t + 2.0 * a as f64).sin(),
    );
    body.iter()
        .map(|b| {
            let b = [b[0] * 20.0, b[1] * 20.0, b[2] * 20.0];
            [c * b[0] - s * b[1] + px, s * b[0] + c * b[1] + py, b[2]]
        })
        .collect()
}

/// One single-view track over `frames`, as `PosePipeline.track` leaves it.
fn track(id: i64, frames: std::ops::Range<usize>, kp: &[Vec<[f64; 2]>], conf: f64) -> TrackOut {
    let span = frames.len();
    TrackOut {
        id,
        frames: frames.clone().collect(),
        boxes: vec![f64::NAN; span * 4],
        keypoints: kp[frames]
            .iter()
            .flatten()
            .flat_map(|p| [p[0], p[1]])
            .collect(),
        conf: vec![conf; span * K],
        ..TrackOut::default()
    }
}

fn model(kind: ModelKind, cfg: serde_json::Value) -> Model {
    let cfg = ModelConfig::from_json(&cfg).expect("a model config");
    Model::build(kind, &cfg, &Weights::new()).expect("the network builds at its initial weights")
}

#[test]
fn fusion_triangulation_and_3d_models_run_end_to_end() {
    let cams = rig();
    let n = 40;
    let world: Vec<Vec<Vec<Vec3>>> = (0..2)
        .map(|a| (0..n).map(|t| pose(a, t)).collect())
        .collect();
    let mut per_view = Vec::new();
    for (v, cam) in cams.iter().enumerate() {
        let mut out = VideoOut {
            n_frames: n,
            ..VideoOut::default()
        };
        for (a, frames) in world.iter().enumerate() {
            let kp: Vec<Vec<[f64; 2]>> = frames
                .iter()
                .enumerate()
                .map(|(t, pts)| {
                    let uv: Vec<[f64; 2]> = pts
                        .iter()
                        .enumerate()
                        .map(|(k, p)| {
                            let q = cam.project(p);
                            let e = 0.4 * ((t * K + k) as f64 * 1.3 + v as f64).sin();
                            [q[0] + e, q[1] - e]
                        })
                        .collect();
                    // The side view cannot tell left from right: its labels are mirrored.
                    if v == 1 {
                        (0..K).map(|k| uv[FLIP_INDEX[k]]).collect()
                    } else {
                        uv
                    }
                })
                .collect();
            let conf = if v == 1 { 0.6 } else { 0.9 };
            if v == 2 && a == 1 {
                // Animal 1 fragments into two tracks in the angled view.
                out.tracks.insert(7, track(7, 0..18, &kp, conf));
                out.tracks.insert(8, track(8, 22..n, &kp, conf));
            } else {
                let id = (a + v) as i64 % 2;
                out.tracks.insert(id, track(id, 0..n, &kp, conf));
            }
        }
        per_view.push(out);
    }

    let mut mv = MultiViewPipeline::new(cams.clone(), &PipelineCfg::default());
    mv.resolve_left_right = true;
    mv.temporal_window = 16;
    mv.temporal_hop = 8;
    mv.temporal3d = Some(model(
        ModelKind::Temporal,
        serde_json::json!({"num_keypoints": K, "dims": 3, "hidden": 8, "layers": 1,
                           "bidirectional": true, "dropout": 0.0}),
    ));
    let classes = vec![
        "resting".to_string(),
        "walking".to_string(),
        "rearing".to_string(),
    ];
    mv.behavior3d = Some((
        model(
            ModelKind::Behavior,
            serde_json::json!({"num_keypoints": K, "dims": 3, "hidden": 8, "classes": classes, "dropout": 0.0}),
        ),
        classes.clone(),
    ));
    let res = mv.run_tracks(per_view).expect("the multi-view stage runs");

    // Two animals; the fragments of animal 1 share one identity.
    assert_eq!(res.animals.len(), 2);
    let ids = &res.identities;
    assert_eq!(ids[&(2, 7)], ids[&(2, 8)]);
    assert_eq!(ids[&(0, 1)], ids[&(2, 7)]);
    assert_ne!(ids[&(0, 0)], ids[&(0, 1)]);
    assert_eq!(
        ids[&(1, 1)],
        ids[&(0, 0)],
        "view 1 numbers its tracks the other way round"
    );

    for (animal, track0) in [(0usize, 0i64), (1, 1)] {
        let gid = ids[&(0, track0)];
        let a = &res.animals[&gid];
        let mut worst: f64 = 0.0;
        for (t, frame) in a.tri.keypoints3d.iter().enumerate() {
            for (k, p) in frame.iter().enumerate() {
                let w = world[animal][t][k];
                let e =
                    ((p[0] - w[0]).powi(2) + (p[1] - w[1]).powi(2) + (p[2] - w[2]).powi(2)).sqrt();
                assert!(
                    e.is_finite(),
                    "every keypoint is seen by at least two views"
                );
                worst = worst.max(e);
            }
        }
        assert!(worst < 0.5, "animal {animal}: worst 3D error {worst} cm");
        // The mirrored side view is relabelled, so all three views are used where all see it.
        assert_eq!(a.tri.n_views[5][13], 3);
        let smoothed = a.smoothed3d.as_ref().expect("the 3D temporal model ran");
        assert_eq!(smoothed.len(), n * K * 3);
        assert!(smoothed.iter().all(|v| v.is_finite()));
        let probs = a
            .behavior_probs
            .as_ref()
            .expect("the 3D behavior model ran");
        let behavior = a.behavior.as_ref().expect("argmax");
        assert_eq!((probs.len(), behavior.len()), (n * classes.len(), n));
        for (t, row) in probs.chunks(classes.len()).enumerate() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-5);
            let best = row
                .iter()
                .enumerate()
                .fold(0, |b, (j, p)| if *p > row[b] { j } else { b });
            assert_eq!(behavior[t], best as i64);
        }
    }

    let out = temp_dir("stage");
    res.write(&out).expect("outputs are written");
    let arrays = read_npz(out.join("animals3d.npz")).expect("animals3d.npz reads back");
    for g in 0..2 {
        for (name, shape) in [
            ("keypoints3d", vec![n, K, 3]),
            ("reproj_error", vec![n, K]),
            ("n_views", vec![n, K]),
            ("smoothed3d", vec![n, K, 3]),
            ("confidence3d", vec![n, K]),
            ("imputed", vec![n, K]),
            ("behavior_probs", vec![n, 3]),
            ("behavior", vec![n]),
        ] {
            let key = format!("animal{g}_{name}");
            assert_eq!(
                arrays.get(&key).map(|a| a.shape.clone()),
                Some(shape),
                "{key}"
            );
        }
        assert!(matches!(
            arrays[&format!("animal{g}_n_views")].data,
            NpyData::Int(_)
        ));
    }
    for v in 0..3 {
        let csv =
            std::fs::read_to_string(out.join(format!("view{v}_tracks.csv"))).expect("per-view CSV");
        assert!(csv.starts_with("frame,time_s,track,keypoint,x,y,confidence,imputed,behavior\r\n"));
    }
    std::fs::remove_dir_all(&out).ok();
}

fn run(args: &[&str]) {
    let args: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    animalpose::cli::run(&args).expect("the command runs");
}

#[test]
fn data_calibrate_writes_cams_json_from_landmarks() {
    let truth = rig();
    let dir = temp_dir("calibrate");
    // Landmarks: arena corners at two heights plus a few points on a calibration object.
    let mut landmarks: Vec<Vec3> = Vec::new();
    for (x, y) in [(-40.0, -40.0), (40.0, -40.0), (40.0, 40.0), (-40.0, 40.0)] {
        landmarks.push([x, y, 0.0]);
        landmarks.push([x * 0.5, y * 0.5, 12.0]);
    }
    landmarks.extend([[5.0, -10.0, 3.0], [-15.0, 20.0, 7.5]]);
    let cameras: Vec<serde_json::Value> = truth
        .iter()
        .enumerate()
        .map(|(v, c)| {
            let uv: Vec<[f64; 2]> = landmarks
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let q = c.project(p);
                    [
                        q[0] + 0.2 * (i as f64 + v as f64).sin(),
                        q[1] + 0.2 * (i as f64 * 0.7).cos(),
                    ]
                })
                .collect();
            let mut entry = serde_json::json!({"name": c.name, "size": [c.size.0, c.size.1],
                                               "points3d": landmarks, "points2d": uv});
            if v == 1 {
                // Known intrinsics: PnP instead of DLT.
                entry["K"] = serde_json::json!(c.k);
                entry["dist"] = serde_json::json!(c.dist);
            }
            entry
        })
        .collect();
    let points = dir.join("rig_points.json");
    std::fs::write(
        &points,
        serde_json::json!({ "cameras": cameras }).to_string(),
    )
    .expect("written");

    // Matched keypoints of one animal for --refine.
    let pts: Vec<Vec3> = (0..30).step_by(3).flat_map(|t| pose(0, t)).collect();
    let mut obs = Vec::new();
    for c in &truth {
        for p in &pts {
            obs.extend(c.project(p));
        }
    }
    obs[0] = f64::NAN; // one unseen observation
    obs[1] = f64::NAN;
    let refine = dir.join("obs.npz");
    write_npz(
        &refine,
        &[("obs".to_string(), Member::F64(vec![3, pts.len(), 2], obs))],
    )
    .expect("written");

    let out = dir.join("cams.json");
    run(&[
        "data",
        "calibrate",
        points.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
        "--refine",
        refine.to_str().expect("utf-8"),
    ]);
    let cams = load_cameras(&out).expect("cams.json is a list of Camera.to_dict()");
    assert_eq!(cams.len(), 3);
    for (got, want) in cams.iter().zip(&truth) {
        assert_eq!(
            (got.name.as_str(), got.size),
            (want.name.as_str(), want.size)
        );
        let (gc, wc) = (got.center(), want.center());
        let d =
            ((gc[0] - wc[0]).powi(2) + (gc[1] - wc[1]).powi(2) + (gc[2] - wc[2]).powi(2)).sqrt();
        assert!(d < 3.0, "{}: centre off by {d} cm", got.name);
        for p in &landmarks {
            let (a, b) = (got.project(p), want.project(p));
            assert!(
                (a[0] - b[0]).hypot(a[1] - b[1]) < 2.0,
                "{}: reprojection",
                got.name
            );
        }
    }
    assert_eq!(
        std::fs::read_to_string(&out).expect("text"),
        dumps_cameras(&cams)
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Frames of a bright blob for one view.
fn write_frames(dir: &Path, frames: usize, shift: u32) {
    std::fs::create_dir_all(dir).expect("frame dir");
    for t in 0..frames {
        let mut image = image::GrayImage::from_pixel(96, 96, image::Luma([40u8]));
        let cx = 12 + (t as u32 * 8 + shift) % 72;
        for y in 40..56 {
            for x in (cx - 6)..(cx + 6) {
                image.put_pixel(x, y, image::Luma([220u8]));
            }
        }
        image
            .save(dir.join(format!("frame_{t:04}.png")))
            .expect("frame written");
    }
}

#[test]
#[ignore = "the pose checkpoint is a resnet18; run it with --release --ignored"]
fn predict_views_writes_the_multi_view_outputs() {
    let fixtures = animalpose::data::fixtures_dir().expect("tests/fixtures exists");
    let models = fixtures.join("models");
    let base = temp_dir("predict");
    let views: Vec<PathBuf> = (0..2).map(|v| base.join(format!("view{v}"))).collect();
    for (v, dir) in views.iter().enumerate() {
        write_frames(dir, 6, 4 * v as u32);
    }
    let cams: Vec<Camera> = rig()
        .into_iter()
        .take(2)
        .map(|mut c| {
            c.size = (96, 96);
            c
        })
        .collect();
    let cams_json = base.join("cams.json");
    std::fs::write(&cams_json, dumps_cameras(&cams)).expect("cams.json");
    let config = base.join("pipeline.yaml");
    std::fs::write(
        &config,
        format!(
            concat!(
                "checkpoints:\n  detector: {}\n  pose: {}\n",
                "pipeline:\n  det_size: [96, 96]\n  det_thresh: 0.0\n  crop_size: 48\n  max_det: 2\n",
                "  embedder: none\n",
                "tracker:\n  high_thresh: 0.0\n  low_thresh: 0.0\n  max_animals: 1\n",
            ),
            models.join("detector_tiny.safetensors").display(),
            models.join("pose_general.safetensors").display(),
        ),
    )
    .expect("config");
    let out = base.join("out");
    let mut args = vec![
        "predict".to_string(),
        "--config".into(),
        config.display().to_string(),
        "--views".into(),
    ];
    args.extend(views.iter().map(|v| v.display().to_string()));
    args.extend([
        "--cameras".into(),
        cams_json.display().to_string(),
        "--out".into(),
        out.display().to_string(),
    ]);
    animalpose::cli::run(&args).expect("`predict --views` runs end to end");
    assert!(out.join("animals3d.npz").is_file());
    for v in 0..2 {
        let csv =
            std::fs::read_to_string(out.join(format!("view{v}_tracks.csv"))).expect("per-view CSV");
        assert!(csv.lines().count() > 1, "view {v} was tracked");
    }
    let arrays = read_npz(out.join("animals3d.npz")).expect("reads back");
    let by_name: BTreeMap<_, _> = arrays
        .iter()
        .map(|(k, v)| (k.clone(), v.shape.clone()))
        .collect();
    for (key, shape) in &by_name {
        if key.ends_with("_keypoints3d") {
            assert_eq!(&shape[1..], &[K, 3]);
        }
    }
    std::fs::remove_dir_all(&base).ok();
}
