//! Smoke tests of the image side of the port: `animalpose train pose` / `train detector` on
//! rendered frames with a tiny config, the checkpoints they write loading back through
//! `crate::models::load`, and `animalpose predict --synthetic` running on them.

use std::path::{Path, PathBuf};

use animalpose::models::{load, Model, ModelKind};

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("animalpose-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The `total` of every logged step, from the `.history.json` next to the checkpoint.
fn totals(history: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(history).unwrap();
    let rows: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    rows.iter().map(|r| r["total"].as_f64().unwrap()).collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn train(kind: &str, config: &str, dir: &Path, steps: usize) -> PathBuf {
    let cfg = dir.join(format!("{kind}.yaml"));
    std::fs::write(&cfg, config).unwrap();
    let out = dir.join(format!("{kind}.pt"));
    let args: Vec<String> = [
        "train".to_string(),
        kind.to_string(),
        cfg.to_string_lossy().into_owned(),
        format!("output={}", out.display()),
        format!("train.steps={steps}"),
    ]
    .into();
    animalpose::cli::run(&args).unwrap_or_else(|e| panic!("train {kind}: {e:#}"));
    out.with_extension("safetensors")
}

const POSE: &str = "\
model:
  num_keypoints: 25
  backbone: {name: tiny, width: 8, out_channels: 16}
  embed_dim: 16
data:
  source: synthetic
  crop_size: 64
  image_size: [160, 128]
  n_sims: 4
  frames: 60
  species: [mouse, rat]
train:
  batch_size: 8
  lr: 5.0e-3
  workers: 2
  log_every: 0
";

const DETECTOR: &str = "\
model:
  backbone: {name: tiny, width: 8, out_channels: 16}
  num_species: 2
data:
  source: synthetic
  input_size: [64, 96]
  image_size: [96, 64]
  n_sims: 4
  frames: 60
  species: [mouse, rat]
  degrade: 0.3
  zoom_out: 0.2
train:
  batch_size: 4
  lr: 5.0e-3
  workers: 2
  log_every: 0
";

/// About 10 s in a release build and over 3 minutes in a debug one, so debug runs skip it
/// (`cargo test --release` runs it).
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "trains two networks: run with cargo test --release"
)]
fn train_pose_and_detector_then_predict_synthetic() {
    let dir = temp_dir("render-train");

    let pose = train("pose", POSE, &dir, 24);
    let hist = totals(&pose.with_extension("history.json"));
    assert_eq!(hist.len(), 24);
    assert!(hist.iter().all(|v| v.is_finite()));
    let (first, last) = (mean(&hist[..6]), mean(&hist[18..]));
    assert!(last < 0.9 * first, "pose loss {first} -> {last}: {hist:?}");
    let (cfg, kind, weights) = load(&pose).expect("the pose checkpoint loads back");
    assert_eq!(kind, ModelKind::Pose);
    assert_eq!(cfg.embed_dim(), 16);
    let model = Model::build(kind, &cfg, &weights).expect("every pose slot is filled");
    assert_eq!(model.kind(), ModelKind::Pose);

    // `animalpose finetune` (`--init`): starting from the trained weights, the first step's loss
    // is already far below a fresh network's.
    let cfg_path = dir.join("pose.yaml");
    let tuned = dir.join("pose_tuned.pt");
    let args: Vec<String> = [
        "train".to_string(),
        "pose".to_string(),
        cfg_path.to_string_lossy().into_owned(),
        "--init".to_string(),
        pose.to_string_lossy().into_owned(),
        format!("output={}", tuned.display()),
        "train.steps=2".to_string(),
    ]
    .into();
    animalpose::cli::run(&args).unwrap_or_else(|e| panic!("finetune pose: {e:#}"));
    let tuned_hist = totals(&tuned.with_extension("history.json"));
    assert!(tuned_hist[0] < hist[0] * 0.9, "{tuned_hist:?} vs {hist:?}");
    load(&tuned.with_extension("safetensors")).expect("the fine-tuned checkpoint loads back");

    let det = train("detector", DETECTOR, &dir, 24);
    let hist = totals(&det.with_extension("history.json"));
    assert_eq!(hist.len(), 24);
    assert!(hist.iter().all(|v| v.is_finite()));
    let (first, last) = (mean(&hist[..6]), mean(&hist[18..]));
    assert!(last < first, "detector loss {first} -> {last}: {hist:?}");
    let (cfg, kind, weights) = load(&det).expect("the detector checkpoint loads back");
    assert_eq!(kind, ModelKind::Detector);
    assert_eq!(cfg.num_species(), 2);
    Model::build(kind, &cfg, &weights).expect("every detector slot is filled");
    let extra = weights.metadata_value("extra").unwrap_or_default();
    assert!(extra.contains("input_size"), "{extra}");

    // `animalpose predict --synthetic` on the two checkpoints.
    let pipeline = dir.join("pipeline.yaml");
    std::fs::write(
        &pipeline,
        format!(
            "checkpoints:\n  detector: {}\n  pose: {}\npipeline:\n  det_size: [64, 96]\n  \
             crop_size: 64\n  det_thresh: 0.05\ntracker:\n  max_animals: 2\n",
            det.display(),
            pose.display()
        ),
    )
    .unwrap();
    let out = dir.join("pred");
    let args: Vec<String> = [
        "predict",
        "--synthetic",
        "--config",
        &pipeline.to_string_lossy(),
        "--max-frames",
        "3",
        "--out",
        &out.to_string_lossy(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    animalpose::cli::run(&args).unwrap_or_else(|e| panic!("predict --synthetic: {e:#}"));
    assert!(out.join("tracks.csv").is_file());
    std::fs::remove_dir_all(&dir).ok();
}

/// The demo clip of `predict --synthetic`: two mice, both annotated in (nearly) every frame.
#[test]
fn demo_clip_shows_two_mice() {
    let frames = animalpose::data::render::demo_clip(5).unwrap();
    assert_eq!(frames.len(), 5);
    for f in &frames {
        assert_eq!((f.image.width, f.image.height), (320, 256));
        assert_eq!(f.n_animals(), 2);
    }
    let present: usize = frames
        .iter()
        .map(|f| f.present.iter().filter(|p| **p).count())
        .sum();
    assert!(present >= 8, "{present} of 10 animals present");
}
