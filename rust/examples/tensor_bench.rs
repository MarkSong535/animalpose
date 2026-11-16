//! Release-mode timing harness for the tensor core.
//!
//! ```text
//! cargo run --release --example tensor_bench -- [workload ...] [--reps N]
//! ```
//!
//! Workloads (all by default):
//!
//! * `pose1` / `pose8` — the general pose network (ResNet-18 backbone, 25 keypoints) forward in eval mode
//!   on a `(1|8, 3, 192, 192)` crop batch, with the trained weights of
//!   `tests/fixtures/models/pose_general.safetensors`;
//! * `det` — the general detector (tiny backbone, width 32) forward in eval mode on `(1, 3, 256, 320)`,
//!   with `tests/fixtures/models/detector_general.safetensors`;
//! * `train` — one training step (forward, loss, backward) of the tiny pose network of
//!   `configs/pose.yaml` (width 32, 25 keypoints) on a `(8, 3, 128, 128)` batch;
//! * `lstm` / `lstmtrain` — the temporal network of `configs/temporal.yaml` (2-layer bidirectional
//!   LSTM, hidden 256) forward in eval mode, and one training step, on 8 tracks of 64 frames.
//!
//! The fixture weights are looked up from the working directory upwards (or under
//! `$ANIMALPOSE_FIXTURES`); a workload whose weights are absent is skipped.

use animalpose::models::config::ModelConfig;
use animalpose::models::weights::Weights;
use animalpose::models::{load, Model, ModelKind};
use animalpose::tensor::{Ctx, Graph, LossKind};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn fixtures_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("ANIMALPOSE_FIXTURES") {
        let base = PathBuf::from(dir);
        if base.join("models").is_dir() {
            return Some(base.join("models"));
        }
        return Some(base);
    }
    let mut at = env::current_dir().ok()?;
    for _ in 0..6 {
        let candidate = at.join("tests").join("fixtures").join("models");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !at.pop() {
            break;
        }
    }
    None
}

/// A deterministic pseudo-random image batch in `[0, 1)`.
fn images(shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut s = 0x1234_5678_u32;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

/// Times `f` `reps` times after one warm-up run; returns the median and the fastest run in
/// milliseconds (the minimum is the steadier number on a busy machine).
fn time<F: FnMut()>(reps: usize, mut f: F) -> (f64, f64) {
    f();
    let mut runs: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (runs[runs.len() / 2], runs[0])
}

fn eval_forward(model: &Model, shape: &[usize], reps: usize) -> (f64, f64) {
    let data = images(shape);
    time(reps, || {
        let mut g = Graph::new();
        let x = g.from_data(data.clone(), shape);
        let outs = model.forward(&mut Ctx::eval(), &mut g, &[x]).unwrap();
        std::hint::black_box(g.value(outs[0])[0]);
    })
}

fn train_step(reps: usize) -> (f64, f64) {
    let cfg = ModelConfig::from_json(&serde_json::json!({
        "backbone": {"name": "tiny", "width": 32, "out_channels": 64},
        "num_keypoints": 25,
        "embed_dim": 64
    }))
    .unwrap();
    let mut model = Model::build(ModelKind::Pose, &cfg, &Weights::new()).unwrap();
    model.assign_slots(0);
    let shape = [8, 3, 128, 128];
    let data = images(&shape);
    let mut seed = 0u64;
    time(reps, || {
        seed += 1;
        let mut g = Graph::new();
        let x = g.from_data(data.clone(), &shape);
        let outs = model.forward(&mut Ctx::train(seed), &mut g, &[x]).unwrap();
        let heat_shape = g.shape(outs[0]).clone();
        let target = g.constant(0.0, &heat_shape);
        let heat = g.bce_with_logits(outs[0], target);
        let off = g.weighted_loss(outs[1], LossKind::SmoothL1, 0.05);
        let emb = g.reduce_mean(outs[2], &[0, 1]);
        let loss = g.add(heat, off);
        let loss = g.add(loss, emb);
        g.backward(loss).unwrap();
        std::hint::black_box(g.param_grads().len());
    })
}

fn temporal(train: bool, reps: usize) -> (f64, f64) {
    let cfg = ModelConfig::from_json(&serde_json::json!({
        "num_keypoints": 25, "dims": 2, "hidden": 256, "layers": 2, "bidirectional": true,
        "dropout": 0.1, "prior_bandwidth": 2.0, "prior": "hybrid"
    }))
    .unwrap();
    let mut model = Model::build(ModelKind::Temporal, &cfg, &Weights::new()).unwrap();
    model.assign_slots(0);
    let (n, t, k) = (8, 64, 25);
    let coords: Vec<f32> = images(&[n, t, k, 2]).iter().map(|v| v * 200.0).collect();
    let conf = images(&[n, t, k]);
    let observed: Vec<f32> = conf.iter().map(|v| f32::from(*v > 0.2)).collect();
    let mut seed = 0u64;
    time(reps, || {
        seed += 1;
        let mut g = Graph::new();
        let x = g.from_data(coords.clone(), &[n, t, k, 2]);
        let c = g.from_data(conf.clone(), &[n, t, k]);
        let o = g.from_data(observed.clone(), &[n, t, k]);
        let mut ctx = if train { Ctx::train(seed) } else { Ctx::eval() };
        let outs = model.forward(&mut ctx, &mut g, &[x, c, o]).unwrap();
        if train {
            let loss = g.weighted_loss(outs[1], LossKind::SmoothL1, 1.0);
            g.backward(loss).unwrap();
            std::hint::black_box(g.param_grads().len());
        } else {
            std::hint::black_box(g.value(outs[0])[0]);
        }
    })
}

fn main() {
    let mut reps = 5usize;
    let mut wanted: Vec<String> = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--reps" {
            reps = args.next().and_then(|v| v.parse().ok()).unwrap_or(reps);
        } else if arg != "--bench" {
            wanted.push(arg);
        }
    }
    if wanted.is_empty() {
        wanted = ["pose1", "pose8", "det", "train", "lstm", "lstmtrain"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    println!(
        "threads: {}, reps: {reps} after one warm-up; median and min in ms",
        rayon::current_num_threads()
    );
    let dir = fixtures_dir();
    let model_of = |name: &str| -> Option<Model> {
        let path = dir.as_ref()?.join(format!("{name}.safetensors"));
        if !path.is_file() {
            println!("skipping {name}: {} is absent", path.display());
            return None;
        }
        let (cfg, kind, weights) = load(&path).unwrap();
        Some(Model::build(kind, &cfg, &weights).unwrap())
    };
    let mut pose: Option<Option<Model>> = None;
    for w in &wanted {
        let (median, min) = match w.as_str() {
            "pose1" | "pose8" => {
                let batch = if w == "pose1" { 1 } else { 8 };
                let model = pose.get_or_insert_with(|| model_of("pose_general"));
                match model {
                    Some(m) => eval_forward(m, &[batch, 3, 192, 192], reps),
                    None => continue,
                }
            }
            "det" => match model_of("detector_general") {
                Some(m) => eval_forward(&m, &[1, 3, 256, 320], reps),
                None => continue,
            },
            "train" => train_step(reps),
            "lstm" => temporal(false, reps),
            "lstmtrain" => temporal(true, reps),
            other => {
                println!("unknown workload {other}");
                continue;
            }
        };
        println!("{w:<6} {median:>10.1} {min:>10.1}");
    }
}
