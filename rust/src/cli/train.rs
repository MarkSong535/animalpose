//! `animalpose train {behavior,temporal,pose,detector} CONFIG [--init CHECKPOINT] [dotted.overrides...]`.
//!
//! The port of `src/animalpose/train/loops.py:train_behavior` / `train_temporal` (and, in
//! `train/images.rs`, `train_pose` / `train_detector`): the same
//! optimizer (`torch.optim.AdamW` plus the warmup-cosine schedule of `_optimizer`), the same
//! gradient clipping and log line of `_loop`, the same batch sources and mixing (`_pool`, `_Mix`,
//! `_KeypointSubset`, `behavior_windows`), and a checkpoint plus a `.history.json` next to the
//! configured `output`.
//!
//! `// differs from Python:`
//! * The checkpoint written is `.safetensors`, not `torch.save`'s pickle (`animalpose export
//!   import-safetensors` turns it into a `.pt`); an `init:` may be either, `.pt` files are read by
//!   [`crate::models::torch_pt`].
//! * Batches come from [`crate::data::synthetic::Rng`], not `numpy.random.default_rng`, so one
//!   seed draws different — but identically distributed — batches.
//! * `data.real_root` names a home-cage DLC tree that `data/tracks.py:load_sessions` reads through
//!   HDF5; that reader needs a C library, so the CLI rejects `real_root` instead of approximating it.
//! * The Gaussian NLL of `temporal/model.py:temporal_loss` multiplies by `exp(-2 * log_sigma)`, and
//!   [`crate::tensor::Graph`] has no `exp`: the factor enters as a per-step constant, so the
//!   NLL gradient reaches `coords_norm` exactly while `log_sigma` only sees its `d * log_sigma`
//!   term. Every temporal run prints that once.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::data::config::{Config, DataConfig, TrainConfig};
use crate::data::sessions::{behavior_windows, load_session_dir, Session};
use crate::data::synthetic::{
    apply_mask_mask, behavior_batch, canonical_mask, keypoint_subset_masks, temporal_batch,
    Corruption, Rng, SimulationPool, TemporalBatch, MILD_CORRUPTION,
};
use crate::models::config::ModelConfig;
use crate::models::layers::LayerTree;
use crate::models::weights::Weights;
use crate::models::{layers, Model, ModelKind};
use crate::tensor::{adamw_step, AdamW, Ctx, Graph, NodeId, Schedule, Slot};
use anyhow::{bail, Context, Result};
use serde_json::json;

mod images;

/// `skeleton/species.py:CANONICAL.num_keypoints`.
const NUM_KEYPOINTS: usize = 25;

/// `SimulationPool`'s `max_animals` default (`data/synthetic.py:SimulationPool.__init__`), which
/// `train/loops.py:_pool` does not override.
const POOL_MAX_ANIMALS: usize = 3;

/// `data/tracks.py:behavior_windows`' `conf_thresh`.
const BEHAVIOR_CONF_THRESH: f64 = 0.6;

/// `data/tracks.py:behavior_windows`' `min_labeled`.
const BEHAVIOR_MIN_LABELED: f64 = 0.25;

/// The layout of one training batch: `(batch, window, NUM_KEYPOINTS, dims)`.
#[derive(Clone, Copy, Debug)]
struct Geom {
    /// Sequences per batch (`train.batch_size`).
    batch: usize,
    /// Frames per sequence (`data.window`).
    window: usize,
    /// Coordinates per point, 2 for image tracks (`model.dims`).
    dims: usize,
}

/// `train/loops.py:train_behavior` / `train_temporal` for one config file.
pub fn run(model: &str, config: &Path, overrides: &[String]) -> Result<()> {
    let cfg = Config::load(config, overrides)?;
    match model {
        "behavior" => train_behavior(&cfg),
        "temporal" => train_temporal(&cfg),
        "pose" => images::train_pose(&cfg),
        "detector" => images::train_detector(&cfg),
        other => bail!("unknown model {other:?}: choose behavior, temporal, pose or detector"),
    }
}

/// `train/loops.py:train_behavior`.
fn train_behavior(cfg: &Config) -> Result<()> {
    let train = cfg.train();
    let data = cfg.data();
    let mcfg = model_config(cfg)?;
    let classes: Vec<String> = mcfg.classes();
    ensure_keypoints(&mcfg)?;
    let dims = mcfg.dims();
    let window = data.window;
    let geom = Geom {
        batch: train.batch_size,
        window,
        dims,
    };

    let mut run = Trainer::prepare(cfg, ModelKind::Behavior, &train)?;
    let class_refs: Vec<&str> = classes.iter().map(String::as_str).collect();
    let mut sessions = Vec::new();
    for dir in &data.session_dirs {
        sessions.extend(load_session_dir(
            repo_path(dir),
            SESSION_TARGET_FPS,
            data.session_groups.as_deref(),
            &class_refs,
        )?);
    }
    if let Some(root) = &data.real_root {
        reject_dlc_root(root)?;
    }
    if !sessions.is_empty() {
        println!(
            "{} labeled frames from {} sessions ({}% real batches)",
            sessions.iter().map(|s| s.frames).sum::<usize>(),
            sessions.len(),
            (data.real_fraction * 100.0).round()
        );
    }

    let pool = pool(&data, &train)?;
    let subsets = subset_masks(&data)?;
    let weights = class_weights(cfg, &sessions, &classes)?;
    let mut rng = Rng::new(train.seed as u64);
    let mut history = History::new();
    let started = Instant::now();

    for step in 1..=train.steps {
        // `train/loops.py:_Mix.__call__` chooses the source with one draw per batch; with no sessions
        // there is no `_Mix` at all, so no draw happens and every batch is synthetic.
        let (x, mut mask, y) = if sessions.is_empty() || rng.random() >= data.real_fraction {
            // `behavior/data.py:synthetic_batch`'s default corruption, `behavior/data.py:_MILD`.
            let batch = behavior_batch(
                train.batch_size,
                window,
                dims,
                &pool,
                &mut rng,
                &MILD_CORRUPTION,
            );
            (batch.x, batch.mask, batch.labels)
        } else {
            behavior_windows(
                &sessions,
                train.batch_size,
                window,
                &mut rng,
                BEHAVIOR_CONF_THRESH,
                BEHAVIOR_MIN_LABELED,
            )
            .with_context(|| {
                format!(
                    "no labeled window of {window} frames could be drawn from {} sessions \
                     (conf_thresh {BEHAVIOR_CONF_THRESH}, min_labeled {BEHAVIOR_MIN_LABELED})",
                    sessions.len()
                )
            })?
        };
        // `_KeypointSubset` wraps the mixed source — so real windows are thinned too — but
        // `train_behavior` only builds that wrapper `if sessions:`, which is why a purely synthetic
        // run neither draws nor applies a subset mask.
        if !sessions.is_empty() {
            if let Some(keep) = keypoint_subset_masks(&subsets, data.kp_subset_p, &mut rng) {
                apply_mask_mask(&mut mask, &keep, NUM_KEYPOINTS);
            }
        }
        let losses = run.step_behavior(&x, &mask, &y, geom, weights.as_deref())?;
        history.push(step, &losses);
        report(&train, step, "behavior", &losses, started);
    }
    run.finish(cfg, &history)
}

/// `train/loops.py:train_temporal`.
fn train_temporal(cfg: &Config) -> Result<()> {
    let train = cfg.train();
    let data = cfg.data();
    let mcfg = model_config(cfg)?;
    ensure_keypoints(&mcfg)?;
    let dims = mcfg.dims();
    let window = data.window;
    let geom = Geom {
        batch: train.batch_size,
        window,
        dims,
    };
    let mut run = Trainer::prepare(cfg, ModelKind::Temporal, &train)?;
    if let Some(root) = &data.real_root {
        reject_dlc_root(root)?;
    }
    println!(
        "[temporal] note: exp(-2*log_sigma) of temporal/model.py:temporal_loss is applied as a \
         constant factor (the tape has no `exp`), so log_sigma receives no NLL gradient"
    );

    let pool = pool(&data, &train)?;
    let mut rng = Rng::new(train.seed as u64);
    let mut history = History::new();
    let started = Instant::now();

    for step in 1..=train.steps {
        let batch = temporal_batch(
            train.batch_size,
            window,
            dims,
            &pool,
            &mut rng,
            &Corruption::default(),
        );
        // `train_temporal` wraps the synthetic source in `_KeypointSubset` only when
        // `data.real_root` is set, and that source is rejected above, so no batch is thinned here.
        let losses = run.step_temporal(&batch, geom)?;
        history.push(step, &losses);
        report(&train, step, "temporal", &losses, started);
    }
    run.finish(cfg, &history)
}

/// `data/behavior_sets.py:load_session_dir`'s `target_fps` default.
const SESSION_TARGET_FPS: f64 = 15.0;

/// `train/loops.py:_pool`: `SimulationPool(pool_sims, pool_frames, species, seed = seed + 100)`.
fn pool(data: &DataConfig, train: &TrainConfig) -> Result<SimulationPool> {
    let mut rng = Rng::new((train.seed + 100) as u64);
    SimulationPool::new(
        data.pool_sims,
        data.pool_frames,
        &data.species,
        POOL_MAX_ANIMALS,
        &mut rng,
    )
    .with_context(|| format!("cannot simulate the training pool of {:?}", data.species))
}

/// `train/loops.py:_KeypointSubset.__init__`'s `self.keep`: one canonical mask per mapping.
fn subset_masks(data: &DataConfig) -> Result<Vec<Vec<f32>>> {
    data.kp_subsets
        .iter()
        .map(|mapping| canonical_mask(mapping))
        .collect()
}

/// The keypoint count the model head and the batch builders must agree on.
fn ensure_keypoints(mcfg: &ModelConfig) -> Result<()> {
    anyhow::ensure!(
        mcfg.num_keypoints() == NUM_KEYPOINTS,
        "unsupported: the batch builders emit {NUM_KEYPOINTS} canonical keypoints, this model has {}",
        mcfg.num_keypoints()
    );
    Ok(())
}

/// `data.real_root` names a home-cage DLC tree; `load_sessions` reads it through HDF5.
fn reject_dlc_root(root: &str) -> Result<()> {
    bail!(
        "unsupported: `data.real_root: {root}` is read by \
         src/animalpose/data/tracks.py:load_sessions (load_h5, HDF5), which needs a C library; \
         train from the simulator, or write the recordings as session .npz directories with \
         animalpose behavior sessions and list them under `data.session_dirs`"
    )
}

/// Config paths are relative to the repository root, which is how `configs/*.yaml` writes them.
pub(crate) fn repo_path(path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() || candidate.exists() {
        return candidate;
    }
    match crate::data::repo_root() {
        Some(root) => root.join(candidate),
        None => candidate,
    }
}

fn model_config(cfg: &Config) -> Result<ModelConfig> {
    ModelConfig::from_json(&cfg.model_json()?).context("the `model:` block is not a model config")
}

/// `train_behavior`'s inverse-square-root class weights (`loss.class_balance`).
fn class_weights(
    cfg: &Config,
    sessions: &[Session],
    classes: &[String],
) -> Result<Option<Vec<f32>>> {
    if !cfg.bool_at("loss.class_balance", false) || sessions.is_empty() {
        return Ok(None);
    }
    let mut counts = vec![0f64; classes.len()];
    for session in sessions {
        for &label in &session.labels {
            if label >= 0 && (label as usize) < classes.len() {
                counts[label as usize] += 1.0;
            }
        }
    }
    let present: Vec<f64> = counts.iter().copied().filter(|c| *c > 0.0).collect();
    if present.is_empty() {
        return Ok(None);
    }
    let mean = present.iter().sum::<f64>() / present.len() as f64;
    // `np.where(counts > 0, 1 / sqrt(maximum(counts, 1) / counts[counts > 0].mean()), 1.0)`
    let mut w: Vec<f64> = counts
        .iter()
        .map(|c| {
            if *c > 0.0 {
                1.0 / (c.max(1.0) / mean).sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let normalise = w.iter().sum::<f64>() / w.len() as f64;
    for v in w.iter_mut() {
        *v /= normalise;
    }
    let pairs: Vec<String> = classes
        .iter()
        .zip(w.iter())
        .map(|(name, v)| format!("{name} {v:.2}"))
        .collect();
    println!("class weights {}", pairs.join(", "));
    Ok(Some(w.iter().map(|v| *v as f32).collect()))
}

/// `_log`: one line every `train.log_every` steps, `[name] step N <key>=<value>... (Ns)`, the loss
/// terms in the order `step_fn` put them in its dict.
fn report(train: &TrainConfig, step: usize, name: &str, losses: &[(&str, f64)], started: Instant) {
    if train.log_every == 0 || step % train.log_every != 0 {
        return;
    }
    let secs = started.elapsed().as_secs_f64();
    let terms: Vec<String> = losses
        .iter()
        .map(|(key, value)| format!("{key}={value:.4}"))
        .collect();
    println!("[{name}] step {step} {} ({secs:.0}s)", terms.join(" "));
}

/// The `_optimizer` pair, the model, and its trainable slots as plain buffers the optimizer rewrites.
struct Trainer {
    model: Model,
    names: Vec<String>,
    params: Vec<Vec<f32>>,
    clip: f64,
    opt: AdamW,
    schedule: Schedule,
    ctx: Ctx,
}

impl Trainer {
    /// `torch.manual_seed` + `build_model` + `_init` + `_optimizer`.
    fn prepare(cfg: &Config, kind: ModelKind, train: &TrainConfig) -> Result<Self> {
        let mcfg = model_config(cfg)?;
        let mut model = Model::build(kind, &mcfg, &Weights::new())
            .with_context(|| format!("cannot build {kind}"))?;
        if let Some(init) = cfg.get("init").and_then(|v| v.as_str()) {
            let path = repo_path(init);
            let (init_cfg, init_kind, weights) =
                crate::models::load(&path).with_context(|| format!("cannot read {init}"))?;
            if init_kind != kind {
                bail!("{init} holds a {init_kind} network, not a {kind} one");
            }
            if init_cfg.to_json_string()? != mcfg.to_json_string()? {
                bail!(
                    "{init} was built from a different model block:\n  checkpoint: {}\n  config:     {}",
                    init_cfg.to_json_string()?,
                    mcfg.to_json_string()?
                );
            }
            let report = model.load_weights(&weights);
            println!(
                "initialised from {init} (missing {}, unexpected {})",
                report.missing.len(),
                report.ignored.len()
            );
        }

        // `assign_slots` numbers the parameters so `Graph::param_grads` hands them back in order.
        assign_slots(&mut model, 0);
        let mut slots = slots_of(&mut model);
        slots.sort_by_key(|s| s.slot);
        let mut names = Vec::new();
        let mut params = Vec::new();
        for slot in slots.iter().filter(|s| s.trainable) {
            names.push(slot.name.clone());
            params.push(slot.data.clone());
        }
        Ok(Self {
            model,
            names,
            params,
            clip: train.clip,
            opt: AdamW::new(train.lr as f32, train.weight_decay as f32),
            schedule: Schedule::new(train.steps),
            ctx: Ctx::train(train.seed as u64),
        })
    }

    /// `_loop`'s body for one behavior batch: forward, `CrossEntropyLoss`, clip, `opt.step()`.
    ///
    /// `train_behavior`'s `step` returns the one-entry dict `{"total": ...}`, so that is the only
    /// term logged.
    fn step_behavior(
        &mut self,
        x: &[f32],
        mask: &[f32],
        labels: &[i64],
        geom: Geom,
        weights: Option<&[f32]>,
    ) -> Result<Vec<(&'static str, f64)>> {
        let (batch, window, dims) = (geom.batch, geom.window, geom.dims);
        let mut g = Graph::new();
        let input = g.from_data(x.to_vec(), &[batch, window, NUM_KEYPOINTS, dims]);
        let mask = g.from_data(mask.to_vec(), &[batch, window, NUM_KEYPOINTS]);
        let out = self
            .model
            .forward_named(&mut self.ctx, &mut g, &[input, mask])?;
        let logits = output(&out, "logits")?;
        let classes = g.shape(logits).last().copied().unwrap_or_default();
        let target = g.from_data(labels.iter().map(|v| *v as f32).collect(), &[batch, window]);
        let weight = weights.map(|w| g.from_data(w.to_vec(), &[classes]));
        // `nn.CrossEntropyLoss(weight=..., label_smoothing=0.05, ignore_index=-100)`.
        let loss = g.cross_entropy(logits, target, weight, 0.05, -100);
        self.optimize(&mut g, loss, &[])
    }

    /// `_loop`'s body for one temporal batch, with `temporal/model.py:temporal_loss` on the tape.
    ///
    /// The loss dict that function returns is `{"nll", "mse", "total"}`, in that order, and `_log`
    /// prints every entry, so all three come back.
    fn step_temporal(
        &mut self,
        batch_data: &TemporalBatch,
        geom: Geom,
    ) -> Result<Vec<(&'static str, f64)>> {
        let (batch, window, dims) = (geom.batch, geom.window, geom.dims);
        let (x, conf, observed) = (
            batch_data.x.as_slice(),
            batch_data.conf.as_slice(),
            batch_data.observed.as_slice(),
        );
        let (target, target_mask) = (
            batch_data.target.as_slice(),
            batch_data.target_mask.as_slice(),
        );
        let mut g = Graph::new();
        let input = g.from_data(x.to_vec(), &[batch, window, NUM_KEYPOINTS, dims]);
        let conf = g.from_data(conf.to_vec(), &[batch, window, NUM_KEYPOINTS]);
        let obs = g.from_data(observed.to_vec(), &[batch, window, NUM_KEYPOINTS]);
        let out = self
            .model
            .forward_named(&mut self.ctx, &mut g, &[input, conf, obs])?;
        let coords_norm = output(&out, "coords_norm")?;
        let log_sigma = output(&out, "log_sigma")?;

        // `tn = (nan_to_num(target) - center) / scale`. `center` `(B,1,1,D)` and `scale` `(B,1,1,1)`
        // are computed from the input coordinates, so they carry no gradient here either; folding
        // them into `tn` on the host is what Python's no-grad normalizer does.
        let center = g.value(output(&out, "center")?).to_vec();
        let scale = g.value(output(&out, "scale")?).to_vec();
        let mut tn = vec![0.0f32; target.len()];
        for (at, &t) in target.iter().enumerate() {
            let d = at % dims;
            let b = at / (window * NUM_KEYPOINTS * dims);
            let v = if t.is_nan() { 0.0 } else { t };
            tn[at] = (v - center[b * dims + d]) / scale[b];
        }
        let tn = g.from_data(tn, &[batch, window, NUM_KEYPOINTS, dims]);
        let diff = g.sub(coords_norm, tn);
        let square = g.mul(diff, diff);
        let err2 = g.reduce_sum(square, &[3]);
        let m = g.from_data(target_mask.to_vec(), &[batch, window, NUM_KEYPOINTS]);
        // `n = m.sum().clamp(min=1)`, and both terms divide by it.
        let n = f64::from(target_mask.iter().sum::<f32>()).max(1.0);
        let frozen = log_sigma_values(g.value(log_sigma));
        let frozen = g.from_data(frozen, &[batch, window, NUM_KEYPOINTS]);
        let weighted = g.mul(err2, frozen);
        let weighted = g.scale(weighted, 0.5);
        let entropy = g.scale(log_sigma, dims as f32);
        let nll_terms = g.add(weighted, entropy);
        let nll_masked = g.mul(nll_terms, m);
        let nll_sum = g.reduce_sum(nll_masked, &[0, 1, 2]);
        let nll = g.scale(nll_sum, (1.0 / n) as f32);
        let mse_masked = g.mul(err2, m);
        let mse_sum = g.reduce_sum(mse_masked, &[0, 1, 2]);
        let mse = g.scale(mse_sum, (1.0 / n) as f32);
        let total = g.add(nll, mse);
        self.optimize(&mut g, total, &[("nll", nll), ("mse", mse)])
    }

    /// `clip_grad_norm_` + `opt.step()` + `sched.step()`, then the new values go back to the slots.
    ///
    /// The values of `total` and of every node in `logged` are read before the step, in the order
    /// `step_fn` returns them; `total` last, as `_loop`'s dict gains that key last.
    fn optimize(
        &mut self,
        g: &mut Graph,
        total: NodeId,
        logged: &[(&'static str, NodeId)],
    ) -> Result<Vec<(&'static str, f64)>> {
        let mut losses: Vec<(&'static str, f64)> = logged
            .iter()
            .map(|(name, node)| Ok((*name, f64::from(g.value(*node)[0]))))
            .collect::<Result<_>>()?;
        losses.push(("total", f64::from(g.value(total)[0])));
        g.backward(total)?;
        let mut grads: Vec<Vec<f32>> = vec![Vec::new(); self.params.len()];
        for (slot, grad) in g.param_grads() {
            if slot < grads.len() {
                grads[slot] = grad;
            }
        }
        let clipped = clip_norm(&grads, self.clip);
        self.opt.set_scale(self.schedule.at(self.opt.step_count()));
        adamw_step(&mut self.params, &clipped, &self.names, &mut self.opt);
        self.write_back();
        Ok(losses)
    }

    /// Copies the optimizer's buffers back into the model, so the next forward uses them.
    fn write_back(&mut self) {
        let params = std::mem::take(&mut self.params);
        let mut next = 0usize;
        visit_slots(&mut self.model, &mut |slot| {
            if slot.trainable {
                if let Some(values) = params.get(next) {
                    slot.data = values.clone();
                }
                next += 1;
            }
        });
        self.params = params;
    }

    /// `save_checkpoint(cfg["output"], kind, model, cfg["model"], ...)`, plus the history log
    /// `animalpose train` writes next to it.
    fn finish(&mut self, cfg: &Config, history: &History) -> Result<()> {
        let Some(output) = cfg.get("output").and_then(|v| v.as_str()) else {
            return Ok(());
        };
        let mcfg = model_config(cfg)?;
        let kind = self.model.kind();
        let mut weights = Weights::new();
        for slot in slots_of(&mut self.model) {
            let data = if slot.trainable && slot.slot < self.params.len() {
                self.params[slot.slot].clone()
            } else {
                slot.data.clone()
            };
            weights.insert(&slot.name, &slot.shape, data);
        }
        // `save_checkpoint(..., {"window": window})` and friends: the non-tensor fields of the
        // checkpoint, which `animalpose export safetensors` puts in the header's `extra` block and
        // `animalpose export import-safetensors` folds back into a `.pt`. The values are the ones the
        // trainers read from the config (`loops.py:train_pose`'s `crop`, `train_behavior`'s `window`).
        let extra = match kind {
            ModelKind::Behavior | ModelKind::Temporal => {
                json!({"window": cfg.usize_at("data.window", 64)})
            }
            ModelKind::Pose => json!({"crop_size": cfg.usize_at("data.crop_size", 128)}),
            // Arena detectors are trained by scripts/train_arena.py; their input size is part of
            // the arena definition in the model config.
            ModelKind::Arena => json!({}),
            ModelKind::Detector => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let size: Vec<i64> = cfg
                    .f64_list_at("data.input_size", &[256.0, 320.0])
                    .iter()
                    .map(|v| v.round() as i64)
                    .collect();
                json!({"input_size": size})
            }
        };
        weights.set_metadata("extra", extra.to_string());
        let path = repo_path(output).with_extension("safetensors");
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        crate::models::save(&path, kind, &mcfg, &mut weights)
            .with_context(|| format!("cannot write {}", path.display()))?;
        let log = path.with_extension("history.json");
        std::fs::File::create(&log)
            .and_then(|mut f| f.write_all(history.json().as_bytes()))
            .with_context(|| format!("cannot write {}", log.display()))?;
        println!("saved {} and {}", path.display(), log.display());
        Ok(())
    }
}

/// The checkpoint tensor of `out` named `name`.
fn output(out: &[(&'static str, NodeId)], name: &str) -> Result<NodeId> {
    out.iter()
        .find(|(n, _)| *n == name)
        .map(|(_, node)| *node)
        .with_context(|| format!("the model returned no {name} output"))
}

/// `exp(-2 * log_sigma)`: the NLL's factor, held constant for the lack of an `exp` on the tape.
fn log_sigma_values(log_sigma: &[f32]) -> Vec<f32> {
    log_sigma
        .iter()
        .map(|v| (-2.0f32 * v).exp().min(f32::MAX))
        .collect()
}

/// `torch.nn.utils.clip_grad_norm_`: rescale the whole list once its global norm passes `clip`.
fn clip_norm(grads: &[Vec<f32>], clip: f64) -> Vec<Vec<f32>> {
    let norm = grads
        .iter()
        .flat_map(|g| g.iter())
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        .sqrt();
    let factor = if norm > clip && norm > 0.0 {
        (clip / norm) as f32
    } else {
        1.0
    };
    grads
        .iter()
        .map(|g| g.iter().map(|v| v * factor).collect())
        .collect()
}

/// Every checkpoint tensor of `model`, trainable and not.
fn slots_of(model: &mut Model) -> Vec<Slot> {
    let mut out = Vec::new();
    visit_slots(model, &mut |slot| out.push(slot.clone()));
    out
}

/// `layers::assign_slots` for the [`Model`] enum, whose variants are the layer trees.
fn assign_slots(model: &mut Model, base: usize) -> usize {
    match model {
        Model::Pose(m) => layers::assign_slots(&mut **m, base),
        Model::Detector(m) => layers::assign_slots(&mut **m, base),
        Model::Behavior(m) => layers::assign_slots(&mut **m, base),
        Model::Temporal(m) => layers::assign_slots(&mut **m, base),
    }
}

fn visit_slots(model: &mut Model, f: &mut dyn FnMut(&mut Slot)) {
    match model {
        Model::Pose(m) => m.each_layer_mut(&mut |layer| layer.visit_mut(f)),
        Model::Detector(m) => m.each_layer_mut(&mut |layer| layer.visit_mut(f)),
        Model::Behavior(m) => m.each_layer_mut(&mut |layer| layer.visit_mut(f)),
        Model::Temporal(m) => m.each_layer_mut(&mut |layer| layer.visit_mut(f)),
    }
}

/// `_log`'s `history`, which `animalpose train` dumps with `json.dumps`: one object per step holding
/// `step` and every loss term, in the order the terms are logged.
struct History(Vec<String>);

impl History {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn push(&mut self, step: usize, losses: &[(&str, f64)]) {
        let mut terms = vec![format!("\"step\":{step}")];
        for (name, value) in losses {
            // JSON has no spelling for a NaN loss; `json.dumps` writes `NaN`, which no parser reads.
            let number = serde_json::to_string(value).unwrap_or_else(|_| "null".to_string());
            terms.push(format!("\"{name}\":{number}"));
        }
        self.0.push(format!("{{{}}}", terms.join(",")));
    }

    fn json(&self) -> String {
        format!("[{}]", self.0.join(","))
    }
}
