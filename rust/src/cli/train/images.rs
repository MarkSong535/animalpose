//! `train/loops.py:train_pose` / `train_detector`: the image networks, trained on rendered or COCO
//! frames.
//!
//! The same loop as the keypoint trainers of the parent module — `torch.manual_seed`,
//! `build_model` + `_init`, `_optimizer` (AdamW plus warmup-cosine), `_loop`'s gradient clipping and
//! log line, `save_checkpoint` with `{"crop_size": ...}` / `{"input_size": ...}` — over the batches
//! of [`PoseCropDataset`] / [`DetectionDataset`] fed by `_source` ([`source_from_config`]):
//! item `i` of batch `s` is dataset index `s * batch_size + i`, exactly the `DataLoader(...,
//! shuffle=False)` order. Items are built on `train.workers` threads while the previous step runs,
//! the way the `DataLoader` workers overlap with training.
//!
//! The losses are [`crate::models::losses::pose_loss`] / [`crate::models::losses::detector_loss`],
//! evaluated on the host; their exact output gradients enter the tape through the surrogate
//! `sum(output * grad)` (see that module), so `loss.backward()` reaches the same parameter
//! gradients.

use std::ops::Range;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};

use super::{model_config, output, report, History, Trainer};
use crate::data::config::{Config, Value};
use crate::data::datasets::{
    source_from_config, DetectionDataset, DetectionItem, PoseCropDataset, PoseItem,
};
use crate::data::degrade::load_profile;
use crate::models::losses::{detector_loss, detector_targets, pose_loss, pose_targets};
use crate::models::{Model, ModelKind};
use crate::tensor::{Graph, NodeId};

/// `train/loops.py:train_pose`.
pub(super) fn train_pose(cfg: &Config) -> Result<()> {
    let train = cfg.train();
    let mcfg = model_config(cfg)?;
    let mut run = Trainer::prepare(cfg, ModelKind::Pose, &train)?;
    let stride = match &run.model {
        Model::Pose(m) => m.stride(),
        _ => bail!("the pose trainer built a {} network", run.model.kind()),
    };
    let k = mcfg.num_keypoints();
    ensure!(
        k == crate::data::datasets::K,
        "unsupported: the pose datasets label the {} canonical keypoints, this model has {k}",
        crate::data::datasets::K
    );
    let crop = cfg.usize_at("data.crop_size", 128);
    ensure!(
        crop % 32 == 0 && crop > 0,
        "data.crop_size must be a multiple of 32, got {crop}"
    );
    let bs = train.batch_size;
    let data = data_block(cfg);
    let ds = PoseCropDataset::new(
        source_from_config(&data, 1)?,
        train.steps * bs,
        crop,
        cfg.f64_at("data.degrade", 0.0),
        load_profile(cfg.get("data.camera_profile"))?,
    );
    let hw = (crop / stride, crop / stride);
    let sigma = cfg.f64_at("loss.sigma", 1.5);
    let occluded_weight = cfg.f64_at("loss.occluded_weight", 0.5);
    let offset_weight = cfg.f64_at("loss.offset_weight", 0.05);
    let embed_weight = cfg.f64_at("loss.embed_weight", 0.1);
    let group_by_scene = cfg.bool_at("loss.group_by_scene", true);

    let mut history = History::new();
    let started = Instant::now();
    let build = |step: usize| -> Result<Vec<PoseItem>> {
        build_items(batch_range(step, bs), train.workers, &|i| ds.get(i))
    };
    run_steps(train.steps, &build, |step, items| {
        let mut crops = Vec::with_capacity(bs * 3 * crop * crop);
        let mut kpts = Vec::with_capacity(bs * k);
        let mut state = Vec::with_capacity(bs * k);
        let mut ids = Vec::with_capacity(bs);
        for it in &items {
            crops.extend_from_slice(&it.crop);
            kpts.extend_from_slice(&it.keypoints);
            state.extend_from_slice(&it.state);
            ids.push(it.id);
        }
        let b = items.len();
        let mut g = Graph::new();
        let x = g.from_data(crops, &[b, 3, crop, crop]);
        let out = run.model.forward_named(&mut run.ctx, &mut g, &[x])?;
        let (heat, offset, emb) = (
            output(&out, "heat")?,
            output(&out, "offset")?,
            output(&out, "embedding")?,
        );
        let embed_dim = g.shape(emb).get(1).copied().unwrap_or(0);
        let tgt = pose_targets(&kpts, &state, b, k, hw, stride, sigma, occluded_weight, 2.0);
        // Synthetic ids are `scene * 16 + animal`; real datasets give per-video track ids.
        let groups: Option<Vec<i64>> =
            group_by_scene.then(|| ids.iter().map(|i| i.div_euclid(16)).collect());
        let loss = pose_loss(
            g.value(heat),
            g.value(offset),
            g.value(emb),
            embed_dim,
            &tgt,
            Some(&ids),
            offset_weight,
            embed_weight,
            10.0,
            groups.as_deref(),
        );
        let seeds = [
            (heat, &loss.grads[0]),
            (offset, &loss.grads[1]),
            (emb, &loss.grads[2]),
        ];
        step_with_grads(&mut run, &mut g, &seeds)?;
        history.push(step, &loss.terms);
        report(&train, step, "pose", &loss.terms, started);
        Ok(())
    })?;
    run.finish(cfg, &history)
}

/// `train/loops.py:train_detector`.
pub(super) fn train_detector(cfg: &Config) -> Result<()> {
    let train = cfg.train();
    let mut run = Trainer::prepare(cfg, ModelKind::Detector, &train)?;
    let (stride, num_species) = match &run.model {
        Model::Detector(m) => (m.stride(), m.num_species()),
        _ => bail!("the detector trainer built a {} network", run.model.kind()),
    };
    let size = cfg.f64_list_at("data.input_size", &[256.0, 320.0]);
    ensure!(size.len() == 2, "data.input_size is [H, W]");
    let size = (size[0] as usize, size[1] as usize);
    ensure!(
        size.0 % 32 == 0 && size.1 % 32 == 0 && size.0 > 0 && size.1 > 0,
        "data.input_size must be multiples of 32, got {size:?}"
    );
    let bs = train.batch_size;
    let data = data_block(cfg);
    let ds = DetectionDataset {
        source: source_from_config(&data, 0)?,
        length: train.steps * bs,
        size,
        augment: true,
        degrade: cfg.f64_at("data.degrade", 0.0),
        zoom_out: cfg.f64_at("data.zoom_out", 0.0),
        profile: load_profile(cfg.get("data.camera_profile"))?,
    };
    let (h, w) = (size.0 / stride, size.1 / stride);

    let mut history = History::new();
    let started = Instant::now();
    let build = |step: usize| -> Result<Vec<DetectionItem>> {
        build_items(batch_range(step, bs), train.workers, &|i| ds.get(i))
    };
    run_steps(train.steps, &build, |step, items| {
        let b = items.len();
        let mut images = Vec::with_capacity(b * 3 * size.0 * size.1);
        let mut boxes = Vec::with_capacity(b);
        let mut species = Vec::with_capacity(b);
        for it in items {
            images.extend_from_slice(&it.image);
            boxes.push(it.boxes);
            species.push(it.species);
        }
        let mut g = Graph::new();
        let x = g.from_data(images, &[b, 3, size.0, size.1]);
        let out = run.model.forward_named(&mut run.ctx, &mut g, &[x])?;
        let (heat, size_map, offset) = (
            output(&out, "heat")?,
            output(&out, "size")?,
            output(&out, "offset")?,
        );
        let spc = if num_species > 0 {
            Some(output(&out, "species")?)
        } else {
            None
        };
        let tgt = detector_targets(
            &boxes,
            (h, w),
            stride,
            (num_species > 0).then_some(species.as_slice()),
        );
        let loss = detector_loss(
            g.value(heat),
            g.value(size_map),
            g.value(offset),
            spc.map(|s| (g.value(s), num_species)),
            &tgt,
            0.1,
            1.0,
        );
        let mut seeds = vec![
            (heat, &loss.grads[0]),
            (size_map, &loss.grads[1]),
            (offset, &loss.grads[2]),
        ];
        if let Some(s) = spc {
            seeds.push((s, &loss.grads[3]));
        }
        step_with_grads(&mut run, &mut g, &seeds)?;
        history.push(step, &loss.terms);
        report(&train, step, "detector", &loss.terms, started);
        Ok(())
    })?;
    run.finish(cfg, &history)
}

/// The `data:` block, or an empty mapping.
fn data_block(cfg: &Config) -> Value {
    cfg.get("data")
        .cloned()
        .unwrap_or_else(|| Value::Map(Default::default()))
}

/// Dataset indices of 1-based training step `step`.
fn batch_range(step: usize, bs: usize) -> Range<u64> {
    ((step - 1) * bs) as u64..(step * bs) as u64
}

/// Builds `range` with `workers` threads (`0` behaves like `1`, the main thread), keeping the order.
fn build_items<T: Send>(
    range: Range<u64>,
    workers: usize,
    item: &(dyn Fn(u64) -> Result<T> + Sync),
) -> Result<Vec<T>> {
    let idx: Vec<u64> = range.collect();
    let threads = workers.max(1).min(idx.len().max(1));
    if threads == 1 {
        return idx.into_iter().map(item).collect();
    }
    let chunk = idx.len().div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = idx
            .chunks(chunk)
            .map(|part| s.spawn(move || part.iter().map(|i| item(*i)).collect::<Result<Vec<T>>>()))
            .collect();
        let mut out = Vec::with_capacity(idx.len());
        for h in handles {
            out.extend(
                h.join()
                    .map_err(|_| anyhow::anyhow!("a data worker panicked"))??,
            );
        }
        Ok(out)
    })
}

/// Steps `1..=steps`, building the next batch on a background thread while the current one trains.
fn run_steps<T: Send>(
    steps: usize,
    build: &(dyn Fn(usize) -> Result<Vec<T>> + Sync),
    mut train_step: impl FnMut(usize, Vec<T>) -> Result<()>,
) -> Result<()> {
    if steps == 0 {
        return Ok(());
    }
    let mut next = Some(build(1)?);
    for step in 1..=steps {
        let items = next.take().context("the batch of this step is missing")?;
        next = std::thread::scope(|s| -> Result<Option<Vec<T>>> {
            let ahead = (step < steps).then(|| s.spawn(move || build(step + 1)));
            train_step(step, items)?;
            match ahead {
                Some(h) => {
                    Ok(Some(h.join().map_err(|_| {
                        anyhow::anyhow!("the data thread panicked")
                    })??))
                }
                None => Ok(None),
            }
        })?;
    }
    Ok(())
}

/// `loss.backward()` + `_loop`'s clip / `opt.step()` / `sched.step()` for a loss whose output
/// gradients are known: the surrogate `sum_k sum(out_k * grad_k)` has exactly the loss's parameter
/// gradient.
fn step_with_grads(run: &mut Trainer, g: &mut Graph, seeds: &[(NodeId, &Vec<f32>)]) -> Result<()> {
    let mut total: Option<NodeId> = None;
    for (node, grad) in seeds {
        let shape = g.shape(*node).clone();
        ensure!(
            grad.len() == Graph::numel(&shape),
            "a loss gradient of {} values for an output of shape {shape:?}",
            grad.len()
        );
        let seed = g.from_data(grad.to_vec(), &shape);
        let prod = g.mul(*node, seed);
        let sum = g.sum_all(prod);
        total = Some(match total {
            Some(t) => g.add(t, sum),
            None => sum,
        });
    }
    let total = total.context("no output to train")?;
    run.optimize(g, total, &[])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_keep_the_dataloader_order_on_any_number_of_workers() {
        let one = build_items(batch_range(3, 5), 1, &|i| Ok(i * 10)).unwrap();
        let four = build_items(batch_range(3, 5), 4, &|i| Ok(i * 10)).unwrap();
        assert_eq!(one, vec![100, 110, 120, 130, 140]);
        assert_eq!(one, four);
        assert!(build_items(0..3, 2, &|i| if i == 1 { bail!("bad item") } else { Ok(i) }).is_err());
    }

    /// `PoseNet(context=True)`: the FiLM head starts as the identity and trains like the rest.
    #[test]
    fn a_pose_model_with_film_context_trains() {
        let cfg = Config::parse(
            "model: {num_keypoints: 25, backbone: {name: tiny, width: 8, out_channels: 16}, \
             embed_dim: 8, context: true}\n\
             data: {source: synthetic, crop_size: 32, image_size: [96, 64], n_sims: 2, frames: 30}\n\
             train: {steps: 2, batch_size: 4, log_every: 0}",
            &[],
        )
        .unwrap();
        let film = Trainer::prepare(&cfg, ModelKind::Pose, &cfg.train())
            .unwrap()
            .model
            .slot_names()
            .into_iter()
            .filter(|n| n.starts_with("context."))
            .count();
        assert_eq!(film, 4);
        let mut model = Trainer::prepare(&cfg, ModelKind::Pose, &cfg.train())
            .unwrap()
            .model;
        let mut zero = true;
        super::super::visit_slots(&mut model, &mut |s| {
            if s.name.starts_with("context.2.") {
                zero &= s.data.iter().all(|v| *v == 0.0);
            }
        });
        assert!(zero, "context.2 starts at zero");
        train_pose(&cfg).unwrap();
    }
}
