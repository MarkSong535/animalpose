"""Training loops for the four networks.

Each ``train_*`` function takes a plain config dict (see ``configs/*.yaml``),
trains on the configured data source, saves a checkpoint with
:func:`~animalpose.train.checkpoint.save_checkpoint` and returns
``(model, history)``. Synthetic data is the default source so every loop runs
out of the box; real data plugs in through ``data.source: coco``.
"""

from __future__ import annotations

import math
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from animalpose.behavior.data import synthetic_batch as behavior_batch
from animalpose.data.behavior_sets import load_session_dir, remap_labels
from animalpose.data.datasets import (
    CocoKeypoints,
    DetectionDataset,
    MixedSource,
    PoseCropDataset,
    SyntheticScenes,
    detection_collate,
)
from animalpose.data.degrade import load_profile
from animalpose.data.synthetic import SimulationPool
from animalpose.data.tracks import behavior_windows, load_sessions, smooth_sessions, track_list
from animalpose.detector.model import build_targets as det_targets
from animalpose.detector.model import detector_loss
from animalpose.device import select_device
from animalpose.labels import BEHAVIORS_EXT
from animalpose.pose.model import build_targets as pose_targets
from animalpose.pose.model import pose_loss
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS
from animalpose.temporal.data import self_supervised_batch
from animalpose.temporal.data import synthetic_batch as temporal_batch
from animalpose.temporal.model import temporal_loss
from animalpose.train.checkpoint import build_model, load_model, save_checkpoint


def _optimizer(model, cfg, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    return opt, sched


def _source(data_cfg: dict, seed_offset: int = 0):
    """``data.source`` = synthetic | coco, or ``data.sources`` = list of such dicts with ``weight``."""
    if "sources" in data_cfg:
        subs = [_source({k: v for k, v in {**data_cfg, **s}.items() if k != "sources"}, seed_offset)
                for s in data_cfg["sources"]]
        return MixedSource(subs, [s.get("weight", 1.0) for s in data_cfg["sources"]], data_cfg.get("seed", 0) + seed_offset)
    kind = data_cfg.get("source", "synthetic")
    if kind == "synthetic":
        return SyntheticScenes(data_cfg.get("n_sims", 64), data_cfg.get("frames", 120),
                               tuple(data_cfg.get("image_size", (320, 256))),
                               tuple(data_cfg.get("species", ("mouse", "rat"))),
                               data_cfg.get("max_animals", 4), data_cfg.get("seed", 0) + seed_offset,
                               data_cfg.get("low_view_fraction", 0.0))
    if kind == "coco":
        return CocoKeypoints(data_cfg["json"], data_cfg["images"], data_cfg.get("mapping", "synthetic"),
                             data_cfg.get("species", "mouse"))
    raise ValueError(f"unknown data source {kind}")


def _init(model, cfg: dict):
    """Optionally start from an existing checkpoint (``init: runs/pose.pt``)."""
    if cfg.get("init"):
        state = torch.load(cfg["init"], map_location="cpu", weights_only=False)["state_dict"]
        own = model.state_dict()
        skipped = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        state = {k: v for k, v in state.items() if k not in skipped}  # e.g. a classifier with more classes
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"initialised from {cfg['init']} (missing {len(missing)}, unexpected {len(unexpected)}, "
              f"reshaped/skipped {skipped})")
    return model


def _log(history, step, losses, every, t0, name):
    history.append({"step": step, **{k: float(v) for k, v in losses.items()}})
    if every and step % every == 0:
        print(f"[{name}] step {step} " + " ".join(f"{k}={float(v):.4f}" for k, v in losses.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)


def _loop(model, loader, steps, cfg, step_fn, name):
    dev = next(model.parameters()).device
    opt, sched = _optimizer(model, cfg, steps)
    history, step, t0 = [], 0, time.time()
    model.train()
    while step < steps:
        for batch in loader:
            losses = step_fn(batch, dev)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 5.0))
            opt.step()
            sched.step()
            step += 1
            _log(history, step, {k: v.detach() for k, v in losses.items()}, cfg.get("log_every", 50), t0, name)
            if step >= steps:
                break
    return history


def train_detector(cfg: dict):
    tc = cfg.get("train", {})
    dev = select_device(cfg.get("device", "auto"))
    torch.manual_seed(tc.get("seed", 0))
    model = _init(build_model("detector", cfg["model"]), cfg).to(dev)
    size = tuple(cfg["data"].get("input_size", (256, 320)))
    ds = DetectionDataset(_source(cfg["data"]), tc.get("steps", 1000) * tc.get("batch_size", 16), size,
                          degrade=cfg["data"].get("degrade", 0.0), zoom_out=cfg["data"].get("zoom_out", 0.0),
                          profile=load_profile(cfg["data"].get("camera_profile")))
    loader = DataLoader(ds, tc.get("batch_size", 16), shuffle=False, num_workers=tc.get("workers", 0),
                        collate_fn=detection_collate)
    h, w = size[0] // model.stride, size[1] // model.stride

    def step(batch, dev):
        imgs, boxes, species, _ = batch
        out = model(imgs.to(dev))
        tgt = det_targets(boxes, (h, w), model.stride, species if model.num_species else None)
        return detector_loss(out, tgt)

    hist = _loop(model, loader, tc.get("steps", 1000), tc, step, "detector")
    if cfg.get("output"):
        save_checkpoint(cfg["output"], "detector", model, cfg["model"], {"input_size": size})
    return model.eval(), hist


def train_pose(cfg: dict):
    tc = cfg.get("train", {})
    dev = select_device(cfg.get("device", "auto"))
    torch.manual_seed(tc.get("seed", 0))
    model = _init(build_model("pose", cfg["model"]), cfg).to(dev)
    crop = cfg["data"].get("crop_size", 128)
    ds = PoseCropDataset(_source(cfg["data"], 1), tc.get("steps", 1000) * tc.get("batch_size", 32), crop,
                         degrade=cfg["data"].get("degrade", 0.0), profile=load_profile(cfg["data"].get("camera_profile")))
    loader = DataLoader(ds, tc.get("batch_size", 32), shuffle=False, num_workers=tc.get("workers", 0))
    hw = (crop // model.stride, crop // model.stride)
    lc = cfg.get("loss", {})

    def step(batch, dev):
        crops, kp, state, ids, _ = batch
        out = model(crops.to(dev))
        tgt = pose_targets(kp, state, hw, model.stride, lc.get("sigma", 1.5), lc.get("occluded_weight", 0.5))
        # Synthetic ids are scene * 16 + animal; real datasets give per-video track ids.
        groups = torch.div(ids, 16, rounding_mode="floor") if lc.get("group_by_scene", True) else None
        return pose_loss(out, tgt, ids, lc.get("offset_weight", 0.05), lc.get("embed_weight", 0.1), groups=groups)

    hist = _loop(model, loader, tc.get("steps", 1000), tc, step, "pose")
    if cfg.get("output"):
        save_checkpoint(cfg["output"], "pose", model, cfg["model"], {"crop_size": crop})
    return model.eval(), hist


class _NumpyBatches(torch.utils.data.IterableDataset):
    def __init__(self, fn, seed):
        self.fn, self.seed = fn, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((self.seed, info.id if info else 0))
        while True:
            yield tuple(torch.from_numpy(a) for a in self.fn(rng))


class _Mix:
    """Picklable chooser between a synthetic and a real batch function."""

    def __init__(self, synthetic, real, real_fraction: float):
        self.synthetic, self.real, self.real_fraction = synthetic, real, real_fraction

    def __call__(self, rng):
        return self.real(rng) if rng.random() < self.real_fraction else self.synthetic(rng)


class _KeypointSubset:
    """With probability ``p``, hide every keypoint one of the ``mappings`` does not label, so
    training also covers the sparse-skeleton regime of the real data: a home-cage DLC session has 6 of
    the 25 canonical points while a track our own pose model produced has all 25, and a classifier
    that only ever saw the first cannot read the second. Coordinates stay put (index 0 of a batch is
    the input the loss regresses); the models multiply points by the mask, so masking the mask is the
    same as deleting the point. ``mappings`` is one dataset name or a list, one skeleton per batch."""

    def __init__(self, fn, mappings, p: float = 0.5):
        self.fn, self.p = fn, p
        names = [mappings] if isinstance(mappings, str) else list(mappings)
        self.keep = [CANONICAL.mask(list(DATASET_MAPPINGS[m].values())).astype(np.float32) for m in names]

    def __call__(self, rng):
        batch = list(self.fn(rng))
        if self.p > 0 and rng.random() < self.p:
            keep = self.keep[int(rng.integers(len(self.keep)))]
            for i, a in enumerate(batch):  # observed / mask / target-mask arrays are (B, T, K)
                if a.ndim == 3 and a.shape[-1] == CANONICAL.num_keypoints and a.dtype == np.float32 and i > 0:
                    batch[i] = a * keep
        return tuple(batch)


def _pool(data_cfg: dict, species, seed: int) -> SimulationPool:
    return SimulationPool(data_cfg.get("pool_sims", 64), data_cfg.get("pool_frames", 600), species, seed=seed + 100)


def train_temporal(cfg: dict):
    tc = cfg.get("train", {})
    dev = select_device(cfg.get("device", "auto"))
    torch.manual_seed(tc.get("seed", 0))
    model = _init(build_model("temporal", cfg["model"]), cfg).to(dev)
    dims, window, bs = cfg["model"].get("dims", 2), cfg["data"].get("window", 64), tc.get("batch_size", 32)
    species = tuple(cfg["data"].get("species", ("mouse", "rat")))
    pool = _pool(cfg["data"], species, tc.get("seed", 0))
    fn = partial(temporal_batch, bs, window, dims=dims, species=species, pool=pool)
    if cfg["data"].get("real_root"):  # self-supervised masked reconstruction on real tracks
        tracks = track_list(load_sessions(cfg["data"]["real_root"], cfg["data"].get("real_animals")))
        fn = _Mix(_KeypointSubset(fn, cfg["data"].get("real_mapping", "homecage_dlc")),
                  partial(self_supervised_batch, tracks, bs, window), cfg["data"].get("real_fraction", 0.5))
    loader = DataLoader(_NumpyBatches(fn, tc.get("seed", 0)),
                        batch_size=None, num_workers=tc.get("workers", 0))

    def step(batch, dev):
        x, conf, obs, target, tmask = (b.to(dev) for b in batch)
        return temporal_loss(model(x, conf, obs), target, tmask)

    hist = _loop(model, loader, tc.get("steps", 2000), tc, step, "temporal")
    if cfg.get("output"):
        save_checkpoint(cfg["output"], "temporal", model, cfg["model"], {"window": window})
    return model.eval(), hist


def train_behavior(cfg: dict):
    tc = cfg.get("train", {})
    dev = select_device(cfg.get("device", "auto"))
    torch.manual_seed(tc.get("seed", 0))
    model = _init(build_model("behavior", cfg["model"]), cfg).to(dev)
    dims, window, bs = cfg["model"].get("dims", 2), cfg["data"].get("window", 64), tc.get("batch_size", 32)
    species = tuple(cfg["data"].get("species", ("mouse", "rat")))
    pool = _pool(cfg["data"], species, tc.get("seed", 0))
    fn = partial(behavior_batch, bs, window, dims=dims, species=species, pool=pool)
    sessions = []
    classes = tuple(model.classes)
    if cfg["data"].get("real_root"):  # real labelled tracks; unlabelled frames are ignored by the loss
        sessions = load_sessions(cfg["data"]["real_root"], cfg["data"].get("real_animals"))
        for s in sessions:
            s.labels = remap_labels(s.labels, BEHAVIORS_EXT, classes)
    for d in cfg["data"].get("session_dirs", []):  # video datasets tracked by the pose model
        sessions += load_session_dir(d, groups=cfg["data"].get("session_groups"), classes=classes)
    if sessions and cfg["data"].get("real_temporal"):  # clean the tracks with a temporal model first
        sessions = smooth_sessions(sessions, load_model(cfg["data"]["real_temporal"]).to(dev))
    if sessions:
        mapping = cfg["data"].get("real_mapping", "homecage_dlc")
        mixed = _Mix(fn, partial(behavior_windows, sessions, bs, window), cfg["data"].get("real_fraction", 0.5))
        # Applied after mixing so the sparse-skeleton regime reaches the real batches too: tracks made
        # by our own pose model are dense (all 25 points), tracks from DLC are not (6 of 25).
        fn = _KeypointSubset(mixed, cfg["data"].get("kp_subsets", [mapping]), cfg["data"].get("kp_subset_p", 0.5))
    loader = DataLoader(_NumpyBatches(fn, tc.get("seed", 0)),
                        batch_size=None, num_workers=tc.get("workers", 0))
    weight = None
    if cfg.get("loss", {}).get("class_balance") and sessions:
        # Inverse square-root frequency of the real labels; classes absent from real data keep weight 1.
        labels = np.concatenate([s.labels for s in sessions])
        counts = np.array([(labels == k).sum() for k in range(len(model.classes))], float)
        w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1) / counts[counts > 0].mean()), 1.0)
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        print("class weights", dict(zip(model.classes, np.round(w / w.mean(), 2))))
    ce = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05, ignore_index=-100)

    def step(batch, dev):
        x, mask, y = (b.to(dev) for b in batch)
        logits = model(x, mask)
        return {"total": ce(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))}

    hist = _loop(model, loader, tc.get("steps", 2000), tc, step, "behavior")
    if cfg.get("output"):
        save_checkpoint(cfg["output"], "behavior", model, cfg["model"], {"window": window})
    return model.eval(), hist


TRAINERS = {"detector": train_detector, "pose": train_pose, "temporal": train_temporal, "behavior": train_behavior}
