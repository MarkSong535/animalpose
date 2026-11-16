"""Evaluate temporal and behavior checkpoints on held-out real DLC tracks with human behavior labels.

Home-cage DeepLabCut tracks with BORIS labels, read in place.

Temporal: confident points inside random windows are hidden in contiguous
gaps (2-30 frames, per keypoint and whole-animal) and reconstructed; RMSE vs
linear interpolation on the hidden points only.

Behavior: frame-wise macro-F1 on human-labelled frames of the held-out
animal. Synthetic-only (5-class) models are scored on the four classes both
label sets share (walking, rearing, grooming, eating), with predictions
restricted to those classes; real-fine-tuned models are scored on the same
4-class protocol and on all 5 labelled classes (adding "other").

    animalpose eval real-tracks --root data/raw/home_cage --animals MOUSE1 \\
        --temporal runs/temporal.pt runs/temporal_real.pt --behavior runs/behavior.pt runs/behavior_real.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from animalpose.data.tracks import load_sessions, smooth_sessions
from animalpose.eval.behavior import protocol_scores
from animalpose.eval.metrics import linear_interpolate
from animalpose.temporal.data import Corruption, corrupt
from animalpose.train.checkpoint import load_model

NAME = "real-tracks"


def eval_temporal(model, sessions, n_windows: int = 400, window: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dev = next(model.parameters()).device
    gaps = Corruption(jitter=(0.0, 0.0), point_drop=0.0, point_gaps=0.03, point_gap_len=(2, 30), animal_gaps=0.7,
                      animal_gap_len=(5, 30), outliers=0.0, swaps=0.0)
    lin_e, lstm_e, gap_len = [], [], []
    for _ in range(n_windows):
        s = sessions[int(rng.integers(len(sessions)))]
        t0 = int(rng.integers(0, len(s.coords) - window))
        clean = np.nan_to_num(s.coords[t0:t0 + window]).astype(np.float32)
        conf = s.conf[t0:t0 + window].astype(np.float32)
        known = (conf >= 0.9) & np.isfinite(s.coords[t0:t0 + window]).all(-1)
        if known.sum() < window:
            continue
        _, c2, obs = corrupt(clean, known, rng, gaps)  # obs: known minus artificial gaps
        hidden = known & (obs == 0)
        if hidden.sum() == 0:
            continue
        x = clean * obs[..., None]
        lin = linear_interpolate(x, obs > 0)
        inputs = (x, np.where(obs > 0, conf, 0.0).astype(np.float32), obs)
        with torch.no_grad():
            out = model.eval()(*(torch.from_numpy(a[None]).to(dev) for a in inputs))["coords"][0].cpu().numpy()
        ok = hidden & np.isfinite(lin).all(-1)
        lin_e.append(np.linalg.norm(lin[ok] - clean[ok], axis=-1))
        lstm_e.append(np.linalg.norm(out[ok] - clean[ok], axis=-1))
        gap_len.append(_run_lengths(obs == 0)[ok])
    lin_e, lstm_e, gap_len = np.concatenate(lin_e), np.concatenate(lstm_e), np.concatenate(gap_len)
    res = {"gap_rmse_linear_px": float(np.sqrt(np.mean(lin_e**2))), "gap_rmse_lstm_px": float(np.sqrt(np.mean(lstm_e**2))),
           "gap_median_linear_px": float(np.median(lin_e)), "gap_median_lstm_px": float(np.median(lstm_e)),
           "n_hidden_points": int(len(lin_e))}
    for lo, hi in ((1, 5), (6, 15), (16, 64)):
        sel = (gap_len >= lo) & (gap_len <= hi)
        if sel.sum() > 50:
            res[f"median_linear_px_gap{lo}-{hi}"] = float(np.median(lin_e[sel]))
            res[f"median_lstm_px_gap{lo}-{hi}"] = float(np.median(lstm_e[sel]))
    return res


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Per element of a (T, K) boolean mask: length of the True run it belongs to (0 if False)."""
    out = np.zeros(mask.shape, int)
    for k in range(mask.shape[1]):
        t = 0
        while t < mask.shape[0]:
            if mask[t, k]:
                e = t
                while e < mask.shape[0] and mask[e, k]:
                    e += 1
                out[t:e, k] = e - t
                t = e
            else:
                t += 1
    return out


def add_arguments(parser) -> None:
    parser.add_argument("--root", default="data/raw/home_cage")
    parser.add_argument("--animals", nargs="+", required=True)
    parser.add_argument("--temporal", nargs="*", default=[])
    parser.add_argument("--behavior", nargs="*", default=[])
    parser.add_argument("--smooth", default=None, help="temporal checkpoint that cleans tracks before behavior")
    parser.add_argument("--out", default="results/real_tracks.json")


def run(args) -> None:
    sessions = load_sessions(args.root, args.animals)
    beh_sessions = smooth_sessions(sessions, load_model(args.smooth)) if args.smooth else sessions
    views = sorted({s.view for s in sessions})
    res: dict = {"held_out_animals": args.animals, "temporal": {}, "behavior": {}}
    for path in args.temporal:
        model = load_model(path)
        res["temporal"][path] = {v: eval_temporal(model, [s for s in sessions if s.view == v]) for v in views}
        print(path, json.dumps(res["temporal"][path]), flush=True)
    for path in args.behavior:
        model = load_model(path)
        res["behavior"][path] = {v: protocol_scores(model, [s for s in beh_sessions if s.view == v])
                            for v in views}
        print(path, json.dumps({v: {k: r[k]["macro_f1"] for k in r} for v, r in res["behavior"][path].items()}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
