"""Real two-mouse tracking test on CRIM13 out-of-distribution videos.

CRIM13 pairs a black resident with a white intruder, so the coat brightness
inside each tracked box gives a per-frame identity proxy without manual
labels. For each track we take its majority coat; a *swap* is a run of at
least ``--min-run`` frames whose coat disagrees with the track's majority
(the track jumped to the other mouse). Also reported: identity purity (share
of frames matching the majority coat), the number of track ids created, and
coverage (share of frames with both mice tracked).

Configurations compared: no appearance term, color-histogram appearance, and
the pose network's learned embedding; ``max_animals=2`` in all of them.

    animalpose eval real-tracking --detector runs/detector_real.pt --pose runs/pose_real.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.checkpoint import load_model

NAME = "real-tracking"


def coat_brightness(image: np.ndarray, box: np.ndarray) -> float:
    x0, y0, x1, y1 = box
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
    patch = image[int(max(0, cy - h)):int(cy + h + 1), int(max(0, cx - w)):int(cx + w + 1)]
    return float(np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))) if patch.size else np.nan


def score(result, frames) -> dict:
    vals = {tid: np.array([coat_brightness(frames[t], tr.boxes[i]) if np.isfinite(tr.boxes[i]).all() else np.nan
                           for i, t in enumerate(tr.frames)]) for tid, tr in result.tracks.items()}
    allv = np.concatenate([v[np.isfinite(v)] for v in vals.values()]) if vals else np.zeros(0)
    if len(allv) < 10:
        return {"swaps": 0, "purity": float("nan"), "n_ids": len(vals), "coverage": 0.0}
    thr, _ = cv2.threshold(allv.astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    swaps, agree, total = 0, 0, 0
    for v in vals.values():
        ok = np.isfinite(v)
        if ok.sum() == 0:
            continue
        coat = v[ok] > thr
        major = coat.mean() >= 0.5
        wrong = coat != major
        agree += int((~wrong).sum())
        total += int(len(coat))
        run = 0
        for w in wrong:  # count runs of disagreeing frames longer than min_run
            run = run + 1 if w else 0
            if run == score.min_run:
                swaps += 1
    per_frame = np.zeros(len(frames))
    for tr in result.tracks.values():
        per_frame[tr.frames[np.isfinite(tr.boxes).all(-1)]] += 1
    return {"swaps": swaps, "purity": agree / max(total, 1), "n_ids": len(vals), "coverage": float((per_frame >= 2).mean())}


def add_arguments(parser) -> None:
    parser.add_argument("--detector", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--videos", default="data/raw/lp_crim13/extracted/crim13/videos_OOD")
    parser.add_argument("--n-videos", type=int, default=6)
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--min-run", type=int, default=10)
    parser.add_argument("--configs", nargs="+", default=["no_appearance", "color_histogram", "learned_embedding"])
    parser.add_argument("--w-app", type=float, default=0.2, help="appearance weight in the association cost")
    parser.add_argument("--tag", default="", help="prefix for result keys, e.g. the pose model")
    parser.add_argument("--out", default="results/real_tracking.json")


def run(args) -> None:
    score.min_run = args.min_run
    det, pose = load_model(args.detector), load_model(args.pose)
    videos = sorted(Path(args.videos).glob("*.mp4"))[: args.n_videos]
    configs = {"no_appearance": ("none", args.w_app), "color_histogram": ("color", args.w_app),
               "learned_embedding": ("model", args.w_app)}
    configs = {k: v for k, v in configs.items() if k in args.configs}
    res = {}
    for name, (embedder, w_app) in configs.items():
        cfg = PipelineConfig(crop_size=args.crop_size, embedder=embedder,
                             tracker=TrackerConfig(max_animals=2, w_app=w_app))
        pipe = PosePipeline(det, pose, config=cfg)
        per = []
        for v in videos:
            cap = cv2.VideoCapture(str(v))
            frames = []
            while len(frames) < args.frames:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            per.append(score(pipe.track(frames), frames))
        res[name] = {"swaps_total": int(sum(p["swaps"] for p in per)),
                     "purity_mean": float(np.nanmean([p["purity"] for p in per])),
                     "ids_per_video": float(np.mean([p["n_ids"] for p in per])),
                     "coverage_mean": float(np.mean([p["coverage"] for p in per])), "videos": len(per),
                     "frames_per_video": args.frames}
        print(name, json.dumps(res[name]), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update({f"{args.tag}{k}@w_app={args.w_app}": v for k, v in res.items()})
    out.write_text(json.dumps(prev, indent=2))
