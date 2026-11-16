"""Pseudo-label an unlabeled collection of recordings, for self-training.

Every video file, and every folder that holds image frames, under ``root`` is one recording.
About 1 in 5 recordings (fixed by a hash of its path) is held out and never labeled. From the
others, frames are sampled evenly, and the current models (the teacher) label them. Labels are
kept only when the teacher is confident and self-consistent:

* **detections**: score >= ``hi`` after the static-object (median background) filter. Frames with
  an uncertain detection (``lo`` <= score < ``hi``) are skipped entirely, so the detector is never
  taught that a missed animal is background.
* **keypoints**: confidence >= ``kp_conf`` *and* the same point found again on the horizontally
  flipped frame (left/right swapped back) within ``flip_tol`` box widths. Animals with fewer than
  ``min_kps`` such points are skipped.
* **moving pixels** (recordings with a background): the box is shrunk to the pixels that differ
  from the background, and keypoints off them are dropped, so an animal touching a static object
  (a cup, a wall) does not teach the object as part of the animal.

``boxes="blob"`` is for a fixed camera with one animal that contrasts with the floor (a dark mouse
on white): the box comes from the largest moving blob instead of the detector: the pixels darker
than the median background inside the bright arena, opened to drop the thin tail. Frames with no
blob of plausible size, or with a second blob of comparable size, are skipped, and so are frames
whose labelled nose is closer to the visible tail than the labelled tail base (a head/tail flip).

Writes ``<out>/train.json`` (COCO, canonical names), ``<out>/images/`` and ``<out>/split.json``.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from animalpose.data.real import CocoWriter
from animalpose.frames import find_sources, is_held_out
from animalpose.pipeline import PipelineConfig, PosePipeline, arena_roi, blob_box, foreground_fraction, median_background
from animalpose.skeleton import CANONICAL
from animalpose.train.checkpoint import load_model


@dataclass(frozen=True)
class LabelOptions:
    """The teacher and the acceptance thresholds of a labeling run (see the module docstring).

    Field metadata drives ``animalpose label --help``: ``help`` is the flag help, ``flag``/``nargs``/
    ``metavar``/``choices``/``action`` shape the flag. One default, one place to read it.
    """

    detector: str | Path = field(default="weights/detector_general.pt",
                                 metadata={"help": "the teacher detector"})
    pose: str | Path = field(default="weights/pose_general.pt",
                             metadata={"help": "the teacher pose model"})
    out: str | Path = field(default="data/processed/unlabeled_pseudo",
                            metadata={"help": "where train.json, split.json and images/ go"})
    det_size: tuple[int, int] = field(default=(480, 640),
                                      metadata={"flag": "--det-size", "nargs": 2, "type": int,
                                                "metavar": ("H", "W"), "help": "detector input size"})
    crop_size: int = field(default=192, metadata={"help": "pose crop size"})
    per_source: int = field(default=200,
                            metadata={"flag": "--per-source", "type": int, "help": "frames sampled per recording"})
    hi: float = field(default=0.5, metadata={"type": float, "help": "detection score that counts as an animal"})
    lo: float = field(default=0.2,
                      metadata={"type": float, "help": "below this the frame is background; between: skip"})
    kp_conf: float = field(default=0.5, metadata={"flag": "--kp-conf", "type": float,
                                                  "help": "keypoint confidence to keep"})
    flip_tol: float = field(default=0.05, metadata={"flag": "--flip-tol", "type": float,
                                                    "help": "keypoint flip agreement, in box widths"})
    min_kps: int = field(default=4, metadata={"flag": "--min-kps", "type": int,
                                              "help": "keypoints an animal needs to be labelled"})
    min_seq: int = field(default=30, metadata={"flag": "--min-seq", "type": int,
                                               "help": "frames needed to use the background filter"})
    boxes: str = field(default="detector", metadata={"choices": ["detector", "blob"],
                                                     "help": "'blob': one animal contrasting with the floor"})
    time_range: tuple[float, float] = field(default=(0.0, 1.0),
                                            metadata={"flag": "--range", "nargs": 2, "type": float,
                                                      "metavar": ("FROM", "TO"),
                                                      "help": "fraction of each recording to label"})
    holdout: bool = field(default=True, metadata={"flag": "--no-holdout", "action": "store_false",
                                                  "help": "label every recording, not 4 in 5"})
    include: tuple[str, ...] | None = field(default=None,
                                            metadata={"nargs": "+",
                                                      "help": "only recordings whose path contains one of these"})
    progress: bool = field(default=True, metadata={"flag": "--quiet", "action": "store_false",
                                                   "help": "print nothing until the summary"})


def source_id(path: str | Path, root: str | Path) -> str:
    """Stable folder name for one recording, so relabels overwrite the same images."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return f"{zlib.crc32(rel.encode()):08x}_{re.sub(r'[^A-Za-z0-9]+', '_', Path(rel).stem)[-40:]}"


def flip_estimate(pipe: PosePipeline, img: np.ndarray, boxes: np.ndarray):
    """Pose on the mirrored frame, mapped back to the original frame and labels."""
    w = img.shape[1]
    fb = boxes.copy()
    fb[:, [0, 2]] = w - boxes[:, [2, 0]]
    kp, cf, _ = pipe.estimate(np.ascontiguousarray(img[:, ::-1]), fb)
    kp[..., 0] = w - 1 - kp[..., 0]
    perm = CANONICAL.flip_index
    return kp[:, perm], cf[:, perm]


def consistent_keypoints(pipe: PosePipeline, img: np.ndarray, boxes: np.ndarray, kp_conf: float, flip_tol: float):
    """``(kp (N,K,2), ok (N,K), conf)``: keypoints confident and stable under a horizontal flip."""
    kp, cf, _ = pipe.estimate(img, boxes)
    kf, cff = flip_estimate(pipe, img, boxes)
    size = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])[:, None]
    ok = (cf >= kp_conf) & (cff >= kp_conf) & (np.linalg.norm(kp - kf, axis=-1) <= flip_tol * size)
    return (kp + kf) / 2, ok, cf


def foreground_mask(img: np.ndarray, bg: np.ndarray, diff: float, grow: int = 3) -> np.ndarray:
    """Foreground pixels near ``img``, dilated to cover the animal's outline."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = (np.abs(g - bg) > diff).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)


def fit_to_foreground(box: np.ndarray, fg: np.ndarray, pad: float = 0.1) -> np.ndarray | None:
    """Box shrunk to the foreground pixels inside it (``None`` if there are almost none)."""
    h, w = fg.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    ys, xs = np.nonzero(fg[y0:y1, x0:x1])
    if len(xs) < 20:
        return None
    lo, hi = np.array([xs.min() + x0, ys.min() + y0], float), np.array([xs.max() + x0 + 1, ys.max() + y0 + 1], float)
    d = (hi - lo) * pad
    return np.array([max(lo[0] - d[0], 0), max(lo[1] - d[1], 0), min(hi[0] + d[0], w), min(hi[1] + d[1], h)], np.float32)


def tail_centroid(img: np.ndarray, bg: np.ndarray, box: np.ndarray, diff: float, open_px: int = 5):
    """Centre of the thin dark pixels (the tail) near ``box``: dark pixels the body opening removed."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = ((bg - g) > diff).astype(np.uint8)
    body = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8)), np.ones((3, 3), np.uint8))
    w, h = box[2] - box[0], box[3] - box[1]
    x0, y0, x1, y1 = int(max(box[0] - w / 2, 0)), int(max(box[1] - h / 2, 0)), int(box[2] + w / 2), int(box[3] + h / 2)
    ys, xs = np.nonzero((m & (1 - body))[y0:y1, x0:x1])
    return None if len(xs) < 15 else np.array([xs.mean() + x0, ys.mean() + y0])


def background(src, n_frames: int, k: int = 64) -> np.ndarray:
    """Median of ``k`` frames spread over the recording: the static scene."""
    idx = np.linspace(0, n_frames - 1, min(k, n_frames)).astype(int)
    return median_background([f for _, f in src.frames(idx)], k)


def label_videos(root: str | Path, options: LabelOptions | None = None,
                 **changes: Any) -> dict:
    """Label sampled frames of every recording under ``root``; returns the run statistics.

    ``options`` carries the teacher and every threshold; ``changes`` override single fields, so
    ``label_videos(root, boxes="blob")`` is the short form of passing a full options object.
    """
    opts = replace(options or LabelOptions(), **changes)
    pipe = PosePipeline(load_model(opts.detector), load_model(opts.pose),
                        config=PipelineConfig(crop_size=opts.crop_size, det_size=tuple(opts.det_size),
                                            det_thresh=opts.lo))
    out = Path(opts.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sources = find_sources(root)
    if opts.include:
        sources = [s for s in sources if any(k in str(s.path) for k in opts.include)]
        if opts.progress:
            print(f"{len(sources)} recordings match the filter", flush=True)
    split = {"root": os.path.abspath(root), "train": [], "held_out": []}
    writer, stats = CocoWriter(), {"sources": len(sources), "frames": 0, "labeled": 0, "skipped_uncertain": 0,
                                   "no_animal": 0, "animals": 0, "keypoints": 0, "head_tail_flip": 0}
    for src in sources:
        rel = os.path.relpath(os.path.abspath(src.path), os.path.abspath(root))
        if opts.holdout and is_held_out(src, root):
            split["held_out"].append(rel)
            continue
        split["train"].append(rel)
        n = src.n_frames
        if n == 0:
            continue
        bg = background(src, n) if n >= opts.min_seq else None
        if opts.boxes == "blob" and bg is None:
            continue
        roi = arena_roi(bg) if opts.boxes == "blob" else None
        areas = None
        sid = source_id(src.path, root)
        (out / "images" / sid).mkdir(exist_ok=True)
        lo_f, hi_f = int(opts.time_range[0] * (n - 1)), int(opts.time_range[1] * (n - 1))
        idx = np.linspace(lo_f, hi_f, min(opts.per_source, hi_f - lo_f + 1)).astype(int)
        if opts.boxes == "blob":  # typical blob size from a quick pass, to reject odd ones
            sizes = []
            for _, im in src.frames(idx[:: max(1, len(idx) // 50)]):
                b = blob_box(im, bg, pipe.cfg.foreground_diff, roi=roi)
                if b is not None:
                    sizes.append((b[2] - b[0]) * (b[3] - b[1]) / 1.69)  # undo the 15 % padding
            med = float(np.median(sizes)) if sizes else None
            areas = (0.15 * med, 2.5 * med) if med else None
        for t, img in src.frames(idx):
            stats["frames"] += 1
            if opts.boxes == "blob":
                b = blob_box(img, bg, pipe.cfg.foreground_diff, area=None, roi=roi)
                if b is None or (areas and not areas[0] <= (b[2] - b[0]) * (b[3] - b[1]) / 1.69 <= areas[1]):
                    stats["no_animal"] += 1
                    continue
                boxes_, scores = b[None], np.ones(1, np.float32)
            else:
                boxes_, scores = pipe.detect(img)
            if bg is not None and len(boxes_):
                keep = [foreground_fraction(img, bg, b, pipe.cfg.foreground_diff) >= pipe.cfg.min_foreground
                        for b in boxes_]
                boxes_, scores = boxes_[keep], scores[keep]
            if ((scores >= opts.lo) & (scores < opts.hi)).any():
                stats["skipped_uncertain"] += 1
                continue
            boxes_ = boxes_[scores >= opts.hi]
            if len(boxes_) == 0:
                stats["no_animal"] += 1
                continue
            if bg is not None:
                fg = foreground_mask(img, bg, pipe.cfg.foreground_diff)
                fitted = boxes_ if opts.boxes == "blob" else [fit_to_foreground(b, fg) for b in boxes_]
                if any(f is None for f in fitted):
                    continue
                boxes_ = np.stack(fitted)
            kp, ok, _ = consistent_keypoints(pipe, img, boxes_, opts.kp_conf, opts.flip_tol)
            if bg is not None:
                xy = np.clip(np.round(kp).astype(int), 0, [fg.shape[1] - 1, fg.shape[0] - 1])
                ok &= fg[xy[..., 1], xy[..., 0]]
            if opts.boxes == "blob":
                tail = tail_centroid(img, bg, boxes_[0], pipe.cfg.foreground_diff)
                nose, base = CANONICAL.index("nose"), CANONICAL.index("tail_base")
                if (tail is not None and ok[0, nose] and ok[0, base]
                        and np.linalg.norm(kp[0, base] - tail) >= np.linalg.norm(kp[0, nose] - tail)):
                    stats["head_tail_flip"] += 1
                    continue
            animals = [{"keypoints": kp[i], "visibility": np.where(ok[i], 2, 0), "box": boxes_[i]}
                       for i in range(len(boxes_)) if ok[i].sum() >= opts.min_kps]
            if len(animals) < len(boxes_):  # an animal we cannot label well: do not teach it as background
                continue
            name = f"{sid}/{t:07d}.jpg"
            cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            writer.add(name, img.shape[:2], animals)
            stats["labeled"] += 1
            stats["animals"] += len(animals)
            stats["keypoints"] += int(sum(a["visibility"].astype(bool).sum() for a in animals))
        if opts.progress:
            print(rel, json.dumps(stats), flush=True)
    stats.update(writer.save(out / "train.json"))
    (out / "split.json").write_text(json.dumps(split, indent=1))
    return stats
