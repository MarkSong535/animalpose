"""Label-free checks on held-out recordings of an unlabeled collection.

The recordings held out by ``animalpose label`` (about 1 in 5, chosen by
path hash) are never pseudo-labeled. With no ground truth, models are compared on proxies,
each computed on short runs of consecutive frames:

* ``detected``: share of frames with an animal (score >= ``--thresh``, after the static-object
  filter for sequences);
* ``flip_agreement``: share of confident keypoints found again within 5 % of the box on the
  mirrored frame (self-consistency; higher is better);
* ``jitter``: mean frame-to-frame motion of confident keypoints relative to the box size,
  for consecutive frames (lower is steadier, but a model that ignores the animal is steady too,
  so read it together with ``detected``);
* ``conf``: mean keypoint confidence;
* ``blob_iou`` (with ``--blob``, fixed camera, one dark animal): share of frames where the
  detected box overlaps the moving-blob box (``animalpose.pipeline.blob_box``) with IoU >= 0.5,
  over frames where the blob is unambiguous.

``--range a b`` evaluates only that part of each recording and ``--all-sources`` uses every
recording, e.g. the held-out last fifth of a single video labeled with ``--range 0 0.8``.
``--candidates blob`` adds the moving-blob boxes of a fixed camera (PipelineConfig
``detector_candidates``), which is how the recall of the network detector is compared to it.

    animalpose eval unlabeled --root ~/animal/training_data --pairs \\
        runs/detector_real_robust.pt:runs/pose_real_r18_robust.pt runs/detector_real_adapt.pt:runs/pose_real_r18_adapt.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from animalpose.frames import find_sources, is_held_out
from animalpose.pipeline import PipelineConfig, PosePipeline, arena_roi, blob_box, foreground_fraction
from animalpose.selftrain import background, consistent_keypoints
from animalpose.train.checkpoint import load_model

NAME = "unlabeled"


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def evaluate(pipe, sources, args) -> dict:
    det, agree, jit, conf, ious, per = [], [], [], [], [], {}
    for src in sources:
        n = src.n_frames
        if n == 0:
            continue
        bg = background(src, n) if n >= args.min_seq else None
        roi = arena_roi(bg) if (args.blob and bg is not None) else None
        prop = pipe.proposals(bg) if (bg is not None and "blob" in args.candidates) else None
        a, b = int(args.range[0] * (n - 1)), int(args.range[1] * (n - 1))
        starts = np.linspace(a, max(b - args.run_frames, a), args.runs).astype(int)
        s_det, s_agree, s_jit, s_iou = [], [], [], []
        for s0 in starts:
            prev = None
            for _, img in src.frames(range(s0, min(s0 + args.run_frames, n))):
                ref = (blob_box(img, bg, pipe.cfg.foreground_diff, roi=roi) if roi is not None else None)
                boxes, scores = pipe.detect_candidates(img, prop)
                if bg is not None and len(boxes):
                    keep = [foreground_fraction(img, bg, b, pipe.cfg.foreground_diff) >= pipe.cfg.min_foreground
                            for b in boxes]
                    boxes, scores = boxes[keep], scores[keep]
                boxes = boxes[scores >= args.thresh][:1]
                s_det.append(len(boxes) > 0)
                if ref is not None:
                    hit = len(boxes) > 0 and _iou(boxes[0], ref) >= 0.5
                    ious.append(hit)
                    s_iou.append(hit)
                if not len(boxes):
                    prev = None
                    continue
                kp, ok, cf = consistent_keypoints(pipe, img, boxes, 0.0, 0.05)
                confident = cf[0] >= 0.5
                if confident.any():
                    s_agree.append(ok[0][confident].mean())
                conf.append(cf[0].mean())
                size = max(boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1], 1.0)
                if prev is not None:
                    both = confident & prev[1]
                    if both.any():
                        s_jit.append(float(np.median(np.linalg.norm(kp[0][both] - prev[0][both], axis=-1)) / size))
                prev = (kp[0], confident)
        det += s_det
        agree += s_agree
        jit += s_jit
        per[str(src.path.name)] = {"detected": float(np.mean(s_det)) if s_det else 0.0,
                                    "flip_agreement": float(np.mean(s_agree)) if s_agree else None,
                                    "jitter": float(np.median(s_jit)) if s_jit else None,
                                    "blob_iou50": float(np.mean(s_iou)) if s_iou else None}

    return {"detected": float(np.mean(det)) if det else 0.0,
            "flip_agreement": float(np.mean(agree)) if agree else None,
            "jitter": float(np.mean(jit)) if jit else None,
            "conf": float(np.mean(conf)) if conf else None, "frames": len(det),
            "blob_iou50": float(np.mean(ious)) if ious else None, "per_source": per}


def add_arguments(parser) -> None:
    parser.add_argument("--root", required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="detector.pt:pose.pt")
    parser.add_argument("--det-size", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--runs", type=int, default=5, help="runs of consecutive frames per recording")
    parser.add_argument("--run", dest="run_frames", metavar="RUN", type=int, default=20, help="frames per run")
    parser.add_argument("--min-seq", type=int, default=30)
    parser.add_argument("--range", type=float, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--all-sources", action="store_true")
    parser.add_argument("--candidates", nargs="+", choices=["network", "blob"], default=["network"],
                        help="detection candidate sources (blob needs a static background; blob_iou50 "
                             "then measures self-agreement, not detector accuracy)")
    parser.add_argument("--blob", action="store_true", help="also score boxes against the moving-blob box")
    parser.add_argument("--out", default="results/unlabeled.json")
    parser.add_argument("--include", nargs="+", help="only recordings whose path contains one of these strings")


def run(args) -> None:
    held = [s for s in find_sources(args.root) if args.all_sources or is_held_out(s, args.root)]
    if args.include:
        held = [s for s in held if any(k in str(s.path) for k in args.include)]
    print(f"{len(held)} held-out recordings", flush=True)
    out = Path(args.out)
    allres = json.loads(out.read_text()) if out.exists() else {}
    for pair in args.pairs:
        det, pose = pair.split(":")
        if not (Path(det).exists() and Path(pose).exists()):
            print("missing", pair)
            continue
        pipe = PosePipeline(load_model(det), load_model(pose),
                            config=PipelineConfig(crop_size=args.crop_size, det_size=tuple(args.det_size), det_thresh=0.1,
                                                  detector_candidates=tuple(args.candidates)))
        tag = f"{Path(det).stem}+{Path(pose).stem}"
        allres[tag] = evaluate(pipe, held, args)
        print(tag, json.dumps({k: v for k, v in allres[tag].items() if k != "per_source"}), flush=True)
        out.write_text(json.dumps(allres, indent=2))
