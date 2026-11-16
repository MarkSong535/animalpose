"""Export checkpoints to TorchScript and ONNX, verify parity with ONNX Runtime, and write manifest.json for C++ consumers.

    animalpose export models --detector runs/detector.pt --pose runs/pose.pt \\
        --temporal runs/temporal.pt --behavior runs/behavior.pt --out exported/
"""

from __future__ import annotations

import argparse
import json

import torch

from animalpose.export.export import export_all
from animalpose.train.checkpoint import load_model

NAME = "models"


def add_arguments(parser) -> None:
    for k in ("detector", "pose", "temporal", "behavior"):
        parser.add_argument(f"--{k}")
    parser.add_argument("--out", default="exported")
    parser.add_argument("--det-size", type=int, nargs=2, default=None, help="H W")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--pair-decoding", action=argparse.BooleanOptionalAction, default=True,
                        help="bake left/right pair decoding into the pose graph")


def run(args) -> None:
    models = {k: load_model(getattr(args, k)) if getattr(args, k) else None
              for k in ("detector", "pose", "temporal", "behavior")}
    det_size = tuple(args.det_size) if args.det_size else tuple(
        torch.load(args.detector, weights_only=False).get("input_size", (256, 320)) if args.detector else (256, 320))
    crop = args.crop_size or (torch.load(args.pose, weights_only=False).get("crop_size", 128) if args.pose else 128)
    report = export_all(args.out, **models, det_size=det_size, crop_size=crop, window=args.window,
                        pair_decoding=args.pair_decoding)
    print(json.dumps({k: {"ok": v["ok"], "max_abs_err": v["max_abs_err"], "exporter": v.get("exporter")}
                      for k, v in report.items()}, indent=2))
    if not all(v["ok"] for v in report.values()):
        raise SystemExit("export parity check failed")
