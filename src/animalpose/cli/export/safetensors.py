"""``animalpose export safetensors``: a ``.pt`` checkpoint as a ``.safetensors`` file.

    animalpose export safetensors --checkpoint runs/pose.pt
    animalpose export safetensors --checkpoint runs/pose.pt --out exported/pose.safetensors

The format and what goes in its header are described in :mod:`animalpose.export.safetensors`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from animalpose.export.safetensors import metadata_for, split_checkpoint, write_safetensors
from animalpose.paths import EXPORT_DIR

NAME = "safetensors"


def add_arguments(parser) -> None:
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: exported/<checkpoint stem>.safetensors")


def run(args) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind, cfg, tensors, extra = split_checkpoint(ckpt)
    if kind not in ("pose", "detector", "behavior", "temporal"):
        raise ValueError(f"{args.checkpoint}: unknown model kind {kind!r}")
    out = args.out or EXPORT_DIR / f"{args.checkpoint.stem}.safetensors"
    write_safetensors(out, tensors, metadata_for(kind, cfg, extra))
    print(f"wrote {out} ({len(tensors)} tensors, {out.stat().st_size / 1e6:.2f} MB, kind={kind})")
    print(f"  model config: {json.dumps(cfg)}")
