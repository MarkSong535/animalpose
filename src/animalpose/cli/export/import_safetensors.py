"""``animalpose export import-safetensors``: rebuild a ``.pt`` checkpoint from a ``.safetensors`` file.

    animalpose export import-safetensors --safetensors exported/pose.safetensors
    animalpose export import-safetensors --safetensors exported/pose.safetensors \\
        --out runs/pose_restored.pt --check runs/pose.pt

The inverse of :mod:`animalpose.export.safetensors`: the header's ``__metadata__`` block is unpacked back
into the ``{"kind", "model_cfg", "state_dict", ...}`` entries :func:`animalpose.train.checkpoint.load_model`
expects. ``--check <original.pt>`` additionally compares the rebuilt checkpoint with the original tensor by
tensor and prints the largest absolute difference, which is how the interchange is verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from animalpose.export.safetensors import checkpoint_from_safetensors, compare
from animalpose.paths import EXPORT_DIR

NAME = "import-safetensors"


def add_arguments(parser) -> None:
    parser.add_argument("--safetensors", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: exported/<name>.pt")
    parser.add_argument("--check", type=Path, default=None, help="original .pt checkpoint to compare against")


def run(args) -> None:
    ckpt = checkpoint_from_safetensors(args.safetensors)
    out = args.out or EXPORT_DIR / f"{args.safetensors.stem}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"wrote {out} ({len(ckpt['state_dict'])} tensors, kind={ckpt['kind']})")
    print(f"  model config: {json.dumps(ckpt['model_cfg'])}")
    if args.check:
        original = torch.load(args.check, map_location="cpu", weights_only=False)
        checked, worst = compare(original, ckpt)
        print(f"round trip vs {args.check}: {checked} tensors, max abs diff {worst:g}")
        if worst != 0.0:
            raise ValueError("round trip is not bit-exact")
