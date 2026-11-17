"""Checkpoints store the model config next to the weights so any model can be
rebuilt with :func:`load_model` (by the pipeline, export script or benchmark)."""

from __future__ import annotations

from pathlib import Path

import torch

from animalpose.behavior.model import BehaviorTCN
from animalpose.detector.model import CenterDetector
from animalpose.labels import BEHAVIORS
from animalpose.pose.model import PoseNet
from animalpose.temporal.model import TemporalLSTM

BUILDERS = {
    "detector": lambda c: CenterDetector(c.get("backbone"), c.get("num_species", 0)),
    "pose": lambda c: PoseNet(c.get("num_keypoints", 25), c.get("backbone"), c.get("embed_dim", 64),
                                 c.get("context", False)),
    # Arena detectors (scripts/train_arena.py): the pose network over the whole frame, one point per
    # arena landmark; model_cfg["arena"] holds the arena definition (animalpose.arena).
    "arena": lambda c: PoseNet(c.get("num_keypoints", 4), c.get("backbone"), c.get("embed_dim", 64),
                                  c.get("context", False)),
    "temporal": lambda c: TemporalLSTM(c.get("num_keypoints", 25), c.get("dims", 2), c.get("hidden", 128),
                                       c.get("layers", 2), c.get("bidirectional", True), c.get("dropout", 0.1),
                                       c.get("prior_bandwidth", 0.0), c.get("prior", "kernel")),
    "behavior": lambda c: BehaviorTCN(c.get("num_keypoints", 25), c.get("dims", 2), tuple(c.get("classes", BEHAVIORS)),
                                      c.get("hidden", 96), dropout=c.get("dropout", 0.1)),
}


def build_model(kind: str, model_cfg: dict) -> torch.nn.Module:
    return BUILDERS[kind](model_cfg or {})


def save_checkpoint(path: str | Path, kind: str, model: torch.nn.Module, model_cfg: dict, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "model_cfg": model_cfg, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_model(path: str | Path, map_location="cpu") -> torch.nn.Module:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"no checkpoint at {source}")
    ckpt = torch.load(source, map_location=map_location, weights_only=False)
    model = build_model(ckpt["kind"], ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
