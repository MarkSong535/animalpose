"""YAML config loading: ``extends:`` inheritance, dotted overrides (``train.steps=200``) and key checking.

``extends`` names one base config or a list of them, resolved relative to the file that declares them.
Mappings merge recursively, every other value replaces the base value wholesale, and the child wins.
The key itself is removed from the resolved dict, so callers never see it.

The key check catches the one failure this loader cannot otherwise report: a mistyped key trains with
defaults and says nothing (``tarin.steps=500`` is a run at the default step count). Config kinds share
this loader and share no keys, so the check is opt-in per kind: :data:`TRAIN_SCHEMA` for a trainer
config, :func:`pipeline_schema` for a pipeline config. Camera profiles and dataset source lists are
checked where they are read (:func:`animalpose.data.degrade.load_profile`).
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import yaml

EXTENDS = "extends"


class ConfigError(ValueError):
    """A config that would be ignored rather than honoured: an unknown key or a broken base."""


# The keys a trainer config may carry. Anything else is a typo: the trainers read what they read through
# cfg.get(...), so an unknown key silently leaves the default in place.
TRAIN_SCHEMA: dict[str | None, set[str]] = {
    None: {"device", "init", "output", "model", "data", "train", "loss", EXTENDS},
    "model": {"backbone", "num_species", "num_keypoints", "dims", "classes", "hidden", "layers", "dropout",
              "bidirectional", "context", "embed_dim", "prior", "prior_bandwidth"},
    "data": {"source", "sources", "json", "images", "mapping", "species", "seed", "n_sims", "frames",
             "image_size", "input_size", "crop_size", "max_animals", "low_view_fraction", "degrade", "zoom_out",
             "camera_profile", "window", "pool_sims", "pool_frames", "real_root", "real_animals", "session_dirs",
             "session_groups", "real_temporal", "real_fraction", "real_mapping", "kp_subsets", "kp_subset_p"},
    "train": {"steps", "batch_size", "lr", "weight_decay", "workers", "log_every", "seed", "clip"},
    "loss": {"class_balance", "embed_weight", "offset_weight", "occluded_weight", "sigma"},
}


def pipeline_schema() -> dict[str | None, set[str]]:
    """The keys a pipeline config may carry, taken from the dataclasses that consume them."""
    from dataclasses import fields

    from animalpose.multiview.drift import DriftConfig
    from animalpose.pipeline import PipelineConfig
    from animalpose.tracking.tracker import TrackerConfig

    # `assay` (experiment rules) and `checksums` are read by the Rust engine; Python carries them through.
    return {None: {"device", "checkpoints", "pipeline", "tracker", "drift", "assay", "checksums", EXTENDS},
            "checkpoints": {"detector", "pose", "temporal", "behavior"},
            "pipeline": {f.name for f in fields(PipelineConfig)},
            "tracker": {f.name for f in fields(TrackerConfig)},
            "drift": {f.name for f in fields(DriftConfig)}}


def load_config(path: str | Path, overrides: list[str] | None = None,
                schema: dict[str | None, set[str]] | None = None) -> dict:
    """The config at ``path``: its bases merged underneath, ``overrides`` applied, keys checked."""
    cfg = load_tree(Path(path))
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    if schema:
        validate(cfg, schema, str(path))
    return cfg


def load_tree(path: str | Path, chain: tuple[Path, ...] = ()) -> dict:
    """The file's own tree with every base it ``extends`` merged underneath it."""
    resolved = Path(path).resolve()
    if resolved in chain:
        raise ConfigError(f"config extends itself: {' -> '.join(str(p) for p in [*chain, resolved])}")
    try:
        own = yaml.safe_load(resolved.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"config not found: {resolved}") from None
    if not isinstance(own, dict):
        raise ConfigError(f"{resolved}: a config must be a mapping")
    own = dict(own)
    bases = _bases(resolved, own.pop(EXTENDS, None))
    merged: dict = {}
    for base in bases:
        merge_into(merged, load_tree(base, (*chain, resolved)))
    merge_into(merged, own)
    return merged


def merge_into(dst: dict, src: dict) -> None:
    """``src`` on top of ``dst``: mappings merge key by key, every other value replaces."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge_into(dst[key], value)
        else:
            dst[key] = value


def validate(cfg: dict, schema: dict[str | None, set[str]], name: str = "<config>") -> None:
    """Reject keys nothing reads, naming the closest known key."""
    for key, value in cfg.items():
        known = schema.get(None, set())
        if key not in known:
            raise ConfigError(f"{name}: unknown key {key!r}{_suggest(key, known)}")
        section = schema.get(key)
        if section and isinstance(value, dict):
            for sub, sub_value in value.items():
                if sub not in section:
                    raise ConfigError(f"{name}: unknown key {key}.{sub} = {sub_value!r}{_suggest(sub, section)}")


def _bases(path: Path, value: Any) -> list[Path]:
    """An ``extends`` value as paths, relative to the file that wrote it."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value) if isinstance(value, list) else None
    if items is None or not all(isinstance(v, str) and v for v in items):
        raise ConfigError(f"{path}: {EXTENDS} must name a path or a list of paths, got {value!r}")
    return [Path(v) if Path(v).is_absolute() else path.parent / v for v in items]


def _suggest(key: str, known: set[str]) -> str:
    close = difflib.get_close_matches(key, sorted(known), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""
