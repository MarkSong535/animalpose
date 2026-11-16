"""Dump end-to-end pipeline fixtures for the Rust CLI (`rust/crates/animalpose-cli`).

    animalpose fixtures pipeline
    animalpose fixtures pipeline --frames 8 --out tests/fixtures/pipeline
    animalpose fixtures pipeline --name general \
        --detector exported/detector_general.safetensors --pose exported/pose_general.safetensors \\
        --temporal exported/temporal_real.safetensors --behavior exported/behavior_real.safetensors

Runs :class:`animalpose.pipeline.PosePipeline` over a deterministic clip and writes everything the
Rust `animalpose infer` has to reproduce, byte for byte where the two ports can agree:

* ``frames/frame_NNNN.png`` — grayscale PNGs of `data/synthetic.py:render_video` (one seeded
  simulation, one top camera); both ports read the same files through their own frame readers.
* ``pipeline.yaml`` — the ``PipelineConfig`` / ``TrackerConfig`` the run used, in the layout
  ``configs/pipeline.yaml`` has, so the Rust CLI reads the very same file.
* ``tracks.csv`` / ``behavior.csv`` — ``VideoResult.to_csv`` / ``to_behavior_csv`` output.
* ``fixture.json`` — the reference values: per frame the network boxes, the blob candidates, the
  merged boxes and the pose output per box, plus the classifier's own input tensors, per track every
  ``TrackResult`` array, and the ``times``. Arrays are the ``{"shape", "data"}`` encoding of
  ``animalpose fixtures models`` (``null`` for NaN); the *inputs* (letterboxed detector
  tensor, warped pose crops) go in as SHA-256 of their float32 bytes plus their shape, so the Rust
  test asserts it feeds bit-identical arrays.
* ``<kind>.safetensors`` — the weights of the tiny randomly-initialised networks (the default),
  written by ``animalpose export safetensors``'s writer, so both ports load the same file. Trained
  checkpoints are referenced where they already live (``exported/*.safetensors``) and copied next to
  the fixture when they are given as ``--pose``/``--detector``/... paths.

``--skip sessions`` also dumps ``tests/fixtures/sessions/session.npz`` + ``session.json``: a slice of
a real ``data/behavior_sets.py:save_session`` file (dtypes included) with the values and the
``remap_labels`` result, which ``animalpose-data``'s npz reader is checked against.

The pipeline stage values are only comparable down to the difference between ``cv2.warpAffine``'s
fixed-point bilinear sampling and the Rust exact-weight sampler, so the Rust test reports the largest
deviation it finds instead of assuming one; the CSVs are the printed contract and are compared row by
row.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from animalpose.cli.fixtures.models import enc, random_weights
from animalpose.data.datasets import letterbox, to_tensor
from animalpose.data.synthetic import make_cameras, render_video, simulate
from animalpose.data.tracks import IGNORE
from animalpose.export.safetensors import checkpoint_from_safetensors, metadata_for, write_safetensors
from animalpose.frames import frame_timestamps, read_frames, sampled_background
from animalpose.geometry import box_xyxy_to_cs, crop_affine, warp
from animalpose.paths import FIXTURES_DIR, REPO
from animalpose.pipeline import (
    BlobProposer,
    PipelineConfig,
    PosePipeline,
    arena_roi,
    median_background,
    merge_candidates,
)
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.checkpoint import build_model

NAME = "pipeline"

SEED = 20261103

# 25 canonical keypoints are the only skeleton the Rust batch builders and pipeline speak, so the
# tiny fixture networks carry the full vocabulary (unlike the 5-point `tests/fixtures/models` ones).
TINY_BACKBONE = {"name": "tiny", "width": 4, "out_channels": 8}
CLASSES = ["resting", "sniffing", "walking", "turning", "rearing", "grooming", "other"]

TINY_MODELS = {
    "detector": {"backbone": TINY_BACKBONE, "num_species": 0},
    "pose": {"backbone": TINY_BACKBONE, "num_keypoints": 25, "embed_dim": 8},
    "behavior": {"num_keypoints": 25, "dims": 2, "hidden": 16, "classes": CLASSES, "dropout": 0.0},
    "temporal": {"num_keypoints": 25, "dims": 2, "hidden": 16, "layers": 1, "bidirectional": True,
                 "dropout": 0.0, "prior_bandwidth": 2.0, "prior": "hybrid"},
}

# A config both ports read: tiny random networks score ~0.5 everywhere, so the thresholds sit at 0
# and the blob candidate plus the static-object filter are on to cover those paths too.
PIPELINE = {
    "device": "cpu",
    "checkpoints": {},  # filled with the weight files this script writes or copies
    "pipeline": {
        "det_size": [96, 96],
        "det_thresh": 0.0,
        "max_det": 3,
        "crop_size": 64,
        "crop_padding": 1.25,
        "kp_conf_thresh": 0.3,
        "pair_decoding": True,
        "pair_identity": True,
        "pair_motion_frac": 0.1,
        "background_filter": True,
        "background_frames": 64,
        "foreground_diff": 25.0,
        "min_foreground": 0.05,
        "detector_candidates": ["network", "blob"],
        "blob_score": 0.9,
        "blob_diff": 25.0,
        "blob_open_px": 3,
        "blob_arena_roi": True,
        "blob_calibrate": 0,
        "merge_iou": 0.5,
        "behavior_conf": 0.6,
        "behavior_top_k": 3,
        "behavior_smooth": 3,
        "behavior_prior_adapt": True,
        "behavior_prior_max_ratio": 4.0,
        "embedder": "model",
        "temporal_window": 16,
        "batch_crops": 8,
    },
    "tracker": {"high_thresh": 0.0, "low_thresh": 0.0, "max_animals": 2, "max_lost": 5, "n_init": 1},
}


def sha_f32(array: np.ndarray) -> str:
    """SHA-256 of the row-major float32 bytes of ``array`` (what the Rust side hashes too)."""
    return hashlib.sha256(np.ascontiguousarray(array, np.float32).tobytes()).hexdigest()


def input_entry(array: np.ndarray) -> dict:
    """One network input: shape plus hash, and the values for a human to diff."""
    a = np.ascontiguousarray(array, np.float32)
    return {"shape": list(a.shape), "sha256": sha_f32(a), "nbytes": int(a.nbytes)}


def tensor_entry(array, extra: dict | None = None) -> dict:
    """A reference value: the ``{"shape", "data"}`` encoding, with NaNs as ``null``."""
    entry = dict(enc(torch.as_tensor(np.asarray(array))))
    if extra:
        entry.update(extra)
    return entry


def write_frames(folder: Path, n_frames: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Seeded synthetic clip, written as grayscale PNGs; returns the rendered grey frames.

    The names carry a `frames.py:_timestamp` suffix 100 ms apart, so both ports read the same
    frame rate (10 fps) off the file names, and `catalog.py:body_kinematics`' speed kernel
    (`round(0.3 * fps)` frames) stays shorter than a track, which it must for a clip this short.
    """
    folder.mkdir(parents=True, exist_ok=True)
    sim = simulate(1, n_frames, "mouse", seed=SEED)
    cam = make_cameras(sim.arena_cm, ("top",), size=size)[0]
    greys = []
    for t, frame in enumerate(render_video(sim, cam, seed=SEED)):
        grey = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        greys.append(grey)
        seconds = t / 10.0
        stamp = f"{int(seconds // 3600):02d}_{int(seconds // 60) % 60:02d}_" \
                f"{int(seconds % 60):02d}_{int(round(seconds * 1000) % 1000):03d}"
        cv2.imwrite(str(folder / f"frame_{stamp}.png"), grey)
    return greys


def build_tiny(kind: str, out: Path, extra: dict) -> tuple[object, dict]:
    """A randomly initialised network of ``TINY_MODELS[kind]``, stored as ``out/<kind>.safetensors``."""
    cfg = dict(TINY_MODELS[kind])
    model = build_model(kind, cfg)
    # `hash(str)` is salted per process; the digest is not, so the fixture is reproducible.
    random_weights(model, SEED + int(hashlib.sha256(kind.encode()).hexdigest()[:8], 16) % 9973)
    model.eval()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    write_safetensors(out / f"{kind}.safetensors", state, metadata_for(kind, cfg, extra))
    return model, cfg


def load_checkpoint(path: Path) -> tuple[object, dict]:
    """``(model, model_cfg)`` of a ``.safetensors`` export (or a ``.pt`` checkpoint)."""
    if path.suffix == ".safetensors":
        ck = checkpoint_from_safetensors(path)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(str(ck["kind"]), dict(ck["model_cfg"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), dict(ck["model_cfg"])


def record_behavior_inputs(pipe: PosePipeline) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(track id, coords, mask)`` of every classifier call `pipeline.py:_behavior_probs` makes.

    The wrapper records what the classifier was handed without touching the reference code path:
    `_behavior_probs` notes which track it is on, and the subclassed network appends the tensors it
    is called with, so the Rust port can compare its own classifier input array for array.
    """
    calls: list[tuple[int, np.ndarray, np.ndarray]] = []
    current: dict[str, int] = {}
    original = pipe._behavior_probs

    def wrapper(tr, times):  # noqa: ANN001, ANN002
        current["tid"] = int(tr.track_id)
        try:
            return original(tr, times)
        finally:
            current.pop("tid", None)

    pipe._behavior_probs = wrapper
    inner = pipe.behavior
    if inner is None:
        return calls

    class _Recorder:
        """A callable wrapper around the classifier that keeps the tensors of every call."""

        def __init__(self, model):
            self.model, self.classes = model, getattr(model, "classes", ())

        def __call__(self, *args, **kwargs):  # noqa: ANN001, ANN002
            calls.append((current.get("tid", -1), args[0].detach().numpy(), args[1].detach().numpy()))
            return self.model(*args, **kwargs)

    pipe.behavior = _Recorder(inner)
    return calls


def pose_crop(frame: np.ndarray, box: np.ndarray, crop_size: int, padding: float) -> np.ndarray:
    """The exact tensor `pipeline.py:PosePipeline.estimate` feeds the pose net for ``box``."""
    matrix = crop_affine(*box_xyxy_to_cs(box, padding=padding), crop_size)
    return to_tensor(warp(frame, matrix, crop_size)).numpy()


def frame_stages(pipe: PosePipeline, frames: list[np.ndarray], background: np.ndarray | None,
                 cfg: PipelineConfig) -> list[dict]:
    """Per frame: network boxes, blob candidates, merged boxes and the pose output per box."""
    propose = None
    if "blob" in cfg.detector_candidates:
        assert background is not None
        propose = BlobProposer(background, diff=cfg.blob_diff or cfg.foreground_diff, score=cfg.blob_score,
                               open_px=cfg.blob_open_px, area=cfg.blob_area, calibrate=cfg.blob_calibrate,
                               roi=arena_roi(background) if cfg.blob_arena_roi else None)
    stages = []
    for _index, img in enumerate(frames):
        net_boxes, net_scores = pipe.detect(img)
        extra_boxes, extra_scores = propose(img) if propose is not None else (np.zeros((0, 4), np.float32), np.zeros(0))
        boxes, scores = merge_candidates(net_boxes, net_scores, extra_boxes, extra_scores,
                                         cfg.merge_iou, cfg.max_det)
        kp, cf, em, margins = pipe.estimate(img, boxes, return_margins=True)
        tensor, _, _ = letterbox(img, cfg.det_size)
        crops = [pose_crop(img, b, cfg.crop_size, cfg.crop_padding) for b in boxes]
        stage = {
            "detector_input": input_entry(to_tensor(tensor).numpy()),
            "net_boxes": tensor_entry(net_boxes), "net_scores": tensor_entry(net_scores),
            "blob_boxes": tensor_entry(extra_boxes), "blob_scores": tensor_entry(extra_scores),
            "boxes": tensor_entry(boxes), "scores": tensor_entry(scores),
            "keypoints": tensor_entry(kp), "conf": tensor_entry(cf),
            "embeddings": tensor_entry(em), "pair_margin": tensor_entry(margins),
            "pose_inputs": [input_entry(c) for c in crops],
        }
        stages.append(stage)
    return stages


def track_entries(result) -> dict:
    """Every ``TrackResult`` array of one track, keyed the way ``to_npz`` names them."""
    out = {}
    for tid, tr in result.tracks.items():
        entry = {}
        for name in ("frames", "boxes", "keypoints", "conf", "smoothed", "smoothed_conf", "imputed",
                     "pair_margin", "pair_swapped", "behavior", "behavior_probs", "behavior_full"):
            value = getattr(tr, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype == bool:
                entry[name] = {"shape": list(array.shape), "data": [int(v) for v in array.reshape(-1)]}
            elif array.dtype.kind in "iu":
                entry[name] = {"shape": list(array.shape), "data": [int(v) for v in array.reshape(-1)]}
            else:
                entry[name] = tensor_entry(array)
        out[str(tid)] = entry
    return out


# The override strings the CLI passes to `config.py:load_config`, one per line of `--set`.
# `yaml.safe_load` decides what each means, so the fixture records Python's answer instead of a
# guess at it. `KNOWN_DIFFERENCES` below names the strings the two ports do not read the same way;
# the fixture carries that flag and the reason
# cases the ports are expected to disagree on.
CONFIG_OVERRIDES: list[str] = [
    "train.steps=200",
    "train.steps=-3",
    "train.steps=+7",
    "train.lr=5.0e-4",
    "train.lr=.5",
    "train.lr=-.inf",
    "train.steps=1_000",
    "train.steps=0x10",
    "train.steps=200 # a comment",
    "train.log_every=True",
    "train.log_every=true",
    "train.log_every=False",
    "train.log_every=yes",
    "train.log_every=off",
    "data.species=null",
    "data.species=",
    "data.species=[mouse, rat]",
    "data.species=mouse",
    "train.name=\"two  spaces\"",
    "train.note=a=b",
    "train.seed=0755",
    "model.head.temporal.window=16",
    "tracker.max_animals=4",
    "tracker.max_lost=0",
    "x.time=10:30",
    "x.version=1.2.3",
    "x.nan=nan",
    "x.dotnan=.nan",
    "x.inf=inf",
    "x.dotinf=.inf",
    "train.steps.deeper=1",
    "train=1",
    "=1",
]

# Where the two ports are known to disagree, and why, so `config_parity.rs` can insist the list
# stays true instead of pretending the parsers are identical.
KNOWN_DIFFERENCES: dict[str, str] = {
    # PyYAML resolves plain scalars with the YAML 1.1 rules, serde_yaml with the 1.2 core schema.
    "train.log_every=yes": "yaml 1.1 spells a boolean 'yes'; yaml 1.2 reads the string",
    "train.log_every=off": "yaml 1.1 spells a boolean 'off'; yaml 1.2 reads the string",
    "train.steps=1_000": "yaml 1.1 allows '_' digit separators; yaml 1.2 reads the string",
    "train.seed=0755": "yaml 1.1 reads a leading 0 as octal (493); yaml 1.2 reads the string",
    "x.time=10:30": "yaml 1.1 has sexagesimal numbers (630); yaml 1.2 reads the string",
    # `load_config` walks whatever key it is given; the CLI refuses to store an unnamed one.
    "=1": "the CLI rejects an override whose key is empty; python stores it under ''",
}


# ``extends:`` trees the two loaders have to resolve alike: what a child keeps, what replaces it, in
# which order several bases lose, and what is an error rather than a default. Paths are relative to the
# file that writes them, exactly as they are in ``configs/``.
EXTENDS_FILES: dict[str, str] = {
    "parent.yaml": ("train:\n  steps: 30\n  lr: 0.5\ndata:\n  species: [rat]\n"
                    "  sources: [{source: coco, weight: 0.5}]\nmodel:\n  backbone: {name: tiny, width: 32}\n"),
    "second.yaml": "train:\n  seed: 7\ndata:\n  species: [mouse, rat]\n",
    "child.yaml": ("extends: parent.yaml\ntrain:\n  steps: 5\nmodel:\n  backbone: {width: 64}\n"
                   "data:\n  sources: [{source: synthetic, weight: 1.0}]\n"),
    "grandchild.yaml": "extends: child.yaml\ntrain:\n  seed: 9\n",
    "two_parents.yaml": "extends: [parent.yaml, second.yaml]\ntrain:\n  lr: 0.25\n",
    "of_base.yaml": "extends: base.yaml\ntrain:\n  steps: 3\ndata:\n  species: [rat]\n",
    "overridden.yaml": "extends: parent.yaml\ntrain:\n  steps: 5\n",
    "no_base.yaml": "train:\n  steps: 2\n",
    "missing_base.yaml": "extends: nowhere.yaml\ntrain:\n  steps: 1\n",
    "cycle_a.yaml": "extends: cycle_b.yaml\n",
    "cycle_b.yaml": "extends: cycle_a.yaml\n",
    "extends_is_a_number.yaml": "extends: 3\n",
}

# file -> the dotted overrides applied on top of what it extends ("" means none)
EXTENDS_CASES: dict[str, list[str]] = {
    "child.yaml": [],
    "grandchild.yaml": [],
    "two_parents.yaml": [],
    "of_base.yaml": [],
    "no_base.yaml": [],
    "overridden.yaml": ["train.steps=1", "data.species=[pig]"],
    "missing_base.yaml": [],
    "cycle_a.yaml": [],
    "extends_is_a_number.yaml": [],
}


def value_entry(value) -> dict:
    """Any value `yaml.safe_load` returns, tagged so the Rust side can compare it exactly."""
    if value is None:
        return {"kind": "null"}
    if isinstance(value, bool):
        return {"kind": "bool", "data": value}
    if isinstance(value, int):
        return {"kind": "int", "data": value}
    if isinstance(value, float):
        return {
            "kind": "float",
            "data": None if not math.isfinite(value) else float(f"{value:.17g}"),
            "repr": repr(value),
        }
    if isinstance(value, str):
        return {"kind": "str", "data": value}
    if isinstance(value, list):
        return {"kind": "list", "data": [value_entry(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "map", "data": {str(k): value_entry(v) for k, v in value.items()}}
    return {"kind": "unsupported", "data": repr(value)}


def dump_config(out: Path) -> None:
    """What `config.py:load_config` makes of every dotted override the CLI accepts."""
    from animalpose.config import load_config

    folder = out / "config"
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / "base.yaml"
    base.write_text(
        "train:\n  steps: 10\n  seed: 0\ndata:\n  species: [mouse]\ntracker:\n  max_animals: 2\n",
        encoding="utf-8",
        newline="\n",
    )
    cases, extends = [], []
    for item in CONFIG_OVERRIDES:
        key = item.partition("=")[0]
        try:
            cfg = load_config(base, [item])
        except Exception as exc:  # noqa: BLE001
            cases.append({"override": item, "key": key, "error": type(exc).__name__})
            continue
        node = cfg
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        cases.append(
            {
                "override": item,
                "key": key,
                "value": value_entry(node),
                "known_difference": item in KNOWN_DIFFERENCES,
                "note": KNOWN_DIFFERENCES.get(item),
            }
        )
    for name, text in EXTENDS_FILES.items():
        (folder / name).write_text(text, encoding="utf-8", newline="\n")
    for name, item in sorted(EXTENDS_CASES.items()):
        try:
            tree = load_config(folder / name, item)
        except Exception as exc:  # noqa: BLE001
            extends.append({"file": name, "overrides": item, "error": type(exc).__name__})
            continue
        extends.append({"file": name, "overrides": item, "value": value_entry(tree)})
    (folder / "overrides.json").write_text(
        json.dumps({"base": base.read_text(encoding="utf-8"), "cases": cases, "extends": extends}, indent=1),
        encoding="utf-8",
        newline="\n",
    )
    print(f"  config/overrides.json {len(cases)} override strings, {len(KNOWN_DIFFERENCES)} expected to differ, "
          f"{len(extends)} extends trees")


def dump_sessions(out: Path) -> None:
    """A slice of a behavior session `.npz`, plus the values `sessions.rs` must read back.

    The source is a simulated session under ``data/processed/behavior/synthetic`` (``save_session`` of a
    ``simulate`` run), so the committed fixture carries no recorded animal."""
    from animalpose.data.behavior_sets import load_session_dir, remap_labels
    from animalpose.labels import BEHAVIORS_EXT

    roots = [REPO / "data/processed/behavior/synthetic"]
    source = next((p for root in roots for p in sorted(Path(root).rglob("*.npz"))), None)
    if source is None:
        print("  no behavior session .npz found: sessions fixture skipped")
        return
    z = np.load(source, allow_pickle=False)
    keep = 24
    names = [str(n) for n in z["label_names"]]
    session_dir = out / "sessions"
    save_session_stub(session_dir / "session.npz", z, keep)
    stored = np.load(session_dir / "session.npz", allow_pickle=False)
    labels = stored["labels"]
    modelled = ["resting", "sniffing", "walking", "turning", "rearing", "grooming", "other"]
    remapped = remap_labels(labels, names, modelled)
    loaded = load_session_dir(session_dir, 15.0, None, modelled)[0]
    payload = {
        "source": str(source.relative_to(REPO)),
        "frames": int(stored["coords"].shape[0]),
        "coords": tensor_entry(stored["coords"]),
        "conf": tensor_entry(stored["conf"]),
        "labels": {"shape": list(labels.shape), "data": [int(v) for v in labels]},
        "fps": float(stored["fps"]),
        "animal": str(stored["animal"]),
        "view": str(stored["view"]),
        "label_names": names,
        "dtypes": {k: str(stored[k].dtype) for k in stored.files},
        "classes": modelled,
        "remapped": {"shape": list(remapped.shape), "data": [int(v) for v in remapped]},
        "ignored": int(IGNORE),
        # `load_session_dir` resamples to ~15 fps and reports the frames that survive.
        "resampled_frames": int(loaded.coords.shape[0]),
        "resampled_fps": float(loaded.fps),
        "resampled_labels": {"shape": list(loaded.labels.shape), "data": [int(v) for v in loaded.labels]},
        "ext_classes": list(BEHAVIORS_EXT),
    }
    (out / "sessions" / "session.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"  sessions/session.npz {keep} frames of {source.name}, {len(names)} label names")


def save_session_stub(path: Path, z, keep: int) -> None:
    """The first ``keep`` frames of a stored session, re-saved with the same members and dtypes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "coords": z["coords"][:keep].astype(np.float32),
        "conf": z["conf"][:keep].astype(np.float32),
        "labels": z["labels"][:keep].astype(np.int64),
        "fps": np.array(float(z["fps"])),
        "animal": np.array(str(z["animal"])),
        "view": np.array(str(z["view"])),
        "label_names": np.array([str(n) for n in z["label_names"]]),
    }
    if "boxes" in z.files:
        arrays["boxes"] = z["boxes"][:keep].astype(np.float32)
    np.savez_compressed(path, **arrays)


def add_arguments(parser) -> None:
    parser.add_argument("--out", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--name", default="pipeline", help="the fixture directory under --out")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, nargs=2, default=(96, 96), metavar=("H", "W"))
    parser.add_argument("--detector", type=Path, help="trained detector .safetensors/.pt (default: tiny random)")
    parser.add_argument("--pose", type=Path, help="trained pose .safetensors/.pt (default: tiny random)")
    parser.add_argument("--temporal", type=Path, help="trained temporal .safetensors/.pt (default: tiny random)")
    parser.add_argument("--behavior", type=Path, help="trained behavior .safetensors/.pt (default: tiny random)")
    parser.add_argument("--skip", choices=["config", "sessions"], action="append", default=[])


def run(args) -> None:
    out = (args.out if args.out.is_absolute() else REPO / args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    greys = write_frames(out / "frames", args.frames, (args.size[1], args.size[0]))
    print(f"  frames/: {len(greys)} grey png of {greys[0].shape[1]}x{greys[0].shape[0]} at 10 fps")

    cfg = json.loads(json.dumps(PIPELINE))
    pcfg = cfg["pipeline"]
    # The non-tensor entries `train/loops.py:train_*` store next to the weights.
    extra = {"pose": {"crop_size": pcfg["crop_size"]}, "detector": {"input_size": pcfg["det_size"]},
             "behavior": {"window": pcfg["temporal_window"]}, "temporal": {"window": pcfg["temporal_window"]}}
    models, configs = {}, {}
    for kind, path in (("detector", args.detector), ("pose", args.pose),
                       ("temporal", args.temporal), ("behavior", args.behavior)):
        sidecar = out / f"{kind}.safetensors"
        if path is not None:
            source = path if path.is_absolute() else REPO / path
            model, model_cfg = load_checkpoint(source)
            if source.suffix == ".safetensors":
                sidecar.write_bytes(source.read_bytes())
            else:
                write_safetensors(sidecar, {k: v.contiguous() for k, v in model.state_dict().items()},
                                  metadata_for(kind, model_cfg, extra[kind]))
        else:
            model, model_cfg = build_tiny(kind, out, extra[kind])
        # Repository fixtures keep repo-relative paths, so `pipeline.yaml` reads the same in either
        # port; a dump made outside the checkout (a trained checkpoint under /tmp, say) keeps its
        # absolute path, which both readers resolve as-is.
        cfg["checkpoints"][kind] = (
            sidecar.relative_to(REPO).as_posix() if sidecar.is_relative_to(REPO) else str(sidecar)
        )
        models[kind], configs[kind] = model, model_cfg
        print(f"  {kind}: {model_cfg} <- {cfg['checkpoints'][kind]}")

    (out / "pipeline.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    pipeline_cfg = PipelineConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                                     for k, v in cfg["pipeline"].items()},
                                  tracker=TrackerConfig(**cfg["tracker"]))
    pipe = PosePipeline(models["detector"], models["pose"], models["temporal"], models["behavior"],
                        pipeline_cfg, device="cpu")
    recorded = record_behavior_inputs(pipe) if models["behavior"] is not None else []

    frames, fps = read_frames(out / "frames")
    times = frame_timestamps(out / "frames")
    fps = float(fps or 30.0)
    need_bg = pipe.cfg.background_filter or "blob" in pipe.cfg.detector_candidates
    background = sampled_background(out / "frames") if need_bg else None
    result = pipe.run(frames, background=background, fps=fps, times=times)

    stages = frame_stages(pipe, frames, background, pipe.cfg)
    payload = {
        "seed": SEED,
        "frames": args.frames,
        "fps": fps,
        "size": [int(frames[0].shape[0]), int(frames[0].shape[1])],
        "models": {k: configs[k] for k in configs},
        "pipeline": cfg["pipeline"],
        "tracker": cfg["tracker"],
        "times": {"shape": list(np.asarray(result.times).shape),
                  "data": [float(f"{v:.9g}") for v in np.asarray(result.times).reshape(-1)]},
        "behaviors": list(result.behaviors),
        "background": input_entry(np.asarray(background, np.float32)) if background is not None else None,
        "background_median": input_entry(median_background(frames, pipe.cfg.background_frames)),
        "stages": stages,
        "tracks": track_entries(result),
        "n_frames": int(result.n_frames),
    }
    if recorded:
        payload["behavior_inputs"] = [
            {"track": str(tid), "frames": int(x.shape[1]),
             "coords": input_entry(x[0]), "mask": input_entry(m[0]),
             "coords_values": tensor_entry(x[0]), "mask_values": tensor_entry(m[0])}
            for tid, x, m in recorded
        ]
    (out / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")

    result.to_csv(out / "tracks.csv")
    if result.behavior_info:
        result.to_behavior_csv(out / "behavior.csv", pipe.cfg.behavior_top_k)
    size = (out / "fixture.json").stat().st_size / 1e3
    print(f"  fixture.json {size:.0f} KiB: {len(stages)} frames, {len(result.tracks)} tracks, "
          f"{len(result.behaviors)} behavior names")

    out_dir = args.out if args.out.is_absolute() else REPO / args.out
    if "config" not in args.skip:
        dump_config(out_dir)
    if "sessions" not in args.skip:
        dump_sessions(out_dir)
    print(f"wrote {out}")
