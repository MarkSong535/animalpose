"""Dump torch reference values for the Rust networks (`rust/crates/animalpose-models`).

    animalpose fixtures models
    animalpose fixtures models --out /tmp/mfx \
        && ANIMALPOSE_FIXTURES=/tmp/mfx ...

    animalpose fixtures models \
        --checkpoint weights/pose_general.pt --checkpoint weights/detector_general.pt

Trained checkpoints go through ``build_model`` with their own ``model_cfg`` and weights, so their
fixtures cover the shapes a random initialisation never reaches — the resnet18 encoder of ``pose_general``,
the width-32 backbone of ``detector_general``, the canonical keypoint groups of ``behavior_general``. Only the
config, inputs and outputs are written to JSON: the weights of a trained model are tens of megabytes
and live in the sibling ``.safetensors`` (which ``.gitignore`` keeps out of the tree), the same file the
Rust test loads in production. A missing weights file makes the Rust test skip that fixture.

One JSON file per network under ``tests/fixtures/models/`` — ``pose_tiny``, ``detector_tiny``,
``behavior_tiny``, ``temporal_tiny`` and their variants — built with
:func:`animalpose.train.checkpoint.build_model` from the same config dicts the YAML configs carry, at
small sizes (a 4-wide `tiny` backbone with 8-wide maps, 5 keypoints, ``64x64`` images,
``T=16`` sequences — the ``*_general`` fixtures then cover the widths the trained models use). Every
parameter is replaced by noise at its initialisation scale and every batch norm's running statistics by
random values (so evaluation-mode normalisation is exercised, not just the identity), the model runs in ``eval()``
mode under ``torch.no_grad()``, and the config, the inputs, the outputs and the whole state dict are
written out. The Rust tests (``rust/crates/animalpose-models/src/tests.rs``) rebuild the model from the
config, load the weights, run the same inputs and compare with a ``2e-4`` absolute tolerance; a missing
fixture directory makes those tests print a line and skip.

One file per case, ``<kind>_<name>.json``::

    {"config":  {...model config...},
     "inputs":  {"name": {"shape": [...], "data": [...]}, ...},   # row-major float32
     "outputs": {"name": {"shape": [...], "data": [...]}, ...},   # the Python forward's dict order
     "weights": {"tensor_name": {"shape": [...], "data": [...]}, ...}}

and the same weights as ``<kind>_<name>.safetensors``, written by
``animalpose export safetensors``'s writer with the ``"animalpose"`` / ``"kind"`` metadata, so the Rust
test loads weights exactly the way production does. ``manifest.json`` records a SHA-256 per file: the
generator seeds ``torch`` once and works in a fixed order, so re-running it reproduces the fixtures
byte for byte, and it says so — the maximum absolute deviation between two runs is printed at the end
and is the deviation the Rust comparison has to tolerate on top of ``2e-4``.

Behaviour fixtures with fewer than 21 keypoints swap ``skeleton.CANONICAL``'s anterior/posterior groups
for a small pair (the canonical indices would be out of range); the groups are buffers
(``features.ant`` / ``features.post``), so the Rust model reads them from the dumped weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from animalpose.behavior import model as behavior_model
from animalpose.export.safetensors import metadata_for, write_safetensors
from animalpose.paths import FIXTURES_DIR
from animalpose.train.checkpoint import build_model

NAME = "models"

SEED = 4242
TINY_BACKBONE = {"name": "tiny", "width": 4, "out_channels": 8}

# The small keypoint groups used when a fixture has fewer keypoints than the canonical vocabulary.
SMALL_GROUPS = {"anterior": [0, 1], "posterior": [4, 3]}

# `skeleton.CANONICAL`'s most posterior keypoint index plus one: fewer keypoints than this and the
# default groups do not fit.
CANONICAL_KEYPOINTS = 21


def enc(t: torch.Tensor) -> dict:
    """One tensor -> ``{"shape": ..., "data": ...}``, row-major float32 (``%.9g`` round-trips f32).

    NaN and inf are written as ``null``: strict JSON has no spelling for them, and the Rust test reads a
    ``null`` back as NaN, which is what an unobserved keypoint coordinate is.
    """
    a = np.ascontiguousarray(np.asarray(t.detach().cpu().numpy(), dtype=np.float32))
    flat = a.reshape(-1)
    return {
        "shape": list(a.shape),
        "data": [None if not math.isfinite(v) else float(f"{v:.9g}") for v in flat],
    }


def dec(entry: dict) -> torch.Tensor:
    """The inverse of :func:`enc`: ``null`` becomes NaN again."""
    data = [float("nan") if v is None else v for v in entry["data"]]
    return torch.tensor(data, dtype=torch.float32).reshape(entry["shape"])


class _Groups:
    """Stand-in for ``skeleton.CANONICAL`` with the small groups of the ``K = 5`` fixtures."""

    anterior = SMALL_GROUPS["anterior"]
    posterior = SMALL_GROUPS["posterior"]


def random_weights(model: torch.nn.Module, seed: int) -> None:
    """Replaces every parameter by noise at its own initialisation scale, and every running statistic
    by random values.

    ``N(0, 1 / sqrt(fan_in))`` for kernels and weight matrices, ``U(-1 / sqrt(fan_in), 1 / sqrt(fan_in))``
    for biases — the spread ``nn.Conv2d``/``nn.Linear`` start from — so the networks produce outputs of
    order one instead of the thousands a flat ``N(0, 0.4)`` draws. The absolute tolerance of the Rust
    comparison is only meaningful when the reference values are.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            fan_in = max(1.0, param.numel() / max(1, param.shape[0]))
            scale = 1.0 / math.sqrt(fan_in)
            if ".norm." in name:  # batch-norm gamma and beta
                if name.endswith(".weight"):
                    param.uniform_(0.6, 1.4)
                else:
                    param.normal_(0.0, 0.5)
            elif name.endswith(".weight"):
                param.normal_(0.0, scale)
            else:
                param.uniform_(-scale, scale)
        for name, buf in model.named_buffers():
            # `norm.mean` / `norm.std` and the keypoint groups stay at their constructor values.
            if name.endswith("running_mean"):
                buf.normal_(0.0, 0.3)
            elif name.endswith("running_var"):
                buf.uniform_(0.4, 1.6)


def masked_nan(coords: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Sets every unobserved coordinate to NaN, the way the Python datasets deliver missing points."""
    out = coords.clone()
    out[seen < 0.5] = float("nan")
    return out


def cases() -> list[tuple[str, str, dict, dict[str, torch.Tensor]]]:
    """``(kind, name, model config, inputs)`` for every fixture, in the order they are written."""
    torch.manual_seed(SEED)
    pose = {"backbone": TINY_BACKBONE, "num_keypoints": 5, "embed_dim": 8}
    detector = {"backbone": TINY_BACKBONE}
    crops = torch.rand(1, 3, 64, 64)

    # `build_model` does not forward `dilations`, so a behavior model always gets the constructor's
    # (1, 2, 4, 8, 16) stack; the key stays out of these configs so both ports agree.
    behaviour2 = {"num_keypoints": 5, "dims": 2, "hidden": 8,
                  "classes": ["resting", "walking", "rearing"], "dropout": 0.0}
    behaviour3 = {"num_keypoints": 5, "dims": 3, "hidden": 6,
                  "classes": ["resting", "walking"], "dropout": 0.0}
    seen2 = (torch.rand(2, 16, 5) > 0.25).float()
    seen3 = (torch.rand(2, 16, 5) > 0.15).float()

    temporal_k = {"num_keypoints": 5, "dims": 2, "hidden": 8, "layers": 2, "bidirectional": False,
                  "dropout": 0.0, "prior_bandwidth": 2.0, "prior": "kernel"}
    temporal_h = {"num_keypoints": 5, "dims": 3, "hidden": 6, "layers": 2, "bidirectional": True,
                  "dropout": 0.0, "prior_bandwidth": 3.0, "prior": "hybrid"}
    temporal_0 = {"num_keypoints": 5, "dims": 2, "hidden": 8, "layers": 1, "bidirectional": True,
                  "dropout": 0.0, "prior_bandwidth": 0.0}

    track2 = masked_nan(torch.randn(2, 16, 5, 2) * 20.0, seen2)
    track3 = masked_nan(torch.randn(2, 16, 5, 3) * 20.0, seen3)
    conf = torch.rand(2, 16, 5)

    return [
        ("pose", "pose_tiny", pose, {"crops": crops}),
        ("pose", "pose_tiny_context", {**pose, "context": True}, {"crops": crops}),
        ("detector", "detector_tiny", {**detector, "num_species": 0}, {"images": crops}),
        ("detector", "detector_tiny_species", {**detector, "num_species": 2},
         {"images": torch.rand(1, 3, 64, 64)}),
        ("behavior", "behavior_tiny", behaviour2,
         {"coords": masked_nan(torch.randn(2, 16, 5, 2) * 15.0, seen2), "mask": seen2}),
        ("behavior", "behavior_tiny_3d", behaviour3,
         {"coords": masked_nan(torch.randn(2, 16, 5, 3) * 15.0, seen3), "mask": seen3}),
        ("temporal", "temporal_tiny", temporal_k,
         {"coords": track2, "conf": conf, "observed": seen2}),
        ("temporal", "temporal_tiny_bidir", temporal_h,
         {"coords": track3, "conf": conf, "observed": seen3}),
        ("temporal", "temporal_tiny_noprior", temporal_0,
         {"coords": track2, "conf": conf, "observed": seen2}),
    ]


def outputs_of(kind: str, model, args: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """The Python forward's outputs, named the way the Rust model names them.

    Behaviour fixtures carry `features` as well — the hand-written `EgocentricFeatures` block the
    convolution stack sits on top of — so a Rust mismatch says which of the two went wrong.
    """
    with torch.no_grad():
        out = model(*args)
    if kind == "behavior":
        return {"logits": out, "features": model.features(*args)}
    return dict(out)


def build_case(kind: str, name: str, cfg: dict, inputs: dict[str, torch.Tensor]) -> dict:
    """The fixture document for one case: config, inputs, outputs and the whole state dict."""
    restore = behavior_model.CANONICAL
    if kind == "behavior" and cfg.get("num_keypoints", 25) < CANONICAL_KEYPOINTS:
        behavior_model.CANONICAL = _Groups  # the canonical groups need 21 keypoints
    try:
        model = build_model(kind, cfg)
    finally:
        behavior_model.CANONICAL = restore
    # A stable per-case seed: `hash(str)` is salted per process, the digest is not.
    random_weights(model, SEED + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 9973)
    model.eval()
    args = [inputs[key] for key in inputs]  # the models' forward argument order
    out = outputs_of(kind, model, args)
    return {
        "config": cfg,
        "inputs": {k: enc(v) for k, v in inputs.items()},
        "outputs": {k: enc(v) for k, v in out.items()},
        "weights": {k: enc(v) for k, v in model.state_dict().items()},
    }


def checkpoint_inputs(kind: str, cfg: dict) -> dict[str, torch.Tensor]:
    """The input batch a real checkpoint is run on, seeded so the fixture is reproducible."""
    torch.manual_seed(SEED + 1)
    if kind in ("pose", "detector"):
        # A 32x32 crop: the smallest size the stride-32 encoder accepts, and small enough to keep the
        # committed reference JSON light.
        batch = torch.rand(1, 3, 32, 32)
        return {"crops": batch} if kind == "pose" else {"images": batch}
    if kind == "behavior":
        k = cfg.get("num_keypoints", 25)
        seen = (torch.rand(2, 32, k) > 0.2).float()
        return {"coords": masked_nan(torch.randn(2, 32, k, cfg.get("dims", 2)) * 15.0, seen), "mask": seen}
    k = cfg.get("num_keypoints", 25)
    dims = cfg.get("dims", 2)
    seen = (torch.rand(2, 32, k) > 0.25).float()
    return {
        "coords": masked_nan(torch.randn(2, 32, k, dims) * 20.0, seen),
        "conf": torch.rand(2, 32, k),
        "observed": seen,
    }


def build_checkpoint_case(path: Path) -> tuple[str, str, dict, dict, dict[str, torch.Tensor]]:
    """The fixture for a trained checkpoint: its own config, weights and outputs, nothing randomised.

    The weights stay out of the JSON (a resnet18 backbone is 46 MB of f32) and go into the sibling
    ``<name>.safetensors``, which is what the Rust test loads; the JSON carries the config, the inputs
    and the reference outputs, so it is committable on its own and the test skips when the weights file
    has not been exported.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    kind, cfg, state = blob["kind"], dict(blob["model_cfg"]), blob["state_dict"]
    name = f"{kind}_{path.stem.removeprefix(kind + '_')}"
    restore = behavior_model.CANONICAL
    if kind == "behavior" and cfg.get("num_keypoints", 25) < CANONICAL_KEYPOINTS:
        behavior_model.CANONICAL = _Groups
    try:
        model = build_model(kind, cfg)
        model.load_state_dict(state)
    finally:
        behavior_model.CANONICAL = restore
    model.eval()
    inputs = checkpoint_inputs(kind, cfg)
    args = [inputs[key] for key in inputs]
    out = outputs_of(kind, model, args)
    fixture = {
        "config": cfg,
        "inputs": {k: enc(v) for k, v in inputs.items()},
        "outputs": {k: enc(v) for k, v in out.items()},
        "weights": {},
        "weights_file": f"{name}.safetensors",
    }
    return name, kind, cfg, fixture, state


def deviation(a: dict, b: dict) -> float:
    """The largest absolute difference between the outputs of two fixture documents."""
    worst = 0.0
    for name, entry in a["outputs"].items():
        other = b["outputs"][name]
        if entry["shape"] != other["shape"]:
            return float("inf")
        worst = max(worst, float((dec(entry) - dec(other)).abs().max()))
    for name, tensor in a["weights"].items():
        other = b["weights"][name]
        if tensor["shape"] != other["shape"]:
            return float("inf")
        worst = max(worst, float((dec(tensor) - dec(other)).abs().max()))
    return worst


def sha256(path: Path) -> str:
    """The checksum of one fixture file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_arguments(parser) -> None:
    parser.add_argument("--out", type=Path, default=FIXTURES_DIR / "models")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        metavar="RUN.PT",
        help="also dump a fixture from a trained checkpoint, e.g. --checkpoint weights/pose_general.pt",
    )


def run(args) -> None:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    worst = 0.0
    manifest: dict[str, str] = {}
    for kind, name, cfg, inputs in cases():
        fixture = build_case(kind, name, cfg, inputs)
        again = build_case(kind, name, cfg, inputs)  # the same seed again: fixtures must be deterministic
        worst = max(worst, deviation(fixture, again))
        weights = {k: dec(v) for k, v in fixture["weights"].items()}
        meta = metadata_for(kind, cfg, {})
        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(fixture), encoding="utf-8")
        write_safetensors(out_dir / f"{name}.safetensors", weights, meta)
        manifest[f"{name}.json"] = sha256(json_path)
        manifest[f"{name}.safetensors"] = sha256(out_dir / f"{name}.safetensors")
        written.append(name)
        sizes = json_path.stat().st_size / 1e3, (out_dir / f"{name}.safetensors").stat().st_size / 1e6
        print(f"  {name}: {len(weights)} tensors, {sizes[0]:.0f} KiB json, {sizes[1]:.2f} MB st")

    for path in args.checkpoint:
        name, kind, cfg, fixture, state = build_checkpoint_case(path)
        again = build_checkpoint_case(path)[3]
        worst = max(worst, deviation(fixture, again))
        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(fixture), encoding="utf-8")
        sidecar = out_dir / f"{name}.safetensors"
        write_safetensors(sidecar, state, metadata_for(kind, cfg, {}))
        manifest[f"{name}.json"] = sha256(json_path)
        manifest[f"{name}.safetensors"] = sha256(sidecar)
        written.append(name)
        print(f"  {name}: {kind} from {path}, {len(state)} tensors, "
              f"{json_path.stat().st_size / 1e3:.0f} KiB json, {sidecar.stat().st_size / 1e6:.2f} MB st")

    (out_dir / "manifest.json").write_text(
        json.dumps({"seed": SEED, "tolerance": 2e-4, "files": manifest}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(written)} fixtures to {out_dir}")
    print(f"max abs deviation between two runs (the deviation the Rust tests must tolerate): {worst:g}")
