"""The JSON contract between the torch fixture dumpers and the Rust parity tests.

``animalpose fixtures tensor`` and ``animalpose fixtures models`` write the reference values that
``rust/src/tensor/tests.rs`` and ``rust/src/models/tests.rs`` compare their own forward and backward passes
against. Those Rust readers are strict: ``serde`` parses ``{"shape", "data"}`` entries out of strict JSON (no
``NaN``/``Infinity`` tokens, and in the tensor fixtures no ``null`` either), it panics on a fixture whose
``<kind>_`` file stem names no model, it demands the fixture's input and output names be exactly the model's
``INPUTS``/``OUTPUTS`` constants, it re-checks every JSON weight against the sibling ``.safetensors`` to 1e-6,
and it compares with a fixed ``2e-4`` tolerance. Both dumpers are run here into a temp directory and every one
of those expectations is asserted on the files they produced -- including that each listed operation and
network got exactly one file, that the whole dump is reproducible from its seed, and that ``manifest.json``
digests every file it wrote. The fixture files themselves are never touched: nothing is written outside
``tmp_path``.
"""

import argparse
import hashlib
import json
import math
import re
from collections import namedtuple
from pathlib import Path

import pytest
import torch

from animalpose.behavior import model as behavior_model
from animalpose.cli import main
from animalpose.cli.fixtures import models
from animalpose.cli.fixtures import tensor as tensor_fixtures
from animalpose.export.safetensors import read_safetensors
from animalpose.train.checkpoint import build_model, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
RUST_TENSOR = REPO / "rust" / "src" / "tensor" / "tests.rs"
RUST_MODELS = REPO / "rust" / "src" / "models" / "tests.rs"
RUST_MODEL_DIR = REPO / "rust" / "src" / "models"

requires_rust = pytest.mark.skipif(not (RUST_TENSOR.is_file() and RUST_MODELS.is_file()),
                                   reason="the Rust parity tests are not in this checkout")

TENSOR_KEYS = {"shape", "inputs", "params", "output", "grad_input", "grad_params"}
KINDS = ("pose", "detector", "behavior", "temporal")
# The behaviour and temporal fixtures carry unobserved keypoints, encoded as `null` (Rust reads NaN back).
NULLABLE_INPUTS = {"coords"}
# The largest tensor the tensor fixtures are allowed to hold (the dumper promises "nothing larger than 3x4x8x8").
MAX_TENSOR_VALUES = 3 * 4 * 8 * 8
# How `Fixture` in `rust/src/tensor/tests.rs` reads each param key. A JSON float under `usize`/`signed`
# makes `as_u64`/`as_i64` return `None` and the operation silently runs with its default, so the JSON type
# of every key below is part of the contract, not just its value.
USIZE_PARAMS = {"axis", "dilation", "end", "h", "hidden", "k", "layers", "pad", "start", "stride", "w"}
FLOAT_PARAMS = {"dropout", "eps", "smoothing", "weight"}  # `dropout` is written but never read
SIGNED_PARAMS = {"ignore"}
FLAG_PARAMS = {"bidirectional"}
LIST_PARAMS = {"axes", "perm"}
# Discriminators the Rust test takes from the fixture's name and from a hardcoded `LossKind`, never from JSON.
TAG_PARAMS = {"kind", "mode"}

Dump = namedtuple("Dump", "dir names")


def _args(module, out: Path, extra: list[str] | tuple[str, ...] = ()) -> argparse.Namespace:
    """The namespace the ``animalpose fixtures <part>`` verb builds for ``--out <out>``."""
    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    return parser.parse_args(["--out", str(out), *extra])


def _rust_tensor_names() -> set[str]:
    """Every fixture name ``rust/src/tensor/tests.rs`` asks ``compare()`` for."""
    src = RUST_TENSOR.read_text(encoding="utf-8")
    names = set(re.findall(r'compare\("([a-z0-9_]+)"', src))
    for tags in re.findall(r"for tag in \[([^\]]*)\]", src):
        listed = re.findall(r'"([a-z0-9_]+)"', tags)
        names |= {f"reduce_{op}_{tag}" for op in ("sum", "mean") for tag in listed}
    return names


def _rust_tolerance(path: Path) -> float:
    """The tolerance a Rust parity test compares with."""
    (found,) = re.findall(r"const TOL: f32 = ([0-9.eE+-]+);", path.read_text(encoding="utf-8"))
    return float(found)


def _rust_io_names() -> dict[str, dict[str, tuple[str, ...]]]:
    """The ``INPUTS``/``OUTPUTS`` constants of each network in ``rust/src/models/``."""
    found: dict[str, dict[str, tuple[str, ...]]] = {}
    for path in sorted(RUST_MODEL_DIR.glob("*.rs")):
        src = path.read_text(encoding="utf-8")
        tables = {}
        for key, label in (("inputs", "INPUTS"), ("outputs", "OUTPUTS")):
            rows = re.findall(rf"pub const {label}: \[&str; \d+\] = \[([^\]]*)\];", src)
            if rows:
                tables[key] = tuple(re.findall(r'"([a-z_]+)"', rows[0]))
        if len(tables) == 2:
            found[path.stem] = tables
    return found


def _real(value) -> bool:
    """A JSON number the Rust accessors accept: not a bool, and finite (strict JSON has no NaN token)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int(value) -> bool:
    """A JSON integer: `as_u64`/`as_i64` refuse a float, and a bool is not a number to them."""
    return isinstance(value, int) and not isinstance(value, bool)


def _flat(values: list, *, allow_null: bool) -> list:
    """A fixture tensor's flat values: strict JSON numbers, with ``null`` only where the port expects NaN."""
    for v in values:
        if v is None:
            assert allow_null
        else:
            assert _real(v)
    return values


def _entry(entry: dict, *, allow_null: bool) -> None:
    """One ``{"shape", "data"}`` entry in the shape ``serde`` wants."""
    assert set(entry) == {"shape", "data"}
    assert all(isinstance(d, int) and d > 0 for d in entry["shape"])
    assert len(entry["data"]) == math.prod(entry["shape"])
    _flat(entry["data"], allow_null=allow_null)


def _strict_json(text: str, *, allow_null: bool) -> dict:
    """Parse a fixture and reject the non-standard tokens ``serde_json`` refuses to read."""
    assert "NaN" not in text and "Infinity" not in text
    if not allow_null:
        assert "null" not in text
    return json.loads(text)


def _fixture_jsons(directory: Path) -> list[Path]:
    """The JSON files ``fixtures()`` in ``rust/src/models/tests.rs`` walks: every one but the manifest."""
    return sorted(p for p in directory.glob("*.json") if p.stem != "manifest")


def _small_groups():
    """The stand-in keypoint groups the sub-21-keypoint behaviour fixtures need."""

    class Groups:
        anterior = models.SMALL_GROUPS["anterior"]
        posterior = models.SMALL_GROUPS["posterior"]

    return Groups


@pytest.fixture(scope="module")
def tensor_dump(tmp_path_factory):
    """One run of the tensor dumper: the files it wrote and the operation names it reported writing."""
    out = tmp_path_factory.mktemp("tensor-fixtures")
    names: list[str] = []
    original = tensor_fixtures.dump

    def record(out_dir: Path, name: str, *rest, **kwargs):
        names.append(name)
        return original(out_dir, name, *rest, **kwargs)

    tensor_fixtures.dump = record
    try:
        tensor_fixtures.run(_args(tensor_fixtures, out))
    finally:
        tensor_fixtures.dump = original
    return Dump(out, names)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """One tiny checkpoint per model kind, for the ``--checkpoint`` path of the model dumper."""
    root = tmp_path_factory.mktemp("model-checkpoints")
    tiny = {"name": "tiny", "width": 4, "out_channels": 8}
    cases = [("pose", {"backbone": tiny, "num_keypoints": 5, "embed_dim": 8}, {"crop_size": 32}),
             ("detector", {"backbone": tiny, "num_species": 2}, {"input_size": (32, 32)}),
             ("temporal", {"num_keypoints": 5, "dims": 2, "hidden": 6, "prior_bandwidth": 2.0}, {"window": 32}),
             ("behavior", {"num_keypoints": 25, "dims": 2, "hidden": 6, "classes": ["a", "b"]}, {"window": 32})]
    saved = {}
    for kind, cfg, extra in cases:
        saved[f"{kind}_fx"] = root / f"{kind}_fx.pt"
        save_checkpoint(saved[f"{kind}_fx"], kind, build_model(kind, cfg), cfg, extra)
    # Fewer keypoints than `skeleton.CANONICAL` has: the model needs the small groups to be constructed.
    cfg = {"num_keypoints": 5, "dims": 2, "hidden": 6, "classes": ["resting", "walking"], "dropout": 0.0}
    restore = behavior_model.CANONICAL
    behavior_model.CANONICAL = _small_groups()
    try:
        model = build_model("behavior", cfg)
    finally:
        behavior_model.CANONICAL = restore
    saved["behavior_small"] = root / "behavior_small.pt"
    save_checkpoint(saved["behavior_small"], "behavior", model, cfg, {"window": 32})
    return saved


@pytest.fixture(scope="module")
def model_dump(tmp_path_factory, checkpoints):
    """One run of the model dumper: the tiny cases and the five checkpoint cases."""
    out = tmp_path_factory.mktemp("model-fixtures")
    extra = [item for path in checkpoints.values() for item in ("--checkpoint", str(path))]
    models.run(_args(models, out, extra))
    return Dump(out, sorted(p.stem for p in out.glob("*.json") if p.stem != "manifest"))


# -------------------------------------------------------------------------------------------- tensor core


@requires_rust
def test_the_tensor_dumper_writes_one_file_per_operation_the_rust_reads(tensor_dump):
    """A missing file makes the matching Rust test skip silently, so every name it wants must exist."""
    files = sorted(p.name for p in tensor_dump.dir.iterdir())
    assert files == sorted(f"{name}.json" for name in tensor_dump.names)
    assert len(tensor_dump.names) == len(set(tensor_dump.names))  # one file per operation
    wanted = _rust_tensor_names()
    assert len(wanted) >= 25, "the Rust fixture-name extraction found almost nothing"
    assert wanted <= set(tensor_dump.names), sorted(wanted - set(tensor_dump.names))
    assert {n for n in tensor_dump.names if n.startswith("reduce_")} == {
        f"reduce_{op}_{tag}" for op in ("sum", "mean") for tag in ("axes1", "axes02", "all")}


def test_every_tensor_fixture_has_the_documented_shape(tensor_dump):
    """The keys, shapes and gradient bookkeeping ``Fixture``/``compare()`` in the Rust test read."""
    for path in sorted(tensor_dump.dir.glob("*.json")):
        doc = _strict_json(path.read_text(encoding="utf-8"), allow_null=False)
        assert set(doc) == TENSOR_KEYS, path.name
        assert isinstance(doc["params"], dict), path.name
        for key, entry in doc["inputs"].items():
            _entry(entry, allow_null=False)
            if key == "x":
                assert doc["shape"] == entry["shape"], f"{path.name}: `shape` is the primary input's shape"
        for key, entry in doc["grad_params"].items():
            _entry(entry, allow_null=False)
            assert key != "x" and key in doc["inputs"], path.name
        _entry(doc["output"], allow_null=False)
        _entry(doc["grad_input"], allow_null=False)
        assert doc["grad_input"]["shape"] == doc["shape"], path.name
        assert doc["output"]["data"] and doc["output"]["shape"]


def test_tensor_fixtures_are_strict_json_and_finite(tensor_dump):
    """``data: Vec<f32>`` and ``serde_json``: no ``null``, no ``NaN``, no ``Infinity``, all finite."""
    for path in sorted(tensor_dump.dir.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        doc = _strict_json(text, allow_null=False)
        groups = [{"output": doc["output"], "grad_input": doc["grad_input"], **doc["grad_params"]}]
        for group in groups:
            for name, entry in group.items():
                assert all(v is not None and math.isfinite(v) for v in entry["data"]), f"{path.name}:{name}"


def test_tensor_params_carry_only_the_scalars_the_rust_accessors_read(tensor_dump):
    """Every ``params`` key has the JSON type of the ``usize``/``flag``/``float``/``signed``/``list`` read."""
    for path in sorted(tensor_dump.dir.glob("*.json")):
        doc = _strict_json(path.read_text(encoding="utf-8"), allow_null=False)
        for key, value in doc["params"].items():
            where = f"{path.name}:{key}"
            if key in TAG_PARAMS:
                assert isinstance(value, str), where
            elif key in FLAG_PARAMS:
                assert isinstance(value, bool), where
            elif key in LIST_PARAMS:
                assert all(_int(v) and v >= 0 for v in value), where
            elif key in USIZE_PARAMS:
                assert _int(value) and value >= 0, where
            elif key in SIGNED_PARAMS:
                assert _int(value), where
            elif key in FLOAT_PARAMS:
                assert _real(value), where
            else:
                raise AssertionError(f"{where} is not a key any Rust accessor reads")
    by_name = {p.stem: _strict_json(p.read_text(encoding="utf-8"), allow_null=False)
               for p in tensor_dump.dir.glob("*.json")}
    assert by_name["conv2d_dilated"]["params"] == {"stride": 1, "pad": 2, "dilation": 2}
    # The batch-norm mode and the loss kind name the variant the Rust test already picked by name.
    assert by_name["batch_norm_train"]["params"]["mode"] == "train"
    assert by_name["batch_norm_eval"]["params"]["mode"] == "eval"
    assert by_name["smooth_l1"]["params"]["kind"] == "smooth_l1"
    assert by_name["mse"]["params"]["kind"] == "mse"
    assert by_name["batch_norm_train"]["params"]["eps"] == pytest.approx(1e-5)
    assert by_name["lstm_2layer_bidir"]["params"]["bidirectional"] is True
    assert by_name["lstm_1layer"]["params"] == {"hidden": 4, "layers": 1, "bidirectional": False, "dropout": 0.0}
    assert by_name["cross_entropy"]["params"] == {"smoothing": 0.1, "ignore": -100}
    assert by_name["upsample_nearest"]["params"] == {"h": 6, "w": 8}


def test_tensor_fixtures_stay_tiny_and_float32(tensor_dump):
    """The parity is a numerical comparison of two f32 implementations, so nothing may be big or wide."""
    for path in sorted(tensor_dump.dir.glob("*.json")):
        doc = _strict_json(path.read_text(encoding="utf-8"), allow_null=False)
        entries = [doc["output"], doc["grad_input"], *doc["inputs"].values(), *doc["grad_params"].values()]
        for entry in entries:
            assert len(entry["shape"]) <= 4, path.name
            assert len(entry["data"]) <= MAX_TENSOR_VALUES, path.name
        for key, entry in doc["inputs"].items():
            if key in ("mask", "observed"):
                assert set(entry["data"]) <= {0.0, 1.0}, path.name


def test_the_tensor_dumper_is_deterministic_from_its_seed(tmp_path, tensor_dump):
    """The Rust comparison has one tolerance budget: a re-run must reproduce the reference bit for bit."""
    again = tmp_path / "tensor-again"
    tensor_fixtures.run(_args(tensor_fixtures, again))
    assert sorted(p.name for p in again.iterdir()) == sorted(p.name for p in tensor_dump.dir.iterdir())
    differs = [p.name for p in sorted(again.iterdir()) if p.read_bytes() != (tensor_dump.dir / p.name).read_bytes()]
    assert differs == []


def test_the_tensor_verb_on_the_command_line_writes_the_same_files(tmp_path, tensor_dump, capsys):
    """``animalpose fixtures tensor --out`` is the same run under the same seed."""
    out = tmp_path / "cli-tensor"
    assert main(["fixtures", "tensor", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"wrote {len(tensor_dump.names)} fixtures to {out}" in printed
    differs = [p.name for p in sorted(out.iterdir()) if p.read_bytes() != (tensor_dump.dir / p.name).read_bytes()]
    assert differs == []


# --------------------------------------------------------------------------------------- model networks


def test_the_model_dumper_writes_one_pair_of_files_per_case(model_dump, checkpoints):
    """Every case in ``models.cases()`` and every ``--checkpoint`` gets a JSON and a weights sidecar."""
    listed = [name for _, name, _, _ in models.cases()]
    derived = []
    for path in checkpoints.values():  # the dumper's own naming rule: `<kind>_<stem without the kind>`
        kind = path.stem.split("_")[0]
        derived.append(f"{kind}_{path.stem.removeprefix(kind + '_')}")
    assert listed and set(listed) <= set(model_dump.names)
    assert set(derived) <= set(model_dump.names), sorted(derived - set(model_dump.names))
    assert set(model_dump.names) == set(listed) | set(derived)
    files = sorted(p.name for p in model_dump.dir.iterdir())
    assert files == sorted([*(f"{n}.json" for n in model_dump.names),
                            *(f"{n}.safetensors" for n in model_dump.names), "manifest.json"])


@requires_rust
def test_every_model_fixture_carries_what_the_rust_reader_parses(model_dump):
    """``config``/``inputs``/``outputs`` are required fields, and the names must be the model's own."""
    io_names = _rust_io_names()
    assert set(io_names) >= set(KINDS), io_names
    for path in _fixture_jsons(model_dump.dir):
        doc = _strict_json(path.read_text(encoding="utf-8"), allow_null=True)
        # `Fixture` deserialises these three and `weights` with `#[serde(default)]`; the extra
        # `weights_file` of a checkpoint fixture is unknown to serde and ignored.
        assert {"config", "inputs", "outputs"} <= set(doc), path.name
        kind = path.stem.split("_")[0]
        assert kind in KINDS, f"{path.name}: <kind>_<name> is how the Rust test picks the model"
        cfg, io = doc["config"], io_names[kind]
        assert isinstance(cfg, dict) and cfg
        assert set(doc["inputs"]) == set(io["inputs"]), path.name
        expected = set(io["outputs"])
        if kind == "detector" and not cfg.get("num_species"):
            expected -= {"species"}
        if kind == "behavior":
            expected |= {"features"}  # dumped too, so a failure says which block drifted
        assert set(doc["outputs"]) == expected, path.name
        assert isinstance(doc.get("weights"), dict), path.name
        for name, entry in doc["inputs"].items():
            _entry(entry, allow_null=name in NULLABLE_INPUTS)
        for name, entry in doc["outputs"].items():
            _entry(entry, allow_null=False)
            assert all(v is not None for v in entry["data"]), f"{path.name}: {name} must be finite"
        for entry in doc["weights"].values():
            _entry(entry, allow_null=False)
        for name in NULLABLE_INPUTS & set(doc["inputs"]):
            assert any(v is None for v in doc["inputs"][name]["data"]), f"{path.name}: {name} should be sparse"


def test_the_sidecar_weights_match_the_json_and_the_metadata(model_dump):
    """The Rust test loads the sidecar the production way and re-checks it against the JSON to 1e-6."""
    for path in _fixture_jsons(model_dump.dir):
        doc = _strict_json(path.read_text(encoding="utf-8"), allow_null=True)
        kind = path.stem.split("_")[0]
        sidecar = path.with_suffix(".safetensors")
        if not doc["weights"]:  # a checkpoint fixture: its weights live only in the sidecar
            assert doc.get("weights_file") == sidecar.name, path.name
        assert sidecar.is_file(), path.name  # `check()` skips a fixture whose weights file is absent
        tensors, meta = read_safetensors(sidecar)
        assert meta["kind"] == kind, path.name
        assert json.loads(meta["animalpose"]) == doc["config"], path.name
        assert set(tensors) >= set(doc["weights"])
        if doc["weights"]:
            assert set(tensors) == set(doc["weights"]), path.name
        for name, entry in doc["weights"].items():
            got = tensors[name]
            assert list(got.shape) == entry["shape"], f"{path.name}: {name} has the wrong shape"
            assert got.dtype == torch.float32, f"{path.name}: {name} is not f32"
            worst = max(abs(float(a) - float(b)) for a, b in zip(entry["data"], got.reshape(-1).tolist())
                        if a is not None)
            assert worst <= 1e-6, f"{path.name}: {name} deviates by {worst}"
        assert all(torch.isfinite(v).all() for v in tensors.values()), path.name


@requires_rust
def test_the_manifest_digests_every_file_and_states_the_shared_tolerance(model_dump):
    """``manifest.json`` is the parity contract: seed, the tolerance both ports use, and one digest per file."""
    manifest = json.loads((model_dump.dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"seed", "tolerance", "files"}
    assert manifest["seed"] == models.SEED
    assert manifest["tolerance"] == 2e-4 == _rust_tolerance(RUST_TENSOR) == _rust_tolerance(RUST_MODELS)
    assert set(manifest["files"]) == {p.name for p in model_dump.dir.iterdir()} - {"manifest.json"}
    wrong = [name for name, digest in manifest["files"].items()
             if hashlib.sha256((model_dump.dir / name).read_bytes()).hexdigest() != digest]
    assert wrong == []


def test_the_model_dumper_is_deterministic_from_its_seed(tmp_path, model_dump, checkpoints):
    """The reference JSONs are reproduced byte for byte; only a sidecar header's ordering may move."""
    again = tmp_path / "models-again"
    extra = [item for path in checkpoints.values() for item in ("--checkpoint", str(path))]
    models.run(_args(models, again, extra))
    assert sorted(p.name for p in again.iterdir()) == sorted(p.name for p in model_dump.dir.iterdir())
    first = json.loads((model_dump.dir / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((again / "manifest.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in second.items() if k != "files"} == {k: v for k, v in first.items() if k != "files"}
    assert set(second["files"]) == set(first["files"])
    assert {k: v for k, v in second["files"].items() if k.endswith(".json")} == {
        k: v for k, v in first["files"].items() if k.endswith(".json")}
    for path in _fixture_jsons(model_dump.dir):  # the very values the Rust comparison reads
        assert (again / path.name).read_bytes() == path.read_bytes(), path.name
    # The safetensors wheel writes its metadata map in a free order, so a sidecar's bytes -- and the
    # manifest digest of them -- are not run-stable; what they hold is, which is what parity depends on.
    for name in model_dump.names:
        a, meta_a = read_safetensors(model_dump.dir / f"{name}.safetensors")
        b, meta_b = read_safetensors(again / f"{name}.safetensors")
        assert meta_a == meta_b and sorted(a) == sorted(b), name
        assert all(torch.equal(a[n], b[n]) and a[n].dtype == b[n].dtype for n in a), name


def test_the_model_verb_on_the_command_line_dumps_the_tiny_cases(tmp_path, capsys):
    """``animalpose fixtures models --out`` runs without any ``--checkpoint`` argument at all."""
    out = tmp_path / "cli-models"
    assert main(["fixtures", "models", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    listed = [name for _, name, _, _ in models.cases()]
    assert f"wrote {len(listed)} fixtures to {out}" in printed
    assert {p.stem for p in out.glob("*.json")} - {"manifest"} == set(listed)


# ------------------------------------------------------------------------------------------ encodings


def test_enc_writes_nan_as_null_and_dec_reads_it_back():
    """Strict JSON has no NaN; ``null`` is the agreed spelling and ``dec`` turns it back into NaN."""
    src = torch.tensor([[1.5, float("nan")], [-0.25, float("inf")]], dtype=torch.float32)
    entry = models.enc(src)
    assert entry["shape"] == [2, 2] and entry["data"] == [1.5, None, -0.25, None]
    back = models.dec(entry)
    assert back.shape == src.shape and back.dtype == torch.float32
    assert not math.isnan(back[0, 0].item()) and math.isnan(back[0, 1].item())
    assert math.isnan(models.dec(models.enc(torch.tensor([float("inf")])))[0].item())  # inf is `null` too
    assert json.dumps(entry)  # the entry is serialisable without a non-standard token
    # `%.9g` round-trips a float32 exactly, so the JSON weights are the safetensors weights.
    awkward = torch.randn(64, dtype=torch.float32) * 1e6
    assert models.dec(models.enc(awkward)).view(torch.int32).equal(awkward.view(torch.int32))


def test_masked_nan_hides_exactly_the_unobserved_points():
    """The datasets deliver missing keypoints as NaN; the fixtures must carry them as NaN, not as 0."""
    source = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2)
    seen = torch.ones(2, 4, 3)  # `(B, T, K)`: one flag per keypoint, never per coordinate
    seen[0, 1] = seen[0, 3] = 0.0
    before = source.clone()
    out = models.masked_nan(source, seen)
    assert out.shape == source.shape and out.dtype == source.dtype
    assert torch.isnan(out[0, 1]).all() and torch.isnan(out[0, 3]).all()
    assert torch.equal(out[0, 0], source[0, 0]) and torch.equal(out[1], source[1])  # the rest stay put
    assert not torch.isnan(out[0, 0]).any() and torch.isfinite(out[1]).all()
    assert torch.equal(source, before)  # the input is never mutated


def test_random_weights_follow_the_initialisation_scale_and_the_seed():
    """Noise at each layer's own scale, seeded per case: that is what makes a 2e-4 tolerance meaningful."""
    cfg = {"backbone": {"name": "tiny", "width": 4, "out_channels": 8}, "num_keypoints": 5, "embed_dim": 8}
    first, second, fresh = build_model("pose", cfg), build_model("pose", cfg), build_model("pose", cfg)
    models.random_weights(first, 11)
    models.random_weights(second, 11)
    a, b, c = first.state_dict(), second.state_dict(), fresh.state_dict()
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert not all(torch.equal(a[n], c[n]) for n in a)
    for name, buf in a.items():
        if name.endswith("running_var"):
            assert bool(((buf >= 0.4) & (buf <= 1.6)).all()), name
        elif name.endswith("running_mean"):
            assert bool((buf.abs() < 5).all()), name
        elif name.endswith("num_batches_tracked"):
            assert torch.equal(buf, c[name]), f"{name}: torch's counter is nobody's business"
    conv = next(t for n, t in a.items() if n.endswith("encoder.stem.0.0.weight"))
    scale = 1.0 / math.sqrt(conv.numel() / conv.shape[0])
    assert 0.2 * scale < float(conv.std()) < 5 * scale, conv.shape

    behaviour = build_model("behavior", {"num_keypoints": 25, "dims": 2, "hidden": 6, "classes": ["a", "b"],
                                         "dropout": 0.0})
    models.random_weights(behaviour, 3)
    named = dict(behaviour.named_parameters())
    gamma = {n: t for n, t in named.items() if ".norm." in n and n.endswith(".weight")}
    beta = {n: t for n, t in named.items() if ".norm." in n and n.endswith(".bias")}
    assert gamma and all(bool(((v >= 0.6) & (v <= 1.4)).all()) for v in gamma.values())
    assert beta and bool((torch.cat(list(beta.values())).abs() > 0).any())
    other = {n: t for n, t in named.items() if ".norm." not in n and n.endswith(".bias")}
    assert other and all(float(t.abs().max()) < 5 for t in other.values())


def test_deviation_and_sha256_report_the_differences_the_generator_promises(tmp_path):
    """The printed deviation is what the Rust test has to tolerate on top of its own tolerance."""
    left = {"outputs": {"heat": {"shape": [2], "data": [0.5, 1.0]}}, "weights": {"w": {"shape": [1], "data": [3.0]}}}
    assert models.deviation(left, json.loads(json.dumps(left))) == 0.0
    shifted = json.loads(json.dumps(left))
    shifted["outputs"]["heat"]["data"][1] = 1.25
    assert models.deviation(left, shifted) == pytest.approx(0.25)
    reshaped = json.loads(json.dumps(left))
    reshaped["weights"]["w"]["shape"] = [2]
    reshaped["weights"]["w"]["data"] = [3.0, 4.0]
    reshaped["outputs"]["heat"]["shape"] = [3]
    reshaped["outputs"]["heat"]["data"] = [0.5, 1.0, 2.0]
    assert models.deviation(left, reshaped) == float("inf")
    path = tmp_path / "fixture.json"
    path.write_text("{}", encoding="utf-8")
    assert models.sha256(path) == hashlib.sha256(b"{}").hexdigest()
