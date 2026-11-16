"""The ``.safetensors`` interchange contract the Rust port is validated against.

``rust/src/models/tests.rs`` loads ``<kind>_<name>.safetensors`` through ``load()`` and expects, from a file
``animalpose export safetensors`` wrote: tensor entries named exactly like the Python ``state_dict`` keys, a
``__metadata__`` block whose ``kind`` / ``animalpose`` / ``extra`` strings rebuild the checkpoint, dtype names
from the safetensors table, IEEE NaN and inf kept as NaN and inf rather than flattened to 0, and weights that
come back bit for bit. Every check below is a real round trip: a tiny pose network trained for two steps on
synthetic scenes, written, re-read, and compared parameter by parameter (names, shapes, dtypes, values). Both
writers and both readers the module offers are exercised -- the ``safetensors`` wheel and the dependency-free
fallback -- and each must read what the other wrote. The ONNX/TorchScript parity of these checkpoints is
``tests/test_export.py``'s contract and is not repeated here.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import pytest
import torch

from animalpose.cli import main
from animalpose.cli.export import import_safetensors as import_cli
from animalpose.cli.export import models as models_cli
from animalpose.cli.export import safetensors as export_cli
from animalpose.detector.model import CenterDetector
from animalpose.export import safetensors as st
from animalpose.export.safetensors import (
    DTYPES,
    READ_DTYPES,
    checkpoint_from_safetensors,
    compare,
    metadata_for,
    read_safetensors,
    split_checkpoint,
    write_safetensors,
)
from animalpose.pose.model import PoseNet
from animalpose.train.checkpoint import build_model, load_model, save_checkpoint
from animalpose.train.loops import train_pose

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
TRAIN = {"steps": 2, "batch_size": 2, "log_every": 0, "workers": 0, "seed": 7}
SCENES = {"n_sims": 1, "frames": 24, "image_size": [160, 128]}

# safetensors dtype name -> struct type char of one element (`struct` spells the width in the char).
STRUCT = {"F16": "e", "F32": "f", "F64": "d", "I8": "b", "I16": "h", "I32": "i", "I64": "q", "U8": "B",
          "BOOL": "B"}
# safetensors dtype name -> the module's numpy-style element code, cross-checked against STRUCT.
CODE_BY_NAME = {name: code for name, code in DTYPES.values()}
_BITS = {torch.float16: torch.int16, torch.bfloat16: torch.uint16, torch.float32: torch.int32,
         torch.float64: torch.int64}


@pytest.fixture(params=["wheel", "fallback"])
def backend(request, monkeypatch):
    """Each test runs against both code paths: the ``safetensors`` wheel, and the pure-Python fallback."""
    if request.param == "fallback":
        for name in ("_save_file", "_load_file", "_safe_open"):
            monkeypatch.setattr(st, name, None)
    return request.param


@pytest.fixture(scope="module")
def pose_ckpt(tmp_path_factory):
    """A pose checkpoint from two training steps: BN running statistics, int64 step counters, ``crop_size``."""
    out = tmp_path_factory.mktemp("st-runs") / "pose.pt"
    _, history = train_pose({"device": "cpu", "output": str(out), "model": {"backbone": TINY, "embed_dim": 16},
                             "data": {**SCENES, "crop_size": 64}, "train": TRAIN})
    assert len(history) == TRAIN["steps"]
    return out


def _split(path: Path) -> tuple[dict, str, dict, dict[str, torch.Tensor], dict]:
    """``(original, kind, cfg, tensors, extra)`` of a ``.pt`` checkpoint, via the module's own splitter."""
    original = torch.load(path, map_location="cpu", weights_only=False)
    kind, cfg, tensors, extra = split_checkpoint(original)
    return original, kind, cfg, tensors, extra


def _round_trip(path: Path, out: Path) -> tuple[dict, dict]:
    """Export ``path`` and import it again, exactly as the two CLI verbs do."""
    original, kind, cfg, tensors, extra = _split(path)
    write_safetensors(out, tensors, metadata_for(kind, cfg, extra))
    return original, checkpoint_from_safetensors(out)


def _header_and_body(path: Path) -> tuple[dict, bytes]:
    """The parsed header and the data section, read with nothing but ``json`` and ``struct``."""
    blob = path.read_bytes()
    (length,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8:8 + length]), blob[8 + length:]


def _element(name: str) -> tuple[str, int]:
    """``("f", 4)``: the ``struct`` type char of one element of that dtype and its width in bytes."""
    char = STRUCT[name]
    assert struct.calcsize(f"<{char}") == int(CODE_BY_NAME[name][2:])  # both tables agree on the width
    return char, struct.calcsize(f"<{char}")


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The raw bits of ``t`` as an integer tensor, so NaN and -0.0 compare exactly."""
    if t.dtype == torch.bool:
        return t.to(torch.uint8)
    if not t.is_floating_point():
        return t
    return t.view(_BITS[t.dtype])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True when ``a`` and ``b`` are the same dtype, shape and bit pattern."""
    return a.dtype == b.dtype and a.shape == b.shape and bool((_bits(a) == _bits(b)).all())


def test_a_trained_checkpoint_round_trips_tensor_for_tensor(pose_ckpt, tmp_path, backend):
    """The contract the Rust loader depends on: every parameter returns with its name, dtype and bits."""
    original, rebuilt = _round_trip(pose_ckpt, tmp_path / "pose.safetensors")
    want, got = original["state_dict"], rebuilt["state_dict"]
    assert sorted(got) == sorted(want)
    assert rebuilt["kind"] == original["kind"] == "pose"
    assert rebuilt["model_cfg"] == original["model_cfg"]
    assert rebuilt["crop_size"] == original["crop_size"] == 64
    mismatched = [n for n, t in want.items() if not _same(t, got[n])]
    assert mismatched == []
    checked, worst = compare(original, rebuilt)
    assert (checked, worst) == (len(want), 0.0)
    # torch's step counters are int64 buffers: they travel as I64 and must not be rounded through float.
    counters = [n for n, t in want.items() if t.dtype == torch.int64]
    assert counters and all(got[n].dtype == torch.int64 and _same(got[n], want[n]) for n in counters)
    # Rebuilding the network from the re-imported checkpoint is what `load_model` does in production.
    restored = tmp_path / "restored.pt"
    torch.save(rebuilt, restored)
    model = load_model(restored)
    assert isinstance(model, PoseNet) and model.num_keypoints == 25
    restored_state = model.state_dict()
    assert all(_same(restored_state[n], want[n]) for n in want)


def test_the_header_is_plain_json_and_the_bytes_decode_without_torch(pose_ckpt, tmp_path, backend):
    """A consumer with no Python at all can parse the file: 8-byte length, JSON header, raw little-endian data."""
    path = tmp_path / "pose.safetensors"
    original, kind, cfg, tensors, extra = _split(pose_ckpt)
    write_safetensors(path, tensors, metadata_for(kind, cfg, extra))
    header, body = _header_and_body(path)
    meta = header.pop("__metadata__")
    assert set(meta) == {"animalpose", "kind", "extra"}
    assert meta["kind"] == "pose"
    assert json.loads(meta["animalpose"]) == cfg
    assert json.loads(meta["extra"]) == {"crop_size": 64}
    assert set(header) == set(original["state_dict"])

    spans = sorted(tuple(v["data_offsets"]) for v in header.values())
    assert spans[0][0] == 0 and spans[-1][1] == len(body)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # contiguous, no overlap, no gaps

    for name, entry in header.items():
        dtype = original["state_dict"][name].dtype
        assert entry["dtype"] == DTYPES[dtype][0]
        assert entry["shape"] == list(original["state_dict"][name].shape)
        begin, end = entry["data_offsets"]
        assert end - begin == original["state_dict"][name].numel() * original["state_dict"][name].element_size()

    # Two tensors decoded with `struct` alone (no torch, no numpy, no safetensors) to the last bit.
    picked = [next(n for n, t in original["state_dict"].items() if t.dtype == wanted)
              for wanted in (torch.float32, torch.int64)]
    assert len(set(picked)) == 2
    for name in picked:
        entry = header[name]
        begin, end = entry["data_offsets"]
        raw = body[begin:end]
        char, size = _element(entry["dtype"])
        flat = list(struct.unpack(f"<{len(raw) // size}{char}", raw))
        assert flat == original["state_dict"][name].reshape(-1).tolist()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_nan_and_inf_survive_the_encoding_instead_of_becoming_zero(tmp_path, backend, dtype):
    """Unobserved keypoints are NaN; an exporter that flushes them to 0 invents a coordinate of (0, 0)."""
    src = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 1.5, 3.25], dtype=dtype)
    path = tmp_path / f"special-{str(dtype).split('.')[-1]}.safetensors"
    write_safetensors(path, {"special": src}, {"note": "non-finite"})
    got, meta = read_safetensors(path)
    assert meta == {"note": "non-finite"}
    back = got["special"]
    assert _same(back, src)
    assert math.isnan(back[0].item()) and back[0].item() != 0.0
    assert back[1].item() == float("inf") and back[2].item() == -float("inf")

    # The file itself, not just the reader: the element bytes are the IEEE pattern.
    header, body = _header_and_body(path)
    begin, end = header["special"]["data_offsets"]
    raw = body[begin:end]
    assert raw != b"\x00" * len(raw)
    char, size = _element(DTYPES[dtype][0])
    (first,) = struct.unpack(f"<{char}", raw[:size])
    assert math.isnan(first)


@pytest.mark.parametrize("dtype", sorted(DTYPES, key=str))
def test_every_documented_dtype_keeps_its_name_and_values(tmp_path, backend, dtype):
    """The header's dtype string is the contract: the Rust loader picks its element type from it."""
    src = _probe(dtype).clone()
    name = f"t_{str(dtype).split('.')[-1]}"
    path = tmp_path / f"dtypes-{str(dtype).split('.')[-1]}.safetensors"
    write_safetensors(path, {name: src}, {"dtype": str(dtype)})
    got, meta = read_safetensors(path)
    assert meta == {"dtype": str(dtype)}
    assert _same(got[name], src)
    header, _ = _header_and_body(path)
    assert header[name]["dtype"] == DTYPES[dtype][0]
    assert READ_DTYPES[header[name]["dtype"]][0] == dtype
    assert header[name]["shape"] == list(src.shape)


def _probe(dtype: torch.dtype) -> torch.Tensor:
    """A tiny tensor of ``dtype`` including every non-finite pattern that dtype can hold."""
    if dtype == torch.bool:
        return torch.tensor([[True], [False]])
    if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.tensor([[1, 2], [3, 4]], dtype=dtype)
    return torch.tensor([[1.5, -2.25], [float("nan"), float("inf")]], dtype=dtype)


def test_the_fallback_and_the_wheel_writes_read_each_other(pose_ckpt, tmp_path):
    """The dependency-free writer is a drop-in for the wheel: same entries, same bytes, both directions."""
    _, kind, cfg, tensors, extra = _split(pose_ckpt)
    meta = metadata_for(kind, cfg, extra)
    wheel, fallback = tmp_path / "wheel.safetensors", tmp_path / "fallback.safetensors"

    write_safetensors(wheel, tensors, meta)
    saved = st._save_file, st._load_file, st._safe_open
    st._save_file = st._load_file = st._safe_open = None
    try:
        write_safetensors(fallback, tensors, meta)
        from_fallback, fallback_meta = read_safetensors(fallback)
    finally:
        st._save_file, st._load_file, st._safe_open = saved
    from_wheel, wheel_meta = read_safetensors(wheel)

    assert fallback_meta == wheel_meta == meta
    assert sorted(from_fallback) == sorted(from_wheel) == sorted(tensors)
    assert all(_same(from_fallback[n], from_wheel[n]) for n in tensors)
    assert all(_same(from_wheel[n], tensors[n]) for n in tensors)

    wheel_header, wheel_body = _header_and_body(wheel)
    fallback_header, fallback_body = _header_and_body(fallback)
    for name, entry in fallback_header.items():
        if name == "__metadata__":
            continue
        other = wheel_header[name]
        assert entry["dtype"] == other["dtype"] and entry["shape"] == other["shape"]
        b0, b1 = entry["data_offsets"]
        o0, o1 = other["data_offsets"]
        assert wheel_body[o0:o1] == fallback_body[b0:b1]


def test_a_checkpoint_without_a_state_dict_key_splits_to_itself(tmp_path):
    """A bare ``{name: tensor}`` dict is a checkpoint too; ``model`` is accepted for ``model_cfg``."""
    tensors = {"heat_bias": torch.full((3,), -4.0), "step": torch.tensor(7)}
    kind, cfg, found, extra = split_checkpoint({**tensors, "note": "bare"})
    assert kind == "None" and cfg == {} and set(found) == set(tensors) and extra == {"note": "bare"}
    assert all(_same(found[n], tensors[n]) for n in tensors)

    kind, cfg, _, _ = split_checkpoint({"kind": "pose", "model": {"num_keypoints": 5}, "state_dict": {}})
    assert kind == "pose" and cfg == {"num_keypoints": 5}

    path = tmp_path / "bare.safetensors"
    write_safetensors(path, found, None)
    ckpt = checkpoint_from_safetensors(path)
    assert set(ckpt) == {"kind", "model_cfg", "state_dict"}
    assert ckpt["kind"] is None and ckpt["model_cfg"] == {}
    assert all(_same(ckpt["state_dict"][n], tensors[n]) for n in tensors)


def test_metadata_for_only_carries_extra_when_there_is_one():
    """The header block is strings only, as the format requires; ``extra`` is omitted when empty."""
    meta = metadata_for("pose", {"num_keypoints": 5}, {})
    assert set(meta) == {"animalpose", "kind"} and meta["kind"] == "pose"
    assert json.loads(meta["animalpose"]) == {"num_keypoints": 5}
    meta = metadata_for("temporal", {"hidden": 8}, {"window": 64, "input_size": [128, 160]})
    assert set(meta) == {"animalpose", "kind", "extra"}
    assert json.loads(meta["extra"]) == {"window": 64, "input_size": [128, 160]}
    assert all(isinstance(v, str) for v in meta.values())


def test_compare_names_the_first_difference_it_finds():
    """The ``--check`` verdict: a clean pair reports a count and a zero, every other pair raises by name."""
    a = {"kind": "pose", "state_dict": {"w": torch.ones(2, 2), "n": torch.tensor([1, 2])}}
    assert compare(a, _clone(a)) == (2, 0.0)
    with pytest.raises(ValueError, match="tensor w is missing"):
        compare(a, {"kind": "pose", "state_dict": {"n": a["state_dict"]["n"]}})
    with pytest.raises(ValueError, match=r"tensor w: shape \(2, 2\) != \(2, 1\)"):
        compare(a, {"kind": "pose", "state_dict": {"w": torch.ones(2, 1), "n": a["state_dict"]["n"]}})
    with pytest.raises(ValueError, match="tensor w: dtype"):
        compare(a, {"kind": "pose", "state_dict": {"w": torch.ones(2, 2, dtype=torch.float64),
                                                   "n": a["state_dict"]["n"]}})
    with pytest.raises(ValueError, match="integer/buffer values differ"):
        compare(a, {"kind": "pose", "state_dict": {"w": a["state_dict"]["w"], "n": torch.tensor([1, 3])}})
    with pytest.raises(ValueError, match="tensor extra is not in the original"):
        compare(a, {**_clone(a), "state_dict": {**a["state_dict"], "extra": torch.zeros(1)}})
    with pytest.raises(ValueError, match="checkpoint entry kind differs"):
        compare(a, {**_clone(a), "kind": "detector"})


def _clone(ckpt: dict) -> dict:
    return {"kind": ckpt["kind"], "state_dict": {k: v.clone() for k, v in ckpt["state_dict"].items()}}


def test_a_tuple_valued_entry_comes_back_as_a_list(pose_ckpt, tmp_path):
    """Non-tensor entries travel as JSON, so a tuple (``input_size``) returns as a list: ``compare`` says so."""
    det = tmp_path / "detector.pt"
    save_checkpoint(det, "detector", build_model("detector", {"backbone": TINY}), {"backbone": TINY},
                    {"input_size": (64, 96)})
    original, rebuilt = _round_trip(det, tmp_path / "detector.safetensors")
    assert rebuilt["input_size"] == [64, 96]
    assert all(_same(rebuilt["state_dict"][n], t) for n, t in original["state_dict"].items())
    with pytest.raises(ValueError, match="checkpoint entry input_size differs"):
        compare(original, rebuilt)


def test_export_safetensors_writes_what_the_command_writes(pose_ckpt, tmp_path, capsys, monkeypatch):
    """``animalpose export safetensors`` is a thin wrapper: same file, same entries, same printed kind."""
    library = tmp_path / "library.safetensors"
    original, kind, cfg, tensors, extra = _split(pose_ckpt)
    write_safetensors(library, tensors, metadata_for(kind, cfg, extra))

    out = tmp_path / "cli.safetensors"
    export_cli.run(argparse.Namespace(checkpoint=pose_ckpt, out=out))
    printed = capsys.readouterr().out
    assert f"wrote {out}" in printed and "kind=pose" in printed
    assert f"{len(tensors)} tensors" in printed and json.dumps(cfg) in printed

    from_library, meta_a = read_safetensors(library)
    from_cli, meta_b = read_safetensors(out)
    assert meta_a == meta_b
    assert sorted(from_cli) == sorted(tensors) and all(_same(from_cli[n], tensors[n]) for n in tensors)
    assert all(_same(from_cli[n], from_library[n]) for n in tensors)

    # The default output lands in the export directory, not in the working tree.
    exported = tmp_path / "export-dir"
    monkeypatch.setattr(export_cli, "EXPORT_DIR", exported)
    export_cli.run(argparse.Namespace(checkpoint=pose_ckpt, out=None))
    capsys.readouterr()
    assert (exported / "pose.safetensors").is_file()


def test_import_safetensors_restores_a_loadable_checkpoint_and_checks_it(pose_ckpt, tmp_path, capsys,
                                                                         monkeypatch):
    """``import-safetensors --check`` compares against the original and refuses anything but a zero diff."""
    safetensors = tmp_path / "pose.safetensors"
    original, kind, cfg, tensors, extra = _split(pose_ckpt)
    write_safetensors(safetensors, tensors, metadata_for(kind, cfg, extra))

    restored = tmp_path / "pose_restored.pt"
    import_cli.run(argparse.Namespace(safetensors=safetensors, out=restored, check=pose_ckpt))
    printed = capsys.readouterr().out
    assert f"wrote {restored}" in printed and f"{len(tensors)} tensors" in printed
    assert "max abs diff 0" in printed and json.dumps(cfg) in printed

    back = torch.load(restored, map_location="cpu", weights_only=False)
    assert set(back["state_dict"]) == set(original["state_dict"])
    assert all(_same(back["state_dict"][n], original["state_dict"][n]) for n in original["state_dict"])
    model = load_model(restored)
    assert model.num_keypoints == original["model_cfg"].get("num_keypoints", 25)

    exported = tmp_path / "import-dir"
    monkeypatch.setattr(import_cli, "EXPORT_DIR", exported)
    import_cli.run(argparse.Namespace(safetensors=safetensors, out=None, check=None))
    capsys.readouterr()
    assert (exported / "pose.pt").is_file()


def test_import_safetensors_refuses_a_round_trip_that_is_not_bit_exact(pose_ckpt, tmp_path, capsys):
    """A header that lies about one tensor must fail the check, not print a small number and pass."""
    original, kind, cfg, tensors, extra = _split(pose_ckpt)
    drifted = {k: (v + 0.5 if k == "heat_bias" else v) for k, v in tensors.items()}
    path = tmp_path / "drift.safetensors"
    write_safetensors(path, drifted, metadata_for(kind, cfg, extra))
    with pytest.raises(ValueError, match="not bit-exact"):
        import_cli.run(argparse.Namespace(safetensors=path, out=tmp_path / "drift.pt", check=pose_ckpt))
    capsys.readouterr()


def test_the_cli_verbs_drive_the_same_round_trip(tmp_path, pose_ckpt, capsys):
    """``animalpose export safetensors`` / ``export import-safetensors`` as parsed commands."""
    safetensors = tmp_path / "pose.safetensors"
    restored = tmp_path / "pose.pt"
    assert main(["export", "safetensors", "--checkpoint", str(pose_ckpt), "--out", str(safetensors)]) == 0
    assert main(["export", "import-safetensors", "--safetensors", str(safetensors), "--out",
                 str(restored), "--check", str(pose_ckpt)]) == 0
    printed = capsys.readouterr().out
    assert "max abs diff 0" in printed and "kind=pose" in printed

    original = torch.load(pose_ckpt, map_location="cpu", weights_only=False)
    back = torch.load(restored, map_location="cpu", weights_only=False)
    assert compare(original, back) == (len(original["state_dict"]), 0.0)


def test_the_export_cli_rejects_an_unknown_model_kind(tmp_path, capsys):
    """The kind in the header is what the Rust ``ModelKind::from_str`` reads, so an unknown one never ships."""
    ckpt = tmp_path / "mystery.pt"
    save_checkpoint(ckpt, "spline", build_model("pose", {"backbone": TINY, "embed_dim": 16}),
                    {"backbone": TINY, "embed_dim": 16}, {})
    out = tmp_path / "mystery.safetensors"
    assert main(["export", "safetensors", "--checkpoint", str(ckpt), "--out", str(out)]) == 1
    assert "unknown model kind" in capsys.readouterr().err
    assert not out.exists()


def test_the_detector_command_forwards_the_checkpoint_entries(tmp_path, monkeypatch):
    """``export models`` reads ``input_size``/``crop_size`` off the checkpoints -- the same entries ``export
    safetensors`` puts in its ``extra`` metadata -- and passes them to the exporter. ONNX/TorchScript parity
    itself is ``tests/test_export.py``'s contract, so the exporter is stubbed here to keep this cheap."""
    seen: dict = {}

    def fake_export_all(out, **kwargs):
        seen["out"] = out
        seen.update(kwargs)
        return {f"{k}.torchscript": {"ok": True, "max_abs_err": 0.0, "exporter": None}
                for k, v in kwargs.items() if v is not None}

    monkeypatch.setattr(models_cli, "export_all", fake_export_all)
    det = tmp_path / "detector.pt"
    save_checkpoint(det, "detector", build_model("detector", {"backbone": TINY}), {"backbone": TINY},
                    {"input_size": (64, 96)})
    pose = tmp_path / "pose.pt"
    save_checkpoint(pose, "pose", build_model("pose", {"backbone": TINY, "embed_dim": 16}),
                    {"backbone": TINY, "embed_dim": 16}, {"crop_size": 64})

    models_cli.run(argparse.Namespace(detector=str(det), pose=str(pose), temporal=None, behavior=None,
                                      out=str(tmp_path / "exported"), det_size=None, crop_size=None, window=16,
                                      pair_decoding=True))
    assert seen["det_size"] == (64, 96) and seen["crop_size"] == 64
    assert seen["window"] == 16 and seen["pair_decoding"] is True
    assert isinstance(seen["detector"], CenterDetector) and isinstance(seen["pose"], PoseNet)
    assert seen["temporal"] is None and seen["behavior"] is None
    assert split_checkpoint(torch.load(det, map_location="cpu", weights_only=False))[3] == {"input_size": (64, 96)}

    models_cli.run(argparse.Namespace(detector=None, pose=None, temporal=None, behavior=None,
                                      out=str(tmp_path / "exported"), det_size=[96, 128], crop_size=32,
                                      window=8, pair_decoding=False))
    assert seen["det_size"] == (96, 128) and seen["crop_size"] == 32 and seen["pose"] is None

    monkeypatch.setattr(models_cli, "export_all",
                        lambda out, **kw: {"pose.onnx": {"ok": False, "max_abs_err": 1.0, "exporter": "legacy"}})
    with pytest.raises(SystemExit, match="parity check failed"):
        models_cli.run(argparse.Namespace(detector=None, pose=str(pose), temporal=None, behavior=None,
                                          out=str(tmp_path / "exported"), det_size=[64, 64], crop_size=64,
                                          window=16, pair_decoding=False))
