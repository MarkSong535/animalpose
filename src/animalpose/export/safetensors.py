"""Safetensors interchange: the weights and architecture of a checkpoint in one file.

The tensor names are exactly the checkpoint's ``state_dict`` keys (``backbone.encoder.stages.0.0.conv1.0.weight``,
``heat_queries``, ...), so the Rust ``models`` module can drop them straight into its layers by name. The
architecture travels with them: the header's metadata block keeps ``"kind"`` (the model kind),
``"animalpose"`` (the JSON model config :func:`animalpose.train.checkpoint.build_model` consumes) and
``"extra"`` (any other checkpoint entry, e.g. ``crop_size``, ``window``, ``input_size``), which is what
:func:`checkpoint_from_safetensors` puts back.

One file per checkpoint::

    {
      "TENSOR_NAME": {"dtype": "F32", "shape": [...], "data_offsets": [begin, end]},
      ...,
      "__metadata__": {"animalpose": "{...model config json...}", "kind": "pose", "extra": "{...}"}
    }

:func:`write_safetensors` uses the ``safetensors`` wheel when it is installed and otherwise falls back to a
dependency-free writer of the same format (8-byte little-endian header length, JSON header, raw
little-endian tensor bytes); :func:`read_safetensors` mirrors it. Both directions are exercised
round-trip by :func:`compare`, which the export command runs against the original checkpoint.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

try:  # the reference bindings, used when available
    from safetensors import safe_open as _safe_open
    from safetensors.torch import load_file as _load_file
    from safetensors.torch import save_file as _save_file
except Exception:  # pragma: no cover - falls back to the local reader and writer
    _safe_open = _load_file = _save_file = None

__all__ = ["DTYPES", "checkpoint_from_safetensors", "compare", "metadata_for", "read_safetensors",
           "split_checkpoint", "write_safetensors"]

# torch dtype -> safetensors dtype name / numpy struct code (little-endian).
DTYPES = {
    torch.float32: ("F32", "<f4"),
    torch.float64: ("F64", "<f8"),
    torch.float16: ("F16", "<f2"),
    torch.bfloat16: ("BF16", None),
    torch.int64: ("I64", "<i8"),
    torch.int32: ("I32", "<i4"),
    torch.int16: ("I16", "<i2"),
    torch.int8: ("I8", "<i1"),
    torch.uint8: ("U8", "<u1"),
    torch.bool: ("BOOL", "<u1"),
}

# safetensors dtype name -> (element torch dtype, numpy dtype for the fallback reader).
READ_DTYPES = {
    "F32": (torch.float32, "float32"),
    "F64": (torch.float64, "float64"),
    "F16": (torch.float16, "float16"),
    "BF16": (torch.bfloat16, None),
    "I64": (torch.int64, "int64"),
    "I32": (torch.int32, "int32"),
    "I16": (torch.int16, "int16"),
    "I8": (torch.int8, "int8"),
    "U8": (torch.uint8, "uint8"),
    "BOOL": (torch.bool, "uint8"),
}

# Checkpoint entries that are not weights; everything else is a tensor.
META_KEYS = ("kind", "model_cfg", "state_dict")


def write_safetensors(path: str | Path, tensors: dict[str, torch.Tensor],
                      metadata: dict[str, str] | None = None) -> None:
    """Writes ``tensors`` (and optional string ``metadata``) in the safetensors format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # the wheel refuses to create the folder
    if _save_file is not None:
        clean = {k: v.detach().cpu().clone().contiguous() for k, v in tensors.items()}
        _save_file(clean, str(path), metadata=metadata)
        return
    header: dict[str, dict] = {}
    body = bytearray()
    for name, tensor in tensors.items():
        t = tensor.detach().cpu().contiguous()
        dtype, code = DTYPES[t.dtype]
        if code is None:  # bfloat16 has no numpy counterpart; move the bits through uint16
            raw = t.view(torch.uint16).numpy().tobytes()
        else:
            raw = t.numpy().tobytes()
        begin, end = len(body), len(body) + len(raw)
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [begin, end]}
        body += raw
    if metadata:
        header["__metadata__"] = dict(metadata)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8  # the data section starts on an 8-byte boundary
    blob += b" " * pad
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(body)


def read_safetensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """``(tensors, metadata)`` of a ``.safetensors`` file."""
    path = Path(path)
    if _load_file is not None:
        tensors = _load_file(str(path))
        with _safe_open(str(path), framework="pt", device="cpu") as fh:
            meta = dict(fh.metadata() or {})
        return tensors, meta
    import numpy as np

    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        blob = fh.read(n)
        body = fh.read()
    header = json.loads(blob)
    meta = dict(header.pop("__metadata__", {}))
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        dtype, np_dtype = READ_DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        raw = body[begin:end]
        shape = tuple(info["shape"])
        if info["dtype"] == "BF16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16)
        else:
            t = torch.from_numpy(np.frombuffer(raw, dtype=np_dtype).copy())
        tensors[name] = t.reshape(shape).to(dtype if info["dtype"] != "BOOL" else torch.bool)
    return tensors, meta


def split_checkpoint(ckpt: dict) -> tuple[str, dict, dict[str, torch.Tensor], dict]:
    """``(kind, model_cfg, tensors, extra)`` of a checkpoint written by :func:`save_checkpoint`."""
    kind = ckpt.get("kind")
    cfg = ckpt.get("model_cfg", ckpt.get("model")) or {}
    state = ckpt.get("state_dict", ckpt)
    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    extra = {k: v for k, v in ckpt.items() if k not in META_KEYS and not isinstance(v, torch.Tensor)}
    return str(kind), dict(cfg), tensors, extra


def metadata_for(kind: str, cfg: dict, extra: dict) -> dict[str, str]:
    """The header's ``__metadata__`` block: string values only, as the format requires."""
    meta = {"animalpose": json.dumps(cfg), "kind": kind}
    if extra:
        meta["extra"] = json.dumps(extra)
    return meta


def checkpoint_from_safetensors(path: str | Path) -> dict:
    """The ``{"kind", "model_cfg", "state_dict", ...}`` dict stored in a safetensors file."""
    tensors, meta = read_safetensors(path)
    ckpt: dict = {"kind": meta.get("kind"), "model_cfg": json.loads(meta.get("animalpose", "{}")),
                  "state_dict": tensors}
    ckpt.update(json.loads(meta.get("extra", "{}")))
    return ckpt


def compare(a: dict, b: dict) -> tuple[int, float]:
    """``(number of entries checked, largest absolute difference)`` between two checkpoints."""
    worst = 0.0
    checked = 0
    for name, t in a["state_dict"].items():
        u = b["state_dict"].get(name)
        if u is None:
            raise ValueError(f"tensor {name} is missing from the rebuilt checkpoint")
        if tuple(t.shape) != tuple(u.shape):
            raise ValueError(f"tensor {name}: shape {tuple(t.shape)} != {tuple(u.shape)}")
        if t.dtype != u.dtype:
            raise ValueError(f"tensor {name}: dtype {t.dtype} != {u.dtype}")
        checked += 1
        if t.is_floating_point():
            worst = max(worst, float((t.float() - u.float()).abs().max()))
        elif torch.any(t != u):
            raise ValueError(f"tensor {name}: integer/buffer values differ")
    for key in set(a) - {"state_dict"} | set(b) - {"state_dict"}:
        if a.get(key) != b.get(key):
            raise ValueError(f"checkpoint entry {key} differs: {a.get(key)!r} != {b.get(key)!r}")
    for name in set(b["state_dict"]) - set(a["state_dict"]):
        raise ValueError(f"tensor {name} is not in the original checkpoint")
    return checked, worst
