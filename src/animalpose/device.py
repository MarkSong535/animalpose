"""Device selection: CUDA, then Apple MPS, then CPU."""

from __future__ import annotations

import torch

_CHAIN = ("cuda", "mps", "cpu")


def _available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    return kind == "cpu"


def select_device(preferred: str | None = "auto") -> torch.device:
    """Return the best available device.

    ``"auto"``/``None`` tries CUDA -> MPS -> CPU. An explicit request such as
    ``"cuda:1"`` is honoured when available and otherwise falls down the same
    chain, so a config written on a GPU server still runs on a laptop.
    """
    if preferred in (None, "auto"):
        start = 0
    else:
        dev = torch.device(preferred)
        if _available(dev.type):
            return dev
        start = _CHAIN.index(dev.type) + 1 if dev.type in _CHAIN else 0
    for kind in _CHAIN[start:]:
        if _available(kind):
            return torch.device(kind)
    return torch.device("cpu")
