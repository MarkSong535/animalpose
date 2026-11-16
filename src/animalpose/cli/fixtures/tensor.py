"""Dump torch reference values for the Rust tensor core (`rust/crates/animalpose-tensor`).

Every operation in `rust/crates/animalpose-tensor/src/ops.rs` gets one JSON file under
``tests/fixtures/tensor/``: the inputs, the operation's scalars, the forward output, and the gradient
of ``output.sum()`` with respect to every input. The Rust tests
(``rust/crates/animalpose-tensor/src/tests.rs``) compare their own forward values and
``Graph::backward`` gradients against these with a 2e-4 absolute tolerance; a missing file makes the
matching test print and skip, so a fresh clone without fixtures still passes.

    animalpose fixtures tensor
    animalpose fixtures tensor --out /tmp/fx && ANIMALPOSE_FIXTURES=/tmp/fx ...

One file per operation::

    {"shape": [primary input shape],
     "inputs":  {"name": {"shape": [...], "data": [...]}, ...},   # row-major float32
     "params":  {"stride": 1, "pad": 1, ...},                      # scalars / index lists
     "output":  {"shape": [...], "data": [...]},
     "grad_input":  {"shape": [...], "data": [...]},               # d(sum(output))/d(inputs["x"])
     "grad_params": {"name": {"shape": [...], "data": [...]}}}     # same for the other inputs

Everything is float32 and tiny (nothing larger than ``3x4x8x8``) so the comparison is a true numerical
parity check between two independent f32 implementations rather than a test of accumulation order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from animalpose.paths import FIXTURES_DIR

NAME = "tensor"

SEED = 1234


def enc(t: torch.Tensor) -> dict:
    """One tensor -> ``{"shape": ..., "data": ...}`` in row-major float32."""
    a = np.ascontiguousarray(np.asarray(t.detach().cpu().numpy(), dtype=np.float32))
    return {"shape": list(a.shape), "data": [float(v) for v in a.reshape(-1)]}


def randn(*shape: int, scale: float = 1.0, grad: bool = False) -> torch.Tensor:
    """A deterministic random tensor (``torch.manual_seed`` is set once in :func:`run`)."""
    t = torch.randn(*shape, dtype=torch.float32) * scale
    return t.requires_grad_(grad)


def dump(out_dir: Path, name: str, inputs: dict[str, torch.Tensor], params: dict, out: torch.Tensor) -> str:
    """Back-propagates ``sum(output)``, then writes one fixture file."""
    (out if out.numel() == 1 else out.sum()).backward()
    x = inputs["x"]
    fixture = {
        "shape": list(x.shape),
        "inputs": {k: enc(v) for k, v in inputs.items()},
        "params": params,
        "output": enc(out),
        "grad_input": enc(x.grad),
        "grad_params": {
            k: enc(v.grad) for k, v in inputs.items() if k != "x" and v.grad is not None
        },
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    return path.name


def conv2d_case(out_dir: Path, name: str, n: int, c: int, h: int, o: int, k: int, s: int, p: int, d: int) -> str:
    x = randn(n, c, h, h, grad=True)
    w = randn(o, c, k, k, scale=0.4, grad=True)
    b = randn(o, scale=0.3, grad=True)
    out = F.conv2d(x, w, b, stride=s, padding=p, dilation=d)
    return dump(out_dir, name, {"x": x, "weight": w, "bias": b},
                {"stride": s, "pad": p, "dilation": d}, out)


def pool_case(out_dir: Path, name: str, kind: str, k: int, s: int, p: int) -> str:
    x = randn(2, 3, 6, 6, grad=True)
    out = F.max_pool2d(x, k, stride=s, padding=p) if kind == "max" else F.avg_pool2d(x, k, stride=s, padding=p)
    return dump(out_dir, name, {"x": x}, {"k": k, "stride": s, "pad": p}, out)


def norm_case(out_dir: Path, name: str, train: bool) -> str:
    x = randn(3, 4, 5, 5, grad=True)
    gamma = (torch.rand(4, dtype=torch.float32) * 0.6 + 0.7).requires_grad_(True)
    beta = randn(4, scale=0.3, grad=True)
    if train:
        out = F.batch_norm(x, None, None, weight=gamma, bias=beta, training=True)
        inputs = {"x": x, "gamma": gamma, "beta": beta}
    else:
        mean = randn(4, scale=0.4)
        var = (torch.rand(4, dtype=torch.float32) * 1.5 + 0.5)
        out = F.batch_norm(x, mean, var, weight=gamma, bias=beta, training=False)
        inputs = {"x": x, "gamma": gamma, "beta": beta, "running_mean": mean, "running_var": var}
    return dump(out_dir, name, inputs, {"eps": 1e-5, "mode": "train" if train else "eval"}, out)


def matmul_case(out_dir: Path, name: str, a_shape: tuple, b_shape: tuple) -> str:
    a = randn(*a_shape, scale=0.5, grad=True)
    b = randn(*b_shape, scale=0.5, grad=True)
    return dump(out_dir, name, {"x": a, "other": b}, {}, a @ b)


def loss_case(out_dir: Path, name: str, out: torch.Tensor, inputs: dict, params: dict) -> str:
    return dump(out_dir, name, inputs, params, out)


def lstm_case(out_dir: Path, name: str, n: int, t: int, inp: int, hidden: int, layers: int,
              bidirectional: bool) -> str:
    lstm = nn.LSTM(inp, hidden, layers, batch_first=True, bidirectional=bidirectional, dropout=0.0)
    lstm.eval()
    for pname, param in list(lstm.named_parameters()):  # random weights, replacing nn.LSTM's init
        setattr(lstm, pname, nn.Parameter(torch.randn_like(param) * 0.4))
    inputs = {"x": randn(n, t, inp, scale=0.6, grad=True)}
    inputs.update({pname: param for pname, param in lstm.named_parameters()})
    out, _ = lstm(inputs["x"])
    params = {"hidden": hidden, "layers": layers, "bidirectional": bidirectional, "dropout": 0.0}
    return dump(out_dir, name, inputs, params, out)


def add_arguments(parser) -> None:
    parser.add_argument("--out", type=Path, default=FIXTURES_DIR / "tensor")


def run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    out = args.out
    written: list[str] = []

    written.append(conv2d_case(out, "conv2d_k3_s1_p1", 2, 3, 6, 4, 3, 1, 1, 1))
    written.append(conv2d_case(out, "conv2d_dilated", 2, 3, 6, 3, 3, 1, 2, 2))
    written.append(pool_case(out, "max_pool2d", "max", 2, 2, 0))
    written.append(pool_case(out, "avg_pool2d", "avg", 2, 2, 1))
    written.append(norm_case(out, "batch_norm_train", True))
    written.append(norm_case(out, "batch_norm_eval", False))

    up = randn(1, 2, 3, 4, grad=True)
    written.append(dump(out, "upsample_nearest", {"x": up}, {"h": 6, "w": 8},
                        F.interpolate(up, size=(6, 8), mode="nearest")))

    written.append(matmul_case(out, "matmul_batched", (2, 5, 3), (2, 3, 4)))
    written.append(matmul_case(out, "matmul_broadcast", (1, 4, 3), (2, 3, 5)))
    written.append(matmul_case(out, "matmul_2d", (3, 6), (6, 2)))

    pm = randn(2, 3, 4, grad=True)
    written.append(dump(out, "permute", {"x": pm}, {"perm": [2, 0, 1]}, pm.permute(2, 0, 1)))

    a, b = randn(2, 3, 4, grad=True), randn(2, 3, 2, scale=0.7, grad=True)
    written.append(dump(out, "concat", {"x": a, "other": b}, {"axis": 2}, torch.cat([a, b], dim=2)))

    sl = randn(2, 3, 4, grad=True)
    written.append(dump(out, "slice", {"x": sl}, {"axis": 1, "start": 1, "end": 3}, sl[:, 1:3, :]))

    for tag, axes in (("axes1", (1,)), ("axes02", (0, 2)), ("all", (0, 1, 2))):
        rx = randn(2, 3, 4, grad=True)
        written.append(dump(out, f"reduce_sum_{tag}", {"x": rx}, {"axes": list(axes)}, rx.sum(dim=axes)))
        ry = randn(2, 3, 4, grad=True)
        written.append(dump(out, f"reduce_mean_{tag}", {"x": ry}, {"axes": list(axes)}, ry.mean(dim=axes)))

    sx = randn(2, 3, 4, scale=0.8, grad=True)
    written.append(dump(out, "softmax", {"x": sx}, {"axis": 1}, torch.softmax(sx, dim=1)))
    sy = randn(2, 3, 4, scale=0.8, grad=True)
    written.append(dump(out, "log_softmax", {"x": sy}, {"axis": 1}, torch.log_softmax(sy, dim=1)))
    nz = randn(2, 3, 4, grad=True)
    written.append(dump(out, "l2_normalize", {"x": nz}, {"axis": 1}, F.normalize(nz, dim=1)))

    logits = randn(2, 3, 4, scale=0.9, grad=True)
    bce_target = torch.sigmoid(logits.detach() * 1.3)
    written.append(loss_case(out, "bce_with_logits",
                             F.binary_cross_entropy_with_logits(logits, bce_target),
                             {"x": logits, "target": bce_target}, {}))
    diff = randn(2, 3, 4, grad=True)
    written.append(loss_case(out, "smooth_l1", F.smooth_l1_loss(diff, torch.zeros_like(diff)),
                             {"x": diff}, {"kind": "smooth_l1", "weight": 1.0}))
    mse = randn(2, 3, 4, grad=True)
    written.append(loss_case(out, "mse", F.mse_loss(mse, torch.zeros_like(mse)), {"x": mse},
                             {"kind": "mse", "weight": 1.0}))

    ce_logits = randn(10, 6, scale=1.2, grad=True)  # (N, C): classes on the last axis, like the Rust op
    ce_target = torch.randint(0, 6, (10,), dtype=torch.float32).clone()
    ce_target[2] = -100.0
    ce_target[7] = -100.0
    ce = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
    written.append(loss_case(out, "cross_entropy", ce(ce_logits, ce_target.long()),
                             {"x": ce_logits, "target": ce_target}, {"smoothing": 0.1, "ignore": -100}))
    cw_logits = randn(10, 6, scale=1.2, grad=True)
    cw_weight = torch.rand(6, dtype=torch.float32) * 1.6 + 0.2
    cwe = nn.CrossEntropyLoss(weight=cw_weight, label_smoothing=0.1, ignore_index=-100)
    written.append(loss_case(out, "cross_entropy_weighted", cwe(cw_logits, ce_target.long()),
                             {"x": cw_logits, "target": ce_target, "weight": cw_weight},
                             {"smoothing": 0.1, "ignore": -100}))

    written.append(lstm_case(out, "lstm_2layer_bidir", 2, 3, 5, 4, 2, True))
    written.append(lstm_case(out, "lstm_1layer", 2, 4, 3, 4, 1, False))

    sizes = sorted((out / n).stat().st_size for n in written)
    print(f"wrote {len(written)} fixtures to {out} ({sizes[-1] / 1024:.1f} KiB largest)")
