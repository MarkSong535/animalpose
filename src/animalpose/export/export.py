"""TorchScript and ONNX export with numerical parity checks.

Every exported graph takes plain float tensors and contains only standard ops
(conv, batch-norm, LSTM, resize, top-k, gather, elementwise), so it can be run
from C++ with LibTorch (``torch::jit::load``) or ONNX Runtime (CUDA / CoreML /
CPU execution providers). Pre-/post-processing that stays outside the graph is
deliberately simple and documented in ``manifest.json``:

* detector input: RGB float32 in [0, 1], NCHW, H and W multiples of 32
  (letterbox-resize the frame);
* pose input: RGB crops from ``cv::warpAffine`` with the 2x3 matrix described
  by :func:`animalpose.geometry.crop_affine` (square box, 1.25 padding);
* score thresholds, the tracker (Hungarian assignment) and triangulation run
  outside the networks.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from animalpose.behavior.model import BehaviorExport, BehaviorTCN
from animalpose.detector.model import CenterDetector, DetectorExport
from animalpose.labels import BEHAVIORS
from animalpose.pose.model import PoseExport, PoseNet
from animalpose.skeleton import CANONICAL
from animalpose.temporal.model import TemporalExport, TemporalLSTM


def _as_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def export_torchscript(module: nn.Module, inputs: tuple, path: str | Path, atol: float = 1e-4) -> dict:
    module = module.eval().cpu()
    inputs = tuple(i.cpu() for i in inputs)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(module, inputs, check_trace=False)
        traced.save(str(path))
        loaded = torch.jit.load(str(path))
        ref = _as_tuple(module(*inputs))
        got = _as_tuple(loaded(*inputs))
    err = max(float((r - g).abs().max()) for r, g in zip(ref, got))
    return {"path": str(path), "max_abs_err": err, "ok": err <= atol}


def export_onnx(module: nn.Module, inputs: tuple, path: str | Path, input_names: list[str],
                output_names: list[str], dynamic_axes: dict, opset: int = 18, exporter: str = "auto",
                atol: float = 1e-3, verify_inputs: tuple | None = None) -> dict:
    """Export to ONNX and compare against PyTorch with ONNX Runtime.

    ``exporter``: ``"dynamo"`` (torch.export based), ``"legacy"`` (TorchScript
    based) or ``"auto"`` (dynamo, falling back to legacy). ``verify_inputs``
    (e.g. a different batch size / sequence length) checks that dynamic axes
    really are dynamic.
    """
    # Example batch/time sizes must not be 1: torch.export specialises size-1 dims.
    module = module.eval().cpu()
    inputs = tuple(i.cpu() for i in inputs)
    errors = []
    used = None
    for kind in (["dynamo", "legacy"] if exporter == "auto" else [exporter]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "dynamo":
                    dims = {}
                    for name, axes in dynamic_axes.items():
                        if name in input_names:
                            dims[name] = {ax: torch.export.Dim.AUTO for ax in axes}
                    torch.onnx.export(module, inputs, str(path), input_names=input_names, output_names=output_names,
                                      dynamic_shapes=[dims.get(n, {}) for n in input_names], opset_version=opset,
                                      dynamo=True, verbose=False)
                else:
                    torch.onnx.export(module, inputs, str(path), input_names=input_names, output_names=output_names,
                                      dynamic_axes=dynamic_axes, opset_version=opset, dynamo=False)
            used = kind
            report = verify_onnx(module, path, input_names, [inputs] + ([verify_inputs] if verify_inputs else []), atol)
            if report["ok"]:
                break
            errors.append(f"{kind}: parity failed {report['max_abs_err']:.2e}")
        except Exception as e:  # noqa: BLE001 - try the next exporter
            errors.append(f"{kind}: {type(e).__name__}: {str(e)[:300]}")
            report = {"ok": False, "max_abs_err": float("nan")}
    return {"path": str(path), "exporter": used, **report, "errors": errors}


def verify_onnx(module: nn.Module, path, input_names, input_sets, atol: float = 1e-3) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for inputs in input_sets:
        with torch.no_grad():
            ref = [r.numpy() for r in _as_tuple(module(*inputs))]
        got = sess.run(None, {n: i.numpy() for n, i in zip(input_names, inputs)})
        for r, g in zip(ref, got):
            if r.dtype.kind in "iu":  # indices: compare exactly
                worst = max(worst, float(np.abs(r.astype(np.int64) - g.astype(np.int64)).max()))
            else:
                worst = max(worst, float(np.abs(r - g).max()))
    return {"ok": worst <= atol, "max_abs_err": worst}


def export_all(out_dir: str | Path, detector: CenterDetector | None = None, pose: PoseNet | None = None,
               temporal: TemporalLSTM | None = None, behavior: BehaviorTCN | None = None,
               det_size: tuple[int, int] = (256, 320), crop_size: int = 128, window: int = 64,
               max_det: int = 20, formats=("torchscript", "onnx"), pair_decoding: bool = False) -> dict:
    """Export every provided model and write ``manifest.json`` for C++ consumers.

    ``pair_decoding`` bakes :func:`animalpose.pose.model.decode_pairs` into the pose graph.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = CANONICAL.num_keypoints
    report: dict = {}
    manifest: dict = {"keypoints": CANONICAL.names, "edges": CANONICAL.edge_index.tolist(),
                      "behaviors": list(BEHAVIORS), "models": {}}
    specs = []
    if detector is not None:
        h, w = det_size
        specs.append(("detector", DetectorExport(detector, max_det), (torch.rand(2, 3, h, w),),
                      (torch.rand(1, 3, h + 32, w + 64),), ["images"], ["boxes", "scores"],
                      {"images": {0: "batch", 2: "height", 3: "width"}},
                      {"input": "RGB float32 [0,1] NCHW, H/W multiple of 32", "max_det": max_det,
                       "stride": detector.stride, "det_size": [h, w]}))
    if pose is not None:
        specs.append(("pose", PoseExport(pose, pair_decoding), (torch.rand(2, 3, crop_size, crop_size),),
                      (torch.rand(3, 3, crop_size, crop_size),), ["crops"], ["keypoints", "confidence", "embedding"],
                      {"crops": {0: "batch"}},
                      {"input": f"RGB float32 [0,1] crops {crop_size}x{crop_size}", "crop_padding": 1.25,
                       "crop_size": crop_size, "pair_decoding": pair_decoding}))
    if temporal is not None:
        d = temporal.d
        ex = (torch.rand(2, window, k, d) * 100, torch.rand(2, window, k), (torch.rand(2, window, k) > 0.2).float())
        ver = (torch.rand(3, window // 2, k, d) * 100, torch.rand(3, window // 2, k),
               (torch.rand(3, window // 2, k) > 0.2).float())
        specs.append(("temporal", TemporalExport(temporal), ex, ver, ["coords", "conf", "observed"],
                      ["coords_out", "confidence", "imputed"],
                      {n: {0: "batch", 1: "time"} for n in ["coords", "conf", "observed"]},
                      {"dims": d, "window": window, "bidirectional": temporal.bidirectional}))
    if behavior is not None:
        d = behavior.features.d
        ex = (torch.rand(2, window, k, d) * 100, (torch.rand(2, window, k) > 0.2).float())
        ver = (torch.rand(3, window * 2, k, d) * 100, (torch.rand(3, window * 2, k) > 0.2).float())
        specs.append(("behavior", BehaviorExport(behavior), ex, ver, ["keypoints", "mask"], ["probs"],
                      {n: {0: "batch", 1: "time"} for n in ["keypoints", "mask"]},
                      {"dims": d, "classes": list(behavior.classes)}))
    for name, module, ex, ver, ins, outs, dyn, info in specs:
        module.eval()
        entry = {"inputs": ins, "outputs": outs, **info}
        if "torchscript" in formats:
            report[f"{name}.torchscript"] = export_torchscript(module, ex, out / f"{name}.ts")
            entry["torchscript"] = f"{name}.ts"
        if "onnx" in formats:
            report[f"{name}.onnx"] = export_onnx(module, ex, out / f"{name}.onnx", ins, outs, dyn, verify_inputs=ver)
            entry["onnx"] = f"{name}.onnx"
        manifest["models"][name] = entry
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    return report
