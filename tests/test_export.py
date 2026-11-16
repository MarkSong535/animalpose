import json

import pytest
import torch
from torch import nn

from animalpose.behavior.model import BehaviorTCN
from animalpose.detector.model import CenterDetector, DetectorExport
from animalpose.export.export import export_all, export_onnx, verify_onnx
from animalpose.pose.model import PoseNet
from animalpose.temporal.model import TemporalLSTM

pytest.importorskip("onnxruntime")


def test_detector_export_handles_tied_peaks_and_unused_boxes(tmp_path):
    class TwoPeaks(nn.Module):
        stride = 4

        def forward(self, images):
            heat = images[:, :1]
            zeros = torch.zeros_like(images[:, :2])
            return {"heat": heat, "size": zeros, "offset": zeros}

    images = torch.arange(100, dtype=torch.float32).view(1, 1, 10, 10) / 100
    images[:, :, 1, 1] = images[:, :, 7, 7] = 10
    images = images.repeat(2, 3, 1, 1)
    detector = DetectorExport(TwoPeaks(), 8)
    report = export_onnx(detector, (images,), tmp_path / "detector.onnx",
                         ["images"], ["boxes", "scores"], {}, exporter="legacy")
    assert report["ok"], report

    class Reordered(nn.Module):
        def forward(self, images):
            boxes, scores = detector(images)
            return boxes.flip(1), scores.flip(1)

    path = tmp_path / "reordered.onnx"
    report = export_onnx(Reordered(), (images,), path, ["images"], ["boxes", "scores"], {}, exporter="legacy")
    assert report["ok"], report
    assert verify_onnx(detector, path, ["images"], [(images,)])["ok"]


@pytest.mark.parametrize("pair_decoding", [False, True])
def test_export_all_models_parity(tmp_path, tiny_backbone, pair_decoding):
    report = export_all(tmp_path, CenterDetector(tiny_backbone), PoseNet(25, tiny_backbone, 16, context=pair_decoding),
                        TemporalLSTM(25, 2, hidden=16), BehaviorTCN(25, 3, hidden=16),
                        det_size=(64, 96), crop_size=64, window=16, pair_decoding=pair_decoding)
    assert len(report) == 8
    for name, r in report.items():
        assert r["ok"], (name, r)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["models"]) == {"detector", "pose", "temporal", "behavior"}
    assert len(manifest["keypoints"]) == 25
