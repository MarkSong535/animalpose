import json

import pytest

from animalpose.behavior.model import BehaviorTCN
from animalpose.detector.model import CenterDetector
from animalpose.export.export import export_all
from animalpose.pose.model import PoseNet
from animalpose.temporal.model import TemporalLSTM

pytest.importorskip("onnxruntime")


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
