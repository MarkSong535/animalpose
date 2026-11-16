"""The two real-data rules no other suite pins: Rat 7M's body-side occlusion rule, and
starting a wider behavior classifier from a narrower checkpoint. Mapping, splitting and
session loading are exercised by test_real_converters, test_behavior_sets and
test_tracks_and_degrade, so they are deliberately absent here."""

import numpy as np
import torch

from animalpose.behavior.model import BehaviorTCN
from animalpose.data.real import rat7m_visibility
from animalpose.labels import BEHAVIORS_EXT
from animalpose.multiview.camera import Camera
from animalpose.train.checkpoint import save_checkpoint
from animalpose.train.loops import _init


def test_rat7m_visibility_uses_body_side():
    markers = {"SpineF": np.array([10.0, 0, 50]), "SpineL": np.array([-10.0, 0, 50]),
               "HipL": np.array([-8.0, 5, 40]), "HipR": np.array([-8.0, -5, 40]), "SpineM": np.array([0.0, 0, 55])}
    cam_left = Camera.look_at([0, 500, 100], [0, 0, 40])  # camera on the animal's left (+y)
    vis = rat7m_visibility(markers, cam_left)
    assert vis["HipL"] == 2 and vis["HipR"] == 1 and vis["SpineM"] == 2


def test_init_skips_resized_heads(tmp_path):
    src = BehaviorTCN(25, 2, hidden=16)
    save_checkpoint(tmp_path / "b.pt", "behavior", src, {"hidden": 16})
    dst = _init(BehaviorTCN(25, 2, classes=BEHAVIORS_EXT, hidden=16), {"init": str(tmp_path / "b.pt")})
    torch.testing.assert_close(dst.inp.weight, src.inp.weight)
    assert dst.out.weight.shape[0] == 6
