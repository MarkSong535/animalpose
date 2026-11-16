"""Encoder backbones, keypoint-box geometry and the skeleton registries.

Checks that the torchvision encoder really is a stride-4/8/16/32 pyramid with the
channel counts its block family implies (bottleneck stages are 4x wider), that a box
fitted to keypoints is padded and ignores invalid points instead of running off to
them, that the crop affine is exactly invertible and lands the box centre on the crop
centre, that IoU is exact on known overlaps, that a species profile and a dataset
mapping actually decide which canonical points the models are told exist, and that a
registered mapping is usable by the remapper while source names it does not list stay
unlabeled.
"""

import numpy as np
import pytest
import torch

from animalpose.geometry import apply_affine, box_iou, box_xyxy_to_cs, crop_affine, invert_affine, keypoints_to_box
from animalpose.models.backbone import TorchvisionResNet, build_backbone
from animalpose.skeleton import CANONICAL
from animalpose.skeleton.species import (
    DATASET_MAPPINGS,
    SPECIES,
    SpeciesProfile,
    register_mapping,
    register_species,
    remap_keypoints,
)


def test_torchvision_encoder_is_a_stride_four_pyramid():
    pytest.importorskip("torchvision")
    enc = TorchvisionResNet("resnet18")
    assert enc.channels == [64, 128, 256, 512]  # basic-block widths
    feats = enc(torch.rand(1, 3, 128, 128))
    assert [f.shape[-2:] for f in feats] == [(32, 32), (16, 16), (8, 8), (4, 4)]


def test_bottleneck_encoders_report_the_four_times_wider_channels():
    pytest.importorskip("torchvision")
    assert TorchvisionResNet("resnet50").channels == [256, 512, 1024, 2048]


def test_resnet_backbone_merges_the_pyramid_into_stride_four():
    pytest.importorskip("torchvision")
    bb = build_backbone({"name": "resnet18", "out_channels": 16})
    assert (bb.stride, bb.out_channels, bb.deep_channels) == (4, 16, 512)
    fused, deep = bb(torch.rand(2, 3, 64, 64))
    assert fused.shape == (2, 16, 16, 16) and deep.shape == (2, 512, 2, 2)


def test_keypoints_to_box_pads_the_extent_of_the_valid_points_only():
    kpts = np.array([[0.0, 0.0], [10.0, 0.0], [900.0, 900.0]])
    box = keypoints_to_box(kpts, np.array([True, True, False]))
    np.testing.assert_allclose(box, [-1.0, -0.4, 11.0, 0.4])  # 10% of the extent, 4px floor per axis


def test_keypoints_to_box_falls_back_to_finiteness_and_gives_up_without_points():
    kpts = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.nan]])
    # No validity mask: only the fully finite point counts, and a lone point still
    # gets the 4px minimum extent rather than a degenerate box.
    np.testing.assert_allclose(keypoints_to_box(kpts), [0.6, 1.6, 1.4, 2.4])
    assert keypoints_to_box(np.full((4, 2), np.nan)) is None
    assert keypoints_to_box(kpts, np.zeros(3, dtype=bool)) is None


def test_skeleton_mask_marks_exactly_the_named_points():
    m = CANONICAL.mask(["nose", "left_ear"])
    assert m.dtype == bool and m.shape == (CANONICAL.num_keypoints,)
    assert np.nonzero(m)[0].tolist() == [CANONICAL.index("nose"), CANONICAL.index("left_ear")]
    assert CANONICAL.mask(CANONICAL.names).all() and not CANONICAL.mask([]).any()


def test_a_species_profile_declares_which_canonical_points_exist(monkeypatch):
    assert SPECIES["mouse"].mask().all() and SPECIES["mouse"].body_length_cm > 0
    monkeypatch.setitem(SPECIES, "zebra_test", None)
    profile = SpeciesProfile("zebra_test", 180.0, ("nose", "tail_base", "left_front_paw"))
    register_species(profile)
    assert SPECIES["zebra_test"] is profile
    m = SPECIES["zebra_test"].mask()
    assert m.sum() == 3 and m[CANONICAL.indices(["nose", "tail_base", "left_front_paw"])].all()
    assert not m[CANONICAL.index("left_eye")]  # anatomy this profile does not declare


def test_a_registered_mapping_is_usable_and_unlisted_source_names_stay_unlabeled(monkeypatch):
    monkeypatch.setitem(DATASET_MAPPINGS, "tiny_dlc_test", None)
    register_mapping("tiny_dlc_test", {"snout": "nose", "ear_l": "left_ear"})
    assert DATASET_MAPPINGS["tiny_dlc_test"]["ear_l"] == "left_ear"
    raw = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    canon, labeled = remap_keypoints(["snout", "implant", "ear_l", "tailtip"], raw, "tiny_dlc_test")
    assert canon.shape == (CANONICAL.num_keypoints, 2)
    np.testing.assert_allclose(canon[CANONICAL.index("nose")], [3.0, 4.0])
    np.testing.assert_allclose(canon[CANONICAL.index("left_ear")], [7.0, 8.0])
    # Source columns the mapping does not list are dropped, and canonical points the
    # dataset never labels stay NaN instead of becoming a spurious point at the origin.
    assert labeled.sum() == 2 and labeled[CANONICAL.indices(["nose", "left_ear"])].all()
    assert np.isnan(canon[np.nonzero(~labeled)[0]]).all()


def test_the_pose_crop_affine_is_exactly_invertible_and_centres_the_box():
    center, side = box_xyxy_to_cs(np.array([10, 20, 50, 40]))
    m = crop_affine(center, side, 64, rotation=0.7)
    pts = np.random.rand(10, 2) * 100
    back = apply_affine(invert_affine(m), apply_affine(m, pts))
    np.testing.assert_allclose(back, pts, atol=1e-9)
    np.testing.assert_allclose(apply_affine(m, center[None])[0], [31.5, 31.5], atol=1e-9)


def test_box_iou_is_exact_on_known_overlaps():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 1 / 3, 0.0])
