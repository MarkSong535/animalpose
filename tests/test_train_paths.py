"""The data paths of the four training loops, the datasets, the temporal real-track
batches, the export fallbacks and device selection.

Every case here is a branch the default synthetic run never takes: an unknown data
source name, a real COCO source and one whose files are missing, a weighted mixture that
inherits the parent's keys, loss logging, the sparse-skeleton keypoint subset and the
synthetic/real batch mixer, self-supervised windows from real DeepLabCut tracks
(including a track shorter than the clip), COCO image counting, the zoom-out and
degradation branches of the training datasets, a scene that holds no animals, the legacy
ONNX exporter and its error reporting, integer export outputs, and device selection when
no accelerator exists.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from animalpose import device as device_mod
from animalpose.data.behavior_sets import save_session
from animalpose.data.datasets import (
    Annotation,
    CocoKeypoints,
    DetectionDataset,
    MixedSource,
    PoseCropDataset,
    SyntheticScenes,
)
from animalpose.data.synthetic import SimulationPool
from animalpose.data.tracks import Session, load_sessions
from animalpose.export.export import export_onnx
from animalpose.labels import BEHAVIORS
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS
from animalpose.temporal.data import sample_windows, self_supervised_batch
from animalpose.train import loops
from animalpose.train.checkpoint import load_model
from animalpose.train.loops import train_behavior, train_detector, train_pose, train_temporal

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
DLC_BODYPARTS = ["nose", "leftear", "rightear", "headstage", "midbody", "tailbase"]
K = CANONICAL.num_keypoints


# --------------------------------------------------------------------------------- fixtures
def _dlc_recording(root: Path, animal: str, stamp: str, t: int, spans: dict[str, tuple[int, int]]) -> None:
    """A ``<animal>_<view>`` directory: DLC predictions plus a BORIS-style label CSV."""
    import pandas as pd

    cols = pd.MultiIndex.from_product([["DLC_model"], DLC_BODYPARTS, ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.uniform(5.0, 60.0, (t, len(cols))), columns=cols)
    frame.loc[:, (slice(None), slice(None), "likelihood")] = rng.uniform(0.5, 1.0, (t, len(DLC_BODYPARTS)))
    (root / animal / "DLC").mkdir(parents=True, exist_ok=True)
    frame.to_hdf(root / animal / "DLC" / f"{animal}-{stamp}DLC_model.h5", key="df_with_missing", format="table")
    labels = pd.DataFrame(0.0, index=range(t), columns=["Locomotion", "Grooming", "Rearing"])
    for col, (a, b) in spans.items():
        labels.loc[a:b, col] = 1
    (root / animal / "Behavior_Labels" / "CSV").mkdir(parents=True, exist_ok=True)
    labels.to_csv(root / animal / "Behavior_Labels" / "CSV" / f"{animal}-{stamp}_merged.csv")


@pytest.fixture(scope="module")
def real_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("dlc")
    _dlc_recording(root, "MOUSE1_top", "20220101T000000-010000", 100,
                      {"Locomotion": (0, 59), "Grooming": (60, 69), "Rearing": (70, 74)})
    _dlc_recording(root, "MOUSE2_side", "20220102T000000-010000", 100,
                      {"Locomotion": (0, 29), "Grooming": (30, 39), "Rearing": (40, 44)})
    return root


def _write_coco(tmp: Path) -> tuple[Path, Path]:
    """Three images; the third has no usable annotation and the first has one to ignore."""
    (tmp / "images").mkdir(exist_ok=True)
    for i in range(3):
        img = np.full((40, 60, 3), 40 + 30 * i, np.uint8)
        img[8:32, 18:48] = 200
        cv2.imwrite(str(tmp / "images" / f"im{i}.png"), img)
    anns = [
        {"id": 1, "image_id": 0, "category_id": 1, "num_keypoints": 3, "track_id": 7, "bbox": [18.0, 8.0, 30.0, 24.0],
         "keypoints": [25, 12, 2, 30, 14, 1, 35, 12, 0, 0, 0, 0]},
        {"id": 2, "image_id": 0, "category_id": 1, "num_keypoints": 0, "bbox": [1.0, 1.0, 5.0, 5.0],
         "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"id": 3, "image_id": 1, "category_id": 1, "num_keypoints": 1, "bbox": [10.0, 5.0, 20.0, 20.0],
         "keypoints": [15, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    ]
    payload = {"categories": [{"id": 1, "keypoints": ["snout", "earL", "earR", "marker"], "skeleton": []}],
               "images": [{"id": i, "file_name": f"im{i}.png", "width": 60, "height": 40} for i in range(3)],
               "annotations": anns}
    path = tmp / "ann.json"
    path.write_text(json.dumps(payload))
    return path, tmp / "images"


MAPPING = {"snout": "nose", "earL": "left_ear", "earR": "right_ear"}


class _OneBox:
    """A fixed frame and box, so the augmentation's box arithmetic can be checked exactly."""

    def __init__(self, boxes: np.ndarray | None = None):
        self.image = np.full((100, 200, 3), 200, np.uint8)
        self.boxes = (np.array([[10.0, 10.0, 60.0, 60.0]], np.float32) if boxes is None
                      else np.asarray(boxes, np.float32).reshape(-1, 4))

    def sample(self, index: int, variant: int = 0) -> Annotation:
        n = len(self.boxes)
        return Annotation(self.image, self.boxes.copy(), np.zeros((n, K, 2), np.float32),
                          np.full((n, K), 2, np.int64), np.arange(n) + index, np.zeros(n, np.int64))


class _EmptyFirst:
    """Scene 0 holds no animal; the pose dataset has to step past it."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def sample(self, index: int, variant: int = 0) -> Annotation:
        self.calls.append((index, variant))
        boxes = np.zeros((0, 4), np.float32) if index == 0 else np.array([[10.0, 10.0, 60.0, 60.0],
                                                                          [70.0, 20.0, 120.0, 90.0]], np.float32)
        ids = np.zeros(len(boxes), np.int64) if index == 0 else np.array([11, 12], np.int64)
        n = len(boxes)
        centers = ((boxes[:, :2] + boxes[:, 2:]) / 2)[:, None, :].repeat(K, 1) if n else np.zeros((0, K, 2))
        return Annotation(np.full((100, 200, 3), 180, np.uint8), boxes, centers.astype(np.float32),
                          np.full((n, K), 2, np.int64), ids, np.zeros(n, np.int64))


# --------------------------------------------------------------------------------- data sources
def test_an_unknown_data_source_is_named_instead_of_silently_training_on_synthetic():
    with pytest.raises(ValueError, match="unknown data source dlc_npz"):
        train_pose({"device": "cpu", "model": {"backbone": TINY, "embed_dim": 16},
                    "data": {"source": "dlc_npz", "crop_size": 64}, "train": {"steps": 1, "batch_size": 1}})


def test_a_real_source_whose_json_is_missing_fails_loudly(tmp_path):
    with pytest.raises((FileNotFoundError, json.JSONDecodeError)) as exc:
        train_detector({"device": "cpu", "model": {"backbone": TINY},
                        "data": {"source": "coco", "json": str(tmp_path / "gone.json"),
                                 "images": str(tmp_path), "input_size": [128, 160]},
                        "train": {"steps": 2, "batch_size": 2}})
    assert "gone.json" in str(exc.value)


def test_a_coco_source_trains_the_detector_and_logs_every_step(tmp_path, capsys):
    js, images = _write_coco(tmp_path)
    model, hist = train_detector({"device": "cpu", "model": {"backbone": TINY},
                                  "data": {"source": "coco", "json": str(js), "images": str(images),
                                           "mapping": MAPPING, "species": "rat", "input_size": [128, 160]},
                                  "train": {"steps": 4, "batch_size": 2, "log_every": 0}})
    assert not model.training
    assert [h["step"] for h in hist] == [1, 2, 3, 4]
    assert np.isfinite([h["total"] for h in hist]).all()
    assert capsys.readouterr().out == ""  # log_every=0 silences the progress lines entirely


def test_mixed_sources_inherit_the_parent_config_and_obey_their_weights(tmp_path):
    js, images = _write_coco(tmp_path)
    src = loops._source({"sources": [{"weight": 1.0}, {"weight": 0.0}], "source": "coco", "json": str(js),
                          "images": str(images), "mapping": MAPPING, "seed": 5}, seed_offset=2)
    assert isinstance(src, MixedSource) and len(src.sources) == 2
    assert all(isinstance(s, CocoKeypoints) for s in src.sources)  # json/images/mapping came from the parent
    np.testing.assert_allclose(src.p, [1.0, 0.0])
    assert src.seed == 7  # the configured seed plus the loop's per-stage offset
    assert {src.sample(i).image.shape for i in range(12)} == {(40, 60, 3)}  # only the weighted source is drawn

    other = loops._source({"sources": [{"weight": 0.0}, {"source": "synthetic", "n_sims": 2, "frames": 30,
                                                                    "image_size": [48, 64]}], "source": "coco",
                           "json": str(js), "images": str(images), "mapping": MAPPING})
    assert all(other.sample(i).image.shape != (40, 60, 3) for i in range(6))
    assert isinstance(other.sources[1], SyntheticScenes)


def test_logging_prints_only_on_the_requested_interval(tmp_path, capsys):
    train_pose({"device": "cpu", "model": {"backbone": TINY, "embed_dim": 16},
                "data": {"source": "synthetic", "n_sims": 2, "frames": 40, "image_size": [96, 96],
                         "crop_size": 48}, "train": {"steps": 4, "batch_size": 2, "log_every": 2}})
    out = capsys.readouterr().out
    assert "[pose] step 2" in out and "[pose] step 4" in out and "total=" in out
    assert "step 1 " not in out and "step 3 " not in out


# --------------------------------------------------------------------------------- batch wrappers
def _five_arrays(rng: np.random.Generator):
    """The temporal batch layout: x, conf, observed, target, target mask."""
    pts = np.ones((2, 5, K, 2), np.float32)
    conf = np.full((2, 5, K), 0.8, np.float32)
    mask = np.ones((2, 5, K), np.float32)
    target = np.full((2, 5, K, 2), 3.0, np.float32)
    target_mask = np.ones((2, 5, K), np.float32)
    return pts, conf, mask, target, target_mask


def test_the_keypoint_subset_blanks_only_the_points_a_mapping_lacks():
    keep = CANONICAL.mask(list(DATASET_MAPPINGS["homecage_dlc"].values()))
    sub = loops._KeypointSubset(_five_arrays, "homecage_dlc", p=1.0)
    out = sub(np.random.default_rng(0))
    assert keep.sum() == 6
    np.testing.assert_array_equal(out[0], np.ones_like(out[0]))  # the regressed coordinates stay put
    np.testing.assert_array_equal(out[3], np.full_like(out[3], 3.0))  # ... and so does the target
    for i in (1, 2, 4):
        expected = np.tile(keep, (2, 5, 1)) * (0.8 if i == 1 else 1.0)
        np.testing.assert_allclose(out[i], expected)  # the confidence/mask arrays follow the skeleton

    same = loops._KeypointSubset(_five_arrays, "homecage_dlc", p=0.0)
    assert all(np.array_equal(a, b) for a, b in zip(same(np.random.default_rng(1)), _five_arrays(None), strict=True))

    both = loops._KeypointSubset(_five_arrays, ["homecage_dlc", "calms21"], p=1.0)
    patterns = {tuple(np.nonzero(out[2][0, 0])[0]) for out in (both(np.random.default_rng(s)) for s in range(12))}
    assert len(patterns) == 2  # one skeleton per batch, so training sees both sparse regimes


def test_the_batch_mixer_honours_the_real_data_fraction():
    calls = {"s": 0, "r": 0}

    def synth(rng):
        calls["s"] += 1
        return "s"

    def real(rng):
        calls["r"] += 1
        return "r"

    rng = np.random.default_rng(0)
    mix = loops._Mix(synth, real, 0.5)
    assert (mix.synthetic, mix.real, mix.real_fraction) == (synth, real, 0.5)
    assert [loops._Mix(synth, real, 1.0)(rng) for _ in range(5)] == ["r"] * 5
    assert [loops._Mix(synth, real, 0.0)(rng) for _ in range(5)] == ["s"] * 5
    kinds = {mix(rng) for _ in range(80)}
    assert kinds == {"s", "r"} and calls["s"] + calls["r"] == 90


# --------------------------------------------------------------------------------- training branches
def test_temporal_trains_self_supervised_on_real_tracks(real_root, tmp_path, capsys):
    ckpt = tmp_path / "temporal.pt"
    _, hist = train_temporal({"device": "cpu", "output": str(ckpt), "model": {"dims": 2, "hidden": 16},
                              "data": {"window": 32, "pool_sims": 2, "pool_frames": 120,
                                       "real_root": str(real_root), "real_fraction": 1.0},
                              "train": {"steps": 8, "batch_size": 4, "log_every": 0}})
    assert len(hist) == 8 and np.isfinite([h["total"] for h in hist]).all()
    assert torch.load(ckpt, map_location="cpu", weights_only=False)["window"] == 32
    assert isinstance(load_model(ckpt), torch.nn.Module)


def test_behavior_training_uses_real_sessions_and_weights_the_rare_classes(real_root, tmp_path, capsys):
    sessions = load_sessions(real_root)
    assert [s.animal for s in sessions] == ["MOUSE1", "MOUSE2"]  # the tree the trainer will read
    sessions_dir = tmp_path / "video_sessions"
    coords = np.repeat(np.arange(80, dtype=np.float64)[:, None, None], K * 2, -1).reshape(80, K, 2)
    conf = np.full((80, K), 0.9)
    labels = np.zeros(80, np.int64)
    labels[70:] = -100  # unlabeled tail: ignored by the loss and by the class weights
    labels[40:70] = 3  # grooming
    save_session(sessions_dir / "cage7" / "m1.npz", Session("m1", "side", coords, conf, labels, 30.0))
    ckpt = tmp_path / "temporal.pt"
    train_temporal({"device": "cpu", "output": str(ckpt), "model": {"dims": 2, "hidden": 16},
                    "data": {"window": 32, "pool_sims": 2, "pool_frames": 120},
                    "train": {"steps": 2, "batch_size": 4, "log_every": 0}})
    out = tmp_path / "behavior.pt"
    _, hist = train_behavior({"device": "cpu", "output": str(out), "model": {"dims": 2, "hidden": 16},
                              "data": {"window": 32, "pool_sims": 2, "pool_frames": 120, "real_root": str(real_root),
                                       "session_dirs": [str(sessions_dir)], "real_temporal": str(ckpt),
                                       "real_mapping": "homecage_dlc", "real_fraction": 0.5},
                              "loss": {"class_balance": True},
                              "train": {"steps": 8, "batch_size": 4, "log_every": 0}})
    assert len(hist) == 8 and np.isfinite([h["total"] for h in hist]).all()
    printed = capsys.readouterr().out
    weights = {m.group(1): float(m.group(2)) for m in
               re.finditer(r"(\w+)['\"]:\s*(?:np\.float64\()?([0-9.]+)", printed.split("class weights")[1])}
    assert set(weights) == set(BEHAVIORS)
    # Inverse-square-root class frequency over the real labels (the DLC recordings plus the
    # 40 resampled frames of the video session), renormalised to mean 1; a class the real data
    # never shows keeps the unweighted value instead of being boosted to infinity.
    counts = {"resting": 20, "walking": 90, "rearing": 10, "grooming": 35, "eating": 0}
    mean = np.mean([c for c in counts.values() if c])
    raw = {k: 1.0 / np.sqrt(c / mean) if c else 1.0 for k, c in counts.items()}
    norm = np.mean(list(raw.values()))
    assert all(weights[k] == pytest.approx(raw[k] / norm, abs=0.015) for k in counts)
    assert weights["walking"] < weights["grooming"] < weights["resting"] < weights["rearing"]
    assert torch.load(out, map_location="cpu", weights_only=False)["window"] == 32


# --------------------------------------------------------------------------------- temporal data
def test_windows_come_from_fresh_simulations_when_no_pool_is_given():
    rng = np.random.default_rng(4)
    windows = sample_windows(3, 32, rng, ("mouse",), pool=None)
    assert len(windows) == 3
    for sim, animal, start in windows:
        assert sim.species == "mouse" and sim.n_frames >= 32 + 60
        assert 30 <= start and start + 32 <= sim.n_frames and 0 <= animal < sim.n_animals

    pool = SimulationPool(2, 120, ("mouse",), seed=1)
    from_pool = sample_windows(2, 32, np.random.default_rng(0), ("mouse",), pool=pool)
    assert all(any(sim is s for s in pool.sims) for sim, _, _ in from_pool)  # the pool branch, not fresh sims


def test_self_supervised_windows_skip_tracks_shorter_than_the_clip():
    window, t = 20, 100
    ramp = np.arange(t, dtype=np.float32)[:, None, None] * np.ones((1, K, 2), np.float32)
    conf = np.full((t, K), 0.9, np.float32)
    conf[10:14] = 0.1  # four frames the tracker does not believe
    short = np.full((window - 1, K, 2), 999.0, np.float32)
    tracks = [(ramp, conf), (short, np.full((window - 1, K), 0.9, np.float32))]
    x, c, obs, target, mask = self_supervised_batch(tracks, 6, window, np.random.default_rng(0))
    assert x.shape == (6, window, K, 2) and c.shape == (6, window, K) and obs.shape == mask.shape
    assert np.isfinite(x).all() and target.max() < 900  # the short track never filled a clip
    starts = target[:, 0, 0, 0].astype(int)
    assert (starts >= 0).all() and (starts + window <= t).all()
    for i, s in enumerate(starts):
        np.testing.assert_array_equal(mask[i] > 0, conf[s:s + window] > 0.3)  # confident frames are the targets
        np.testing.assert_allclose(target[i][:, 0, 0], np.arange(s, s + window, dtype=np.float32))


# --------------------------------------------------------------------------------- dataset branches
def test_coco_length_counts_images_with_usable_annotations(tmp_path):
    js, images = _write_coco(tmp_path)
    ds = CocoKeypoints(js, images, MAPPING)
    assert len(ds) == 2 and sorted(ds.by_image) == [0, 1]  # the num_keypoints=0 ann and image 2 drop out
    assert len(ds.sample(0).boxes) == 1 and len(ds.sample(1).boxes) == 1
    np.testing.assert_allclose(ds.sample(3).boxes[0], ds.sample(1).boxes[0])  # indices wrap around
    ann = ds.sample(0)
    nose = CANONICAL.index("nose")
    assert ann.state[0, nose] == 2 and ann.state[0, CANONICAL.index("left_ear")] == 1
    assert ann.state[0, CANONICAL.index("right_ear")] == 0  # v=0 carries no label
    assert ann.ids.tolist() == [7]


def test_coco_sample_raises_when_the_image_file_is_missing(tmp_path):
    js, images = _write_coco(tmp_path)
    (images / "im1.png").unlink()
    with pytest.raises(FileNotFoundError, match="im1.png"):
        CocoKeypoints(js, images, MAPPING).sample(1)


def test_zoom_out_shrinks_the_frame_and_keeps_the_boxes_inside_it():
    src = _OneBox()
    s = min(128 / 100, 160 / 200)  # the letterbox scale is unchanged: zoom_out paints on the same canvas
    base = 50.0 * 50.0 * s * s
    areas = []
    for i in range(6):
        img, boxes, _species, _flag = DetectionDataset(src, 6, (128, 160), zoom_out=1.0)[i]
        b = boxes.numpy()
        assert img.shape == (3, 128, 160)
        assert (b >= -1e-3).all() and b[:, 2].max() <= 160 + 1e-3 and b[:, 3].max() <= 128 + 1e-3
        areas.append(float(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sum()))
    ratio = np.array(areas) / base
    assert (ratio < 1 - 1e-4).all() and (ratio > 0.35**2 - 1e-4).all()  # f in [0.35, 1), applied twice
    control = DetectionDataset(src, 3, (128, 160))[0][1].numpy()
    assert abs(float(((control[:, 2] - control[:, 0]) * (control[:, 3] - control[:, 1])).sum()) - base) < 1e-2


def test_detection_degradation_is_flagged_only_when_there_is_an_animal_to_blur():
    src = _OneBox()
    ds = DetectionDataset(src, 4, (128, 160), degrade=1.0)
    imgs = [ds[i] for i in range(4)]
    assert all(bool(flag) for _, _, _, flag in imgs)
    assert all(im.dtype == torch.float32 and im.max() <= 1.0 for im, _, _, _ in imgs)
    empty = DetectionDataset(_OneBox(np.zeros((0, 4))), 2, (128, 160), degrade=1.0)
    assert not bool(empty[0][3]) and empty[0][1].shape == (0, 4)  # nothing to size, nothing to degrade


def test_pose_crop_steps_past_a_scene_without_animals():
    src = _EmptyFirst()
    crop, kp, state, ids, degraded = PoseCropDataset(src, 4, 64, augment=False)[0]
    assert crop.shape == (3, 64, 64) and kp.shape == (K, 2) and not degraded
    assert src.calls[0] == (0, 0) and src.calls[1] == (1_000_003, 0)  # the empty scene is jumped, not retried
    assert int(ids) in (11, 12)
    assert (kp >= 0).all() and (kp <= 63).all() and (state == 2).all()  # a real crop of a real animal


def test_pose_crop_can_degrade_at_the_camera_scale():
    crop, kp, _state, _ids, degraded = PoseCropDataset(SyntheticScenes(2, 40, seed=0), 6, 64, degrade=1.0)[0]
    assert bool(degraded) and crop.shape == (3, 64, 64) and torch.isfinite(kp).all()


# --------------------------------------------------------------------------------- export / device
class _Top3(nn.Module):
    def forward(self, x):
        return x.argmax(dim=-1)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def _export(module, path, exporter="auto", atol=1e-3, dynamo_error=None, both_fail=False):
    real_export = torch.onnx.export

    def fake(model, args, where, **kw):
        if both_fail or (dynamo_error is not None and kw.get("dynamo")):
            raise dynamo_error or RuntimeError("no exporter works")
        return real_export(model, args, where, **kw)

    torch.onnx.export = fake
    try:
        return export_onnx(module, (torch.rand(2, 4),), path, ["x"], ["y"], {"x": {0: "batch"}},
                           exporter=exporter, atol=atol)
    finally:
        torch.onnx.export = real_export


def test_integer_export_outputs_are_compared_exactly(tmp_path):
    pytest.importorskip("onnxruntime")
    assert not _Top3()(torch.rand(3, 5)).dtype.is_floating_point  # indices, not values: no float tolerance
    report = export_onnx(_Top3(), (torch.rand(2, 4),), tmp_path / "top.onnx", ["x"], ["idx"], {"x": {0: "batch"}})
    assert report["ok"] and report["max_abs_err"] == 0.0
    assert (tmp_path / "top.onnx").stat().st_size > 0


def test_the_legacy_exporter_is_still_available(tmp_path):
    pytest.importorskip("onnxruntime")
    report = _export(_Net(), tmp_path / "m.onnx", exporter="legacy")
    assert report["exporter"] == "legacy" and report["ok"] and report["errors"] == []
    assert report["max_abs_err"] < 1e-3


def test_auto_exporter_falls_back_to_legacy_when_dynamo_raises(tmp_path):
    pytest.importorskip("onnxruntime")
    report = _export(_Net(), tmp_path / "m.onnx", dynamo_error=RuntimeError("dynamo exploded"))
    assert report["exporter"] == "legacy" and report["ok"]
    assert len(report["errors"]) == 1 and report["errors"][0].startswith("dynamo: RuntimeError: dynamo exploded")


def test_when_both_exporters_fail_the_report_says_so_and_keeps_both_errors(tmp_path):
    pytest.importorskip("onnxruntime")
    report = _export(_Net(), tmp_path / "m.onnx", both_fail=True)
    assert report["exporter"] is None and report["ok"] is False and np.isnan(report["max_abs_err"])
    assert [e.split(":")[0] for e in report["errors"]] == ["dynamo", "legacy"]


def test_a_parity_failure_is_recorded_for_every_exporter_tried(tmp_path):
    pytest.importorskip("onnxruntime")
    report = _export(_Net(), tmp_path / "m.onnx", atol=-1.0)  # no error can ever be under this tolerance
    assert report["ok"] is False and report["exporter"] == "legacy"
    assert len(report["errors"]) == 2 and all("parity failed" in e for e in report["errors"])


def test_select_device_reaches_cpu_only_when_nothing_is_available(monkeypatch):
    monkeypatch.setattr(device_mod, "_available", lambda kind: False)
    assert device_mod.select_device("cuda").type == "cpu"
    assert device_mod.select_device("auto").type == "cpu"
    monkeypatch.setattr(device_mod, "_available", lambda kind: kind == "mps")
    assert device_mod.select_device("cuda:1").type == "mps"  # a GPU request still finds an accelerator
