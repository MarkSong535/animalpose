"""The held-out synthetic benchmark run end to end: tiny models, tiny scenes, every section.

Checks the report a benchmark run produces: every section present with the views and thresholds the
module promises, every reported number finite and inside its range, the thresholds and alphas ordered
against each other, frames, views and annotation-less images with nothing to score dropped rather than
scored, the JSON and markdown writers round-tripping, and the evaluation seeds disjoint from the seeds
the trainers used. Not checked: the accuracy itself - these networks train for seconds, so AP or PCK
near zero is expected and no accuracy threshold is asserted.
"""

import json
from dataclasses import replace

import numpy as np
import pytest

from animalpose.cli.evaluate.benchmark import to_markdown
from animalpose.data.datasets import SyntheticScenes
from animalpose.data.synthetic import SimulationPool
from animalpose.eval import benchmark as B
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.skeleton import CANONICAL, SPECIES
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.loops import train_behavior, train_detector, train_pose, train_temporal

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
DATA = {"n_sims": 4, "frames": 40, "image_size": [160, 128]}
RENDER = (160, 128)  # (W, H) for the simulator's cameras
DET = (128, 160)  # (H, W) for the detector
TRAIN_SEED = 0  # what train_* defaults to; its simulation pool sits at TRAIN_SEED + 100
TEMPORAL_KEYS = {"observed_raw_rmse_{u}", "observed_smoothed_rmse_{u}", "gap_linear_interp_rmse_{u}",
                 "gap_lstm_rmse_{u}", "note"}


def _numbers(obj, path="root"):
    """Every numeric leaf of a nested report, with its path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


class _LabeledImages:
    """The only thing :func:`eval_coco` needs of a real dataset: a sized source of canonical annotations.

    ``empty_at`` indexes return an image with no animals, the way a COCO frame without instances does.
    """

    def __init__(self, n_sims: int, seed: int = 0, empty_at=()):
        self.src = SyntheticScenes(n_sims, 40, RENDER, ("mouse", "rat"), 3, seed=seed)
        self.empty_at = set(empty_at)

    def __len__(self) -> int:
        return 4 * len(self.src.sims)

    def sample(self, index: int):
        ann = self.src.sample(int(index))
        return replace(ann, boxes=ann.boxes[:0]) if int(index) in self.empty_at else ann


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = tmp_path_factory.mktemp("benchmark")
    train = {"steps": 150, "batch_size": 8, "log_every": 0, "workers": 0}
    det, _ = train_detector({"device": "cpu", "output": str(out / "detector.pt"), "model": {"backbone": TINY},
                             "data": {**DATA, "input_size": list(DET)}, "train": train})
    pose, _ = train_pose({"device": "cpu", "output": str(out / "pose.pt"), "model": {"backbone": TINY, "embed_dim": 16},
                          "data": {**DATA, "crop_size": 64}, "train": train})
    quick = {"steps": 30, "batch_size": 4, "log_every": 0, "workers": 0}
    small = {"window": 32, "pool_sims": 2, "pool_frames": 120}
    trained = {"det": det, "pose": pose}
    for kind, dims in (("t2", 2), ("t3", 3), ("b2", 2), ("b3", 3)):
        loop = train_temporal if kind.startswith("t") else train_behavior
        trained[kind], _ = loop({"device": "cpu", "model": {"dims": dims, "hidden": 16}, "data": small,
                                 "train": quick})
    return trained


@pytest.fixture(scope="module")
def pipe(models):
    cfg = PipelineConfig(det_size=DET, crop_size=64, temporal_window=32, tracker=TrackerConfig(max_animals=3))
    return PosePipeline(models["det"], models["pose"], models["t2"], models["b2"], cfg, device="cpu")


@pytest.fixture(scope="module")
def report(pipe, models):
    """One full benchmark pass, with the sample counts that keep it to seconds."""
    r = {"detection": B.eval_detection(pipe, 12, RENDER), "pose": B.eval_pose(pipe, 12, RENDER)}
    r["temporal2d"] = B.eval_temporal(models["t2"], 2, batches=2, window=32)
    r["temporal3d"] = B.eval_temporal(models["t3"], 3, batches=2, window=32)
    r["behavior2d"] = B.eval_behavior(models["b2"], 2, batches=2, window=32)
    r["behavior3d"] = B.eval_behavior(models["b3"], 3, batches=2, window=32)
    r["tracking_model_embedding"] = B.eval_tracking(pipe, videos=1, frames=40, n_animals=2)
    pipe.cfg.embedder = "color"
    r["tracking_color_histogram"] = B.eval_tracking(pipe, videos=1, frames=40, n_animals=2)
    pipe.cfg.embedder = "model"
    r["multiview"] = B.eval_multiview(pipe, models["t3"], frames=40, n_animals=2)
    r["throughput"] = [B.throughput(pipe, n=8, size=RENDER)]
    return r


def test_every_number_the_benchmark_reports_is_finite(report):
    leaves = list(_numbers(report))
    assert len(leaves) > 40 and np.isfinite([v for _, v in leaves]).all(), [p for p, v in leaves if not np.isfinite(v)]
    assert set(report) >= {"detection", "pose", "temporal2d", "temporal3d", "behavior2d", "behavior3d", "multiview",
                           "throughput"}


def test_detection_covers_every_preset_view_and_holds_the_iou_threshold_order(pipe, report):
    assert set(report["detection"]) == {"random_views", *B.VIEWS} and set(B.VIEWS) == {"top", "angled", "side"}
    for view, m in report["detection"].items():
        assert set(m) == {"ap50", "ap75"}, view
        assert 0.0 <= m["ap75"] <= m["ap50"] <= 1.0, view
        assert all(v is None or np.isfinite(v) for v in m.values())


def test_pose_scores_each_species_and_their_pooled_together(report):
    assert set(report["pose"]) == {"random_views", *B.VIEWS}
    for view, per_species in report["pose"].items():
        assert set(per_species) <= {"mouse", "rat", "all"} and "all" in per_species, view
        assert per_species["all"]["n_instances"] == sum(
            per_species[s]["n_instances"] for s in ("mouse", "rat") if s in per_species), view
        for sp, m in per_species.items():
            assert m["n_instances"] > 0 and m["rmse_px"] > 0.0, (view, sp)
            assert 0.0 <= m["pck@0.05"] <= m["pck@0.1"] <= 1.0, (view, sp)
            assert 0.0 <= m["conf_visible"] <= 1.0, (view, sp)
            if m["conf_hidden"] is not None:
                assert 0.0 <= m["conf_hidden"] <= 1.0, (view, sp)


def test_pose_skips_frames_that_hold_no_annotatable_animal(pipe, monkeypatch):
    seen = []
    real = pipe.estimate

    def estimate(image, boxes):
        seen.append(len(boxes))
        return real(image, boxes)

    monkeypatch.setattr(pipe, "estimate", estimate)
    out = B.eval_pose(pipe, n=6, size=(32, 24))  # a field of view that pushes most animals out of frame
    assert set(out) == {"random_views", *B.VIEWS}
    assert 0 < len(seen) < 15 and all(n > 0 for n in seen)  # 15 frames rendered, the empty ones cost no pass
    assert out["angled"] == {}  # every frame of this view lost every animal
    assert set(out["side"]) == {"rat", "all"} and set(out["random_views"]) == {"mouse", "all"}
    for view, m in out.items():
        assert m.get("all", {}).get("n_instances", 0) == sum(
            v["n_instances"] for k, v in m.items() if k != "all"), view


@pytest.mark.parametrize(("key", "unit"), (("temporal2d", "px"), ("temporal3d", "cm")))
def test_temporal_separates_the_observed_from_the_filled_gaps(report, key, unit):
    m = report[key]
    assert set(m) == {k.format(u=unit) for k in TEMPORAL_KEYS}
    errors = {k: v for k, v in m.items() if k != "note"}
    assert all(v > 0.0 and np.isfinite(v) for v in errors.values()), errors
    assert "linear interpolation" in m["note"]


def test_behavior_scores_every_class_of_the_classifier(report, models):
    for key, model in (("behavior2d", models["b2"]), ("behavior3d", models["b3"])):
        m = report[key]
        assert list(m["classes"]) == list(model.classes), key
        assert set(m) == {"macro_f1", "accuracy", "per_class_f1", "confusion", "classes"}
        assert len(m["per_class_f1"]) == len(m["classes"])
        assert all(0.0 <= f <= 1.0 for f in m["per_class_f1"]), key
        assert 0.0 <= m["macro_f1"] <= 1.0 and 0.0 <= m["accuracy"] <= 1.0, key
        assert sum(sum(r) for r in m["confusion"]) == 16 * 2 * 32, key  # batches x window frames
        assert m["accuracy"] == pytest.approx(sum(m["confusion"][i][i] for i in range(len(m["classes"]))) / 1024)


def test_tracking_is_scored_per_view_against_the_simulated_identities(pipe, report):
    for key in ("tracking_model_embedding", "tracking_color_histogram"):
        assert set(report[key]) == {"top", "angled"}, key
        for view, m in report[key].items():
            assert set(m) == {"mota", "idf1", "id_switches", "n_pred_ids", "n_gt_ids"}, (key, view)
            assert m["n_gt_ids"] == 2 and m["id_switches"] >= 0.0 and m["n_pred_ids"] >= 0.0, (key, view)
            assert 0.0 <= m["idf1"] <= 1.0 and m["mota"] <= 1.0, (key, view)


def test_multiview_triangulates_the_two_rats_from_three_views(report, pipe):
    m = report["multiview"]
    assert m["n_animals"] == 2 and m["n_global_ids"] >= 1
    assert 0.0 < m["triangulated_fraction"] <= 1.0
    assert m["mpjpe_cm"] > 0.0 and m["median_error_cm"] <= m["mpjpe_cm"]
    assert SPECIES["rat"].body_length_cm * 0.8 < m["body_length_cm"] < SPECIES["rat"].body_length_cm * 1.2
    assert m["mpjpe_smoothed_all_frames_cm"] > 0.0  # the 3D temporal model was part of the run
    assert pipe.cfg.embedder == "model"


def test_throughput_times_the_device_it_ran_on(report, pipe):
    t = report["throughput"][0]
    assert t["device"] == str(pipe.device) == "cpu"
    assert t["detector_ms_per_frame"] > 0.0 and t["pose_ms_per_animal"] > 0.0 and t["detect_pose_track_fps"] > 0.0


def test_the_report_survives_json_and_renders_every_section(report, tmp_path):
    md = to_markdown(report)
    (tmp_path / "benchmark.json").write_text(json.dumps(report, indent=2))
    assert json.loads((tmp_path / "benchmark.json").read_text()) == report
    for heading in ("## Detection", "## Pose", "## Temporal LSTM, 2D", "## Temporal LSTM, 3D",
                    "## Behavior, 2D random views", "## Behavior, 3D", "## Tracking model embedding",
                    "## Tracking color histogram", "## Multi-camera", "## Throughput"):
        assert heading in md, heading
    assert "nan" not in md.lower() and "inf" not in md.lower()
    assert md.count("|") > 100 and md.startswith("# Synthetic benchmark")


def test_the_benchmark_scores_simulations_the_trainers_never_saw():
    assert B.SEED > TRAIN_SEED + 100  # train_temporal/train_behavior build their pool at seed + 100
    trained = SimulationPool(2, 120, seed=TRAIN_SEED + 100)
    held = SimulationPool(2, 120, seed=B.SEED + 3)  # the pool eval_temporal draws from
    assert all(not np.array_equal(a.kpts3d, b.kpts3d) for a in trained.sims for b in held.sims)

    def frames(seed, view):
        return [f.image for _, f in B._frames(2, seed, RENDER, view)]

    a, b = frames(B.SEED + 1, "top"), frames(B.SEED + 1, "top")
    assert [x.tobytes() for x in a] == [x.tobytes() for x in b]
    assert frames(B.SEED + 1, "side")[0].tobytes() != a[0].tobytes()


def test_coco_style_scoring_runs_on_a_sized_annotation_source(pipe):
    out = B.eval_coco(pipe, _LabeledImages(4), max_images=10)
    assert out["n_instances"] > 0 and out["rmse_px"] > 0.0 and out["median_box_px"] > 0.0
    assert 0.0 <= out["pck@0.05"] <= out["pck@0.1"] <= 1.0
    assert 0.0 <= out["median_err_px"] and 0.0 <= out["conf_auroc_visible_vs_occluded"] <= 1.0
    per_kp = out["per_keypoint_pck@0.1"]
    assert per_kp and set(per_kp) <= set(CANONICAL.names) and all(0.0 <= v <= 1.0 for v in per_kp.values())
    assert 0.0 <= out["det_ap75"] <= out["det_ap50"] <= 1.0
    src = _LabeledImages(4, seed=1, empty_at=range(0, 16, 2))  # four images per simulation, half are empty
    idx = np.linspace(0, len(src) - 1, 6).astype(int)
    no_detect = B.eval_coco(pipe, src, max_images=6, detect=False)
    assert not {"det_ap50", "det_ap75"} & set(no_detect)
    kept = [int(i) for i in idx if i not in src.empty_at]  # images with no instance are skipped outright
    assert no_detect["n_instances"] == sum(len(src.sample(i).boxes) for i in kept) > 0
    assert len(kept) < len(idx)
