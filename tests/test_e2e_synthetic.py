"""Small end-to-end run on CPU: train every network for a few steps on
synthetic data, then run the single-view and multi-camera pipelines,
export, and write outputs. Checks plumbing and output contracts, not accuracy
(see animalpose eval benchmark for accuracy)."""

import numpy as np
import pytest

from animalpose.data.synthetic import make_cameras, render_video, simulate
from animalpose.pipeline import MultiViewPipeline, PipelineConfig, PosePipeline
from animalpose.skeleton import CANONICAL
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.checkpoint import load_model
from animalpose.train.loops import train_behavior, train_detector, train_pose, train_temporal

TINY = {"name": "tiny", "width": 8, "out_channels": 16}


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    train = {"steps": 12, "batch_size": 4, "log_every": 0, "workers": 0}
    data = {"n_sims": 3, "frames": 40, "image_size": [160, 128]}
    det, _ = train_detector({"device": "cpu", "output": str(out / "det.pt"), "model": {"backbone": TINY},
                             "data": {**data, "input_size": [128, 160]}, "train": train})
    pose, _ = train_pose({"device": "cpu", "output": str(out / "pose.pt"), "model": {"backbone": TINY, "embed_dim": 16},
                          "data": {**data, "crop_size": 64}, "train": train})
    t2, _ = train_temporal({"device": "cpu", "model": {"dims": 2, "hidden": 16},
                            "data": {"window": 32, "pool_sims": 2, "pool_frames": 120}, "train": train})
    t3, _ = train_temporal({"device": "cpu", "model": {"dims": 3, "hidden": 16},
                            "data": {"window": 32, "pool_sims": 2, "pool_frames": 120}, "train": train})
    b2, hist = train_behavior({"device": "cpu", "model": {"dims": 2, "hidden": 16},
                               "data": {"window": 32, "pool_sims": 2, "pool_frames": 120}, "train": train})
    assert np.isfinite([h["total"] for h in hist]).all()
    # Checkpoints round-trip.
    assert load_model(out / "det.pt").stride == 4 and load_model(out / "pose.pt").num_keypoints == 25
    return {"det": det, "pose": pose, "t2": t2, "t3": t3, "b2": b2}


def _pipe(models, **kw):
    cfg = PipelineConfig(det_size=(128, 160), crop_size=64, temporal_window=32,
                         tracker=TrackerConfig(max_animals=2), **kw)
    return PosePipeline(models["det"], models["pose"], models["t2"], models["b2"], cfg, device="cpu")


def test_single_view_pipeline(models, tmp_path):
    sim = simulate(2, 40, "mouse", seed=99, coats=["black", "white"])
    frames = render_video(sim, make_cameras(sim.arena_cm, ("angled",), size=(160, 128))[0])
    gt = lambda t, img: (frames[t].boxes[frames[t].present], np.full(frames[t].present.sum(), 0.9))  # noqa: E731
    pipe = _pipe(models, embedder="color")
    result = pipe.run([f.image for f in frames], detections_fn=gt)  # GT boxes: untrained detector is unreliable
    assert 1 <= len(result.tracks) <= 2
    for tr in result.tracks.values():
        t = len(tr.frames)
        assert tr.keypoints.shape == (t, CANONICAL.num_keypoints, 2)
        if t >= 5:
            assert tr.smoothed.shape == tr.keypoints.shape and np.isfinite(tr.smoothed).all()
            assert set(np.unique(tr.imputed)) <= {0.0, 1.0}
            assert tr.behavior.shape == (t,) and np.allclose(tr.behavior_probs.sum(-1), 1, atol=1e-4)
    result.to_npz(tmp_path / "tracks.npz")
    result.to_csv(tmp_path / "tracks.csv")
    assert (tmp_path / "tracks.csv").read_text().count("\n") > 25
    # The real detector path also runs end to end (accuracy not asserted).
    pipe.run([f.image for f in frames[:5]])


def test_multiview_pipeline(models):
    sim = simulate(2, 30, "rat", seed=98, coats=["black", "white"])
    cams = make_cameras(sim.arena_cm, ("top", "side", "angled"), size=(160, 128))
    views = [render_video(sim, c, seed=i) for i, c in enumerate(cams)]
    fns = [lambda t, img, fr=fr: (fr[t].boxes[fr[t].present], np.full(fr[t].present.sum(), 0.9)) for fr in views]
    mv = MultiViewPipeline(_pipe(models, embedder="color"), cams, temporal3d=models["t3"], fusion_max_cost=1e9,
                           reproj_thresh=1e9)
    res = mv.run([[f.image for f in fr] for fr in views], fns)
    assert len(res["per_view"]) == 3 and res["animals"]
    for a in res["animals"].values():
        assert a["keypoints3d"].shape == (30, 25, 3)
        if "smoothed3d" in a:
            assert np.isfinite(a["smoothed3d"]).all()
