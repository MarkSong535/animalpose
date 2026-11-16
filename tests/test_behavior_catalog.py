import numpy as np

from animalpose.behavior.catalog import NAMES, body_kinematics, describe, full_probabilities, top_k
from animalpose.data.behavior_sets import remap_labels
from animalpose.data.tracks import IGNORE
from animalpose.skeleton import CANONICAL


def _mouse(t, x0, vx, y=0.0):
    """Straight mouse of length 10 px facing +x, moving at vx px/frame."""
    kp = np.zeros((t, CANONICAL.num_keypoints, 2))
    for i in range(t):
        cx = x0 + vx * i
        for name, dx in [("nose", 5), ("head_center", 3), ("neck", 2), ("spine_upper", 1), ("spine_mid", 0),
                         ("spine_lower", -2), ("tail_base", -5)]:
            kp[i, CANONICAL.index(name)] = (cx + dx, y)
    return kp, np.ones((t, CANONICAL.num_keypoints))


def test_full_probabilities_sum_to_one_and_split_running():
    t = 60
    kp, cf = _mouse(t, 0, 1.0)  # 1 px/frame at 30 fps = 3 body lengths/s: running
    kin = body_kinematics(kp, cf, 30.0)
    probs = np.tile([[0.1, 0.8, 0.1]], (t, 1))
    full, modelled = full_probabilities(["resting", "walking", "grooming"], probs, kin)
    assert np.allclose(full.sum(1), 1.0)
    assert full[30, NAMES.index("run")] > full[30, NAMES.index("walk")]
    assert modelled[NAMES.index("run")] and not modelled[NAMES.index("eat")]
    assert full[:, NAMES.index("eat")].max() == 0
    idx, p = top_k(full, 3)
    assert NAMES[idx[30, 0]] == "run" and p[30, 0] >= p[30, 1] >= p[30, 2]
    info = {d["name"]: d for d in describe(modelled)}
    assert info["eat"]["source"].startswith("not modelled") and info["follow"]["source"] == "pair rule"


def test_social_rules_respond_to_geometry():
    t = 40
    a_kp, cf = _mouse(t, 0, 0.0)
    near, _ = _mouse(t, 10, 0.0)  # partner 10 px ahead: a's nose (x=5) is at b's tail base (x=5)
    nose_to_nose, _ = _mouse(t, 0, 0.0, y=0.5)
    far, _ = _mouse(t, 200, 0.0)
    probs = np.tile([[1.0]], (t, 1))
    kin = body_kinematics(a_kp, cf, 30.0)
    social = [NAMES.index(n) for n in ("sniff_face", "sniff_anogenital", "huddle")]
    f_far, _ = full_probabilities(["resting"], probs, kin, [body_kinematics(far, cf, 30.0)])
    f_near, _ = full_probabilities(["resting"], probs, kin, [body_kinematics(near, cf, 30.0)])
    f_nose, _ = full_probabilities(["resting"], probs, kin, [body_kinematics(nose_to_nose, cf, 30.0)])
    assert f_far[:, social].sum(1).max() < 0.05
    assert f_near[20, NAMES.index("sniff_anogenital")] > 0.3  # a's nose at b's tail base
    assert f_nose[20, NAMES.index("sniff_face")] > 0.3
    assert np.allclose(f_near.sum(1), 1.0)


def test_remap_labels_between_label_sets():
    lab = np.array([0, 1, IGNORE, 2])
    out = remap_labels(lab, ("resting", "sniffing", "eating"), ("sniffing", "resting"))
    assert out.tolist() == [1, 0, IGNORE, IGNORE]


def test_kinematics_use_real_timestamps():
    kp, cf = _mouse(30, 0, 1.0)
    even = body_kinematics(kp, cf, 30.0)
    slow_clock = body_kinematics(kp, cf, 30.0, times=np.arange(30) * 2 / 30)  # frames really 1/15 s apart
    assert abs(np.median(slow_clock["speed"][5:-5]) * 2 - np.median(even["speed"][5:-5])) < 1e-6


def test_behavior_probs_on_variable_frame_rate():
    import torch

    from animalpose.behavior.model import BehaviorTCN
    from animalpose.pipeline import PipelineConfig, PosePipeline, TrackResult

    pipe = PosePipeline.__new__(PosePipeline)
    pipe.cfg, pipe.device = PipelineConfig(behavior_fps=15.0), torch.device("cpu")
    pipe.behavior = BehaviorTCN(25, 2, ("resting", "walking"), hidden=8).eval()
    kp, cf = _mouse(60, 0, 1.0)
    times = np.r_[np.arange(30) / 30, 2.0 + np.arange(30) / 60]  # 30 fps, a 1 s drop, then 60 fps
    tr = TrackResult(0, np.arange(60), np.zeros((60, 4)), kp, cf)
    probs = pipe._behavior_probs(tr, times)
    assert probs.shape == (60, 2) and np.allclose(probs.sum(1), 1.0, atol=1e-5)


def test_image_timestamps_cross_midnight(tmp_path):
    import cv2

    from animalpose.frames import frame_timestamps, timing_report

    for name in ("img_23_59_59_800.png", "img_23_59_59_900.png", "img_00_00_00_000.png", "img_00_00_00_300.png"):
        cv2.imwrite(str(tmp_path / name), np.zeros((4, 4), np.uint8))
    files = sorted(tmp_path.iterdir(), key=lambda p: p.name.replace("img_00", "img_24"))
    sub = tmp_path / "seq"
    sub.mkdir()
    for i, p in enumerate(files):
        p.rename(sub / f"{i:02d}_{p.name}")
    t = frame_timestamps(sub)
    assert np.allclose(t, [0.0, 0.1, 0.2, 0.5])
    rep = timing_report(t)
    assert rep["gaps"] == 1 and rep["variable"]
