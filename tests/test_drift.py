"""Camera drift (``animalpose.multiview.drift``): the same cases as ``rust/src/multiview/drift.rs``."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from animalpose.config import load_config, pipeline_schema
from animalpose.multiview.camera import Camera, rodrigues, rotation_to_rvec
from animalpose.multiview.drift import DriftConfig, References, correct, register, turned

ROOT = Path(__file__).resolve().parents[1]


def camera() -> Camera:
    """Looking straight down from 2 units above a textured floor."""
    k = [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
    r = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    return Camera(k, r, [0.0, 0.0, 2.0], [-0.15, 0.01], (320, 240), "top")


def floor(x, y):
    """Smooth blobs and an edge, like a desk and walls."""
    blobs = 60 * np.sin(3.1 * x) * np.cos(2.3 * y) + 40 * np.sin(7 * x + 5 * y)
    edge = np.where((np.abs(x - 0.3) < 0.2) & (np.abs(y + 0.1) < 0.3), 50.0, 0.0)
    return 120 + blobs + edge


def picture(cam: Camera, gain=1.0, offset=0.0) -> np.ndarray:
    """What ``cam`` sees of the floor, lit with ``gain`` and ``offset`` (grey levels, truncated like u8)."""
    w, h = cam.size
    ys, xs = np.mgrid[0:h, 0:w]
    n = cam.undistort_points(np.stack([xs, ys], -1).reshape(-1, 2).astype(float))
    d = np.column_stack([n, np.ones(len(n))]) @ cam.R  # R.T @ ray
    c = cam.center
    s = -c[2] / d[:, 2]
    v = floor(c[0] + s * d[:, 0], c[1] + s * d[:, 1])
    return np.floor(np.clip(gain * v + offset, 0, 255)).reshape(h, w)


def off_by(knock, rotation) -> float:
    return float(np.degrees(np.linalg.norm(rotation_to_rvec(knock.T @ rotation))))


def test_a_turned_camera_is_measured_and_corrected():
    cam = camera()
    knock = rodrigues([0.004, -0.007, 0.010])
    moved = turned(cam, knock)
    reg = register(cam, picture(cam), picture(moved, 0.85, 12.0))  # dimmer, turned ~0.7°
    assert off_by(knock, reg.rotation) < 0.02
    assert abs(reg.degrees - 0.7) < 0.05 and reg.after > reg.before
    p = np.array([0.2, -0.15, 0.0])
    assert np.linalg.norm(turned(cam, reg.rotation).project(p) - moved.project(p)) < 0.2


def test_a_changed_scene_does_not_pull_the_camera():
    cam = camera()
    knock = rodrigues([-0.006, 0.003, 0.005])
    now = picture(turned(cam, knock))
    ys, xs = np.mgrid[30:110, 180:260]
    now[30:110, 180:260] = np.where((xs // 8 + ys // 8) % 2 == 0, 250.0, 10.0)  # a cup put down
    reg = register(cam, picture(cam), now)
    assert off_by(knock, reg.rotation) < 0.05 and reg.after > reg.before


def test_a_camera_that_stayed_put_stays_put():
    cam = camera()
    assert register(cam, picture(cam), picture(cam, 1.1, -5.0)).degrees < 0.01


def test_the_first_session_becomes_the_reference(tmp_path):
    refs = References.beside(tmp_path / "rig.json")
    cam = camera()
    cams, entries = correct([cam], [picture(cam)], refs)
    assert entries[0].status == "reference saved" and cams[0] is cam
    assert (tmp_path / "rig.references" / "top.png").is_file()

    knock = rodrigues([0.0, 0.006, -0.004])
    cams, entries = correct([cam], [picture(turned(cam, knock))], refs)
    assert entries[0].status == "corrected", entries[0]
    assert off_by(knock, cams[0].R @ cam.R.T) < 0.05
    assert entries[0].line().startswith("top corrected: turned")
    assert set(entries[0].to_dict()) == {"camera", "status", "degrees", "before", "after", "reference"}

    cams, entries = correct([cam], [np.zeros((120, 160))], refs)
    assert entries[0].status == "skipped" and cams[0] is cam


def test_the_registration_follows_the_config():
    cam = camera()
    knock = rodrigues([0.004, -0.007, 0.010])
    now = picture(turned(cam, knock))
    still = register(cam, picture(cam), now, DriftConfig(iterations=0))
    assert still.degrees == 0.0 and still.after == still.before  # no steps: no turn
    coarse = register(cam, picture(cam), now, DriftConfig(samples=2000, coarsest=32))
    assert off_by(knock, coarse.rotation) < 0.1


def test_the_shipped_config_spells_out_the_defaults():
    """``configs/pipeline.yaml`` and the Rust ``DriftCfg`` share these numbers."""
    shipped = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text())["drift"]
    assert shipped == asdict(DriftConfig())
    assert pipeline_schema()["drift"] == set(shipped)


def test_drift_can_be_switched_off_from_a_config(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(f"extends: {ROOT / 'configs/pipeline.yaml'}\ndrift:\n  enabled: false\n")
    cfg = load_config(child, ["drift.huber=2.0"], pipeline_schema())
    assert DriftConfig(**cfg["drift"]) == DriftConfig(enabled=False, huber=2.0)
