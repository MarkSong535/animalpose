"""``animalpose data audit`` measured against recordings built in this file, and ``data calibrate``.

Frames are rendered from ``animalpose.data.synthetic`` and written out as an .mp4 (and as an image
folder), so the animal size, frame rate, edge width and calibrated PSF the audit reports have a
truth we know: the projected skeleton, the encoder's own frame rate, and the Gaussian blur we
deliberately applied to a second copy of the same recording. The YAML it writes is read back the way
a training config reads it (``data.camera_profile`` -> :func:`animalpose.data.degrade.load_profile`).
``data calibrate`` is checked the same way, against corner points projected by cameras we chose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from animalpose.cli import main
from animalpose.cli.data.audit import audit, calibrate_psf, profile_from, surrogate_crops, write_yaml
from animalpose.data.degrade import Profile, load_profile
from animalpose.data.synthetic import camera_from_view, make_cameras, render, render_video, simulate
from animalpose.frames import open_source
from animalpose.multiview.camera import Camera
from animalpose.skeleton import CANONICAL

FPS, N_FRAMES, SIZE = 30.0, 60, (320, 256)
BLUR_SIGMA = 2.0  # what we add to the second copy of the recording
COAT = CANONICAL.indices(["nose", "head_center", "neck", "spine_upper", "spine_mid", "spine_lower", "tail_base",
                          "left_hip", "right_hip", "left_shoulder", "right_shoulder"])


def write_video(path: Path, images, fps: float = FPS) -> Path:
    assert len(images) > 2, path
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE[0], SIZE[1]))
    for img in images:
        vw.write(img)
    vw.release()
    return path


def render_frames(seed: int, n: int = N_FRAMES):
    sim = simulate(1, n, "mouse", seed=seed, coats=["black"])
    cam = make_cameras(sim.arena_cm, ("top",), size=SIZE)[0]
    return sim, cam, render_video(sim, cam, seed=3)


def coat_extent(frames) -> float:
    """The extent (px) of the coat the renderer drew: the projected skeleton without tail and limbs."""
    ext = [max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()) for p in
           (f.keypoints[0][COAT] for f in frames)]
    return float(np.median(np.asarray(ext, float)))


def stamp(t: int, step_ms: int = 100) -> str:
    """``img_HH_MM_SS_mmm`` names, the frame stamps :func:`animalpose.frames.estimate_fps` reads."""
    ms = t * step_ms
    return f"img_{ms // 3_600_000:02d}_{ms // 60_000 % 60:02d}_{ms // 1000 % 60:02d}_{ms % 1000:03d}.png"


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A 2 s recording of one black mouse on a bright floor, sharp and blurred by ``BLUR_SIGMA``."""
    out = tmp_path_factory.mktemp("audit")
    sim, cam, frames = render_frames(7)
    images = [f.image for f in frames]
    sharp = write_video(out / "sharp.mp4", images)
    blurred = write_video(out / "blur.mp4", [cv2.GaussianBlur(i, (0, 0), BLUR_SIGMA) for i in images])
    idx = np.linspace(0, len(frames) - 1, 24).astype(int)
    return {"dir": out, "sharp": sharp, "blurred": blurred, "frames": frames,
            "truth_px": coat_extent([frames[i] for i in idx]), "n_sampled": len(idx)}


@pytest.fixture(scope="module")
def surrogate(tmp_path_factory):
    """A prepared dataset (COCO JSON + images) for the PSF calibration, as ``--surrogate`` reads it."""
    base = tmp_path_factory.mktemp("surrogate")
    root = base / "crim13"
    (root / "images").mkdir(parents=True)
    sim, _, _ = render_frames(11, 12)
    cam = camera_from_view(sim.arena_cm, 60.0, size=SIZE)
    images, anns = [], []
    for t in range(8):
        f = render(sim, cam, t, t)
        rel = f"{t:03d}.png"
        cv2.imwrite(str(root / "images" / rel), f.image)
        images.append({"id": t, "file_name": rel, "height": SIZE[1], "width": SIZE[0]})
        keep = np.nonzero(f.present)[0]
        for a in keep[:1]:
            kp = [float(v) for xy in f.keypoints[a] for v in (*xy, 2)]
            x0, y0, x1, y1 = (float(v) for v in f.boxes[a])
            anns.append({"id": len(anns), "image_id": t, "category_id": 1, "keypoints": kp,
                         "num_keypoints": CANONICAL.num_keypoints, "track_id": int(a),
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": 1.0, "iscrowd": 0})
    (root / "val.json").write_text(json.dumps({"images": images, "annotations": anns, "categories":
                                              [{"id": 1, "name": "mouse", "keypoints": list(CANONICAL.names),
                                                "skeleton": []}]}))
    return base


def test_the_recording_is_read_as_it_was_written(recording):
    m = audit(open_source(recording["sharp"]), n_frames=24, windows=2, window=8)
    assert m["frames"] == N_FRAMES and m["source"] == str(recording["sharp"])
    assert m["fps"] == pytest.approx(FPS, abs=1e-3)
    assert m["duration_s"] == pytest.approx(N_FRAMES / FPS, abs=0.1)
    assert m["colour"] is True  # the renderer draws a colour floor, not a grey one
    assert m["frames_with_blob"] >= int(0.8 * recording["n_sampled"])  # the mouse is found in the frames
    assert m["edge_width_px"]["blocks"] == 4  # four time blocks, so focus drift can be seen


def test_the_animal_measures_what_the_renderer_drew(recording):
    m = audit(open_source(recording["sharp"]), n_frames=24, windows=2, window=8)
    truth = recording["truth_px"]
    assert abs(m["animal_px"]["median"] - truth) < 8.0, (m["animal_px"], truth)
    assert m["animal_px"]["p10"] <= m["animal_px"]["median"] <= m["animal_px"]["p90"]
    # The floor we painted is 120-200 grey, the wall 40-110; the median of the background is the floor.
    lo, med, hi = m["background_grey"]
    assert 110 <= med <= 210 and lo < med < hi
    assert m["frame_contrast"][1] - m["frame_contrast"][0] > 50
    assert m["noise_sigma"]["median"] < 8.0  # a clean render, so an almost noiseless camera
    assert m["sharpness_lapvar"]["median"] > m["sharpness_lapvar"]["p10"]


def test_displacement_and_smear_agree_with_the_exposure(recording):
    exposure = 0.5
    m = audit(open_source(recording["sharp"]), n_frames=24, exposure=exposure, windows=2, window=8)
    assert m["displacement_px"]["windows"] == 16
    assert m["displacement_px"]["median"] > 0.0  # the mouse walks, so frames are not identical
    assert m["motion_smear_px"] == pytest.approx(m["displacement_px"]["median"] * exposure, abs=0.02)
    assert m["motion_smear_body_frac"] == pytest.approx(m["motion_smear_px"] / m["animal_px"]["median"], abs=1e-3)


def test_a_blurred_copy_measures_a_wider_edge_and_a_bigger_psf(recording, surrogate):
    sharp = audit(open_source(recording["sharp"]), n_frames=24, windows=2, window=8)
    blur = audit(open_source(recording["blurred"]), n_frames=24, windows=2, window=8)
    edge_sharp, edge_blur = sharp["edge_width_px"]["median"], blur["edge_width_px"]["median"]
    assert edge_sharp < 2.5  # an unblurred silhouette measures its sampling grid plus fur softness
    # A Gaussian step of width sigma has its peak gradient at C / (sigma * sqrt(2*pi)), i.e. width
    # sigma * sqrt(2*pi) per the edge_width estimator; the audit has to report that blur.
    assert abs(edge_blur - BLUR_SIGMA * np.sqrt(2 * np.pi)) < 1.5, edge_blur
    assert edge_blur > edge_sharp + 2.0
    assert blur["animal_px"]["median"] == pytest.approx(sharp["animal_px"]["median"], abs=3.0)

    crops = surrogate_crops(surrogate, n=12, crop_size=64)
    assert len(crops) == 12 and crops[0].dtype == np.uint8 and crops[0].shape == (64, 64, 3)
    psf_blur, curve_blur = calibrate_psf(edge_blur, crops)
    psf_sharp, curve_sharp = calibrate_psf(edge_sharp, crops)
    assert psf_blur > psf_sharp, (psf_blur, psf_sharp)
    widths = list(curve_blur["edge_width_by_psf_px"].values())
    assert all(b >= a for a, b in zip(widths, widths[1:])) and widths[-1] > widths[0]  # monotone in the radius
    assert curve_blur["target_edge_width_px"] == round(edge_blur, 3)
    assert min(curve_blur["edge_width_by_psf_px"], key=lambda r: abs(curve_blur["edge_width_by_psf_px"][r]
                                                                    - edge_blur)) == psf_blur
    # A defocus disk of radius r measures ~1.4 r of edge, so the chosen radius is the target width / 1.4.
    assert abs(psf_blur - edge_blur / 1.4) <= 1.0, (psf_blur, edge_blur)


def test_the_profile_yaml_is_what_a_training_config_reads(recording, surrogate):
    m = audit(open_source(recording["blurred"]), n_frames=24, windows=2, window=8)
    crops = surrogate_crops(surrogate, n=12, crop_size=64)
    psf, calibration = calibrate_psf(m["edge_width_px"]["median"], crops)
    p, notes = profile_from(m, psf)
    assert p.psf_px == psf
    out = recording["dir"] / "camera.yaml"
    write_yaml(out, m, p, notes, calibration)
    text = out.read_text()
    assert text.startswith(f"# animalpose data audit on {m['source']} ({m['frames']} frames, {m['fps']:.2f} fps")
    assert all(f"# assumed: {n}" in text for n in notes)
    doc = yaml.safe_load(text)
    trained = {k: v for k, v in doc.items() if k not in ("measured", "notes", "source")}
    assert set(trained) == set(Profile.__dataclass_fields__)  # the keys data.camera_profile reads
    prof = load_profile(out)  # exactly what train/loops.py does with data.camera_profile
    assert prof.psf_px == psf and prof.jpeg == (40.0, 80.0)
    body = m["animal_px"]["median"]
    assert prof.downscale_to == (max(8.0, round(0.8 * body)), round(1.15 * body))
    contrast = m["frame_contrast"][1] - m["frame_contrast"][0]
    assert 100 < contrast < 150  # a bright floor against dark walls, but not a full-range camera
    assert prof.mono == 0.0 and prof.low_contrast == (0.0 if contrast > 150 else 1.0)
    assert doc["measured"]["psf_calibration"]["edge_width_by_psf_px"] == calibration["edge_width_by_psf_px"]
    assert doc["measured"]["edge_width_px"]["median"] == m["edge_width_px"]["median"]
    assert "source" not in doc["measured"]


def test_an_uncalibrated_psf_and_a_missing_animal_are_said_so(recording):
    m = audit(open_source(recording["sharp"]), n_frames=24, windows=2, window=8)
    p, notes = profile_from(m, None)
    assert p.psf_px == pytest.approx(m["edge_width_px"]["median"] / 1.4, abs=0.01)
    assert any("psf_px not calibrated" in n for n in notes)
    assert any("jpeg quality is assumed" in n for n in notes)
    assert 0.05 <= p.psf_jitter <= 0.6  # autofocus drift, clipped into a usable band
    assert p.noise[0] <= p.noise[1] and p.motion_frac[0] == 0.0

    blind = dict(m, animal_px={"p10": 0.0, "median": None, "p90": 0.0}, motion_smear_body_frac=None)
    p2, notes2 = profile_from(blind, 2.0)
    assert p2.motion_frac == (0.0, 0.2) and p2.downscale_to == (16.0, 48.0)
    assert any("no animal found" in n for n in notes2)
    assert not any("psf_px not calibrated" in n for n in notes2)


def test_surrogate_crops_only_read_prepared_datasets(surrogate, tmp_path):
    assert surrogate_crops(tmp_path / "nowhere") == []
    assert surrogate_crops(tmp_path) == []  # a folder that is not a prepared dataset
    crops = surrogate_crops(surrogate, n=5, crop_size=96)
    assert len(crops) == 5 and crops[0].shape == (96, 96, 3)
    assert crops[0].std() > 5  # a real image, not a blank patch


def test_the_cli_writes_a_calibrated_profile_and_the_measurements(recording, surrogate, capsys):
    out, js = recording["dir"] / "cli.yaml", recording["dir"] / "measured.json"
    code = main(["data", "audit", str(recording["blurred"]), "--frames", "16", "--windows", "2", "--window", "6",
                 "--surrogate", str(surrogate), "--crops", "8", "--crop-size", "64", "--out", str(out),
                 "--json", str(js)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "psf calibration:" in printed and f"-> {out}" in printed
    first = json.loads(printed.splitlines()[0])
    assert set(first) == {"source", "fps", "animal_px", "edge_width_px", "motion_smear_px", "noise_sigma",
                          "sharpness_lapvar"}
    doc = yaml.safe_load(out.read_text())
    measured = json.loads(js.read_text())
    assert len(measured) == 1 and measured[0]["source"] == str(recording["blurred"])
    assert doc["measured"]["psf_calibration"]["target_edge_width_px"] == measured[0]["edge_width_px"]["median"]
    assert "clean_crop_edge_width_px" in measured[0]["edge_width_px"]
    assert measured[0]["animal_px"]["median"] == pytest.approx(recording["truth_px"], abs=8.0)


def test_root_and_image_folders_are_audited_too(recording, tmp_path, capsys):
    root = tmp_path / "recordings"
    frames = recording["frames"][20:40]  # a stretch of the same recording, saved as a frame folder
    images = [f.image for f in frames]
    folder = root / "caicai_frames"
    folder.mkdir(parents=True)
    for t, img in enumerate(images):  # timestamps in the names are the frame rate of an image folder
        cv2.imwrite(str(folder / stamp(t)), img)
    write_video(root / "three.mp4", images[:3], fps=10.0)
    js = tmp_path / "all.json"
    code = main(["data", "audit", "--root", str(root), "--frames", "12", "--windows", "1", "--window", "4",
                 "--surrogate", str(tmp_path / "unprepared"), "--json", str(js)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "no surrogate crops under" in printed and "too few frames" in printed
    measured = json.loads(js.read_text())
    assert len(measured) == 2
    short = [m for m in measured if "error" in m]
    assert len(short) == 1 and short[0]["error"] == "too few frames"
    assert short[0]["source"] == str(root / "three.mp4")
    full = [m for m in measured if "error" not in m][0]
    assert full["frames"] == len(images) and full["source"] == str(folder)
    assert full["fps"] == pytest.approx(10.0, abs=0.2)  # from the file-name timestamps
    assert full["duration_s"] == pytest.approx(len(images) / 10.0, abs=0.2)
    assert full["animal_px"]["median"] > 0.0 and full["edge_width_px"]["median"] > 0.0


def test_the_cli_wants_a_path(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["data", "audit", "--frames", "8"])
    assert "give a path" in str(stop.value)


# --------------------------------------------------------------------------------- data calibrate


def rig_spec(views=("top", "side")):
    """Arena corners at two heights, projected by known cameras: the input of ``data calibrate``."""
    cams = make_cameras(40.0, views, size=SIZE)
    half = 20.0
    X = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (0.0, 10.0)], float)
    spec = {"cameras": [{"name": c.name, "size": list(SIZE), "points3d": X.tolist(),
                         "points2d": c.project(X).tolist()} for c in cams]}
    return cams, X, spec


def test_a_rig_is_solved_from_its_corner_points(tmp_path, capsys):
    cams, X, spec = rig_spec()
    points, out = tmp_path / "rig.json", tmp_path / "cams.json"
    points.write_text(json.dumps(spec))
    assert main(["data", "calibrate", str(points), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "wrote" in printed and all(f"{c.name}: mean reprojection error 0.0" in printed for c in cams)
    written = json.loads(out.read_text())
    assert [c["name"] for c in written] == ["top", "side"]
    for entry, cam in zip(written, cams):  # the DLT solution re-projects the corners exactly
        solved = Camera.from_dict(entry)
        assert np.abs(solved.project(X) - cam.project(X)).max() < 1e-6
        assert solved.size == SIZE


def test_known_intrinsics_and_a_bundle_adjustment_are_taken_when_offered(tmp_path, capsys):
    cams, X, spec = rig_spec(("top", "angled"))
    with_k = tmp_path / "rig_k.json"
    with_k.write_text(json.dumps({"cameras": [dict(c, K=cam.K.tolist(), dist=[0.0] * 5)
                                              for c, cam in zip(spec["cameras"], cams)]}))
    out = tmp_path / "cams_k.json"
    assert main(["data", "calibrate", str(with_k), "--out", str(out)]) == 0
    assert all(np.abs(Camera.from_dict(c).project(X) - cam.project(X)).max() < 1e-6
               for c, cam in zip(json.loads(out.read_text()), cams))

    # --refine: the same animal seen by both cameras (one keypoint missed in the second view).
    sim = simulate(1, 12, "mouse", seed=7, coats=["black"])
    obs = np.stack([render_video(sim, cam, seed=0)[t].keypoints[0] for t, cam in enumerate(cams)])
    obs[1, 3:5] = np.nan
    npz = tmp_path / "obs.npz"
    np.savez_compressed(npz, obs=obs.astype(np.float32))
    refined = tmp_path / "cams_refined.json"
    assert main(["data", "calibrate", str(tmp_path / "rig_k.json"), "--out", str(refined),
                 "--refine", str(npz)]) == 0
    before, after = (float(v) for v in re.findall(r"bundle adjustment: ([\d.]+)px -> ([\d.]+)px",
                                                  capsys.readouterr().out)[0])
    assert 0.0 < after < before < 1.0  # joint refinement only tightens the fit
    assert len(json.loads(refined.read_text())) == 2
