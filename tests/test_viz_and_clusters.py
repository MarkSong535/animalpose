"""Overlays and the label-free behavior path, on tracks we build ourselves: skeleton drawing, the
kinematics that feed clustering, the cluster-to-behavior naming, and the ethogram figure and clip."""

import json
import math

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from animalpose.behavior.catalog import NAMES, canonical, describe
from animalpose.behavior.clusters import (
    NAME_RULES,
    bouts,
    cluster_tracks,
    kinematics,
    label_clusters,
    montage,
    naming_agreement,
    parse_mapping,
    write_naming,
)
from animalpose.behavior.plot import clip, ethogram, plot_run
from animalpose.pipeline import TrackResult, VideoResult
from animalpose.skeleton import CANONICAL
from animalpose.viz import color, draw_frame, draw_pose

FPS, T, MOVE = 30.0, 120, 60
OFFSETS = {"nose": (20, 0), "head_center": (15, 0), "neck": (11, 0), "spine_upper": (7, 0), "spine_mid": (0, 0),
           "spine_lower": (-7, 0), "tail_base": (-20, 0), "tail_mid": (-28, 0), "tail_end": (-35, 0),
           "left_eye": (17, 2), "right_eye": (17, -2), "left_ear": (13, 3), "right_ear": (13, -3),
           "left_shoulder": (7, 4), "right_shoulder": (7, -4), "left_front_knee": (11, 6),
           "right_front_knee": (11, -6), "left_front_paw": (14, 7), "right_front_paw": (14, -7),
           "left_hip": (-7, 4), "right_hip": (-7, -4), "left_hind_knee": (-11, 6),
           "right_hind_knee": (-11, -6), "left_hind_paw": (-14, 7), "right_hind_paw": (-14, -7)}


def _body(centre, curl: float = 0.0) -> np.ndarray:
    """A 40 px straight animal pointing +x, its rear half curled by ``curl`` when given."""
    kp = np.zeros((CANONICAL.num_keypoints, 2), np.float32)
    for name, (a, b) in OFFSETS.items():
        kp[CANONICAL.index(name)] = centre + np.array([a, b], np.float32)
    if curl:
        r = np.array([[np.cos(curl), -np.sin(curl)], [np.sin(curl), np.cos(curl)]])
        for name in ("spine_lower", "tail_base", "tail_mid", "tail_end"):
            k = CANONICAL.index(name)
            kp[k] = centre + (kp[k] - centre) @ r.T
    return kp


def _pixel(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The pixel cv2 would have drawn at ``(x, y)`` is stored at ``[y, x]``."""
    return image[int(round(float(xy[1]))), int(round(float(xy[0])))]


def _painted(image: np.ndarray, col, tol: int = 60) -> int:
    return int((np.abs(image.astype(int) - np.array(col)).sum(-1) < tol).sum())


def _is_col(pixel: np.ndarray, col, tol: int = 60) -> bool:
    return bool(np.abs(np.asarray(pixel, int) - np.array(col)).sum() < tol)


def _colored(image: np.ndarray) -> int:
    return int((image.astype(int).max(-1) - image.astype(int).min(-1) > 40).sum())


def _decoded(path) -> list[np.ndarray]:
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def _track(tid: int, start: int, kp: np.ndarray, behavior) -> TrackResult:
    conf = np.full(kp.shape[:2], 0.9, np.float32)
    return TrackResult(tid, np.arange(start, start + len(kp)), np.zeros((len(kp), 4), np.float32), kp, conf,
                       smoothed=kp, smoothed_conf=conf, imputed=np.zeros_like(conf), behavior=behavior)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A track that runs straight at 4 body lengths/s for 1.3 s, then curls up and stops; its video,
    a ``tracks.npz`` and the behavior catalogue a run would leave next to it."""
    out = tmp_path_factory.mktemp("behavior-run")
    kp = np.stack([_body(np.array([40.0 + 4.0 * min(t, MOVE - 1), 120.0]),
                         0.0 if t < MOVE else np.radians(90)) for t in range(T)])
    conf = np.full((T, CANONICAL.num_keypoints), 0.9, np.float32)
    times = np.arange(T) / FPS
    modelled = np.zeros(len(NAMES), bool)
    used = [NAMES.index(n) for n in ("rest", "walk", "self_groom")]
    modelled[used] = True
    full = np.zeros((T, len(NAMES)))
    full[:, used[0]], full[:, used[1]], full[:, used[2]] = np.where(times > 1.2, 0.8, 0.1), \
        np.where(times > 1.2, 0.1, 0.8), 0.1
    np.savez_compressed(out / "tracks.npz", track0_smoothed=kp, track0_smoothed_conf=conf,
                        track0_frames=np.arange(T), times=times, track0_behavior_full=full,
                        keypoint_names=np.array(CANONICAL.names), behaviors=np.array(NAMES))
    (out / "behavior_catalog.json").write_text(json.dumps(describe(modelled)))
    vw = cv2.VideoWriter(str(out / "synth.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 240))
    for t in range(T):
        img = np.full((240, 320, 3), 150, np.uint8)
        cv2.circle(img, tuple(kp[t, CANONICAL.index("spine_mid")].astype(int)), 12, (25, 25, 25), -1)
        vw.write(img)
    vw.release()
    return {"dir": out, "tracks": out / "tracks.npz", "video": out / "synth.avi", "kp": kp, "conf": conf,
            "times": times, "full": full, "names": list(NAMES), "modelled": modelled}


def test_kinematics_measures_speed_in_body_lengths_per_second(recording):
    kp, conf = recording["kp"], recording["conf"]
    feats, names = kinematics(kp, conf, FPS)
    assert names == ["speed", "length", "turn", "posture", "head_motion", "bend", "speed_sd", "length_sd",
                     "posture_sd"] and feats.shape == (T, 9) and np.isfinite(feats).all()
    length = np.linalg.norm(kp[:, CANONICAL.index("nose")] - kp[:, CANONICAL.index("tail_base")], axis=-1)
    per_frame = 4.0
    assert feats[:20, 0] == pytest.approx(per_frame * FPS / np.median(length), abs=0.05)  # running
    assert feats[-20:, 0] == pytest.approx(0.0, abs=0.05)  # stopped
    half = kinematics(kp, conf, FPS, np.arange(T) / (FPS / 2))[0]
    assert half[:20, 0] == pytest.approx(feats[:20, 0] / 2, abs=0.05)  # the same track played half as fast
    assert (feats[:20, 1] > 1.1).all() and (feats[-20:, 1] < 0.9).all()  # stretched, then curled up
    assert (feats[:20, 5] < 0.2).all() and feats[-20:, 5].mean() > 0.5  # straight, then bent over
    assert feats[:20, 4] == pytest.approx(0.0, abs=0.05) and feats[:20, 3] == pytest.approx(0.0, abs=0.05)
    assert bouts(np.array([0, 0, 1, 1, 1, 0])) == [(0, 2, 0), (2, 5, 1), (5, 6, 0)]
    unsmoothed = [kinematics(kp, conf, f, recording["times"])[0] for f in (1.0, 2.0)]
    assert unsmoothed[0][:, :6] == pytest.approx(unsmoothed[1][:, :6])  # a one-frame window: no smoothing
    assert unsmoothed[0][MOVE, 0] > feats[MOVE, 0]  # the frame it stopped on stays sharp, not smeared
    assert (unsmoothed[0][:, 6:] >= 0).all()


def test_clusters_split_moving_from_still(recording, tmp_path):
    out = tmp_path / "clusters"
    summary = cluster_tracks(recording["tracks"], recording["video"], out, fps=FPS, train_range=(0.0, 0.8), k=2)
    assert sorted(summary) == [0, 1] and abs(sum(s["share_of_train"] for s in summary.values()) - 1) < 1e-6
    assert (out / "clusters.npz").exists() and (out / "summary.json").read_text() == json.dumps(summary, indent=1)
    for c in summary:
        assert (out / f"cluster{c}.png").exists()
    thin = cluster_tracks(recording["tracks"], recording["video"], out / "thin", fps=FPS,
                          train_range=(0.0, 0.05), k=2)
    assert sorted(thin) == [0, 1] and (out / "thin" / "clusters.npz").exists()
    assert not any((out / "thin" / f"cluster{c}.png").exists() for c in thin)  # no bout long enough to show
    fast = max(summary, key=lambda c: summary[c]["speed"])
    assert summary[fast]["speed"] > 3.0 and summary[fast]["length"] > 0.9 and summary[fast]["bend"] < 1.0
    assert summary[1 - fast]["speed"] < 0.35 and summary[1 - fast]["length"] < 0.9
    npz = np.load(out / "clusters.npz")
    assert set(np.unique(npz["labels"])) == {0, 1} and npz["features"].shape == (T, 9)
    assert float(np.median(npz["labels"][:20])) == float(fast) and float(np.median(npz["labels"][-20:])) != float(fast)


def test_naming_and_label_files_follow_the_kinematics(recording, tmp_path):
    clusters = tmp_path / "clusters"
    summary = cluster_tracks(recording["tracks"], recording["video"], clusters, fps=FPS, train_range=(0.0, 0.8), k=2)
    fast = max(summary, key=lambda c: summary[c]["speed"])
    report = write_naming(clusters)
    assert (clusters / "naming.json").exists()  # every published rule is recorded with its thresholds
    assert {n: tuple(tuple(c) for c in crit) for n, crit in report["rules"].items()} == dict(NAME_RULES)
    assert report["naming"][str(fast)]["name"] == "running"  # the fast, stretched cluster
    assert report["naming"][str(1 - fast)]["name"] == "resting"  # the stopped, curled one is not locomotion
    assert {canonical(n["name"]) for n in report["naming"].values()} == {"run", "rest"}  # catalogue classes
    mapping = {fast: "running", 1 - fast: "resting"}
    labels = tmp_path / "labels"
    share = label_clusters(clusters, mapping, labels, name="synth", view="top", min_bout=0.5)
    assert set(share) == {"running", "resting"} and sum(share.values()) == pytest.approx(1.0)
    assert share["running"] == pytest.approx(MOVE / T, abs=0.08) and share["resting"] > 0.4
    for part in ("train", "heldout"):
        assert (labels / part / f"synth_{part}.npz").exists()
    held = np.load(labels / "heldout" / "synth_heldout.npz")
    assert float(held["fps"]) == FPS and str(held["view"]) == "top" and str(held["animal"]) == "synth_heldout"
    assert set(str(n) for n in held["label_names"]) == {"running", "resting"}
    assert [str(n) for n in held["label_names"][held["labels"]]] == ["resting"] * len(held["labels"])
    train = np.load(labels / "train" / "synth_train.npz")
    names = np.array([str(n) for n in train["label_names"]])
    assert (names[train["labels"]][:45] == "running").all() and (names[train["labels"]][75:] == "resting").all()
    written = json.loads((labels / "labels.json").read_text())
    assert written["mapping"] == {str(k): v for k, v in mapping.items()} and written["min_bout_s"] == 0.5
    assert written["share"] == share
    agree = write_naming(clusters, agree_with=labels / "labels.json")["agreement"]
    assert agree["n_named_clusters"] == 2 and agree["weighted_agreement"] == 1.0 and agree["mismatches"] == []
    assert naming_agreement(report["naming"], {1 - fast: "resting"})["weighted_agreement"] == 1.0
    wrong = naming_agreement(report["naming"], {fast: "grooming"})
    assert wrong["n_named_clusters"] == 1 and wrong["weighted_agreement"] == 0.0
    assert wrong["mismatches"] == [f"cluster {fast}: rule says running, human said grooming"]
    nobody = naming_agreement(report["naming"], {"7": "resting"})
    assert nobody["n_named_clusters"] == 0 and nobody["mismatches"] == []
    assert math.isnan(nobody["weighted_agreement"])  # nothing was named by hand, nothing to agree with
    assert parse_mapping(["0=resting", "2=grooming"]) == {0: "resting", 2: "grooming"}


def test_label_clusters_rejects_names_outside_the_catalogue_and_merges_short_bouts(recording, tmp_path):
    clusters = tmp_path / "clusters"
    summary = cluster_tracks(recording["tracks"], recording["video"], clusters, fps=FPS, train_range=(0.0, 0.8), k=2)
    fast = max(summary, key=lambda c: summary[c]["speed"])
    with pytest.raises(ValueError, match="catalogue"):
        label_clusters(clusters, {fast: "sidewalking_xyz"}, tmp_path / "none")
    share = label_clusters(clusters, {fast: "running", 1 - fast: "resting"}, tmp_path / "merged", min_bout=60.0)
    assert share == {"running": 1.0}
    for part in ("train", "heldout"):
        z = np.load(tmp_path / "merged" / part / f"recording_{part}.npz")
        assert set(str(n) for n in z["label_names"][z["labels"]]) == {"running"}


def test_montage_strips_are_cropped_around_the_animal(recording, tmp_path):
    centre = recording["kp"][:, CANONICAL.index("spine_mid")]
    strip = montage(recording["video"], centre, [5, 10, 15, 20, 25, 30])
    assert strip.shape == (180, 6 * 180, 3) and strip.dtype == np.uint8
    grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    assert (grey < 90).mean() > 0.01 and grey.mean() < 160  # the animal is in every tile
    empty = montage(tmp_path / "absent.avi", centre, [0, 1])
    assert empty.shape == (10, 10, 3) and not empty.any()


def test_ethogram_figure_reports_the_time_share_of_each_behavior(recording, tmp_path, monkeypatch):
    closed = {}
    monkeypatch.setattr(plt, "close", lambda fig: closed.__setitem__("fig", fig))
    share = ethogram(tmp_path / "ethogram.png", recording["names"], recording["full"], recording["times"],
                     recording["modelled"], heldout_from=2.0)
    assert set(share) == {"rest", "walk", "self_groom"} and sum(share.values()) == pytest.approx(1.0)
    assert share["walk"] == pytest.approx(37 / 120, abs=0.01) and share["self_groom"] == 0.0
    fig = closed["fig"]
    assert matplotlib.get_backend() == "Agg" and len(fig.axes) == 3
    ax0, ax1, ax2 = fig.axes
    assert ax0.get_ylabel() == "probability" and ax0.get_ylim() == (0.0, 1.0)
    assert ax0.get_xlim() == (recording["times"][0], recording["times"][-1])
    assert [t.get_text() for t in ax0.legend().get_texts()] == ["rest", "walk", "self_groom"]
    assert len(ax0.lines) == len(ax1.lines) == 1 and len(ax0.texts) == 1  # the held-out line and its label
    assert len(ax2.patches) == 3 and len(ax1.patches) == 2  # one barh per behavior, one band per bout
    assert ax2.get_xlabel().startswith("time as most likely") and ax1.get_yticks().size == 0
    png = cv2.imread(str(tmp_path / "ethogram.png"))
    assert png is not None and png.std() > 20 and (png < 250).any()
    ethogram(tmp_path / "plain.png", recording["names"], recording["full"], recording["times"],
             recording["modelled"])
    assert len(closed["fig"].axes[0].lines) == 0 and len(closed["fig"].axes[0].texts) == 0


def test_clip_zooms_on_the_animal_with_the_skeleton_and_behavior_panel(recording, tmp_path):
    z = np.load(recording["tracks"], allow_pickle=False)
    out = clip(tmp_path / "clip.mp4", recording["video"], list(NAMES), z, 0.5, 1.0, 2, scale=2, half=30)
    assert out == tmp_path / "clip.mp4" and out.exists() and out.stat().st_size > 1000
    frames = _decoded(out)
    assert len(frames) == 30 and frames[0].shape == (120, 420, 3)
    for f in frames:
        assert _colored(f[:, :120]) > 200  # the skeleton and keypoints, not just the crop
        assert f[:, 120:].mean() > 150 and f[:, :120].mean() < f[:, 120:].mean()  # the white panel
    assert _colored(frames[0][:, 120:]) > 100  # the behavior bars are colored too
    gapped = dict(np.load(recording["tracks"]))
    gapped["track0_frames"] = np.r_[np.arange(20), np.arange(25, T)]  # the tracker lost five frames
    assert len(_decoded(clip(tmp_path / "gapped.mp4", recording["video"], list(NAMES), gapped,
                             0.5, 1.0, 2, scale=2, half=30))) == 25  # lost frames are skipped, not blanked
    assert len(_decoded(clip(tmp_path / "past.mp4", recording["video"], list(NAMES), z, 3.5, 2.0, 2,
                             scale=2, half=30))) == T - 105  # the clip stops where the video does
    stale = dict(np.load(recording["tracks"]))
    stale["times"] = np.arange(2 * T) / FPS  # the track timeline is longer than the video it came from
    assert len(_decoded(clip(tmp_path / "stale.mp4", recording["video"], list(NAMES), stale, 3.5, 4.0, 2,
                             scale=2, half=30))) == T - 105  # an unreadable frame ends it, nothing is padded
    faded = dict(np.load(recording["tracks"]))
    faded["track0_smoothed_conf"][:, CANONICAL.index("nose")] = 0.1
    dim = _decoded(clip(tmp_path / "dim.mp4", recording["video"], list(NAMES), faded, 0.5, 0.3, 2, scale=2, half=30))
    lit = _decoded(clip(tmp_path / "lit.mp4", recording["video"], list(NAMES), z, 0.5, 0.3, 2, scale=2, half=30))
    assert len(dim) == len(lit) == 9
    assert _colored(dim[0][:, :120]) < _colored(lit[0][:, :120])  # the shy nose: no dot, no limb to it
    assert np.abs(dim[0][:, 120:].astype(int) - lit[0][:, 120:].astype(int)).mean() < 1.0  # panel unchanged


def test_plot_run_writes_the_ethogram_and_clip_next_to_the_tracks(recording):
    run = recording["dir"]
    share = plot_run(run, recording["video"], clip_start=0.5, clip_seconds=1.0, top=3, heldout_from=2.0)
    assert share["rest"] == pytest.approx(0.69, abs=0.02)
    eth, movie = cv2.imread(str(run / "ethogram.png")), cv2.VideoCapture(str(run / "behavior_clip.mp4"))
    ok, first = movie.read()
    movie.release()
    assert eth is not None and eth.shape[0] > 500 and ok and first.shape == (360, 660, 3)


def test_draw_pose_paints_skeleton_keypoints_and_label(tmp_path):
    frame = np.full((80, 100, 3), 60, np.uint8)
    kp = _body(np.array([50.0, 40.0]))
    conf = np.full(CANONICAL.num_keypoints, 0.9, np.float32)
    assert color(1) != color(2) and color(7) == color(1)  # the palette cycles with the track id
    out = draw_pose(frame.copy(), kp, conf, track_id=1, label="walk")
    assert out.shape == frame.shape and not np.array_equal(out, frame)
    assert _painted(frame, color(1)) == 0 and _painted(out, color(1)) > 200 and _painted(out, color(0)) == 0
    for name in ("nose", "tail_end", "left_hind_paw"):
        assert _is_col(_pixel(out, kp[CANONICAL.index(name)]), color(1)), name
    paws = CANONICAL.indices(["left_front_paw", "right_front_paw", "left_hind_paw", "right_hind_paw"])
    limbs_off = conf.copy()
    limbs_off[paws] = 0.0
    fewer = draw_pose(frame.copy(), kp, limbs_off, track_id=1, label="walk")
    assert 0 < _painted(fewer, color(1)) < _painted(out, color(1))  # an edge needs both endpoints confident
    paw, col = CANONICAL.index("left_front_paw"), color(0)
    only = np.zeros_like(conf)
    only[paw] = 0.9
    assert _is_col(_pixel(draw_pose(frame.copy(), kp, only), kp[paw]), col)  # one confident point: a solid dot
    hollow = draw_pose(frame.copy(), kp, only, imputed=np.ones_like(conf))
    assert not _is_col(_pixel(hollow, kp[paw]), col) and _painted(hollow, col) > 0  # imputed points are drawn hollow
    assert np.array_equal(draw_pose(frame.copy(), kp, conf, thresh=0.95), frame)  # below the threshold
    plain = draw_pose(frame.copy(), kp)  # with no confidence array, every finite point counts
    assert np.array_equal(plain, draw_pose(frame.copy(), kp, np.ones_like(conf)))
    broken = kp.copy()
    broken[paw] = np.nan
    assert np.array_equal(draw_pose(frame.copy(), broken, only), frame)  # not finite: nothing to draw
    named = _painted(draw_pose(frame.copy(), kp, conf, label="groom"), col)
    assert named > _painted(draw_pose(frame.copy(), kp, conf), col)  # the behavior name adds pixels
    assert cv2.imwrite(str(tmp_path / "overlay.png"), out)
    assert np.array_equal(cv2.imread(str(tmp_path / "overlay.png")), out)  # what was drawn is what was written


def test_draw_frame_draws_only_the_tracks_alive_at_that_frame():
    image = np.full((80, 100, 3), 60, np.uint8)
    left = np.stack([_body(np.array([25.0, 30.0]))] * 10)
    right = np.stack([_body(np.array([70.0, 60.0]))] * 6)
    named = {0: _track(0, 10, left, np.zeros(10, np.int64)), 1: _track(1, 0, right, np.ones(6, np.int64))}
    result = VideoResult(named, 20, NAMES)
    early = draw_frame(image, result, 3)  # only track 1 has started
    assert _painted(early, color(1)) > 200 and _painted(early, color(0)) == 0
    late = draw_frame(image, result, 15)  # track 1 has ended
    assert _painted(late, color(0)) > 200 and _painted(late, color(1)) == 0
    assert np.array_equal(draw_frame(image, result, 25), image)  # past every track
    assert np.array_equal(image, np.full((80, 100, 3), 60, np.uint8))  # the input is never touched
    quiet = VideoResult({0: _track(0, 10, left, None), 1: _track(1, 0, right, None)}, 20, ())
    nameless = draw_frame(image, quiet, 3)
    assert _painted(nameless, color(1)) > 0
    assert _painted(early, color(1)) > _painted(nameless, color(1))  # the catalogue name adds the text
