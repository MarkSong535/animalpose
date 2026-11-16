"""Video-driven session builders fed tiny fake dataset trees, checked against the .npz sessions they write.

Covers MIT home-cage, ABODe and BORIS-labelled videos: tracking picks the animal rather than
debris, frames are strided and resampled to the requested fps, frame-wise / per-second / state-event
labels land on the right frames, absent boxes and uncoded time stay IGNORED instead of being guessed
as a class, and held-out animals / recordings / groups can be excluded when sessions are loaded back.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from animalpose.data.behavior_sets import (
    abode_boxes,
    build_abode,
    build_boris_videos,
    build_mit,
    load_session_dir,
    mit_labels,
    save_session,
    track_main_animal,
)
from animalpose.data.tracks import IGNORE, Session
from animalpose.labels import BEHAVIORS, BEHAVIORS_EXT
from animalpose.skeleton import CANONICAL

MOUSE = np.array([[4.0, 2.0, 24.0, 22.0]], np.float32)
DEBRIS = np.array([[1.0, 1.0, 5.0, 5.0]], np.float32)
MIT_TXT = "frame: 00000001-00000004 rest\nframe: 00000005-00000008 groom\nframe: 00000009-00000010 unlabeled\n"


class FakePipe:
    """Duck-typed stand-in for the pose pipeline: one animal box plus a smaller distractor per frame."""

    def __init__(self, blanks=()):
        self.blanks, self.calls = set(blanks), 0

    def detect(self, image_bgr):
        blank = self.calls in self.blanks
        self.calls += 1
        if blank:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return np.vstack([MOUSE, DEBRIS]), np.array([0.6, 0.99], np.float32)

    def estimate(self, image_bgr, boxes):
        boxes = np.asarray(boxes, np.float32)
        kp = np.stack([_kp_from_box(b) for b in boxes]) if len(boxes) else np.zeros((0, 25, 2), np.float32)
        return kp.astype(np.float32), np.full((len(boxes), 25), 0.8, np.float32), None


def _ext(*names: str) -> list[int]:
    return [BEHAVIORS_EXT.index(n) for n in names]


def _kp_from_box(box) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    out = np.zeros((CANONICAL.num_keypoints, 2))
    for j, kp in enumerate(CANONICAL.keypoints):
        fx = {"left": 0.25, "right": 0.75}.get(kp.side, 0.5)
        out[j, 0] = x0 + fx * (x1 - x0)
        out[j, 1] = y0 + (0.15 if kp.region == "head" else 0.6) * (y1 - y0)
    return out


def _video(path: Path, n: int, size_wh: tuple[int, int], fps: float, fourcc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size_wh
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    for t in range(n):
        img = np.full((h, w, 3), 40 + 30 * (t % 4), np.uint8)
        c = (5 * t) % (w - 8)
        img[4:h - 4, c:c + 8] = 235
        vw.write(img)
    vw.release()


def _mpg(path: Path, n: int, size_wh: tuple[int, int], fps: float = 30.0) -> None:
    """MJPEG/AVI bytes named .mpg: ffmpeg reads them by content, and a real MPEG program stream has no frame index."""
    _video(path.with_suffix(".avi"), n, size_wh, fps, "MJPG")
    path.with_suffix(".avi").replace(path)


def _mp4(path: Path, n: int, size_wh: tuple[int, int], fps: float = 25.0) -> None:
    _video(path, n, size_wh, fps, "mp4v")


def test_track_main_animal_prefers_the_animal_sized_detection(tmp_path):
    video = tmp_path / "v.mp4"
    _mp4(video, 5, (32, 24))
    coords, conf, boxes = track_main_animal(FakePipe(blanks={3}), video, stride=1)
    assert coords.shape == (5, CANONICAL.num_keypoints, 2) and boxes.shape == (5, 4)
    assert np.allclose(boxes[[0, 1, 2, 4]], MOUSE[0])  # the smaller, more confident blob is rejected
    np.testing.assert_allclose(coords[0], _kp_from_box(MOUSE[0]), atol=1e-4)
    assert (conf[[0, 1, 2, 4]] == 0.8).all()
    assert np.isnan(coords[3]).all() and (conf[3] == 0).all() and np.isnan(boxes[3]).all()
    coords, _, boxes = track_main_animal(FakePipe(), video, stride=2)
    assert len(coords) == len(boxes) == 3  # stride 2 of 5 frames


def test_build_mit_strides_frames_and_aligns_frame_wise_labels(tmp_path):
    root, out = tmp_path / "mit", tmp_path / "sessions"
    for name in ("v1", "v2", "v3"):
        _mpg(root / f"{name}.mpg", 10, (32, 24), fps=30.0)
    for name in ("v1", "v2"):
        (root / "Annotator_group_1").mkdir(parents=True, exist_ok=True)
        (root / "Annotator_group_1" / f"{name}.txt").write_text(MIT_TXT)

    assert build_mit(FakePipe(blanks={8}), root, out, stride=1) == ["v1", "v2"]
    assert build_mit(FakePipe(), root, out / "filtered", stride=1, only=["absent"]) == []
    assert not (out / "filtered" / "mit").exists()

    rest, groom = BEHAVIORS_EXT.index("resting"), BEHAVIORS_EXT.index("grooming")
    full = mit_labels(root / "Annotator_group_1" / "v1.txt", 10)
    assert full.tolist() == [rest] * 4 + [groom] * 4 + [IGNORE] * 2

    sessions = {s.animal: s for s in load_session_dir(out, target_fps=15.0)}
    assert sorted(sessions) == ["v1", "v2"]  # the video without a label file produced no session
    s = sessions["v1"]
    assert s.view == "side" and s.fps == 15.0 and len(s.labels) == 5
    assert s.labels.tolist() == full[::2].tolist() == [rest, rest, groom, groom, IGNORE]
    np.testing.assert_allclose(s.coords[0], _kp_from_box(MOUSE[0]), atol=1e-4)
    assert np.isnan(s.coords[4]).all()  # frame 8 lost its detection
    assert load_session_dir(out, target_fps=15.0, groups=["v1"])[0].animal == "v1"
    assert load_session_dir(out, target_fps=15.0, groups=["held_out"]) == []
    z = np.load(out / "mit" / "v1.npz")
    assert np.allclose(z["boxes"][::2][:4], MOUSE[0]) and np.isnan(z["boxes"][::2][4]).all()  # never the distractor
    assert [str(n) for n in z["label_names"]] == list(BEHAVIORS_EXT)


def _abode_annotations(path: Path) -> None:
    rows = [(1, 2, 3, bti) for bti in range(3)]
    idx = pd.MultiIndex.from_tuples(rows, names=["AnimalGroup", "Segment", "Snippet", "BTI"])
    fields = ["X", "Y", "Width", "Height"]
    cols = pd.MultiIndex.from_tuples([(m, f) for m in ("0", "1") for f in fields + ["GT.Behaviour"]]
                                     + [("2", f) for f in fields], names=["MouseId", "Field"])
    beh = {"0": ["Feed", "N/Obs", "S-Grm"], "1": ["Drink", "Imm", "Loco"]}
    data = {(m, f): ([float(b) for b in range(3)] if f != "GT.Behaviour" else beh[m]) for m, f in cols}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=idx, columns=cols).to_pickle(path, compression="bz2")


def _abode_identities(path: Path, n: int) -> None:
    t = np.arange(n)
    absent, collapsed = t < 10, 6
    cols, data = [], {}
    for m in ("0", "1", "2"):
        x, y, w, h = 2.0 + 0.3 * t, 1.0 + 0.2 * t, np.full(n, 10.0), np.full(n, 10.0)
        if m == "0":
            w[collapsed] = 0.0  # a zero-width box is not an animal
        if m == "1":
            x, y, w, h = (np.where(absent, np.nan, v) for v in (x, y, w, h))
        for f, v in zip(("X", "Y", "Width", "Height"), (x, y, w, h), strict=True):
            cols.append((m, f))
            data[(m, f)] = v
    idx = pd.MultiIndex.from_arrays([t // 2, t], names=["BTI", "FrameTimepoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_tuples(cols, names=["MouseId", "Field"])).to_pickle(path)


def test_abode_boxes_follow_the_identity_frame_index(tmp_path):
    n = 20
    ids = tmp_path / "Identities.df"
    _abode_identities(ids, n)
    boxes = abode_boxes(pd.read_pickle(ids), n)
    assert sorted(boxes) == ["0", "1", "2"]
    assert boxes["0"].shape == (n, 4)
    np.testing.assert_allclose(boxes["0"][7], [2.0 + 2.1, 1.0 + 1.4, 2.0 + 2.1 + 10.0, 1.0 + 1.4 + 10.0])
    assert np.isnan(boxes["1"][:10]).all() and np.isfinite(boxes["1"][10:]).all()


def test_build_abode_poses_annotated_boxes_and_spreads_per_second_labels(tmp_path):
    root, out, n = tmp_path / "abode", tmp_path / "sessions", 60
    _abode_annotations(root / "Annotations.df")
    _mp4(root / "Snippets" / "1_2_3" / "Video.mp4", n, (32, 24), fps=25.0)
    _abode_identities(root / "Snippets" / "1_2_3" / "Identities.df", n)
    _mp4(root / "Snippets" / "9_9_9" / "Video.mp4", 4, (32, 24), fps=25.0)

    assert build_abode(FakePipe(), root, out, stride=2) == ["1_2_3"]  # 9_9_9 has no annotation rows
    assert build_abode(FakePipe(), root, out / "off", stride=2, only=["4_4_4"]) == []
    sessions = load_session_dir(out / "abode", target_fps=12.5)
    assert sorted(s.animal for s in sessions) == ["cage1"] * 3
    assert [round(s.fps, 3) for s in sessions] == [12.5] * 3
    m0, m1, m2 = sessions
    assert len(m0.labels) == 30

    eat, rest, groom, walk, other = _ext("eating", "resting", "grooming", "walking", "other")
    sampled = list(range(0, n, 2))
    expect0 = [eat if t < 25 else (groom if t >= 50 else IGNORE) for t in sampled]
    assert m0.labels.tolist() == expect0  # a 1 s label covers the 25 frames it annotates
    assert m1.labels.tolist() == [other if t < 25 else (walk if t >= 50 else rest) for t in sampled]
    assert (m2.labels == IGNORE).all()  # an identity without a GT.Behaviour column stays unlabeled

    want = np.c_[2 + 0.3 * np.asarray(sampled, float), 1 + 0.2 * np.asarray(sampled, float),
                 12 + 0.3 * np.asarray(sampled, float), 11 + 0.2 * np.asarray(sampled, float)]
    want[sampled.index(6), 2] = want[sampled.index(6), 0]  # the collapsed box is kept, but never posed
    boxes0 = np.load(out / "abode" / "1_2_3_0.npz")["boxes"]
    np.testing.assert_allclose(boxes0, want, atol=1e-4)
    assert np.isnan(m0.coords[3]).all() and np.isfinite(m0.coords[4]).all()  # the collapsed box at t == 6
    assert np.isnan(m1.coords[:5]).all() and np.isfinite(m1.coords[5:]).all()  # mouse 1 enters at frame 10
    np.testing.assert_allclose(m0.coords[4], _kp_from_box(boxes0[4]), atol=1e-4)
    assert np.allclose(m0.conf[4], 0.8) and (m1.conf[:5] == 0).all()
    assert load_session_dir(out / "abode", target_fps=12.5, groups=["cage9"]) == []

    core = load_session_dir(out / "abode", target_fps=12.5, classes=BEHAVIORS)
    assert (m1.labels[:13] == other).all() and (core[1].labels[:13] == IGNORE).all()  # "other" has no core class


def test_build_boris_videos_uses_boris_state_events(tmp_path):
    vids = [tmp_path / f"M{7 + i}-20200101T000000-001000.mp4" for i in range(2)]
    for v in vids:
        _mp4(v, 10, (32, 24), fps=30.0)
    proj = {"behaviors_conf": {"0": {"code": "eating", "type": "State event"},
                               "1": {"code": "grooming", "type": "State event"},
                               "2": {"code": "approach", "type": "Point event"}},
            "observations": {"1": {"file": {"1": [str(vids[0])]},
                                   "media_info": {"length": {str(vids[0]): 1.0}},
                                   "events": [[0.05, "", "grooming", "", "", 0], [0.1, "", "eating", "", "", 0],
                                              [0.15, "", "approach", "", "", 0], [0.2, "", "eating", "", "", 0],
                                              [0.3, "", "grooming", "", "", 0]]}}}
    (tmp_path / "capture.boris").write_text(json.dumps(proj))
    out = tmp_path / "sessions"
    assert build_boris_videos(FakePipe(), vids, tmp_path, out, stride=1) == [v.stem for v in vids]

    other, eat = _ext("other", "eating")
    got = {s.animal: s for s in load_session_dir(out, target_fps=30.0)}
    assert sorted(got) == ["M7", "M8"]  # the animal is the recording id before the first dash
    assert got["M7"].view == "cage" and got["M7"].fps == 30.0
    assert got["M7"].labels.tolist() == [IGNORE, other, other, eat, eat, eat, eat, other, other, other]
    assert (got["M8"].labels == IGNORE).all()  # BORIS covers only one of the two media files
    assert len(got["M7"].coords) == 10
    assert [s.animal for s in load_session_dir(out, target_fps=30.0, groups=["M8"])] == ["M8"]

    bare = tmp_path / "unlabeled"
    assert build_boris_videos(FakePipe(), vids[:1], None, bare, stride=2) == [vids[0].stem]
    (only,) = load_session_dir(bare, target_fps=15.0)
    assert (only.labels == IGNORE).all() and only.fps == 15.0 and len(only.labels) == 5


def test_load_session_dir_remaps_label_sets_and_needs_every_key(tmp_path):
    lab = np.array([0, 1, 0, IGNORE, 1, 1], np.int64)
    s = Session("m", "side", np.zeros((6, 25, 2)), np.ones((6, 25)), lab, 30.0)
    save_session(tmp_path / "named.npz", s, label_names=("resting", "walking"))
    (back,) = load_session_dir(tmp_path, target_fps=30.0)
    want = _ext("resting", "walking", "resting", "resting", "walking", "walking")
    want[3] = IGNORE
    assert back.labels.tolist() == want
    np.savez_compressed(tmp_path / "legacy.npz", coords=s.coords.astype(np.float32), conf=s.conf.astype(np.float32),
                        labels=lab, fps=10.0, animal="legacy", view="top")
    legacy = {x.animal: x for x in load_session_dir(tmp_path, target_fps=10.0)}
    assert legacy["legacy"].labels[:3].tolist() == [0, 1, 0]  # unlabeled files are read with the default label set
    assert legacy["m"].fps == 10.0 and len(legacy["m"].coords) == 2  # 30 fps -> 10 fps keeps every third frame
    np.savez_compressed(tmp_path / "broken.npz", coords=s.coords, labels=lab)
    with pytest.raises(KeyError):
        load_session_dir(tmp_path)
