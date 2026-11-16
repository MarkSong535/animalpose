"""DeepLabCut/HDF5 track loaders and the camera-profile degradations, on tiny generated fixtures.

Checks that DLC bodyparts land on canonical keypoints and that unpaired or held-out recordings are
skipped, that mismatched track/label lengths are truncated rather than misaligned, that invalid
prediction files raise, that smoothing only ever fills in points the source tracks, and that the
profile-driven degradations behave physically: a larger psf_px measurably removes high-frequency
energy and widens the subject silhouette, a mono sensor gives three identical channels, and a fixed
seed reproduces every draw.
"""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from animalpose.data.degrade import DEFAULT_PROFILE, Profile, degrade, edge_width, load_profile, motion_smear
from animalpose.data.tracks import (
    IGNORE,
    Session,
    behavior_windows,
    load_dlc_h5,
    load_label_csv,
    load_sessions,
    smooth_sessions,
    track_list,
)
from animalpose.labels import BEHAVIORS_EXT
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS
from animalpose.temporal.model import TemporalLSTM

DLC_BODYPARTS = ["nose", "leftear", "rightear", "headstage", "midbody", "tailbase"]
TRACKED = CANONICAL.indices(DATASET_MAPPINGS["homecage_dlc"][b] for b in DLC_BODYPARTS)
UNTRACKED = [i for i in range(CANONICAL.num_keypoints) if i not in TRACKED]


def _ext(*names: str) -> list[int]:
    return [BEHAVIORS_EXT.index(n) for n in names]


def _dlc(path: Path, t: int = 40, coords: tuple[str, ...] = ("x", "y", "likelihood"), seed: int = 0) -> pd.DataFrame:
    pytest.importorskip("tables")
    cols = pd.MultiIndex.from_product([["DLC_model"], DLC_BODYPARTS, coords], names=["scorer", "bodyparts", "coords"])
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.random((t, len(cols))) * 60.0, columns=cols)
    frame.loc[:, (slice(None), slice(None), coords[-1])] = rng.random((t, len(DLC_BODYPARTS)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_hdf(path, key="df_with_missing", format="table")
    return frame


def _labels(path: Path, t: int, active: dict[str, tuple[int, int]] | None = None) -> None:
    frame = pd.DataFrame(0.0, index=range(t), columns=["Locomotion", "Grooming", "Nesting"])
    for col, (a, b) in (active or {}).items():
        frame.loc[a:b, col] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _recording(root: Path, animal: str, view: str, stamp: str, h5_t: int | None, csv_t: int | None,
               active: dict[str, tuple[int, int]] | None = None) -> None:
    """One ``<animal>_<view>`` directory; None omits the DLC file or the label CSV."""
    d = root / f"{animal}_{view}"
    if h5_t is not None:
        _dlc(d / "DLC" / f"{animal}-{stamp}DLC_model.h5", h5_t)
    if csv_t is not None:
        _labels(d / "Behavior_Labels" / "CSV" / f"{animal}-{stamp}_merged.csv", csv_t, active)


def _dlc_tree(root: Path) -> list[Session]:
    _recording(root, "MOUSE1", "top", "20220101T000000-010000", 40, 40,
               {"Locomotion": (0, 20), "Grooming": (5, 9), "Nesting": (30, 33)})
    _recording(root, "MOUSE2", "side", "20220102T000000-010000", 40, 25)  # its labels stop early
    _recording(root, "MOUSE3", "top", "20220103T000000-010000", None, 30)  # labeled but never tracked
    _recording(root, "MOUSE4", "top", "20220104T000000-010000", 30, None)  # tracked but never labeled
    (root / "notes").mkdir()  # no underscore: not a recording directory
    (root / "readme.txt").write_text("a stray file is not a recording either\n")
    return load_sessions(root)


def test_dlc_h5_rejects_an_unexpected_coordinate_layout(tmp_path):
    _dlc(tmp_path / "bad.h5", t=6, coords=("x", "y", "score"))
    with pytest.raises(ValueError, match="unexpected DLC column layout"):
        load_dlc_h5(tmp_path / "bad.h5")


def test_load_sessions_pairs_tracks_with_labels_by_recording(tmp_path):
    sessions = _dlc_tree(tmp_path)
    assert [s.animal for s in sessions] == ["MOUSE1", "MOUSE2"]  # MOUSE3 has no track, MOUSE4 no labels
    assert [s.view for s in sessions] == ["top", "side"]
    assert [len(s.labels) for s in sessions] == [40, 25]  # a shorter label CSV truncates, never misaligns
    assert [s.animal for s in load_sessions(tmp_path, animals=["MOUSE2"])] == ["MOUSE2"]  # held-out animals
    assert all(s.fps == 15.0 for s in sessions)

    s = sessions[0]
    h5 = tmp_path / "MOUSE1_top" / "DLC" / "MOUSE1-20220101T000000-010000DLC_model.h5"
    bps, xy, lk = load_dlc_h5(h5)
    assert bps == DLC_BODYPARTS and xy.shape == (40, 6, 2) and (lk <= 1).all()
    np.testing.assert_allclose(s.coords[:, CANONICAL.index("spine_mid")], xy[:, DLC_BODYPARTS.index("midbody")])
    np.testing.assert_allclose(s.conf[:, CANONICAL.index("head_center")], lk[:, DLC_BODYPARTS.index("headstage")])
    assert np.isfinite(s.coords[:, TRACKED]).all()
    assert np.isnan(s.coords[:, UNTRACKED]).all() and (s.conf[:, UNTRACKED] == 0).all()

    walk, groom, other = _ext("walking", "grooming", "other")
    csv = tmp_path / "MOUSE1_top" / "Behavior_Labels" / "CSV" / "MOUSE1-20220101T000000-010000_merged.csv"
    y = load_label_csv(csv)
    assert y[2] == walk and y[7] == groom and y[31] == other and y[36] == IGNORE  # overlap: grooming wins
    assert s.labels.tolist() == y.tolist()


def test_label_csv_ignores_frames_with_no_active_class(tmp_path):
    _labels(tmp_path / "l.csv", 12, {"Nesting": (2, 4)})
    y = load_label_csv(tmp_path / "l.csv")
    assert y[3] == _ext("other")[0]
    assert (y[[0, 1, 5, 11]] == IGNORE).all()
    sub = load_label_csv(tmp_path / "l.csv", classes=("walking", "grooming", "other"))
    assert sub[[3, 5]].tolist() == [2, IGNORE]  # a reduced label set keeps the classes it has


def test_smooth_sessions_only_fills_points_the_source_tracks(tmp_path):
    raw = _dlc_tree(tmp_path)
    temporal = TemporalLSTM(CANONICAL.num_keypoints, 2, hidden=16, layers=1)
    smooth = smooth_sessions(raw, temporal, conf_thresh=0.5, window=48)
    assert [s.animal for s in smooth] == [s.animal for s in raw]
    for before, after in zip(raw, smooth, strict=True):
        assert after.coords.shape == before.coords.shape and after.conf.shape == before.conf.shape
        assert after.fps == before.fps and after.labels.tolist() == before.labels.tolist()
        assert np.isfinite(after.coords[:, TRACKED]).all()  # dropouts and shaky detections are filled in
        assert np.isnan(after.coords[:, UNTRACKED]).all()  # anatomy the source never tracks is not invented
        assert (after.conf[:, UNTRACKED] == 0).all() and (after.conf[:, TRACKED] > 0).all()
        assert (after.conf <= 1).all()
        assert not np.allclose(after.coords[:, TRACKED], before.coords[:, TRACKED])
    pairs = track_list(smooth)
    assert len(pairs) == 2
    assert all(c.dtype == np.float32 and f.dtype == np.float32 for c, f in pairs)
    assert np.isnan(pairs[0][0][:, UNTRACKED]).all()


def _track(animal: str, t: int, labels: np.ndarray, conf_value: float = 0.9) -> Session:
    coords = np.repeat(np.arange(t, dtype=np.float64)[:, None, None], CANONICAL.num_keypoints * 2, -1)
    conf = np.full((t, CANONICAL.num_keypoints), conf_value)
    conf[:, 7] = 0.1  # one point the tracker never believes
    return Session(animal, "top", coords.reshape(t, CANONICAL.num_keypoints, 2), conf, labels, 15.0)


def test_behavior_windows_draw_only_long_enough_labeled_spans():
    walk = _ext("walking")[0]
    short = _track("short", 6, np.full(6, walk))
    with pytest.raises(ValueError):
        behavior_windows([short], 2, 8, np.random.default_rng(0))  # no 8-frame window fits
    unlabeled = _track("unlabeled", 40, np.full(40, IGNORE))
    with pytest.raises(ValueError):
        behavior_windows([unlabeled], 2, 8, np.random.default_rng(0), min_labeled=0.25)

    long = _track("long", 40, np.arange(40) % 3)
    long.coords[5] = np.nan  # one lost frame
    x, m, y = behavior_windows([short, long], 4, 8, np.random.default_rng(3), conf_thresh=0.5)
    assert x.shape == (4, 8, CANONICAL.num_keypoints, 2) and m.shape == (4, 8, CANONICAL.num_keypoints)
    assert y.shape == (4, 8)
    starts = x[:, 0, 0, 0].astype(int)
    assert ((starts >= 0) & (starts <= 31)).all()
    for i, t0 in enumerate(starts):
        want = np.ones((8, CANONICAL.num_keypoints), bool)
        want[:, 7] = False
        if t0 <= 5 < t0 + 8:
            want[5 - t0] = False
        assert (m[i] == want).all()  # low confidence and dropouts are masked out, never trusted
        np.testing.assert_allclose(x[i], np.nan_to_num(long.coords[t0:t0 + 8]), rtol=0, atol=1e-4)
        assert y[i].tolist() == [t % 3 for t in range(t0, t0 + 8)]


SHARP = Profile(psf_jitter=0.0, defocus_frac=(0.0, 0.0), motion_frac=(0.0, 0.0), downscale_to=None, mono=0.0,
                low_contrast=0.0, noise=(0.0, 0.0), jpeg=None)


def _disc(n: int = 64, r: float = 14.0) -> np.ndarray:
    img = np.full((n, n, 3), 210, np.uint8)
    y, x = np.mgrid[:n, :n]
    img[(x - n // 2) ** 2 + (y - n // 2) ** 2 <= r * r] = 40
    return img


class _Chooser:
    """A random source with fixed draws, so a stochastic branch is chosen deliberately."""

    def __init__(self, *values: float):
        self.values = list(values)

    def random(self) -> float:
        return self.values.pop(0)

    def uniform(self, low: float, high: float) -> float:
        return self.values.pop(0)

    def normal(self, mean, sd, size):
        return np.zeros(size)


def _high_freq(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    power = np.abs(np.fft.rfft2(g - g.mean())) ** 2
    fy, fx = np.fft.fftfreq(g.shape[0]), np.fft.rfftfreq(g.shape[1])
    return float((power * (np.hypot(fy[:, None], fx[None, :]) > 0.15)).sum())


def test_a_larger_psf_remeasurably_removes_high_frequency_detail():
    out = [degrade(_disc(), np.random.default_rng(0), animal_px=28, psf_px=p, profile=SHARP) for p in (1.0, 3.0, 6.0)]
    assert all(o.shape == (64, 64, 3) and o.dtype == np.uint8 for o in out)
    assert _high_freq(out[0]) > _high_freq(out[1]) > _high_freq(out[2]) > 0.0
    assert _high_freq(out[0]) < _high_freq(_disc())  # even a 1 px defocus disc takes contrast out of the edge
    widths = [edge_width(np.asarray([o]))[0] for o in out]
    assert widths[0] < widths[1] < widths[2] and widths[2] > 2.0 * widths[0]
    sharpness = [cv2.Laplacian(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for o in out]
    assert sharpness[0] > sharpness[1] > sharpness[2]
    assert edge_width(np.asarray([out[2]]))[1] == 0.0  # one patch has no spread

    same = degrade(_disc(), np.random.default_rng(0), 28, profile=replace(SHARP, psf_px=6.0))
    assert np.array_equal(same, out[2])  # psf_px and profile.psf_px are the same knob
    far = degrade(_disc(), np.random.default_rng(0), animal_px=400, psf_px=6.0, profile=SHARP)
    assert np.array_equal(far, out[2])  # a lens blurs a 28 px and a 400 px animal identically


def test_edge_width_needs_a_real_subject_and_reports_nothing_otherwise():
    assert edge_width(np.zeros((3, 32, 32, 3), np.uint8)) == (0.0, 0.0)  # no contrast
    assert edge_width(np.full((2, 64, 64, 3), 5, np.uint8)) == (0.0, 0.0)  # flat
    assert edge_width(np.asarray([_disc(13, 1.6)])) == (0.0, 0.0)  # a subject too small to measure
    w, spread = edge_width(np.asarray([_disc(), _disc(64, 20)]))
    assert w > 1.0 and spread >= 0.0


def test_camera_profile_channels_and_resolution_are_measurable():
    color = _disc().astype(np.int16)
    color[..., 2] = np.clip(color[..., 2] + 40, 0, 255)
    color = color.astype(np.uint8)
    assert not (color[..., 0] == color[..., 2]).all()  # a colour sensor to begin with
    mono = degrade(color, np.random.default_rng(4), 28, profile=replace(SHARP, mono=1.0))
    assert (mono[..., 0] == mono[..., 1]).all() and (mono[..., 1] == mono[..., 2]).all()

    low = degrade(_disc(), np.random.default_rng(4), 28, profile=replace(SHARP, low_contrast=1.0))
    assert low.std() < _disc().std()
    shrunk = Profile(**{**SHARP.to_dict(), "downscale_to": (8.0, 10.0)})
    # no defocus, downscale to a 9 px animal, no mono, no contrast change: only the resolution is touched
    tiny = degrade(_disc(), _Chooser(0.9, 0.1, 9.0, 0.9, 0.9), 28, profile=shrunk)
    sharp_levels = len(np.unique(cv2.cvtColor(_disc(), cv2.COLOR_BGR2GRAY)))
    assert len(np.unique(cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY))) > 2 * sharp_levels  # a resolution ramp
    assert np.array_equal(degrade(_disc(), np.random.default_rng(0), 28, strength=0.0, profile=SHARP), _disc())
    still_blurred = degrade(_disc(), np.random.default_rng(0), 28, strength=0.0, psf_px=3.0, profile=SHARP)
    assert not np.array_equal(still_blurred, _disc())  # defocus is a camera property, not a strength-scaled one


def test_degrade_is_reproducible_under_a_fixed_seed():
    noisy = Profile(noise=(2.0, 12.0), jpeg=(20.0, 60.0))
    a = degrade(_disc(), np.random.default_rng(11), 28, profile=noisy)
    b = degrade(_disc(), np.random.default_rng(11), 28, profile=noisy)
    c = degrade(_disc(), np.random.default_rng(12), 28, profile=noisy)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (64, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(degrade(_disc(), np.random.default_rng(5), 28, strength=0.0), _disc())


def test_camera_profiles_round_trip_and_reject_typos(tmp_path):
    assert Profile.from_dict(None) is DEFAULT_PROFILE and load_profile(None) is DEFAULT_PROFILE
    assert set(DEFAULT_PROFILE.to_dict()) == set(Profile.__dataclass_fields__)
    assert Profile.from_dict(DEFAULT_PROFILE.to_dict()) == DEFAULT_PROFILE
    assert Profile.from_dict({"downscale_to": [8.0, 12.0], "psf_px": 3}).downscale_to == (8.0, 12.0)
    with pytest.raises(ValueError, match="unknown camera profile keys"):
        Profile.from_dict({"psf_px": 2.0, "psf_pix": 2.0})
    (tmp_path / "audit.yaml").write_text("psf_px: 2.5\nnoise: [1.0, 3.0]\nmeasured:\n  psf_px: 2.4\n"
                                         "notes: n/a\nsource: animalpose data audit\n")
    p = load_profile(tmp_path / "audit.yaml")  # audit's documentation keys are skipped, a typo still fails
    assert (p.psf_px, p.noise) == (2.5, (1.0, 3.0))
    assert load_profile({"mono": 1.0}).mono == 1.0
    with pytest.raises(ValueError, match="unknown camera profile keys"):
        load_profile({"psf_px": 1.0, "nofocus": 1.0})


def test_motion_smear_is_a_fraction_of_the_per_frame_displacement():
    assert motion_smear(np.array([[3.0, 4.0], [10.0, 0.0]])) == pytest.approx(1.75)
    assert motion_smear(np.array([2.0]), exposure_fraction=1.0) == pytest.approx(2.0)
    assert motion_smear(np.array([10.0, 10.0, 10.0]), 0.2) == pytest.approx(2.0)
