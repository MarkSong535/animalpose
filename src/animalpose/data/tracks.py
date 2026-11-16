"""Real keypoint tracks with frame-wise behavior labels (DeepLabCut output +
BORIS-style label CSVs), for training/evaluating the temporal and behavior
models on real animals without images.

The layout is a home-cage DeepLabCut tree (``<animal>_<view>/DLC/*.h5``
and ``<animal>_<view>/Behavior_Labels/CSV/*_merged.csv``); that data is
private and is only read in place. Any DLC project with per-frame labels can
be loaded with :func:`load_dlc_h5` and :func:`load_label_csv`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animalpose.labels import BEHAVIORS_EXT
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS

# Source label -> extended behavior class. Earlier entries win when several
# labels are active in the same frame. Frames without any label are ignored
# (-100): unlabelled home-cage time is mostly, but not reliably, inactivity.
LABEL_PRIORITY: list[tuple[str, str]] = [
    ("Grooming", "grooming"), ("Chewing", "eating"), ("Drinking", "eating"),
    ("Rearing", "rearing"), ("Scaling a Wall", "rearing"), ("Locomotion", "walking"),
    ("Digging", "other"), ("Nesting", "other"), ("Playing w/ Ball", "other"), ("Playing w/ Box", "other"),
    ("Riding the Ball", "other"), ("Riding the Box", "other"), ("Porthole Interaction", "other"),
    ("Stretching", "other"), ("Microarousal", "other"),
]
IGNORE = -100


def load_dlc_h5(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """DeepLabCut prediction file -> ``(bodyparts, xy (T, J, 2), likelihood (T, J))``."""
    import pandas as pd

    df = pd.read_hdf(path)
    bps = list(dict.fromkeys(df.columns.get_level_values(-2)))
    coords = list(df.columns.get_level_values(-1))
    if coords != ["x", "y", "likelihood"] * len(bps):
        raise ValueError(f"unexpected DLC column layout in {path}")
    arr = df.to_numpy(float).reshape(len(df), len(bps), 3)
    return bps, arr[..., :2], arr[..., 2]


def load_label_csv(path: str | Path, classes=BEHAVIORS_EXT) -> np.ndarray:
    """Multi-hot behavior CSV -> per-frame class index (``IGNORE`` where unlabeled)."""
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    out = np.full(len(df), IGNORE, dtype=np.int64)
    for src, dst in reversed(LABEL_PRIORITY):  # higher priority overwrites
        if src in df.columns:
            out[df[src].to_numpy() > 0] = classes.index(dst)
    return out


@dataclass
class Session:
    animal: str
    view: str
    coords: np.ndarray  # (T, K, 2) canonical, NaN where undefined
    conf: np.ndarray  # (T, K)
    labels: np.ndarray  # (T,)
    fps: float


def _key(name: str) -> str:
    m = re.search(r"(\d{8}T\d{6}-\d{6})", name)
    return m.group(1) if m else name


def load_sessions(root: str | Path, animals: list[str] | None = None, target_fps: float = 15.0,
                  mapping: str = "homecage_dlc") -> list[Session]:
    """Pair DLC files with label CSVs by recording timestamp; resample to ``target_fps``."""
    sessions = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir() or "_" not in d.name:
            continue
        animal, view = d.name.split("_", 1)
        if animals is not None and animal not in animals:
            continue
        h5s = {_key(p.name): p for p in (d / "DLC").glob("*.h5")} if (d / "DLC").exists() else {}
        for csv in sorted((d / "Behavior_Labels" / "CSV").glob("*.csv")) if (d / "Behavior_Labels").exists() else []:
            h5 = h5s.get(_key(csv.name))
            if h5 is None:
                continue
            bps, xy, lk = load_dlc_h5(h5)
            labels = load_label_csv(csv)
            t = min(len(xy), len(labels))
            fps = 30.0 if t > 80_000 else 15.0  # hour-long files: 108k frames at 30 fps, 54k at 15 fps
            step = max(1, int(round(fps / target_fps)))
            coords = np.full((t, CANONICAL.num_keypoints, 2), np.nan)
            conf = np.zeros((t, CANONICAL.num_keypoints))
            for j, bp in enumerate(bps):
                c = DATASET_MAPPINGS[mapping].get(bp)
                if c is not None:
                    coords[:, CANONICAL.index(c)] = xy[:t, j]
                    conf[:, CANONICAL.index(c)] = lk[:t, j]
            sessions.append(Session(animal, view.lower(), coords[::step], conf[::step], labels[:t:step], fps / step))
    return sessions


def smooth_sessions(sessions: list[Session], temporal, conf_thresh: float = 0.6, window: int = 64) -> list[Session]:
    """Clean tracks with a temporal model before behavior classification.

    Keypoints below ``conf_thresh`` (DLC's nose is rarely confident) and
    dropouts are imputed, jitter is smoothed, and confidence is replaced by
    the model's calibrated confidence. Only keypoints the source tracks at all
    are kept.
    """
    from animalpose.temporal.model import smooth_sequence

    out = []
    for s in sessions:
        tracked = np.isfinite(s.coords).all(-1).any(0)
        observed = (s.conf >= conf_thresh) & np.isfinite(s.coords).all(-1)
        coords, conf, _ = smooth_sequence(temporal, s.coords, s.conf, observed, window)
        coords[:, ~tracked], conf[:, ~tracked] = np.nan, 0.0
        out.append(Session(s.animal, s.view, coords.astype(np.float64), conf.astype(np.float64), s.labels, s.fps))
    return out


def behavior_windows(sessions: list[Session], batch: int, window: int, rng: np.random.Generator,
                     conf_thresh: float = 0.6, min_labeled: float = 0.25):
    """Random labeled windows -> ``x (B,T,K,2)``, ``mask (B,T,K)``, ``labels (B,T)``."""
    xs, ms, ys = [], [], []
    tries = 0
    while len(xs) < batch and tries < batch * 50:
        tries += 1
        s = sessions[int(rng.integers(len(sessions)))]
        if len(s.labels) <= window:
            continue
        t0 = int(rng.integers(0, len(s.labels) - window))
        y = s.labels[t0:t0 + window]
        if (y != IGNORE).mean() < min_labeled:
            continue
        m = (s.conf[t0:t0 + window] >= conf_thresh) & np.isfinite(s.coords[t0:t0 + window]).all(-1)
        xs.append(np.nan_to_num(s.coords[t0:t0 + window]).astype(np.float32))
        ms.append(m.astype(np.float32))
        ys.append(y)
    return np.stack(xs), np.stack(ms), np.stack(ys)


def track_list(sessions: list[Session]) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(coords, conf)`` pairs for :func:`animalpose.temporal.data.self_supervised_batch`."""
    return [(s.coords.astype(np.float32), s.conf.astype(np.float32)) for s in sessions]
