"""Training data for the temporal model.

Two sources, one corruption model:

* synthetic: clean projected tracks from :mod:`animalpose.data.synthetic`
  (ground truth everywhere, including through occlusion);
* self-supervised on real tracker output: confident observations are the
  targets, and extra artificial gaps/noise are injected on top, so the model
  learns to reconstruct points it is not shown (masked modelling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animalpose.data.synthetic import SimulationPool, random_camera, simulate
from animalpose.skeleton import CANONICAL


@dataclass
class Corruption:
    jitter: tuple[float, float] = (0.005, 0.03)  # noise std, fraction of track spread
    point_drop: float = 0.05  # independent per-point dropout
    point_gaps: float = 0.02  # rate of per-keypoint contiguous gaps per frame
    point_gap_len: tuple[int, int] = (2, 20)
    animal_gaps: float = 0.6  # probability of >= 1 whole-animal gap per window
    animal_gap_len: tuple[int, int] = (5, 40)
    outliers: float = 0.01  # per point: jump to a random location, mid confidence
    swaps: float = 0.005  # per frame: left/right swap
    conf_threshold: float = 0.3


def corrupt(clean: np.ndarray, valid: np.ndarray, rng: np.random.Generator, c: Corruption = Corruption()):
    """``clean (T, K, D)``, ``valid (T, K)`` -> ``(x, conf, observed)``."""
    t, k, d = clean.shape
    x = clean.copy()
    vals = clean[valid]
    spread = float(np.std(vals)) if vals.size else 1.0
    x += rng.normal(0, rng.uniform(*c.jitter) * spread, size=x.shape)
    conf = rng.uniform(0.6, 1.0, size=(t, k))
    obs = valid.copy()
    obs &= rng.random((t, k)) > c.point_drop
    n_gaps = rng.poisson(c.point_gaps * t * k / 5)
    for _ in range(n_gaps):
        j, s = rng.integers(k), rng.integers(t)
        obs[s:s + rng.integers(*c.point_gap_len), j] = False
    if rng.random() < c.animal_gaps:
        for _ in range(rng.integers(1, 3)):
            s = rng.integers(t)
            obs[s:s + rng.integers(*c.animal_gap_len)] = False
    out = rng.random((t, k)) < c.outliers
    if out.any() and vals.size:
        lo, hi = vals.reshape(-1, d).min(0), vals.reshape(-1, d).max(0)
        x[out] = rng.uniform(lo, hi, size=(out.sum(), d))
        conf[out] = rng.uniform(0.3, 0.7, size=out.sum())
    flip = CANONICAL.flip_index
    for s in np.nonzero(rng.random(t) < c.swaps)[0]:
        x[s] = x[s, flip]
    conf[~obs] = rng.uniform(0.0, c.conf_threshold, size=(~obs).sum())
    x[~obs] = 0.0
    return x.astype(np.float32), conf.astype(np.float32), obs.astype(np.float32)


def synthetic_batch(batch: int, window: int, rng: np.random.Generator, dims: int = 2,
                    species=("mouse", "rat"), corruption: Corruption = Corruption(),
                    pool: SimulationPool | None = None):
    """Returns numpy arrays ``x, conf, observed, target, target_mask``.

    Windows come from ``pool`` when given, otherwise from fresh simulations.
    """
    xs, cs, os_, ts, ms = [], [], [], [], []
    for sim, a, s in sample_windows(batch, window, rng, species, pool):
        pts3 = sim.kpts3d[a, s:s + window]
        if dims == 3:
            clean, valid = pts3, np.ones(pts3.shape[:2], bool)
        else:
            cam = random_camera(sim.arena_cm, rng)
            clean = cam.project(pts3)
            valid = cam.in_image(clean)
        x, conf, obs = corrupt(clean, valid, rng, corruption)
        xs.append(x), cs.append(conf), os_.append(obs)
        ts.append(clean.astype(np.float32)), ms.append(np.ones(valid.shape, np.float32))
    return tuple(np.stack(v) for v in (xs, cs, os_, ts, ms))


def sample_windows(batch, window, rng, species, pool):
    out = []
    while len(out) < batch:
        if pool is not None:
            out.append(pool.window(rng, window))
            continue
        sim = simulate(int(rng.integers(1, 4)), window + 60, str(rng.choice(species)), seed=rng)
        out.extend((sim, a, int(rng.integers(30, 60))) for a in range(sim.n_animals))
    return out[:batch]


def self_supervised_batch(tracks: list[tuple[np.ndarray, np.ndarray]], batch: int, window: int,
                          rng: np.random.Generator, corruption: Corruption = Corruption()):
    """Masked-reconstruction batch from real tracks ``[(coords (T,K,D), conf (T,K)), ...]``.

    Targets are the confident observations; corruption hides additional ones.
    """
    xs, cs, os_, ts, ms = [], [], [], [], []
    while len(xs) < batch:
        coords, conf = tracks[int(rng.integers(len(tracks)))]
        if len(coords) < window:
            continue
        s = int(rng.integers(0, len(coords) - window + 1))
        clean, cf = coords[s:s + window], conf[s:s + window]
        target_mask = (cf > corruption.conf_threshold) & np.all(np.isfinite(clean), -1)
        clean = np.nan_to_num(clean)
        x, c, o = corrupt(clean, target_mask, rng, corruption)
        xs.append(x), cs.append(c), os_.append(o), ts.append(clean.astype(np.float32))
        ms.append(target_mask.astype(np.float32))
    return tuple(np.stack(v) for v in (xs, cs, os_, ts, ms))
