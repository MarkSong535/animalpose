"""Synthetic training windows for the behavior classifier.

Pose sequences come from the simulator, projected through random cameras
(any elevation/azimuth) for 2D models so the classifier does not overfit a
viewpoint, and corrupted with the same noise/gap model used for the temporal
network so it tolerates real tracking output.
"""

from __future__ import annotations

import numpy as np

from animalpose.data.synthetic import SimulationPool, random_camera
from animalpose.temporal.data import Corruption, corrupt, sample_windows

_MILD = Corruption(jitter=(0.003, 0.02), point_drop=0.05, animal_gaps=0.2, animal_gap_len=(3, 10))


def synthetic_batch(batch: int, window: int, rng: np.random.Generator, dims: int = 2,
                    species=("mouse", "rat"), corruption: Corruption = _MILD, pool: SimulationPool | None = None):
    """Returns ``x (B,T,K,D)``, ``mask (B,T,K)``, ``labels (B,T)``."""
    xs, ms, ys = [], [], []
    for sim, a, s in sample_windows(batch, window, rng, species, pool):
        pts3 = sim.kpts3d[a, s:s + window]
        if dims == 3:
            clean, valid = pts3, np.ones(pts3.shape[:2], bool)
        else:
            cam = random_camera(sim.arena_cm, rng)
            clean = cam.project(pts3)
            valid = cam.in_image(clean)
        x, _, obs = corrupt(clean, valid, rng, corruption)
        xs.append(x), ms.append(obs), ys.append(sim.behavior[a, s:s + window])
    return np.stack(xs), np.stack(ms), np.stack(ys).astype(np.int64)
