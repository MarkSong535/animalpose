"""Confidence-weighted, outlier-robust multi-view triangulation."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np

from animalpose.multiview.camera import Camera


def triangulate_dlt(cameras: list[Camera], uv: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Linear triangulation of points seen in several views.

    ``uv (V, N, 2)`` pixels (NaN = unseen), ``weights (V, N)`` (e.g. keypoint
    confidence). Points are undistorted first. Returns ``(N, 3)``; NaN where
    fewer than two views are available.
    """
    v, n = uv.shape[:2]
    w = np.ones((v, n)) if weights is None else np.nan_to_num(np.asarray(weights, float))
    seen = np.isfinite(uv).all(-1) & (w > 0)
    xn = np.stack([c.undistort_points(np.nan_to_num(uv[i])) for i, c in enumerate(cameras)])  # (V,N,2)
    Ps = np.stack([np.hstack([c.R, c.t[:, None]]) for c in cameras])  # normalised projections
    A = np.zeros((n, 2 * v, 4))
    A[:, 0::2] = xn[..., 0].T[..., None] * Ps[None, :, 2] - Ps[None, :, 0]
    A[:, 1::2] = xn[..., 1].T[..., None] * Ps[None, :, 2] - Ps[None, :, 1]
    ww = (w * seen).T.repeat(2, axis=1)[..., None]
    _, _, vt = np.linalg.svd(A * ww)
    Xh = vt[:, -1]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:]) < 1e-12, 1e-12, Xh[:, 3:])
    X[seen.sum(0) < 2] = np.nan
    return X


def triangulate_robust(cameras: list[Camera], uv: np.ndarray, conf: np.ndarray | None = None,
                       conf_thresh: float = 0.3, reproj_thresh: float = 10.0):
    """Triangulate with view selection.

    For each point, views below ``conf_thresh`` are discarded; if the
    all-view solution has a view with reprojection error above
    ``reproj_thresh`` px, every view subset (size >= 2) is tried and the
    largest subset with all errors under the threshold (lowest error on ties)
    is kept.

    Returns ``X (N, 3)``, ``err (N,)`` mean reprojection error over used views,
    ``used (V, N)`` boolean.
    """
    v, n = uv.shape[:2]
    conf = np.ones((v, n)) if conf is None else np.nan_to_num(np.asarray(conf, float))
    ok = np.isfinite(uv).all(-1) & (conf >= conf_thresh)
    X = triangulate_dlt(cameras, np.where(ok[..., None], uv, np.nan), np.where(ok, conf, 0))
    used = ok.copy()
    err = _errors(cameras, X, uv, used)
    bad = np.nanmax(np.where(used, _per_view(cameras, X, uv), -np.inf), axis=0) > reproj_thresh
    for j in np.nonzero(bad & (ok.sum(0) > 2))[0]:
        views = np.nonzero(ok[:, j])[0]
        best = None
        for size in range(len(views) - 1, 1, -1):
            for subset in combinations(views, size):
                sub = list(subset)
                Xj = triangulate_dlt([cameras[i] for i in sub], uv[sub, j:j + 1], conf[sub, j:j + 1])
                e = np.array([np.linalg.norm(cameras[i].project(Xj[0]) - uv[i, j]) for i in sub])
                if np.all(e <= reproj_thresh) and (best is None or e.mean() < best[1]):
                    best = (sub, e.mean(), Xj[0])
            if best is not None:
                break
        if best is not None:
            X[j] = best[2]
            used[:, j] = False
            used[best[0], j] = True
            err[j] = best[1]
    return X, err, used


def _per_view(cameras, X, uv):
    return np.stack([np.linalg.norm(c.project(np.nan_to_num(X)) - uv[i], axis=-1) for i, c in enumerate(cameras)])


def _errors(cameras, X, uv, used):
    e = _per_view(cameras, X, uv)
    e = np.where(used, e, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # points seen by no view -> NaN
        return np.nanmean(e, axis=0)
