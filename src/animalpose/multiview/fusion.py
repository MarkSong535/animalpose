"""Cross-view identity fusion.

Each camera is tracked independently, so the same animal has a different
track id per view. Two tracks in different views belong to the same animal
when their keypoints are epipolar-consistent over the frames both are seen.
We score every cross-view track pair by the median symmetric epipolar
distance (converted to approximate pixels) and build global identities by
constrained agglomerative clustering (:func:`fuse_identities`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animalpose.multiview.camera import Camera, essential_matrix
from animalpose.multiview.symmetry import resolve_left_right_cameras
from animalpose.multiview.triangulation import triangulate_robust


@dataclass
class ViewTrack:
    """One single-view track: dense over ``frames``."""

    view: int
    track_id: int
    frames: np.ndarray  # (T,) int
    keypoints: np.ndarray  # (T, K, 2) pixels, NaN where missing
    conf: np.ndarray  # (T, K)


def epipolar_cost(a: ViewTrack, b: ViewTrack, cam_a: Camera, cam_b: Camera, conf_thresh: float = 0.3,
                  min_points: int = 10) -> float:
    """Median symmetric epipolar distance (px) over co-visible keypoints; inf if too few."""
    common, ia, ib = np.intersect1d(a.frames, b.frames, return_indices=True)
    if len(common) == 0:
        return np.inf
    pa, pb = a.keypoints[ia].reshape(-1, 2), b.keypoints[ib].reshape(-1, 2)
    ok = (a.conf[ia].ravel() > conf_thresh) & (b.conf[ib].ravel() > conf_thresh)
    ok &= np.isfinite(pa).all(-1) & np.isfinite(pb).all(-1)
    if ok.sum() < min_points:
        return np.inf
    xa = np.c_[cam_a.undistort_points(pa[ok]), np.ones(ok.sum())]
    xb = np.c_[cam_b.undistort_points(pb[ok]), np.ones(ok.sum())]
    E = essential_matrix(cam_a, cam_b)
    la = xa @ E.T  # epipolar lines in b
    lb = xb @ E  # epipolar lines in a
    num = np.abs(np.sum(xb * la, axis=1))
    d_b = num / np.linalg.norm(la[:, :2], axis=1) * cam_b.K[0, 0]
    d_a = num / np.linalg.norm(lb[:, :2], axis=1) * cam_a.K[0, 0]
    d = 0.5 * (d_a + d_b)
    return float(np.median(d))


def _overlap(a: ViewTrack, b: ViewTrack, tolerance: int = 2) -> bool:
    return len(np.intersect1d(a.frames, b.frames)) > tolerance


def fuse_identities(tracks: list[ViewTrack], cameras: list[Camera], max_cost: float = 15.0,
                    max_animals: int | None = None) -> dict[tuple[int, int], int]:
    """Assign a global id to every ``(view, track_id)`` by agglomerative clustering.

    Every single-view track starts as its own cluster. The two clusters with
    the lowest average cross-view epipolar cost are merged repeatedly, as long
    as the cost is below ``max_cost`` and the merge keeps tracks *of the same
    view* temporally disjoint. Allowing several disjoint tracks per view means
    a track that fragmented in one camera (occlusion, ID switch) is stitched
    back together through the other cameras instead of becoming a new animal.
    With ``max_animals``, merging continues past ``max_cost`` (still subject
    to the disjointness constraint) until at most that many identities remain.
    """
    n = len(tracks)
    pair = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracks[i], tracks[j]
            if a.view != b.view:
                pair[i, j] = pair[j, i] = epipolar_cost(a, b, cameras[a.view], cameras[b.view])
    clusters: list[list[int]] = [[i] for i in range(n)]

    def linkage(ca: list[int], cb: list[int]) -> float:
        for i in ca:
            for j in cb:
                if tracks[i].view == tracks[j].view and _overlap(tracks[i], tracks[j]):
                    return np.inf
        costs = pair[np.ix_(ca, cb)]
        finite = costs[np.isfinite(costs)]
        return float(finite.mean()) if finite.size else np.inf

    while len(clusters) > 1:
        best, bi, bj = np.inf, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c = linkage(clusters[i], clusters[j])
                if c < best:
                    best, bi, bj = c, i, j
        forced = max_animals is not None and len(clusters) > max_animals
        if bi < 0 or not np.isfinite(best) or (best > max_cost and not forced):
            break
        clusters[bi] += clusters.pop(bj)
    clusters.sort(key=lambda c: -sum(len(tracks[i].frames) for i in c))
    return {(tracks[i].view, tracks[i].track_id): g for g, c in enumerate(clusters) for i in c}


def triangulate_identities(tracks: list[ViewTrack], cameras: list[Camera], ids: dict[tuple[int, int], int],
                           n_frames: int, conf_thresh: float = 0.3, reproj_thresh: float = 10.0,
                           resolve_left_right: bool = False):
    """Per global identity: ``(T, K, 3)`` 3D keypoints, ``(T, K)`` reprojection error and
    number of views used. Frames/points with < 2 views are NaN. ``resolve_left_right``
    first makes bilateral labels epipolar-consistent across views
    (:mod:`animalpose.multiview.symmetry`); robust triangulation already rejects a
    single swapped view, so this mainly matters for two-camera rigs."""
    k = tracks[0].keypoints.shape[1]
    v = len(cameras)
    out = {}
    for gid in sorted(set(ids.values())):
        uv = np.full((v, n_frames, k, 2), np.nan)
        cf = np.zeros((v, n_frames, k))
        for t in tracks:
            if ids[(t.view, t.track_id)] == gid:
                uv[t.view, t.frames] = t.keypoints
                cf[t.view, t.frames] = np.nan_to_num(t.conf)
        if resolve_left_right:
            for f in range(n_frames):
                seen = np.isfinite(uv[:, f]).all(-1).any(-1)
                if seen.sum() >= 2:
                    idx = np.nonzero(seen)[0]
                    fixed, fixed_conf, _ = resolve_left_right_cameras(np.nan_to_num(uv[idx, f]), cf[idx, f],
                                                                      [cameras[i] for i in idx], conf_thresh=conf_thresh)
                    uv[idx, f] = np.where(np.isfinite(uv[idx, f]), fixed, np.nan)
                    cf[idx, f] = fixed_conf
        X, err, used = triangulate_robust(cameras, uv.reshape(v, -1, 2), cf.reshape(v, -1), conf_thresh, reproj_thresh)
        out[gid] = {"keypoints3d": X.reshape(n_frames, k, 3), "reproj_error": err.reshape(n_frames, k),
                    "n_views": used.sum(0).reshape(n_frames, k)}
    return out
