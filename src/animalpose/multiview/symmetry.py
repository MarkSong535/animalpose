"""Multi-view left/right disambiguation.

A single side view often cannot tell the near paw from the far one, so
single-view pose swaps bilateral keypoints. Another synchronised view usually
can. For each bilateral pair (left_X, right_X) we decide, view by view,
whether keeping or swapping the pair is more consistent with the views
already decided, using symmetric epipolar distance. Views are processed in
order of how confidently they see the pair, so the clearest view anchors the
labelling.

Geometry is given as fundamental matrices in pixel coordinates,
``F[(i, j)]`` with ``x_j^T F x_i = 0``. :func:`fundamental_from_cameras`
builds them from calibrated pinhole cameras; a planar mirror rig whose two
views share the horizontal axis uses :data:`F_SHARED_X`.
"""

from __future__ import annotations

import numpy as np

from animalpose.multiview.camera import Camera, essential_matrix
from animalpose.skeleton import CANONICAL

# Corresponding points have equal x (vertical epipolar lines): x_b^T F x_a = u_a - u_b.
F_SHARED_X = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def fundamental_from_cameras(cam_a: Camera, cam_b: Camera) -> np.ndarray:
    """Pixel-space F for undistorted points (undistort first for lenses with distortion)."""
    return np.linalg.inv(cam_b.K).T @ essential_matrix(cam_a, cam_b) @ np.linalg.inv(cam_a.K)


def epipolar_distance(F: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Symmetric point-to-epipolar-line distance (px) for ``(..., 2)`` point arrays."""
    xa = np.concatenate([pa, np.ones(pa.shape[:-1] + (1,))], -1)
    xb = np.concatenate([pb, np.ones(pb.shape[:-1] + (1,))], -1)
    la = xa @ F.T  # lines in b
    lb = xb @ F  # lines in a
    num = np.abs(np.sum(xb * la, -1))
    return 0.5 * (num / np.maximum(np.linalg.norm(la[..., :2], axis=-1), 1e-12)
                  + num / np.maximum(np.linalg.norm(lb[..., :2], axis=-1), 1e-12))


def bilateral_pairs(skeleton=CANONICAL) -> list[tuple[int, int]]:
    flip = skeleton.flip_index
    return [(i, int(flip[i])) for i in range(skeleton.num_keypoints) if skeleton.keypoints[i].side == "left"]


def resolve_left_right(kpts: np.ndarray, conf: np.ndarray, F: dict, conf_thresh: float = 0.3,
                       margin: float = 0.0, pairs: list[tuple[int, int]] | None = None,
                       view_priority: list[int] | None = None):
    """Make bilateral labels consistent across views.

    ``kpts (V, K, 2)``, ``conf (V, K)`` for one animal at one instant.
    ``F[(i, j)]`` fundamental matrices for the view pairs that have geometry.
    ``margin`` (px) is the minimum epipolar improvement required to swap.
    ``view_priority`` fixes the anchoring order (most trusted view first)
    instead of ordering by confidence, e.g. when one view is known to resolve
    left/right reliably (a bottom or top view) and another does not (a side view).
    Returns corrected copies and a boolean ``(V, n_pairs)`` array of swaps.
    """
    kpts, conf = kpts.copy(), conf.copy()
    pairs = pairs or bilateral_pairs()
    v = kpts.shape[0]
    swapped = np.zeros((v, len(pairs)), bool)

    def fm(i, j):
        if (i, j) in F:
            return F[(i, j)]
        if (j, i) in F:
            return F[(j, i)].T
        return None

    for p, (lft, rgt) in enumerate(pairs):
        seen = (conf[:, lft] >= conf_thresh) & (conf[:, rgt] >= conf_thresh)
        order = view_priority if view_priority is not None else np.argsort(-np.minimum(conf[:, lft], conf[:, rgt]))
        decided: list[int] = []
        for j in order:
            if not seen[j]:
                continue
            keep = swap = 0.0
            for i in decided:
                f = fm(i, j)
                if f is None:
                    continue
                keep += epipolar_distance(f, kpts[i, [lft, rgt]], kpts[j, [lft, rgt]]).sum()
                swap += epipolar_distance(f, kpts[i, [lft, rgt]], kpts[j, [rgt, lft]]).sum()
            if decided and swap + margin < keep:
                kpts[j, [lft, rgt]] = kpts[j, [rgt, lft]]
                conf[j, [lft, rgt]] = conf[j, [rgt, lft]]
                swapped[j, p] = True
            decided.append(int(j))
    return kpts, conf, swapped


def resolve_left_right_cameras(kpts: np.ndarray, conf: np.ndarray, cameras: list[Camera], **kw):
    """:func:`resolve_left_right` for calibrated (possibly distorted) cameras."""
    und = np.stack([c.undistort_points(np.nan_to_num(k)) @ c.K[:2, :2].T + c.K[:2, 2] for c, k in zip(cameras, kpts)])
    F = {(i, j): fundamental_from_cameras(cameras[i], cameras[j])
         for i in range(len(cameras)) for j in range(i + 1, len(cameras))}
    _, conf_out, swapped = resolve_left_right(und, conf, F, **kw)
    out = kpts.copy()
    for p, (lft, rgt) in enumerate(kw.get("pairs") or bilateral_pairs()):
        for v in np.nonzero(swapped[:, p])[0]:
            out[v, [lft, rgt]] = out[v, [rgt, lft]]
    return out, conf_out, swapped
