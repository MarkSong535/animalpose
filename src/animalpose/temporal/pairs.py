"""Temporal left/right identity for bilateral keypoint pairs.

In a side view the per-frame evidence for which paw is the left one comes from
subtle cues (the near paw is brighter and occludes the far one) and is weak
whenever the paws overlap or blur. Over time, however, each paw moves
continuously. For every pair this module runs a two-state Viterbi (keep the
decoded labels or swap them) per track: the emission cost is the pose
network's left/right margin (:func:`animalpose.pose.model.decode_pairs`), the
transition cost is how far each labelled paw would have to jump between
consecutive frames. Confident frames therefore carry the identity through
ambiguous ones, while a genuine crossing (both labellings equally smooth) is
left to the per-frame evidence.
"""

from __future__ import annotations

import numpy as np


def resolve_pair_identity(kpts: np.ndarray, margin: np.ndarray, pairs, scale: np.ndarray | float,
                          motion_frac: float = 0.1, margin_weight: float = 1.0):
    """Relabel bilateral pairs consistently over one track.

    ``kpts (T, K, 2)`` decoded keypoints (NaN where unobserved), ``margin (T, P)``
    per-frame left/right evidence for the decoded labelling (positive = keep),
    ``pairs`` ``(P, 2)`` left/right keypoint indices, ``scale`` animal size in
    px (scalar or ``(T,)``). A labelled paw jumping ``motion_frac * scale``
    between frames costs as much as one unit of margin.
    Returns corrected keypoints and a boolean ``(T, P)`` array of swapped frames.
    """
    kpts = np.array(kpts, dtype=np.float64, copy=True)
    t_len = len(kpts)
    scale = np.broadcast_to(np.asarray(scale, np.float64), (t_len,))
    swapped = np.zeros((t_len, len(pairs)), bool)
    if t_len == 0:
        return kpts.astype(np.float32), swapped
    for p, (lft, rgt) in enumerate(pairs):
        a, b = kpts[:, lft], kpts[:, rgt]
        m = np.nan_to_num(np.asarray(margin[:, p], np.float64)) * margin_weight
        emit = np.stack([-m / 2, m / 2], 1)  # cost of state 0 (keep) / 1 (swap)
        # Labels keep their paws (same) or trade them (cross) between the last observed frame
        # and frame t; the allowed motion grows with the number of frames skipped.
        ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        last = np.maximum.accumulate(np.where(ok, np.arange(t_len), -1))
        prev = np.maximum(last[:-1], 0)
        gap = np.where(last[:-1] >= 0, np.arange(1, t_len) - prev, 1)
        unit = motion_frac * np.maximum(scale[1:], 1e-6) * gap
        pa, pb = a[prev], b[prev]
        same = np.linalg.norm(a[1:] - pa, axis=-1) + np.linalg.norm(b[1:] - pb, axis=-1)
        cross = np.linalg.norm(a[1:] - pb, axis=-1) + np.linalg.norm(b[1:] - pa, axis=-1)
        same, cross = np.nan_to_num(same / unit), np.nan_to_num(cross / unit)
        # An unobserved frame inherits the previous state, so the chain bridges it.
        cross[~ok[1:]] = 1e9
        cost = emit[0].copy()
        back = np.zeros((t_len, 2), np.int64)
        for t in range(1, t_len):
            tot = cost[:, None] + np.array([[same[t - 1], cross[t - 1]], [cross[t - 1], same[t - 1]]])
            back[t] = tot.argmin(0)
            cost = tot.min(0) + emit[t]
        s = int(cost.argmin())
        for t in range(t_len - 1, -1, -1):
            swapped[t, p] = s == 1
            s = back[t, s]
        sw = swapped[:, p]
        kpts[sw, lft], kpts[sw, rgt] = b[sw].copy(), a[sw].copy()
    return kpts.astype(np.float32), swapped
