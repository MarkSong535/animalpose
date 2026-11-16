"""Overlay drawing for inspection videos."""

from __future__ import annotations

import cv2
import numpy as np

from animalpose.skeleton import CANONICAL

_PALETTE = [(66, 135, 245), (245, 66, 102), (66, 245, 135), (245, 200, 66), (190, 66, 245), (66, 230, 245)]


def color(i: int) -> tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def draw_pose(image: np.ndarray, kpts: np.ndarray, conf: np.ndarray | None = None, imputed: np.ndarray | None = None,
              track_id: int = 0, label: str = "", thresh: float = 0.3) -> np.ndarray:
    """Skeleton overlay; imputed (predicted, unobserved) points are drawn hollow."""
    col = color(track_id)
    ok = np.isfinite(kpts).all(-1)
    if conf is not None:
        ok &= np.nan_to_num(conf) >= thresh
    p = np.nan_to_num(kpts).round().astype(int)
    for a, b in CANONICAL.edge_index:
        if ok[a] and ok[b]:
            cv2.line(image, tuple(p[a]), tuple(p[b]), col, 1, cv2.LINE_AA)
    for k in np.nonzero(ok)[0]:
        hollow = imputed is not None and imputed[k] > 0.5
        cv2.circle(image, tuple(p[k]), 2, col, 1 if hollow else -1, cv2.LINE_AA)
    if ok.any():
        x, y = p[ok].min(0)
        cv2.putText(image, f"{track_id} {label}".strip(), (int(x), max(10, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, col, 1, cv2.LINE_AA)
    return image


def draw_frame(image: np.ndarray, result, t: int) -> np.ndarray:
    """Draw every track of a :class:`~animalpose.pipeline.VideoResult` at frame ``t``."""
    out = image.copy()
    for tid, tr in result.tracks.items():
        i = t - int(tr.frames[0])
        if not 0 <= i < len(tr.frames):
            continue
        conf = tr.smoothed_conf if tr.smoothed_conf is not None else tr.conf
        imp = tr.imputed[i] if tr.imputed is not None else None
        label = result.behaviors[tr.behavior[i]] if tr.behavior is not None else ""
        draw_pose(out, tr.best[i], conf[i], imp, tid, label)
    return out
