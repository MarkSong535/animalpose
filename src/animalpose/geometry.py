"""Box and affine-crop helpers shared by datasets, inference and tracking.

Crops are plain 2x3 affine matrices so the same preprocessing can be reproduced
in C++ with ``cv::warpAffine``.
"""

from __future__ import annotations

import cv2
import numpy as np


def box_xyxy_to_cs(box: np.ndarray, aspect: float = 1.0, padding: float = 1.25) -> tuple[np.ndarray, float]:
    """Box -> (center, side) of the square (for ``aspect == 1``) crop region."""
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float64)
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, float(w * padding)


def crop_affine(center: np.ndarray, side: float, out_size: int, rotation: float = 0.0) -> np.ndarray:
    """2x3 matrix mapping image pixels to an ``out_size`` square crop.

    ``rotation`` (radians) rotates the crop content counter-clockwise; used for
    augmentation and optional heading-aligned inference.
    """
    s = out_size / side
    cos, sin = np.cos(rotation) * s, np.sin(rotation) * s
    half = (out_size - 1) / 2
    cx, cy = center
    return np.array(
        [[cos, sin, half - cos * cx - sin * cy], [-sin, cos, half + sin * cx - cos * cy]], dtype=np.float64
    )


def invert_affine(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(m, dtype=np.float64))


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:, :2].T + m[:, 2]


def warp(image: np.ndarray, m: np.ndarray, out_size: int) -> np.ndarray:
    return cv2.warpAffine(image, m, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=-1)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=-1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def keypoints_to_box(kpts: np.ndarray, valid: np.ndarray | None = None, pad: float = 0.1) -> np.ndarray | None:
    """Tight xyxy box around valid keypoints, expanded by ``pad`` of its size."""
    kpts = np.asarray(kpts, dtype=np.float64)
    if valid is None:
        valid = np.all(np.isfinite(kpts), axis=-1)
    if not np.any(valid):
        return None
    p = kpts[valid]
    lo, hi = p.min(0), p.max(0)
    d = np.maximum(hi - lo, 4.0) * pad
    return np.concatenate([lo - d, hi + d])
