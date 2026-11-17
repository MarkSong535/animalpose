"""Camera drift: how far each camera has turned since a reference picture of its scene, measured from
the videos themselves, and the camera model corrected by it (``rust/src/multiview/drift.rs``).

A calibration holds while the cameras stay put; a knocked mount turns a camera by a fraction of a
degree and every 3D point it helps triangulate moves with it. Each camera's reference is the median
background of the first session fused with the camera model (``<model>.references/<camera>.png``).
Every later session's median background is registered against it: a coarse-to-fine alignment over
the camera's rotation through its own ``K`` and lens distortion, with the lighting's gain and offset
solved alongside and Huber weights scaled by the residuals' own spread, so a moved object counts
less than the walls and the desk. The rotation is applied when it makes the two pictures agree
better than the uncorrected camera did; ``DriftConfig`` sets how the registration runs, never how
far a camera may move. A camera carried to another place needs a new calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, rodrigues, rotation_to_rvec


@dataclass
class DriftConfig:
    """The ``drift:`` block of a pipeline config."""

    enabled: bool = True  # correct multiview cameras for drift
    coarsest: int = 64  # the smallest pyramid level is about this many pixels across
    iterations: int = 30  # Gauss-Newton steps per level at most
    samples: int = 40000  # pixels per level at most (an even subsample above it)
    huber: float = 1.345  # Huber's constant in robust standard deviations (95% Gaussian efficiency)


@dataclass
class _Level:
    scale: float  # full-resolution pixels per pixel of this level
    img: np.ndarray
    gx: np.ndarray
    gy: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.img.shape

    def at(self, u: np.ndarray) -> np.ndarray:
        """Full-resolution pixel positions in this level's pixels."""
        return (u + 0.5) / self.scale - 0.5

    def sample(self, p: np.ndarray):
        """Bilinear value and gradient at positions ``(N, 2)`` inside a one-pixel margin."""
        h, w = self.shape
        x, y = p[:, 0], p[:, 1]
        ok = (x >= 1) & (y >= 1) & (x < w - 2) & (y < h - 2)
        x0 = np.floor(np.where(ok, x, 1)).astype(int)
        y0 = np.floor(np.where(ok, y, 1)).astype(int)
        fx, fy = np.where(ok, x, 1) - x0, np.where(ok, y, 1) - y0

        def lerp(v):
            top = v[y0, x0] * (1 - fx) + v[y0, x0 + 1] * fx
            bottom = v[y0 + 1, x0] * (1 - fx) + v[y0 + 1, x0 + 1] * fx
            return top * (1 - fy) + bottom * fy

        return ok, lerp(self.img), lerp(self.gx), lerp(self.gy)


def _blur(img: np.ndarray) -> np.ndarray:
    """``[1 2 1] / 4`` in both directions, edges clamped."""
    p = np.pad(img, ((0, 0), (1, 1)), mode="edge")
    img = 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]
    p = np.pad(img, ((1, 1), (0, 0)), mode="edge")
    return 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]


def _level(img: np.ndarray, scale: float) -> _Level:
    img = _blur(img)
    gx, gy = np.zeros_like(img), np.zeros_like(img)
    gx[1:-1, 1:-1] = 0.5 * (img[1:-1, 2:] - img[1:-1, :-2])
    gy[1:-1, 1:-1] = 0.5 * (img[2:, 1:-1] - img[:-2, 1:-1])
    return _Level(scale, img, gx, gy)


def _pyramid(gray: np.ndarray, coarsest: int) -> list[_Level]:
    """Levels from full resolution down to about ``coarsest`` pixels across."""
    img = np.asarray(gray, dtype=np.float64)
    levels = [_level(img, 1.0)]
    while img.shape[1] // 2 >= coarsest and img.shape[0] // 2 >= coarsest // 2:
        h2, w2 = img.shape[0] // 2, img.shape[1] // 2
        img = 0.25 * (img[0:2 * h2:2, 0:2 * w2:2] + img[0:2 * h2:2, 1:2 * w2:2]
                      + img[1:2 * h2:2, 0:2 * w2:2] + img[1:2 * h2:2, 1:2 * w2:2])
        levels.append(_level(img, levels[-1].scale * 2))
    return levels


def _project(cam: Camera, b: np.ndarray):
    """Pixels where camera-frame directions ``(N, 3)`` land, through the lens, and which are in front."""
    ok = b[:, 2] > 1e-9
    z = np.where(ok, b[:, 2], 1.0)
    d = cam._distort(b[:, :2] / z[:, None])
    return d @ cam.K[:2, :2].T + cam.K[:2, 2], ok


def _samples(cam: Camera, level: _Level, most: int):
    """The pixels of ``level`` of the current picture that take part: viewing directions and grey levels."""
    h, w = level.shape
    inner = max(w - 2, 0) * max(h - 2, 0)
    step = max(1, math.ceil(math.sqrt(inner / most)))
    ys, xs = np.meshgrid(np.arange(1, h - 1, step), np.arange(1, w - 1, step), indexing="ij")
    ys, xs = ys.ravel(), xs.ravel()
    u = np.stack([(xs + 0.5) * level.scale - 0.5, (ys + 0.5) * level.scale - 0.5], -1)
    n = cam.undistort_points(u)
    return np.column_stack([n, np.ones(len(n))]), level.img[ys, xs]


def _photometric(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least-squares ``y ≈ a·x + c``."""
    sw, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    det = sw * sxx - sx * sx
    if abs(det) < 1e-12 or sw <= 0:
        return 1.0, 0.0
    a = (sw * sxy - sx * sy) / det
    return a, (sy - a * sx) / sw


def _huber(r: np.ndarray, constant: float) -> np.ndarray:
    """Huber weights, scaled by the residuals' median absolute deviation."""
    sigma = 1.4826 * np.sort(np.abs(r))[len(r) // 2] if len(r) else 0.0
    if sigma <= 1e-12:
        return np.ones_like(r)
    k = constant * sigma
    return np.where(np.abs(r) <= k, 1.0, k / np.maximum(np.abs(r), 1e-300))


def _agreement(cam: Camera, reference: _Level, bearings, values, rd) -> float | None:
    """Pearson correlation between the current picture and the reference seen through ``rd``."""
    uv, front = _project(cam, bearings @ rd)  # rows: rd.T @ b
    ok, ref, _, _ = reference.sample(reference.at(uv))
    ok &= front
    if ok.sum() < 2:
        return None
    x, y = ref[ok] - ref[ok].mean(), values[ok] - values[ok].mean()
    sxx, syy = (x * x).sum(), (y * y).sum()
    return float((x * y).sum() / math.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else None


@dataclass
class Registration:
    rotation: np.ndarray  # b_now = rotation @ b_reference, camera coordinates
    degrees: float
    before: float  # correlation of the two pictures before and after the turn is taken out
    after: float


def register(cam: Camera, reference: np.ndarray, current: np.ndarray,
             cfg: DriftConfig | None = None) -> Registration | None:
    """Registers ``current`` against ``reference`` (grey pictures of ``cam``'s scene at its resolution).
    ``None`` when the two share too few pixels to compare."""
    cfg = cfg or DriftConfig()
    if reference.shape != current.shape:
        return None
    refs, curs = _pyramid(reference, cfg.coarsest), _pyramid(current, cfg.coarsest)
    rd, eps, axes = np.eye(3), 1e-6, np.eye(3)
    for r, c in reversed(list(zip(refs, curs))):
        bearings, values = _samples(cam, c, cfg.samples)
        for _ in range(cfg.iterations):
            b0 = bearings @ rd  # rd.T @ b for every row
            uv, front = _project(cam, b0)
            ok, iref, gx, gy = r.sample(r.at(uv))
            ok &= front
            if ok.sum() < 64:
                return None
            b0, iref, gx, gy, cur = b0[ok], iref[ok], gx[ok], gy[ok], values[ok]
            # d(pixel)/d(rotation) by central differences of the lens projection: a turn by δ moves
            # the reference direction by δ × b0.
            g = np.zeros((len(b0), 3))
            for k in range(3):
                d = np.cross(axes[k], b0)
                (p, okp), (m, okm) = _project(cam, b0 + eps * d), _project(cam, b0 - eps * d)
                du = (p - m) / (2 * eps)
                g[:, k] = np.where(okp & okm, (gx * du[:, 0] + gy * du[:, 1]) / r.scale, 0.0)
            a, c0 = _photometric(iref, cur, np.ones(len(cur)))
            w = _huber(cur - a * iref - c0, cfg.huber)
            a, c0 = _photometric(iref, cur, w)
            res = cur - a * iref - c0
            j = a * g
            try:
                delta = np.linalg.solve((j * w[:, None]).T @ j, (j * (w * res)[:, None]).sum(0))
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(delta)):
                break
            rd = rd @ rodrigues(-delta)
            if np.linalg.norm(delta) < 1e-7:
                break
    bearings, values = _samples(cam, curs[0], cfg.samples)
    before = _agreement(cam, refs[0], bearings, values, np.eye(3))
    after = _agreement(cam, refs[0], bearings, values, rd)
    if before is None or after is None:
        return None
    return Registration(rd, float(np.degrees(np.linalg.norm(rotation_to_rvec(rd)))), before, after)


def turned(cam: Camera, rotation: np.ndarray) -> Camera:
    """``cam`` after turning by ``rotation`` about its centre: ``R' = rotation @ R``, ``t' = rotation @ t``."""
    return Camera(cam.K, rotation @ cam.R, rotation @ cam.t, cam.dist, cam.size, cam.name)


@dataclass
class References:
    """The reference pictures of a camera model, one PNG per camera: ``<model>.references/<camera>.png``."""

    dir: Path

    @classmethod
    def beside(cls, model: str | Path) -> References:
        return cls(Path(model).with_suffix(".references"))

    def path(self, camera: str) -> Path:
        return self.dir / ("".join(c if c.isalnum() or c in "-_" else "-" for c in camera) + ".png")

    def load(self, camera: str) -> np.ndarray | None:
        path = self.path(camera)
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f"cannot read {path}")
        return img.astype(np.float64)

    def save(self, camera: str, picture: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path(camera)
        if not cv2.imwrite(str(path), np.clip(picture, 0, 255).astype(np.uint8)):
            raise OSError(f"cannot write {path}")


@dataclass
class Entry:
    """What happened to one camera (a row of ``camera_drift.json``)."""

    camera: str
    status: str  # corrected | unchanged | reference saved | skipped
    reference: Path
    degrees: float | None = None
    before: float | None = None
    after: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        out = {"camera": self.camera, "status": self.status}
        for key in ("degrees", "before", "after"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        out["reference"] = str(self.reference)
        if self.note:
            out["note"] = self.note
        return out

    def line(self) -> str:
        measured = ""
        if None not in (self.degrees, self.before, self.after):
            measured = (f": turned {self.degrees:.3f}° since its reference "
                        f"(agreement {self.before:.3f} → {self.after:.3f})")
        note = f" ({self.note})" if self.note else ""
        return f"{self.camera} {self.status}{measured}{note}"


def correct(cameras: list[Camera], backgrounds: list[np.ndarray], references: References,
            cfg: DriftConfig | None = None) -> tuple[list[Camera], list[Entry]]:
    """Corrects each camera for the turn its session background shows against its reference. A camera
    without a reference keeps its calibration and this session's background becomes its reference."""
    cfg = cfg or DriftConfig()
    out, entries = [], []
    for cam, bg in zip(cameras, backgrounds):
        entry = Entry(cam.name, "unchanged", references.path(cam.name))
        used = cam
        h, w = bg.shape[:2]
        if (w, h) != tuple(cam.size):
            entry.status = "skipped"
            entry.note = f"the video is {w}×{h}, the camera model {cam.size[0]}×{cam.size[1]}"
        elif (reference := references.load(cam.name)) is None:
            references.save(cam.name, bg)
            entry.status, entry.note = "reference saved", "later sessions are measured against this one"
        elif reference.shape != bg.shape:
            entry.status = "skipped"
            entry.note = (f"the reference is {reference.shape[1]}×{reference.shape[0]}; "
                          "delete it to take a new one")
        elif (reg := register(cam, reference, bg, cfg)) is None:
            entry.note = "the pictures share too little to compare"
        else:
            entry.degrees, entry.before, entry.after = reg.degrees, reg.before, reg.after
            if reg.after > reg.before:
                used, entry.status = turned(cam, reg.rotation), "corrected"
            else:
                entry.note = "correcting did not improve the agreement"
        out.append(used)
        entries.append(entry)
    return out, entries
