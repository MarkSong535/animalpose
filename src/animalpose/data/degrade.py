"""Image degradations for training on poor recordings: motion and defocus blur, tiny animals,
low contrast, monochrome sensors, noise and compression.

Real setups (e.g. a top camera over a 3-chamber arena) often give a ~30 px mouse,
smeared by motion at a 5 Hz frame rate, in low-contrast grayscale. ``degrade`` turns
clean training images into that regime; strengths are relative to the animal's size
in pixels so the same settings work for crops and full frames.

A :class:`Profile` separates what the *camera* does from what the *animal* does. Its
``psf_px`` is a defocus radius in image pixels: every animal in the frame blurs by the same
amount, whether it is 25 px or 250 px away, which is what a fixed lens actually does.
``animalpose data audit`` measures these values on a recording and writes them to YAML;
``data.camera_profile`` in a training config then samples degradations around them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


def motion_kernel(length: float, angle: float) -> np.ndarray:
    """Normalized line kernel of ``length`` px at ``angle`` radians."""
    n = max(3, int(np.ceil(length)) | 1)
    k = np.zeros((n, n), np.float32)
    c = (n - 1) / 2
    dx, dy = np.cos(angle) * (length - 1) / 2, np.sin(angle) * (length - 1) / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1,
             cv2.LINE_AA)
    return k / max(k.sum(), 1e-6)


def disk_kernel(radius: float) -> np.ndarray:
    r = max(1, int(np.ceil(radius)))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = (x * x + y * y <= radius * radius + 0.5).astype(np.float32)
    return k / k.sum()


@dataclass(frozen=True)
class Profile:
    """How a camera degrades a recording. Ranges are ``(low, high)`` and sampled per image.

    * ``psf_px``: defocus radius in **image pixels** (a property of the lens and the subject
      distance, not of the animal). When set, defocus is always applied, jittered by
      ``psf_jitter`` (autofocus and the animal drifting in and out of the focal plane); the
      ``defocus_frac`` branch is then unused.
    * ``defocus_frac`` / ``motion_frac``: radii and smear lengths as a fraction of the animal's
      body length, for setups where nothing is known about the camera.
    * ``motion_frac`` also encodes frame rate: a 30 fps recording with short exposure gives a
      small fraction, a 5 fps one a large fraction.
    * ``downscale_to``: the animal ends up this many pixels long (sensor resolution and
      distance), or ``None`` to keep resolution.
    * ``mono`` / ``low_contrast``: probability of each; ``noise`` is the sensor sigma in grey
      levels, ``jpeg`` the quality range (``None`` disables either).
    """

    psf_px: float | None = None
    psf_jitter: float = 0.35
    defocus_frac: tuple[float, float] = (0.0, 0.06)
    motion_frac: tuple[float, float] = (0.0, 0.2)
    downscale_to: tuple[float, float] | None = (16.0, 48.0)
    mono: float = 0.5
    low_contrast: float = 0.5
    noise: tuple[float, float] = (2.0, 12.0)
    jpeg: tuple[float, float] | None = (15.0, 70.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping | None) -> Profile:
        if d is None:
            return DEFAULT_PROFILE
        known = {f for f in Profile.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown camera profile keys: {sorted(unknown)}")
        out = {}
        for k, v in d.items():
            out[k] = tuple(v) if isinstance(v, list) else v
        return Profile(**out)


DEFAULT_PROFILE = Profile()

# Keys a profile YAML may carry for documentation (written by animalpose data audit).
DOC_KEYS = ("measured", "notes", "source")


def load_profile(value: Mapping | str | Path | None) -> Profile:
    """A profile from a config mapping, a YAML path, or ``None`` for the default one.

    Documentation keys (``measured``, ``notes``, ``source``) are skipped, so the file
    ``animalpose data audit`` writes can be handed to this directly; a typo in a real
    profile key is still an error.
    """
    if value is None:
        return DEFAULT_PROFILE
    if isinstance(value, (str, Path)):
        import yaml

        value = yaml.safe_load(Path(value).read_text())
    return Profile.from_dict({k: v for k, v in (value or {}).items() if k not in DOC_KEYS})


def degrade(image: np.ndarray, rng: np.random.Generator, animal_px: float, strength: float = 1.0,
            psf_px: float | None = None, profile: Profile | None = None) -> np.ndarray:
    """Randomly degrade a BGR uint8 ``image`` whose animal is about ``animal_px`` px long.

    Each effect is drawn independently; ``strength`` (0-1) scales how likely and how
    severe they are. ``profile`` selects the camera behaviour; ``psf_px`` overrides its
    defocus radius (see :meth:`Profile`).
    """
    p = profile or DEFAULT_PROFILE
    psf = p.psf_px if psf_px is None else psf_px
    img = image.astype(np.float32)
    a = max(float(animal_px), 8.0)
    s = float(np.clip(strength, 0.0, 1.0))
    h, w = img.shape[:2]
    if p.motion_frac[1] > 0 and rng.random() < 0.6 * s:  # motion blur, up to motion_frac of the body length
        length = rng.uniform(3, max(4, p.motion_frac[1] * a * s))
        img = cv2.filter2D(img, -1, motion_kernel(length, rng.uniform(0, np.pi)))
    if psf is not None:  # defocus set by the lens: same radius for every animal in the frame
        r = psf * rng.uniform(1 - p.psf_jitter, 1 + p.psf_jitter)
        img = cv2.filter2D(img, -1, disk_kernel(max(1.0, r)))
    elif rng.random() < 0.4 * s:  # defocus relative to the animal
        img = cv2.filter2D(img, -1, disk_kernel(rng.uniform(1, max(1.5, p.defocus_frac[1] * a * s))))
    if p.downscale_to is not None and rng.random() < 0.5 * s:  # low resolution, then back up
        target = rng.uniform(*p.downscale_to)
        f = min(1.0, target / a)
        if f < 0.95:
            small = cv2.resize(img, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < p.mono * s:  # monochrome (IR / machine-vision cameras)
        img = np.repeat(img.mean(-1, keepdims=True), 3, -1)
    if rng.random() < p.low_contrast * s:  # low contrast, gamma
        m = img.mean()
        img = m + (img - m) * rng.uniform(0.35, 0.9)
        img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** rng.uniform(0.7, 1.5)
    if p.noise[1] > 0 and rng.random() < 0.5 * s:  # sensor noise, signal-dependent
        sigma = rng.uniform(*p.noise) * s
        img = img + rng.normal(0, 1, img.shape) * sigma * np.sqrt(np.clip(img, 1, 255) / 128.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if p.jpeg is not None and rng.random() < 0.4 * s:  # compression
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(*p.jpeg))])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def zoom_out(image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator, min_scale: float = 0.35):
    """Shrink the frame into a same-size canvas (filled with its mean colour) so animals become small.
    Returns the new image and boxes."""
    h, w = image.shape[:2]
    f = rng.uniform(min_scale, 1.0)
    nw, nh = max(2, int(w * f)), max(2, int(h * f))
    x0, y0 = int(rng.integers(0, w - nw + 1)), int(rng.integers(0, h - nh + 1))
    canvas = np.empty_like(image)
    canvas[:] = image.reshape(-1, image.shape[-1]).mean(0).astype(image.dtype)
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, boxes * f + np.array([x0, y0, x0, y0], np.float32)


def edge_width(patches: np.ndarray, ring: int = 2, percentile: float = 95.0) -> tuple[float, float]:
    """Transition width of the subject silhouette in ``patches``, in patch px, and its spread.

    The peak gradient across a step of contrast ``C`` blurred by a kernel of width ``w`` is
    ``C / w`` (a defocus disk of radius ``r`` gives ``w ≈ 1.4 r``), so the width follows from the
    contrast and the peak gradient, which needs no assumption about the image content. Patches
    hold a dark subject on a brighter surround; the gradient is measured in a thin ring around
    the subject's silhouette, where the edge actually is.

    An unblurred silhouette still measures a pixel or two (the sampling grid) plus the subject's
    own softness (fur, paws, the third dimension), so this compares like with like: use it to
    match the blur of degraded training crops to that of a target recording, not as an absolute
    lens measurement.
    """
    out = []
    k = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    for g in patches:
        hi, lo = np.percentile(g, 95), np.percentile(g, 5)
        if hi - lo < 8:
            continue
        t = np.clip((hi - g) / (hi - lo), 0.0, 1.0)  # 1 inside the subject, 0 on the surround
        m = (t > 0.5).astype(np.uint8)
        if int(m.sum()) < 30:
            continue
        edge = cv2.dilate(m, k) & (1 - cv2.erode(m, k))
        if int(edge.sum()) < 20:
            continue
        peak = float(np.percentile(np.hypot(*np.gradient(g))[edge > 0], percentile))
        if peak > 1e-6:
            out.append((hi - lo) / peak)
    if not out:
        return 0.0, 0.0
    return float(np.median(out)), float(np.std(out))


def motion_smear(displacement_px: np.ndarray, exposure_fraction: float = 0.5) -> float:
    """Expected motion smear in px: how far the animal moves during one exposure."""
    return float(np.median(np.asarray(displacement_px, float)) * exposure_fraction)
