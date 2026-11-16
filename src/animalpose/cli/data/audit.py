"""Measure what a recording does to the image, and write a camera profile from it.

``animalpose.data.degrade`` invents degradations; this measures them, so training degrades
images the way the target camera does.

The core measurement is the **edge transition width** of the animal's silhouette against the
arena floor (:func:`animalpose.data.degrade.edge_width`): the peak gradient across a step of
contrast ``C`` blurred over a width ``w`` is ``C / w``, so width follows from the contrast and
the peak gradient with no assumption about image content. An unblurred silhouette still
measures a pixel or two plus the animal's own softness, so the number is only meaningful
relative to another recording of a similar subject -- which is exactly how it is used:

* the target recording's width is measured, spread over time blocks (autofocus drift becomes
  ``psf_jitter``) and over the silhouette;
* clean training crops are degraded with candidate PSF radii and measured the same way
  (``--calibrate``, using ``--surrogate`` datasets);
* the radius that makes the crops measure like the target becomes ``psf_px``, so the model
  trains on the blur it will be shown.

Also reported: animal size, per-frame displacement and the motion smear it implies at the
given exposure (``--exposure``), frame contrast, background level, sensor noise, and global
sharpness. The YAML it writes goes to ``data.camera_profile`` in a training config, as a path
or inline. Compression cannot be measured from decoded frames, so ``jpeg`` is assumed.

    animalpose data audit myvideo.mp4 --out configs/camera.yaml
    animalpose data audit --root ~/animal/training_data --frames 48
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from animalpose.data.datasets import CocoKeypoints, PoseCropDataset
from animalpose.data.degrade import Profile, degrade, edge_width, motion_smear
from animalpose.frames import find_sources, open_source, source_fps
from animalpose.geometry import box_xyxy_to_cs, crop_affine, warp
from animalpose.pipeline import arena_roi, largest_moving_blob, median_background

NAME = "audit"

PATCH = 64  # animal patches are measured on this grid
BOX_PAD = 1.3  # largest_moving_blob pads a box by 15 % of its longest side on each side
PSF_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
SURROGATE_SETS = (("crim13/val.json", "crim13/images", "mouse"),
                  ("mirror_mouse/side_val.json", "mirror_mouse/images", "mouse"))


def _gray(img: np.ndarray) -> np.ndarray:
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img).astype(np.float32)


def noise_sigma(g: np.ndarray, roi: np.ndarray | None) -> float:
    """Sensor noise: robust sigma of the high-pass residual, in grey levels."""
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    v = hp[roi > 0] if roi is not None else hp
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def animal_patch(img: np.ndarray, box: np.ndarray, out: int = PATCH) -> tuple[np.ndarray, float]:
    """``(grey patch, px per image px)``: ``box`` plus 40 % context resampled to ``out``."""
    center, side = box_xyxy_to_cs(box, padding=1.4)
    crop = warp(img, crop_affine(center, side, out, 0.0), out)
    return _gray(crop), out / side


def _displacement(src, bg, roi, n: int, windows: int, window: int, diff: float) -> list[float]:
    """Blob displacement per frame, measured on consecutive frames in windows over the video."""
    out = []
    for w in range(windows):
        s0 = int((w + 0.5) * n / max(windows, 1))
        frames = [f for _, f in src.frames(range(s0, min(s0 + window, n)))]
        cents = []
        for f in frames:
            b = largest_moving_blob(f, bg, diff, roi=roi)[0]
            cents.append(None if b is None else np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]))
        for a, b in zip(cents, cents[1:]):
            if a is not None and b is not None:
                out.append(float(np.linalg.norm(b - a)))
    return out


def audit(src, n_frames: int = 64, exposure: float = 0.5, windows: int = 6, window: int = 30,
          diff: float = 25.0) -> dict:
    """Measure the camera degradations of one recording (see the module docstring)."""
    n = src.n_frames
    if n < 4:
        return {"source": str(src.path), "error": "too few frames"}
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else 0.0
    idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)
    frames = [f for _, f in src.frames(idx)]
    gs = [_gray(f) for f in frames]
    bg = median_background(frames, len(frames))
    roi = arena_roi(bg)
    boxes = [largest_moving_blob(f, bg, diff, roi=roi)[0] for f in frames]
    edge, spread, sizes, noise, lap = [], [], [], [], []
    for g, b in zip(gs, boxes):
        lap.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        noise.append(noise_sigma(g, roi))
        if b is not None:
            sizes.append(float(max(b[2] - b[0], b[3] - b[1])) / BOX_PAD)  # undo the blob-box padding
    for part in np.array_split(np.arange(len(frames)), 4):  # per time block: focus can drift
        sel = [i for i in part if boxes[i] is not None]
        if len(sel) < 3:
            continue
        pts, scales = [], []
        for i in sel:
            p, k = animal_patch(frames[i], boxes[i])
            pts.append(p)
            scales.append(k)
        w, s = edge_width(np.stack(pts))
        k = float(np.mean(scales))  # patch px -> image px
        edge.append(w / k)
        spread.append(s / k)
    disp = _displacement(src, bg, roi, n, windows, window, diff)
    smear = motion_smear(disp, exposure) if disp else 0.0
    body = float(np.median(sizes)) if sizes else None
    fps = source_fps(src.path)
    return {
        "source": str(src.path),
        "frames": int(n),
        "fps": fps,
        "duration_s": round(n / fps, 1) if fps else None,
        "colour": bool(not np.allclose(frames[0][..., 0].astype(int), frames[0][..., 2].astype(int))),
        "background_grey": [q(bg.ravel(), 1), q(bg.ravel(), 50), q(bg.ravel(), 99)],
        "frame_contrast": [q(gs[0].ravel(), 1), q(gs[0].ravel(), 99)],
        "sharpness_lapvar": {"p10": q(lap, 10), "median": q(lap, 50), "p90": q(lap, 90)},
        "noise_sigma": {"p10": q(noise, 10), "median": q(noise, 50), "p90": q(noise, 90)},
        "animal_px": {"p10": q(sizes, 10), "median": body, "p90": q(sizes, 90)},
        "frames_with_blob": int(sum(b is not None for b in boxes)),
        "displacement_px": {"median": q(disp, 50), "p90": q(disp, 90), "windows": int(windows * window)},
        "edge_width_px": {"median": round(q(edge, 50), 2), "p10": round(q(edge, 10), 2),
                          "p90": round(q(edge, 90), 2), "silhouette_spread": round(q(spread, 50), 2),
                          "blocks": len(edge)},
        "exposure_fraction": exposure,
        "motion_smear_px": round(smear, 2),
        "motion_smear_body_frac": round(smear / body, 3) if body else None,
    }


def surrogate_crops(root: str | Path, n: int = 96, crop_size: int = 192, sets=SURROGATE_SETS) -> list[np.ndarray]:
    """Clean BGR crops from prepared training datasets, to calibrate a PSF against."""
    out = []
    for js, imgs, species in sets:
        j, im = Path(root) / js, Path(root) / imgs
        if not j.exists():
            continue
        ds = PoseCropDataset(CocoKeypoints(j, im, "synthetic", species), n, crop_size, augment=False)
        for i in range(len(ds)):
            out.append((np.ascontiguousarray(ds[i][0].numpy()[::-1].transpose(1, 2, 0)) * 255).astype(np.uint8))
    return out


def calibrate_psf(target_px: float, crops: list[np.ndarray], grid=PSF_GRID) -> tuple[float, dict]:
    """The PSF radius whose degraded crops measure ``target_px`` of edge width, and the curve.

    Only defocus is applied (motion, noise, contrast and compression switched off), so the
    curve is monotone and the match is the blur alone.
    """
    curve = {}
    bare = Profile(psf_jitter=0.0, motion_frac=(0.0, 0.0), downscale_to=None, mono=0.0, low_contrast=0.0,
                   noise=(0.0, 0.0), jpeg=None)
    for r in grid:
        prof = Profile(**{**bare.__dict__, "psf_px": max(r, 1e-3)})
        degraded = np.stack([_gray(degrade(c, np.random.default_rng(17 + i), 40, profile=prof))
                             for i, c in enumerate(crops)])
        curve[round(float(r), 2)] = round(edge_width(degraded)[0], 3)
    best = min(curve, key=lambda r: abs(curve[r] - target_px))
    return best, {"target_edge_width_px": round(target_px, 3), "edge_width_by_psf_px": curve}


def profile_from(m: dict, psf_px: float | None = None) -> tuple[Profile, list[str]]:
    """A training profile from the measurements, plus notes on what is assumed."""
    notes = []
    body, edge = m["animal_px"]["median"], m["edge_width_px"]["median"]
    spread = m["edge_width_px"]["silhouette_spread"]
    if body and m["motion_smear_body_frac"] is not None:
        motion_hi = min(0.2, round(max(0.0, m["motion_smear_body_frac"] * 1.5), 3))
    else:
        motion_hi = 0.2
        notes.append("no animal found: motion_frac kept at the default 0.2")
    if psf_px is None:
        notes.append("psf_px not calibrated: rerun with --calibrate (it matches training crops to this "
                     "edge width); the uncalibrated width/1.4 is used")
        psf_px = round(edge / 1.4, 2)
    p = Profile(
        psf_px=psf_px if psf_px > 0 else None,
        psf_jitter=float(np.clip(m["edge_width_px"]["p90"] / max(edge, 1e-6) - 1, 0.05, 0.6)),
        motion_frac=(0.0, motion_hi),
        downscale_to=(max(8.0, round(0.8 * body)), round(1.15 * body)) if body else (16.0, 48.0),
        mono=0.0 if m["colour"] else 1.0,
        low_contrast=0.0 if m["frame_contrast"][1] - m["frame_contrast"][0] > 150 else 1.0,
        noise=(round(m["noise_sigma"]["p10"], 1), round(max(m["noise_sigma"]["p90"], 1.0), 1)),
        jpeg=(40.0, 80.0),
    )
    if spread > 0.5 and spread > 0.4 * edge:
        notes.append(f"the edge width varies by {spread:.1f} px around the silhouette (mean {edge:.1f} px): part of "
                     f"it is the animal's own shape, so treat psf_px as an upper bound")
    if m["motion_smear_px"] > 0.5 * edge:
        notes.append(f"motion smear ({m['motion_smear_px']:.1f} px) is comparable to the edge width: raise "
                     f"motion_frac if the recording is at a low frame rate")
    notes.append("jpeg quality is assumed: compressed sources (h264/jpg) are not measurable per frame")
    return p, notes


def write_yaml(path: Path, m: dict, p: Profile, notes: list[str], calibration: dict | None) -> None:
    dur = f", {m['duration_s']:.0f} s" if m.get("duration_s") else ""
    head = [f"# animalpose data audit on {m['source']} ({m['frames']} frames, {m['fps']:.2f} fps{dur}, "
            f"{'colour' if m['colour'] else 'greyscale'}).",
            "# Feed to training with data.camera_profile=<this file>; see animalpose.data.degrade.Profile.",
            f"# Animal ~{m['animal_px']['median']:.0f} px, edge width {m['edge_width_px']['median']:.1f} px, "
            f"PSF {p.psf_px} px (+-{100 * p.psf_jitter:.0f} %), motion smear {m['motion_smear_px']:.1f} px, "
            f"noise sigma {m['noise_sigma']['median']:.1f}."]
    head += [f"# assumed: {n}" for n in notes]
    measured = {k: v for k, v in m.items() if k != "source"}
    if calibration:
        measured["psf_calibration"] = calibration
    body = {"measured": measured,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in p.to_dict().items()}}
    path.write_text("\n".join(head) + "\n" + yaml.safe_dump(body, sort_keys=False))


def add_arguments(parser) -> None:
    parser.add_argument("paths", nargs="*", help="video files or frame folders")
    parser.add_argument("--root", help="audit every recording under this folder")
    parser.add_argument("--frames", type=int, default=64, help="frames sampled per recording")
    parser.add_argument("--windows", type=int, default=6, help="windows for per-frame displacement")
    parser.add_argument("--window", type=int, default=30, help="consecutive frames per window")
    parser.add_argument("--exposure", type=float, default=0.5, help="exposure as a fraction of the frame interval")
    parser.add_argument("--diff", type=float, default=25.0, help="grey levels that count as moving pixels")
    parser.add_argument("--no-calibrate", action="store_true", help="skip matching PSF candidates to training crops")
    parser.add_argument("--surrogate", default="data/processed", help="prepared datasets to calibrate the PSF on")
    parser.add_argument("--crops", type=int, default=96, help="clean crops used for the PSF calibration")
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--out", help="write this camera profile YAML (single recording)")
    parser.add_argument("--json", help="write the full measurements here")


def run(args) -> None:
    sources = [open_source(p) for p in args.paths]
    if args.root:
        sources += find_sources(args.root)
    if not sources:
        raise SystemExit("give a path or --root")
    crops = None if args.no_calibrate else surrogate_crops(args.surrogate, args.crops, args.crop_size)
    if crops is not None and not crops:
        print(f"no surrogate crops under {args.surrogate}: PSF stays uncalibrated", flush=True)
    allm = []
    for src in sources:
        m = audit(src, args.frames, args.exposure, args.windows, args.window, args.diff)
        allm.append(m)
        if "error" in m:
            print(m["source"], m["error"], flush=True)
            continue
        target = m["edge_width_px"]["median"]
        calibration, psf = None, None
        if crops:
            psf, calibration = calibrate_psf(target, crops)
            m["edge_width_px"]["clean_crop_edge_width_px"] = round(calibration["edge_width_by_psf_px"][0.0], 2)
        p, notes = profile_from(m, psf)
        print(json.dumps({k: m[k] for k in ("source", "fps", "animal_px", "edge_width_px", "motion_smear_px",
                                            "noise_sigma", "sharpness_lapvar")}))
        print("  profile:", json.dumps({k: v for k, v in p.to_dict().items() if k != "defocus_frac"}))
        if calibration:
            print("  psf calibration:", json.dumps(calibration["edge_width_by_psf_px"]))
        for n in notes:
            print("  note:", n)
        if args.out and len(sources) == 1:
            write_yaml(Path(args.out), m, p, notes, calibration)
            print("  ->", args.out)
    if args.json:
        Path(args.json).write_text(json.dumps(allm, indent=1))
