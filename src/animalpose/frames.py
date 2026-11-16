"""Frame sources: a video file or a folder of images (e.g. TIF sequences from machine-vision cameras).

Images are returned as 8-bit BGR whatever their storage: grayscale is replicated to
three channels, 16-bit frames are rescaled with a fixed per-sequence range so
brightness stays consistent over time. For image folders the frame rate is estimated
from timestamps in the file names (``..._HH_MM_SS_mmm`` as in ``img1030_13_47_28_113.tif``).
"""

from __future__ import annotations

import os
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _timestamp(name: str) -> float | None:
    m = re.search(r"(\d{1,2})_(\d{2})_(\d{2})_(\d{1,3})(?:\D*)$", Path(name).stem)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


def image_files(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)


def estimate_fps(files: list[Path]) -> float | None:
    ts = [_timestamp(p.name) for p in files[:500]]
    if len(ts) < 3 or any(t is None for t in ts):
        return None
    d = np.diff(np.asarray(ts, float))
    d = d[d > 0]
    return float(1.0 / np.median(d)) if len(d) else None


def to_bgr8(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if img.dtype != np.uint8:
        lo = float(img.min()) if lo is None else lo
        hi = float(img.max()) if hi is None else hi
        img = np.clip((img.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def read_frames(path: str | Path, max_frames: int | None = None, stride: int = 1):
    """``(frames, fps)`` from a video file or an image folder; ``fps`` is None when unknown."""
    path = Path(path)
    frames: list[np.ndarray] = []
    if path.is_dir():
        files = image_files(path)[::stride][:max_frames]
        fps = estimate_fps(files)
        raw = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in files]
        raw = [r for r in raw if r is not None]
        lo = hi = None
        if raw and raw[0].dtype != np.uint8:  # one intensity range for the whole sequence
            sample = np.stack(raw[:: max(1, len(raw) // 50)])
            lo, hi = np.percentile(sample, 0.5), np.percentile(sample, 99.5)
        frames = [to_bgr8(r, lo, hi) for r in raw]
        return frames, (fps / stride if fps else None)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    t = 0
    while max_frames is None or len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if t % stride == 0:
            frames.append(to_bgr8(f))
        t += 1
    cap.release()
    return frames, (fps / stride if fps else None)


# ----------------------------------------------------------------------------- unlabeled collections

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv"}


@dataclass
class Source:
    """One recording: a video file, or a folder of image frames (``files`` in order)."""

    path: Path
    kind: str  # "video" | "images"
    files: list[Path] | None = None

    @property
    def n_frames(self) -> int:
        if self.kind == "images":
            return len(self.files)
        return int(cv2.VideoCapture(str(self.path)).get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self, indices) -> list[tuple[int, np.ndarray]]:
        """Frames at the given indices as 8-bit BGR (skips unreadable ones)."""
        out = []
        if self.kind == "images":
            for i in indices:
                img = cv2.imread(str(self.files[i]), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    out.append((int(i), to_bgr8(img)))
            return out
        cap = cv2.VideoCapture(str(self.path))
        for i in sorted(int(i) for i in indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if ok:
                out.append((i, to_bgr8(img)))
        cap.release()
        return out


def find_sources(root: str | Path) -> list[Source]:
    """Every video file and every folder that directly holds image files, anywhere under ``root``."""
    out = []
    for d, _, names in sorted(os.walk(root)):
        d = Path(d)
        imgs = sorted(d / n for n in names if Path(n).suffix.lower() in IMAGE_EXT)
        if imgs:
            out.append(Source(d, "images", imgs))
        out += [Source(d / n, "video") for n in sorted(names) if Path(n).suffix.lower() in VIDEO_EXT]
    return out


def is_held_out(source: Source, root: str | Path, every: int = 5) -> bool:
    """Deterministic split by recording: about 1 in ``every`` sources is held out for evaluation."""
    rel = os.path.relpath(os.path.abspath(source.path), os.path.abspath(root))
    return zlib.crc32(rel.encode()) % every == 0


def open_source(path: str | Path) -> Source:
    """A video file or an image folder as a :class:`Source`."""
    path = Path(path)
    return Source(path, "images", image_files(path)) if path.is_dir() else Source(path, "video")


def iter_frames(path: str | Path, max_frames: int | None = None):
    """Stream 8-bit BGR frames from a video or image folder without holding them in memory."""
    src = open_source(path)
    if src.kind == "images":
        for i in range(min(len(src.files), max_frames or len(src.files))):
            for _, f in src.frames([i]):
                yield f
        return
    cap = cv2.VideoCapture(str(path))
    t = 0
    while max_frames is None or t < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        yield to_bgr8(f)
        t += 1
    cap.release()


def source_fps(path: str | Path) -> float | None:
    src = open_source(path)
    if src.kind == "images":
        return estimate_fps(src.files)
    return cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or None


def sampled_background(path: str | Path, n: int = 64, max_frames: int | None = None) -> np.ndarray:
    """Grey median background from ``n`` frames spread over the recording."""
    src = open_source(path)
    total = min(src.n_frames, max_frames or src.n_frames)
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
              for _, f in src.frames(np.linspace(0, total - 1, min(n, total)).astype(int))]
    return np.median(np.stack(frames), 0)


def frame_timestamps(path: str | Path, max_frames: int | None = None) -> np.ndarray:
    """Per-frame presentation times in seconds, from 0, for variable-frame-rate recordings.

    Videos: ffprobe packet timestamps when ffprobe is available (fast, exact), else OpenCV's
    position after each grab. Image folders: times parsed from the file names, else
    ``index / estimated fps`` (or 30 fps).
    """
    import shutil
    import subprocess

    src = open_source(path)
    if src.kind == "images":
        files = src.files[:max_frames]
        ts = [_timestamp(p.name) for p in files]
        if ts and all(t is not None for t in ts):
            t = np.asarray(ts, float)
            t = np.where(t < t[0], t + 86400.0, t)  # past midnight
            return t - t[0]
        return np.arange(len(files)) / (estimate_fps(files) or 30.0)
    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False).stdout
        t = np.sort(np.array([float(x) for x in out.split() if x.strip() not in ("", "N/A")]))
        if len(t):
            return (t - t[0])[:max_frames]
    cap = cv2.VideoCapture(str(path))
    t = []
    while (max_frames is None or len(t) < max_frames) and cap.grab():
        t.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()
    t = np.asarray(t, float)
    return t - t[0] if len(t) else t


def timing_report(times: np.ndarray) -> dict:
    """Frame-rate summary: mean rate, interval spread, dropped-frame gaps (> 1.5 median intervals).
    ``variable`` is set when intervals spread by more than 10 % or frames were dropped."""
    if len(times) < 2:
        return {"frames": int(len(times))}
    dt = np.diff(times)
    med = float(np.median(dt))
    gaps = dt > 1.5 * med
    return {"frames": int(len(times)), "duration_s": float(times[-1]), "mean_fps": float(1.0 / dt.mean()),
            "interval_ms_p1_p99": [float(np.percentile(dt, 1) * 1e3), float(np.percentile(dt, 99) * 1e3)],
            "gaps": int(gaps.sum()), "longest_gap_s": float(dt.max()),
            "variable": bool(np.std(dt) > 0.1 * med or gaps.any())}
