"""Ethogram figure and annotated clip from the ``tracks.npz`` and ``behavior_catalog.json`` of a run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from animalpose.behavior.catalog import top_k
from animalpose.skeleton import CANONICAL

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
           "#bab0ac", "#86bcb6", "#d37295"]


def ethogram(out: str | Path, names, full: np.ndarray, times: np.ndarray, modelled,
             heldout_from: float | None = None) -> dict:
    """Stacked probabilities, a top-1 colour band and the time share of each behavior; returns the shares."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = [i for i in np.argsort(-full.mean(0)) if modelled[i] and full[:, i].max() > 0.05][:12]
    colors = {i: PALETTE[j % len(PALETTE)] for j, i in enumerate(used)}
    top = full.argmax(1)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 0.5, 1.6]})
    ax0.stackplot(times, full[:, used].T, colors=[colors[i] for i in used], labels=[names[i] for i in used], lw=0)
    ax0.set_ylabel("probability")
    ax0.set_xlim(times[0], times[-1])
    ax0.set_ylim(0, 1)
    ax0.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    starts = np.r_[0, np.nonzero(np.diff(top))[0] + 1]
    ends = np.r_[starts[1:], len(top)]
    for s, e in zip(starts, ends):
        t1 = times[e] if e < len(times) else times[-1]
        ax1.axvspan(times[s], t1, color=colors.get(top[s], "#dddddd"), lw=0)
    ax1.set_xlim(times[0], times[-1])
    ax1.set_yticks([])
    ax1.set_ylabel("top 1", rotation=0, ha="right", va="center")
    for ax in (ax0, ax1):
        if heldout_from is not None:
            ax.axvline(heldout_from, color="k", lw=1.2, ls="--")
    if heldout_from is not None:
        ax0.text(heldout_from, 1.02, " held out", fontsize=9, va="bottom")
    dt = np.diff(times, append=times[-1] + np.median(np.diff(times)))
    share = {names[i]: float(dt[top == i].sum() / dt.sum()) for i in used}
    order = sorted(share, key=share.get, reverse=True)
    ax2.barh(order[::-1], [share[n] * 100 for n in order[::-1]],
             color=[colors[names.index(n)] for n in order[::-1]])
    ax2.set_xlabel("time as most likely behavior (%)")
    ax0.set_title("Behavior probabilities over time (time axis in seconds from frame timestamps)")
    ax1.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return share


def clip(out: str | Path, video: str | Path, names, z, start_s: float, seconds: float, k: int,
         scale: int = 3, half: int = 60) -> Path:
    """A window of the video zoomed on the animal, with its skeleton and the top-``k`` behaviors."""
    times = z["times"]
    kp, cf, full = z["track0_smoothed"], z["track0_smoothed_conf"], z["track0_behavior_full"]
    frames = z["track0_frames"].astype(int)
    first = int(np.searchsorted(times, start_s))
    last = int(np.searchsorted(times, start_s + seconds))
    fps = 1.0 / float(np.median(np.diff(times[first:last + 1])))
    idx_k, p_k = top_k(full, k)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    size = 2 * half * scale
    panel = 300
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{size + panel}x{size}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                           "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
    mid = CANONICAL.index("spine_mid")
    centre = None
    for f in range(first, last):
        ok, img = cap.read()
        if not ok:
            break
        i = int(np.searchsorted(frames, f))
        if i >= len(frames) or frames[i] != f:
            continue
        c = kp[i, mid] if np.isfinite(kp[i, mid]).all() else centre
        centre = c if centre is None or c is None else 0.85 * centre + 0.15 * c  # steady camera
        cx, cy = (np.nan_to_num(centre) if centre is not None else np.array(img.shape[1::-1]) / 2).astype(int)
        pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REPLICATE)
        view = cv2.resize(pad[cy:cy + 2 * half, cx:cx + 2 * half], (size, size), interpolation=cv2.INTER_CUBIC)
        pts = (kp[i] - [cx - half, cy - half]) * scale
        ok_k = cf[i] > 0.3
        for a, b in CANONICAL.edge_index:
            if ok_k[a] and ok_k[b]:
                cv2.line(view, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), (255, 160, 40), 2, cv2.LINE_AA)
        for j in np.nonzero(ok_k)[0]:
            cv2.circle(view, tuple(np.int32(pts[j])), 4, (40, 40, 230), -1, cv2.LINE_AA)
        side = np.full((size, panel, 3), 255, np.uint8)
        cv2.putText(side, f"t = {times[f]:7.2f} s", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        for r in range(k):
            y = 90 + r * 70
            name, p = names[idx_k[i, r]], p_k[i, r]
            cv2.putText(side, f"{r + 1}. {name}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(side, (12, y + 12), (12 + int(260 * p), y + 32), (180, 120, 60), -1)
            cv2.putText(side, f"{p:.2f}", (16, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ff.stdin.write(np.concatenate([view, side], 1).tobytes())
    ff.stdin.close()
    ff.wait()
    cap.release()
    return Path(out)


def plot_run(run: str | Path, video: str | Path, *, clip_start: float = 0.0, clip_seconds: float = 30.0,
             top: int = 3, heldout_from: float | None = None) -> dict:
    """Write ``ethogram.png`` and ``behavior_clip.mp4`` next to a run's ``tracks.npz``; returns the shares."""
    run = Path(run)
    z = np.load(run / "tracks.npz", allow_pickle=False)
    names = [str(n) for n in z["behaviors"]]
    modelled = [d["modelled"] for d in json.loads((run / "behavior_catalog.json").read_text())]
    times = z["times"][z["track0_frames"].astype(int)]
    share = ethogram(run / "ethogram.png", names, z["track0_behavior_full"], times, modelled, heldout_from)
    clip(run / "behavior_clip.mp4", video, names, z, clip_start, clip_seconds, top)
    return share
