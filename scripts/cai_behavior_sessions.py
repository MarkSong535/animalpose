"""Behavior sessions from the Cai two-view results, for training the single-view behavior classifier.

The two-view 3D rules (`~/Cai/<experiment>/behavior3d/<top>.csv`, made by ~/upload/pose_3d) label
each top-view frame from both cameras: rearing from the nose's true height, grooming from posture
and motion, social from the nose at a cup. A classifier trained on those labels and the top view's
2D keypoints (`position/<top>.csv`) brings the 3D-informed labels to a single top-view video.

    3D rule state        session label
    rearing              rearing
    grooming             grooming
    other, moving        walking
    other, still         resting
    social_L / social_R  ignored (where the cups are is not in the keypoints; the assay's zones
                         and contact rules score it)
    no valid 3D          ignored

Writes one session per top-view video (`animalpose.data.behavior_sets.load_session_dir` format:
coords, conf, labels, label_names, fps, animal, view) into `<out>/train/` or `<out>/heldout/`; a
fifth of each experiment's videos (every fifth, from the third) are held out.

    python scripts/cai_behavior_sessions.py --cai ~/Cai --out ~/upload/behavior/sessions
"""

import argparse
import csv
from pathlib import Path

import numpy as np

KEYPOINTS = (
    "nose left_eye right_eye left_ear right_ear head_center neck spine_upper spine_mid spine_lower tail_base "
    "tail_mid tail_end left_shoulder right_shoulder left_front_knee right_front_knee left_front_paw "
    "right_front_paw left_hip right_hip left_hind_knee right_hind_knee left_hind_paw right_hind_paw"
).split()
LABELS = ("resting", "walking", "rearing", "grooming")


def label(state, activity):
    if state in ("rearing", "grooming"):
        return LABELS.index(state)
    if state == "other" and activity in ("moving", "still"):
        return LABELS.index("walking" if activity == "moving" else "resting")
    return -100  # social, or no valid 3D: ignored by the loss


def number(text):
    try:
        return float(text)
    except ValueError:
        return np.nan


def session(position_csv, behavior_csv):
    pos = list(csv.DictReader(open(position_csv)))
    beh = {int(r["frame"]): r for r in csv.DictReader(open(behavior_csv))}
    t = len(pos)
    coords = np.full((t, len(KEYPOINTS), 2), np.nan)
    conf = np.zeros((t, len(KEYPOINTS)))
    labels = np.full(t, -100, np.int64)
    times = np.full(t, np.nan)
    for i, r in enumerate(pos):
        times[i] = number(r["time_s"])
        for k, name in enumerate(KEYPOINTS):
            coords[i, k] = number(r[f"{name}_x"]), number(r[f"{name}_y"])
            conf[i, k] = np.nan_to_num(number(r[f"{name}_conf"]))
        b = beh.get(int(r["frame"]))
        if b is not None:
            labels[i] = label(b["state"], b["activity"])
    dt = np.nanmedian(np.diff(times))
    fps = float(1.0 / dt) if dt > 0 else 30.0
    return coords, conf, labels, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cai", type=Path, default=Path.home() / "Cai")
    parser.add_argument("--out", type=Path, default=Path.home() / "upload/behavior/sessions")
    args = parser.parse_args()
    counts = {split: np.zeros(len(LABELS), int) for split in ("train", "heldout")}
    for exp in sorted(p for p in args.cai.iterdir() if (p / "behavior3d").is_dir()):
        tops = sorted((exp / "behavior3d").glob("*.csv"))
        held = set(tops[2::5] or tops[:1])
        for beh in tops:
            pos = exp / "position" / beh.name
            if not pos.is_file():
                continue
            coords, conf, labels, fps = session(pos, beh)
            split = "heldout" if beh in held else "train"
            out = args.out / split / f"{exp.name}__{beh.stem}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                coords=coords,
                conf=conf,
                labels=labels,
                label_names=np.array(LABELS),
                fps=fps,
                animal=f"{exp.name}/{beh.stem}",
                view="top",
            )
            counts[split] += np.bincount(labels[labels >= 0], minlength=len(LABELS))
            print(
                f"{split:7s} {exp.name}/{beh.stem}: {len(labels)} frames at {fps:.1f} fps, "
                f"{int((labels >= 0).sum())} labelled",
                flush=True,
            )
    for split, c in counts.items():
        print(f"{split}: " + ", ".join(f"{n} {v}" for n, v in zip(LABELS, c)))


if __name__ == "__main__":
    main()
