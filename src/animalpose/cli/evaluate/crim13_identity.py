"""Two-mouse tracking scored against human identity labels (CRIM13 OOD, Lightning Pose release).

The unlabeled snippet videos are cut from the same recordings as some of the
labeled frames, but their frame offsets are not published. Each labeled frame
(and its context frames) is located in its session's snippet by thumbnail
matching. A session is used only when at least ``--min-matches`` labeled
frames agree on a single offset. The full snippet is then tracked
(``max_animals=2``). At every located labeled frame, the tracked animals are
matched to the black and white mouse by keypoint distance (Hungarian).

Scores:
* ``identity_accuracy``: share of labeled (frame, mouse) pairs whose track's
  majority mouse is that mouse;
* ``switches``: changes of the mouse a track covers between consecutive
  labeled frames;
* ``missed``: labeled mice with no track within ``--max-dist`` box widths.

Unlike ``animalpose eval real-tracking`` (coat-brightness proxy), identity here
comes from the annotators' black/white labels.

    animalpose eval crim13-identity --pose runs/pose_real_r18_side.pt --configs learned_embedding color_histogram
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.checkpoint import load_model

NAME = "crim13-identity"

EMBEDDERS = {"no_appearance": "none", "color_histogram": "color", "learned_embedding": "model"}


def thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_labels(csv: Path) -> dict[str, dict[int, np.ndarray]]:
    """session -> {frame index: (2, P, 2) black/white keypoints}."""
    import pandas as pd

    df = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
    parts = sorted({bp[len("black_mouse_"):] for _, bp, _ in df.columns if bp.startswith("black_mouse_")})
    out: dict[str, dict[int, np.ndarray]] = {}
    for rel, row in df.iterrows():
        session = Path(rel).parent.name
        frame = int(re.search(r"img(\d+)", rel).group(1))
        kp = np.array([[[row[(df.columns[0][0], f"{c}_mouse_{p}", a)] for a in ("x", "y")] for p in parts]
                       for c in ("black", "white")], float)
        out.setdefault(session, {})[frame] = kp
    return out


def locate(frames_thumb: np.ndarray, labeled: dict[Path, int], thresh: float) -> int | None:
    """Offset ``labeled frame index - snippet frame index`` agreed by most matches."""
    votes = Counter()
    for path, idx in labeled.items():
        d = np.abs(frames_thumb - thumb(cv2.imread(str(path)))).mean((1, 2))
        j = int(d.argmin())
        if d[j] < thresh:
            votes[idx - j] += 1
    return votes.most_common(1)[0] if votes else None


def score_session(result, gt: dict[int, np.ndarray], offset: int, n_frames: int, max_dist: float) -> dict:
    seq: dict[int, list[tuple[int, int]]] = {}  # track -> [(frame, mouse)]
    pairs, missed = [], 0
    for f_lab, kp in sorted(gt.items()):
        t = f_lab - offset
        if not 0 <= t < n_frames:
            continue
        gt_c = np.nanmean(kp, 1)  # (2, 2) centroids
        size = np.nanmax(np.nanmax(kp, 1) - np.nanmin(kp, 1))
        tids, cents = [], []
        for tid, tr in result.tracks.items():
            i = t - tr.frames[0]
            if 0 <= i < len(tr.frames) and np.isfinite(tr.keypoints[i]).any():
                ok = tr.conf[i] >= 0.3
                if ok.any():
                    tids.append(tid)
                    cents.append(tr.keypoints[i][ok].mean(0))
        if not tids:
            missed += 2
            continue
        cost = np.linalg.norm(np.asarray(cents)[:, None] - gt_c[None], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        hit = set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_dist * max(size, 1.0):
                seq.setdefault(tids[r], []).append((t, int(c)))
                pairs.append((tids[r], int(c)))
                hit.add(int(c))
        missed += 2 - len(hit)
    major = {tid: Counter(m for _, m in s).most_common(1)[0][0] for tid, s in seq.items()}
    correct = sum(major[tid] == m for tid, m in pairs)
    switches = sum(sum(a[1] != b[1] for a, b in zip(s[:-1], s[1:])) for s in seq.values())
    return {"labeled_mice": len(pairs) + missed, "assigned": len(pairs), "correct": int(correct),
            "switches": int(switches), "missed": int(missed), "tracks": len(seq)}


def add_arguments(parser) -> None:
    parser.add_argument("--pose", required=True)
    parser.add_argument("--detector", default="weights/detector_general.pt")
    parser.add_argument("--root", default="data/raw/lp_crim13/extracted/crim13")
    parser.add_argument("--split", default="OOD")
    parser.add_argument("--configs", nargs="+", default=list(EMBEDDERS))
    parser.add_argument("--w-app", type=float, default=0.2)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--match-thresh", type=float, default=1.5, help="mean abs grey difference of 64x48 thumbnails")
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--max-dist", type=float, default=0.5, help="max centroid distance, in mouse sizes")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", default="results/crim13_identity.json")


def run(args) -> None:
    root = Path(args.root)
    labels = load_labels(root / f"labels_{args.split}.csv")
    det, pose = load_model(args.detector), load_model(args.pose)
    sessions = []
    for video in sorted((root / f"videos_{args.split}").glob("*.mp4")):
        lab_dir = root / f"labeled-data_{args.split}" / video.stem
        if video.stem not in labels or not lab_dir.exists():
            continue
        cap, frames = cv2.VideoCapture(str(video)), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        thumbs = np.stack([thumb(f) for f in frames])
        labeled = {p: int(re.search(r"img(\d+)", p.name).group(1)) for p in lab_dir.glob("*.png")}
        best = locate(thumbs, labeled, args.match_thresh)
        status = {"video": video.stem, "frames": len(frames), "offset_votes": best[1] if best else 0}
        if best is None or best[1] < args.min_matches:
            print("skip", json.dumps(status), flush=True)
            continue
        status["offset"] = best[0]
        sessions.append((video.stem, frames, best[0], status))
        print("use", json.dumps(status), flush=True)
    res = {"sessions": [s[3] for s in sessions]}
    for name in args.configs:
        cfg = PipelineConfig(crop_size=args.crop_size, embedder=EMBEDDERS[name],
                             tracker=TrackerConfig(max_animals=2, w_app=args.w_app))
        pipe = PosePipeline(det, pose, config=cfg)
        per = {}
        for stem, frames, offset, _ in sessions:
            per[stem] = score_session(pipe.track(frames), labels[stem], offset, len(frames), args.max_dist)
        tot = {k: sum(p[k] for p in per.values()) for k in ("labeled_mice", "assigned", "correct", "switches", "missed")}
        res[name] = {"identity_accuracy": tot["correct"] / max(tot["assigned"], 1),
                     "coverage": tot["assigned"] / max(tot["labeled_mice"], 1), **tot, "per_session": per}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "per_session"}), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[f"{args.tag}{Path(args.pose).stem}@w_app={args.w_app}"] = res
    out.write_text(json.dumps(allres, indent=2))
