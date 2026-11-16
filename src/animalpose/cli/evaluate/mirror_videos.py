"""Side-view paw identity over time on the unlabeled mirror-mouse OOD videos.

There are no labels for these videos, so the bottom view is the reference.
Bottom-view paws are near-perfect (PCK 0.99), and the bottom view shares the
side view's horizontal axis. For each frame the side-view paw pair is labelled
by matching x with the bottom view; this was 97-98 % correct on the labeled
OOD frames. Only frames where the two side paws are at least
``--min-dx`` box widths apart in x are scored, since only there is the x
match decisive. Reported: how often the side view *alone* labels the pair the
same way, per frame and after :func:`animalpose.temporal.pairs.resolve_pair_identity`.

    animalpose eval mirror-videos --pose runs/pose_real_r18_side.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from animalpose.data.real import MIRROR_SPLIT_Y
from animalpose.multiview.symmetry import F_SHARED_X, bilateral_pairs, resolve_left_right
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.skeleton import CANONICAL
from animalpose.temporal.pairs import resolve_pair_identity
from animalpose.train.checkpoint import load_model

NAME = "mirror-videos"

PAWS = ["left_front_paw", "left_hind_paw"]


def add_arguments(parser) -> None:
    parser.add_argument("--pose", required=True)
    parser.add_argument("--detector", default="weights/detector_general.pt")
    parser.add_argument("--videos", default="data/raw/lp_mirror_mouse/extracted/mirror-mouse/videos_OOD")
    parser.add_argument("--frames", type=int, default=1500, help="frames per video (from the start)")
    parser.add_argument("--stride", type=int, default=2, help="frame step (videos are 250 Hz)")
    parser.add_argument("--min-dx", type=float, default=0.05)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default="results/mirror_videos.json")


def run(args) -> None:
    pipe = PosePipeline(load_model(args.detector), load_model(args.pose), config=PipelineConfig(crop_size=args.crop_size))
    pairs = bilateral_pairs()
    paw_p = [i for i, (lft, _) in enumerate(pairs) if CANONICAL.names[lft] in PAWS]
    fracs = (0.02, 0.05, 0.1, 0.2)
    agree = {"per_frame": [], **{f"temporal_{f}": [] for f in fracs}}
    per_video = {}
    for video in sorted(Path(args.videos).glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        side_k, margins, sizes, ref_swap, scored = [], [], [], [], []
        for t in range(args.frames):
            ok, img = cap.read()
            if not ok:
                break
            if t % args.stride:
                continue
            side, bottom = img[:MIRROR_SPLIT_Y], img[MIRROR_SPLIT_Y:]
            bs, _ = pipe.detect(side)
            bb, _ = pipe.detect(bottom)
            if len(bs) == 0 or len(bb) == 0:
                side_k.append(np.full((CANONICAL.num_keypoints, 2), np.nan)), margins.append(np.zeros(len(pairs)))
                sizes.append(np.nan), ref_swap.append(np.zeros(len(pairs), bool)), scored.append(np.zeros(len(pairs), bool))
                continue
            ks, cs, _, ms = pipe.estimate(side, bs[:1], return_margins=True)
            kb, cb, _ = pipe.estimate(bottom, bb[:1])
            K = np.stack([ks[0], kb[0] + [0, MIRROR_SPLIT_Y]])
            _, _, sw = resolve_left_right(K, np.stack([cs[0], cb[0]]), {(0, 1): F_SHARED_X}, view_priority=[1, 0])
            size = float(max(bs[0][2] - bs[0][0], bs[0][3] - bs[0][1]))
            dx = np.array([abs(ks[0][lft][0] - ks[0][rgt][0]) for lft, rgt in pairs])
            conf_ok = np.array([min(cs[0][lft], cs[0][rgt], cb[0][lft], cb[0][rgt]) >= 0.3 for lft, rgt in pairs])
            side_k.append(ks[0]), margins.append(ms[0]), sizes.append(size)
            ref_swap.append(sw[0]), scored.append((dx >= args.min_dx * size) & conf_ok)
        side_k, margins, ref_swap, scored = map(np.asarray, (side_k, margins, ref_swap, scored))
        sizes = np.nan_to_num(np.asarray(sizes), nan=np.nanmedian(sizes))
        res_v = {"frames": int(len(side_k)), "scored_pairs": int(scored[:, paw_p].sum())}
        # Per-frame: the decoded labels need no swap; temporal: its swaps match the reference's.
        hits = {"per_frame": ~ref_swap}
        for f in fracs:
            _, tsw = resolve_pair_identity(side_k, margins, pairs, sizes, motion_frac=f)
            hits[f"temporal_{f}"] = tsw == ref_swap
        for name, h in hits.items():
            sel = h[:, paw_p][scored[:, paw_p]]
            agree[name].append(sel)
            res_v[name] = float(sel.mean())
        per_video[video.stem] = res_v
        print(video.stem, {k: round(v, 3) if isinstance(v, float) else v for k, v in res_v.items()}, flush=True)
    res = {name: float(np.concatenate(v).mean()) for name, v in agree.items()}
    res["scored_pairs"] = int(sum(len(np.concatenate([x])) for x in agree["per_frame"]))
    res["per_video"] = per_video
    print(json.dumps({k: v for k, v in res.items() if k != "per_video"}, indent=1))
    out = Path(args.out)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[args.tag or Path(args.pose).stem] = res
    out.write_text(json.dumps(allres, indent=2))
