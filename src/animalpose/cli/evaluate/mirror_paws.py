"""Side-view paw identity on mirror-mouse OOD (head-fixed mouse, side view + mirrored bottom view).

Reports, for the side view, paw PCK@0.1 and how left/right pairs fail:
*collapsed* (both labels on one paw) or *swapped*. Decoders compared:

* ``argmax``: independent per-keypoint argmax;
* ``pairs``: :func:`animalpose.pose.model.decode_pairs` (distinct peaks per pair);
* ``+bottom``: the side labels are then made consistent with the bottom view,
  which shares the horizontal image axis with the side view (``F_SHARED_X``).

    animalpose eval mirror-paws --pose runs/pose_real_r18.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from animalpose.data.datasets import CocoKeypoints
from animalpose.data.real import MIRROR_SPLIT_Y
from animalpose.multiview.symmetry import F_SHARED_X, resolve_left_right
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.skeleton import CANONICAL
from animalpose.train.checkpoint import load_model

NAME = "mirror-paws"

PAIRS = [CANONICAL.indices(["left_front_paw", "right_front_paw"]), CANONICAL.indices(["left_hind_paw", "right_hind_paw"])]


def score(rows) -> dict:
    paws = [j for p in PAIRS for j in p]
    ok = n = ok_all = n_all = coll = swap = npair = 0
    for k, g, v, s in rows:
        hit = np.linalg.norm(k - g, axis=-1) <= 0.1 * s
        ok, n = ok + hit[paws][v[paws]].sum(), n + v[paws].sum()
        ok_all, n_all = ok_all + hit[v].sum(), n_all + v.sum()
        for a, b in PAIRS:
            if v[a] and v[b] and np.linalg.norm(g[a] - g[b]) > 0.1 * s:
                npair += 1
                coll += np.linalg.norm(k[a] - k[b]) < 0.04 * s
                swap += (np.linalg.norm(k[a] - g[b]) <= 0.1 * s) and (np.linalg.norm(k[b] - g[a]) <= 0.1 * s)
    return {"paw_pck@0.1": ok / n, "side_pck@0.1": ok_all / n_all, "collapsed_pairs": coll / npair,
            "swapped_pairs": swap / npair, "n_paws": int(n)}


def add_arguments(parser) -> None:
    parser.add_argument("--pose", required=True)
    parser.add_argument("--detector", default="weights/detector_general.pt")
    parser.add_argument("--data", default="data/processed/mirror_mouse")
    parser.add_argument("--split", default="val")
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default="results/mirror_paws.json")


def run(args) -> None:
    pipe = PosePipeline(load_model(args.detector), load_model(args.pose), config=PipelineConfig(crop_size=args.crop_size))
    side = CocoKeypoints(Path(args.data) / f"side_{args.split}.json", Path(args.data) / "images", "synthetic", "mouse")
    bot = CocoKeypoints(Path(args.data) / f"bottom_{args.split}.json", Path(args.data) / "images", "synthetic", "mouse")
    bidx = {bot.images[i]["file_name"].split("/", 1)[1]: k for k, i in enumerate(bot.ids)}
    rows: dict[str, list] = {m: [] for m in ("argmax", "pairs", "argmax+bottom", "pairs+bottom")}
    for i in range(len(side)):
        a = side.sample(i)
        name = side.images[side.ids[i]]["file_name"].split("/", 1)[1]
        b = bot.sample(bidx[name]) if name in bidx else None
        g, v = a.keypoints[0], a.state[0] == 2
        s = max(a.boxes[0][2] - a.boxes[0][0], a.boxes[0][3] - a.boxes[0][1])
        for dec in ("argmax", "pairs"):
            pipe.pose.pair_decoding = dec == "pairs"
            k, c, _ = pipe.estimate(a.image, a.boxes)
            rows[dec].append((k[0], g, v, s))
            if b is None:
                continue
            kb, cb, _ = pipe.estimate(b.image, b.boxes)
            K = np.stack([k[0], kb[0] + [0, MIRROR_SPLIT_Y]])
            K, _, _ = resolve_left_right(K, np.stack([c[0], cb[0]]), {(0, 1): F_SHARED_X}, view_priority=[1, 0])
            rows[f"{dec}+bottom"].append((K[0], g, v, s))
    res = {m: score(r) for m, r in rows.items()}
    for m, r in res.items():
        print(f"{m:14s} " + " ".join(f"{k} {v:.3f}" for k, v in r.items() if k != "n_paws"))
    out = Path(args.out)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[args.tag or Path(args.pose).stem] = res
    out.write_text(json.dumps(allres, indent=2))
