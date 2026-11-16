"""Frame-wise behavior scores of one or more classifiers on labeled session files (``*.npz``).

Used to compare a general classifier with a per-recording fine-tune on the same held-out minutes
(``--model`` the general checkpoint, then the fine-tuned one). Each model is scored on the classes
that appear in the labels, and every post-processing variant of
:func:`animalpose.eval.behavior.behavior_report` (moving-median bout smoothing, class-prior
re-estimation on the recording being scored) is reported, because a new recording usually has a
different behavior mix than the training sessions.

    animalpose eval behavior-sessions --model runs/behavior_real.pt runs/behavior_mine.pt \\
        --sessions data/processed/behavior/mine/heldout --out results/behavior_mine.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from animalpose.data.behavior_sets import load_session_dir
from animalpose.data.tracks import IGNORE
from animalpose.eval.behavior import behavior_report
from animalpose.train.checkpoint import load_model

NAME = "behavior-sessions"


def _rnd(x, nd: int = 4):
    if isinstance(x, dict):
        return {k: _rnd(v, nd) for k, v in x.items()}
    if isinstance(x, float):
        return round(x, nd)
    return x


def add_arguments(parser) -> None:
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--conf", type=float, default=0.6, help="keypoint confidence above which a point is used")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="score these classes for every model, so a general and a fine-tuned classifier are "
                             "compared on the same label set (default: the classes that appear in the labels)")
    parser.add_argument("--prob-smooth", default="1,5,15", help="moving-median windows (frames) to score")
    parser.add_argument("--prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default=None)


def run(args) -> None:
    smoothing = tuple(int(w) for w in args.prob_smooth.split(",") if w.strip())
    for path in args.model:
        model = load_model(path).eval()
        classes = list(model.classes)
        sessions = load_session_dir(args.sessions, classes=classes)
        labels = np.concatenate([np.asarray(s.labels) for s in sessions])
        present = sorted(set(int(c) for c in labels[labels != IGNORE]))
        shared = [c for c in (args.classes or [classes[c] for c in present]) if c in classes]
        rep = behavior_report(model, sessions, {"present": shared}, smoothing=smoothing,
                              prior_adapt=args.prior, conf_thresh=args.conf)
        best = max(rep.items(), key=lambda kv: kv[1]["present"]["macro_f1"])
        res = {"model": path, "frames": int((labels != IGNORE).sum()), "classes_present": shared,
               "label_share": {classes[c]: round(float((labels == c).mean()), 3) for c in present},
               "variants": {k: _rnd({m: v for m, v in r["present"].items() if m != "confusion"})
                            for k, r in rep.items()},
               "best": [best[0], round(best[1]["present"]["macro_f1"], 4)],
               "per_class_f1_raw": {k: round(v, 3) for k, v in rep["raw"]["present"]["per_class_f1"].items()}}
        print(Path(path).name, json.dumps({"best": res["best"], "raw": round(rep["raw"]["present"]["macro_f1"], 4),
                                           "per_class_f1_raw": res["per_class_f1_raw"]}))
        if args.out:
            out = Path(args.out)
            allres = json.loads(out.read_text()) if out.exists() else {}
            allres[path] = res
            out.write_text(json.dumps(allres, indent=2))
