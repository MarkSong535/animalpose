"""Leave-one-animal-out behavior evaluation on home-cage DeepLabCut tracks.

For every animal (DLC tracks, plus any ``--session-dirs`` made by
``animalpose behavior sessions``: an MIT video, an ABODe cage, a BORIS-labelled
video), the behavior TCN is trained on all other animals with ``--config``
and scored on the held-out one (frame-wise macro-F1 on the 4 shared classes and
the 5 labelled ones, per camera view). Summaries are given per data source.
The held-out logits are scored for a grid of post-processing variants at once:
moving-median bout smoothing (``--prob-smooth``) and re-estimating the class prior
on the held-out animal (``--prior``), so one training run measures them all.
``--root none`` drops the DLC tracks.

    animalpose eval behavior-loao --config configs/behavior_real.yaml --tag dlc
    animalpose eval behavior-loao --config configs/behavior_real.yaml --tag smoothed \\
        data.real_temporal=runs/temporal_real.pt --smooth runs/temporal_real.pt
    animalpose eval behavior-loao --config configs/behavior_public.yaml --tag public --root none \\
        --session-dirs data/processed/behavior/mit data/processed/behavior/abode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from animalpose.config import load_config
from animalpose.data.behavior_sets import load_session_dir
from animalpose.data.tracks import load_sessions, smooth_sessions
from animalpose.eval.behavior import PROTOCOLS, behavior_report
from animalpose.train.checkpoint import load_model
from animalpose.train.loops import TRAINERS

NAME = "behavior-loao"


def add_arguments(parser) -> None:
    parser.add_argument("--config", default="configs/behavior_real.yaml")
    parser.add_argument("--root", default="data/raw/home_cage")
    parser.add_argument("--session-dirs", nargs="*", default=[])
    parser.add_argument("--smooth", default=None, help="temporal checkpoint applied to the held-out tracks too")
    parser.add_argument("--prob-smooth", default="1,3,5,9,15", help="moving-median windows (frames) to score")
    parser.add_argument("--prior", action=argparse.BooleanOptionalAction, default=True,
                        help="also score probabilities whose class prior is re-estimated on the held-out animal")
    parser.add_argument("--behavior-conf", type=float, default=0.6, help="keypoint confidence above which a point is used")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", default="results/behavior_loao.json")
    parser.add_argument("overrides", nargs="*")


def run(args) -> None:
    source = {}
    sessions = [] if args.root == "none" else load_sessions(args.root)
    source.update({s.animal: "homecage_dlc" for s in sessions})
    dlc = sorted(source)
    for d in args.session_dirs:
        extra = load_session_dir(d)
        source.update({s.animal: Path(d).name for s in extra})
        sessions += extra
    animals = sorted({s.animal for s in sessions})
    if args.smooth:
        sessions = smooth_sessions(sessions, load_model(args.smooth))
    protocols = PROTOCOLS
    smoothing = tuple(int(w) for w in args.prob_smooth.split(",") if w.strip())
    folds = {}
    for held in animals:
        train = [a for a in animals if a != held]
        cfg = load_config(args.config, args.overrides + [
            f"data.real_root={'null' if args.root == 'none' else args.root}",
            f"data.real_animals={[a for a in train if a in dlc]}",
            f"data.session_dirs={list(args.session_dirs)}",
            f"data.session_groups={[a for a in train if a not in dlc]}", "output=null"])
        model, _ = TRAINERS["behavior"](cfg)
        test = [s for s in sessions if s.animal == held]
        views = sorted({s.view for s in test})
        folds[held] = {v: behavior_report(model, [s for s in test if s.view == v], protocols, smoothing=smoothing,
                                         prior_adapt=args.prior, conf_thresh=args.behavior_conf) for v in views}
        shown = {}
        for v, res in folds[held].items():
            shown[v] = {}
            for proto in protocols:
                scored = [(r[proto]["macro_f1"], var) for var, r in res.items() if r[proto]["n_frames"]]
                best = max(scored) if scored else (float("nan"), "none")
                shown[v][proto] = {"raw": round(res["raw"][proto]["macro_f1"], 3),
                                   "best": [round(best[0], 3), best[1]]}
        print(held, json.dumps(shown), flush=True)
    summary = {}
    variants = [v for r in folds.values() for f in r.values() for v in f]
    for src in sorted(set(source.values())):
        for proto in protocols:
            for var in variants:
                vals = [r[var][proto]["macro_f1"] for a, f in folds.items() if source[a] == src
                        for r in f.values() if r[var][proto]["n_frames"]]
                if vals:
                    summary[f"{src}/{proto}/{var}"] = {"mean_macro_f1": float(np.mean(vals)), "min": float(np.min(vals)),
                                                       "max": float(np.max(vals)), "n_animal_views": len(vals)}
    print(args.tag, json.dumps(summary))
    out = Path(args.out)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres[args.tag] = {"summary": summary, "folds": folds, "config": args.config, "overrides": args.overrides,
                        "smooth": args.smooth, "protocols": protocols}
    out.write_text(json.dumps(allres, indent=2))
