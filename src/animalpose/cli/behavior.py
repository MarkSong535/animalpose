"""``animalpose behavior``: turn the tracks of a new recording into behavior labels.

``cluster`` groups its kinematics, ``name`` labels the groups by published signatures, ``label``
writes train/held-out sessions, ``plot`` draws the ethogram of a run, and ``sessions`` rebuilds the
public labeled datasets from their videos.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from animalpose.api import behavior
from animalpose.behavior.clusters import parse_mapping
from animalpose.behavior.plot import plot_run


def add_subparser(sub) -> None:
    p = sub.add_parser("behavior", help="bootstrap behavior labels from pose tracks")
    b = p.add_subparsers(dest="task", required=True, metavar="TASK")

    c = b.add_parser("cluster", help="cluster the kinematics of a tracks.npz")
    c.add_argument("--tracks", required=True)
    c.add_argument("--video", required=True)
    c.add_argument("--fps", type=float, default=30.0)
    c.add_argument("--train-range", type=float, nargs=2, default=[0.0, 0.8])
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--out", required=True)
    c.set_defaults(run=_cluster)

    n = b.add_parser("name", help="name the clusters by rule")
    n.add_argument("--clusters", required=True)
    n.add_argument("--agree-with", default=None, help="labels.json of hand-named clusters, to score the rules")
    n.set_defaults(run=_name)

    lb = b.add_parser("label", help="write session files from a cluster=behavior mapping")
    lb.add_argument("--clusters", required=True)
    lb.add_argument("--map", nargs="+", required=True, help="cluster=behavior, e.g. 0=resting")
    lb.add_argument("--min-bout", type=float, default=0.5, help="seconds")
    lb.add_argument("--name", default="recording")
    lb.add_argument("--view", default="top")
    lb.add_argument("--out", required=True)
    lb.set_defaults(run=_label)

    pl = b.add_parser("plot", help="ethogram and annotated clip of a run")
    pl.add_argument("--run", dest="run_dir", required=True, help="output folder of `animalpose predict`")
    pl.add_argument("--video", required=True)
    pl.add_argument("--clip-start", type=float, default=0.0, help="seconds")
    pl.add_argument("--clip-seconds", type=float, default=30.0)
    pl.add_argument("--top", type=int, default=3)
    pl.add_argument("--heldout-from", type=float, default=None, help="seconds; drawn as a line")
    pl.set_defaults(run=_plot)

    s = b.add_parser("sessions", help="track and label the public behavior datasets")
    s.add_argument("dataset", choices=["mit", "abode", "boris"])
    s.add_argument("--root", help="extracted dataset folder (mit, abode)")
    s.add_argument("--videos-list", help="text file, one video path per line (boris)")
    s.add_argument("--boris-dir", help="folder of BORIS projects covering the listed videos (boris)")
    s.add_argument("--pose", default="weights/pose_general.pt")
    s.add_argument("--detector", default="weights/detector_general.pt")
    s.add_argument("--crop-size", type=int, default=192)
    s.add_argument("--stride", type=int, default=None, help="frame step (default: mit 2, abode 2, boris 1)")
    s.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    s.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    s.add_argument("--out", default="data/processed/behavior")
    s.set_defaults(run=_sessions)


def _cluster(args) -> None:
    print(json.dumps(behavior("cluster", tracks=args.tracks, video=args.video, out=args.out, fps=args.fps,
                              train_range=args.train_range, k=args.k), indent=1))


def _name(args) -> None:
    report = behavior("name", clusters=args.clusters, agree_with=args.agree_with)
    naming = report["naming"]
    order = sorted(naming, key=lambda c: -float(naming[c]["share_of_train"] or 0.0))
    for c in order:
        f = naming[c]["features"]
        print(f"cluster {c} -> {naming[c]['name']:<9} share {float(naming[c]['share_of_train'] or 0):.3f}  "
              + "  ".join(f"{k}={f[k]:.2f}" for k in ("speed", "length", "turn", "posture", "head_motion", "bend")))
    if "agreement" in report:
        print("rule naming vs hand naming:", json.dumps(report["agreement"]))
    print("map:", " ".join(f"{c}={naming[c]['name']}" for c in sorted(naming, key=int)))


def _label(args) -> None:
    print(json.dumps(behavior("label", clusters=args.clusters, mapping=parse_mapping(args.map), out=args.out,
                              name=args.name, view=args.view, min_bout=args.min_bout)))


def _plot(args) -> None:
    share = plot_run(args.run_dir, args.video, clip_start=args.clip_start, clip_seconds=args.clip_seconds,
                     top=args.top, heldout_from=args.heldout_from)
    print(json.dumps({k: round(v, 3) for k, v in share.items()}))


def _sessions(args) -> None:
    from animalpose.data import behavior_sets as bs
    from animalpose.pipeline import PipelineConfig, PosePipeline
    from animalpose.train.checkpoint import load_model

    pipe = PosePipeline(load_model(args.detector), load_model(args.pose), config=PipelineConfig(crop_size=args.crop_size))
    out = Path(args.out)

    def mine(items):
        return [x for i, x in enumerate(sorted(items)) if i % args.nshards == args.shard]

    if args.dataset == "mit":
        stems = mine(p.stem for p in Path(args.root).rglob("*.mpg"))
        done = bs.build_mit(pipe, Path(args.root), out, stride=args.stride or 2, only=stems)
    elif args.dataset == "abode":
        snips = mine(p.parent.name for p in Path(args.root).rglob("Snippets/*/Video.mp4"))
        done = bs.build_abode(pipe, Path(args.root), out, stride=args.stride or 2, only=snips)
    else:
        videos = [Path(v.strip()) for v in Path(args.videos_list).read_text().splitlines()
                  if v.strip() and not v.startswith("#")]
        done = bs.build_boris_videos(pipe, mine(videos), Path(args.boris_dir) if args.boris_dir else None, out,
                                      stride=args.stride or 1)
    print(f"{args.dataset} shard {args.shard}/{args.nshards}: {len(done)} done -> {out}")
