"""``animalpose predict``: detect, track and classify a recording."""

from __future__ import annotations

from animalpose.api import predict


def add_subparser(sub) -> None:
    p = sub.add_parser("predict", help="run detection, tracking and behavior on a video")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--video", help="video file, or a folder of image frames (e.g. a TIF sequence)")
    p.add_argument("--views", nargs="+", help="synchronised videos, one per view")
    p.add_argument("--cameras", help="JSON list of Camera.to_dict() entries, one per view")
    p.add_argument("--synthetic", action="store_true", help="generate a 2-mouse demo clip")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--fps", type=float, default=None, help="default: from the video or its timestamps, else 30")
    p.add_argument("--no-overlay", action="store_true", help="skip writing overlay.mp4")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("overrides", nargs="*", help="dotted config overrides")
    p.set_defaults(run=_run)


def _run(args) -> None:
    if not (args.video or args.views or args.synthetic):
        raise SystemExit("predict needs --video, --views or --synthetic")
    res = predict(args.video, config=args.config, out=args.out, overrides=args.overrides, views=args.views,
                  cameras=args.cameras, synthetic=args.synthetic, max_frames=args.max_frames, fps=args.fps,
                  overlay=not args.no_overlay)
    for entry in res.drift:
        print(entry.line())
    if res.views:
        print(f"{res.animals} animals fused across {res.views} views -> {res.out}")
    else:
        print(f"{res.frames} frames, {res.tracks} tracks -> {res.out}")
