"""``animalpose label``: pseudo-label boxes and keypoints for unlabeled recordings."""

from __future__ import annotations

import json

from animalpose.api import label
from animalpose.cli.base import add_optionals, build
from animalpose.selftrain import LabelOptions


def add_subparser(sub) -> None:
    p = sub.add_parser("label", help="auto-label frames of every recording under --root")
    p.add_argument("--root", required=True, help="folder of videos and image folders, searched recursively")
    add_optionals(p, LabelOptions)
    p.set_defaults(run=_run)


def _run(args) -> None:
    print(json.dumps(label(args.root, options=build(LabelOptions, args))))
