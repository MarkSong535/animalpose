"""Convert DeepLabCut labeled data (CollectedData_<scorer>.csv) to COCO keypoint JSON.

Keypoint names are kept as-is; map them onto the canonical vocabulary at training time with
``data.mapping`` (a registered dataset name or a YAML dict).

    animalpose data convert labeled-data/video1/CollectedData_me.csv --out data/video1.json
"""

from __future__ import annotations

import json
from pathlib import Path

from animalpose.data.dlc import read_dlc_csv, to_coco

NAME = "convert"


def add_arguments(parser) -> None:
    parser.add_argument("csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-root", default=None, help="project root to read image sizes from")


def run(args) -> None:
    images, inds, bps, coords = read_dlc_csv(args.csv)
    coco = to_coco(images, inds, bps, coords, Path(args.image_root) if args.image_root else None)
    Path(args.out).write_text(json.dumps(coco))
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} animals, keypoints: {bps}")
