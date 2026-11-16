"""Pose and detection on held-out real images under fixed, increasingly severe degradations.

Each condition is deterministic and scaled to the animal's size, so models are compared
on identical images:

* ``clean``
* ``motion_10`` / ``motion_20``: motion blur of 10 / 20 % of the body length
* ``defocus``: disk blur, radius 5 % of the body length
* ``defocus_px_<r>``: disk blur of a fixed radius in **image pixels**, which is what a real
  lens does (the animal's size does not matter) and what ``animalpose data audit``
  measures for a recording; train on it with ``data.camera_profile``
* ``tiny_32``: the frame is downscaled so the animal is 32 px long, then upscaled back
* ``arena``: what a top camera over a 3-chamber arena gives (animal ~35 px, 15 % motion
  blur, grayscale, halved contrast, sensor noise)

    animalpose eval blur --pairs runs/detector_real.pt:runs/pose_real_r18_side.pt \\
        runs/detector_real_robust.pt:runs/pose_real_r18_robust.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from animalpose.data.datasets import CocoKeypoints
from animalpose.data.degrade import disk_kernel, motion_kernel
from animalpose.eval.benchmark import eval_coco
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.train.checkpoint import load_model

NAME = "blur"

SETS = {
    "crim13_ood": ("crim13/val.json", "crim13/images", "mouse"),
    "rat7m_val": ("rat7m/val.json", "rat7m/images", "rat"),
    "mirror_side_ood": ("mirror_mouse/side_val.json", "mirror_mouse/images", "mouse"),
}


def _tiny(img: np.ndarray, size: float, target: float) -> np.ndarray:
    f = min(1.0, target / max(size, 1.0))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply(cond: str, img: np.ndarray, size: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi)
    x = img.astype(np.float32)
    if cond.startswith("motion_"):
        x = cv2.filter2D(x, -1, motion_kernel(max(3.0, int(cond[7:]) / 100 * size), angle))
    elif cond.startswith("defocus_px_"):  # a fixed sensor PSF, as animalpose data audit measures
        x = cv2.filter2D(x, -1, disk_kernel(float(cond[len("defocus_px_"):])))
    elif cond == "defocus":
        x = cv2.filter2D(x, -1, disk_kernel(max(1.0, 0.05 * size)))
    elif cond == "tiny_32":
        x = _tiny(x, size, 32)
    elif cond == "arena":
        x = _tiny(x, size, 35)
        x = cv2.filter2D(x, -1, motion_kernel(max(3.0, 0.15 * size), angle))
        x = np.repeat(x.mean(-1, keepdims=True), 3, -1)
        x = x.mean() + (x - x.mean()) * 0.5
        x = x + rng.normal(0, 4, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


class Degraded:
    """Wraps a CocoKeypoints source; degrades every image in a fixed way."""

    def __init__(self, src, cond: str):
        self.src, self.cond = src, cond

    def __len__(self):
        return len(self.src)

    def sample(self, i, *a):
        ann = self.src.sample(i, *a)
        if self.cond != "clean" and len(ann.boxes):
            size = float(np.median(np.maximum(ann.boxes[:, 2] - ann.boxes[:, 0], ann.boxes[:, 3] - ann.boxes[:, 1])))
            ann.image = apply(self.cond, ann.image, size, seed=i)
        return ann


def add_arguments(parser) -> None:
    parser.add_argument("--pairs", nargs="+", required=True, help="detector.pt:pose.pt")
    parser.add_argument("--conditions", nargs="+",
                        default=["clean", "motion_10", "motion_20", "defocus", "tiny_32", "arena"])
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--out", default="results/blur.json")


def run(args) -> None:
    out = Path(args.out)
    allres = json.loads(out.read_text()) if out.exists() else {}
    for pair in args.pairs:
        det, pose = pair.split(":")
        pipe = PosePipeline(load_model(det), load_model(pose), config=PipelineConfig(crop_size=args.crop_size))
        tag = f"{Path(det).stem}+{Path(pose).stem}"
        res = allres.setdefault(tag, {})
        for name, (js, imgs, species) in SETS.items():
            if not (Path(args.data) / js).exists():
                continue
            src = CocoKeypoints(Path(args.data) / js, Path(args.data) / imgs, "synthetic", species)
            for cond in args.conditions:
                r = eval_coco(pipe, Degraded(src, cond), args.max_images)
                res.setdefault(name, {})[cond] = {"pck@0.1": r["pck@0.1"], "det_ap50": r["det_ap50"]}
                print(f"{tag} {name} {cond}: PCK@0.1 {r['pck@0.1']:.3f} AP50 {r['det_ap50']:.3f}", flush=True)
        out.write_text(json.dumps(allres, indent=2))
