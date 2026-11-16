"""Evaluate detector + pose checkpoints on the held-out real datasets and write results/real.json / results/real.md.

The sets are the ones prepared by ``animalpose data prepare`` under ``data/processed``.

    animalpose eval real --detector runs/detector.pt --pose runs/pose.pt --tag synthetic_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from animalpose.data.datasets import CocoKeypoints
from animalpose.eval.benchmark import eval_coco
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.train.checkpoint import load_model

NAME = "real"

SETS = {  # name: (json, images, species)
    "crim13_ood (top, 2 mice)": ("crim13/val.json", "crim13/images", "mouse"),
    "rat7m_val (6 cams, 8-32° elev.)": ("rat7m/val.json", "rat7m/images", "rat"),
    "mirror_side_ood": ("mirror_mouse/side_val.json", "mirror_mouse/images", "mouse"),
    "mirror_bottom_ood": ("mirror_mouse/bottom_val.json", "mirror_mouse/images", "mouse"),
    "rat7m_subject2 (unseen rat)": ("rat7m_s2/val.json", "rat7m_s2/images", "rat"),
    "irodent (zero-shot, wild rodents)": ("irodent/test.json", "@irodent/IMAGE_ROOT", "rat"),
}


def add_arguments(parser) -> None:
    parser.add_argument("--detector", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--tag", default="model")
    parser.add_argument("--pair-decoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--out", default="results/real.json")


def run(args) -> None:
    pipe = PosePipeline(load_model(args.detector), load_model(args.pose),
                        config=PipelineConfig(crop_size=args.crop_size, pair_decoding=args.pair_decoding))
    res = {}
    for name, (js, imgs, species) in SETS.items():
        js_path = Path(args.data) / js
        if js_path.exists():
            img_root = (Path((Path(args.data) / imgs[1:]).read_text().strip()) if imgs.startswith("@")
                        else Path(args.data) / imgs)
            src = CocoKeypoints(js_path, img_root, "synthetic", species)
            res[name] = eval_coco(pipe, src, args.max_images)
            r = res[name]
            print(f"{name}: PCK@0.1 {r['pck@0.1']:.3f} PCK@0.05 {r['pck@0.05']:.3f} median err {r['median_err_px']:.1f}px "
                  f"(box {r['median_box_px']:.0f}px) AP50 {r['det_ap50']:.3f} "
                  f"conf-AUROC {r.get('conf_auroc_visible_vs_occluded', float('nan')):.3f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_res = json.loads(out.read_text()) if out.exists() else {}
    all_res[args.tag] = res
    out.write_text(json.dumps(all_res, indent=2))
