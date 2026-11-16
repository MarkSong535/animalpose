"""3D accuracy on Rat 7M against motion capture.

For synchronised frames of the held-out chunks, every camera image goes through
the detector and pose network; the six views are then triangulated with
confidence-weighted robust DLT (optionally after multi-view left/right
resolution) and compared with the mocap markers (millimetres).

    animalpose eval rat7m-3d --detector runs/detector_real.pt --pose runs/pose_real_r18.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

from animalpose.data.datasets import CocoKeypoints
from animalpose.data.real import rat7m_cameras
from animalpose.multiview.symmetry import resolve_left_right_cameras
from animalpose.multiview.triangulation import triangulate_robust
from animalpose.pipeline import PipelineConfig, PosePipeline
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS
from animalpose.train.checkpoint import load_model

NAME = "rat7m-3d"


def add_arguments(parser) -> None:
    parser.add_argument("--detector", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--mocap", default="data/raw/rat7m/mocap-s1-d1.mat")
    parser.add_argument("--data", default="data/processed/rat7m")
    parser.add_argument("--pair-decoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--max-groups", type=int, default=150)
    parser.add_argument("--out", default="results/rat7m_3d.json")


def run(args) -> None:
    mat = sio.loadmat(args.mocap, squeeze_me=True, struct_as_record=False)
    cams = rat7m_cameras(mat)
    frame_of = {n: np.asarray(getattr(mat["cameras"], n).frame) for n in cams}
    mapping = DATASET_MAPPINGS["rat7m"]
    gt_names = list(mapping)
    gt_idx = CANONICAL.indices([mapping[n] for n in gt_names])
    src = CocoKeypoints(Path(args.data) / "val.json", Path(args.data) / "images", "synthetic", "rat")
    groups: dict = {}
    for k, i in enumerate(src.ids):
        m = re.search(r"camera(\d+)-(\d+)_(\d+)", src.images[i]["file_name"])
        groups.setdefault((int(m.group(2)), int(m.group(3))), []).append((f"Camera{m.group(1)}", k))
    pipe = PosePipeline(load_model(args.detector), load_model(args.pose),
                        config=PipelineConfig(crop_size=args.crop_size, pair_decoding=args.pair_decoding))
    errs = {"raw": [], "lr_resolved": []}
    per_joint = {"raw": [], "lr_resolved": []}
    n_views, used = [], 0
    for (start, j), members in sorted(groups.items())[: args.max_groups]:
        if len(members) < 3:
            continue
        # Mocap sample: the per-camera frame->mocap mappings differ by a few samples; take the median.
        idx = [np.nonzero(frame_of[c] == start + j)[0] for c, _ in members]
        idx = [i[0] for i in idx if len(i)]
        if not idx:
            continue
        mi = int(np.median(idx))
        gt = np.stack([np.asarray(getattr(mat["mocap"], n), float)[mi] for n in gt_names])
        kps, confs, cl = [], [], []
        for cam, k in members:
            ann = src.sample(k)
            boxes, scores = pipe.detect(ann.image)
            if len(boxes) == 0:
                continue
            kp, cf, _ = pipe.estimate(ann.image, boxes[:1])
            kps.append(kp[0]), confs.append(cf[0]), cl.append(cams[cam])
        if len(cl) < 2:
            continue
        K, C = np.stack(kps), np.stack(confs)
        used += 1
        for name in errs:
            KK, CC = (resolve_left_right_cameras(K, C, cl)[:2] if name == "lr_resolved" else (K, C))
            X, _, u = triangulate_robust(cl, KK, CC, conf_thresh=0.3, reproj_thresh=15.0)
            d = np.linalg.norm(X[gt_idx] - gt, axis=-1)
            errs[name].append(d[np.isfinite(d)])
            per_joint[name].append(d)
            if name == "raw":
                n_views.append(u[:, gt_idx].sum(0))
    res = {"frames": used, "cameras_per_frame": float(np.mean([len(m) for m in groups.values()]))}
    for name in errs:
        e = np.concatenate(errs[name])
        pj = np.nanmean(np.stack(per_joint[name]), 0)
        res[name] = {"mpjpe_mm": float(e.mean()), "median_mm": float(np.median(e)),
                     "pck_20mm": float((e <= 20).mean()), "triangulated_fraction":
                     float(np.isfinite(np.stack(per_joint[name])).mean()),
                     "per_joint_mm": {mapping[n]: round(float(v), 1) for n, v in zip(gt_names, pj)}}
    res["mean_views_used"] = float(np.mean(np.concatenate(n_views)))
    print(json.dumps(res, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=2))
