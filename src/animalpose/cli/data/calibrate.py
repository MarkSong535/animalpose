"""Calibrate a multi-camera rig from known 3D landmarks (arena corners, a calibration object).

Write cams.json for animalpose predict --views.

Input JSON::

    {"cameras": [
       {"name": "top", "size": [640, 480],
        "points3d": [[x, y, z], ...],         # world units (e.g. cm), >= 6, non-coplanar
        "points2d": [[u, v], ...],
        "K": [[...]], "dist": [...]           # optional: known intrinsics -> PnP instead of DLT
       }, ...]}

Optionally refine all extrinsics jointly by bundle adjustment over matched
observations (``--refine obs.npz`` holding ``obs`` of shape (V, N, 2), NaN = unseen),
e.g. confident keypoints of a single animal seen by all cameras.

    animalpose data calibrate rig_points.json --out cams.json [--refine obs.npz]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from animalpose.multiview.calibration import bundle_adjust, calibrate_dlt, reprojection_errors, solve_extrinsics
from animalpose.multiview.triangulation import triangulate_dlt

NAME = "calibrate"


def add_arguments(parser) -> None:
    parser.add_argument("points")
    parser.add_argument("--out", default="cams.json")
    parser.add_argument("--refine")


def run(args) -> None:
    spec = json.loads(Path(args.points).read_text())
    cams = []
    for c in spec["cameras"]:
        X, x = np.asarray(c["points3d"], float), np.asarray(c["points2d"], float)
        size = tuple(c.get("size", (640, 480)))
        cam = (solve_extrinsics(c["K"], c.get("dist", [0] * 5), X, x, size, c["name"]) if "K" in c
               else calibrate_dlt(X, x, size, c["name"]))
        err = reprojection_errors([cam], X, x[None])[0]
        print(f"{c['name']}: mean reprojection error {err.mean():.2f}px (max {err.max():.2f})")
        cams.append(cam)
    if args.refine:
        obs = np.load(args.refine)["obs"]
        X0 = triangulate_dlt(cams, obs)
        ok = np.isfinite(X0).all(-1)
        cams, _, info = bundle_adjust(cams, X0[ok], obs[:, ok])
        print(f"bundle adjustment: {info['rmse_before']:.2f}px -> {info['rmse_after']:.2f}px")
    Path(args.out).write_text(json.dumps([c.to_dict() for c in cams], indent=2))
    print(f"wrote {args.out}")
