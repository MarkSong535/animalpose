"""DeepLabCut exports (``CollectedData_<scorer>.csv``) as COCO keypoint annotations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_dlc_csv(path: str | Path):
    """Returns ``(image_paths, individuals, bodyparts, coords)`` with
    ``coords (N_images, N_individuals, N_bodyparts, 2)`` (NaN = unlabeled).

    Handles single-animal (3 header rows: scorer/bodyparts/coords) and multi-animal (4 header rows:
    scorer/individuals/bodyparts/coords) CSVs.
    """
    rows = list(csv.reader(Path(path).open()))
    multi = rows[1][0].strip().lower() == "individuals"
    n_header = 4 if multi else 3
    header = rows[:n_header]
    # Image path occupies 1 column (old format) or 3 (labeled-data/video/img.png split).
    first = next(i for i, v in enumerate(header[n_header - 1]) if v in ("x", "y"))
    cols = list(range(first, len(header[0])))
    individuals = header[1][first:] if multi else ["single"] * len(cols)
    bodyparts = header[n_header - 2][first:]
    coords_kind = header[n_header - 1][first:]
    ind_names = list(dict.fromkeys(individuals))
    bp_names = list(dict.fromkeys(b for b, i in zip(bodyparts, individuals) if i == ind_names[0]))
    images, data = [], []
    for r in rows[n_header:]:
        if not r:
            continue
        images.append("/".join(v for v in r[:first] if v))
        arr = np.full((len(ind_names), len(bp_names), 2), np.nan)
        for c, ind, bp, kind in zip(cols, individuals, bodyparts, coords_kind):
            if bp not in bp_names or r[c] == "":
                continue
            arr[ind_names.index(ind), bp_names.index(bp), 0 if kind == "x" else 1] = float(r[c])
        data.append(arr)
    return images, ind_names, bp_names, np.stack(data)


def to_coco(images, individuals, bodyparts, coords, image_root: Path | None = None, pad: float = 0.1) -> dict:
    """COCO dict for those keypoints; boxes come from the labelled points. Names are kept as-is, so
    map them onto the canonical vocabulary at training time with ``data.mapping``."""
    import cv2

    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "animal", "keypoints": bodyparts, "skeleton": []}]}
    ann_id = 0
    for i, (name, arr) in enumerate(zip(images, coords)):
        entry = {"id": i, "file_name": name}
        if image_root is not None:
            img = cv2.imread(str(image_root / name))
            if img is not None:
                entry["height"], entry["width"] = img.shape[:2]
        coco["images"].append(entry)
        for a, kp in enumerate(arr):
            ok = np.isfinite(kp).all(-1)
            if ok.sum() < 2:
                continue
            lo, hi = kp[ok].min(0), kp[ok].max(0)
            d = (hi - lo) * pad + 5
            x0, y0 = lo - d
            w, h = hi - lo + 2 * d
            flat = []
            for p, v in zip(kp, ok):
                flat += [float(p[0]), float(p[1]), 2] if v else [0.0, 0.0, 0]
            coco["annotations"].append({"id": ann_id, "image_id": i, "category_id": 1, "keypoints": flat,
                                        "num_keypoints": int(ok.sum()), "bbox": [float(x0), float(y0), float(w), float(h)],
                                        "area": float(w * h), "iscrowd": 0, "track_id": a})
            ann_id += 1
    return coco
