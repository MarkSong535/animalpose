"""The DeepLabCut CSV path end to end (project CSV -> COCO -> canonical keypoints ->
training crop) and the letterbox inverse map. Synthetic scene batches, degradation at
the camera scale and the COCO/real loaders are pinned by test_train_paths,
test_tracks_and_degrade and test_real_converters, so they are not repeated here."""

import json

import cv2
import numpy as np
import torch

from animalpose.data.datasets import CocoKeypoints, PoseCropDataset, letterbox
from animalpose.data.dlc import read_dlc_csv, to_coco
from animalpose.skeleton import CANONICAL


def _write_dlc_project(tmp):
    (tmp / "labeled-data" / "v1").mkdir(parents=True)
    bps = ["snout", "leftear", "rightear", "tailbase"]
    header = [["scorer", "", ""] + ["me"] * 16,
              ["individuals", "", ""] + ["mouse1"] * 8 + ["mouse2"] * 8,
              ["bodyparts", "", ""] + [b for b in bps for _ in range(2)] * 2,
              ["coords", "", ""] + ["x", "y"] * 8]
    rows = []
    for i in range(3):
        name = f"img{i}.png"
        cv2.imwrite(str(tmp / "labeled-data" / "v1" / name), np.full((120, 160, 3), 128, np.uint8))
        vals = [str(v) for v in (20 + i, 30, 15, 25, 25, 25, 50, 32)] + [str(v) for v in (100, 80, 95, 75, 105, 75, 130, 82)]
        if i == 2:
            vals[8:10] = ["", ""]  # mouse2 snout unlabeled
        rows.append(["labeled-data", "v1", name] + vals)
    path = tmp / "labeled-data" / "v1" / "CollectedData_me.csv"
    with path.open("w") as fh:
        for r in header + rows:
            fh.write(",".join(r) + "\n")
    return path


def test_dlc_to_coco_to_canonical(tmp_path):
    csv_path = _write_dlc_project(tmp_path)
    images, inds, bps, coords = read_dlc_csv(csv_path)
    assert inds == ["mouse1", "mouse2"] and bps == ["snout", "leftear", "rightear", "tailbase"]
    assert coords.shape == (3, 2, 4, 2) and np.isnan(coords[2, 1, 0]).all()
    coco = to_coco(images, inds, bps, coords, tmp_path)
    js = tmp_path / "ann.json"
    js.write_text(json.dumps(coco))
    mapping = {"snout": "nose", "leftear": "left_ear", "rightear": "right_ear", "tailbase": "tail_base"}
    ds = CocoKeypoints(js, tmp_path, mapping)
    ann = ds.sample(2)
    assert ann.keypoints.shape == (2, 25, 2)
    nose = CANONICAL.index("nose")
    assert ann.state[0, nose] == 2 and ann.state[1, nose] == 0
    assert ann.state[:, CANONICAL.index("left_front_paw")].max() == 0
    crop, kp, state, ids, degraded = PoseCropDataset(ds, 4, 64)[0]
    assert crop.shape == (3, 64, 64) and kp.shape == (25, 2) and state.dtype == torch.int64
    assert not degraded


def test_letterbox_maps_back():
    img = np.zeros((100, 300, 3), np.uint8)
    out, s, pad = letterbox(img, (128, 160))
    assert out.shape == (128, 160, 3)
    p = np.array([300.0, 100.0]) * s + pad
    assert np.allclose(p, [160, 128 - pad[1]], atol=1)
