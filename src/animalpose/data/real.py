"""Converters from public rodent datasets to canonical COCO keypoint JSON.

Each converter writes images plus one COCO JSON per split whose keypoint
names are already canonical (``mapping: synthetic`` / identity), so training
code treats all sources alike. Raw downloads are only read, never modified.

* Lightning Pose **mirror-mouse** (CC BY 4.0): head-fixed mouse, one camera
  with a 45° mirror. Each frame is split into its *side* view and *bottom*
  view, which become separate images. Splits follow the dataset: InD -> train,
  OOD (different sessions/animals) -> val.
* Lightning Pose **CRIM13** (CC BY 4.0): two mice (black, white), top view.
* **Rat 7M** (CC BY 4.0): six calibrated cameras (8-32° elevation) and 3D
  motion capture. 2D labels are obtained by projecting the 3D markers.
  Lateral markers on the far side of the body are marked occluded
  (visibility 1), using the camera position relative to the body's left axis.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import cv2
import numpy as np

from animalpose.multiview.camera import Camera
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS

MIRROR_MOUSE = {"paw1LH": "left_hind_paw", "paw2LF": "left_front_paw", "paw3RF": "right_front_paw",
                "paw4RH": "right_hind_paw", "tailBase": "tail_base", "tailMid": "tail_mid", "nose": "nose"}
MIRROR_SPLIT_Y = 160  # side view above, mirrored bottom view below


class CocoWriter:
    def __init__(self):
        self.images, self.annotations = [], []

    def add(self, file_name: str, size_hw, animals: list[dict]) -> None:
        img_id = len(self.images)
        self.images.append({"id": img_id, "file_name": file_name, "height": int(size_hw[0]), "width": int(size_hw[1])})
        for a in animals:
            kp = np.asarray(a["keypoints"], float)  # (K, 2) canonical, NaN unlabeled
            vis = np.asarray(a["visibility"], int)  # (K,) 0/1/2
            flat = []
            for p, v in zip(kp, vis):
                flat += [float(p[0]), float(p[1]), int(v)] if v > 0 and np.isfinite(p).all() else [0.0, 0.0, 0]
            x0, y0, x1, y1 = a["box"]
            self.annotations.append({"id": len(self.annotations), "image_id": img_id, "category_id": 1,
                                     "keypoints": flat, "num_keypoints": int((vis > 0).sum()),
                                     "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                     "area": float((x1 - x0) * (y1 - y0)), "iscrowd": 0,
                                     "track_id": int(a.get("track_id", -1))})

    def save(self, path: Path) -> dict:
        coco = {"images": self.images, "annotations": self.annotations,
                "categories": [{"id": 1, "name": "animal", "keypoints": CANONICAL.names, "skeleton": []}]}
        path.write_text(json.dumps(coco))
        return {"images": len(self.images), "animals": len(self.annotations)}


def _box(kp: np.ndarray, pad: float, size_hw, min_side: float = 16.0) -> list[float] | None:
    ok = np.isfinite(kp).all(-1)
    if ok.sum() < 2:
        return None
    lo, hi = kp[ok].min(0), kp[ok].max(0)
    d = np.maximum(hi - lo, min_side) * pad
    lo, hi = lo - d, hi + d
    h, w = size_hw
    return [max(0.0, lo[0]), max(0.0, lo[1]), min(w, hi[0]), min(h, hi[1])]


def _canonical(names, xy, mapping) -> np.ndarray:
    out = np.full((CANONICAL.num_keypoints, 2), np.nan)
    for n, p in zip(names, xy):
        if n in mapping:
            out[CANONICAL.index(mapping[n])] = p
    return out


# ----------------------------------------------------------------------------- mirror-mouse

def convert_mirror_mouse(root: Path, out: Path) -> dict:
    import pandas as pd

    root, out = Path(root), Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, csv in (("train", "labels_InD.csv"), ("val", "labels_OOD.csv")):
        df = pd.read_csv(root / csv, header=[0, 1, 2], index_col=0)
        writers = {"side": CocoWriter(), "bottom": CocoWriter()}
        for rel, row in df.iterrows():
            img = cv2.imread(str(root / rel))
            if img is None:
                continue
            views = (("side", "_top", 0, MIRROR_SPLIT_Y), ("bottom", "_bot", MIRROR_SPLIT_Y, img.shape[0]))
            for view, suffix, y0, y1 in views:
                names, xy = [], []
                for bp in MIRROR_MOUSE:
                    x, y = row[(df.columns[0][0], bp + suffix, "x")], row[(df.columns[0][0], bp + suffix, "y")]
                    names.append(bp)
                    xy.append([x, y - y0])
                kp = _canonical(names, np.asarray(xy, float), MIRROR_MOUSE)
                part = img[y0:y1]
                box = _box(kp, 0.25, part.shape[:2])
                if box is None:
                    continue
                name = f"{view}/{rel.replace('/', '_').replace('.png', '.jpg')}"
                (out / "images" / view).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "images" / name), part, [cv2.IMWRITE_JPEG_QUALITY, 95])
                vis = np.where(np.isfinite(kp).all(-1), 2, 0)
                writers[view].add(name, part.shape[:2], [{"keypoints": kp, "visibility": vis, "box": box}])
        for view, w in writers.items():
            stats[f"{view}_{split}"] = w.save(out / f"{view}_{split}.json")
    return stats


# ----------------------------------------------------------------------------- CRIM13 (Lightning Pose)

def _lp_crim13_mapping(bodypart: str) -> tuple[int, str] | None:
    """'black_mouse_nose'-style names -> (animal index, canonical name)."""
    m = re.match(r"(black|white)_mouse_(.+)$", bodypart) or re.match(r"(.+?)_(black|white)$", bodypart)
    if m is None:
        return None
    color, part = (m.group(1), m.group(2)) if m.group(1) in ("black", "white") else (m.group(2), m.group(1))
    canon = {"nose": "nose", "right_ear": "right_ear", "left_ear": "left_ear", "top_of_neck": "neck",
             "neck": "neck", "right_rear_knee": "right_hind_knee", "left_rear_knee": "left_hind_knee",
             "base_tail": "tail_base", "tail_base": "tail_base", "base_of_tail": "tail_base",
             "right_side": "right_hip", "left_side": "left_hip"}.get(part)
    return (0 if color == "black" else 1, canon) if canon else None


def convert_lp_crim13(root: Path, out: Path) -> dict:
    import pandas as pd

    root, out = Path(root), Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, csv in (("train", "labels_InD.csv"), ("val", "labels_OOD.csv")):
        if not (root / csv).exists():
            continue
        df = pd.read_csv(root / csv, header=[0, 1, 2], index_col=0)
        scorer = df.columns[0][0]
        parts = list(dict.fromkeys(df.columns.get_level_values(1)))
        unmapped = [p for p in parts if _lp_crim13_mapping(p) is None]
        w = CocoWriter()
        for rel, row in df.iterrows():
            img = cv2.imread(str(root / rel))
            if img is None:
                continue
            animals = []
            for a in (0, 1):
                kp = np.full((CANONICAL.num_keypoints, 2), np.nan)
                for p in parts:
                    m = _lp_crim13_mapping(p)
                    if m and m[0] == a:
                        kp[CANONICAL.index(m[1])] = [row[(scorer, p, "x")], row[(scorer, p, "y")]]
                box = _box(kp, 0.3, img.shape[:2])
                if box is not None:
                    vis = np.where(np.isfinite(kp).all(-1), 2, 0)
                    session = Path(rel).parent.name
                    animals.append({"keypoints": kp, "visibility": vis, "box": box,
                                    "track_id": (zlib.crc32(session.encode()) % 100000) * 16 + a})
            name = rel.replace("/", "_").replace(".png", ".jpg")
            cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            w.add(name, img.shape[:2], animals)
        stats[split] = {**w.save(out / f"{split}.json"), "unmapped_bodyparts": unmapped}
    return stats


# ----------------------------------------------------------------------------- Rat 7M

_RAT7M_LATERAL = {"L": ["HeadL", "HipL", "ElbowL", "ArmL", "ShoulderL", "KneeL", "ShinL"],
                  "R": ["HipR", "ElbowR", "ArmR", "ShoulderR", "KneeR", "ShinR"]}


def rat7m_cameras(mat) -> dict[str, Camera]:
    """MATLAB camera structs (row-vector convention, 1-based pixels) -> :class:`Camera`."""
    cams = {}
    for name in mat["cameras"]._fieldnames:
        c = getattr(mat["cameras"], name)
        K = np.asarray(c.IntrinsicMatrix, float).T.copy()
        K[0, 2] -= 1.0
        K[1, 2] -= 1.0
        rad = np.asarray(c.RadialDistortion, float).ravel()
        tan = np.asarray(c.TangentialDistortion, float).ravel()
        dist = [rad[0], rad[1] if len(rad) > 1 else 0.0, tan[0], tan[1], rad[2] if len(rad) > 2 else 0.0]
        cams[name] = Camera(K, np.asarray(c.rotationMatrix, float).T, np.asarray(c.translationVector, float),
                            dist, (1328, 1048), name)
    return cams


def rat7m_visibility(markers: dict[str, np.ndarray], cam: Camera) -> dict[str, int]:
    """2 for midline markers and lateral markers facing the camera, 1 for far-side lateral markers."""
    spine = markers["SpineF"] - markers["SpineL"]
    left = np.cross([0.0, 0.0, 1.0], spine)
    vis = {}
    for name, X in markers.items():
        side = 1.0 if name in _RAT7M_LATERAL["L"] else (-1.0 if name in _RAT7M_LATERAL["R"] else 0.0)
        if side == 0.0 or not np.isfinite(left).all() or np.linalg.norm(left) < 1e-6:
            vis[name] = 2
        else:
            vis[name] = 2 if np.dot(side * left, cam.center - X) > 0 else 1
    return vis


def convert_rat7m(mocap_mat: Path, video_dir: Path, out: Path, every: int = 25,
                  val_starts: tuple[int, ...] | None = (105000, 122500)) -> dict:
    """Rat 7M subject/day -> COCO with mocap projected to every camera. ``val_starts=None``
    puts every chunk in ``val`` (a held-out subject used only for testing)."""
    import scipy.io as sio

    mat = sio.loadmat(str(mocap_mat), squeeze_me=True, struct_as_record=False)
    cams = rat7m_cameras(mat)
    marker_names = list(mat["mocap"]._fieldnames)
    mocap = {n: np.asarray(getattr(mat["mocap"], n), float) for n in marker_names}
    mapping = DATASET_MAPPINGS["rat7m"]
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    writers = {"train": CocoWriter(), "val": CocoWriter()}
    for video in sorted(Path(video_dir).glob("*.mp4")):
        m = re.search(r"camera(\d+)-(\d+)\.mp4$", video.name)
        cam = cams[f"Camera{m.group(1)}"]
        start = int(m.group(2))
        frame_of = np.asarray(getattr(mat["cameras"], f"Camera{m.group(1)}").frame)
        split = "val" if val_starts is None or start in val_starts else "train"
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for j in range(n):
            if not cap.grab():
                break
            if j % every:
                continue
            ok, img = cap.retrieve()
            if not ok:
                break
            idx = np.nonzero(frame_of == start + j)[0]
            if len(idx) == 0:
                continue
            X = {k: v[idx[0]] for k, v in mocap.items()}
            valid = {k: x for k, x in X.items() if np.isfinite(x).all()}
            if len(valid) < 6 or "SpineF" not in valid or "SpineL" not in valid:
                continue
            vis = rat7m_visibility(valid, cam)
            names = [k for k in valid if k in mapping]
            uv = cam.project(np.stack([valid[k] for k in names]))
            kp = _canonical(names, uv, mapping)
            v = np.zeros(CANONICAL.num_keypoints, int)
            for k in names:
                v[CANONICAL.index(mapping[k])] = vis[k]
            inside = cam.in_image(kp)
            v[~inside] = 0
            all_uv = cam.project(np.stack(list(valid.values())))
            box = _box(all_uv[cam.in_image(all_uv)], 0.35, img.shape[:2])
            if box is None or (v > 0).sum() < 4:
                continue
            name = f"{video.stem}_{j:05d}.jpg"
            cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            writers[split].add(name, img.shape[:2], [{"keypoints": kp, "visibility": v, "box": box, "track_id": 0}])
        cap.release()
    (out / "cameras.json").write_text(json.dumps([c.to_dict() for c in cams.values()], indent=1))
    return {s: w.save(out / f"{s}.json") for s, w in writers.items()}


# ----------------------------------------------------------------------------- generic COCO keypoints (iRodent, AP-10K)

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def detect_mapping(names: list[str]) -> tuple[str, dict]:
    """Pick the DATASET_MAPPINGS entry whose source names best match ``names``
    (case- and punctuation-insensitive). Returns ``(mapping name, {source name: canonical})``."""
    best, best_map = "", {}
    for key, mapping in DATASET_MAPPINGS.items():
        lookup = {_norm(k): v for k, v in mapping.items()}
        hit = {n: lookup[_norm(n)] for n in names if _norm(n) in lookup}
        if len(hit) > len(best_map):
            best, best_map = key, hit
    return best, best_map


def convert_coco_keypoints(coco_json: Path, out_json: Path, mapping: str | None = None) -> dict:
    """Remap any COCO keypoint file (e.g. iRodent, AP-10K) onto the canonical vocabulary.

    Images stay where they are: ``file_name`` is kept, so load the result with the
    original image directory. The source mapping is detected from keypoint names
    unless ``mapping`` names a DATASET_MAPPINGS entry; unmapped names are reported.
    Category names (species) are kept in ``species``.
    """
    src = json.loads(Path(coco_json).read_text())
    cats = {c["id"]: c for c in src["categories"]}
    names = list(src["categories"][0]["keypoints"])
    used, table = (mapping, {n: DATASET_MAPPINGS[mapping][n] for n in names if n in DATASET_MAPPINGS[mapping]}) \
        if mapping else detect_mapping(names)
    images = {im["id"]: im for im in src["images"]}
    writer = CocoWriter()
    per_image: dict[int, list] = {}
    for a in src["annotations"]:
        if a.get("num_keypoints", 1) == 0 or a.get("iscrowd", 0):
            continue
        kps = np.asarray(a["keypoints"], float).reshape(-1, 3)
        cat_names = cats[a["category_id"]]["keypoints"]
        kp = np.full((CANONICAL.num_keypoints, 2), np.nan)
        vis = np.zeros(CANONICAL.num_keypoints, int)
        for (x, y, v), n in zip(kps, cat_names):
            c = table.get(n)
            if c is not None and v > 0:
                kp[CANONICAL.index(c)] = (x, y)
                vis[CANONICAL.index(c)] = 2 if v == 2 else 1
        x0, y0, w, h = a["bbox"]
        per_image.setdefault(a["image_id"], []).append(
            {"keypoints": kp, "visibility": vis, "box": [x0, y0, x0 + w, y0 + h],
             "species": cats[a["category_id"]].get("name", "")})
    species = {}
    for img_id, animals in per_image.items():
        im = images[img_id]
        writer.add(im["file_name"], (im["height"], im["width"]), animals)
        for a in animals:
            species[a["species"]] = species.get(a["species"], 0) + 1
    stats = writer.save(Path(out_json))
    stats.update({"mapping": used, "mapped": sorted(table), "unmapped": sorted(set(names) - set(table)),
                  "species": species})
    return stats
