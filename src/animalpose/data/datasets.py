"""Training datasets for the detector and the pose network.

* :class:`SyntheticScenes` — a pool of simulations rendered on the fly from
  random cameras; shared by both stages.
* :class:`CocoKeypoints` — COCO-style keypoint JSON (the format used by
  AP-10K, SuperAnimal exports and :mod:`animalpose.data.dlc`), remapped onto
  the canonical vocabulary through a dataset mapping.

Both feed :class:`DetectionDataset` (letterboxed frames + boxes) and
:class:`PoseCropDataset` (augmented top-down crops + keypoints + per-keypoint
state 2/1/0 = visible / labeled-but-occluded / unlabeled).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from animalpose.data.degrade import DEFAULT_PROFILE, Profile, degrade, zoom_out
from animalpose.data.synthetic import random_camera, render, simulate
from animalpose.geometry import apply_affine, box_xyxy_to_cs, crop_affine, warp
from animalpose.skeleton import CANONICAL, remap_keypoints


@dataclass
class Annotation:
    """One image with its animals, in canonical keypoint order."""

    image: np.ndarray  # BGR uint8
    boxes: np.ndarray  # (N, 4)
    keypoints: np.ndarray  # (N, K, 2)
    state: np.ndarray  # (N, K) int: 2 visible, 1 occluded-labeled, 0 unlabeled
    ids: np.ndarray  # (N,) identity for the embedding loss, -1 unknown
    species: np.ndarray  # (N,) index into species list


class SyntheticScenes:
    """Random access to rendered frames from a fixed pool of simulations."""

    def __init__(self, n_sims: int = 64, frames: int = 120, size=(320, 256), species=("mouse", "rat"),
                 max_animals: int = 4, seed: int = 0, low_view_fraction: float = 0.0):
        rng = np.random.default_rng(seed)
        self.size = size
        self.species = list(species)
        self.sims = [simulate(int(rng.integers(1, max_animals + 1)), frames, str(rng.choice(self.species)), seed=rng)
                     for _ in range(n_sims)]
        self.seed = seed
        self.low_view_fraction = low_view_fraction  # extra share of 5-25 degree (side-like) cameras

    def sample(self, index: int, variant: int = 0) -> Annotation:
        """``variant`` re-renders the same simulation at another time and
        viewpoint (positive pairs for the identity embedding)."""
        s = int(np.random.default_rng((self.seed, index)).integers(len(self.sims)))
        rng = np.random.default_rng((self.seed, index, variant))
        sim = self.sims[s]
        t = int(rng.integers(20, sim.n_frames))
        low = rng.random() < self.low_view_fraction
        cam = random_camera(sim.arena_cm, rng, self.size, elevation=(5.0, 25.0) if low else (10.0, 90.0))
        f = render(sim, cam, t, rng)
        keep = f.present
        n = int(keep.sum())
        state = np.where(f.visible[keep], 2, 1)
        inside = cam.in_image(f.keypoints[keep])
        state = np.where(inside, state, 0)  # out-of-frame points carry no label
        ids = s * 16 + np.nonzero(keep)[0]
        species = np.full(n, self.species.index(sim.species))
        return Annotation(f.image, f.boxes[keep], f.keypoints[keep], state, ids, species)


class CocoKeypoints:
    """COCO keypoint JSON -> canonical annotations.

    ``mapping`` is a dataset name registered in
    :data:`animalpose.skeleton.DATASET_MAPPINGS` or an explicit dict. COCO
    visibility ``v=2`` -> visible, ``v=1`` -> occluded-labeled, ``v=0`` -> unlabeled.
    """

    def __init__(self, json_path: str | Path, image_root: str | Path, mapping, species: str = "mouse",
                 species_list=("mouse", "rat")):
        data = json.loads(Path(json_path).read_text())
        self.root = Path(image_root)
        cats = {c["id"]: c for c in data["categories"]}
        self.images = {im["id"]: im for im in data["images"]}
        self.by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            if ann.get("num_keypoints", 1) > 0:
                self.by_image.setdefault(ann["image_id"], []).append(ann)
        self.ids = sorted(self.by_image)
        self.cats = cats
        self.mapping = mapping
        self.species_idx = list(species_list).index(species) if species in species_list else -1

    def __len__(self) -> int:
        return len(self.ids)

    def sample(self, index: int, variant: int = 0) -> Annotation:
        im = self.images[self.ids[index % len(self.ids)]]
        image = cv2.imread(str(self.root / im["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.root / im["file_name"])
        anns = self.by_image[im["id"]]
        boxes, kps, states, ids = [], [], [], []
        for a in anns:
            names = self.cats[a["category_id"]]["keypoints"]
            raw = np.asarray(a["keypoints"], dtype=np.float32).reshape(-1, 3)
            canon, labeled = remap_keypoints(names, raw, self.mapping)
            v = np.nan_to_num(canon[:, 2])
            st = np.where(labeled & (v >= 2), 2, np.where(labeled & (v >= 1), 1, 0))
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            kps.append(np.nan_to_num(canon[:, :2]))
            states.append(st)
            ids.append(a.get("track_id", -1))
        n = len(anns)
        return Annotation(image, np.asarray(boxes, np.float32), np.asarray(kps, np.float32),
                          np.asarray(states, np.int64), np.asarray(ids, np.int64), np.full(n, self.species_idx))


class MixedSource:
    """Weighted mixture of annotation sources (synthetic and/or real datasets)."""

    def __init__(self, sources: list, weights: list[float], seed: int = 0):
        self.sources = sources
        self.p = np.asarray(weights, float) / np.sum(weights)
        self.seed = seed

    def sample(self, index: int, variant: int = 0) -> Annotation:
        k = int(np.random.default_rng((self.seed, 99, index)).choice(len(self.sources), p=self.p))
        return self.sources[k].sample(int(np.random.default_rng((self.seed, 98, index)).integers(1 << 30)), variant)


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, np.ndarray]:
    """Resize keeping aspect ratio and pad to ``size=(H, W)``. Returns image, scale, pad (x, y)."""
    h, w = image.shape[:2]
    H, W = size
    s = min(H / h, W / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    out = np.zeros((H, W, 3), dtype=image.dtype)
    py, px = (H - nh) // 2, (W - nw) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, s, np.array([px, py], dtype=np.float32)


def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> RGB float CHW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(image_bgr[..., ::-1].transpose(2, 0, 1))).float() / 255.0


def photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = image.astype(np.float32)
    if rng.random() < 0.2:
        g = img.mean(-1, keepdims=True)
        img = np.repeat(g, 3, -1)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    img = img * rng.uniform(0.9, 1.1, size=3)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        img += rng.normal(0, rng.uniform(2, 8), size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class DetectionDataset(Dataset):
    """Letterboxed frames with boxes. ``degrade`` (probability) applies blur/low-quality
    degradations (:mod:`animalpose.data.degrade`); ``zoom_out`` (probability) shrinks the
    frame so animals become as small as in wide arena views. ``profile`` selects the camera
    behaviour of those degradations (a :class:`animalpose.data.degrade.Profile`)."""

    def __init__(self, source, length: int, size=(256, 320), augment: bool = True, degrade: float = 0.0,
                 zoom_out: float = 0.0, profile: Profile | None = None):
        self.source, self.length, self.size, self.augment = source, length, size, augment
        self.degrade, self.zoom_out, self.profile = degrade, zoom_out, profile or DEFAULT_PROFILE

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i):
        ann = self.source.sample(i)
        rng = np.random.default_rng((7, i))
        img, boxes = ann.image, ann.boxes.copy()
        if self.augment and rng.random() < 0.5:
            img = img[:, ::-1]
            w = img.shape[1]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        if self.augment and len(boxes) and rng.random() < self.zoom_out:
            img, boxes = zoom_out(np.ascontiguousarray(img), boxes, rng)
        if self.augment:
            img = photometric(img, rng)
        flag = False
        if self.augment and len(boxes) and rng.random() < self.degrade:
            size = float(np.median(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])))
            img = degrade(img, rng, size, profile=self.profile)
            flag = True
        img, s, pad = letterbox(np.ascontiguousarray(img), self.size)
        boxes = boxes * s + np.tile(pad, 2)
        return to_tensor(img), torch.from_numpy(boxes.astype(np.float32)), torch.tensor(ann.species), torch.tensor(flag)


def detection_collate(batch):
    imgs, boxes, species, degraded = zip(*batch)
    return torch.stack(imgs), list(boxes), list(species), torch.stack(degraded)



class PoseCropDataset(Dataset):
    """Top-down crops with rotation/scale/shift/flip augmentation.

    The crop box is jittered so the network sees detector-like framing.
    """

    # Degrading happens on a native-scale patch of the frame of at most this many pixels.
    MAX_BLUR_BUFFER = 256

    def __init__(self, source, length: int, crop_size: int = 128, augment: bool = True,
                 rotation: float = np.pi, scale=(0.8, 1.25), shift: float = 0.1, degrade: float = 0.0,
                 profile: Profile | None = None):
        self.source, self.length, self.crop, self.augment = source, length, crop_size, augment
        self.degrade = degrade  # probability of blur/low-quality degradation (animalpose.data.degrade)
        self.profile = profile or DEFAULT_PROFILE
        self.rotation, self.scale, self.shift = rotation, scale, shift
        self.flip = CANONICAL.flip_index

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i):
        # Items 4n..4n+3 come from one scene at different times/viewpoints:
        # two crops of animal A, then two of animal B (when the scene has one),
        # so every batch has same-scene positives *and* negatives for the
        # identity embedding, which is what re-identification needs.
        rng = np.random.default_rng((11, i))
        base, variant = divmod(i, 4)
        ann = self.source.sample(base, variant)
        while len(ann.boxes) == 0:
            base += 1_000_003
            ann = self.source.sample(base, variant)
        scene_ids = np.unique(self.source.sample(base, 0).ids) if variant else np.unique(ann.ids)
        order = np.random.default_rng((13, base)).permutation(scene_ids) if len(scene_ids) else scene_ids
        target = order[min(variant // 2, len(order) - 1)] if len(order) else -1
        same = np.nonzero(ann.ids == target)[0]
        j = int(same[0]) if len(same) else int(rng.integers(len(ann.boxes)))
        center, side = box_xyxy_to_cs(ann.boxes[j])
        rot = 0.0
        kpts, state = ann.keypoints[j].copy(), ann.state[j].copy()
        img = ann.image
        if self.augment:
            center = center + rng.uniform(-self.shift, self.shift, 2) * side
            side *= rng.uniform(*self.scale)
            rot = rng.uniform(-self.rotation, self.rotation) if rng.random() < 0.8 else 0.0
        m = crop_affine(center, side, self.crop, rot)
        kp = apply_affine(m, kpts).astype(np.float32)
        flag = False
        if self.augment and rng.random() < self.degrade:
            crop, flag = self.degraded_crop(img, center, side, rot, rng), True
        else:
            crop = warp(img, m, self.crop)
        if self.augment and rng.random() < 0.5:
            crop = crop[:, ::-1]
            kp[:, 0] = self.crop - 1 - kp[:, 0]
            kp, state = kp[self.flip], state[self.flip]
        if self.augment:
            crop = photometric(np.ascontiguousarray(crop), rng)
        inside = (kp >= 0).all(-1) & (kp <= self.crop - 1).all(-1)
        state = np.where(inside, state, 0)
        return (to_tensor(np.ascontiguousarray(crop)), torch.from_numpy(kp), torch.from_numpy(state.astype(np.int64)),
                torch.tensor(int(ann.ids[j])), torch.tensor(flag))

    def degraded_crop(self, img: np.ndarray, center: np.ndarray, side: float, rot: float,
                      rng: np.random.Generator) -> np.ndarray:
        """The crop after camera degradations, applied at the camera's own pixel scale.

        Defocus, sensor noise and compression belong to the recording, not to the crop box: a
        40 px mouse blurred by a 2 px PSF is equally blurry however tightly it is cropped, which
        is what a fixed lens does and what ``animalpose eval blur`` measures. A native-scale patch
        slightly larger than the crop box is degraded first and then resampled into the crop;
        animals larger than that are shrunk into the patch, the PSF scaling along with them.
        """
        side_buf = side * 1.12
        scale = min(1.0, self.MAX_BLUR_BUFFER / max(side_buf, 1.0))
        buf = max(8, int(round(side_buf * scale)))
        src = warp(img, crop_affine(center, side_buf, buf, 0.0), buf)
        psf = None if self.profile.psf_px is None else self.profile.psf_px * scale
        src = degrade(src, rng, side * scale, profile=self.profile, psf_px=psf)
        return warp(src, crop_affine(np.array([buf / 2.0, buf / 2.0]), side * scale, self.crop, rot), self.crop)
