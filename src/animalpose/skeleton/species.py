"""Species profiles and dataset label mappings onto the canonical vocabulary.

A *species profile* records anatomy that matters at inference time (typical
body length, which canonical points exist). A *dataset mapping* converts a
dataset's own keypoint names to canonical names; unmapped canonical points
are treated as unlabeled (zero loss weight), never as negatives.

Adding a species or dataset is a data-only change: register a profile or a
mapping here (or pass a dict at runtime). No model head is resized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animalpose.skeleton.vocabulary import CANONICAL, Skeleton


@dataclass(frozen=True)
class SpeciesProfile:
    name: str
    body_length_cm: float  # nose to tail base, adult, typical
    keypoints: tuple[str, ...]  # canonical points that exist anatomically

    def mask(self, skeleton: Skeleton = CANONICAL) -> np.ndarray:
        return skeleton.mask(self.keypoints)


_ALL = tuple(CANONICAL.names)

SPECIES: dict[str, SpeciesProfile] = {
    "mouse": SpeciesProfile("mouse", 8.5, _ALL),
    "rat": SpeciesProfile("rat", 20.0, _ALL),
    # Generic quadruped: used for unseen species; all canonical points exist.
    "quadruped": SpeciesProfile("quadruped", float("nan"), _ALL),
}

# Dataset keypoint name -> canonical name. Names absent from the canonical
# vocabulary (e.g. arbitrary implant markers) are simply not listed.
#
# Correspondences marked "approx." are anatomical judgement calls (e.g. a spine
# marker that sits between two canonical points); check them against each
# dataset's labelling guide before mixing it with precisely labelled data.
# Keypoint names were checked against the sources listed in docs/DATASETS.md.
DATASET_MAPPINGS: dict[str, dict[str, str]] = {
    # The project's own synthetic renderer labels the full vocabulary.
    "synthetic": {n: n for n in _ALL},
    # DeepLabCut SuperAnimal-TopViewMouse (27 keypoints, superanimal_topviewmouse.yaml).
    "superanimal_topviewmouse": {
        "nose": "nose", "left_eye": "left_eye", "right_eye": "right_eye",
        "left_ear": "left_ear", "right_ear": "right_ear", "head_midpoint": "head_center",
        "neck": "neck", "mid_back": "spine_upper", "mouse_center": "spine_mid",
        "mid_backend2": "spine_lower",  # approx.
        "tail_base": "tail_base", "tail3": "tail_mid", "tail_end": "tail_end",  # tail3: approx.
        "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder",
        "left_hip": "left_hip", "right_hip": "right_hip",
    },
    # DeepLabCut SuperAnimal-Quadruped (39 keypoints; "thai" is the source spelling).
    "superanimal_quadruped": {
        "nose": "nose", "left_eye": "left_eye", "right_eye": "right_eye",
        "left_earbase": "left_ear", "right_earbase": "right_ear",
        "neck_base": "neck",  # approx.
        "back_base": "spine_upper", "back_middle": "spine_mid", "back_end": "spine_lower",  # approx.
        "tail_base": "tail_base", "tail_end": "tail_end",
        "front_left_thai": "left_shoulder", "front_right_thai": "right_shoulder",  # approx.
        "front_left_knee": "left_front_knee", "front_right_knee": "right_front_knee",
        "front_left_paw": "left_front_paw", "front_right_paw": "right_front_paw",
        "back_left_thai": "left_hip", "back_right_thai": "right_hip",  # approx.
        "back_left_knee": "left_hind_knee", "back_right_knee": "right_hind_knee",
        "back_left_paw": "left_hind_paw", "back_right_paw": "right_hind_paw",
    },
    # AP-10K (17 keypoints, COCO-style names).
    "ap10k": {
        "Left Eye": "left_eye", "Right Eye": "right_eye", "Nose": "nose", "Neck": "neck",
        "Root of Tail": "tail_base",
        "Left Shoulder": "left_shoulder", "Right Shoulder": "right_shoulder",
        "Left Elbow": "left_front_knee", "Right Elbow": "right_front_knee",
        "Left Front Paw": "left_front_paw", "Right Front Paw": "right_front_paw",
        "Left Hip": "left_hip", "Right Hip": "right_hip",
        "Left Knee": "left_hind_knee", "Right Knee": "right_hind_knee",
        "Left Back Paw": "left_hind_paw", "Right Back Paw": "right_hind_paw",
    },
    # CalMS21 / MARS top-view mouse (7 keypoints; the arrays are unnamed, in this order).
    "calms21": {
        "nose": "nose", "left_ear": "left_ear", "right_ear": "right_ear", "neck": "neck",
        "left_hip": "left_hip", "right_hip": "right_hip", "tail_base": "tail_base",
    },
    # Rat 7M motion-capture markers (20; names confirmed in mocap-s1-d1.mat). Markers sit on
    # skin/implants, so all correspondences are approx.
    "rat7m": {
        "HeadB": "head_center", "SpineF": "spine_upper", "SpineM": "spine_mid", "SpineL": "spine_lower",
        "ShoulderL": "left_shoulder", "ShoulderR": "right_shoulder",
        "ElbowL": "left_front_knee", "ElbowR": "right_front_knee",
        "HipL": "left_hip", "HipR": "right_hip", "KneeL": "left_hind_knee", "KneeR": "right_hind_knee",
    },
}
# Lightning Pose mirror-mouse (side + mirrored bottom view; suffixes _top/_bot stripped).
DATASET_MAPPINGS["lp_mirror_mouse"] = {
    "paw1LH": "left_hind_paw", "paw2LF": "left_front_paw", "paw3RF": "right_front_paw", "paw4RH": "right_hind_paw",
    "tailBase": "tail_base", "tailMid": "tail_mid", "nose": "nose",
}
# Lightning Pose CRIM13 (top view; prefix black_mouse_/white_mouse_ stripped).
DATASET_MAPPINGS["lp_crim13"] = {
    "nose": "nose", "left_ear": "left_ear", "right_ear": "right_ear", "top_of_neck": "neck",
    "left_rear_knee": "left_hind_knee", "right_rear_knee": "right_hind_knee",  # approx.
    "base_of_tail": "tail_base",
}
# A six-point home-cage DeepLabCut model.
DATASET_MAPPINGS["homecage_dlc"] = {
    "nose": "nose", "leftear": "left_ear", "rightear": "right_ear", "headstage": "head_center",  # implant: approx.
    "midbody": "spine_mid", "tailbase": "tail_base",
}
CALMS21_ORDER = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]


def register_species(profile: SpeciesProfile) -> None:
    SPECIES[profile.name] = profile


def register_mapping(dataset: str, mapping: dict[str, str]) -> None:
    unknown = set(mapping.values()) - set(CANONICAL.names)
    if unknown:
        raise ValueError(f"mapping targets not in canonical vocabulary: {sorted(unknown)}")
    DATASET_MAPPINGS[dataset] = dict(mapping)


def remap_keypoints(
    names: list[str],
    keypoints: np.ndarray,
    mapping: dict[str, str] | str,
    skeleton: Skeleton = CANONICAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Map dataset-ordered keypoints to canonical order.

    Parameters
    ----------
    names: dataset keypoint names, length ``J``.
    keypoints: ``(..., J, C)`` array (``C`` is 2 or 3 coordinates, optionally
        followed by a visibility value).
    mapping: dataset name in :data:`DATASET_MAPPINGS` or an explicit dict.

    Returns
    -------
    ``(out, labeled)`` where ``out`` is ``(..., K, C)`` in canonical order
    (NaN where undefined) and ``labeled`` is a boolean ``(K,)`` mask of
    canonical points this dataset defines.
    """
    if isinstance(mapping, str):
        mapping = DATASET_MAPPINGS[mapping]
    k = skeleton.num_keypoints
    out = np.full(keypoints.shape[:-2] + (k, keypoints.shape[-1]), np.nan, dtype=np.float32)
    labeled = np.zeros(k, dtype=bool)
    for j, name in enumerate(names):
        target = mapping.get(name)
        if target is None:
            continue
        i = skeleton.index(target)
        out[..., i, :] = keypoints[..., j, :]
        labeled[i] = True
    return out, labeled
