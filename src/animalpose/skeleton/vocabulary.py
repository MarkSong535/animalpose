"""Canonical, species-agnostic keypoint vocabulary.

Every model in the package predicts this fixed set of anatomical landmarks.
A species (or a dataset) only declares *which* canonical points it defines and
how its own labels map onto them; no network head changes. Missing points are
masked out of the loss, so datasets with different keypoint sets can be pooled
for training, and a new quadruped can be run with the same weights by
selecting its subset.

Names follow quadruped anatomy rather than rodent-specific marker names so the
same vocabulary covers rodents, carnivores, ungulates and primates on all fours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    name: str
    region: str  # head | trunk | tail | forelimb | hindlimb
    side: str  # left | right | mid
    description: str


_KEYPOINTS: tuple[Keypoint, ...] = (
    Keypoint("nose", "head", "mid", "tip of the snout / rhinarium"),
    Keypoint("left_eye", "head", "left", "centre of the left eye"),
    Keypoint("right_eye", "head", "right", "centre of the right eye"),
    Keypoint("left_ear", "head", "left", "base of the left pinna"),
    Keypoint("right_ear", "head", "right", "base of the right pinna"),
    Keypoint("head_center", "head", "mid", "midpoint between the ears (crown)"),
    Keypoint("neck", "trunk", "mid", "dorsal neck, behind the skull"),
    Keypoint("spine_upper", "trunk", "mid", "withers / between the shoulder blades"),
    Keypoint("spine_mid", "trunk", "mid", "mid back (thoracolumbar junction)"),
    Keypoint("spine_lower", "trunk", "mid", "lower back / rump, between the hips"),
    Keypoint("tail_base", "tail", "mid", "dorsal base of the tail"),
    Keypoint("tail_mid", "tail", "mid", "midpoint along the tail"),
    Keypoint("tail_end", "tail", "mid", "tail tip"),
    Keypoint("left_shoulder", "forelimb", "left", "left shoulder joint"),
    Keypoint("right_shoulder", "forelimb", "right", "right shoulder joint"),
    Keypoint("left_front_knee", "forelimb", "left", "left elbow / carpal region"),
    Keypoint("right_front_knee", "forelimb", "right", "right elbow / carpal region"),
    Keypoint("left_front_paw", "forelimb", "left", "left front paw / hand"),
    Keypoint("right_front_paw", "forelimb", "right", "right front paw / hand"),
    Keypoint("left_hip", "hindlimb", "left", "left hip joint"),
    Keypoint("right_hip", "hindlimb", "right", "right hip joint"),
    Keypoint("left_hind_knee", "hindlimb", "left", "left knee / stifle"),
    Keypoint("right_hind_knee", "hindlimb", "right", "right knee / stifle"),
    Keypoint("left_hind_paw", "hindlimb", "left", "left hind paw / foot"),
    Keypoint("right_hind_paw", "hindlimb", "right", "right hind paw / foot"),
)

_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "head_center"), ("right_ear", "head_center"),
    ("head_center", "neck"), ("neck", "spine_upper"),
    ("spine_upper", "spine_mid"), ("spine_mid", "spine_lower"),
    ("spine_lower", "tail_base"), ("tail_base", "tail_mid"), ("tail_mid", "tail_end"),
    ("spine_upper", "left_shoulder"), ("spine_upper", "right_shoulder"),
    ("left_shoulder", "left_front_knee"), ("left_front_knee", "left_front_paw"),
    ("right_shoulder", "right_front_knee"), ("right_front_knee", "right_front_paw"),
    ("spine_lower", "left_hip"), ("spine_lower", "right_hip"),
    ("left_hip", "left_hind_knee"), ("left_hind_knee", "left_hind_paw"),
    ("right_hip", "right_hind_knee"), ("right_hind_knee", "right_hind_paw"),
)


@dataclass(frozen=True)
class Skeleton:
    keypoints: tuple[Keypoint, ...]
    edges: tuple[tuple[str, str], ...]
    _index: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {k.name: i for i, k in enumerate(self.keypoints)})

    @property
    def names(self) -> list[str]:
        return [k.name for k in self.keypoints]

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def index(self, name: str) -> int:
        return self._index[name]

    def indices(self, names) -> list[int]:
        return [self._index[n] for n in names]

    @property
    def edge_index(self) -> np.ndarray:
        return np.array([(self._index[a], self._index[b]) for a, b in self.edges], dtype=np.int64)

    @property
    def flip_index(self) -> np.ndarray:
        """Permutation that swaps left/right keypoints (for horizontal-flip augmentation)."""
        perm = np.arange(self.num_keypoints)
        for i, kp in enumerate(self.keypoints):
            if kp.side == "left":
                perm[i] = self._index[kp.name.replace("left_", "right_", 1)]
            elif kp.side == "right":
                perm[i] = self._index[kp.name.replace("right_", "left_", 1)]
        return perm

    def mask(self, names) -> np.ndarray:
        """Boolean mask over the vocabulary that is True for ``names``."""
        m = np.zeros(self.num_keypoints, dtype=bool)
        m[self.indices(names)] = True
        return m

    # Groups used by behavior features to define an egocentric frame.
    @property
    def anterior(self) -> list[int]:
        return self.indices(["nose", "head_center", "neck", "spine_upper"])

    @property
    def posterior(self) -> list[int]:
        return self.indices(["tail_base", "spine_lower", "left_hip", "right_hip"])


CANONICAL = Skeleton(_KEYPOINTS, _EDGES)
