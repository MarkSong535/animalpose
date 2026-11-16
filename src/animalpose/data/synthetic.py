"""Procedural 3D rodent simulator and renderer.

This is a *test and bootstrapping* data source, not a substitute for real
labels. It produces:

* 3D trajectories of several animals over the canonical skeleton, driven by a
  behavior Markov chain (resting, walking, rearing, grooming, eating) with
  behavior-specific posture and locomotion;
* renders from arbitrary pinhole cameras (top, side, angled or random), with
  depth-ordered occlusion between animals, per-keypoint visibility, visible-
  extent bounding boxes and per-animal appearance (coat colour, size).

Because the 3D ground truth is known, the same simulation exercises detection,
pose, tracking through occlusion, temporal gap filling, behavior and
multi-camera triangulation end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from animalpose.labels import BEHAVIORS
from animalpose.multiview.camera import Camera
from animalpose.skeleton import CANONICAL, SPECIES

K = CANONICAL.num_keypoints
_I = CANONICAL.index

# Body-frame template in body lengths: x forward, y left, z up; origin under spine_mid.
_TEMPLATE = {
    "nose": (0.52, 0.0, 0.10),
    "left_eye": (0.43, 0.045, 0.14), "right_eye": (0.43, -0.045, 0.14),
    "left_ear": (0.35, 0.08, 0.17), "right_ear": (0.35, -0.08, 0.17),
    "head_center": (0.36, 0.0, 0.17),
    "neck": (0.25, 0.0, 0.18),
    "spine_upper": (0.15, 0.0, 0.20),
    "spine_mid": (0.0, 0.0, 0.21),
    "spine_lower": (-0.17, 0.0, 0.19),
    "tail_base": (-0.30, 0.0, 0.12),
    "tail_mid": (-0.62, 0.0, 0.03),
    "tail_end": (-0.95, 0.0, 0.01),
    "left_shoulder": (0.16, 0.08, 0.11), "right_shoulder": (0.16, -0.08, 0.11),
    "left_front_knee": (0.17, 0.09, 0.05), "right_front_knee": (0.17, -0.09, 0.05),
    "left_front_paw": (0.19, 0.09, 0.0), "right_front_paw": (0.19, -0.09, 0.0),
    "left_hip": (-0.18, 0.09, 0.11), "right_hip": (-0.18, -0.09, 0.11),
    "left_hind_knee": (-0.14, 0.11, 0.05), "right_hind_knee": (-0.14, -0.11, 0.05),
    "left_hind_paw": (-0.20, 0.10, 0.0), "right_hind_paw": (-0.20, -0.10, 0.0),
}
TEMPLATE = np.array([_TEMPLATE[n] for n in CANONICAL.names], dtype=np.float64)

_HEAD = CANONICAL.indices(["nose", "left_eye", "right_eye", "left_ear", "right_ear", "head_center"])
_FRONT_PAWS = CANONICAL.indices(["left_front_paw", "right_front_paw"])
_FRONT_KNEES = CANONICAL.indices(["left_front_knee", "right_front_knee"])
_HIND_PAWS = CANONICAL.indices(["left_hind_paw", "right_hind_paw"])
_HIND_KNEES = CANONICAL.indices(["left_hind_knee", "right_hind_knee"])
_TAIL = CANONICAL.indices(["tail_mid", "tail_end"])
_PITCHED = [i for i, n in enumerate(CANONICAL.names)
            if not n.startswith("tail") and "hind" not in n and "hip" not in n and n != "spine_lower"]

# Posture targets per behavior: crouch, head_down, gait, paws_to_face, pitch(rad), groom, speed(L/s)
_POSTURE = {
    "resting": (1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0),
    "walking": (0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.3),
    "rearing": (0.0, 0.0, 0.0, 0.35, 1.1, 0.0, 0.0),
    "grooming": (0.3, 0.2, 0.0, 1.0, 0.35, 1.0, 0.0),
    "eating": (0.2, 1.0, 0.0, 0.6, 0.0, 0.0, 0.0),
}
_DURATION = {"resting": (60, 240), "walking": (30, 150), "rearing": (20, 60), "grooming": (40, 150), "eating": (40, 150)}
_NEXT = {"resting": 1.0, "walking": 3.0, "rearing": 1.0, "grooming": 1.0, "eating": 1.0}

COAT_COLORS = {  # BGR
    "black": (40, 38, 36), "white": (225, 228, 230), "brown": (50, 85, 130),
    "agouti": (70, 100, 120), "grey": (125, 125, 125), "cream": (150, 190, 215),
}


@dataclass
class Appearance:
    coat: tuple[int, int, int]
    size: float  # body length in cm
    coat_name: str = ""


@dataclass
class Simulation:
    species: str
    fps: float
    arena_cm: float
    kpts3d: np.ndarray  # (N, T, K, 3) cm, world frame, z up
    behavior: np.ndarray  # (N, T) int index into BEHAVIORS
    appearance: list[Appearance]
    food: np.ndarray  # (F, 2) food pellet locations on the floor

    @property
    def n_animals(self) -> int:
        return self.kpts3d.shape[0]

    @property
    def n_frames(self) -> int:
        return self.kpts3d.shape[1]


def _rot_xz(p: np.ndarray, pivot: np.ndarray, angle: float) -> np.ndarray:
    d = p - pivot
    c, s = np.cos(angle), np.sin(angle)
    out = p.copy()
    out[..., 0] = pivot[0] + d[..., 0] * c - d[..., 2] * s
    out[..., 2] = pivot[2] + d[..., 0] * s + d[..., 2] * c
    return out


def body_pose(params: np.ndarray, gait_phase: float, groom_phase: float, bend: float, tail_phase: float) -> np.ndarray:
    """Posture parameters -> body-frame keypoints ``(K, 3)`` in body lengths."""
    crouch, head_down, gait, paws_to_face, pitch, groom = params
    p = TEMPLATE.copy()
    p[:, 0] *= 1 - 0.15 * crouch
    p[:, 2] *= 1 - 0.4 * crouch
    # Head pitches down about the neck (sniffing / eating).
    neck = p[_I("neck")].copy()
    p[_HEAD] = _rot_xz(p[_HEAD], neck, -0.7 * head_down)
    # Diagonal trot: LF with RH, RF with LH.
    for paws, knees, sign in ((_FRONT_PAWS, _FRONT_KNEES, 1.0), (_HIND_PAWS, _HIND_KNEES, -1.0)):
        for j, (pi, ki) in enumerate(zip(paws, knees)):
            ph = gait_phase + np.pi * j + (0 if sign > 0 else np.pi)
            p[pi, 0] += gait * 0.09 * np.sin(ph)
            p[pi, 2] += gait * 0.04 * max(0.0, np.cos(ph))
            p[ki, 0] += gait * 0.05 * np.sin(ph)
    # Front paws toward the face (grooming / holding food / rearing posture).
    nose = p[_I("nose")]
    for j, (pi, ki) in enumerate(zip(_FRONT_PAWS, _FRONT_KNEES)):
        side = 1 if j == 0 else -1
        target = nose + np.array([-0.07, 0.035 * side, -0.02 + 0.04 * groom * np.sin(groom_phase + side)])
        p[pi] = (1 - paws_to_face) * p[pi] + paws_to_face * target
        shoulder = p[_I("left_shoulder" if j == 0 else "right_shoulder")]
        p[ki] = (1 - 0.5 * paws_to_face) * p[ki] + 0.5 * paws_to_face * (target + shoulder) / 2
    # Rear up about the hips.
    pivot = p[_I("left_hip")].copy()
    pivot[1] = 0.0
    p[_PITCHED] = _rot_xz(p[_PITCHED], pivot, pitch)
    # Tail sway and lateral spine bend from turning.
    p[_TAIL, 1] += 0.12 * np.sin(tail_phase) * np.array([0.6, 1.0]) + 0.35 * crouch * np.array([0.5, 1.0])
    ang = bend * p[:, 0]
    c, s = np.cos(ang), np.sin(ang)
    x, y = p[:, 0].copy(), p[:, 1].copy()
    p[:, 0], p[:, 1] = c * x - s * y, s * x + c * y
    p[:, 2] = np.maximum(p[:, 2], 0.0)
    return p


def simulate(
    n_animals: int = 2,
    n_frames: int = 300,
    species: str = "mouse",
    fps: float = 30.0,
    arena_cm: float | None = None,
    seed: int | np.random.Generator = 0,
    coats: list[str] | None = None,
) -> Simulation:
    rng = np.random.default_rng(seed)
    length = SPECIES[species].body_length_cm
    arena = arena_cm or 5.0 * length
    names = list(COAT_COLORS)
    if coats is None:
        coats = list(rng.choice(names, size=n_animals, replace=n_animals > len(names)))
    appearance = [Appearance(COAT_COLORS[c], length * rng.uniform(0.9, 1.1), c) for c in coats]
    food = rng.uniform(-0.4 * arena, 0.4 * arena, size=(2, 2))
    kpts = np.zeros((n_animals, n_frames, K, 3))
    labels = np.zeros((n_animals, n_frames), dtype=np.int64)
    dt = 1.0 / fps
    for a in range(n_animals):
        L = appearance[a].size
        pos = rng.uniform(-0.35 * arena, 0.35 * arena, size=2)
        heading = rng.uniform(-np.pi, np.pi)
        omega = 0.0
        speed = 0.0
        params = np.zeros(6)
        gait_phase, groom_phase, tail_phase = rng.uniform(0, 2 * np.pi, 3)
        beh = str(rng.choice(BEHAVIORS))
        remaining = int(rng.integers(*_DURATION[beh]))
        for t in range(n_frames):
            if remaining <= 0:
                options = [b for b in BEHAVIORS if b != beh]
                w = np.array([_NEXT[b] for b in options])
                beh = str(rng.choice(options, p=w / w.sum()))
                remaining = int(rng.integers(*_DURATION[beh]))
            remaining -= 1
            target = np.array(_POSTURE[beh])
            params += (target[:6] - params) * 0.15
            target_speed = target[6] * L * rng.uniform(0.8, 1.2) if beh == "walking" else 0.0
            speed += (target_speed - speed) * 0.1
            # Heading: OU turning rate, steer away from walls.
            omega += -0.5 * omega * dt * 10 + rng.normal(0, 2.5) * np.sqrt(dt)
            ahead = pos + np.array([np.cos(heading), np.sin(heading)]) * L
            if np.any(np.abs(ahead) > 0.42 * arena):
                desired = np.arctan2(-pos[1], -pos[0])
                omega += 4.0 * np.angle(np.exp(1j * (desired - heading))) * dt * 10
            omega = np.clip(omega, -4.0, 4.0)
            heading += omega * dt * min(1.0, 0.2 + speed / L)
            pos += np.array([np.cos(heading), np.sin(heading)]) * speed * dt
            pos = np.clip(pos, -0.45 * arena, 0.45 * arena)
            gait_phase += 2 * np.pi * 4.0 * dt * (speed / L)
            groom_phase += 2 * np.pi * 5.0 * dt
            tail_phase += 2 * np.pi * 0.7 * dt
            bend = np.clip(omega * 0.15, -0.6, 0.6)
            body = body_pose(params, gait_phase, groom_phase, bend, tail_phase) * L
            if beh in ("eating", "grooming"):
                body[_HEAD] += rng.normal(0, 0.004 * L, size=(len(_HEAD), 3))
                body[:, 2] = np.maximum(body[:, 2], 0.0)
            c, s = np.cos(heading), np.sin(heading)
            world = body.copy()
            world[:, 0] = c * body[:, 0] - s * body[:, 1] + pos[0]
            world[:, 1] = s * body[:, 0] + c * body[:, 1] + pos[1]
            kpts[a, t] = world
            labels[a, t] = BEHAVIORS.index(beh)
    return Simulation(species, fps, arena, kpts, labels, appearance, food)


# ----------------------------------------------------------------------------- cameras

def _fit_focal(cam: Camera, pts: np.ndarray, fill: float = 0.9) -> Camera:
    """Rescale focal length so ``pts`` fill ``fill`` of the image."""
    cam.K[0, 0] = cam.K[1, 1] = 1.0
    uv = cam.project(pts) - cam.K[:2, 2]
    w, h = cam.size
    scale = fill * min(w / (2 * np.abs(uv[:, 0]).max()), h / (2 * np.abs(uv[:, 1]).max()))
    cam.K[0, 0] = cam.K[1, 1] = scale
    return cam


def camera_from_view(arena_cm: float, elevation_deg: float, azimuth_deg: float = -90.0,
                     size=(320, 256), distance: float = 1.6, name: str = "cam", fill: float = 0.9) -> Camera:
    """Camera on a sphere around the arena centre. Elevation 90 = top-down, ~10 = side."""
    el, az = np.deg2rad(elevation_deg), np.deg2rad(azimuth_deg)
    r = distance * arena_cm
    eye = r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    target = np.array([0.0, 0.0, 0.0 if elevation_deg > 30 else 0.03 * arena_cm])
    cam = Camera.look_at(eye, target, size=size, name=name)
    half = arena_cm / 2
    corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (0.0, 0.1 * arena_cm)])
    return _fit_focal(cam, corners, fill)


VIEW_PRESETS = {"top": (90.0, -90.0), "side": (8.0, -90.0), "angled": (45.0, -135.0)}


def make_cameras(arena_cm: float, views=("top", "side", "angled"), size=(320, 256)) -> list[Camera]:
    return [camera_from_view(arena_cm, *VIEW_PRESETS[v], size=size, name=v) for v in views]


def random_camera(arena_cm: float, rng: np.random.Generator, size=(320, 256), elevation=(10.0, 90.0)) -> Camera:
    el = float(rng.uniform(*elevation))
    az = float(rng.uniform(-180.0, 180.0))
    return camera_from_view(arena_cm, el, az, size=size, distance=float(rng.uniform(1.3, 2.2)),
                            name=f"rand_el{el:.0f}_az{az:.0f}", fill=float(rng.uniform(0.75, 1.1)))


# ----------------------------------------------------------------------------- rendering

@dataclass
class Frame:
    image: np.ndarray  # (H, W, 3) uint8 BGR
    keypoints: np.ndarray  # (N, K, 2) pixels (projected even if hidden)
    visible: np.ndarray  # (N, K) bool
    boxes: np.ndarray  # (N, 4) xyxy of the visible extent
    present: np.ndarray  # (N,) bool: animal sufficiently visible to annotate
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _lighter(c, f):
    return tuple(int(np.clip(v * (1 - f) + 230 * f, 0, 255)) for v in c)


def _draw_animal(canvas: np.ndarray, mask: np.ndarray, uv: np.ndarray, px_len: float, coat,
                 depth: np.ndarray | None = None, size: float = 1.0) -> None:
    """Draw one animal. With per-keypoint camera ``depth`` the limbs are depth-ordered:
    the far side's legs go behind the trunk and are shaded darker, the near side's in
    front, which is the occlusion-order cue that tells near paws from far ones in side
    views (top views, where both sides are equally far, are unaffected)."""
    def pt(i):
        return tuple(int(round(v)) for v in uv[i] * 4)  # 2 fractional bits for cv2 shift

    width = max(2, int(0.26 * px_len))
    skin = (150, 160, 215)
    limb = max(1, int(0.06 * px_len))
    tail = max(1, int(0.05 * px_len))

    def line(a, b, color, th):
        cv2.line(canvas, pt(a), pt(b), color, th, cv2.LINE_AA, shift=2)
        cv2.line(mask, pt(a), pt(b), 255, th, cv2.LINE_8, shift=2)

    def circle(a, r, color):
        r = max(1, int(r))
        cv2.circle(canvas, pt(a), r * 4, color, -1, cv2.LINE_AA, shift=2)
        cv2.circle(mask, pt(a), r * 4, 255, -1, cv2.LINE_8, shift=2)

    n = CANONICAL.index
    far_side, shade = "left", 0.0
    if depth is not None:
        lat = [depth[CANONICAL.indices([f"{s}_shoulder", f"{s}_hip"])].mean() for s in ("left", "right")]
        far_side = "left" if lat[0] > lat[1] else "right"
        shade = 0.4 * float(np.clip(abs(lat[0] - lat[1]) / (0.18 * size), 0.0, 1.0))
    near_side = "right" if far_side == "left" else "left"

    def legs(side, hind, front, factor):
        c, sk = tuple(int(v * factor) for v in coat), tuple(int(v * factor) for v in skin)
        for limb_name, on in (("hind", hind), ("front", front)):
            if on:
                top = f"{side}_hip" if limb_name == "hind" else f"{side}_shoulder"
                line(n(top), n(f"{side}_{limb_name}_knee"), c, limb)
                line(n(f"{side}_{limb_name}_knee"), n(f"{side}_{limb_name}_paw"), c, limb)
                circle(n(f"{side}_{limb_name}_paw"), 0.035 * px_len, sk)

    line(n("tail_base"), n("tail_mid"), skin, tail)
    line(n("tail_mid"), n("tail_end"), skin, max(1, tail - 1))
    legs(far_side, True, depth is not None, 1.0 - shade)
    legs(near_side, depth is None, False, 1.0)
    spine = ["tail_base", "spine_lower", "spine_mid", "spine_upper", "neck"]
    for a, b in zip(spine[:-1], spine[1:]):
        line(n(a), n(b), coat, width)
    for s in spine[1:]:
        circle(n(s), width / 2, coat)
    trunk = CANONICAL.indices(["left_hip", "right_hip", "left_shoulder", "right_shoulder"])
    hull = cv2.convexHull(np.round(uv[trunk]).astype(np.int32))
    cv2.fillConvexPoly(canvas, hull, coat, cv2.LINE_AA)
    cv2.fillConvexPoly(mask, hull, 255)
    if depth is None:
        legs("left", False, True, 1.0)
        legs("right", False, True, 1.0)
    else:
        legs(near_side, True, True, 1.0)
    line(n("neck"), n("head_center"), coat, int(width * 0.9))
    line(n("head_center"), n("nose"), coat, int(width * 0.6))
    circle(n("head_center"), width * 0.45, coat)
    circle(n("nose"), 0.04 * px_len, skin)
    for side in ("left", "right"):
        circle(n(f"{side}_ear"), 0.075 * px_len, _lighter(skin, 0.1))
        circle(n(f"{side}_eye"), 0.028 * px_len, (10, 10, 10))


def _background(cam: Camera, sim: Simulation, rng: np.random.Generator) -> np.ndarray:
    w, h = cam.size
    wall = np.array(rng.uniform(40, 110, size=3))
    img = np.ones((h, w, 3), dtype=np.float32) * wall
    half = sim.arena_cm / 2
    floor_poly = cam.project(np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]]))
    floor = np.array(rng.uniform(120, 200, size=3)) * rng.uniform(0.85, 1.15, size=3)
    fmask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(fmask, np.round(floor_poly).astype(np.int32), 255)
    img[fmask > 0] = floor
    low = cv2.resize(rng.normal(0, 12, size=(max(2, h // 32), max(2, w // 32), 1)).astype(np.float32), (w, h))
    img += low[..., None] if low.ndim == 2 else low
    speckle = rng.random((h, w)) < 0.02
    img[speckle] *= rng.uniform(0.7, 1.2)
    for f in sim.food:  # food pellets
        uv = cam.project(np.array([f[0], f[1], 0.0]))
        r = max(1, int(0.06 * sim.appearance[0].size * cam.K[0, 0] / max(np.linalg.norm(cam.center), 1e-6)))
        cv2.circle(img, tuple(int(v) for v in uv), r, (30, 60, 100), -1, cv2.LINE_AA)
    return img


def render(sim: Simulation, cam: Camera, t: int, rng: np.random.Generator | int = 0,
           background: np.ndarray | None = None, noise: bool = True) -> Frame:
    rng = np.random.default_rng(rng)
    w, h = cam.size
    img = (_background(cam, sim, rng) if background is None else background.astype(np.float32)).copy()
    n = sim.n_animals
    uvs, depths = cam.project(sim.kpts3d[:, t].reshape(-1, 3), return_depth=True)
    uvs = uvs.reshape(n, K, 2)
    depths = depths.reshape(n, K)
    label = np.full((h, w), -1, dtype=np.int32)
    order = np.argsort(-depths.mean(1))  # far to near
    for a in order:
        if np.any(depths[a] <= 1e-3):
            continue
        px_len = cam.K[0, 0] * sim.appearance[a].size / depths[a].mean()
        canvas = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), np.uint8)
        coat = tuple(int(np.clip(c * rng.uniform(0.95, 1.05), 0, 255)) for c in sim.appearance[a].coat)
        _draw_animal(canvas, mask, uvs[a], px_len, coat, depths[a], sim.appearance[a].size)
        m = mask > 0
        img[m] = canvas[m]
        label[m] = a
    visible = np.zeros((n, K), dtype=bool)
    boxes = np.zeros((n, 4), dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    xi = np.clip(np.round(uvs[..., 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(uvs[..., 1]).astype(int), 0, h - 1)
    inside = cam.in_image(uvs) & (depths > 0)
    for a in range(n):
        visible[a] = inside[a] & (label[yi[a], xi[a]] == a)
        ys, xs = np.nonzero(label == a)
        full_area = max(1.0, (cam.K[0, 0] * sim.appearance[a].size / max(depths[a].mean(), 1e-3)) ** 2 * 0.15)
        if len(xs) >= max(20, 0.1 * full_area) and visible[a].sum() >= 3:
            boxes[a] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            present[a] = True
    if noise:
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        img += rng.normal(0, rng.uniform(1, 6), size=img.shape)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
    return Frame(np.clip(img, 0, 255).astype(np.uint8), uvs.astype(np.float32), visible, boxes, present,
                 np.arange(n))


def render_video(sim: Simulation, cam: Camera, seed: int = 0) -> list[Frame]:
    """Render every frame with a fixed background (as from a static camera)."""
    rng = np.random.default_rng(seed)
    bg = _background(cam, sim, rng)
    return [render(sim, cam, t, rng, background=bg) for t in range(sim.n_frames)]


class SimulationPool:
    """Pre-simulated long sequences; sampling windows from it is much cheaper
    than simulating per batch (used by the temporal and behavior trainers)."""

    def __init__(self, n_sims: int = 64, frames: int = 600, species=("mouse", "rat"), max_animals: int = 3,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sims = [simulate(int(rng.integers(1, max_animals + 1)), frames, str(rng.choice(list(species))), seed=rng)
                     for _ in range(n_sims)]

    def window(self, rng: np.random.Generator, length: int, warmup: int = 30) -> tuple[Simulation, int, int]:
        """Random ``(sim, animal, start)`` with ``start >= warmup``."""
        sim = self.sims[int(rng.integers(len(self.sims)))]
        start = int(rng.integers(warmup, max(warmup + 1, sim.n_frames - length + 1)))
        return sim, int(rng.integers(sim.n_animals)), start
