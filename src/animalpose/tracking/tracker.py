"""Multi-animal tracker: Hungarian association on motion, pose and appearance,
with re-identification after occlusion.

Per frame:

1. Kalman-predict every track's box.
2. Match active tracks to high-confidence detections. Cost combines
   ``1 - IoU`` with the Kalman-predicted box, ``1 - OKS`` between the
   track's motion-compensated last pose and the detection's pose, and the
   cosine distance between appearance embeddings. Implausible pairs (centre
   jump beyond the Mahalanobis gate and no overlap) are forbidden.
3. Match leftover active tracks to low-confidence detections on IoU only
   (ByteTrack-style recovery of partially occluded animals).
4. Re-identify: leftover detections vs *lost* tracks by appearance (min
   distance to each track's embedding gallery) with a loose spatial prior
   that widens with time lost.
5. New tracks from leftover high-confidence detections. With
   ``max_animals`` set (the usual lab case of a known cohort), identities are
   never created beyond that number: a new detection is instead assigned to
   the best-matching lost identity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from animalpose.geometry import box_iou
from animalpose.tracking.kalman import BoxKalman

_INF = 1e6


@dataclass
class Detection:
    box: np.ndarray  # xyxy
    score: float
    keypoints: np.ndarray | None = None  # (K, 2)
    kp_conf: np.ndarray | None = None  # (K,)
    embedding: np.ndarray | None = None  # (E,), L2-normalised
    pair_margin: np.ndarray | None = None  # (P,) left/right evidence per bilateral pair


@dataclass
class Track:
    id: int
    kf: BoxKalman
    embedding: np.ndarray | None
    gallery: deque
    keypoints: np.ndarray | None
    kp_conf: np.ndarray | None
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: str = "tentative"  # tentative | confirmed | lost
    last_box: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5
    low_thresh: float = 0.2
    w_iou: float = 0.5
    w_oks: float = 0.3
    w_app: float = 0.2
    match_thresh: float = 0.8
    low_match_thresh: float = 0.5
    reid_thresh: float = 0.35
    gate_mahalanobis: float = 50.0
    n_init: int = 3
    max_lost: int = 90  # frames a lost track stays re-identifiable (ignored with max_animals)
    max_animals: int | None = None
    ema: float = 0.9
    gallery_size: int = 30
    kp_conf_thresh: float = 0.3


def oks(kp_a: np.ndarray, conf_a: np.ndarray, kp_b: np.ndarray, conf_b: np.ndarray, scale: float,
        thresh: float = 0.3, kappa: float = 0.1) -> float:
    """Object keypoint similarity with a uniform per-keypoint falloff."""
    valid = (conf_a > thresh) & (conf_b > thresh)
    if valid.sum() < 2:
        return np.nan
    d2 = np.sum((kp_a[valid] - kp_b[valid]) ** 2, axis=-1)
    return float(np.mean(np.exp(-d2 / (2 * (scale * kappa) ** 2 + 1e-9))))


def _cos_dist(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return np.nan
    return float(np.clip(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), 0, 2) / 2)


class MultiAnimalTracker:
    def __init__(self, config: TrackerConfig | None = None, **overrides):
        self.cfg = config or TrackerConfig()
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        self.tracks: list[Track] = []
        self._next_id = 0
        self.frame = 0

    # -------------------------------------------------------------- costs
    def _pair_cost(self, tr: Track, det: Detection, pred_box: np.ndarray, iou: float) -> float:
        c = self.cfg
        if iou <= 0 and tr.kf.mahalanobis(det.box) > c.gate_mahalanobis:
            return _INF
        terms, weights = [1 - iou], [c.w_iou]
        if tr.keypoints is not None and det.keypoints is not None:
            shift = (pred_box[:2] + pred_box[2:]) / 2 - (tr.last_box[:2] + tr.last_box[2:]) / 2
            scale = np.sqrt(max((det.box[2] - det.box[0]) * (det.box[3] - det.box[1]), 1.0))
            s = oks(tr.keypoints + shift, tr.kp_conf, det.keypoints, det.kp_conf, scale, c.kp_conf_thresh)
            if np.isfinite(s):
                terms.append(1 - s), weights.append(c.w_oks)
        a = _cos_dist(tr.embedding, det.embedding)
        if np.isfinite(a):
            terms.append(a), weights.append(c.w_app)
        return float(np.dot(terms, weights) / np.sum(weights))

    def _reid_cost(self, tr: Track, det: Detection) -> float:
        # Loose spatial prior: distance in box diagonals, allowed to grow with time lost.
        diag = np.hypot(*(tr.last_box[2:] - tr.last_box[:2])) + 1e-6
        dist = np.linalg.norm((det.box[:2] + det.box[2:]) / 2 - (tr.last_box[:2] + tr.last_box[2:]) / 2) / diag
        spatial = max(0.0, dist / (1 + 0.1 * tr.time_since_update) - 1.0)
        if not tr.gallery or det.embedding is None:
            # No appearance: only a nearby re-appearance can be the same animal. The cost is
            # scaled so it passes ``reid_thresh`` within ~half a diagonal of the expected range.
            return 0.5 * spatial + 0.1
        app = min(_cos_dist(g, det.embedding) for g in tr.gallery)
        return app + 0.05 * spatial

    @staticmethod
    def _assign(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if cost.size == 0:
            return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
        mr = {r for r, _ in matches}
        mc = {c for _, c in matches}
        return matches, [r for r in range(cost.shape[0]) if r not in mr], [c for c in range(cost.shape[1]) if c not in mc]

    # -------------------------------------------------------------- update
    def _apply(self, tr: Track, det: Detection) -> None:
        c = self.cfg
        tr.kf.update(det.box)
        tr.last_box = np.asarray(det.box, dtype=np.float64)
        if det.keypoints is not None:
            tr.keypoints, tr.kp_conf = det.keypoints, det.kp_conf
        if det.embedding is not None:
            e = det.embedding if tr.embedding is None else c.ema * tr.embedding + (1 - c.ema) * det.embedding
            tr.embedding = e / (np.linalg.norm(e) + 1e-9)
            tr.gallery.append(det.embedding)
        tr.hits += 1
        tr.time_since_update = 0
        if tr.state == "lost" or (tr.state == "tentative" and tr.hits >= c.n_init):
            tr.state = "confirmed"

    def _new_track(self, det: Detection) -> Track:
        tr = Track(self._next_id, BoxKalman(det.box), det.embedding, deque(maxlen=self.cfg.gallery_size),
                   det.keypoints, det.kp_conf, last_box=np.asarray(det.box, dtype=np.float64))
        if det.embedding is not None:
            tr.gallery.append(det.embedding)
        if self.cfg.max_animals is not None or self.frame == 0:
            tr.state = "confirmed"  # known cohort / first frame: trust immediately
        self._next_id += 1
        self.tracks.append(tr)
        return tr

    def update(self, detections: list[Detection]) -> dict[int, int]:
        """Process one frame. Returns ``{track_id: detection_index}`` for
        confirmed tracks matched in this frame."""
        c = self.cfg
        for tr in self.tracks:
            tr.kf.predict()
            tr.age += 1
        active = [t for t in self.tracks if t.state != "lost"]
        lost = [t for t in self.tracks if t.state == "lost"]
        high = [i for i, d in enumerate(detections) if d.score >= c.high_thresh]
        low = [i for i, d in enumerate(detections) if c.low_thresh <= d.score < c.high_thresh]
        assigned: dict[int, int] = {}

        # Stage 1: active tracks vs high-confidence detections.
        preds = np.array([t.kf.box for t in active]).reshape(-1, 4)
        ious = box_iou(preds, np.array([detections[i].box for i in high]).reshape(-1, 4))
        cost = np.array([[self._pair_cost(t, detections[j], preds[r], ious[r, ci]) for ci, j in enumerate(high)]
                         for r, t in enumerate(active)]).reshape(len(active), len(high))
        m, um_t, um_d = self._assign(cost, c.match_thresh)
        for r, ci in m:
            self._apply(active[r], detections[high[ci]])
            assigned[active[r].id] = high[ci]
        rem_tracks = [active[r] for r in um_t]
        rem_high = [high[ci] for ci in um_d]

        # Stage 2: leftover active tracks vs low-confidence detections (IoU only).
        if rem_tracks and low:
            preds2 = np.array([t.kf.box for t in rem_tracks])
            iou2 = box_iou(preds2, np.array([detections[i].box for i in low]))
            m2, um_t2, _ = self._assign(1 - iou2, c.low_match_thresh)
            for r, ci in m2:
                self._apply(rem_tracks[r], detections[low[ci]])
                assigned[rem_tracks[r].id] = low[ci]
            rem_tracks = [rem_tracks[r] for r in um_t2]

        # Stage 3: re-identify lost tracks (also tracks lost this frame stay eligible next frame).
        if lost and rem_high:
            cost3 = np.array([[self._reid_cost(t, detections[j]) for j in rem_high] for t in lost])
            forced = c.max_animals is not None and len(self.tracks) >= c.max_animals
            m3, _, um_d3 = self._assign(cost3, _INF / 2 if forced else c.reid_thresh)
            for r, ci in m3:
                self._apply(lost[r], detections[rem_high[ci]])
                assigned[lost[r].id] = rem_high[ci]
            rem_high = [rem_high[ci] for ci in um_d3]

        # Unmatched active tracks: tentative ones die, confirmed ones become lost.
        for tr in rem_tracks:
            tr.time_since_update += 1
            if tr.state == "tentative":
                tr.state = "dead"
            else:
                tr.state = "lost"
        for tr in lost:
            if tr.id not in assigned:
                tr.time_since_update += 1
        if c.max_animals is None:
            for tr in self.tracks:
                if tr.state == "lost" and tr.time_since_update > c.max_lost:
                    tr.state = "dead"
        self.tracks = [t for t in self.tracks if t.state != "dead"]

        # Stage 4: new identities.
        for j in rem_high:
            if c.max_animals is not None and len(self.tracks) >= c.max_animals:
                break
            tr = self._new_track(detections[j])
            assigned[tr.id] = j
        self.frame += 1
        confirmed = {t.id for t in self.tracks if t.state == "confirmed"}
        return {tid: j for tid, j in assigned.items() if tid in confirmed}


class ColorHistogramEmbedder:
    """Appearance fallback: normalised hue-saturation histogram of the box
    centre. Useful before a pose network with a trained embedding exists."""

    def __init__(self, bins: tuple[int, int] = (12, 6)):
        self.bins = bins

    def __call__(self, image_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image_bgr.shape[:2]
        x0, y0, x1, y1 = box
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.3, (y1 - y0) * 0.3
        xa, xb = int(max(0, cx - bw)), int(min(w, cx + bw + 1))
        ya, yb = int(max(0, cy - bh)), int(min(h, cy + bh + 1))
        patch = image_bgr[ya:yb, xa:xb]
        if patch.size == 0:
            return np.ones(self.bins[0] * self.bins[1] + 1) / np.sqrt(self.bins[0] * self.bins[1] + 1)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, list(self.bins), [0, 180, 0, 256]).ravel()
        v = np.append(hist / max(hist.sum(), 1), hsv[..., 2].mean() / 255.0)
        return v / (np.linalg.norm(v) + 1e-9)
