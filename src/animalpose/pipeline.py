"""End-to-end inference: frames -> detections -> per-animal pose -> tracks ->
temporal refinement -> behavior; and the multi-camera variant with identity
fusion and triangulation."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from animalpose.behavior.catalog import NAMES as CATALOG_NAMES
from animalpose.behavior.catalog import body_kinematics, describe, full_probabilities, top_k
from animalpose.behavior.model import BehaviorTCN
from animalpose.behavior.postprocess import adapt_prior, median_smooth
from animalpose.data.datasets import letterbox, to_tensor
from animalpose.detector.model import CenterDetector, DetectorExport
from animalpose.device import select_device
from animalpose.geometry import apply_affine, box_xyxy_to_cs, crop_affine, invert_affine, warp
from animalpose.multiview.camera import Camera
from animalpose.multiview.fusion import ViewTrack, fuse_identities, triangulate_identities
from animalpose.pose.model import PoseExport, PoseNet
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS
from animalpose.temporal.model import TemporalLSTM, smooth_sequence
from animalpose.temporal.pairs import resolve_pair_identity
from animalpose.tracking.tracker import ColorHistogramEmbedder, Detection, MultiAnimalTracker, TrackerConfig


@dataclass
class PipelineConfig:
    det_size: tuple[int, int] = (256, 320)  # (H, W), multiples of 32
    det_thresh: float = 0.3
    max_det: int = 20
    crop_size: int = 128
    crop_padding: float = 1.25
    kp_conf_thresh: float = 0.3
    # Keep bilateral pairs on distinct heatmap peaks (pose.model.decode_pairs). Fixes both
    # paws of a pair collapsing onto one paw in side views; needed for multi-view L/R resolution.
    pair_decoding: bool = True
    # Viterbi over each track that keeps bilateral labels on continuously moving limbs
    # (temporal.pairs.resolve_pair_identity); needs pair_decoding.
    pair_identity: bool = True
    pair_motion_frac: float = 0.1  # chosen on the mirror-mouse InD videos (animalpose eval mirror-videos)
    # Static-object suppression for fixed cameras: drop detections whose box barely differs from the
    # median background of the recording (cups, bottles, toys). Off by default: an animal that does
    # not move for most of a recording would be removed too.
    background_filter: bool = False
    background_frames: int = 64  # frames, spread over the recording, for the median background
    foreground_diff: float = 25.0  # grey levels that count as "different from background"
    min_foreground: float = 0.08  # minimum share of box pixels that differ
    # Detection candidates in addition to the network ("network", "blob"). A blurry animal often
    # scores below the detector threshold although its silhouette against a static background is
    # unmistakable, so the moving blob of a fixed camera proposes a box the network can then label.
    detector_candidates: tuple[str, ...] = ("network",)
    blob_score: float = 0.9  # score given to blob boxes (above tracker.high_thresh by default)
    blob_diff: float | None = None  # grey levels; defaults to foreground_diff
    blob_open_px: int = 5  # morphological opening that removes the thin tail
    blob_arena_roi: bool = True  # look only inside the bright floor of the background
    blob_area: tuple[float, float] | None = None  # plausible blob area in px^2; default: learned
    blob_calibrate: int = 64  # frames used to learn that range; 0 keeps blob_area as-is
    merge_iou: float = 0.5  # a candidate overlapping a higher-scoring box at or above this is dropped
    # Behavior input must match how the classifier was trained: frame rate (frames are
    # subsampled to it), confidence threshold of the keypoint mask, and optionally only the
    # keypoints of a DATASET_MAPPINGS entry (e.g. "homecage_dlc" for models trained on DLC tracks).
    behavior_fps: float | None = None
    behavior_conf: float = 0.6
    behavior_keypoints: str | None = None
    behavior_top_k: int = 3
    # Post-processing of the classifier's frame-wise probabilities (animalpose.behavior.postprocess):
    # a moving median removes single-frame flicker, and re-estimating the class prior on the recording
    # being scored corrects a behavior mix the training sessions never showed (a corner camera sees
    # rearing far more often than a top camera). Both need no labels; both are transductive over the clip.
    behavior_smooth: int = 5  # moving-median window in frames; 1 disables it
    behavior_prior_adapt: bool = True
    behavior_prior_max_ratio: float = 4.0  # how far a class prior may move
    embedder: str = "model"  # "model" (pose-net embedding), "color" (HSV histogram) or "none"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    temporal_window: int = 64
    batch_crops: int = 32


@dataclass
class TrackResult:
    track_id: int
    frames: np.ndarray  # (T,)
    boxes: np.ndarray  # (T, 4), NaN where unmatched
    keypoints: np.ndarray  # (T, K, 2) raw network output, NaN where unmatched
    conf: np.ndarray  # (T, K)
    smoothed: np.ndarray | None = None  # (T, K, 2) temporal-model output
    smoothed_conf: np.ndarray | None = None
    imputed: np.ndarray | None = None  # (T, K) 1 = predicted, not observed
    pair_margin: np.ndarray | None = None  # (T, P) per-frame left/right evidence
    pair_swapped: np.ndarray | None = None  # (T, P) True where the temporal pass swapped the pair
    behavior: np.ndarray | None = None  # (T,) class index
    behavior_probs: np.ndarray | None = None  # (T, C) over the classifier's own classes
    behavior_full: np.ndarray | None = None  # (T, len(catalog.NAMES)) over the full behavior catalogue

    @property
    def best(self) -> np.ndarray:
        return self.smoothed if self.smoothed is not None else self.keypoints


@dataclass
class VideoResult:
    tracks: dict[int, TrackResult]
    n_frames: int
    behaviors: tuple[str, ...] = ()
    behavior_info: list | None = None  # catalog.describe(): which behaviors are modelled and how
    times: np.ndarray | None = None  # (n_frames,) presentation time in seconds (variable frame rates)

    def to_npz(self, path: str | Path) -> None:
        arrays = {}
        for tid, tr in self.tracks.items():
            for name in ("frames", "boxes", "keypoints", "conf", "smoothed", "smoothed_conf", "imputed",
                         "behavior", "behavior_probs", "behavior_full"):
                v = getattr(tr, name)
                if v is not None:
                    arrays[f"track{tid}_{name}"] = v
        if self.times is not None:
            arrays["times"] = self.times
        np.savez_compressed(path, keypoint_names=np.array(CANONICAL.names), behaviors=np.array(self.behaviors), **arrays)

    def time_of(self, frame) -> str:
        return f"{self.times[int(frame)]:.4f}" if self.times is not None and int(frame) < len(self.times) else ""

    def to_behavior_csv(self, path: str | Path, k: int = 3) -> None:
        """One row per (frame, track): the top-``k`` behaviors with their probabilities, then the
        probability of every behavior in the catalogue."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s", "track"] + [f"top{i + 1}{s}" for i in range(k) for s in ("", "_p")]
                       + [f"p_{n}" for n in self.behaviors])
            for tid, tr in self.tracks.items():
                if tr.behavior_full is None:
                    continue
                idx, pv = top_k(tr.behavior_full, k)
                for i, f in enumerate(tr.frames):
                    top = [x for j in range(k) for x in (self.behaviors[idx[i, j]], f"{pv[i, j]:.3f}")]
                    w.writerow([int(f), self.time_of(f), tid] + top + [f"{p:.4f}" for p in tr.behavior_full[i]])

    def to_csv(self, path: str | Path) -> None:
        """Long format: one row per (frame, track, keypoint)."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s", "track", "keypoint", "x", "y", "confidence", "imputed", "behavior"])
            for tid, tr in self.tracks.items():
                pts = tr.best
                cf = tr.smoothed_conf if tr.smoothed_conf is not None else tr.conf
                for i, f in enumerate(tr.frames):
                    beh = self.behaviors[tr.behavior[i]] if tr.behavior is not None else ""
                    ts = self.time_of(f)
                    for k, name in enumerate(CANONICAL.names):
                        imp = int(tr.imputed[i, k]) if tr.imputed is not None else int(not np.isfinite(pts[i, k, 0]))
                        w.writerow([int(f), ts, tid, name, f"{pts[i, k, 0]:.2f}", f"{pts[i, k, 1]:.2f}",
                                    f"{np.nan_to_num(cf[i, k]):.3f}", imp, beh])


def _gray(img: np.ndarray) -> np.ndarray:
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img).astype(np.float32)


def median_background(frames: list[np.ndarray], n: int = 64) -> np.ndarray:
    """Per-pixel median of up to ``n`` frames spread over the recording (grey levels)."""
    step = max(1, len(frames) // n)
    return np.median(np.stack([_gray(f) for f in frames[::step]]), 0)


def foreground_fraction(img: np.ndarray, bg: np.ndarray, box, diff: float = 25.0) -> float:
    """Share of pixels inside ``box`` (xyxy) that differ from the background by more than ``diff``."""
    h, w = bg.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((np.abs(_gray(img)[y0:y1, x0:x1] - bg[y0:y1, x0:x1]) > diff).mean())


def arena_roi(bg: np.ndarray, grow: int = 15) -> np.ndarray:
    """Bright floor region of the background (where a dark animal can be seen), slightly grown."""
    thr, m = cv2.threshold(bg.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(m, np.ones((grow, grow), np.uint8))


def largest_moving_blob(img: np.ndarray, bg: np.ndarray, diff: float, dark: bool = True, open_px: int = 5,
                        area: tuple[float, float] | None = None, roi: np.ndarray | None = None):
    """``(xyxy box, pixel area)`` of the largest moving blob, or ``(None, 0)`` when absent or ambiguous.

    ``dark`` looks for pixels darker than the background (a dark animal on a bright floor);
    ``False`` takes any change. ``area`` rejects implausible sizes and ``roi`` areas outside the
    floor; a second blob of over 30 % of the largest one makes the frame ambiguous.
    """
    g = _gray(img)
    d = (bg - g) if dark else np.abs(g - bg)
    m = (d > diff).astype(np.uint8)
    if roi is not None:
        m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    k, _, st, _ = cv2.connectedComponentsWithStats(m)
    if k < 2:
        return None, 0.0
    order = np.argsort(-st[1:, 4]) + 1
    a0 = float(st[order[0], 4])
    if area is not None and not area[0] <= a0 <= area[1]:
        return None, a0
    if len(order) > 1 and st[order[1], 4] > 0.3 * a0:
        return None, a0
    x, y, w, h = st[order[0], :4]
    pad = 0.15 * max(w, h)
    H, W = g.shape
    return np.array([max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)], np.float32), a0


def blob_box(img: np.ndarray, bg: np.ndarray, diff: float, dark: bool = True, open_px: int = 5,
             area: tuple[float, float] | None = None, roi: np.ndarray | None = None):
    """The largest moving blob as an xyxy box, or ``None`` when absent or ambiguous."""
    return largest_moving_blob(img, bg, diff, dark, open_px, area, roi)[0]


class BlobProposer:
    """Moving-blob detection candidates for a fixed camera, one call per frame.

    Built from the recording's median background. When ``area`` is not given it is learned from
    the first ``calibrate`` accepted blobs (0.15x to 2.5x their median), so a recording of a
    30 px animal rejects debris and cups of other sizes without any configuration.
    """

    def __init__(self, bg: np.ndarray, diff: float = 25.0, score: float = 0.9, open_px: int = 5,
                 area: tuple[float, float] | None = None, calibrate: int = 64, roi: np.ndarray | None = None,
                 dark: bool = True):
        self.bg, self.diff, self.score, self.open_px, self.dark = bg, diff, score, open_px, dark
        self.roi, self.area, self.calibrate, self.sizes = roi, area, calibrate, []

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        box, a0 = largest_moving_blob(img, self.bg, self.diff, self.dark, self.open_px,
                                      None if self.area is None else self.area, self.roi)
        if box is None:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        if self.area is None and self.calibrate:
            self.sizes.append(a0)
            if len(self.sizes) >= self.calibrate:
                med = float(np.median(self.sizes))
                self.area = (0.15 * med, 2.5 * med)
        return box[None], np.array([self.score], np.float32)


def merge_candidates(boxes: np.ndarray, scores: np.ndarray, extra_boxes: np.ndarray, extra_scores: np.ndarray,
                     iou_thresh: float = 0.5, max_det: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Union of two sets of boxes, best score first; a box overlapping a better one by IoU >=
    ``iou_thresh`` is dropped, and at most ``max_det`` survive."""
    extra = np.asarray(extra_boxes, np.float32).reshape(-1, 4)
    boxes = np.concatenate([np.asarray(boxes, np.float32).reshape(-1, 4), extra])
    scores = np.concatenate([np.asarray(scores, np.float32).ravel(), np.asarray(extra_scores, np.float32).ravel()])
    keep = []
    for i in np.argsort(-scores):
        b = boxes[i]
        if any(_iou(b, boxes[j]) >= iou_thresh for j in keep):
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
    if not keep:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    idx = np.asarray(keep)
    return boxes[idx], scores[idx]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (max(a[2] - a[0], 0) * max(a[3] - a[1], 0) + max(b[2] - b[0], 0) * max(b[3] - b[1], 0) - inter + 1e-9)


class PosePipeline:
    def __init__(self, detector: CenterDetector, pose: PoseNet, temporal: TemporalLSTM | None = None,
                 behavior: BehaviorTCN | None = None, config: PipelineConfig | None = None, device="auto"):
        self.cfg = config or PipelineConfig()
        self.device = select_device(device)
        self.detector = DetectorExport(detector.eval(), self.cfg.max_det).to(self.device).eval()
        self.pose = PoseExport(pose.eval(), self.cfg.pair_decoding).to(self.device).eval()
        self.temporal = temporal.to(self.device).eval() if temporal is not None else None
        self.behavior = behavior.to(self.device).eval() if behavior is not None else None
        self.color = ColorHistogramEmbedder()
        self.behavior_prior: np.ndarray | None = None  # class prior estimated on the last clip scored

    @classmethod
    def from_checkpoints(cls, detector: str, pose: str, temporal: str | None = None, behavior: str | None = None,
                         config: PipelineConfig | None = None, device="auto"):
        from animalpose.train.checkpoint import load_model

        return cls(load_model(detector), load_model(pose), load_model(temporal) if temporal else None,
                   load_model(behavior) if behavior else None, config, device)

    # ------------------------------------------------------------------ per frame
    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img, s, pad = letterbox(image_bgr, self.cfg.det_size)
        boxes, scores = self.detector(to_tensor(img)[None].to(self.device))
        boxes, scores = boxes[0].cpu().numpy(), scores[0].cpu().numpy()
        keep = scores >= min(self.cfg.det_thresh, self.cfg.tracker.low_thresh)
        boxes = (boxes[keep] - np.tile(pad, 2)) / s
        h, w = image_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores[keep]

    @torch.no_grad()
    def estimate(self, image_bgr: np.ndarray, boxes: np.ndarray, return_margins: bool = False):
        """Top-down pose for each box -> keypoints ``(N,K,2)`` image px, conf ``(N,K)``, embeddings ``(N,E)``
        (and pair margins ``(N,P)`` with ``return_margins``)."""
        n = len(boxes)
        k = CANONICAL.num_keypoints
        if n == 0:
            empty = (np.zeros((0, k, 2), np.float32), np.zeros((0, k), np.float32), np.zeros((0, 0), np.float32))
            return (*empty, np.zeros((0, len(self.pose.pairs)), np.float32)) if return_margins else empty
        S = self.cfg.crop_size
        mats = [crop_affine(*box_xyxy_to_cs(b, padding=self.cfg.crop_padding), S) for b in boxes]
        crops = torch.stack([to_tensor(warp(image_bgr, m, S)) for m in mats]).to(self.device)
        kps, confs, embs, margins = [], [], [], []
        for i in range(0, n, self.cfg.batch_crops):
            batch = crops[i:i + self.cfg.batch_crops]
            kp, cf, em, *mg = self.pose.with_margins(batch) if return_margins else self.pose(batch)
            kps.append(kp.cpu().numpy()), confs.append(cf.cpu().numpy()), embs.append(em.cpu().numpy())
            margins.append(mg[0].cpu().numpy() if mg else None)
        kp, cf, em = np.concatenate(kps), np.concatenate(confs), np.concatenate(embs)
        kp = np.stack([apply_affine(invert_affine(m), p) for m, p in zip(mats, kp)]).astype(np.float32)
        return (kp, cf, em, np.concatenate(margins)) if return_margins else (kp, cf, em)

    def proposals(self, background: np.ndarray | None):
        """Extra detection candidates as ``image -> (boxes, scores)``, from ``detector_candidates``.

        Blob candidates come from the recording's median background, so ``background`` is required
        (see :func:`median_background` and :func:`sampled_background`). Returns ``None`` when the
        network is the only candidate source.
        """
        if "blob" not in self.cfg.detector_candidates:
            return None
        if background is None:
            raise ValueError("blob candidates need a median background: pass background= or set background_filter")
        cfg = self.cfg
        return BlobProposer(background, diff=cfg.blob_diff or cfg.foreground_diff, score=cfg.blob_score,
                            open_px=cfg.blob_open_px, area=cfg.blob_area, calibrate=cfg.blob_calibrate,
                            roi=arena_roi(background) if cfg.blob_arena_roi else None)

    def detect_candidates(self, image_bgr: np.ndarray, propose=None) -> tuple[np.ndarray, np.ndarray]:
        """Network boxes merged with ``propose(image) -> (boxes, scores)`` candidates."""
        boxes, scores = self.detect(image_bgr)
        if propose is not None:
            boxes, scores = merge_candidates(boxes, scores, *propose(image_bgr), self.cfg.merge_iou, self.cfg.max_det)
        return boxes, scores

    def frame_detections(self, image_bgr: np.ndarray, boxes=None, scores=None, propose=None) -> list[Detection]:
        if boxes is None:
            boxes, scores = self.detect_candidates(image_bgr, propose)
        use_pairs = self.cfg.pair_decoding and self.cfg.pair_identity
        kp, cf, em, *mg = self.estimate(image_bgr, boxes, return_margins=use_pairs)
        dets = []
        for i, b in enumerate(boxes):
            emb = (self.color(image_bgr, b) if self.cfg.embedder == "color"
                   else None if self.cfg.embedder == "none" else em[i])
            dets.append(Detection(np.asarray(b, np.float64), float(scores[i]), kp[i], cf[i], emb,
                                  mg[0][i] if mg else None))
        return dets

    # ------------------------------------------------------------------ video
    def track(self, frames, detections_fn=None, background: np.ndarray | None = None) -> VideoResult:
        """Detect, estimate and track over an iterable of BGR frames.

        ``detections_fn(t, image) -> (boxes, scores)`` overrides the detector
        (e.g. ground-truth boxes for ablations). With ``background_filter`` or blob
        ``detector_candidates``, pass a precomputed ``background`` (grey median, see
        :func:`median_background`) to stream long recordings; otherwise the frames are held
        in memory to compute it.
        """
        tracker = MultiAnimalTracker(TrackerConfig(**vars(self.cfg.tracker)))
        per_track: dict[int, dict[int, Detection]] = {}
        n = 0
        need_bg = self.cfg.background_filter or "blob" in self.cfg.detector_candidates
        bg = background if need_bg else None
        if need_bg and bg is None:
            frames = list(frames)
            bg = median_background(frames, self.cfg.background_frames)
        propose = self.proposals(bg) if need_bg else None
        for t, img in enumerate(frames):
            boxes, scores = detections_fn(t, img) if detections_fn else (None, None)
            dets = self.frame_detections(img, boxes, scores, propose)
            if bg is not None:
                dets = [d for d in dets if foreground_fraction(img, bg, d.box, self.cfg.foreground_diff)
                        >= self.cfg.min_foreground]
            for tid, j in tracker.update(dets).items():
                per_track.setdefault(tid, {})[t] = dets[j]
            n = t + 1
        k = CANONICAL.num_keypoints
        tracks = {}
        for tid, obs in per_track.items():
            f0, f1 = min(obs), max(obs)
            frames = np.arange(f0, f1 + 1)
            T = len(frames)
            boxes = np.full((T, 4), np.nan, np.float32)
            kps = np.full((T, k, 2), np.nan, np.float32)
            cf = np.zeros((T, k), np.float32)
            n_pairs = len(self.pose.pairs)
            margin = np.zeros((T, n_pairs), np.float32)
            for f, d in obs.items():
                boxes[f - f0], kps[f - f0], cf[f - f0] = d.box, d.keypoints, d.kp_conf
                if d.pair_margin is not None:
                    margin[f - f0] = d.pair_margin
            tracks[tid] = TrackResult(tid, frames, boxes, kps, cf)
            if any(d.pair_margin is not None for d in obs.values()):
                size = np.fmax(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
                tracks[tid].pair_margin = margin
                tracks[tid].keypoints, tracks[tid].pair_swapped = resolve_pair_identity(
                    kps, margin, self.pose.pairs.cpu().numpy(), np.nan_to_num(size, nan=1.0), self.cfg.pair_motion_frac)
        return VideoResult(tracks, n, tuple(self.behavior.classes) if self.behavior is not None else ())

    def refine(self, result: VideoResult, min_len: int = 5, fps: float | None = None,
               times: np.ndarray | None = None) -> VideoResult:
        """Temporal smoothing/gap filling, then behavior: the classifier's probabilities per
        track, mapped onto the full catalogue with kinematic and two-animal rules
        (:mod:`animalpose.behavior.catalog`). ``fps`` is the nominal frame rate; ``times``
        (seconds per frame) handles variable frame rates and dropped frames."""
        fps = fps or 30.0
        n = max([int(tr.frames[-1]) + 1 for tr in result.tracks.values()] + [result.n_frames, 0])
        if times is None or len(times) < n:  # unknown or incomplete: constant rate from fps
            known = np.asarray(times if times is not None else [], float)
            times = np.r_[known, (known[-1] if len(known) else -1.0 / fps) + np.arange(1, n - len(known) + 1) / fps]
        result.times = np.asarray(times[:n], float)
        kin = {}
        for tid, tr in result.tracks.items():
            if len(tr.frames) < min_len:
                continue
            observed = (tr.conf >= self.cfg.kp_conf_thresh) & np.isfinite(tr.keypoints).all(-1)
            if self.temporal is not None:
                tr.smoothed, tr.smoothed_conf, tr.imputed = smooth_sequence(
                    self.temporal, tr.keypoints, tr.conf, observed, self.cfg.temporal_window)
            if self.behavior is not None:
                t_tr = result.times[tr.frames.astype(int)]
                tr.behavior_probs = self._behavior_probs(tr, t_tr)
                conf = tr.smoothed_conf if tr.smoothed_conf is not None else tr.conf
                kin[tid] = body_kinematics(tr.best, conf, fps, t_tr)
        if self.behavior is None or not kin:
            return result
        modelled = None
        for tid, tr in result.tracks.items():
            if tid not in kin:
                continue
            partners = [self._align(result.tracks[o], kin[o], tr.frames) for o in kin if o != tid]
            tr.behavior_full, modelled = full_probabilities(self.behavior.classes, tr.behavior_probs, kin[tid], partners)
            tr.behavior = tr.behavior_full.argmax(-1)
        result.behaviors = CATALOG_NAMES
        result.behavior_info = describe(modelled)
        return result

    def _behavior_probs(self, tr: TrackResult, times: np.ndarray) -> np.ndarray:
        """Classifier probabilities per frame. With ``behavior_fps`` the track is first resampled
        onto an even time grid at that rate (so variable frame rates and dropped frames do not
        change the classifier's sense of time); each frame then takes its nearest grid sample."""
        conf = tr.smoothed_conf if tr.smoothed_conf is not None else tr.conf
        mask = (np.nan_to_num(conf) >= self.cfg.behavior_conf) & np.isfinite(tr.best).all(-1)
        if self.cfg.behavior_keypoints:
            mask &= CANONICAL.mask(list(DATASET_MAPPINGS[self.cfg.behavior_keypoints].values()))
        pts = tr.best
        if self.cfg.behavior_fps:
            grid = np.arange(times[0], times[-1] + 0.5 / self.cfg.behavior_fps, 1.0 / self.cfg.behavior_fps)
            near = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
            prev = np.clip(near - 1, 0, len(times) - 1)
            near = np.where(np.abs(times[prev] - grid) < np.abs(times[near] - grid), prev, near)
            x = np.full((len(grid),) + pts.shape[1:], np.nan)
            for k in range(pts.shape[1]):
                ok = mask[:, k]
                if ok.sum() >= 2:
                    for d in range(pts.shape[2]):
                        x[:, k, d] = np.interp(grid, times[ok], pts[ok, k, d])
            m = mask[near]
        else:
            grid, x, m = times, pts, mask
        with torch.no_grad():
            logits = self.behavior(torch.from_numpy(np.nan_to_num(x).astype(np.float32))[None].to(self.device),
                                   torch.from_numpy(m.astype(np.float32))[None].to(self.device))
        probs = torch.softmax(logits, -1)[0].cpu().numpy()
        if self.cfg.behavior_smooth > 1:
            probs = median_smooth(probs, self.cfg.behavior_smooth)
        if self.cfg.behavior_prior_adapt:
            probs, self.behavior_prior = adapt_prior(probs, max_ratio=self.cfg.behavior_prior_max_ratio)
        back = np.clip(np.searchsorted(grid, times), 0, len(grid) - 1)
        prev = np.clip(back - 1, 0, len(grid) - 1)
        back = np.where(np.abs(grid[prev] - times) < np.abs(grid[back] - times), prev, back)
        return probs[back]

    @staticmethod
    def _align(other: TrackResult, k: dict, frames: np.ndarray) -> dict:
        """Another track's kinematics on ``frames`` (NaN where it is absent)."""
        pos = frames - int(other.frames[0])
        ok = (pos >= 0) & (pos < len(other.frames))
        out = {}
        for key, v in k.items():
            if isinstance(v, np.ndarray):
                a = np.full((len(frames),) + v.shape[1:], np.nan)
                a[ok] = v[pos[ok]]
                out[key] = a
            else:
                out[key] = v
        return out

    def run(self, frames, detections_fn=None, background: np.ndarray | None = None,
            fps: float | None = None, times: np.ndarray | None = None) -> VideoResult:
        return self.refine(self.track(frames, detections_fn, background), fps=fps, times=times)


class MultiViewPipeline:
    """Synchronised cameras: per-view tracking, cross-view identity fusion,
    robust triangulation, optional 3D temporal smoothing and 3D behavior."""

    def __init__(self, pipeline: PosePipeline, cameras: list[Camera], temporal3d: TemporalLSTM | None = None,
                 behavior3d: BehaviorTCN | None = None, fusion_max_cost: float = 15.0, reproj_thresh: float = 10.0,
                 max_animals: int | None = None, resolve_left_right: bool = False):
        self.pipe = pipeline
        self.cameras = cameras
        self.temporal3d = temporal3d.eval() if temporal3d is not None else None
        self.behavior3d = behavior3d.eval() if behavior3d is not None else None
        self.fusion_max_cost = fusion_max_cost
        self.reproj_thresh = reproj_thresh
        self.max_animals = max_animals
        self.resolve_left_right = resolve_left_right

    def run(self, views: list, detections_fns: list | None = None) -> dict:
        per_view = [self.pipe.track(frames, detections_fns[v] if detections_fns else None)
                    for v, frames in enumerate(views)]
        n_frames = max(r.n_frames for r in per_view)
        vtracks = [ViewTrack(v, tid, tr.frames, tr.keypoints, tr.conf)
                   for v, r in enumerate(per_view) for tid, tr in r.tracks.items()]
        ids = fuse_identities(vtracks, self.cameras, self.fusion_max_cost, self.max_animals)
        animals = triangulate_identities(vtracks, self.cameras, ids, n_frames, self.pipe.cfg.kp_conf_thresh,
                                         self.reproj_thresh, self.resolve_left_right)
        for a in animals.values():
            X = a["keypoints3d"]
            obs = np.isfinite(X).all(-1)
            if self.temporal3d is not None and obs.any():
                conf = np.where(obs, 1.0 / (1.0 + np.nan_to_num(a["reproj_error"], nan=1e3) / self.reproj_thresh), 0.0)
                a["smoothed3d"], a["confidence3d"], a["imputed"] = smooth_sequence(
                    self.temporal3d, np.nan_to_num(X), conf.astype(np.float32), obs, self.pipe.cfg.temporal_window)
            if self.behavior3d is not None:
                pts = a.get("smoothed3d", np.nan_to_num(X))
                mask = np.ones(obs.shape, np.float32) if "smoothed3d" in a else obs.astype(np.float32)
                dev = next(self.behavior3d.parameters()).device
                with torch.no_grad():
                    probs = torch.softmax(self.behavior3d(torch.from_numpy(pts.astype(np.float32))[None].to(dev),
                                                          torch.from_numpy(mask)[None].to(dev)), -1)[0].cpu().numpy()
                a["behavior_probs"], a["behavior"] = probs, probs.argmax(-1)
        return {"identities": ids, "animals": animals, "per_view": per_view}
