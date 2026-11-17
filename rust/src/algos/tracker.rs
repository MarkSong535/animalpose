//! Multi-animal tracker: Hungarian association on motion, pose and appearance,
//! with re-identification after occlusion.
//!
//! Port of `animalpose.tracking.tracker`. Appearance embeddings are *inputs*:
//! this crate never computes them, it only scores precomputed vectors.
//!
//! Per frame the Python routine (1) Kalman-predicts every track, (2) matches
//! active tracks to high-confidence detections on IoU + OKS + appearance,
//! (3) matches leftover tracks to low-confidence detections on IoU only,
//! (4) re-identifies lost tracks from their embedding gallery, and (5) opens new
//! identities from the leftover high-confidence detections.

use crate::algos::geometry::{box_iou_pairwise, Box4};
use crate::algos::hungarian::linear_sum_assignment;
use crate::algos::kalman::BoxKalman;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Cost of a pair the gate forbids (`_INF` in
/// `animalpose.tracking.tracker`).
pub const INF: f64 = 1e6;
/// Default `thresh` of `animalpose.tracking.tracker.oks`.
pub const OKS_DEFAULT_THRESH: f64 = 0.3;
/// Default `kappa` of `animalpose.tracking.tracker.oks`.
pub const OKS_DEFAULT_KAPPA: f64 = 0.1;

/// A detection in one frame: box, score and the optional pose/appearance data
/// the association can use.
///
/// Port of `animalpose.tracking.tracker.Detection`.
///
/// // differs from Python: the field is called `bbox` (`box` is a Rust
/// // keyword) and embeddings are `f32` rather than `float64`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// xyxy box in image pixels.
    pub bbox: Box4,
    /// Detection score in `[0, 1]`.
    pub score: f64,
    /// Keypoints `(K, 2)` in image pixels.
    pub keypoints: Option<Vec<[f64; 2]>>,
    /// Keypoint confidences `(K,)`.
    pub kp_conf: Option<Vec<f64>>,
    /// L2-normalised appearance embedding.
    pub embedding: Option<Vec<f32>>,
    /// Per-pair left/right evidence `(P,)`.
    pub pair_margin: Option<Vec<f64>>,
}

impl Detection {
    /// A bare box and score (Python's positional arguments).
    pub fn new(bbox: Box4, score: f64) -> Self {
        Self {
            bbox,
            score,
            ..Self::default()
        }
    }

    /// Attaches pose output (Python's `keypoints=` and `kp_conf=`).
    pub fn with_keypoints(mut self, keypoints: Vec<[f64; 2]>, kp_conf: Vec<f64>) -> Self {
        self.keypoints = Some(keypoints);
        self.kp_conf = Some(kp_conf);
        self
    }

    /// Attaches a precomputed embedding (Python's `embedding=`).
    pub fn with_embedding(mut self, embedding: &[f32]) -> Self {
        self.embedding = Some(embedding.to_vec());
        self
    }

    /// Attaches pair margins (Python's `pair_margin=`).
    pub fn with_pair_margin(mut self, pair_margin: Vec<f64>) -> Self {
        self.pair_margin = Some(pair_margin);
        self
    }
}

/// Track lifecycle, the `state` strings of `animalpose.tracking.tracker.Track`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackState {
    /// Not yet confirmed: fewer than `n_init` observations.
    Tentative,
    /// Reported to the caller.
    Confirmed,
    /// Occluded: kept for re-identification.
    Lost,
    /// Removed at the end of the frame.
    Dead,
}

/// Association thresholds and weights.
///
/// Port of `animalpose.tracking.tracker.TrackerConfig` with the same defaults;
/// instead of Python's `**overrides` the fields are public.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackerConfig {
    /// Score at or above which a detection is trusted.
    pub high_thresh: f64,
    /// Score at or above which a detection can rescue a track.
    pub low_thresh: f64,
    /// Weight of the `1 - IoU` term.
    pub w_iou: f64,
    /// Weight of the `1 - OKS` term.
    pub w_oks: f64,
    /// Weight of the appearance term.
    pub w_app: f64,
    /// Largest cost a stage-1 match may have.
    pub match_thresh: f64,
    /// Largest cost a stage-2 (low-score) match may have.
    pub low_match_thresh: f64,
    /// Largest cost a re-identification may have.
    pub reid_thresh: f64,
    /// Centre jump, in Mahalanobis units, that forbids a non-overlapping pair.
    pub gate_mahalanobis: f64,
    /// Observations a tentative track needs before it is confirmed.
    pub n_init: usize,
    /// Frames a lost track stays re-identifiable (ignored with `max_animals`).
    pub max_lost: usize,
    /// Known cohort size: identities are never created beyond this number.
    pub max_animals: Option<usize>,
    /// Smoothing of the track embedding.
    pub ema: f64,
    /// Number of recent embeddings kept for re-identification.
    pub gallery_size: usize,
    /// Keypoint confidence below which a keypoint is ignored by OKS.
    pub kp_conf_thresh: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            high_thresh: 0.5,
            low_thresh: 0.2,
            w_iou: 0.5,
            w_oks: 0.3,
            w_app: 0.2,
            match_thresh: 0.8,
            low_match_thresh: 0.5,
            reid_thresh: 0.35,
            gate_mahalanobis: 50.0,
            n_init: 3,
            max_lost: 90,
            max_animals: None,
            ema: 0.9,
            gallery_size: 30,
            kp_conf_thresh: 0.3,
        }
    }
}

/// A tracked animal.
///
/// Port of `animalpose.tracking.tracker.Track`; the embedding gallery is the
/// `deque(maxlen=gallery_size)` of recent detection embeddings.
#[derive(Clone, Debug)]
pub struct Track {
    /// Identity, monotonically increasing.
    pub id: i64,
    /// Box filter of this track.
    pub kf: BoxKalman,
    /// Smoothed appearance embedding.
    pub embedding: Option<Vec<f32>>,
    /// Recent detection embeddings used for re-identification.
    pub gallery: VecDeque<Vec<f32>>,
    /// Last observed pose.
    pub keypoints: Option<Vec<[f64; 2]>>,
    /// Confidence of the last observed pose.
    pub kp_conf: Option<Vec<f64>>,
    /// Frames matched so far.
    pub hits: usize,
    /// Frames since the track was created.
    pub age: usize,
    /// Frames since the last match.
    pub time_since_update: usize,
    /// Lifecycle state.
    pub state: TrackState,
    /// Last observed box.
    pub last_box: Box4,
}

/// Object keypoint similarity with a uniform per-keypoint falloff.
///
/// Port of `animalpose.tracking.tracker.oks`; `None` is Python's `NaN` (fewer
/// than two keypoints above `thresh`).
pub fn oks(
    kp_a: &[[f64; 2]],
    conf_a: &[f64],
    kp_b: &[[f64; 2]],
    conf_b: &[f64],
    scale: f64,
    thresh: f64,
    kappa: f64,
) -> Option<f64> {
    let n = kp_a
        .len()
        .min(conf_a.len())
        .min(kp_b.len())
        .min(conf_b.len());
    let mut total = 0.0;
    let mut used = 0usize;
    for i in 0..n {
        if conf_a[i] > thresh && conf_b[i] > thresh {
            let dx = kp_a[i][0] - kp_b[i][0];
            let dy = kp_a[i][1] - kp_b[i][1];
            let d2 = dx * dx + dy * dy;
            total += (-d2 / (2.0 * (scale * kappa).powi(2) + 1e-9)).exp();
            used += 1;
        }
    }
    if used < 2 {
        return None;
    }
    Some(total / used as f64)
}

/// Half the clipped cosine distance of two embeddings; `None` is Python's `NaN`
/// (either side missing).
///
/// Port of `animalpose.tracking.tracker._cos_dist`.
pub fn cos_dist(a: Option<&[f32]>, b: Option<&[f32]>) -> Option<f64> {
    let (a, b) = (a?, b?);
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    Some(((1.0 - dot / (na * nb + 1e-9)).clamp(0.0, 2.0)) / 2.0)
}

/// Which rows/columns a cost matrix matched, plus the leftovers.
///
/// Port of the return value of `animalpose.tracking.tracker.MultiAnimalTracker._assign`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Assignment {
    /// Matched `(row, column)` pairs.
    pub matches: Vec<(usize, usize)>,
    /// Rows without a match.
    pub unmatched_rows: Vec<usize>,
    /// Columns without a match.
    pub unmatched_cols: Vec<usize>,
}

/// Optimal assignment keeping only pairs whose cost is at most `thresh`
/// (port of `MultiAnimalTracker._assign`).
pub fn assign(cost: &[Vec<f64>], thresh: f64) -> Assignment {
    let cols = cost.first().map_or(0, |r| r.len());
    assign_cols(cost, cols, thresh)
}

/// [`assign`] for a cost matrix whose column count is known separately.
///
/// // differs from Python: numpy can express a `(0, M)` matrix, `&[Vec<f64>]`
/// // cannot, so the tracker passes `high.len()` and friends explicitly.
pub fn assign_cols(cost: &[Vec<f64>], cols: usize, thresh: f64) -> Assignment {
    let empty = cols == 0 || cost.is_empty() || cost.iter().any(|r| r.is_empty());
    let pairs = if empty {
        Vec::new()
    } else {
        linear_sum_assignment(cost).unwrap_or_default()
    };
    if empty {
        return Assignment {
            matches: Vec::new(),
            unmatched_rows: (0..cost.len()).collect(),
            unmatched_cols: (0..cols).collect(),
        };
    }
    let mut out = Assignment::default();
    let mut matched_rows = vec![false; cost.len()];
    let mut matched_cols = vec![false; cols];
    for (r, c) in pairs {
        if cost[r][c] <= thresh {
            out.matches.push((r, c));
            matched_rows[r] = true;
            matched_cols[c] = true;
        }
    }
    out.unmatched_rows = (0..cost.len()).filter(|r| !matched_rows[*r]).collect();
    out.unmatched_cols = (0..cols).filter(|c| !matched_cols[*c]).collect();
    out
}

/// The tracker itself.
///
/// Port of `animalpose.tracking.tracker.MultiAnimalTracker`.
#[derive(Clone, Debug)]
pub struct MultiAnimalTracker {
    /// Association configuration.
    pub cfg: TrackerConfig,
    /// Tracks alive after the last frame.
    pub tracks: Vec<Track>,
    next_id: i64,
    frame: usize,
}

impl MultiAnimalTracker {
    /// A tracker with the given configuration (Python's `MultiAnimalTracker(cfg)`).
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
            next_id: 0,
            frame: 0,
        }
    }

    /// A tracker with the default configuration.
    pub fn with_defaults() -> Self {
        Self::new(TrackerConfig::default())
    }

    /// Number of frames processed so far.
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Stage-1 cost of pairing `track` with `det`: `1 - IoU`, `1 - OKS` on the
    /// motion-compensated last pose and the appearance distance, weighted and
    /// averaged over the available terms.
    ///
    /// Port of `MultiAnimalTracker._pair_cost`.
    pub fn pair_cost(&self, track: &Track, det: &Detection, pred_box: Box4, iou: f64) -> f64 {
        let cfg = &self.cfg;
        if iou <= 0.0 && track.kf.mahalanobis(det.bbox) > cfg.gate_mahalanobis {
            return INF;
        }
        let mut terms = vec![1.0 - iou];
        let mut weights = vec![cfg.w_iou];
        if let (Some(kp_a), Some(kp_b), Some(conf_a), Some(conf_b)) = (
            &track.keypoints,
            &det.keypoints,
            &track.kp_conf,
            &det.kp_conf,
        ) {
            // Motion-compensate the last pose with the predicted box centre.
            let shift = [
                (pred_box[0] + pred_box[2]) / 2.0 - (track.last_box[0] + track.last_box[2]) / 2.0,
                (pred_box[1] + pred_box[3]) / 2.0 - (track.last_box[1] + track.last_box[3]) / 2.0,
            ];
            let scale = ((det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]))
                .max(1.0)
                .sqrt();
            let shifted: Vec<[f64; 2]> = kp_a
                .iter()
                .map(|p| [p[0] + shift[0], p[1] + shift[1]])
                .collect();
            if let Some(s) = oks(
                &shifted,
                conf_a,
                kp_b,
                conf_b,
                scale,
                cfg.kp_conf_thresh,
                OKS_DEFAULT_KAPPA,
            ) {
                terms.push(1.0 - s);
                weights.push(cfg.w_oks);
            }
        }
        if let Some(a) = cos_dist(track.embedding.as_deref(), det.embedding.as_deref()) {
            terms.push(a);
            weights.push(cfg.w_app);
        }
        let num: f64 = terms.iter().zip(weights.iter()).map(|(t, w)| t * w).sum();
        let den: f64 = weights.iter().sum();
        num / den
    }

    /// Re-identification cost of a lost track against a detection: the smallest
    /// gallery cosine distance plus a loose spatial prior that widens with the
    /// time lost.
    ///
    /// Port of `MultiAnimalTracker._reid_cost`.
    pub fn reid_cost(&self, track: &Track, det: &Detection) -> f64 {
        let diag = ((track.last_box[2] - track.last_box[0]).powi(2)
            + (track.last_box[3] - track.last_box[1]).powi(2))
        .sqrt()
            + 1e-6;
        let dc = [
            (det.bbox[0] + det.bbox[2]) / 2.0 - (track.last_box[0] + track.last_box[2]) / 2.0,
            (det.bbox[1] + det.bbox[3]) / 2.0 - (track.last_box[1] + track.last_box[3]) / 2.0,
        ];
        let dist = (dc[0] * dc[0] + dc[1] * dc[1]).sqrt() / diag;
        let spatial = (dist / (1.0 + 0.1 * track.time_since_update as f64) - 1.0).max(0.0);
        let Some(embedding) = det.embedding.as_deref() else {
            return 0.5 * spatial + 0.1;
        };
        if track.gallery.is_empty() {
            return 0.5 * spatial + 0.1;
        }
        let app = track
            .gallery
            .iter()
            .filter_map(|g| cos_dist(Some(g.as_slice()), Some(embedding)))
            .fold(f64::INFINITY, f64::min);
        app + 0.05 * spatial
    }

    /// Applies a detection to a track: Kalman update, pose and embedding EMA,
    /// gallery append, hit counters and the tentative/lost promotion.
    ///
    /// Port of `MultiAnimalTracker._apply`.
    fn apply(&mut self, index: usize, det: &Detection) {
        let (ema, gallery_size, n_init) = (self.cfg.ema, self.cfg.gallery_size, self.cfg.n_init);
        let tr = &mut self.tracks[index];
        // A singular innovation covariance (never seen in practice) keeps the
        // prediction, where numpy would raise.
        let _ = tr.kf.update(det.bbox);
        tr.last_box = det.bbox;
        if det.keypoints.is_some() {
            tr.keypoints = det.keypoints.clone();
            tr.kp_conf = det.kp_conf.clone();
        }
        if let Some(e) = &det.embedding {
            let merged: Vec<f32> = match &tr.embedding {
                None => e.clone(),
                Some(old) => old
                    .iter()
                    .zip(e.iter())
                    .map(|(a, b)| (ema * f64::from(*a) + (1.0 - ema) * f64::from(*b)) as f32)
                    .collect(),
            };
            let norm: f64 = merged
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt();
            tr.embedding = Some(
                merged
                    .iter()
                    .map(|v| (f64::from(*v) / (norm + 1e-9)) as f32)
                    .collect(),
            );
            tr.gallery.push_back(e.clone());
            while gallery_size < tr.gallery.len() {
                tr.gallery.pop_front();
            }
        }
        tr.hits += 1;
        tr.time_since_update = 0;
        if tr.state == TrackState::Lost || (tr.state == TrackState::Tentative && tr.hits >= n_init)
        {
            tr.state = TrackState::Confirmed;
        }
    }

    /// Opens a new identity (port of `MultiAnimalTracker._new_track`) and
    /// returns its id.
    fn new_track(&mut self, det: &Detection) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut track = Track {
            id,
            kf: BoxKalman::new(det.bbox),
            embedding: det.embedding.clone(),
            gallery: VecDeque::new(),
            keypoints: det.keypoints.clone(),
            kp_conf: det.kp_conf.clone(),
            hits: 1,
            age: 1,
            time_since_update: 0,
            state: TrackState::Tentative,
            last_box: det.bbox,
        };
        if let Some(e) = &det.embedding {
            track.gallery.push_back(e.clone());
        }
        if self.cfg.max_animals.is_some() || self.frame == 0 {
            track.state = TrackState::Confirmed; // known cohort / first frame
        }
        self.tracks.push(track);
        id
    }

    /// Processes one frame, returning `{track_id: detection_index}` for the
    /// confirmed tracks matched in this frame.
    ///
    /// Port of `MultiAnimalTracker.update`.
    ///
    /// // differs from Python: the mapping is sorted by track id (Python's
    /// // `dict` keeps insertion order).
    pub fn update(&mut self, detections: &[Detection]) -> BTreeMap<i64, usize> {
        let cfg = self.cfg.clone();
        for track in &mut self.tracks {
            track.kf.predict();
            track.age += 1;
        }
        let active: Vec<usize> = (0..self.tracks.len())
            .filter(|i| self.tracks[*i].state != TrackState::Lost)
            .collect();
        let lost: Vec<usize> = (0..self.tracks.len())
            .filter(|i| self.tracks[*i].state == TrackState::Lost)
            .collect();
        let high: Vec<usize> = (0..detections.len())
            .filter(|i| detections[*i].score >= cfg.high_thresh)
            .collect();
        let low: Vec<usize> = (0..detections.len())
            .filter(|i| {
                detections[*i].score >= cfg.low_thresh && detections[*i].score < cfg.high_thresh
            })
            .collect();
        let mut assigned: BTreeMap<i64, usize> = BTreeMap::new();

        // Stage 1: active tracks vs high-confidence detections.
        let preds: Vec<Box4> = active.iter().map(|i| self.tracks[*i].kf.bbox()).collect();
        let high_boxes: Vec<Box4> = high.iter().map(|j| detections[*j].bbox).collect();
        let ious = box_iou_pairwise(&preds, &high_boxes);
        let mut cost = Vec::new();
        for (r, ti) in active.iter().enumerate() {
            let row = high
                .iter()
                .enumerate()
                .map(|(ci, dj)| {
                    let iou = ious[r * high.len() + ci];
                    self.pair_cost(&self.tracks[*ti], &detections[*dj], preds[r], iou)
                })
                .collect();
            cost.push(row);
        }
        let stage1 = assign_cols(&cost, high.len(), cfg.match_thresh);
        for (r, ci) in stage1.matches {
            let ti = active[r];
            let dj = high[ci];
            self.apply(ti, &detections[dj]);
            assigned.insert(self.tracks[ti].id, dj);
        }
        let mut rem_tracks = stage1
            .unmatched_rows
            .iter()
            .map(|r| active[*r])
            .collect::<Vec<_>>();
        let mut rem_high = stage1
            .unmatched_cols
            .iter()
            .map(|c| high[*c])
            .collect::<Vec<_>>();

        // Stage 2: leftover active tracks vs low-confidence detections (IoU only).
        if !rem_tracks.is_empty() && !low.is_empty() {
            let preds2: Vec<Box4> = rem_tracks
                .iter()
                .map(|i| self.tracks[*i].kf.bbox())
                .collect();
            let low_boxes: Vec<Box4> = low.iter().map(|j| detections[*j].bbox).collect();
            let iou2 = box_iou_pairwise(&preds2, &low_boxes);
            let cost2: Vec<Vec<f64>> = (0..rem_tracks.len())
                .map(|r| {
                    (0..low.len())
                        .map(|c| 1.0 - iou2[r * low.len() + c])
                        .collect()
                })
                .collect();
            let stage2 = assign_cols(&cost2, low.len(), cfg.low_match_thresh);
            for (r, ci) in stage2.matches {
                let ti = rem_tracks[r];
                let dj = low[ci];
                self.apply(ti, &detections[dj]);
                assigned.insert(self.tracks[ti].id, dj);
            }
            rem_tracks = stage2
                .unmatched_rows
                .iter()
                .map(|r| rem_tracks[*r])
                .collect::<Vec<_>>();
        }

        // Stage 3: re-identify lost tracks.
        let relost: Vec<usize>;
        if !lost.is_empty() && !rem_high.is_empty() {
            let cost3: Vec<Vec<f64>> = lost
                .iter()
                .map(|ti| {
                    rem_high
                        .iter()
                        .map(|dj| self.reid_cost(&self.tracks[*ti], &detections[*dj]))
                        .collect()
                })
                .collect();
            let forced = cfg.max_animals.is_some() && self.tracks.len() >= cfg.max_animals.unwrap();
            let thresh = if forced { INF / 2.0 } else { cfg.reid_thresh };
            let stage3 = assign_cols(&cost3, rem_high.len(), thresh);
            let matched: Vec<usize> = stage3.matches.iter().map(|(r, _)| lost[*r]).collect();
            for (r, ci) in &stage3.matches {
                let ti = lost[*r];
                let dj = rem_high[*ci];
                self.apply(ti, &detections[dj]);
                assigned.insert(self.tracks[ti].id, dj);
            }
            relost = lost
                .iter()
                .filter(|ti| !matched.contains(ti))
                .copied()
                .collect();
            rem_high = stage3
                .unmatched_cols
                .iter()
                .map(|c| rem_high[*c])
                .collect::<Vec<_>>();
        } else {
            relost = lost.clone();
        }

        // Unmatched active tracks: tentative ones die, confirmed ones go lost.
        for ti in &rem_tracks {
            let tr = &mut self.tracks[*ti];
            tr.time_since_update += 1;
            tr.state = if tr.state == TrackState::Tentative {
                TrackState::Dead
            } else {
                TrackState::Lost
            };
        }
        for ti in &relost {
            self.tracks[*ti].time_since_update += 1;
        }
        if cfg.max_animals.is_none() {
            for tr in &mut self.tracks {
                if tr.state == TrackState::Lost && tr.time_since_update > cfg.max_lost {
                    tr.state = TrackState::Dead;
                }
            }
        }
        self.tracks.retain(|t| t.state != TrackState::Dead);

        // Stage 4: new identities.
        for dj in rem_high {
            if let Some(max) = cfg.max_animals {
                if self.tracks.len() >= max {
                    break;
                }
            }
            let id = self.new_track(&detections[dj]);
            assigned.insert(id, dj);
        }
        self.frame += 1;
        let confirmed: Vec<i64> = self
            .tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .map(|t| t.id)
            .collect();
        assigned
            .into_iter()
            .filter(|(id, _)| confirmed.contains(id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EA: &[f32] = &[1.0, 0.0, 0.0];
    const EB: &[f32] = &[0.0, 1.0, 0.0];
    const EC: &[f32] = &[0.0, 0.0, 1.0];

    /// A straight line of boxes, the synthetic stand-in for the rendered video
    /// of `tests/test_tracking.py::test_tracker_keeps_identities_on_synthetic_video`.
    fn moving(c: usize, t: i64, offset: f64) -> Box4 {
        let x = 2.0 * t as f64 + offset;
        [x, 10.0 + 40.0 * c as f64, x + 20.0, 30.0 + 40.0 * c as f64]
    }

    fn frame_dets(t: i64, animals: &[usize]) -> Vec<Detection> {
        let emb = [EA, EB, EC];
        animals
            .iter()
            .map(|c| Detection::new(moving(*c, t, 0.0), 0.9).with_embedding(emb[*c]))
            .collect()
    }

    /// `{frame: {detection_index: track_id}}` from one run.
    fn run(
        frames: usize,
        cfg: TrackerConfig,
        present: impl Fn(usize) -> Vec<usize>,
    ) -> Vec<BTreeMap<i64, usize>> {
        let mut tracker = MultiAnimalTracker::new(cfg);
        (0..frames)
            .map(|t| tracker.update(&frame_dets(t as i64, &present(t))))
            .collect()
    }

    fn identity_switches(out: &[BTreeMap<i64, usize>]) -> usize {
        let mut switches = 0;
        let mut prev: BTreeMap<usize, i64> = BTreeMap::new();
        for frame in out {
            let cur: BTreeMap<usize, i64> = frame.iter().map(|(id, det)| (*det, *id)).collect();
            if !prev.is_empty() && cur.len() == prev.len() {
                switches += cur
                    .iter()
                    .filter(|(d, i)| prev.get(*d).copied() != Some(**i))
                    .count();
            }
            if !cur.is_empty() {
                prev = cur;
            }
        }
        switches
    }

    #[test]
    fn tracker_keeps_identities_on_a_synthetic_video() {
        let out = run(150, TrackerConfig::default(), |_| vec![0, 1]);
        // Both animals are confirmed from the very first frame (frame 0 tracks
        // are trusted immediately, exactly as in Python).
        assert_eq!(out.iter().filter(|f| f.len() == 2).count(), 150);
        let mut unique: Vec<i64> = out.iter().flat_map(|f| f.keys()).copied().collect();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique, vec![0, 1]);
        assert_eq!(identity_switches(&out), 0);
        assert_eq!(out[60].values().copied().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn reidentification_after_long_occlusion() {
        // Animal 1 vanishes for 70 frames; with a known cohort of two it must
        // come back under its old id (port of test_reidentification_after_long_occlusion).
        let cfg = TrackerConfig {
            max_animals: Some(2),
            ..TrackerConfig::default()
        };
        let out = run(160, cfg, |t| {
            if (40..110).contains(&t) {
                vec![0]
            } else {
                vec![0, 1]
            }
        });
        let mut ids: Vec<i64> = out.iter().flat_map(|f| f.keys()).copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, vec![0, 1], "exactly two identities over the video");
        assert_eq!(identity_switches(&out), 0);
        assert_eq!(out.iter().filter(|f| f.len() == 2).count(), 160 - 70);
        // The re-identified track keeps id 1 on the frame it reappears.
        assert_eq!(out[110].values().copied().collect::<Vec<_>>(), vec![0, 1]);
    }

    /// `_reid_cost` without appearance: the distance between box centres in box diagonals,
    /// allowed to grow with the time lost. A detection straight below the track is as far as one
    /// the same distance to the side (the vertical centre once mixed in the track's bottom edge).
    #[test]
    fn reid_cost_measures_the_distance_between_box_centres() {
        let mut tracker = MultiAnimalTracker::with_defaults();
        tracker.update(&[Detection::new([100.0, 100.0, 130.0, 140.0], 0.9)]);
        let track = tracker.tracks[0].clone();
        let diag = 50.0; // the 30 × 40 box
        let below = Detection::new([100.0, 300.0, 130.0, 340.0], 0.9); // 200 px down
        let aside = Detection::new([300.0, 100.0, 330.0, 140.0], 0.9); // 200 px right
        let expected = 0.5 * (200.0 / diag / (1.0 + 0.1 * track.time_since_update as f64) - 1.0) + 0.1;
        assert!((tracker.reid_cost(&track, &below) - expected).abs() < 1e-6);
        assert!((tracker.reid_cost(&track, &aside) - expected).abs() < 1e-6);
        // In place again: only the base cost.
        let here = Detection::new([100.0, 100.0, 130.0, 140.0], 0.9);
        assert!((tracker.reid_cost(&track, &here) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn reidentification_works_without_a_known_cohort_too() {
        let out = run(160, TrackerConfig::default(), |t| {
            if (40..110).contains(&t) {
                vec![0]
            } else {
                vec![0, 1]
            }
        });
        let mut ids: Vec<i64> = out.iter().flat_map(|f| f.keys()).copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(identity_switches(&out), 0);
    }

    #[test]
    fn new_tracks_stay_tentative_until_n_init_hits() {
        let out = run(40, TrackerConfig::default(), |t| {
            if t >= 20 {
                vec![0, 1, 2]
            } else {
                vec![0, 1]
            }
        });
        for (t, frame) in out.iter().enumerate().take(22).skip(20) {
            assert_eq!(frame.len(), 2, "frame {t} must not report the new animal");
        }
        assert_eq!(out[22].len(), 3);
        assert!(out[22].values().copied().collect::<Vec<_>>().contains(&2));
    }

    #[test]
    fn max_animals_forbids_new_identities() {
        let cfg = TrackerConfig {
            max_animals: Some(2),
            ..TrackerConfig::default()
        };
        let mut tracker = MultiAnimalTracker::new(cfg);
        for t in 0..10i64 {
            let mut dets = frame_dets(t, &[0, 1]);
            if t >= 5 {
                dets.push(Detection::new(moving(2, t, 500.0), 0.9).with_embedding(EC));
            }
            tracker.update(&dets);
        }
        assert_eq!(tracker.tracks.len(), 2);
        assert_eq!(tracker.next_id, 2);
    }

    #[test]
    fn tracks_die_after_max_lost_frames() {
        let cfg = TrackerConfig {
            max_lost: 3,
            ..TrackerConfig::default()
        };
        let mut tracker = MultiAnimalTracker::new(cfg);
        tracker.update(&frame_dets(0, &[0, 1]));
        for t in 1..10i64 {
            tracker.update(&frame_dets(t, &[0]));
        }
        assert_eq!(tracker.tracks.len(), 1);
    }

    #[test]
    fn low_confidence_detections_rescue_a_track() {
        let mut tracker = MultiAnimalTracker::new(TrackerConfig::default());
        tracker.update(&frame_dets(0, &[0, 1]));
        let mut det = frame_dets(1, &[0])[0].clone();
        det.score = 0.3; // between low_thresh and high_thresh
        det.bbox = moving(0, 1, 1.0);
        let out = tracker.update(&[det]);
        assert_eq!(out.values().copied().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn oks_is_nan_below_two_valid_keypoints() {
        let kp = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let conf = [1.0, 1.0, 0.1];
        let same = oks(
            &kp,
            &conf,
            &kp,
            &conf,
            10.0,
            OKS_DEFAULT_THRESH,
            OKS_DEFAULT_KAPPA,
        );
        assert!(same.unwrap() > 0.99);
        let only_one = oks(&kp, &[1.0, 0.1, 0.1], &kp, &[1.0, 0.1, 0.1], 10.0, 0.3, 0.1);
        assert!(only_one.is_none());
        let far = oks(
            &kp,
            &conf,
            &[[100.0, 100.0], [120.0, 120.0], [2.0, 2.0]],
            &conf,
            10.0,
            0.3,
            0.1,
        );
        assert!(far.unwrap() < 0.05);
    }

    #[test]
    fn cos_distance_is_scale_invariant_and_clipped() {
        assert!((cos_dist(Some(EA), Some(EB)).unwrap() - 0.5).abs() < 1e-9);
        assert!(cos_dist(Some(EA), Some(EA)).unwrap().abs() < 1e-9);
        assert!((cos_dist(Some(EA), Some(&[-1.0, 0.0, 0.0])).unwrap() - 1.0).abs() < 1e-9);
        assert!(cos_dist(None, Some(EA)).is_none());
        assert!(cos_dist(Some(EA), None).is_none());
        // Python divides by the norms; an un-normalised vector scores the same.
        let scaled = [3.0f32, 0.0, 0.0];
        assert!(cos_dist(Some(&scaled), Some(EA)).unwrap().abs() < 1e-9);
    }

    #[test]
    fn gated_pairs_cost_the_infinity_marker() {
        let tracker = MultiAnimalTracker::with_defaults();
        let det = Detection::new([500.0, 500.0, 520.0, 520.0], 0.9).with_embedding(EA);
        let mut track = Track {
            id: 0,
            kf: BoxKalman::new([0.0, 0.0, 20.0, 20.0]),
            embedding: Some(EA.to_vec()),
            gallery: VecDeque::from([EA.to_vec()]),
            keypoints: None,
            kp_conf: None,
            hits: 4,
            age: 4,
            time_since_update: 0,
            state: TrackState::Confirmed,
            last_box: [0.0, 0.0, 20.0, 20.0],
        };
        let pred = track.kf.bbox();
        assert_eq!(tracker.pair_cost(&track, &det, pred, 0.0), INF);
        // Overlapping pairs are never gated out.
        let iou = 0.2;
        let cost = tracker.pair_cost(&track, &det, pred, iou);
        assert!(cost < INF);
        // Appearance and pose terms are averaged with their weights.
        track.embedding = Some(EA.to_vec());
        let det_same = Detection::new(pred, 0.9).with_embedding(EA);
        let c = tracker.pair_cost(&track, &det_same, pred, 1.0);
        assert!(c.abs() < 1e-9, "{c}");
    }

    #[test]
    fn reid_prefers_appearance_over_a_distant_box() {
        let tracker = MultiAnimalTracker::with_defaults();
        let track = Track {
            id: 0,
            kf: BoxKalman::new([0.0, 0.0, 20.0, 20.0]),
            embedding: Some(EA.to_vec()),
            gallery: VecDeque::from([EA.to_vec()]),
            keypoints: None,
            kp_conf: None,
            hits: 5,
            age: 20,
            time_since_update: 15,
            state: TrackState::Lost,
            last_box: [0.0, 0.0, 20.0, 20.0],
        };
        let near = Detection::new([40.0, 0.0, 60.0, 20.0], 0.9).with_embedding(EA);
        let other = Detection::new([0.0, 0.0, 20.0, 20.0], 0.9).with_embedding(EB);
        assert!(tracker.reid_cost(&track, &near) < tracker.cfg.reid_thresh);
        assert!(tracker.reid_cost(&track, &other) > tracker.cfg.reid_thresh);
        // Without a gallery only a nearby re-appearance can match.
        let track = Track {
            gallery: VecDeque::new(),
            ..track
        };
        assert!(tracker.reid_cost(&track, &near) < tracker.cfg.reid_thresh);
        let far = Detection::new([400.0, 0.0, 420.0, 20.0], 0.9).with_embedding(EA);
        assert!(tracker.reid_cost(&track, &far) > tracker.cfg.reid_thresh);
    }

    #[test]
    fn an_empty_row_set_still_reports_every_column() {
        // Python: `_assign(np.zeros((0, 3)))` returns all three columns unmatched.
        let a = assign(&[], 0.8);
        assert!(a.unmatched_rows.is_empty());
        assert!(a.unmatched_cols.is_empty());
        let a = assign_cols(&[], 3, 0.8);
        assert!(a.matches.is_empty());
        assert_eq!(a.unmatched_rows, Vec::<usize>::new());
        assert_eq!(a.unmatched_cols, vec![0, 1, 2]);
    }

    #[test]
    fn assignment_thresholds_leave_the_rest_unmatched() {
        let cost = vec![vec![0.1, 0.9], vec![0.9, 0.2]];
        let a = assign(&cost, 0.8);
        assert_eq!(a.matches, vec![(0, 0), (1, 1)]);
        assert!(a.unmatched_rows.is_empty() && a.unmatched_cols.is_empty());
        let a = assign(&cost, 0.15);
        assert_eq!(a.matches, vec![(0, 0)]);
        assert_eq!(a.unmatched_rows, vec![1]);
        assert_eq!(a.unmatched_cols, vec![1]);
        // An empty cost matrix leaves everything unmatched, as in Python.
        let a = assign(&[vec![], vec![]], 0.5);
        assert_eq!(a.unmatched_rows, vec![0, 1]);
        assert!(a.unmatched_cols.is_empty());
        let a = assign(&[], 0.5);
        assert!(a.unmatched_rows.is_empty() && a.unmatched_cols.is_empty());
    }

    #[test]
    fn embeddings_are_smoothed_and_kept_in_a_gallery() {
        let mut tracker = MultiAnimalTracker::new(TrackerConfig {
            gallery_size: 2,
            ..TrackerConfig::default()
        });
        let d0 = Detection::new(moving(0, 0, 0.0), 0.9).with_embedding(EA);
        tracker.update(std::slice::from_ref(&d0));
        let d1 = Detection::new(moving(0, 1, 0.0), 0.9).with_embedding(EB);
        tracker.update(&[d1]);
        let tr = &tracker.tracks[0];
        assert_eq!(tr.gallery.len(), 2);
        let e = tr.embedding.as_ref().unwrap();
        // ema * e_prev + (1 - ema) * e_new, renormalised: 0.9*EA + 0.1*EB.
        let norm = (0.9_f64 * 0.9 + 0.1 * 0.1).sqrt();
        assert!((f64::from(e[0]) - 0.9 / norm).abs() < 1e-6, "{e:?}");
        assert!((f64::from(e[1]) - 0.1 / norm).abs() < 1e-6, "{e:?}");
        assert!(f64::from(e[2]).abs() < 1e-9);
        assert_eq!(tr.state, TrackState::Confirmed);
        assert_eq!(tr.hits, 2);
    }
}
