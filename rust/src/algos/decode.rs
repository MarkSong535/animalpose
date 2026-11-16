//! Inference-time decoding of network heat maps: detector boxes and pose
//! keypoints, with sub-pixel offsets.
//!
//! Port of the `decode` / `decode_pairs` functions of
//! `animalpose.detector.model` and `animalpose.pose.model` (the training losses
//! stay in Python).

use crate::algos::geometry::Box4;
use crate::algos::nms::suppress_non_peaks;
use anyhow::{ensure, Result};

/// Default `max_det` of `animalpose.detector.model.decode`.
pub const DEFAULT_MAX_DET: usize = 20;
/// Default `radius` of `animalpose.pose.model.decode_pairs`.
pub const DEFAULT_PAIR_RADIUS: f64 = 2.5;
/// Default `min_peak` of `animalpose.pose.model.decode_pairs`.
pub const DEFAULT_PAIR_MIN_PEAK: f64 = 0.1;
/// Clamp of the log-size regression (`clamp(max=10.0)` in `decode`).
pub const SIZE_CLAMP: f32 = 10.0;

/// Sigmoid, the `torch.sigmoid` of the Python decoding.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Boxes, scores and flat heat-map cells produced by detector peak extraction.
///
/// // differs from Python: boxes are `f64` rather than `float32`, and the batch
/// // dimension is dropped (call once per image).
#[derive(Clone, Debug, Default)]
pub struct DetectorDecode {
    /// Boxes in image pixels, `(N, 4)` xyxy.
    pub boxes: Vec<Box4>,
    /// Sigmoid scores, descending.
    pub scores: Vec<f32>,
    /// Flat heat-map index each detection came from.
    pub indices: Vec<usize>,
}

/// Peak extraction with export-friendly ops: 3x3 max-pool NMS, top-k, size and
/// offset regression.
///
/// Port of `animalpose.detector.model.decode` for one image.
///
/// * `heat_logits`: `(height * width)` single-channel logits.
/// * `size`/`offset`: `(2 * height * width)`, channel-major (`[c][y][x]`).
///
/// // differs from Python: `torch.topk` gives no ordering guarantee for equal
/// // scores, so equal-scoring detections come back in ascending flat index.
/// // A `max_det` larger than the map is clamped instead of raising.
pub fn detector_decode(
    heat_logits: &[f32],
    size: &[f32],
    offset: &[f32],
    width: usize,
    height: usize,
    stride: f64,
    max_det: usize,
) -> DetectorDecode {
    let n = width * height;
    let heat: Vec<f32> = heat_logits.iter().map(|v| sigmoid(*v)).collect();
    let heat = suppress_non_peaks(&heat, width, height);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        heat[*b]
            .partial_cmp(&heat[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let k = max_det.min(n);
    let mut out = DetectorDecode::default();
    for &i in order.iter().take(k) {
        let (y, x) = (i / width, i % width);
        let off = [offset[i], offset[n + i]];
        let wh = [
            f64::from(size[i].min(SIZE_CLAMP).exp()) * stride,
            f64::from(size[n + i].min(SIZE_CLAMP).exp()) * stride,
        ];
        let cx = (x as f64 + f64::from(off[0])) * stride;
        let cy = (y as f64 + f64::from(off[1])) * stride;
        out.boxes.push([
            cx - wh[0] / 2.0,
            cy - wh[1] / 2.0,
            cx + wh[0] / 2.0,
            cy + wh[1] / 2.0,
        ]);
        out.scores.push(heat[i]);
        out.indices.push(i);
    }
    out
}

/// One crop's heat maps (`(K, h, w)` logits) and offset head (`(2K, h, w)`).
#[derive(Clone, Copy, Debug)]
pub struct HeatMaps<'a> {
    heat: &'a [f32],
    offset: &'a [f32],
    num_keypoints: usize,
    width: usize,
    height: usize,
}

impl<'a> HeatMaps<'a> {
    /// Wraps the two output tensors of `animalpose.pose.model.PoseNet.forward`.
    pub fn new(
        heat: &'a [f32],
        offset: &'a [f32],
        num_keypoints: usize,
        width: usize,
        height: usize,
    ) -> Result<Self> {
        let n = width * height;
        ensure!(
            heat.len() == num_keypoints * n,
            "heat maps hold {} values, expected {} x {}",
            heat.len(),
            num_keypoints,
            n
        );
        ensure!(
            offset.len() == 2 * num_keypoints * n,
            "offset maps hold {} values, expected {} x {}",
            offset.len(),
            2 * num_keypoints,
            n
        );
        Ok(Self {
            heat,
            offset,
            num_keypoints,
            width,
            height,
        })
    }

    /// Number of keypoints.
    pub fn num_keypoints(&self) -> usize {
        self.num_keypoints
    }

    /// Flat cell count of one heat map.
    pub fn cells(&self) -> usize {
        self.width * self.height
    }

    /// Row of the flat probability map of keypoint `k` (sigmoid of the logits).
    pub fn probs(&self, k: usize) -> Vec<f32> {
        self.heat[k * self.cells()..(k + 1) * self.cells()]
            .iter()
            .map(|v| sigmoid(*v))
            .collect()
    }

    /// Probability of keypoint `k` at flat cell `idx`.
    pub fn prob(&self, k: usize, idx: usize) -> f32 {
        sigmoid(self.heat[k * self.cells() + idx])
    }

    /// Cell of the largest logit of keypoint `k`; `torch.max` returns the first
    /// maximum, and so does this.
    pub fn peak(&self, k: usize) -> usize {
        let start = k * self.cells();
        let mut best = start;
        for i in start..start + self.cells() {
            if self.heat[i] > self.heat[best] {
                best = i;
            }
        }
        best - start
    }

    /// Sub-pixel offset predicted for keypoint `k` at flat cell `idx`.
    pub fn offset_at(&self, k: usize, idx: usize) -> [f64; 2] {
        let n = self.cells();
        [
            f64::from(self.offset[2 * k * n + idx]),
            f64::from(self.offset[(2 * k + 1) * n + idx]),
        ]
    }

    /// Grid coordinate `(x, y)` of a flat cell.
    pub fn cell_xy(&self, idx: usize) -> (usize, usize) {
        (idx % self.width, idx / self.width)
    }

    /// Squared cell distance between two flat cells.
    pub fn dist2(&self, a: usize, b: usize) -> f64 {
        let (ax, ay) = self.cell_xy(a);
        let (bx, by) = self.cell_xy(b);
        let dx = ax as f64 - bx as f64;
        let dy = ay as f64 - by as f64;
        dx * dx + dy * dy
    }
}

/// Decoded keypoints (crop pixels), confidences and per-pair margins.
///
/// // differs from Python: coordinates and confidences are `f64`, and `margin`
/// // is always returned (Python only with `return_margin=True`); it is empty
/// // when no pairs were decoded.
#[derive(Clone, Debug, Default)]
pub struct PoseDecode {
    /// Keypoints in crop pixels, `(K, 2)`.
    pub keypoints: Vec<[f64; 2]>,
    /// Sigmoid confidence of every peak, `(K,)`.
    pub conf: Vec<f64>,
    /// Left/right evidence of every decoded pair, `(P,)`.
    pub margin: Vec<f64>,
}

/// Keypoints and confidence at the given flat heat-map cells.
///
/// Port of `animalpose.pose.model._decode_at`.
pub fn pose_decode_at(maps: &HeatMaps<'_>, stride: f64, idx: &[usize]) -> PoseDecode {
    let mut out = PoseDecode::default();
    for (k, &i) in idx.iter().enumerate() {
        let (x, y) = maps.cell_xy(i);
        let off = maps.offset_at(k, i);
        out.keypoints
            .push([(x as f64 + off[0]) * stride, (y as f64 + off[1]) * stride]);
        out.conf.push(f64::from(maps.prob(k, i)));
    }
    out
}

/// Argmax plus learned sub-pixel offset decoding.
///
/// Port of `animalpose.pose.model.decode`: keypoints `(K, 2)` in crop pixels and
/// confidence `(K,)`.
pub fn pose_decode(maps: &HeatMaps<'_>, stride: f64) -> PoseDecode {
    let idx: Vec<usize> = (0..maps.num_keypoints()).map(|k| maps.peak(k)).collect();
    pose_decode_at(maps, stride, &idx)
}

/// Argmax decoding that keeps bilateral pairs on distinct peaks.
///
/// Port of `animalpose.pose.model.decode_pairs`: for each `(left, right)` pair
/// whose argmaxes fall within `radius` cells, the two strongest peaks of
/// `max(p_left, p_right)` separated by more than `radius` are found; when the
/// second one reaches `min_peak` the pair is split across them, in the order the
/// two heat maps prefer. `margin` is
/// `p_l(x_l) + p_r(x_r) - p_l(x_r) - p_r(x_l)` per pair.
pub fn pose_decode_pairs(
    maps: &HeatMaps<'_>,
    stride: f64,
    pairs: &[(usize, usize)],
    radius: f64,
    min_peak: f64,
) -> PoseDecode {
    let mut idx: Vec<usize> = (0..maps.num_keypoints()).map(|k| maps.peak(k)).collect();
    if pairs.is_empty() {
        return pose_decode_at(maps, stride, &idx);
    }
    let mut margin = vec![0.0f64; pairs.len()];
    let mut writes: Vec<(usize, usize)> = Vec::with_capacity(2 * pairs.len());
    for (p, &(lft, rgt)) in pairs.iter().enumerate() {
        let (il, ir) = (idx[lft], idx[rgt]);
        let pl: Vec<f32> = maps.probs(lft);
        let pr: Vec<f32> = maps.probs(rgt);
        let comb: Vec<f32> = pl.iter().zip(pr.iter()).map(|(a, b)| a.max(*b)).collect();
        let i1 = first_max(&comb);
        // Second peak: the strongest cell further than `radius` from `i1`
        // (`torch.where(far, comb, 0).max(dim=2)`).
        let mut i2 = 0usize;
        let mut v2 = 0.0f32;
        for (j, &value) in comb.iter().enumerate() {
            if maps.dist2(i1, j) <= radius * radius {
                continue;
            }
            if value > v2 || (value == v2 && j < i2) {
                v2 = value;
                i2 = j;
            }
        }
        let near = maps.dist2(il, ir) <= radius * radius;
        let split = near && f64::from(v2) >= min_peak;
        let keep = f64::from(maps.prob(lft, i1)) + f64::from(maps.prob(rgt, i2))
            >= f64::from(maps.prob(lft, i2)) + f64::from(maps.prob(rgt, i1));
        let (il, ir) = if split {
            if keep {
                (i1, i2)
            } else {
                (i2, i1)
            }
        } else {
            (il, ir)
        };
        writes.push((lft, il));
        writes.push((rgt, ir));
        margin[p] = f64::from(maps.prob(lft, il)) + f64::from(maps.prob(rgt, ir))
            - f64::from(maps.prob(lft, ir))
            - f64::from(maps.prob(rgt, il));
    }
    for (k, i) in writes {
        idx[k] = i;
    }
    let mut out = pose_decode_at(maps, stride, &idx);
    out.margin = margin;
    out
}

/// Index of the largest value, first occurrence wins (as `torch.max` documents).
fn first_max(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(value: f32, len: usize) -> Vec<f32> {
        vec![value; len]
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-5
    }

    fn close_pt(a: [f64; 2], b: [f64; 2]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1])
    }

    fn close_box(a: Box4, b: Box4) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]) && close(a[3], b[3])
    }

    #[test]
    fn detector_decode_reproduces_python_peak_extraction() {
        // The reference tensors of tests run against animalpose.detector.model.decode.
        let (w, h, stride) = (10usize, 8usize, 4.0f64);
        let mut heat = fill(-20.0, w * h);
        heat[3 * w + 6] = 4.0;
        heat[6 * w + 1] = 2.0;
        let mut size = fill(0.0, 2 * w * h);
        size[..w * h].fill(3.0f32.ln());
        size[w * h..].fill(2.0f32.ln());
        let mut offset = fill(0.0, 2 * w * h);
        offset[..w * h].fill(0.25);
        offset[w * h..].fill(-0.5);
        let out = detector_decode(&heat, &size, &offset, w, h, stride, 5);
        assert_eq!(out.indices[0], 36);
        assert_eq!(out.indices[1], 61);
        // box 0: centre ((6+0.25)*4, (3-0.5)*4) = (25, 10), size (12, 8)
        assert!(
            close_box(out.boxes[0], [19.0, 6.0, 31.0, 14.0]),
            "{:?}",
            out.boxes[0]
        );
        assert!(
            close_box(out.boxes[1], [-1.0, 18.0, 11.0, 26.0]),
            "{:?}",
            out.boxes[1]
        );
        assert!((f64::from(out.scores[0]) - 0.9820137620).abs() < 1e-6);
        assert!((f64::from(out.scores[1]) - 0.8807970285).abs() < 1e-6);
        assert!(out.scores[2] < 1e-8);
        assert!(out.scores.windows(2).all(|s| s[0] >= s[1]));
    }

    #[test]
    fn detector_decode_suppresses_a_neighbouring_peak() {
        let (w, h) = (10usize, 8usize);
        let mut heat = fill(-20.0, w * h);
        heat[3 * w + 5] = 4.0;
        heat[3 * w + 6] = 3.0; // inside the 3x3 window of the first peak
        let size = fill(0.0, 2 * w * h);
        let offset = fill(0.0, 2 * w * h);
        let out = detector_decode(&heat, &size, &offset, w, h, 4.0, 2);
        assert_eq!(out.indices[0], 35);
        assert_ne!(out.indices[1], 36);
        assert!(out.scores[1] < 1e-8);
    }

    #[test]
    fn detector_decode_clamps_the_size_regression() {
        let (w, h) = (4usize, 4usize);
        let mut heat = fill(-20.0, w * h);
        heat[0] = 1.0;
        let mut size = fill(50.0, 2 * w * h); // exp(50) would overflow
        size[0] = 50.0;
        size[w * h] = 50.0;
        let offset = fill(0.0, 2 * w * h);
        let out = detector_decode(&heat, &size, &offset, w, h, 1.0, 1);
        assert!(out.boxes[0][2].is_finite());
        assert!(close(out.boxes[0][2], f64::from(10.0f32.exp()) / 2.0));
    }

    #[test]
    fn pose_decode_adds_the_sub_pixel_offset() {
        let (w, h, k, stride) = (10usize, 8usize, 2usize, 4.0f64);
        let mut heat = fill(-20.0, k * w * h);
        heat[2 * w + 5] = 5.0; // keypoint 0 at (x=5, y=2)
        heat[w * h + 4 * w + 7] = 1.0; // keypoint 1 at (x=7, y=4)
        let mut offset = fill(0.0, 2 * k * w * h);
        offset[2 * w * h..3 * w * h].fill(0.1); // x offset of keypoint 1
        offset[3 * w * h..].fill(0.4); // y offset of keypoint 1
        let maps = HeatMaps::new(&heat, &offset, k, w, h).unwrap();
        let out = pose_decode(&maps, stride);
        assert!(
            close_pt(out.keypoints[1], [28.4, 17.6]),
            "{:?}",
            out.keypoints[1]
        );
        assert!((out.conf[1] - 0.7310585976).abs() < 1e-6);
        assert!(out.margin.is_empty());
    }

    #[test]
    fn pose_decode_rejects_mismatched_tensor_lengths() {
        let err = HeatMaps::new(&[0.0; 8], &[0.0; 16], 1, 4, 4).unwrap_err();
        assert!(err.to_string().contains("heat maps"));
        let err = HeatMaps::new(&[0.0; 32], &[0.0; 31], 2, 4, 4).unwrap_err();
        assert!(err.to_string().contains("offset maps"));
    }

    #[test]
    fn pair_splitting_puts_the_two_paws_on_distinct_peaks() {
        // Reference values from animalpose.pose.model.decode_pairs.
        let (w, h, k, stride) = (10usize, 8usize, 2usize, 4.0f64);
        let mut heat = fill(-20.0, k * w * h);
        heat[3 * w + 3] = 6.0; // both maps peak on the same cell
        heat[w * h + 3 * w + 3] = 5.0;
        heat[w * h + 3 * w + 7] = 2.0; // the far paw, 4 cells away
        heat[3 * w + 7] = 1.0;
        let mut offset = fill(0.0, 2 * k * w * h);
        offset[..w * h].fill(0.3);
        offset[w * h..2 * w * h].fill(-0.2);
        offset[2 * w * h..3 * w * h].fill(0.1);
        offset[3 * w * h..].fill(0.4);
        let maps = HeatMaps::new(&heat, &offset, k, w, h).unwrap();
        let out = pose_decode_pairs(
            &maps,
            stride,
            &[(0, 1)],
            DEFAULT_PAIR_RADIUS,
            DEFAULT_PAIR_MIN_PEAK,
        );
        assert!(
            close_pt(out.keypoints[0], [13.2, 11.2]),
            "{:?}",
            out.keypoints[0]
        );
        assert!(
            close_pt(out.keypoints[1], [28.4, 13.6]),
            "{:?}",
            out.keypoints[1]
        );
        assert!(
            (out.margin[0] - 0.1539587379).abs() < 1e-5,
            "{}",
            out.margin[0]
        );
        // Without pair decoding both heat maps peak on the very same cell.
        let plain = pose_decode(&maps, stride);
        assert!(close_pt(plain.keypoints[0], [13.2, 11.2]));
        assert_eq!(maps.peak(0), maps.peak(1));
    }

    #[test]
    fn a_weak_second_peak_leaves_the_pair_collapsed() {
        let (w, h, k, stride) = (10usize, 8usize, 2usize, 4.0f64);
        let mut heat = fill(-20.0, k * w * h);
        heat[3 * w + 3] = 6.0;
        heat[w * h + 3 * w + 3] = 5.0;
        heat[w * h + 3 * w + 7] = -3.0; // below min_peak
        heat[3 * w + 7] = -3.0;
        let offset = fill(0.0, 2 * k * w * h);
        let maps = HeatMaps::new(&heat, &offset, k, w, h).unwrap();
        let out = pose_decode_pairs(
            &maps,
            stride,
            &[(0, 1)],
            DEFAULT_PAIR_RADIUS,
            DEFAULT_PAIR_MIN_PEAK,
        );
        assert!(close_pt(out.keypoints[0], [12.0, 12.0]));
        assert!(close_pt(out.keypoints[1], [12.0, 12.0]));
        assert_eq!(out.margin[0], 0.0);
    }

    #[test]
    fn distant_argmaxes_are_left_alone_and_empty_pairs_decode_plain() {
        let (w, h, k, stride) = (10usize, 8usize, 2usize, 1.0f64);
        let mut heat = fill(-20.0, k * w * h);
        heat[0] = 3.0;
        heat[w * h + 5 * w + 5] = 3.0;
        let offset = fill(0.0, 2 * k * w * h);
        let maps = HeatMaps::new(&heat, &offset, k, w, h).unwrap();
        let out = pose_decode_pairs(
            &maps,
            stride,
            &[(0, 1)],
            DEFAULT_PAIR_RADIUS,
            DEFAULT_PAIR_MIN_PEAK,
        );
        assert_eq!(out.keypoints[0], [0.0, 0.0]);
        assert_eq!(out.keypoints[1], [5.0, 5.0]);
        let expected = 2.0 / (1.0 + (-3.0f64).exp());
        assert!(
            (out.margin[0] - expected).abs() < 1e-6,
            "{} vs {expected}",
            out.margin[0]
        );
        let plain = pose_decode_pairs(
            &maps,
            stride,
            &[],
            DEFAULT_PAIR_RADIUS,
            DEFAULT_PAIR_MIN_PEAK,
        );
        assert_eq!(plain.keypoints, out.keypoints);
        assert!(plain.margin.is_empty());
    }
}
