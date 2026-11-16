//! Non-maximum suppression, in both shapes the pipeline uses: the 3x3 max-pool
//! peak comparison of the detector head, and the greedy box suppression that
//! merges network and blob candidates.

use crate::algos::geometry::{box_iou, Box4};

/// Default `iou_thresh` of `animalpose.pipeline.merge_candidates`.
pub const DEFAULT_MERGE_IOU: f64 = 0.5;
/// Default `max_det` of the detector and of `merge_candidates`.
pub const DEFAULT_MAX_DET: usize = 20;

/// `true` for every heat-map cell that equals the maximum of its 3x3
/// neighbourhood (out-of-image cells count as minus infinity).
///
/// Port of the `peaks = F.max_pool2d(heat, 3, stride=1, padding=1)` /
/// `heat * (peaks == heat)` test in `animalpose.detector.model.decode`.
pub fn peak_mask(heat: &[f32], width: usize, height: usize) -> Vec<bool> {
    let mut out = vec![false; heat.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let v = heat[i];
            let mut is_peak = true;
            'neighbours: for dy in 0..3usize {
                let sy = y as isize + dy as isize - 1;
                if sy < 0 || sy >= height as isize {
                    continue;
                }
                for dx in 0..3usize {
                    let sx = x as isize + dx as isize - 1;
                    if sx < 0 || sx >= width as isize {
                        continue;
                    }
                    let n = heat[sy as usize * width + sx as usize];
                    // Strictly greater suppresses; an equal neighbour does not,
                    // exactly like `peaks == heat` on identical floats.
                    if n > v {
                        is_peak = false;
                        break 'neighbours;
                    }
                }
            }
            out[i] = is_peak;
        }
    }
    out
}

/// Heat map with every non-peak cell zeroed
/// (`heat * (peaks == heat)` in `animalpose.detector.model.decode`).
pub fn suppress_non_peaks(heat: &[f32], width: usize, height: usize) -> Vec<f32> {
    peak_mask(heat, width, height)
        .iter()
        .zip(heat.iter())
        .map(|(keep, v)| if *keep { *v } else { 0.0 })
        .collect()
}

/// Greedy non-maximum suppression: pick the best-scoring box, drop every later
/// box whose IoU with an already kept box reaches `iou_thresh`, stop at
/// `max_det`, and return the kept indices in score order.
///
/// Port of the keep-loop of `animalpose.pipeline.merge_candidates`.
///
/// // differs from Python: `np.argsort(-scores)` sorts with an unstable
/// // quicksort, so boxes with identical scores come back in arbitrary order;
/// // ties here keep the lower input index.
pub fn nms(boxes: &[Box4], scores: &[f64], iou_thresh: f64, max_det: usize) -> Vec<usize> {
    let n = boxes.len().min(scores.len());
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let mut keep: Vec<usize> = Vec::new();
    for i in order {
        if keep
            .iter()
            .any(|&j| box_iou(boxes[i], boxes[j]) >= iou_thresh)
        {
            continue;
        }
        keep.push(i);
        if keep.len() >= max_det {
            break;
        }
    }
    keep
}

/// Union of two sets of boxes, best score first; a box overlapping a better one
/// by IoU at or above `iou_thresh` is dropped, and at most `max_det` survive.
///
/// Port of `animalpose.pipeline.merge_candidates`.
pub fn merge_candidates(
    boxes: &[Box4],
    scores: &[f64],
    extra_boxes: &[Box4],
    extra_scores: &[f64],
    iou_thresh: f64,
    max_det: usize,
) -> (Vec<Box4>, Vec<f64>) {
    let mut all_boxes: Vec<Box4> = boxes.to_vec();
    all_boxes.extend_from_slice(extra_boxes);
    let mut all_scores: Vec<f64> = scores.to_vec();
    all_scores.extend_from_slice(extra_scores);
    let keep = nms(&all_boxes, &all_scores, iou_thresh, max_det);
    let out_boxes = keep.iter().map(|&i| all_boxes[i]).collect();
    let out_scores = keep.iter().map(|&i| all_scores[i]).collect();
    (out_boxes, out_scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_pool_nms_keeps_only_local_maxima() {
        // Two peaks five cells apart, plus a plateau that stays a single peak.
        let w = 10;
        let h = 3;
        let mut heat = vec![0.1f32; w * h];
        heat[w + 2] = 0.9;
        heat[w + 8] = 0.7;
        let kept = suppress_non_peaks(&heat, w, h);
        assert_eq!(kept[w + 2], 0.9);
        assert_eq!(kept[w + 8], 0.7);
        // Neighbours of a peak are suppressed; the flat floor is its own peak.
        assert_eq!(kept[w + 1], 0.0);
        assert_eq!(kept[0], 0.1);
        // 30 cells minus the 8 neighbours of each of the two peaks.
        assert_eq!(peak_mask(&heat, w, h).iter().filter(|p| **p).count(), 14);
    }

    #[test]
    fn a_plateau_counts_as_peaks_like_the_equality_test() {
        let heat = vec![0.5f32; 9];
        assert!(peak_mask(&heat, 3, 3).iter().all(|p| *p));
    }

    #[test]
    fn greedy_nms_orders_by_score_and_drops_overlaps() {
        let boxes = [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [100.0, 100.0, 110.0, 110.0],
        ];
        let scores = [0.5, 0.9, 0.7];
        let keep = nms(&boxes, &scores, 0.5, 20);
        assert_eq!(keep, vec![1, 2]);
        // A lower threshold lets the near-duplicate survive.
        assert_eq!(nms(&boxes, &scores, 0.99, 20), vec![1, 2, 0]);
        // max_det truncates the list.
        assert_eq!(nms(&boxes, &scores, 0.5, 1), vec![1]);
    }

    #[test]
    fn merge_candidates_prefers_the_blob_box() {
        // Port of the merge assertions in tests/test_degrade_frames.py.
        let blob = [[18.5, 38.5, 31.5, 49.5]];
        let net = [blob[0], [10.0, 10.0, 20.0, 20.0]];
        let (merged, scores) = merge_candidates(&net, &[0.6, 0.4], &blob, &[0.9], 0.5, 20);
        assert_eq!(merged.len(), 2);
        assert!((scores[0] - 0.9).abs() < 1e-6);
        assert_eq!(merged[0], [18.5, 38.5, 31.5, 49.5]);
        assert_eq!(merged[1], [10.0, 10.0, 20.0, 20.0]);
        assert_eq!(scores[1], 0.4);
    }

    #[test]
    fn merge_candidates_honours_max_det_and_empty_inputs() {
        let (merged, scores) = merge_candidates(&[], &[], &[], &[], 0.5, 20);
        assert!(merged.is_empty() && scores.is_empty());
        let boxes = [
            [0.0, 0.0, 1.0, 1.0],
            [10.0, 10.0, 11.0, 11.0],
            [20.0, 20.0, 21.0, 21.0],
        ];
        let scores = [0.3, 0.9, 0.6];
        let (merged, merged_scores) = merge_candidates(&boxes, &scores, &[], &[], 0.5, 2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged_scores, vec![0.9, 0.6]);
    }
}
