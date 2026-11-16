//! Evaluation metrics: detection AP, keypoint PCK/RMSE, tracking MOTA/IDF1/ID switches, the
//! gap-filling baseline, behavior F1 and AUROC — the port of `src/animalpose/eval/metrics.py`.
//!
//! Arrays arrive the way numpy holds them, row-major: keypoints `(N, K, 2)` as `[[f64; 2]]` slices
//! of `N * K` points, masks `(N, K)` as `N * K` booleans.

use crate::algos::geometry::{box_iou_pairwise, Box4};
use crate::algos::hungarian::linear_sum_assignment;
use std::collections::BTreeMap;

/// `average_precision(pred_boxes, pred_scores, gt_boxes, iou_thresh)`: VOC-style (all-point
/// interpolated) AP over a list of images; NaN when there is no ground truth at all.
///
/// `// differs from Python:` predictions of one image are visited in a *stable* descending score
/// order, where `np.argsort(-scores)` (quicksort) may order equal scores either way; only exact
/// score ties on boxes that compete for the same ground truth can tell the two apart.
pub fn average_precision(
    pred_boxes: &[Vec<Box4>],
    pred_scores: &[Vec<f64>],
    gt_boxes: &[Vec<Box4>],
    iou_thresh: f64,
) -> f64 {
    let mut records: Vec<(f64, bool)> = Vec::new();
    let mut n_gt = 0usize;
    for ((pb, ps), gb) in pred_boxes.iter().zip(pred_scores).zip(gt_boxes) {
        n_gt += gb.len();
        let mut used = vec![false; gb.len()];
        let mut order: Vec<usize> = (0..ps.len()).collect();
        order.sort_by(|a, b| ps[*b].total_cmp(&ps[*a]));
        let ious = box_iou_pairwise(pb, gb);
        for i in order {
            let hit = if gb.is_empty() {
                false
            } else {
                let row = &ious[i * gb.len()..(i + 1) * gb.len()];
                let j = argmax(row);
                if row[j] >= iou_thresh && !used[j] {
                    used[j] = true;
                    true
                } else {
                    false
                }
            };
            records.push((ps[i], hit));
        }
    }
    if n_gt == 0 {
        return f64::NAN;
    }
    // `records.sort(key=lambda r: -r[0])` is stable.
    records.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut mrec = vec![0.0];
    let mut mpre = vec![0.0];
    for (_, hit) in &records {
        if *hit {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        mrec.push(tp / n_gt as f64);
        mpre.push(tp / f64::max(tp + fp, 1e-9));
    }
    mrec.push(1.0);
    mpre.push(0.0);
    for i in (0..mpre.len() - 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }
    let mut ap = 0.0;
    for i in 0..mrec.len() - 1 {
        if mrec[i + 1] != mrec[i] {
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
        }
    }
    ap
}

/// `np.argmax`: the first index of the largest value.
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// `pck(pred, gt, mask, norm, alpha)`: the fraction of masked keypoints within `alpha * norm` of
/// the ground truth (`norm` per instance, e.g. the box side); NaN when nothing is masked.
///
/// `pred` / `gt` hold `norm.len() * k` points, `mask` as many flags.
pub fn pck(pred: &[[f64; 2]], gt: &[[f64; 2]], mask: &[bool], norm: &[f64], alpha: f64) -> f64 {
    let k = if norm.is_empty() { 0 } else { pred.len() / norm.len() };
    let mut hits = 0usize;
    let mut total = 0usize;
    for (i, ((p, g), m)) in pred.iter().zip(gt).zip(mask).enumerate() {
        if !*m {
            continue;
        }
        total += 1;
        let d = (p[0] - g[0]).hypot(p[1] - g[1]);
        if d <= alpha * norm[i / k.max(1)] {
            hits += 1;
        }
    }
    if total == 0 {
        f64::NAN
    } else {
        hits as f64 / total as f64
    }
}

/// `rmse(pred, gt, mask)`: root mean squared point error over masked keypoints whose prediction
/// and ground truth are both finite; NaN when none are.
pub fn rmse(pred: &[[f64; 2]], gt: &[[f64; 2]], mask: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for ((p, g), m) in pred.iter().zip(gt).zip(mask) {
        let finite = p.iter().chain(g.iter()).all(|v| v.is_finite());
        if *m && finite {
            sum += (p[0] - g[0]).powi(2) + (p[1] - g[1]).powi(2);
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        (sum / n as f64).sqrt()
    }
}

/// What [`tracking_metrics`] returns, the keys of the Python dict.
#[derive(Clone, Debug, PartialEq)]
pub struct TrackingMetrics {
    /// `1 - (fn + fp + id_switches) / n_gt`.
    pub mota: f64,
    /// `2 * IDTP / (n_gt + n_pred)` under the best global identity mapping.
    pub idf1: f64,
    /// Ground-truth identities whose matched prediction changed.
    pub id_switches: usize,
    /// Missed ground-truth boxes.
    pub fn_: usize,
    /// Unmatched predicted boxes.
    pub fp: usize,
    /// Ground-truth boxes.
    pub n_gt: usize,
    /// Distinct predicted identities.
    pub n_pred_ids: usize,
    /// Distinct ground-truth identities.
    pub n_gt_ids: usize,
}

impl TrackingMetrics {
    /// The Python dict, in its key order.
    pub fn to_py(&self) -> crate::data::pyjson::PyVal {
        crate::data::pyjson::PyVal::dict()
            .with("mota", self.mota)
            .with("idf1", self.idf1)
            .with("id_switches", self.id_switches)
            .with("fn", self.fn_)
            .with("fp", self.fp)
            .with("n_gt", self.n_gt)
            .with("n_pred_ids", self.n_pred_ids)
            .with("n_gt_ids", self.n_gt_ids)
    }
}

/// `tracking_metrics(gt_ids, gt_boxes, pred_ids, pred_boxes, iou_thresh)`: CLEAR-MOT MOTA and ID
/// switches, and IDF1, from per-frame boxes and identities.
///
/// `// differs from Python:` `scipy.optimize.linear_sum_assignment` and
/// [`crate::algos::hungarian::linear_sum_assignment`] may pick different but equally optimal
/// matchings when the IoU matrix has ties.
pub fn tracking_metrics(
    gt_ids: &[Vec<i64>],
    gt_boxes: &[Vec<Box4>],
    pred_ids: &[Vec<i64>],
    pred_boxes: &[Vec<Box4>],
    iou_thresh: f64,
) -> TrackingMetrics {
    let mut matches_per_frame: Vec<Vec<(i64, i64)>> = Vec::new();
    let (mut n_gt, mut n_pred, mut fn_, mut fp, mut idsw) = (0, 0, 0, 0, 0);
    let mut last: BTreeMap<i64, i64> = BTreeMap::new();
    for (((gi, gb), pi), pb) in gt_ids.iter().zip(gt_boxes).zip(pred_ids).zip(pred_boxes) {
        n_gt += gb.len();
        n_pred += pb.len();
        let mut pairs = Vec::new();
        if !gb.is_empty() && !pb.is_empty() {
            let iou = box_iou_pairwise(gb, pb);
            let cost: Vec<Vec<f64>> = (0..gb.len())
                .map(|a| (0..pb.len()).map(|b| -iou[a * pb.len() + b]).collect())
                .collect();
            for (a, b) in linear_sum_assignment(&cost).unwrap_or_default() {
                if iou[a * pb.len() + b] >= iou_thresh {
                    pairs.push((gi[a], pi[b]));
                }
            }
        }
        fn_ += gb.len() - pairs.len();
        fp += pb.len() - pairs.len();
        for (g, p) in &pairs {
            if last.get(g).is_some_and(|q| q != p) {
                idsw += 1;
            }
            last.insert(*g, *p);
        }
        matches_per_frame.push(pairs);
    }
    let g_list: Vec<i64> = sorted_unique(gt_ids);
    let p_list: Vec<i64> = sorted_unique(pred_ids);
    let mut co = vec![vec![0.0; p_list.len()]; g_list.len()];
    for pairs in &matches_per_frame {
        for (g, p) in pairs {
            let gi = g_list.binary_search(g).unwrap_or(0);
            let pi = p_list.binary_search(p).unwrap_or(0);
            co[gi][pi] += 1.0;
        }
    }
    let idtp: f64 = if g_list.is_empty() || p_list.is_empty() {
        0.0
    } else {
        let neg: Vec<Vec<f64>> = co.iter().map(|r| r.iter().map(|v| -v).collect()).collect();
        linear_sum_assignment(&neg)
            .unwrap_or_default()
            .iter()
            .map(|(a, b)| co[*a][*b])
            .sum()
    };
    TrackingMetrics {
        mota: 1.0 - (fn_ + fp + idsw) as f64 / n_gt.max(1) as f64,
        idf1: 2.0 * idtp / (n_gt + n_pred).max(1) as f64,
        id_switches: idsw,
        fn_,
        fp,
        n_gt,
        n_pred_ids: p_list.len(),
        n_gt_ids: g_list.len(),
    }
}

fn sorted_unique(ids: &[Vec<i64>]) -> Vec<i64> {
    let mut all: Vec<i64> = ids.iter().flatten().copied().collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// What [`f1_scores`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct F1Scores {
    /// Mean F1 over the classes present in the ground truth (NaN when none is).
    pub macro_f1: f64,
    /// Share of correct predictions.
    pub accuracy: f64,
    /// F1 of every class.
    pub per_class_f1: Vec<f64>,
    /// `confusion[gt][pred]` counts.
    pub confusion: Vec<Vec<i64>>,
}

/// `f1_scores(pred, gt, n_classes)`: macro-F1 over the classes present in `gt`, accuracy,
/// per-class F1 and the confusion matrix (rows ground truth, columns prediction).
pub fn f1_scores(pred: &[usize], gt: &[usize], n_classes: usize) -> F1Scores {
    let mut cm = vec![vec![0i64; n_classes]; n_classes];
    for (p, g) in pred.iter().zip(gt) {
        cm[*g][*p] += 1;
    }
    let col: Vec<i64> = (0..n_classes).map(|j| cm.iter().map(|r| r[j]).sum()).collect();
    let row: Vec<i64> = cm.iter().map(|r| r.iter().sum()).collect();
    let tp: Vec<f64> = (0..n_classes).map(|i| cm[i][i] as f64).collect();
    let f1: Vec<f64> = (0..n_classes)
        .map(|i| {
            let prec = tp[i] / col[i].max(1) as f64;
            let rec = tp[i] / row[i].max(1) as f64;
            2.0 * prec * rec / f64::max(prec + rec, 1e-9)
        })
        .collect();
    let present: Vec<f64> = (0..n_classes).filter(|i| row[*i] > 0).map(|i| f1[i]).collect();
    let total: i64 = row.iter().sum();
    F1Scores {
        macro_f1: if present.is_empty() {
            f64::NAN
        } else {
            mean(&present)
        },
        accuracy: tp.iter().sum::<f64>() / total.max(1) as f64,
        per_class_f1: f1,
        confusion: cm,
    }
}

/// `np.mean` with numpy's pairwise summation (exact for the short arrays scored here).
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(v) / v.len() as f64
}

/// numpy's pairwise summation (`pairwise_sum` in `umath`), blocks of 8 below 128 elements.
pub fn pairwise_sum(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 8 {
        return v.iter().fold(0.0, |a, b| a + b);
    }
    if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&v[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                r[j] += v[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += v[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&v[..n2]) + pairwise_sum(&v[n2..])
}

/// `linear_interpolate(x, observed)`: per-keypoint linear interpolation (`np.interp`, so the ends
/// hold the nearest observation); a keypoint never observed is NaN throughout.
///
/// `x` is `(T, K, D)` row-major, `observed` `(T, K)`.
pub fn linear_interpolate(x: &[f64], observed: &[bool], t: usize, k: usize, d: usize) -> Vec<f64> {
    let mut out = x.to_vec();
    for kk in 0..k {
        let seen: Vec<usize> = (0..t).filter(|&tt| observed[tt * k + kk]).collect();
        for dd in 0..d {
            let at = |tt: usize| (tt * k + kk) * d + dd;
            if seen.is_empty() {
                for tt in 0..t {
                    out[at(tt)] = f64::NAN;
                }
                continue;
            }
            let xp: Vec<f64> = seen.iter().map(|v| *v as f64).collect();
            let fp: Vec<f64> = seen.iter().map(|tt| x[at(*tt)]).collect();
            for tt in 0..t {
                out[at(tt)] = interp(tt as f64, &xp, &fp);
            }
        }
    }
    out
}

/// `np.interp(x, xp, fp)` for increasing `xp`.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|v| *v <= x) - 1;
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `scipy.stats.rankdata` (method `average`): 1-based ranks, ties share their mean rank.
pub fn rankdata(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|a, b| v[*a].total_cmp(&v[*b]));
    let mut ranks = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for o in &order[i..=j] {
            ranks[*o] = r;
        }
        i = j + 1;
    }
    ranks
}

/// `auroc(scores_pos, scores_neg)`: the probability that a random positive outranks a random
/// negative (Mann-Whitney U, ties counting one half); NaN when either side is empty.
pub fn auroc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let all: Vec<f64> = pos.iter().chain(neg).copied().collect();
    let ranks = rankdata(&all);
    let (np_, nn) = (pos.len() as f64, neg.len() as f64);
    let r_pos: f64 = ranks[..pos.len()].iter().sum();
    (r_pos - np_ * (np_ + 1.0) / 2.0) / (np_ * nn)
}

#[cfg(test)]
mod tests {
    //! `tests/test_eval_metrics.py`, assertion for assertion.
    use super::*;

    const BOX: Box4 = [0.0, 0.0, 10.0, 10.0];

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn average_precision_is_one_when_the_box_matches_and_zero_below_the_threshold() {
        let gt = vec![vec![BOX]];
        assert!(close(average_precision(&[vec![BOX]], &[vec![0.9]], &gt, 0.5), 1.0));
        let shifted = [4.0, 0.0, 14.0, 10.0]; // IoU 60/140 = 0.43
        assert!(close(average_precision(&[vec![shifted]], &[vec![0.9]], &gt, 0.5), 0.0));
        assert!(close(average_precision(&[vec![shifted]], &[vec![0.9]], &gt, 0.4), 1.0));
    }

    #[test]
    fn average_precision_is_the_area_under_the_interpolated_precision_recall_curve() {
        let gt = vec![vec![BOX, [50.0, 50.0, 60.0, 60.0]]];
        let preds = vec![vec![BOX, [20.0, 20.0, 24.0, 24.0], [50.0, 50.0, 60.0, 60.0]]];
        assert!(close(average_precision(&preds, &[vec![0.9, 0.7, 0.5]], &gt, 0.5), 5.0 / 6.0));
        assert!(close(average_precision(&[vec![BOX, BOX]], &[vec![0.9, 0.8]], &gt, 0.5), 0.5));
    }

    #[test]
    fn extra_detections_never_add_recall_and_no_ground_truth_is_undefined() {
        let gt = vec![vec![BOX], vec![]];
        let (perfect, spurious) = (vec![BOX], vec![[80.0, 80.0, 90.0, 90.0]]);
        let both = [perfect.clone(), spurious.clone()];
        assert!(close(average_precision(&both, &[vec![0.99], vec![0.5]], &gt, 0.5), 1.0));
        assert!(close(average_precision(&both, &[vec![0.5], vec![0.99]], &gt, 0.5), 0.5));
        let junk = [spurious.clone(), spurious];
        assert!(close(average_precision(&junk, &[vec![0.9], vec![0.8]], &gt, 0.5), 0.0));
        assert!(average_precision(&[vec![]], &[vec![]], &[vec![]], 0.5).is_nan());
    }

    #[test]
    fn pck_counts_keypoints_within_alpha_times_the_normalising_distance() {
        let gt = [[0.0, 0.0]; 4];
        let pred = [[6.0, 8.0], [0.0, 11.0], [0.0, 1.0], [0.0, 500.0]];
        let mask = [true, true, true, false];
        assert!(close(pck(&pred, &gt, &mask, &[100.0], 0.1), 2.0 / 3.0));
        assert!(close(pck(&pred, &gt, &mask, &[100.0], 0.2), 1.0));
        assert!(close(pck(&gt, &gt, &mask, &[100.0], 0.1), 1.0));
        assert!(pck(&pred, &gt, &[false; 4], &[100.0], 0.1).is_nan());
    }

    #[test]
    fn rmse_averages_squared_error_and_skips_unusable_keypoints() {
        let gt = [[0.0, 0.0]; 3];
        let pred = [[3.0, 0.0], [0.0, 4.0], [f64::NAN, 1.0]];
        assert!(close(rmse(&pred, &gt, &[true; 3]), 12.5f64.sqrt()));
        assert!(close(rmse(&pred, &gt, &[true, true, false]), 12.5f64.sqrt()));
        assert!(close(rmse(&pred, &gt, &[false, true, false]), 4.0));
        assert!(rmse(&pred, &gt, &[false; 3]).is_nan());
    }

    #[test]
    fn tracking_metrics_penalise_an_identity_switch_and_score_idf1_globally() {
        let gt_ids = vec![vec![0], vec![0], vec![0]];
        let boxes = vec![vec![BOX]; 3];
        let m = tracking_metrics(&gt_ids, &boxes, &[vec![1], vec![1], vec![2]], &boxes, 0.5);
        assert_eq!((m.id_switches, m.fn_, m.fp, m.n_gt), (1, 0, 0, 3));
        assert!(close(m.mota, 1.0 - 1.0 / 3.0));
        assert!(close(m.idf1, 4.0 / 6.0));
        assert_eq!((m.n_gt_ids, m.n_pred_ids), (1, 2));
        let stable = tracking_metrics(&gt_ids, &boxes, &[vec![1], vec![1], vec![1]], &boxes, 0.5);
        assert!(close(stable.mota, 1.0) && close(stable.idf1, 1.0));
        assert_eq!(stable.id_switches, 0);
    }

    #[test]
    fn tracking_metrics_count_misses_and_spurious_boxes() {
        let (a, b, junk) = (BOX, [50.0, 50.0, 60.0, 60.0], [90.0, 90.0, 95.0, 95.0]);
        let m = tracking_metrics(&[vec![0, 1]], &[vec![a, b]], &[vec![1, 2]], &[vec![a, junk]], 0.5);
        assert_eq!((m.fn_, m.fp, m.n_pred_ids), (1, 1, 2));
        assert!(close(m.mota, 0.0) && close(m.idf1, 0.5));
        let empty = tracking_metrics(&[vec![0]], &[vec![a]], &[vec![]], &[vec![]], 0.5);
        assert!(close(empty.mota, 0.0) && close(empty.idf1, 0.0));
        assert_eq!((empty.fn_, empty.fp, empty.n_pred_ids), (1, 0, 0));
    }

    #[test]
    fn f1_scores_averages_only_over_classes_present_in_the_ground_truth() {
        assert!(close(f1_scores(&[0, 0], &[0, 0], 3).macro_f1, 1.0));
        let r = f1_scores(&[0, 1, 1, 1], &[0, 0, 1, 1], 2);
        assert_eq!(r.confusion, vec![vec![1, 1], vec![0, 2]]);
        assert!(close(r.accuracy, 0.75));
        assert!(close(r.per_class_f1[0], 2.0 / 3.0) && close(r.per_class_f1[1], 0.8));
        assert!(close(r.macro_f1, (2.0 / 3.0 + 0.8) / 2.0));
        assert!(close(f1_scores(&[1], &[0], 2).macro_f1, 0.0));
    }

    #[test]
    fn linear_interpolate_fills_gaps_straight_line_and_holds_the_ends() {
        let (t, k) = (5, 2);
        let mut x = vec![0.0; t * k];
        for (tt, (a, b)) in [(0.0, 7.0), (5.0, 1.0), (10.0, 2.0), (30.0, 30.0), (40.0, 40.0)]
            .iter()
            .enumerate()
        {
            x[tt * k] = *a;
            x[tt * k + 1] = *b;
        }
        let observed = [
            true, false, false, true, false, true, true, true, true, true,
        ];
        let out = linear_interpolate(&x, &observed, t, k, 1);
        let col = |c: usize| -> Vec<f64> { (0..t).map(|tt| out[tt * k + c]).collect() };
        assert_eq!(col(0), [0.0, 10.0, 20.0, 30.0, 40.0]);
        assert_eq!(col(1), [1.0, 1.0, 2.0, 30.0, 40.0]);
        assert!(linear_interpolate(&x, &[false; 10], t, k, 1)
            .iter()
            .all(|v| v.is_nan()));
        assert!(linear_interpolate(&x, &[true; 10], t, k, 1)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn auroc_is_the_probability_a_positive_outranks_a_negative() {
        assert!(close(auroc(&[1.0, 2.0, 3.0], &[0.5, 0.9]), 1.0));
        assert!(close(auroc(&[0.5, 0.9], &[1.0, 2.0, 3.0]), 0.0));
        assert!(close(auroc(&[1.0, 1.0], &[1.0, 1.0]), 0.5));
        assert!(close(auroc(&[1.0, 2.0], &[2.0, 3.0]), 0.125));
        assert!(close(auroc(&[3.0, 2.0], &[1.0, 4.0]), 0.5));
        assert!(auroc(&[], &[1.0]).is_nan() && auroc(&[1.0], &[]).is_nan());
    }

    #[test]
    fn pinned_against_a_python_run() {
        // Three images (one without predictions), and AUROC with ties across the two sides.
        let preds = vec![
            vec![[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [30.0, 30.0, 40.0, 42.0]],
            vec![[5.0, 5.0, 15.0, 17.0]],
            vec![],
        ];
        let scores = vec![vec![0.8, 0.6, 0.55], vec![0.7], vec![]];
        let gts = vec![
            vec![[0.0, 0.0, 10.0, 10.0], [31.0, 30.0, 41.0, 40.0]],
            vec![[4.0, 6.0, 14.0, 16.0]],
            vec![[0.0, 0.0, 5.0, 5.0]],
        ];
        assert!(close(average_precision(&preds, &scores, &gts, 0.5), PY_AP50));
        assert!(close(average_precision(&preds, &scores, &gts, 0.75), PY_AP75));
        let pos = [0.9, 0.3, 0.75, 0.75, 0.2];
        let neg = [0.1, 0.75, 0.4, 0.3];
        assert!(close(auroc(&pos, &neg), PY_AUROC));
    }

    // `average_precision(..., 0.5)`, `(..., 0.75)` and `auroc(pos, neg)` printed by the Python
    // package on the same inputs.
    const PY_AP50: f64 = 0.6875;
    const PY_AP75: f64 = 0.25;
    const PY_AUROC: f64 = 0.675;
}
