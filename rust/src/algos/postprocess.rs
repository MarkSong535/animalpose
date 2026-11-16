//! Post-processing of classifier probabilities: bout smoothing and prior
//! adaptation.
//!
//! Port of `animalpose.behavior.postprocess`. Probabilities are `(T, C)`
//! (or `(B, T, C)`, which the caller flattens: the routines below only ever see
//! the last axis, exactly like the Python versions that reshape to `(-1, C)`).

use anyhow::{bail, Result};

/// Default `win` of `median_smooth`.
pub const DEFAULT_SMOOTH_WIN: i64 = 5;
/// Default `iters` of `adapt_prior`.
pub const DEFAULT_ADAPT_ITERS: i64 = 25;
/// Default `max_ratio` of `adapt_prior`.
pub const DEFAULT_MAX_RATIO: f64 = 4.0;
/// Default `tol` of `adapt_prior`.
pub const DEFAULT_ADAPT_TOL: f64 = 1e-4;
/// Default search interval and iteration count of `fit_temperature`.
pub const DEFAULT_TEMPERATURE_LO: f64 = 0.2;
/// Default upper bound of the temperature search.
pub const DEFAULT_TEMPERATURE_HI: f64 = 10.0;
/// Default iteration count of the golden-section search.
pub const DEFAULT_TEMPERATURE_ITERS: usize = 40;

/// Per-class median over a window of `win` frames, edges replicated.
///
/// Port of `animalpose.behavior.postprocess.moving_median` for a `(T, C)` array
/// stored row-major. Windows are made odd so the centre frame is the current
/// one; `win <= 1` or fewer than three frames return a copy.
pub fn moving_median(p: &[f64], frames: usize, classes: usize, win: i64) -> Vec<f64> {
    if win <= 1 || frames < 3 || classes == 0 {
        return p.to_vec();
    }
    let win = win as usize + 1 - win as usize % 2; // odd, centre = current frame
    let pad = win / 2;
    let mut out = vec![0.0; frames * classes];
    let mut window = vec![0.0; win];
    for t in 0..frames {
        // `np.pad(..., mode="edge")` replicates the first and last frame.
        let at = |row: usize| -> usize {
            let r = row as i64 - pad as i64;
            r.clamp(0, frames as i64 - 1) as usize
        };
        for c in 0..classes {
            for (i, v) in window.iter_mut().enumerate() {
                *v = p[at(t + i) * classes + c];
            }
            window.sort_by(f64::total_cmp);
            out[t * classes + c] = if win % 2 == 1 {
                window[win / 2]
            } else {
                (window[win / 2 - 1] + window[win / 2]) / 2.0
            };
        }
    }
    out
}

/// Bout-level smoothing of `(T, C)` probabilities with a per-class moving
/// median, then renormalised.
///
/// Port of `animalpose.behavior.postprocess.median_smooth` (`win = 5`,
/// `renormalize = true` by default; a batch `(B, T, C)` is smoothed batch by
/// batch by the caller).
pub fn median_smooth(
    p: &[f64],
    frames: usize,
    classes: usize,
    win: i64,
    renormalize: bool,
) -> Vec<f64> {
    let mut out = moving_median(p, frames, classes, win);
    if renormalize && win > 1 {
        normalize_rows(&mut out, frames, classes, 1e-9);
    }
    out
}

/// Prior-shift of `(T, C)` probabilities; returns `(adjusted, prior)`.
///
/// Port of `animalpose.behavior.postprocess.adapt_prior`: the Saerens-Latinne-
/// Decaestecker expectation step. `prior_source` is the class distribution the
/// model was trained on; when `None` it is the model's own average prediction
/// over these frames. The ratio a class prior may move is capped by
/// `max_ratio`, which stops the EM from switching a class off entirely.
pub fn adapt_prior(
    probs: &[f64],
    n: usize,
    classes: usize,
    prior_source: Option<&[f64]>,
    iters: i64,
    max_ratio: f64,
    tol: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut flat = probs.to_vec();
    normalize_rows(&mut flat, n, classes, 1e-12);
    if classes < 2 {
        let prior = match prior_source {
            Some(src) => src.to_vec(),
            None if n > 0 => column_mean(&flat, n, classes),
            None => vec![1.0; classes],
        };
        return (probs.to_vec(), prior);
    }
    if n == 0 {
        return (probs.to_vec(), vec![1.0 / classes as f64; classes]);
    }
    let mut src = prior_source
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| column_mean(&flat, n, classes));
    let sum = src.iter().sum::<f64>().max(1e-12);
    for v in src.iter_mut() {
        *v = (*v / sum).max(1e-6);
    }
    let bound = max_ratio.max(1.0);
    let mut ratio = vec![1.0; classes];
    for _ in 0..iters.max(0) {
        let post = reweight(&flat, &ratio, n, classes);
        let mut new = column_mean(&post, n, classes);
        for v in new.iter_mut() {
            *v = v.max(1e-6);
        }
        let total: f64 = new.iter().sum();
        for v in new.iter_mut() {
            *v /= total;
        }
        let step: Vec<f64> = new
            .iter()
            .zip(src.iter())
            .map(|(v, s)| (v / s).clamp(1.0 / bound, bound))
            .collect();
        let moved = step
            .iter()
            .zip(ratio.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        ratio = step;
        if moved < tol {
            break;
        }
    }
    let post = reweight(&flat, &ratio, n, classes);
    let prior = ratio.iter().zip(src.iter()).map(|(r, s)| r * s).collect();
    (post, prior)
}

/// Softmax rescaling of probabilities (`probs ** (1/T)` renormalized).
///
/// Port of `animalpose.behavior.postprocess.temperature_scale`.
pub fn temperature_scale(
    probs: &[f64],
    n: usize,
    classes: usize,
    temperature: f64,
) -> Result<Vec<f64>> {
    if temperature <= 0.0 {
        bail!("temperature must be positive");
    }
    let inv = 1.0 / temperature;
    let mut out: Vec<f64> = probs.iter().map(|v| v.max(1e-12).powf(inv)).collect();
    for t in 0..n {
        let row = &mut out[t * classes..(t + 1) * classes];
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
    Ok(out)
}

/// Temperature minimizing cross-entropy on labelled frames (golden-section
/// search).
///
/// Port of `animalpose.behavior.postprocess.fit_temperature`; `labels[t] < 0`
/// marks an unlabelled frame, and fewer than two labelled frames give `1.0`.
pub fn fit_temperature(
    probs: &[f64],
    n: usize,
    classes: usize,
    labels: &[i64],
    lo: f64,
    hi: f64,
    iters: usize,
) -> f64 {
    let clipped: Vec<f64> = probs.iter().map(|v| v.max(1e-12)).collect();
    let keep: Vec<usize> = (0..n.min(labels.len()))
        .filter(|&t| labels[t] >= 0)
        .collect();
    if keep.len() < 2 {
        return 1.0;
    }
    let rows: Vec<&[f64]> = keep
        .iter()
        .map(|&t| &clipped[t * classes..(t + 1) * classes])
        .collect();
    let labs: Vec<usize> = keep.iter().map(|&t| labels[t] as usize).collect();
    let nll = |temp: f64| -> f64 {
        let mut total = 0.0;
        for (row, &lab) in rows.iter().zip(labs.iter()) {
            let mut sum = 0.0;
            let mut picked = 1.0f64;
            for (c, v) in row.iter().enumerate() {
                let q = v.powf(1.0 / temp);
                sum += q;
                if c == lab {
                    picked = q;
                }
            }
            total += -(picked / sum).ln();
        }
        total / rows.len() as f64
    };
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = nll(c);
    let mut fd = nll(d);
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = nll(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = nll(d);
        }
    }
    (a + b) / 2.0
}

/// `p * ratio` per row, renormalised with the `1e-12` floor of the Python code.
fn reweight(flat: &[f64], ratio: &[f64], n: usize, classes: usize) -> Vec<f64> {
    let mut out = vec![0.0; flat.len()];
    for t in 0..n {
        let row = &flat[t * classes..(t + 1) * classes];
        let w: Vec<f64> = row.iter().zip(ratio.iter()).map(|(p, r)| p * r).collect();
        let sum = w.iter().sum::<f64>().max(1e-12);
        for c in 0..classes {
            out[t * classes + c] = w[c] / sum;
        }
    }
    out
}

/// Mean of every column of a row-major `(n, classes)` array.
fn column_mean(values: &[f64], n: usize, classes: usize) -> Vec<f64> {
    let mut out = vec![0.0; classes];
    for row in values.chunks_exact(classes).take(n) {
        for (o, v) in out.iter_mut().zip(row.iter()) {
            *o += v;
        }
    }
    for o in &mut out {
        *o /= n.max(1) as f64;
    }
    out
}

/// Divides every row by the sum of its entries, floored at `floor`.
fn normalize_rows(values: &mut [f64], n: usize, classes: usize, floor: f64) {
    for t in 0..n {
        let row = &mut values[t * classes..(t + 1) * classes];
        let sum = row.iter().sum::<f64>().max(floor);
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of the `_onehot` helper of `tests/test_behavior_postprocess.py`.
    fn onehot(labels: &[usize], classes: usize) -> Vec<f64> {
        let n = labels.len();
        let mut p = vec![0.02; n * classes];
        for (t, &lab) in labels.iter().enumerate() {
            p[t * classes + lab] = 0.94;
        }
        normalize_rows(&mut p, n, classes, 1e-12);
        p
    }

    fn argmax(p: &[f64], classes: usize) -> Vec<usize> {
        p.chunks_exact(classes)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map_or(0, |(i, _)| i)
            })
            .collect()
    }

    fn row_sums(p: &[f64], classes: usize) -> Vec<f64> {
        p.chunks_exact(classes).map(|r| r.iter().sum()).collect()
    }

    /// `np.random.default_rng(0).random(600) > 0.85`, i.e. ~16.5% ones.
    const A0: &[u8] =
        b"000001000100100010000000001100000000011000000100000010000000000000010001010\
        001100000000100110111011000010001000000001000001000100000000000100000010000000000101000010\
        001100100100110100000010000001000100000001000000000101010001000000000000001010000010000100\
        010000000000000010001001100101000000000000100100000001000000000010010000000100001000000000\
        000011000000100000000001000000000000010100000000100000001000100001010100100000000000000000\
        000000010000000000000100110000000000000000000000000000000000001000110000111000001000000000\
        000001000000000100000000100000010000100000000000000100000000000001010000010";

    /// `np.random.default_rng(1).random(400) > 0.05` (95.25% ones).
    const A1: &[u8] =
        b"111111111011111111111111111111111111011111111111111111111111101111111111111\
        011111111111111111011111111111111111111111111111111111111111111111111111111111111111111111\
        111111111110111111101111111110111111111111111111111011111111111011111111110111111111110111\
        101111110111111111111111111110111111011111111111111111111111111111110111011111111111111111\
        1111111111111101111111111111111111111111111111111111111";

    /// The next 400 draws of that generator: `random(400) > 0.5` (50.75% ones).
    const A2: &[u8] =
        b"111010010010100011101001010010000111100001101101100101001100100101111110111\
        110001000010001010111101100010001000010111100010010100101100111011111111100110101000000011\
        011000101101010101111111010001010000011000010111111100110001000110110110001110010011110000\
        101101110110010010001110111010011100111010001100100010010100000110110110011011000001110110\
        1101111011010111001001101011000110100011001000110110111";

    fn bits(s: &[u8]) -> Vec<usize> {
        s.iter()
            .filter(|b| matches!(b, b'0' | b'1'))
            .map(|b| (b - b'0') as usize)
            .collect()
    }

    #[test]
    fn the_embedded_reference_sequences_have_the_python_lengths() {
        assert_eq!(bits(A0).len(), 600);
        assert_eq!(bits(A1).len(), 400);
        assert_eq!(bits(A2).len(), 400);
        assert_eq!(bits(A0).iter().sum::<usize>(), 99);
        assert_eq!(bits(A1).iter().sum::<usize>(), 381);
        assert_eq!(bits(A2).iter().sum::<usize>(), 203);
    }

    #[test]
    fn moving_median_removes_a_single_frame_flicker_but_keeps_bout_edges() {
        let mut labels = vec![0usize; 30];
        labels.extend(std::iter::repeat(0).take(0));
        labels.extend(std::iter::repeat(1).take(30));
        let mut p = onehot(&labels, 2);
        let n = labels.len();
        p[12 * 2] = 0.5;
        p[12 * 2 + 1] = 0.5;
        p[40 * 2] = 0.3;
        p[40 * 2 + 1] = 0.7; // one flickery frame inside the second bout
        let sm = median_smooth(&p, n, 2, 5, true);
        assert_eq!(sm.len(), p.len());
        for s in row_sums(&sm, 2) {
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert_eq!(argmax(&sm, 2), labels); // flicker gone, boundary at frame 30
        let identity = median_smooth(&p, n, 2, 1, true);
        assert_eq!(identity, p);
    }

    #[test]
    fn moving_median_is_identity_on_a_constant_sequence_and_short_inputs() {
        let labels = vec![0usize; 9];
        let p = onehot(&labels, 3);
        assert_eq!(moving_median(&p, 9, 3, 5), p);
        let short = onehot(&[0usize; 2], 3);
        assert_eq!(moving_median(&short, 2, 3, 5), short);
    }

    #[test]
    fn adapt_prior_moves_marginals_towards_the_recordings_own_class_mix() {
        let truth = bits(A0);
        let n = truth.len();
        let base = onehot(&truth, 2);
        let mut p: Vec<f64> = base.iter().map(|v| v * 0.9 + 0.05).collect();
        normalize_rows(&mut p, n, 2, 1e-12);
        let src = [0.5, 0.5];
        let (adj, prior) = adapt_prior(
            &p,
            n,
            2,
            Some(&src),
            DEFAULT_ADAPT_ITERS,
            DEFAULT_MAX_RATIO,
            DEFAULT_ADAPT_TOL,
        );
        assert_eq!(adj.len(), p.len());
        for s in row_sums(&adj, 2) {
            assert!((s - 1.0).abs() < 1e-5);
        }
        assert!(prior[0] > prior[1], "{prior:?}");
        assert!((prior[0] - 0.85).abs() < 0.15, "{prior:?}");
        let marg = column_mean(&adj, n, 2);
        assert!((marg[0] - prior[0]).abs() < 0.05, "{marg:?} vs {prior:?}");
        let max_log = adj
            .iter()
            .zip(p.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a / b).ln().abs()));
        assert!(max_log <= 16.0f64.ln() + 1e-3, "{max_log}");
        // A less certain model gains most from a looser cap.
        let mut soft: Vec<f64> = base.iter().map(|v| v * 0.2 + 0.4).collect();
        normalize_rows(&mut soft, n, 2, 1e-12);
        let (loose, _) = adapt_prior(
            &soft,
            n,
            2,
            Some(&src),
            DEFAULT_ADAPT_ITERS,
            8.0,
            DEFAULT_ADAPT_TOL,
        );
        let before = argmax(&soft, 2).iter().filter(|c| **c == 0).count();
        let after = argmax(&loose, 2).iter().filter(|c| **c == 0).count();
        assert!(after > before, "{after} vs {before}");
    }

    #[test]
    fn adapt_prior_respects_max_ratio_and_is_a_no_op_when_balanced() {
        let truth = bits(A1);
        let n = truth.len();
        let p = onehot(&truth, 2);
        let (_, prior) = adapt_prior(
            &p,
            n,
            2,
            Some(&[0.5, 0.5]),
            DEFAULT_ADAPT_ITERS,
            1.2,
            DEFAULT_ADAPT_TOL,
        );
        assert!(
            prior[0] <= 1.2 * 0.5 + 1e-6 && prior[1] <= 1.2 * 0.5 + 1e-6,
            "{prior:?}"
        );
        assert!(
            prior[0] >= 0.5 / 1.2 - 1e-6 && prior[1] >= 0.5 / 1.2 - 1e-6,
            "{prior:?}"
        );

        let balanced = onehot(&bits(A2), 2);
        let mean = column_mean(&balanced, n, 2);
        let (same, prior) = adapt_prior(
            &balanced,
            n,
            2,
            Some(&mean),
            30,
            DEFAULT_MAX_RATIO,
            DEFAULT_ADAPT_TOL,
        );
        for (a, b) in prior.iter().zip(mean.iter()) {
            assert!((a - b).abs() < 0.05, "{prior:?} vs {mean:?}");
        }
        let diff = argmax(&same, 2)
            .iter()
            .zip(argmax(&balanced, 2).iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(diff < (0.02 * n as f64) as usize, "{diff}");
    }

    #[test]
    fn adapt_prior_handles_batches_and_degenerate_inputs() {
        // A batch (B, T, C) is just more rows: the Python version reshapes to (-1, C).
        let mut labels = vec![0usize; 20];
        labels.extend(std::iter::repeat(1).take(20));
        let p = onehot(&labels, 2);
        let mut batch = p.clone();
        batch.extend_from_slice(&p);
        let (adj, prior) = adapt_prior(
            &batch,
            40,
            2,
            None,
            DEFAULT_ADAPT_ITERS,
            DEFAULT_MAX_RATIO,
            DEFAULT_ADAPT_TOL,
        );
        assert_eq!(adj.len(), batch.len());
        assert_eq!(prior.len(), 2);

        // A single class is returned unchanged.
        let single: Vec<f64> = p.iter().copied().step_by(2).collect();
        let (same, prior) = adapt_prior(
            &single,
            40,
            1,
            None,
            DEFAULT_ADAPT_ITERS,
            DEFAULT_MAX_RATIO,
            DEFAULT_ADAPT_TOL,
        );
        assert_eq!(same, single);
        assert_eq!(prior.len(), 1);

        let empty: Vec<f64> = Vec::new();
        let (out, prior) = adapt_prior(
            &empty,
            0,
            3,
            None,
            DEFAULT_ADAPT_ITERS,
            DEFAULT_MAX_RATIO,
            DEFAULT_ADAPT_TOL,
        );
        assert!(out.is_empty());
        for v in prior {
            assert!((v - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn temperature_scaling_flattens_confident_logits_and_lowers_nll() {
        let mut labels = vec![0usize; 40];
        labels.extend(std::iter::repeat(1).take(40));
        let n = labels.len();
        let base = onehot(&labels, 2);
        let p: Vec<f64> = base.iter().map(|v| v * 0.6 + 0.2).collect();
        let t = fit_temperature(
            &p,
            n,
            2,
            &labels_as_i64(&labels),
            DEFAULT_TEMPERATURE_LO,
            DEFAULT_TEMPERATURE_HI,
            DEFAULT_TEMPERATURE_ITERS,
        );
        assert!(t < 1.0, "{t}");
        let sharp = temperature_scale(&p, n, 2, 10.0).unwrap();
        for s in row_sums(&sharp, 2) {
            assert!((s - 1.0).abs() < 1e-6);
        }
        let mean = column_mean(&sharp, n, 2);
        assert!(
            (mean[0] - 0.5).abs() < 0.02 && (mean[1] - 0.5).abs() < 0.02,
            "{mean:?}"
        );
        assert_eq!(argmax(&sharp, 2), labels);
        let mut over = base;
        for t in (0..n).step_by(5) {
            over.swap(t * 2, t * 2 + 1); // 20 % of the frames are confidently wrong
        }
        let t = fit_temperature(
            &over,
            n,
            2,
            &labels_as_i64(&labels),
            DEFAULT_TEMPERATURE_LO,
            DEFAULT_TEMPERATURE_HI,
            DEFAULT_TEMPERATURE_ITERS,
        );
        assert!(t > 1.0, "{t}");
    }

    #[test]
    fn temperature_scale_rejects_a_non_positive_temperature() {
        let p = [0.5, 0.5];
        assert!(temperature_scale(&p, 1, 2, 0.0).is_err());
        assert!(temperature_scale(&p, 1, 2, -1.0).is_err());
        let same = temperature_scale(&p, 1, 2, 1.0).unwrap();
        assert_eq!(same, p);
    }

    fn labels_as_i64(labels: &[usize]) -> Vec<i64> {
        labels.iter().map(|v| *v as i64).collect()
    }

    #[test]
    fn fit_temperature_ignores_unlabelled_frames() {
        let labels = vec![0usize; 3];
        let p = onehot(&labels, 2);
        let mut lab = labels_as_i64(&labels);
        lab[1] = -1; // only two labelled frames remain: the search still runs
        let t = fit_temperature(
            &p,
            3,
            2,
            &lab,
            DEFAULT_TEMPERATURE_LO,
            DEFAULT_TEMPERATURE_HI,
            DEFAULT_TEMPERATURE_ITERS,
        );
        assert!(t.is_finite() && t > 0.0);
        lab[0] = -1;
        let t = fit_temperature(
            &p,
            3,
            2,
            &lab,
            DEFAULT_TEMPERATURE_LO,
            DEFAULT_TEMPERATURE_HI,
            DEFAULT_TEMPERATURE_ITERS,
        );
        assert_eq!(t, 1.0);
    }
}
