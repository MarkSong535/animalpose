//! The training targets and losses of the image networks: `pose/model.py:build_targets`,
//! `supcon_loss`, `pose_loss` and `detector/model.py:draw_gaussian`, `build_targets`,
//! `focal_loss`, `detector_loss`.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `pose/model.py:build_targets` | [`pose_targets`] → [`PoseTargets`] |
//! | `pose/model.py:supcon_loss` | [`supcon_loss`] |
//! | `pose/model.py:pose_loss` | [`pose_loss`] |
//! | `detector/model.py:draw_gaussian` / `build_targets` | [`draw_gaussian`] / [`detector_targets`] → [`DetectorTargets`] |
//! | `detector/model.py:focal_loss` / `detector_loss` | [`focal_loss`] / [`detector_loss`] |
//!
//! Each loss returns its terms in the order of the Python dict (`total` last) together with the
//! gradient of `total` with respect to every network output it reads. Those gradients are exact
//! (the analytic derivatives of the same expressions torch differentiates, including the
//! zero gradient of `clamp` outside its range and of `masked_fill`); the trainer seeds them into the
//! tape with a surrogate `sum(output * grad)`, whose parameter gradient is the loss's own. Doing
//! the loss on the host keeps it in the Python form (per-element weights, `exp`/`log`) without
//! requiring those operations on [`crate::tensor::Graph`].

/// A loss value plus `d total / d output` for each output, in the order the caller passed them.
#[derive(Clone, Debug, Default)]
pub struct LossOut {
    /// `(name, value)` in the Python dict order, `total` last.
    pub terms: Vec<(&'static str, f64)>,
    /// Gradient of `total` with respect to each output.
    pub grads: Vec<Vec<f32>>,
}

impl LossOut {
    /// The value of `total`.
    pub fn total(&self) -> f64 {
        self.terms.last().map_or(0.0, |(_, v)| *v)
    }
}

// ------------------------------------------------------------------------------------------ pose

/// The dict `pose/model.py:build_targets` returns.
#[derive(Clone, Debug)]
pub struct PoseTargets {
    /// `B`.
    pub batch: usize,
    /// `K`.
    pub keypoints: usize,
    /// Heat-map height.
    pub height: usize,
    /// Heat-map width.
    pub width: usize,
    /// `(B, K, H, W)` Gaussian of the visible points.
    pub heat: Vec<f32>,
    /// `(B, K)` loss weight: 1 visible, `occluded_weight` occluded, 0 unlabeled.
    pub weight: Vec<f32>,
    /// `(B, 2K, H, W)` sub-cell offsets `(x, y)` per keypoint, near the peak only.
    pub offset: Vec<f32>,
    /// `(B, 2K, H, W)` where the offsets are supervised.
    pub offset_mask: Vec<f32>,
}

/// `build_targets(kpts, state, out_hw, stride, sigma, occluded_weight, offset_radius)`.
///
/// `kpts` is `(B, K, 2)` crop pixels, `state` `(B, K)` with 2 = visible, 1 = labeled but occluded
/// (trained as "not visible": a zero heat map at `occluded_weight`), 0 = unlabeled (no loss). The
/// peak sits on the rounded cell so argmax plus offset recovers the location.
#[allow(clippy::too_many_arguments)]
pub fn pose_targets(
    kpts: &[[f32; 2]],
    state: &[i64],
    batch: usize,
    keypoints: usize,
    out_hw: (usize, usize),
    stride: usize,
    sigma: f64,
    occluded_weight: f64,
    offset_radius: f64,
) -> PoseTargets {
    let (h, w) = out_hw;
    let hw = h * w;
    let mut heat = vec![0.0f32; batch * keypoints * hw];
    let mut weight = vec![0.0f32; batch * keypoints];
    let mut offset = vec![0.0f32; batch * 2 * keypoints * hw];
    let mut offset_mask = vec![0.0f32; batch * 2 * keypoints * hw];
    let two_s2 = (2.0 * sigma * sigma) as f32;
    let r2 = (offset_radius * offset_radius) as f32;
    let stride = stride as f32;
    for b in 0..batch {
        for k in 0..keypoints {
            let i = b * keypoints + k;
            let st = state[i];
            weight[i] = match st {
                2 => 1.0,
                1 => occluded_weight as f32,
                _ => 0.0,
            };
            if st != 2 {
                continue;
            }
            let gx = kpts[i][0] / stride;
            let gy = kpts[i][1] / stride;
            let (cx, cy) = (gx.round_ties_even(), gy.round_ties_even());
            let hbase = i * hw;
            let obase = (b * 2 * keypoints + 2 * k) * hw;
            for y in 0..h {
                let yf = y as f32;
                for x in 0..w {
                    let xf = x as f32;
                    let d2 = (xf - cx) * (xf - cx) + (yf - cy) * (yf - cy);
                    heat[hbase + y * w + x] = (-d2 / two_s2).exp();
                    if d2 <= r2 {
                        let at = y * w + x;
                        offset[obase + at] = gx - xf;
                        offset[obase + hw + at] = gy - yf;
                        offset_mask[obase + at] = 1.0;
                        offset_mask[obase + hw + at] = 1.0;
                    }
                }
            }
        }
    }
    PoseTargets {
        batch,
        keypoints,
        height: h,
        width: w,
        heat,
        weight,
        offset,
        offset_mask,
    }
}

/// `supcon_loss(emb, ids, temperature, groups)`: the supervised contrastive loss over identity
/// labels (ids < 0 are ignored); with `groups` (scene / video ids) each anchor is contrasted only
/// against its own group. Returns the loss and its gradient with respect to `emb` `(N, D)`.
pub fn supcon_loss(
    emb: &[f32],
    dim: usize,
    ids: &[i64],
    temperature: f64,
    groups: Option<&[i64]>,
) -> (f64, Vec<f32>) {
    let n_all = ids.len();
    let mut grad = vec![0.0f32; n_all * dim];
    let keep: Vec<usize> = (0..n_all).filter(|&i| ids[i] >= 0).collect();
    let n = keep.len();
    if n < 2 {
        return (0.0, grad);
    }
    let e = |i: usize| &emb[keep[i] * dim..(keep[i] + 1) * dim];
    let mut sim = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = e(i).iter().zip(e(j)).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
            sim[i * n + j] = dot / temperature;
        }
    }
    let id = |i: usize| ids[keep[i]];
    let group = |i: usize| groups.map(|g| g[keep[i]]);
    let excluded = |i: usize, j: usize| i == j || group(i) != group(j);
    let pos = |i: usize, j: usize| id(i) == id(j) && !excluded(i, j);
    let has_pos: Vec<bool> = (0..n)
        .map(|i| {
            (0..n).any(|j| pos(i, j)) && (0..n).any(|j| !excluded(i, j) && !pos(i, j))
        })
        .collect();
    let anchors = has_pos.iter().filter(|v| **v).count();
    if anchors == 0 {
        return (0.0, grad);
    }
    let mut dsim = vec![0.0f64; n * n];
    let mut loss = 0.0f64;
    for i in 0..n {
        if !has_pos[i] {
            continue;
        }
        // `sim.masked_fill(excluded, -1e9)` then `log_softmax` over the row.
        let row: Vec<f64> = (0..n)
            .map(|j| if excluded(i, j) { -1e9 } else { sim[i * n + j] })
            .collect();
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
        let p_count = (0..n).filter(|&j| pos(i, j)).count().max(1) as f64;
        let mut li = 0.0;
        for (j, r) in row.iter().enumerate() {
            if pos(i, j) {
                li -= r - lse;
            }
        }
        loss += li / p_count;
        for j in 0..n {
            if excluded(i, j) {
                continue;
            }
            let softmax = (row[j] - lse).exp();
            let target = if pos(i, j) { 1.0 / p_count } else { 0.0 };
            dsim[i * n + j] = (softmax - target) / anchors as f64;
        }
    }
    // `sim = E E^T / t`: dL/dE = (G + G^T) E / t.
    for i in 0..n {
        let gi = &mut grad[keep[i] * dim..(keep[i] + 1) * dim];
        for j in 0..n {
            let g = (dsim[i * n + j] + dsim[j * n + i]) / temperature;
            if g == 0.0 {
                continue;
            }
            for (o, v) in gi.iter_mut().zip(e(j)) {
                *o += (g * f64::from(*v)) as f32;
            }
        }
    }
    (loss / anchors as f64, grad)
}

/// `F.binary_cross_entropy_with_logits(x, t, reduction="none")` and its derivative in `x`.
#[inline]
fn bce_logits(x: f64, t: f64) -> (f64, f64) {
    let loss = x.max(0.0) - x * t + (-x.abs()).exp().ln_1p();
    let sig = 1.0 / (1.0 + (-x).exp());
    (loss, sig - t)
}

/// `pose_loss(out, tgt, ids, offset_weight, embed_weight, pos_weight, groups)`.
///
/// `heat` is `(B, K, H, W)` logits, `offset` `(B, 2K, H, W)`, `embedding` `(B, D)` (already
/// L2-normalised by the network). Terms: `heat`, `offset`, `embed` (when `ids` is given and
/// `embed_weight > 0`), `total`; the gradients come back as `[heat, offset, embedding]`.
#[allow(clippy::too_many_arguments)]
pub fn pose_loss(
    heat: &[f32],
    offset: &[f32],
    embedding: &[f32],
    embed_dim: usize,
    tgt: &PoseTargets,
    ids: Option<&[i64]>,
    offset_weight: f64,
    embed_weight: f64,
    pos_weight: f64,
    groups: Option<&[i64]>,
) -> LossOut {
    let hw = tgt.height * tgt.width;
    let mut d_heat = vec![0.0f32; heat.len()];
    let weight_sum: f64 = tgt.weight.iter().map(|v| f64::from(*v)).sum();
    let denom = (weight_sum * hw as f64 / 64.0).max(1.0);
    let mut heat_sum = 0.0f64;
    for (i, (x, t)) in heat.iter().zip(&tgt.heat).enumerate() {
        let w = f64::from(tgt.weight[i / hw]);
        if w == 0.0 {
            continue;
        }
        let t = f64::from(*t);
        let (l, g) = bce_logits(f64::from(*x), t);
        let we = (1.0 + pos_weight * t) * w;
        heat_sum += l * we;
        d_heat[i] = (g * we / denom) as f32;
    }
    let heat_loss = heat_sum / denom;

    let m_sum: f64 = tgt.offset_mask.iter().map(|v| f64::from(*v)).sum();
    let m_den = m_sum.max(1.0);
    let mut d_off = vec![0.0f32; offset.len()];
    let mut off_sum = 0.0f64;
    for (i, ((o, t), m)) in offset.iter().zip(&tgt.offset).zip(&tgt.offset_mask).enumerate() {
        if *m == 0.0 {
            continue;
        }
        let d = f64::from(*o) - f64::from(*t);
        // `F.smooth_l1_loss(..., beta=1.0)`.
        let (l, g) = if d.abs() < 1.0 {
            (0.5 * d * d, d)
        } else {
            (d.abs() - 0.5, d.signum())
        };
        off_sum += l * f64::from(*m);
        d_off[i] = (offset_weight * g * f64::from(*m) / m_den) as f32;
    }
    let off_loss = offset_weight * off_sum / m_den;

    let mut terms = vec![("heat", heat_loss), ("offset", off_loss)];
    let mut d_emb = vec![0.0f32; embedding.len()];
    if let Some(ids) = ids {
        if embed_weight > 0.0 {
            let (l, g) = supcon_loss(embedding, embed_dim, ids, 0.1, groups);
            terms.push(("embed", embed_weight * l));
            for (o, v) in d_emb.iter_mut().zip(&g) {
                *o = (embed_weight * f64::from(*v)) as f32;
            }
        }
    }
    let total = terms.iter().map(|(_, v)| v).sum();
    terms.push(("total", total));
    LossOut {
        terms,
        grads: vec![d_heat, d_off, d_emb],
    }
}

// -------------------------------------------------------------------------------------- detector

/// `draw_gaussian(heat, cx, cy, sx, sy)`: `heat = max(heat, g)` over the `3 sigma` window of an
/// axis-aligned Gaussian. `heat` is one `(h, w)` plane.
pub fn draw_gaussian(heat: &mut [f32], h: usize, w: usize, cx: f64, cy: f64, sx: f64, sy: f64) {
    // `int(...)` truncates towards zero.
    let x0 = 0i64.max((cx - 3.0 * sx) as i64);
    let x1 = (w as i64).min((cx + 3.0 * sx) as i64 + 2);
    let y0 = 0i64.max((cy - 3.0 * sy) as i64);
    let y1 = (h as i64).min((cy + 3.0 * sy) as i64 + 2);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let (cxf, cyf) = (cx as f32, cy as f32);
    let (dx2, dy2) = ((2.0 * sx * sx) as f32, (2.0 * sy * sy) as f32);
    for y in y0..y1 {
        for x in x0..x1 {
            let (xf, yf) = (x as f32, y as f32);
            let g = (-((xf - cxf) * (xf - cxf)) / dx2 - ((yf - cyf) * (yf - cyf)) / dy2).exp();
            let at = y as usize * w + x as usize;
            heat[at] = heat[at].max(g);
        }
    }
}

/// The dict `detector/model.py:build_targets` returns.
#[derive(Clone, Debug)]
pub struct DetectorTargets {
    /// `B`.
    pub batch: usize,
    /// Map height.
    pub height: usize,
    /// Map width.
    pub width: usize,
    /// `(B, 1, h, w)` centre heat map, exactly 1 on each box's peak cell.
    pub heat: Vec<f32>,
    /// `(B, 2, h, w)` `log(w, h)` in stride units at the peak cells.
    pub size: Vec<f32>,
    /// `(B, 2, h, w)` sub-cell centre offset at the peak cells.
    pub offset: Vec<f32>,
    /// `(B, 1, h, w)` 1 at the peak cells.
    pub reg_mask: Vec<f32>,
    /// `(B, h, w)` species index at the peak cells, -100 elsewhere.
    pub species: Vec<i64>,
}

/// `build_targets(boxes, out_hw, stride, species)`: dense targets from per-image xyxy boxes in
/// input pixels (a TTFNet-style elliptical Gaussian centred on the integer peak cell).
pub fn detector_targets(
    boxes: &[Vec<[f32; 4]>],
    out_hw: (usize, usize),
    stride: usize,
    species: Option<&[Vec<i64>]>,
) -> DetectorTargets {
    let b = boxes.len();
    let (h, w) = out_hw;
    let hw = h * w;
    let mut t = DetectorTargets {
        batch: b,
        height: h,
        width: w,
        heat: vec![0.0; b * hw],
        size: vec![0.0; b * 2 * hw],
        offset: vec![0.0; b * 2 * hw],
        reg_mask: vec![0.0; b * hw],
        species: vec![-100; b * hw],
    };
    let s = stride as f64;
    for (i, bx) in boxes.iter().enumerate() {
        for (j, bb) in bx.iter().enumerate() {
            let [x0, y0, x1, y1] = bb.map(f64::from);
            let (bw, bh) = ((x1 - x0) / s, (y1 - y0) / s);
            if bw <= 0.0 || bh <= 0.0 {
                continue;
            }
            let (cx, cy) = ((x0 + x1) / 2.0 / s, (y0 + y1) / 2.0 / s);
            let ix = (cx as i64).max(0).min(w as i64 - 1) as usize;
            let iy = (cy as i64).max(0).min(h as i64 - 1) as usize;
            draw_gaussian(
                &mut t.heat[i * hw..(i + 1) * hw],
                h,
                w,
                ix as f64,
                iy as f64,
                (0.54 * bw / 6.0).max(0.8),
                (0.54 * bh / 6.0).max(0.8),
            );
            let at = iy * w + ix;
            t.heat[i * hw + at] = 1.0;
            t.size[(i * 2) * hw + at] = bw.ln() as f32;
            t.size[(i * 2 + 1) * hw + at] = bh.ln() as f32;
            t.offset[(i * 2) * hw + at] = (cx - ix as f64) as f32;
            t.offset[(i * 2 + 1) * hw + at] = (cy - iy as f64) as f32;
            t.reg_mask[i * hw + at] = 1.0;
            if let Some(sp) = species {
                t.species[i * hw + at] = sp[i][j];
            }
        }
    }
    t
}

/// `focal_loss(logits, target, alpha, beta)`: CenterNet's penalty-reduced focal loss, normalised
/// by the number of peaks. Returns the loss and its gradient in `logits`.
pub fn focal_loss(logits: &[f32], target: &[f32], alpha: f64, beta: f64) -> (f64, Vec<f32>) {
    let n_pos = target.iter().filter(|t| **t == 1.0).count() as f64;
    let n = n_pos.max(1.0);
    let (lo, hi) = (1e-4f64, 1.0 - 1e-4);
    let mut loss = 0.0f64;
    let mut grad = vec![0.0f32; logits.len()];
    for (i, (x, t)) in logits.iter().zip(target).enumerate() {
        let s = 1.0 / (1.0 + (-f64::from(*x)).exp());
        // `sigmoid(logits).clamp(1e-4, 1 - 1e-4)`: no gradient where the clamp is active.
        let p = s.clamp(lo, hi);
        let live = (lo..=hi).contains(&s);
        let t = f64::from(*t);
        let dldp = if t == 1.0 {
            loss += -p.ln() * (1.0 - p).powf(alpha);
            -(1.0 - p).powf(alpha) / p + alpha * (1.0 - p).powf(alpha - 1.0) * p.ln()
        } else {
            let w = (1.0 - t).powf(beta);
            loss += -(1.0 - p).ln() * p.powf(alpha) * w;
            w * (p.powf(alpha) / (1.0 - p) - alpha * p.powf(alpha - 1.0) * (1.0 - p).ln())
        };
        if live {
            grad[i] = (dldp * s * (1.0 - s) / n) as f32;
        }
    }
    (loss / n, grad)
}

/// `detector_loss(out, tgt, size_weight, offset_weight)`: focal heat loss, masked L1 on the size
/// and offset maps and, with a species head, cross entropy over the peak cells. Terms: `heat`,
/// `size`, `offset`, (`species`), `total`; gradients `[heat, size, offset, (species)]`.
///
/// `// differs from Python:` `F.cross_entropy` over a batch whose cells are all ignored returns NaN
/// in torch; here the species term is then 0. A species index below zero that is not the ignore
/// value (an unknown species from a COCO source) is also ignored rather than raising.
pub fn detector_loss(
    heat: &[f32],
    size: &[f32],
    offset: &[f32],
    species: Option<(&[f32], usize)>,
    tgt: &DetectorTargets,
    size_weight: f64,
    offset_weight: f64,
) -> LossOut {
    let hw = tgt.height * tgt.width;
    let n = tgt.reg_mask.iter().map(|v| f64::from(*v)).sum::<f64>().max(1.0);
    let (heat_loss, d_heat) = focal_loss(heat, &tgt.heat, 2.0, 4.0);
    let masked_l1 = |out: &[f32], target: &[f32], weight: f64| -> (f64, Vec<f32>) {
        let mut sum = 0.0f64;
        let mut grad = vec![0.0f32; out.len()];
        for (i, (o, t)) in out.iter().zip(target).enumerate() {
            let b = i / (2 * hw);
            let m = f64::from(tgt.reg_mask[b * hw + i % hw]);
            if m == 0.0 {
                continue;
            }
            let d = f64::from(*o) - f64::from(*t);
            sum += d.abs() * m;
            // `sign(0) = 0`, the subgradient torch uses for `abs`.
            let sign = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            grad[i] = (weight * sign * m / n) as f32;
        }
        (weight * sum / n, grad)
    };
    let (size_loss, d_size) = masked_l1(size, &tgt.size, size_weight);
    let (off_loss, d_off) = masked_l1(offset, &tgt.offset, offset_weight);
    let mut terms = vec![("heat", heat_loss), ("size", size_loss), ("offset", off_loss)];
    let mut grads = vec![d_heat, d_size, d_off];
    if let Some((logits, classes)) = species {
        let mut grad = vec![0.0f32; logits.len()];
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (cell, &target) in tgt.species.iter().enumerate() {
            if target < 0 || target as usize >= classes {
                continue;
            }
            count += 1;
            let (b, at) = (cell / hw, cell % hw);
            let idx = |c: usize| (b * classes + c) * hw + at;
            let max = (0..classes)
                .map(|c| f64::from(logits[idx(c)]))
                .fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = (0..classes).map(|c| (f64::from(logits[idx(c)]) - max).exp()).sum();
            sum += -(f64::from(logits[idx(target as usize)]) - max - z.ln());
            for c in 0..classes {
                let p = (f64::from(logits[idx(c)]) - max).exp() / z;
                let onehot = if c == target as usize { 1.0 } else { 0.0 };
                grad[idx(c)] = (p - onehot) as f32;
            }
        }
        let ce = if count > 0 { sum / count as f64 } else { 0.0 };
        if count > 0 {
            for g in grad.iter_mut() {
                *g /= count as f32;
            }
        }
        terms.push(("species", ce));
        grads.push(grad);
    }
    let total = terms.iter().map(|(_, v)| v).sum();
    terms.push(("total", total));
    LossOut { terms, grads }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central differences of a scalar function of one buffer.
    fn numeric_grad(x: &[f32], f: &dyn Fn(&[f32]) -> f64) -> Vec<f64> {
        let eps = 1e-3f32;
        (0..x.len())
            .map(|i| {
                let mut a = x.to_vec();
                let mut b = x.to_vec();
                a[i] += eps;
                b[i] -= eps;
                (f(&a) - f(&b)) / (2.0 * f64::from(eps))
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f64], tol: f64) {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (f64::from(*x) - y).abs() <= tol * (1.0 + y.abs()),
                "grad {i}: {x} vs {y}"
            );
        }
    }

    #[test]
    fn pose_targets_peak_on_the_rounded_cell_and_decode_back() {
        let kp = [[13.0f32, 22.0], [0.0, 0.0]];
        let t = pose_targets(&kp, &[2, 1], 1, 2, (8, 8), 4, 1.5, 0.5, 2.0);
        // 13 / 4 = 3.25 -> cell 3, 22 / 4 = 5.5 -> cell 6 (round half to even).
        let peak = t.heat[..64]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!((peak % 8, peak / 8), (3, 6));
        assert_eq!(t.heat[6 * 8 + 3], 1.0);
        // argmax + offset recovers the keypoint (decode round trip).
        let (ox, oy) = (t.offset[6 * 8 + 3], t.offset[64 + 6 * 8 + 3]);
        assert!(((3.0 + ox) * 4.0 - 13.0).abs() < 1e-5);
        assert!(((6.0 + oy) * 4.0 - 22.0).abs() < 1e-5);
        // The occluded point has weight 0.5 and an empty heat map.
        assert_eq!(t.weight, vec![1.0, 0.5]);
        assert!(t.heat[64..].iter().all(|v| *v == 0.0));
        assert!(t.offset_mask[2 * 64..].iter().all(|v| *v == 0.0));
        // 12 cells lie within radius 2 of the peak (the 13th, at row 8, is off the map).
        assert_eq!(t.offset_mask[..64].iter().filter(|v| **v > 0.0).count(), 12);
    }

    #[test]
    fn pose_loss_gradients_match_finite_differences() {
        let kp = [[9.0f32, 7.0], [3.0, 12.0], [5.5, 5.0], [1.0, 1.0]];
        let state = [2, 1, 2, 0];
        let t = pose_targets(&kp, &state, 2, 2, (4, 4), 4, 1.5, 0.5, 2.0);
        let heat: Vec<f32> = (0..64).map(|i| ((i * 37 % 17) as f32 - 8.0) * 0.3).collect();
        let off: Vec<f32> = (0..128).map(|i| ((i * 13 % 11) as f32 - 5.0) * 0.25).collect();
        let emb = [0.6f32, 0.8, 0.0, 1.0];
        let ids = [3i64, 5];
        let out = pose_loss(&heat, &off, &emb, 2, &t, Some(&ids), 0.05, 0.1, 10.0, None);
        let fh = |h: &[f32]| pose_loss(h, &off, &emb, 2, &t, None, 0.05, 0.1, 10.0, None).total();
        close(&out.grads[0], &numeric_grad(&heat, &fh), 2e-2);
        let fo = |o: &[f32]| pose_loss(&heat, o, &emb, 2, &t, None, 0.05, 0.1, 10.0, None).total();
        close(&out.grads[1], &numeric_grad(&off, &fo), 2e-2);
        assert_eq!(out.terms.iter().map(|t| t.0).collect::<Vec<_>>(), ["heat", "offset", "embed", "total"]);
        // `pose_loss` of `pose/model.py` on the same tensors (torch 2, float32).
        let python = [188.8974609375, 0.02984730154275894, 0.0, 188.92730712890625];
        for ((name, v), p) in out.terms.iter().zip(python) {
            assert!((v - p).abs() < 1e-4 * (1.0 + p), "{name}: {v} vs {p}");
        }
    }

    #[test]
    fn supcon_gradient_matches_finite_differences() {
        let emb: Vec<f32> = vec![0.3, 0.1, -0.2, 0.5, 0.4, 0.0, -0.1, 0.2, 0.25, 0.6, -0.3, 0.1];
        let ids = [1i64, 1, 2, 2, -1, 2];
        let groups = [0i64, 0, 0, 0, 0, 1];
        let (l, g) = supcon_loss(&emb, 2, &ids, 0.1, Some(&groups));
        // `supcon_loss` and its autograd gradient in torch.
        assert!((l - 1.814003348350525).abs() < 1e-5, "{l}");
        let python = [
            2.1197457313537598, -1.8569279909133911, -1.3723225593566895, 0.2867533564567566,
            1.3756320476531982, -0.2606177031993866, -2.135427951812744, 1.8259611129760742,
        ];
        for (a, b) in g.iter().zip(python) {
            assert!((f64::from(*a) - b).abs() < 1e-4, "{a} vs {b}");
        }
        let f = |e: &[f32]| supcon_loss(e, 2, &ids, 0.1, Some(&groups)).0;
        close(&g, &numeric_grad(&emb, &f), 2e-2);
        // The id < 0 row gets no gradient; nor does the lone member of group 1.
        assert_eq!(&g[8..10], &[0.0, 0.0]);
        assert_eq!(&g[10..12], &[0.0, 0.0]);
        // Fewer than two labelled samples: zero.
        assert_eq!(supcon_loss(&emb[..2], 2, &[4], 0.1, None).0, 0.0);
    }

    #[test]
    fn detector_targets_and_loss() {
        let boxes = vec![vec![[8.0f32, 8.0, 40.0, 24.0]], vec![]];
        let t = detector_targets(&boxes, (8, 16), 4, Some(&[vec![1], vec![]]));
        // Centre (24, 16) / 4 = (6, 4).
        let at = 4 * 16 + 6;
        assert_eq!(t.heat[at], 1.0);
        assert_eq!(t.reg_mask.iter().sum::<f32>(), 1.0);
        assert!((t.size[at] - 8.0f32.ln()).abs() < 1e-6);
        assert!((t.size[128 + at] - 4.0f32.ln()).abs() < 1e-6);
        assert_eq!(t.offset[at], 0.0);
        assert_eq!(t.species[at], 1);
        assert!(t.heat[at + 1] > 0.0 && t.heat[at + 1] < 1.0);
        let heat: Vec<f32> = (0..256).map(|i| ((i * 7 % 23) as f32 - 15.0) * 0.4).collect();
        let size: Vec<f32> = (0..512).map(|i| (i % 5) as f32 * 0.7).collect();
        let off: Vec<f32> = (0..512).map(|i| (i % 3) as f32 * 0.2 - 0.1).collect();
        let spc: Vec<f32> = (0..3 * 256).map(|i| ((i * 11 % 7) as f32 - 3.0) * 0.5).collect();
        let out = detector_loss(&heat, &size, &off, Some((&spc, 3)), &t, 0.1, 1.0);
        let fh = |h: &[f32]| detector_loss(h, &size, &off, Some((&spc, 3)), &t, 0.1, 1.0).total();
        close(&out.grads[0], &numeric_grad(&heat, &fh), 2e-2);
        let fs = |s: &[f32]| detector_loss(&heat, &size, &off, Some((s, 3)), &t, 0.1, 1.0).total();
        close(&out.grads[3], &numeric_grad(&spc, &fs), 2e-2);
        assert_eq!(
            out.terms.iter().map(|t| t.0).collect::<Vec<_>>(),
            ["heat", "size", "offset", "species", "total"]
        );
        // `detector_loss` of `detector/model.py` on the same tensors, and the target heat mass.
        let python = [
            106.23928833007812,
            0.27931472659111023,
            0.20000000298023224,
            1.180269718170166,
            107.89886474609375,
        ];
        for ((name, v), p) in out.terms.iter().zip(python) {
            assert!((v - p).abs() < 1e-4 * (1.0 + p), "{name}: {v} vs {p}");
        }
        assert!((t.heat.iter().sum::<f32>() - 4.021_261).abs() < 1e-5);
    }
}
