//! `src/animalpose/temporal/model.py`: an LSTM over keypoint tracks that smooths, corrects outliers
//! and fills gaps.
//!
//! [`TemporalLSTM`] takes coordinates `(B, T, K, D)`, detector confidence `(B, T, K)` and an
//! `observed` mask, and returns refined coordinates for every frame plus a per-point uncertainty
//! (`log_sigma`) and the normalisation it used. With `prior_bandwidth > 0` the network predicts a
//! residual on top of an interpolation prior ([`Prior::Kernel`], [`Prior::Linear`] or
//! [`Prior::Hybrid`]), and `log` of the prior's support becomes an extra input.
//!
//! Checkpoint naming:
//!
//! | Python module | checkpoint key | Rust construction |
//! | --- | --- | --- |
//! | `TemporalLSTM.inp` (`Linear`, `ReLU`) | `inp.0.weight` `(hidden, n_in)`, `inp.0.bias` | [`Dense`] + [`Relu`] |
//! | `TemporalLSTM.lstm` (`nn.LSTM`) | `lstm.weight_ih_l{i}[_reverse]`, `weight_hh_l{i}[_reverse]`, `bias_ih_l{i}[_reverse]`, `bias_hh_l{i}[_reverse]` | [`Lstm`] |
//! | `TemporalLSTM.head` (`Linear`, `ReLU`, `Linear`) | `head.0.*`, `head.2.*` | [`Dense`] + [`Relu`] |
//!
//! The normalisation, the prior and the feature layout are computed over plain values rather than on
//! the tape: they hold no parameters and the Python trainer never asks for a gradient with respect to
//! the input coordinates, so the numbers are identical (`// differs from Python:` at
//! [`TemporalLSTM::forward`]). The eval-mode residual guard of `TemporalLSTM.forward` *is* applied,
//! as a `0`/`1` mask node, which is what `torch.where` does to the gradient too.

use crate::models::config::ModelConfig;
use crate::models::layers::{LayerTree, LoadReport, Relu};
use crate::models::weights::Weights;
use crate::tensor::{Ctx, Dense, Graph, Layer, Lstm, NodeId, Sequential};
use anyhow::{bail, ensure, Result};

/// `RESIDUAL_GUARD` of `temporal/model.py`, in units of the window's keypoint spread.
pub const RESIDUAL_GUARD: f32 = 4.0;

/// `TemporalLSTM`'s default `hidden`, i.e. `BUILDERS`' `c.get("hidden", 128)` for this kind.
pub const TEMPORAL_HIDDEN: usize = 128;

/// The clamp range of the predicted `log_sigma`, `out[..., d].clamp(-7.0, 3.0)`.
pub const LOG_SIGMA_RANGE: (f32, f32) = (-7.0, 3.0);

/// The inputs of `TemporalLSTM.forward(x, conf, observed)`: `(B, T, K, D)` coordinates, `(B, T, K)`
/// detector confidence and the `(B, T, K)` observed mask.
pub const INPUTS: [&str; 3] = ["coords", "conf", "observed"];

/// The output names of `TemporalLSTM.forward`, in the order of the Python dict.
pub const OUTPUTS: [&str; 5] = ["coords", "coords_norm", "log_sigma", "center", "scale"];

/// The shape of a track batch, `(B, T, K, D)`, which the helpers of this module thread around instead
/// of four loose numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Track {
    /// Sequences in the batch.
    pub b: usize,
    /// Frames per sequence.
    pub t: usize,
    /// Keypoints per frame.
    pub k: usize,
    /// Coordinates per keypoint.
    pub d: usize,
}

/// `TemporalLSTM.prior`: the interpolation the network corrects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Prior {
    /// A Laplacian-weighted average of the observed frames ([`kernel_prior`]).
    #[default]
    Kernel,
    /// Exact linear interpolation between the nearest observed frames ([`linear_prior`]).
    Linear,
    /// The kernel average where a point is observed, linear interpolation inside the gaps.
    Hybrid,
}

impl Prior {
    /// The prior named by the config string (`"kernel"`, `"linear"`, `"hybrid"`).
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "kernel" => Ok(Self::Kernel),
            "linear" => Ok(Self::Linear),
            "hybrid" => Ok(Self::Hybrid),
            other => bail!("unknown temporal prior {other:?}: expected kernel, linear or hybrid"),
        }
    }

    /// The Python name of this prior.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Linear => "linear",
            Self::Hybrid => "hybrid",
        }
    }
}

/// `TemporalLSTM.normalizer`: the masked centre `(B, 1, 1, D)` and isotropic scale `(B, 1, 1, 1)` of a
/// window, over row-major `(b, t, k, d)` coordinates and `(b, t, k)` observations.
pub fn normalizer(x: &[f32], observed: &[f32], track: Track) -> (Vec<f32>, Vec<f32>) {
    let Track { b, t, k, d } = track;
    let rows = t * k;
    let mut center = vec![0.0f32; b * d];
    let mut scale = vec![0.0f32; b];
    for bi in 0..b {
        let window = bi * rows * d..(bi + 1) * rows * d;
        let vis = &observed[bi * rows..(bi + 1) * rows];
        let n = (vis.iter().sum::<f32>()).max(1.0);
        let mut mean = vec![0.0f32; d];
        for i in 0..rows {
            if vis[i] == 0.0 {
                continue;
            }
            for c in 0..d {
                mean[c] += x[window.start + i * d + c];
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut var = 0.0f32;
        for i in 0..rows {
            if vis[i] == 0.0 {
                continue;
            }
            for c in 0..d {
                let delta = x[window.start + i * d + c] - mean[c];
                var += delta * delta;
            }
        }
        var /= n * d as f32;
        center[bi * d..(bi + 1) * d].copy_from_slice(&mean);
        scale[bi] = var.sqrt().max(1e-3);
    }
    (center, scale)
}

/// `TemporalLSTM.kernel_prior`: the Laplacian-weighted average of the observed frames, returning
/// `(prior (B, T, K, D), support (B, T, K))`. Causal models (`bidirectional == false`) may only look
/// back.
pub fn kernel_prior(
    xn: &[f32],
    observed: &[f32],
    bandwidth: f32,
    bidirectional: bool,
    track: Track,
) -> (Vec<f32>, Vec<f32>) {
    let Track { b, t, k, d } = track;
    let rows = t * k;
    let mut base = vec![0.0f32; b * rows * d];
    let mut support = vec![0.0f32; b * rows];
    let kernel = |q: usize, s: usize| -> f32 {
        if !bidirectional && s > q {
            0.0
        } else {
            (-(q as f32 - s as f32).abs() / bandwidth).exp()
        }
    };
    for bi in 0..b {
        for q in 0..t {
            for s in 0..t {
                let w = kernel(q, s);
                if w == 0.0 {
                    continue;
                }
                // `support = matmul(w, observed)` is per keypoint, and `num` divides by it per keypoint.
                for j in 0..k {
                    let seen = observed[bi * rows + s * k + j];
                    support[bi * rows + q * k + j] += w * seen;
                    for c in 0..d {
                        base[(bi * rows + q * k + j) * d + c] +=
                            w * seen * xn[(bi * rows + s * k + j) * d + c];
                    }
                }
            }
        }
    }
    // `num / support.clamp(min = 1e-6)`: the support that is returned (and fed to `log`) is unclamped.
    for at in 0..support.len() {
        let total = support[at].max(1e-6);
        for c in 0..d {
            base[at * d + c] /= total;
        }
    }
    (base, support)
}

/// `TemporalLSTM.linear_prior`: linear interpolation between the nearest observed frames of every
/// keypoint, holding the nearest value at the ends and zero when a point is never observed. Causal
/// models hold the last observation instead of interpolating.
pub fn linear_prior(xn: &[f32], observed: &[f32], bidirectional: bool, track: Track) -> Vec<f32> {
    let Track { b, t, k, d } = track;
    let rows = t * k;
    let mut out = vec![0.0f32; b * rows * d];
    for bi in 0..b {
        for j in 0..k {
            let seen = |s: usize| observed[bi * rows + s * k + j] > 0.5;
            for q in 0..t {
                let prev = (0..=q).rev().find(|&s| seen(s));
                // `nxt = min{ s >= q : observed }`, and `has_next` only ever holds for a
                // bidirectional model (`if not self.bidirectional: has_next = zeros_like(...)`).
                let next = bidirectional.then(|| (q..t).find(|&s| seen(s))).flatten();
                let has_prev = prev.is_some();
                let has_next = next.is_some();
                let p_idx = prev.or(next).unwrap_or(0);
                let n_idx = next.unwrap_or(p_idx);
                let span = (n_idx as f32 - p_idx as f32).max(1.0);
                let mut frac = ((q as f32 - p_idx as f32) / span).clamp(0.0, 1.0);
                if !(has_prev && has_next) {
                    frac = 0.0;
                }
                for c in 0..d {
                    let p = xn[(bi * rows + p_idx * k + j) * d + c];
                    let n = xn[(bi * rows + n_idx * k + j) * d + c];
                    let value = if has_prev || has_next {
                        p + frac * (n - p)
                    } else {
                        0.0
                    };
                    out[(bi * rows + q * k + j) * d + c] = value;
                }
            }
        }
    }
    out
}

/// `TemporalLSTM` from `src/animalpose/temporal/model.py`.
pub struct TemporalLSTM {
    inp: Sequential,
    lstm: Lstm,
    head: Sequential,
    k: usize,
    d: usize,
    bidirectional: bool,
    bandwidth: f32,
    prior: Prior,
    input_width: usize,
}

impl TemporalLSTM {
    /// `TemporalLSTM(num_keypoints, dims, hidden, layers, bidirectional, dropout, prior_bandwidth,
    /// prior)`.
    pub fn new(cfg: &ModelConfig) -> Result<Self> {
        let (k, d) = (cfg.num_keypoints(), cfg.dims());
        ensure!(d == 2 || d == 3, "dims must be 2 or 3, got {d}");
        let bandwidth = cfg.prior_bandwidth();
        let prior = Prior::parse(cfg.prior())?;
        let hidden = cfg.hidden_or(TEMPORAL_HIDDEN);
        let input_width = k * (d + 2) + usize::from(bandwidth > 0.0) * k * (d + 1);
        let lstm = Lstm::new(
            "lstm",
            hidden,
            hidden,
            cfg.layers(),
            cfg.bidirectional(),
            cfg.dropout(),
        );
        let out = lstm.out_width();
        Ok(Self {
            inp: Sequential::of(vec![
                Box::new(Dense::new("inp.0", input_width, hidden, true)),
                Box::new(Relu),
            ]),
            head: Sequential::of(vec![
                Box::new(Dense::new("head.0", out, hidden, true)),
                Box::new(Relu),
                Box::new(Dense::new("head.2", hidden, k * (d + 1), true)),
            ]),
            lstm,
            k,
            d,
            bidirectional: cfg.bidirectional(),
            bandwidth,
            prior,
            input_width,
        })
    }

    /// `TemporalLSTM.k`, the keypoint count.
    pub fn k(&self) -> usize {
        self.k
    }

    /// `TemporalLSTM.d`, the coordinate width.
    pub fn d(&self) -> usize {
        self.d
    }

    /// `TemporalLSTM.bidirectional`.
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }

    /// `TemporalLSTM.prior_bandwidth`; `0` disables the interpolation prior.
    pub fn prior_bandwidth(&self) -> f32 {
        self.bandwidth
    }

    /// `TemporalLSTM.prior`.
    pub fn prior(&self) -> Prior {
        self.prior
    }

    /// The input width of `inp`: `k * (d + 2)`, plus `k * (d + 1)` when a prior is active.
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// `TemporalLSTM.forward(x, conf, observed)`, returning the outputs in the Python dict order:
    /// `coords`, `coords_norm`, `log_sigma`, `center`, `scale`.
    ///
    /// `// differs from Python:` the feature block (normalisation, prior, concatenation) runs over
    /// plain values instead of tape nodes, so `Graph::backward` reports no gradient for the input
    /// coordinates; every parameter gradient is unaffected.
    pub fn forward(
        &self,
        g: &mut Graph,
        x: NodeId,
        conf: NodeId,
        observed: NodeId,
        ctx: &mut Ctx,
    ) -> Result<Vec<(&'static str, NodeId)>> {
        let shape = g.shape(x).clone();
        ensure!(
            shape.len() == 4 && shape[2] == self.k && shape[3] == self.d,
            "temporal input must be (B, T, {}, {}), got {shape:?}",
            self.k,
            self.d
        );
        let (b, t, k, d) = (shape[0], shape[1], shape[2], shape[3]);
        let mask_shape = g.shape(observed).clone();
        ensure!(
            mask_shape == vec![b, t, k] && g.shape(conf).clone() == mask_shape,
            "conf and observed must be (B, T, {k}), got {mask_shape:?}"
        );
        let track = Track { b, t, k, d };
        let raw = g.value(x).to_vec();
        let raw_conf = g.value(conf).to_vec();
        let raw_obs = g.value(observed).to_vec();
        let feats = self.features(&raw, &raw_conf, &raw_obs, track);
        let base = feats.base;
        let feature_node = g.from_data(feats.concat, &[b, t, self.input_width]);

        let h = self.inp.forward(g, feature_node, ctx);
        let h = self.lstm.forward(g, h, ctx);
        let h = self.head.forward(g, h, ctx);
        let h = g.reshape(h, &[b, t, k, d + 1]);

        let delta = g.slice(h, 3, 0, d);
        let sigma_raw = g.slice(h, 3, d, d + 1);
        let sigma_flat = g.reshape(sigma_raw, &[b, t, k]);
        let log_sigma = clamp(g, sigma_flat, LOG_SIGMA_RANGE.0, LOG_SIGMA_RANGE.1);

        // The out-of-distribution guard of eval mode: `bad = (delta.abs() > RESIDUAL_GUARD).any(-1)`
        // zeroes the correction and reports maximal uncertainty. `torch.where` passes no gradient to
        // the branch it drops, which is what multiplying by a 0/1 mask does.
        let mut delta_out = delta;
        let mut sigma_out = log_sigma;
        if !ctx.train {
            let keep = guard_mask(g.value(h), d, b, t, k);
            if keep.iter().any(|&v| v < 1.0) {
                let keep4 = g.from_data(keep.clone(), &[b, t, k, 1]);
                let keep3 = g.from_data(keep, &[b, t, k]);
                delta_out = g.mul(delta, keep4);
                let one = g.constant(1.0, &[1]);
                let hi = g.constant(LOG_SIGMA_RANGE.1, &[1]);
                let inverted = g.sub(one, keep3);
                let held = g.mul(inverted, hi);
                let kept = g.mul(log_sigma, keep3);
                sigma_out = g.add(kept, held);
            }
        }

        let base = g.from_data(base, &[b, t, k, d]);
        let coords_norm = g.add(base, delta_out);
        let center = g.from_data(feats.center, &[b, 1, 1, d]);
        let scale = g.from_data(feats.scale, &[b, 1, 1, 1]);
        let scaled = g.mul(coords_norm, scale);
        let coords = g.add(scaled, center);
        Ok(vec![
            (OUTPUTS[0], coords),
            (OUTPUTS[1], coords_norm),
            (OUTPUTS[2], sigma_out),
            (OUTPUTS[3], center),
            (OUTPUTS[4], scale),
        ])
    }

    /// `model.load_state_dict(...)`: copies every matching tensor of `weights` in.
    pub fn load_weights(&mut self, weights: &Weights) -> LoadReport {
        crate::models::layers::load_into(self, weights)
    }

    /// The concatenated LSTM input, the prior it corrects and the normalisation constants.
    fn features(&self, x: &[f32], conf: &[f32], observed: &[f32], track: Track) -> Features {
        let Track { b, t, k, d } = track;
        let rows = t * k;
        // `observed = observed.to(x.dtype)`; `x = torch.nan_to_num(x) * observed.unsqueeze(-1)`.
        let mut xs = vec![0.0f32; b * rows * d];
        for i in 0..x.len() {
            let j = i / d;
            xs[i] = finite(x[i]) * observed[j];
        }
        let (center, scale) = normalizer(&xs, observed, track);
        // `xn = (x - center) / scale * observed.unsqueeze(-1)`
        let mut xn = vec![0.0f32; b * rows * d];
        for bi in 0..b {
            for i in 0..rows {
                for c in 0..d {
                    let at = (bi * rows + i) * d + c;
                    xn[at] = (xs[at] - center[bi * d + c]) / scale[bi] * observed[bi * rows + i];
                }
            }
        }

        let (base, support) = if self.bandwidth > 0.0 {
            let (kernel, support) =
                kernel_prior(&xn, observed, self.bandwidth, self.bidirectional, track);
            let base = match self.prior {
                Prior::Kernel => kernel,
                Prior::Linear => linear_prior(&xn, observed, self.bidirectional, track),
                Prior::Hybrid => {
                    let linear = linear_prior(&xn, observed, self.bidirectional, track);
                    kernel
                        .iter()
                        .zip(&linear)
                        .enumerate()
                        .map(|(i, (&k_val, &l))| {
                            let seen = observed[i / d];
                            seen * k_val + (1.0 - seen) * l
                        })
                        .collect()
                }
            };
            (base, Some(support))
        } else {
            (xn.clone(), None)
        };

        // `feats = [xn.reshape(b, t, k * d), observed, conf * observed]`, then `base.reshape(b, t, k * d)`
        // and `log(support + 1e-6)` when a prior is active; one row per (batch, frame).
        let mut concat = vec![0.0f32; b * t * self.input_width];
        for frame in 0..b * t {
            let at = frame * self.input_width;
            let seen = frame * k;
            concat[at..at + k * d].copy_from_slice(&xn[seen * d..seen * d + k * d]);
            concat[at + k * d..at + k * (d + 1)].copy_from_slice(&observed[seen..seen + k]);
            for j in 0..k {
                concat[at + k * (d + 1) + j] = conf[seen + j] * observed[seen + j];
            }
            if let Some(support) = &support {
                let prior = at + k * (d + 2);
                concat[prior..prior + k * d].copy_from_slice(&base[seen * d..seen * d + k * d]);
                for j in 0..k {
                    concat[prior + k * d + j] = (support[seen + j] + 1e-6).ln();
                }
            }
        }
        Features {
            concat,
            base,
            center,
            scale,
        }
    }
}

/// The pieces [`TemporalLSTM::forward`] needs before the tape work starts.
struct Features {
    /// `(B, T, input_width)`, the concatenated LSTM input.
    concat: Vec<f32>,
    /// `(B, T, K, D)`, the prior the network corrects (`xn` itself when there is none).
    base: Vec<f32>,
    /// `(B, 1, 1, D)`, the masked centre.
    center: Vec<f32>,
    /// `(B, 1, 1, 1)`, the masked isotropic scale.
    scale: Vec<f32>,
}

impl LayerTree for TemporalLSTM {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        f(&self.inp);
        f(&self.lstm);
        f(&self.head);
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        f(&mut self.inp);
        f(&mut self.lstm);
        f(&mut self.head);
    }
}

/// `out[..., :d].abs() > RESIDUAL_GUARD` for any coordinate, as a `1.0` / `0.0` mask of shape
/// `(B, T, K)` over the head output `(B, T, K, D + 1)`.
fn guard_mask(head: &[f32], d: usize, b: usize, t: usize, k: usize) -> Vec<f32> {
    let rows = b * t * k;
    (0..rows)
        .map(|i| {
            let bad = (0..d).any(|c| head[i * (d + 1) + c].abs() > RESIDUAL_GUARD);
            f32::from(!bad)
        })
        .collect()
}

/// `a.clamp(lo, hi)` built from the tape's ReLU: `min` is `hi - relu(hi - a)`, `max` is
/// `lo + relu(a - lo)`.
fn clamp(g: &mut Graph, a: NodeId, lo: f32, hi: f32) -> NodeId {
    let hi_node = g.constant(hi, &[1]);
    let lo_node = g.constant(lo, &[1]);
    let below = g.sub(hi_node, a);
    let above = g.relu(below);
    let capped = g.sub(hi_node, above);
    let under = g.sub(capped, lo_node);
    let raised = g.relu(under);
    g.add(lo_node, raised)
}

/// `torch.nan_to_num`: NaN becomes 0, the infinities become the largest finite magnitude.
fn finite(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else if v == f32::INFINITY {
        f32::MAX
    } else if v == f32::NEG_INFINITY {
        f32::MIN
    } else {
        v
    }
}
