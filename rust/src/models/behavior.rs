//! `src/animalpose/behavior/model.py`: frame-wise behaviour classification from pose sequences.
//!
//! [`EgocentricFeatures`] turns raw keypoints into view- and scale-robust features inside the model
//! (coordinates centred on the animal, rotated so the body axis points along `+x`, scaled by the mean
//! body length, plus per-frame length, centroid velocity, turning rate and per-keypoint speeds);
//! [`BehaviorTCN`] then runs a `1x1` projection and a dilated [`TcnBlock`] stack to per-frame logits.
//!
//! Checkpoint naming:
//!
//! | Python module | checkpoint key | Rust construction |
//! | --- | --- | --- |
//! | `EgocentricFeatures` buffers | `features.ant`, `features.post` | [`Params::buffer`] |
//! | `BehaviorTCN.inp` (`nn.Conv1d(feat, hidden, 1)`) | `inp.weight` `(hidden, feat, 1)`, `inp.bias` | [`Conv1dLayer`] |
//! | `TCNBlock.conv1` (`Conv1d(c, c, 3, padding=d, dilation=d)`) | `blocks.{i}.conv1.weight` `(c, c, 3)`, `.bias` | [`Conv1dLayer`] |
//! | `TCNBlock.norm` (`nn.BatchNorm1d`) | `blocks.{i}.norm.weight` / `.bias` / `.running_*` | [`BatchNorm`] |
//! | `TCNBlock.conv2` (`Conv1d(c, c, 1)`) | `blocks.{i}.conv2.weight` `(c, c, 1)`, `.bias` | [`Conv1dLayer`] |
//! | `BehaviorTCN.out` (`nn.Conv1d(hidden, classes, 1)`) | `out.weight` `(classes, hidden, 1)`, `out.bias` | [`Conv1dLayer`] |
//!
//! The features are computed from the input values rather than on the tape, which is exact for the
//! forward pass: the module owns no parameters and the Python trainers never ask for a gradient with
//! respect to the keypoint input (`// differs from Python:` at [`EgocentricFeatures::compute`]).

use crate::models::config::ModelConfig;
use crate::models::layers::{LayerTree, LoadReport, Params, Relu};
use crate::models::weights::Weights;
use crate::tensor::{BatchNorm, Conv1dLayer, Ctx, Dropout, Graph, Layer, NodeId, Sequential};
use anyhow::{ensure, Result};

/// The `CANONICAL.anterior` keypoint indices of `src/animalpose/skeleton.py`.
pub const CANONICAL_ANTERIOR: [usize; 4] = [0, 5, 6, 7];

/// The `CANONICAL.posterior` keypoint indices of `src/animalpose/skeleton.py`.
pub const CANONICAL_POSTERIOR: [usize; 4] = [10, 9, 19, 20];

/// The `eps` of `EgocentricFeatures.forward`.
pub const EPS: f32 = 1e-6;

/// `BehaviorTCN`'s default `hidden`, i.e. `BUILDERS`' `c.get("hidden", 96)` for this kind.
pub const BEHAVIOR_HIDDEN: usize = 96;

/// The inputs of `BehaviorTCN.forward`: `(B, T, K, D)` coordinates and `(B, T, K)` visibility.
pub const INPUTS: [&str; 2] = ["coords", "mask"];

/// The output name of `BehaviorTCN.forward` (the Python model returns one tensor, the per-frame
/// logits).
pub const OUTPUTS: [&str; 1] = ["logits"];

/// `EgocentricFeatures` from `behavior/model.py`: raw keypoints to view- and scale-robust features.
pub struct EgocentricFeatures {
    ant: Params,
    post: Params,
    k: usize,
    d: usize,
}

impl EgocentricFeatures {
    /// `EgocentricFeatures(num_keypoints, dims)`, with the `CANONICAL` anterior / posterior groups.
    ///
    /// `// differs from Python:` for `k != 25` the groups start empty rather than indexing the
    /// canonical vocabulary (whose indices need 21 keypoints); a checkpoint's `features.ant` /
    /// `features.post` buffers carry the real groups either way.
    pub fn new(k: usize, d: usize) -> Self {
        let group = |indices: &[usize]| -> Vec<f32> {
            (0..k)
                .map(|i| usize::from(indices.contains(&i)) as f32)
                .collect()
        };
        let ant = if k == 25 {
            group(&CANONICAL_ANTERIOR)
        } else {
            vec![0.0f32; k]
        };
        let post = if k == 25 {
            group(&CANONICAL_POSTERIOR)
        } else {
            vec![0.0f32; k]
        };
        Self {
            ant: Params::buffer("features.ant", &[k], ant),
            post: Params::buffer("features.post", &[k], post),
            k,
            d,
        }
    }

    /// `EgocentricFeatures.out_dim`, the feature width `k * d + 2 * k + 6`.
    pub fn out_dim(&self) -> usize {
        self.k * self.d + 2 * self.k + 6
    }

    /// Keypoint count `k`.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Coordinate width `d`.
    pub fn d(&self) -> usize {
        self.d
    }

    /// `EgocentricFeatures.forward(x, mask)`: row-major `(b, t, k, d)` coordinates and `(b, t, k)`
    /// visibility in, `(b, t, out_dim)` features out.
    ///
    /// `// differs from Python:` evaluated over plain values instead of tape nodes. Nothing trainable is
    /// involved and the keypoint input never needs a gradient, so the outputs are the same numbers.
    pub fn compute(
        &self,
        x: &[f32],
        mask: &[f32],
        b: usize,
        t: usize,
        k: usize,
        d: usize,
    ) -> Vec<f32> {
        let rows = b * t;
        let kd = k * d;
        debug_assert_eq!(x.len(), rows * kd);
        debug_assert_eq!(mask.len(), rows * k);

        // `x = torch.nan_to_num(x) * m.unsqueeze(-1)`
        let xs: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| finite(v) * mask[i / d])
            .collect();

        // The masked centroid and the two group centroids give the body axis.
        let mut centroid = vec![0.0f32; rows * d];
        let mut axis = vec![0.0f32; rows * d];
        for i in 0..rows {
            let row = &xs[i * kd..(i + 1) * kd];
            let vis = &mask[i * k..(i + 1) * k];
            let center = self.centroid(row, vis, d);
            let anterior = self.group_centroid(row, vis, self.ant.values(), d);
            let posterior = self.group_centroid(row, vis, self.post.values(), d);
            centroid[i * d..(i + 1) * d].copy_from_slice(&center);
            for c in 0..d {
                axis[i * d + c] = anterior[c] - posterior[c];
            }
        }

        // `length` (the apparent body length), the unit axis `u`, and the window scale `ref`.
        let mut length = vec![0.0f32; rows];
        let mut u = vec![0.0f32; rows * 2];
        for i in 0..rows {
            let len = (axis[i * d] * axis[i * d] + axis[i * d + 1] * axis[i * d + 1] + EPS).sqrt();
            length[i] = len;
            u[i * 2] = axis[i * d] / len;
            u[i * 2 + 1] = axis[i * d + 1] / len;
        }
        let reference: Vec<f32> = (0..b)
            .map(|i| (length[i * t..(i + 1) * t].iter().sum::<f32>() / t as f32).max(EPS))
            .collect();

        // The egocentric coordinates: centred on the animal, body axis along +x, divided by `ref`.
        let mut ego = vec![0.0f32; rows * kd];
        for i in 0..rows {
            let (ux, uy) = (u[i * 2], u[i * 2 + 1]);
            let scale = reference[i / t];
            for j in 0..k {
                let delta: Vec<f32> = (0..d)
                    .map(|c| xs[i * kd + j * d + c] - centroid[i * d + c])
                    .collect();
                let seen = mask[i * k + j];
                for c in 0..d {
                    let v = match c {
                        0 => delta[0] * ux + delta[1] * uy,
                        1 => -delta[0] * uy + delta[1] * ux,
                        _ => delta[c],
                    };
                    ego[i * kd + j * d + c] = v / scale * seen;
                }
            }
        }

        // Centroid velocity in body coordinates, turning, per-keypoint speed.
        let mut vx = vec![0.0f32; rows];
        let mut vy = vec![0.0f32; rows];
        let mut turn_sin = vec![0.0f32; rows];
        let mut turn_cos = vec![0.0f32; rows];
        let mut kp_speed = vec![0.0f32; rows * k];
        for i in 0..rows {
            let prev = if i % t == 0 { i } else { i - 1 }; // `_shift`, along the time axis
            let (ux, uy) = (u[i * 2], u[i * 2 + 1]);
            let v0 = centroid[i * d] - centroid[prev * d];
            let v1 = centroid[i * d + 1] - centroid[prev * d + 1];
            vx[i] = v0 * ux + v1 * uy;
            vy[i] = -v0 * uy + v1 * ux;
            let (pu, pv) = (u[prev * 2], u[prev * 2 + 1]);
            turn_sin[i] = pu * uy - pv * ux;
            turn_cos[i] = pu * ux + pv * uy;
            for j in 0..k {
                let mut acc = 0.0f32;
                for c in 0..d {
                    let step = ego[i * kd + j * d + c] - ego[prev * kd + j * d + c];
                    acc += step * step;
                }
                kp_speed[i * k + j] = (acc + EPS).sqrt() * mask[i * k + j] * mask[prev * k + j];
            }
        }

        // `cat([ego.reshape(b, t, -1), m, kp_speed, vx / ref, vy / ref, speed, length / ref,
        //       turn_sin, turn_cos])`
        let width = self.out_dim();
        let mut out = vec![0.0f32; rows * width];
        for i in 0..rows {
            let scale = reference[i / t];
            let speed = (vx[i] * vx[i] + vy[i] * vy[i] + EPS).sqrt() / scale;
            let row = &mut out[i * width..(i + 1) * width];
            row[..kd].copy_from_slice(&ego[i * kd..(i + 1) * kd]);
            row[kd..kd + k].copy_from_slice(&mask[i * k..(i + 1) * k]);
            row[kd + k..kd + 2 * k].copy_from_slice(&kp_speed[i * k..(i + 1) * k]);
            let tail = kd + 2 * k;
            row[tail] = vx[i] / scale;
            row[tail + 1] = vy[i] / scale;
            row[tail + 2] = speed;
            row[tail + 3] = length[i] / scale;
            row[tail + 4] = turn_sin[i];
            row[tail + 5] = turn_cos[i];
        }
        out
    }

    /// `group(w)`: the masked mean of the keypoints weighted by the group indicator `w`.
    fn group_centroid(&self, xs: &[f32], m: &[f32], w: &[f32], d: usize) -> Vec<f32> {
        let mut acc = vec![0.0f32; d];
        let mut weight = 0.0f32;
        for (j, &seen) in m.iter().enumerate() {
            let ww = seen * w[j];
            weight += ww;
            for c in 0..d {
                acc[c] += xs[j * d + c] * ww;
            }
        }
        let weight = weight.max(EPS);
        for total in acc.iter_mut() {
            *total /= weight;
        }
        acc
    }

    /// The masked centroid of one frame, `(d,)`: `group` with an all-ones weight.
    fn centroid(&self, xs: &[f32], m: &[f32], d: usize) -> Vec<f32> {
        self.group_centroid(xs, m, &vec![1.0f32; m.len()], d)
    }
}

impl LayerTree for EgocentricFeatures {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        f(&self.ant);
        f(&self.post);
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        f(&mut self.ant);
        f(&mut self.post);
    }
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

/// `TCNBlock` from `behavior/model.py`: `x + drop(conv2(relu(norm(conv1(x)))))`.
pub struct TcnBlock {
    main: Sequential,
}

impl TcnBlock {
    /// `TCNBlock(c, dilation, dropout)`, named `blocks.{index}`.
    pub fn new(index: usize, c: usize, dilation: usize, dropout: f32) -> Self {
        let prefix = format!("blocks.{index}");
        Self {
            main: Sequential::of(vec![
                Box::new(Conv1dLayer::new(
                    &format!("{prefix}.conv1"),
                    c,
                    c,
                    3,
                    dilation,
                    true,
                )),
                Box::new(BatchNorm::python(&format!("{prefix}.norm"), c)),
                Box::new(Relu),
                Box::new(Conv1dLayer::new(
                    &format!("{prefix}.conv2"),
                    c,
                    c,
                    1,
                    1,
                    true,
                )),
                Box::new(Dropout::new(dropout)),
            ]),
        }
    }
}

impl Layer for TcnBlock {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        let h = self.main.forward(g, x, ctx);
        g.add(x, h)
    }

    fn params(&self) -> &[String] {
        self.main.params()
    }

    fn visit(&self, f: &mut dyn FnMut(&crate::tensor::Slot)) {
        self.main.visit(f);
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut crate::tensor::Slot)) {
        self.main.visit_mut(f);
    }
}

/// `BehaviorTCN` from `behavior/model.py`: the feature block, a `1x1` projection, dilated [`TcnBlock`]s
/// and a `1x1` classifier, so it runs on any sequence length.
pub struct BehaviorTCN {
    features: EgocentricFeatures,
    inp: Conv1dLayer,
    blocks: Vec<TcnBlock>,
    out: Conv1dLayer,
    classes: Vec<String>,
}

impl BehaviorTCN {
    /// `BehaviorTCN(num_keypoints, dims, classes, hidden, dilations, dropout)`.
    ///
    /// `dilations` comes from [`ModelConfig::dilations`], which is the default `(1, 2, 4, 8, 16)` for
    /// every checkpoint: `build_model` does not forward the key, so the stack a state dict was built
    /// with is the one this constructor reproduces.
    pub fn new(cfg: &ModelConfig) -> Self {
        let hidden = cfg.hidden_or(BEHAVIOR_HIDDEN);
        let features = EgocentricFeatures::new(cfg.num_keypoints(), cfg.dims());
        let classes = cfg.classes();
        Self {
            inp: Conv1dLayer::new("inp", features.out_dim(), hidden, 1, 1, true),
            blocks: cfg
                .dilations()
                .iter()
                .enumerate()
                .map(|(i, &dil)| TcnBlock::new(i, hidden, dil, cfg.dropout()))
                .collect(),
            out: Conv1dLayer::new("out", hidden, classes.len(), 1, 1, true),
            features,
            classes,
        }
    }

    /// `BehaviorTCN.classes`, the frame-wise label vocabulary.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// The feature block, whose `features.ant` / `features.post` buffers a checkpoint carries.
    pub fn features(&self) -> &EgocentricFeatures {
        &self.features
    }

    /// `BehaviorTCN.forward(x, mask)`: `(B, T, K, D)` coordinates and `(B, T, K)` visibility in,
    /// `(B, T, C)` logits out.
    pub fn forward(&self, g: &mut Graph, x: NodeId, mask: NodeId, ctx: &mut Ctx) -> Result<NodeId> {
        let shape = g.shape(x).clone();
        let k = self.features.k();
        let d = self.features.d();
        ensure!(
            shape.len() == 4 && shape[2] == k && shape[3] == d,
            "behavior input must be (B, T, {k}, {d}), got {shape:?}"
        );
        let mask_shape = g.shape(mask).clone();
        ensure!(
            mask_shape == vec![shape[0], shape[1], k],
            "the mask must be (B, T, {k}), got {mask_shape:?}"
        );
        let coords = g.value(x).to_vec();
        let visibility = g.value(mask).to_vec();
        let feats = self
            .features
            .compute(&coords, &visibility, shape[0], shape[1], k, d);
        let width = self.features.out_dim();
        let feats = g.from_data(feats, &[shape[0], shape[1], width]);
        // `self.features(x, mask).transpose(1, 2)`: the convolution stack wants channels first.
        let channels_first = g.permute(feats, &[0, 2, 1]);
        let mut h = self.inp.forward(g, channels_first, ctx);
        for block in &self.blocks {
            h = block.forward(g, h, ctx);
        }
        let h = self.out.forward(g, h, ctx);
        Ok(g.permute(h, &[0, 2, 1]))
    }

    /// `model.load_state_dict(...)`: copies every matching tensor of `weights` in.
    pub fn load_weights(&mut self, weights: &Weights) -> LoadReport {
        crate::models::layers::load_into(self, weights)
    }
}

impl LayerTree for BehaviorTCN {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        self.features.each_layer(f);
        f(&self.inp);
        for block in &self.blocks {
            f(block);
        }
        f(&self.out);
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        self.features.each_layer_mut(f);
        f(&mut self.inp);
        for block in &mut self.blocks {
            f(block);
        }
        f(&mut self.out);
    }
}
