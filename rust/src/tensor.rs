//! f32 tensors with reverse-mode autodiff, and the layers `animalpose` trains.
//!
//! The Python package under `../../src/animalpose` is the reference implementation; this crate is the
//! piece a dependency-free Rust build needs: a dense `f32` array, a tape, the operations the animalpose
//! networks use (convolution, batch norm, pooling, nearest upsampling, LSTM, the losses), and AdamW with
//! the warmup-plus-cosine schedule the Python trainers use. Layouts match PyTorch (row-major, `NCHW`).
//!
//! The heavy kernels live in `kernels`: convolutions run as im2col + SGEMM (and their gradients as
//! SGEMM + col2im), matrix products and the LSTM projections as SGEMM, all through the pure-Rust
//! `matrixmultiply` crate, with the work spread over the cores by `rayon`. Results match the naive
//! loops up to the summation order of the dot products.
//!
//! A [`crate::tensor::Graph`] is a fresh tape per step: parameters are pushed onto it by value with
//! [`crate::tensor::Graph::param`], the step's gradients are read back with [`crate::tensor::Graph::take_param_grads`], and the graph
//! is then dropped, so no tape outlives a step.
//!
//! ```
//! use animalpose::tensor::Graph;
//! let mut g = Graph::new();
//! let a = g.constant(2.0, &[1, 3]);
//! let b = g.constant(3.0, &[1, 3]);
//! let sum = g.add(a, b);
//! let loss = g.reduce_mean(sum, &[0, 1]);
//! assert_eq!(g.value(loss)[0], 5.0);
//! g.backward(loss).unwrap();
//! ```

mod kernels;
#[cfg(all(feature = "mps", target_os = "macos"))]
pub mod mpsgraph;
mod nn;
mod ops;
mod optim;

// `crate::tensor::Shape` is declared in this module (it is the tape's own shape type), so it is not re-exported from
// `crate::tensor::ops`; `crate::tensor::LossKind` names the pointwise losses the public loss constructors take.
pub use nn::{
    conv_bn_act, BatchNorm, Conv1dLayer, Conv2dLayer, ConvBnAct, ConvBnActOpts, Ctx, Dense,
    Dropout, Flatten, Layer, Lstm, LstmCell, Parallel, Residual, Sequential, Slot,
};
pub use ops::{BnMode, LossKind};

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether inference may run on the GPU (see [`set_gpu`]).
static GPU: AtomicBool = AtomicBool::new(true);

/// `true` when this build can run inference on the GPU: the `mps` feature on macOS with a Metal
/// device present ([`mpsgraph`]).
pub fn gpu_available() -> bool {
    #[cfg(all(feature = "mps", target_os = "macos"))]
    {
        static HAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *HAS.get_or_init(|| objc2_metal::MTLCreateSystemDefaultDevice().is_some())
    }
    #[cfg(not(all(feature = "mps", target_os = "macos")))]
    {
        false
    }
}

/// Allows or forbids the GPU for later forward passes and returns whether it is now in use. The
/// configs' `device:` key maps onto it: `cpu` forbids it, `auto` / `mps` use it when [`gpu_available`].
pub fn set_gpu(on: bool) -> bool {
    GPU.store(on, Ordering::Relaxed);
    gpu_active()
}

/// `true` when inference forward passes run on the GPU.
pub fn gpu_active() -> bool {
    GPU.load(Ordering::Relaxed) && gpu_available()
}
pub use optim::{adamw_step, AdamW, Schedule};

use anyhow::{bail, Result};

/// Index of a value on the tape.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// A dense row-major shape, e.g. `[N, C, H, W]`. Conv layouts follow PyTorch: `NCHW`, and sequences are
/// `NCL` (channels first) like `nn.Conv1d`.
pub type Shape = Vec<usize>;

#[derive(Clone, Debug)]
pub(crate) enum Op {
    Constant,
    Param(usize),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Scale(NodeId, f32),
    Relu(NodeId),
    Sigmoid(NodeId),
    Tanh(NodeId),
    Softmax(NodeId, usize),
    LogSoftmax(NodeId, usize),
    Matmul(NodeId, NodeId),
    Conv2d {
        x: NodeId,
        w: NodeId,
        b: Option<NodeId>,
        stride: usize,
        pad: usize,
        dilation: usize,
    },
    MaxPool2d {
        x: NodeId,
        k: usize,
        stride: usize,
        pad: usize,
    },
    AvgPool2d {
        x: NodeId,
        k: usize,
        stride: usize,
        pad: usize,
    },
    UpsampleNearest {
        x: NodeId,
        h: usize,
        w: usize,
    },
    Reshape(NodeId, Shape),
    Permute(NodeId, Vec<usize>),
    Concat(Vec<NodeId>, usize),
    Slice {
        x: NodeId,
        axis: usize,
        start: usize,
        end: usize,
    },
    ReduceSum(NodeId, Vec<usize>),
    ReduceMean(NodeId, Vec<usize>),
    BatchNorm {
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
        mode: BnMode,
        /// A ReLU fused onto the output (see [`crate::tensor::Graph::batch_norm_relu`]).
        relu: bool,
    },
    L2Normalize(NodeId, usize),
    BceWithLogits {
        logits: NodeId,
        target: NodeId,
    },
    WeightedLoss {
        x: NodeId,
        kind: ops::LossKind,
        weight: f32,
    },
    CrossEntropy {
        logits: NodeId,
        target: NodeId,
        weight: Option<NodeId>,
        smoothing: f32,
        ignore: i32,
    },
    Lstm {
        x: NodeId,
        w: Vec<NodeId>,
        hidden: usize,
        layers: usize,
        bidirectional: bool,
    },
    Dropout {
        x: NodeId,
        p: f32,
        seed: u64,
    },
}

struct Node {
    shape: Shape,
    data: Vec<f32>,
    grad: Vec<f32>,
    op: Op,
}

/// The tape: every forward value with its gradient and the operation that produced it.
#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Graph {
    /// Appends a node; the kernels in [`crate::tensor::ops`] build their `(shape, data)` pair and hand it here.
    pub(super) fn push(&mut self, shape: Shape, data: Vec<f32>, op: Op) -> NodeId {
        debug_assert_eq!(
            data.len(),
            Self::numel(&shape),
            "a kernel returned the wrong number of values"
        );
        let grad = vec![0.0; data.len()];
        self.nodes.push(Node {
            shape,
            data,
            grad,
            op,
        });
        NodeId(self.nodes.len() - 1)
    }

    /// A constant filled with one value.
    pub fn constant(&mut self, v: f32, shape: &[usize]) -> NodeId {
        self.push(shape.to_vec(), vec![v; Self::numel(shape)], Op::Constant)
    }

    /// A constant from data already in hand.
    pub fn from_data(&mut self, data: Vec<f32>, shape: &[usize]) -> NodeId {
        self.push(shape.to_vec(), data, Op::Constant)
    }

    /// A node holding a trainable parameter, identified by `slot` (its index in the model's parameter
    /// list). Gradients come back through [`crate::tensor::Graph::take_param_grads`].
    pub fn param(&mut self, slot: usize, shape: &[usize], data: &[f32]) -> NodeId {
        self.push(shape.to_vec(), data.to_vec(), Op::Param(slot))
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn shape(&self, id: NodeId) -> &Shape {
        &self.nodes[id.index()].shape
    }

    pub fn value(&self, id: NodeId) -> &[f32] {
        &self.nodes[id.index()].data
    }

    pub fn to_vec(&self, id: NodeId) -> Vec<f32> {
        self.value(id).to_vec()
    }

    pub fn gradient(&self, id: NodeId) -> &[f32] {
        &self.nodes[id.index()].grad
    }

    /// Empties a node's gradient buffer (used between the two halves of a two-stage loss).
    pub fn clear_grad(&mut self, id: NodeId) {
        self.nodes[id.index()].grad.fill(0.0);
    }

    /// Takes the gradient accumulated for a node, leaving zeros behind.
    pub fn take_param_grads(&mut self, id: NodeId) -> Vec<f32> {
        std::mem::take(&mut self.nodes[id.index()].grad)
    }

    /// Every trainable parameter pushed onto the tape as `(slot, gradient)`, ordered by slot and summed
    /// over the nodes that share one. Layers push their weights during [`crate::tensor::Layer::forward`], so this is how
    /// a training step reads gradients back out of a graph it did not build itself.
    pub fn param_grads(&self) -> Vec<(usize, Vec<f32>)> {
        let mut found: Vec<(usize, Vec<f32>)> = self
            .nodes
            .iter()
            .filter_map(|node| match node.op {
                Op::Param(slot) => Some((slot, node.grad.clone())),
                _ => None,
            })
            .collect();
        found.sort_by_key(|(slot, _)| *slot);
        let mut out: Vec<(usize, Vec<f32>)> = Vec::with_capacity(found.len());
        for (slot, grad) in found {
            match out.last_mut() {
                Some((last, acc)) if *last == slot => {
                    for (a, b) in acc.iter_mut().zip(&grad) {
                        *a += b;
                    }
                }
                _ => out.push((slot, grad)),
            }
        }
        out
    }

    pub fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    fn add_grad(&mut self, id: NodeId, delta: &[f32]) {
        let n = &mut self.nodes[id.index()];
        kernels::add_assign(&mut n.grad, delta);
    }

    /// Backpropagates from a scalar `loss` (gradient seeded to 1) through the whole tape.
    pub fn backward(&mut self, loss: NodeId) -> Result<()> {
        if Self::numel(self.shape(loss)) != 1 {
            bail!(
                "backward expects a scalar loss, got shape {:?}",
                self.shape(loss)
            );
        }
        let n = self.nodes.len();
        for node in self.nodes.iter_mut() {
            node.grad.fill(0.0);
        }
        self.nodes[loss.index()].grad[0] = 1.0;
        for i in (0..n).rev() {
            // A kernel only writes its parents' gradients (they precede it on the tape), so the
            // node's own buffer can be lent out and put back instead of copied.
            let g = std::mem::take(&mut self.nodes[i].grad);
            if !g.iter().all(|v| *v == 0.0) {
                let op = self.nodes[i].op.clone();
                ops::backward(self, i, &op, &g);
            }
            self.nodes[i].grad = g;
        }
        Ok(())
    }

    /// Sum of a node's values: a quick way to build a scalar loss from a per-element one.
    pub fn sum_all(&mut self, a: NodeId) -> NodeId {
        let axes: Vec<usize> = (0..self.shape(a).len()).collect();
        self.reduce_sum(a, &axes)
    }
}

/// Forward operations; every one records the tape, so [`crate::tensor::Graph::backward`] follows.
impl Graph {
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        ops::elementwise2(self, a, b, Op::Add(a, b))
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        ops::elementwise2(self, a, b, Op::Sub(a, b))
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        ops::elementwise2(self, a, b, Op::Mul(a, b))
    }

    pub fn scale(&mut self, a: NodeId, k: f32) -> NodeId {
        let data = kernels::map(self.value(a), |x| x * k);
        let shape = self.shape(a).clone();
        self.push(shape, data, Op::Scale(a, k))
    }

    pub fn relu(&mut self, a: NodeId) -> NodeId {
        let data = kernels::map(self.value(a), |x| x.max(0.0));
        let shape = self.shape(a).clone();
        self.push(shape, data, Op::Relu(a))
    }

    pub fn sigmoid(&mut self, a: NodeId) -> NodeId {
        let data = kernels::map(self.value(a), |x| 1.0 / (1.0 + (-x).exp()));
        let shape = self.shape(a).clone();
        self.push(shape, data, Op::Sigmoid(a))
    }

    pub fn tanh(&mut self, a: NodeId) -> NodeId {
        let data = kernels::map(self.value(a), |x| x.tanh());
        let shape = self.shape(a).clone();
        self.push(shape, data, Op::Tanh(a))
    }

    pub fn softmax(&mut self, a: NodeId, axis: usize) -> NodeId {
        let (shape, data) = ops::softmax_data(self, a, axis);
        self.push(shape, data, Op::Softmax(a, axis))
    }

    pub fn log_softmax(&mut self, a: NodeId, axis: usize) -> NodeId {
        let (shape, p) = ops::softmax_data(self, a, axis);
        let data: Vec<f32> = p.iter().map(|v| v.max(1e-20).ln()).collect();
        self.push(shape, data, Op::LogSoftmax(a, axis))
    }

    /// Batched matrix product of `(..., M, K)` by `(..., K, N)`.
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let (shape, data) = ops::matmul_data(self, a, b);
        self.push(shape, data, Op::Matmul(a, b))
    }

    /// `conv2d(x (N,C,H,W), w (O,C,kH,kW), bias?)` with square stride/padding/dilation.
    pub fn conv2d(
        &mut self,
        x: NodeId,
        w: NodeId,
        b: Option<NodeId>,
        stride: usize,
        pad: usize,
        dilation: usize,
    ) -> NodeId {
        let (shape, data) = ops::conv2d_data(self, x, w, b, stride, pad, dilation);
        self.push(
            shape,
            data,
            Op::Conv2d {
                x,
                w,
                b,
                stride,
                pad,
                dilation,
            },
        )
    }

    /// 1D convolution over `x (N,C,L)` with a kernel of `k`, dilation `d` and padding `d * (k - 1) / 2`
    /// (the causal-free `padding = dilation * (k - 1) / 2` PyTorch convention). Implemented as a 2D
    /// convolution with a `(1, k)` kernel so the backward pass is shared.
    ///
    /// `conv2d` pads both spatial axes, so the padding is materialised on the time axis by
    /// [`crate::tensor::Graph::concat`]enating zero columns: the kernel axis stays unpadded and the output keeps the
    /// input's length, which is what `nn.Conv1d`'s `padding` does.
    pub fn conv1d(
        &mut self,
        x: NodeId,
        w: NodeId,
        b: Option<NodeId>,
        k: usize,
        dilation: usize,
    ) -> NodeId {
        let (n, c, l) = {
            let xs = self.shape(x);
            (xs[0], xs[1], xs[2])
        };
        let pad = dilation * (k - 1) / 2;
        let x = if pad == 0 {
            x
        } else {
            let zeros = self.constant(0.0, &[n, c, pad]);
            self.concat(&[zeros, x, zeros], 2)
        };
        let x4 = self.reshape(x, &[n, c, 1, l + 2 * pad]);
        let ws = self.shape(w).clone();
        let w4 = self.reshape(w, &[ws[0], ws[1], 1, ws[2]]);
        let out = self.conv2d(x4, w4, b, 1, 0, dilation);
        let o = self.shape(out).clone();
        self.reshape(out, &[o[0], o[1], o[3]])
    }

    pub fn max_pool2d(&mut self, x: NodeId, k: usize, stride: usize, pad: usize) -> NodeId {
        let (shape, data) = ops::pool_data(self, x, k, stride, pad, ops::ReduceKind::Max);
        self.push(shape, data, Op::MaxPool2d { x, k, stride, pad })
    }

    pub fn avg_pool2d(&mut self, x: NodeId, k: usize, stride: usize, pad: usize) -> NodeId {
        let (shape, data) = ops::pool_data(self, x, k, stride, pad, ops::ReduceKind::Mean);
        self.push(shape, data, Op::AvgPool2d { x, k, stride, pad })
    }

    /// Nearest-neighbour upsampling to an absolute `(h, w)`; matches `F.interpolate(..., mode="nearest")`.
    pub fn upsample_nearest(&mut self, x: NodeId, h: usize, w: usize) -> NodeId {
        let (shape, data) = ops::upsample_nearest_data(self, x, h, w);
        self.push(shape, data, Op::UpsampleNearest { x, h, w })
    }

    pub fn reshape(&mut self, a: NodeId, shape: &[usize]) -> NodeId {
        let data = self.value(a).to_vec();
        self.push(shape.to_vec(), data, Op::Reshape(a, shape.to_vec()))
    }

    pub fn flatten(&mut self, a: NodeId, start_dim: usize) -> NodeId {
        let s = self.shape(a).clone();
        let mut shape = s[..start_dim].to_vec();
        shape.push(s[start_dim..].iter().product());
        self.reshape(a, &shape)
    }

    pub fn permute(&mut self, a: NodeId, perm: &[usize]) -> NodeId {
        let (shape, data) = ops::permute_data(self, a, perm);
        self.push(shape, data, Op::Permute(a, perm.to_vec()))
    }

    pub fn concat(&mut self, parts: &[NodeId], axis: usize) -> NodeId {
        let (shape, data) = ops::concat_data(self, parts, axis);
        self.push(shape, data, Op::Concat(parts.to_vec(), axis))
    }

    pub fn slice(&mut self, a: NodeId, axis: usize, start: usize, end: usize) -> NodeId {
        let (shape, data) = ops::slice_data(self, a, axis, start, end);
        self.push(
            shape,
            data,
            Op::Slice {
                x: a,
                axis,
                start,
                end,
            },
        )
    }

    pub fn reduce_sum(&mut self, a: NodeId, axes: &[usize]) -> NodeId {
        let (shape, data) = ops::reduce_data(self, a, axes, ops::ReduceKind::Sum);
        self.push(shape, data, Op::ReduceSum(a, axes.to_vec()))
    }

    pub fn reduce_mean(&mut self, a: NodeId, axes: &[usize]) -> NodeId {
        let (shape, data) = ops::reduce_data(self, a, axes, ops::ReduceKind::Mean);
        self.push(shape, data, Op::ReduceMean(a, axes.to_vec()))
    }

    /// Channel-wise batch norm. `x` is `(N,C,*)`; `gamma`/`beta` are `(C,)`. In
    /// [`crate::tensor::BnMode::Train`] the batch statistics are used (and the running statistics of the calling layer
    /// should be updated with [`crate::tensor::ops::update_running`]); in [`crate::tensor::BnMode::Eval`] the running statistics are.
    pub fn batch_norm(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
        mode: BnMode,
    ) -> NodeId {
        self.batch_norm_act(x, gamma, beta, eps, mode, false)
    }

    /// `relu(batch_norm(x))` as one node: the same values and gradients as
    /// [`crate::tensor::Graph::batch_norm`] followed by [`crate::tensor::Graph::relu`], in one pass
    /// over the activations instead of two (`ConvBNAct` is conv, norm, ReLU).
    pub fn batch_norm_relu(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
        mode: BnMode,
    ) -> NodeId {
        self.batch_norm_act(x, gamma, beta, eps, mode, true)
    }

    fn batch_norm_act(
        &mut self,
        x: NodeId,
        gamma: NodeId,
        beta: NodeId,
        eps: f32,
        mode: BnMode,
        relu: bool,
    ) -> NodeId {
        let (shape, data) = ops::batch_norm_data(self, x, gamma, beta, eps, &mode, relu);
        self.push(
            shape,
            data,
            Op::BatchNorm {
                x,
                gamma,
                beta,
                eps,
                mode,
                relu,
            },
        )
    }

    /// `F.normalize(x, dim=axis)`: divides by the L2 norm along `axis`.
    pub fn l2_normalize(&mut self, a: NodeId, axis: usize) -> NodeId {
        let (shape, data) = ops::l2_normalize_data(self, a, axis);
        self.push(shape, data, Op::L2Normalize(a, axis))
    }

    /// Dropout with a fixed seed, so a training run reproduces itself.
    pub fn dropout(&mut self, a: NodeId, p: f32, seed: u64) -> NodeId {
        if p <= 0.0 {
            return a;
        }
        let (shape, data) = ops::dropout_data(self, a, p, seed);
        self.push(shape, data, Op::Dropout { x: a, p, seed })
    }

    /// Mean `nn.BCEWithLogitsLoss` over a `(N, C, ...)` logit/target pair.
    pub fn bce_with_logits(&mut self, logits: NodeId, target: NodeId) -> NodeId {
        let (_, data) = ops::bce_data(self, logits, target);
        self.push(vec![1], data, Op::BceWithLogits { logits, target })
    }

    /// `weight` times the mean of a per-element loss; `x` is `diff` for L1/SmoothL1 and for MSE too.
    pub fn weighted_loss(&mut self, diff: NodeId, kind: ops::LossKind, weight: f32) -> NodeId {
        let (_, data) = ops::weighted_loss_data(self, diff, kind, weight);
        self.push(
            vec![1],
            data,
            Op::WeightedLoss {
                x: diff,
                kind,
                weight,
            },
        )
    }

    /// Cross entropy over `logits` `(N, C)` or `(N, T, C)` with class indices in `target`
    /// (`ignore` selects the frames to skip, `smoothing` is `nn.CrossEntropyLoss`'s label smoothing).
    pub fn cross_entropy(
        &mut self,
        logits: NodeId,
        target: NodeId,
        weight: Option<NodeId>,
        smoothing: f32,
        ignore: i32,
    ) -> NodeId {
        let (_, data) = ops::cross_entropy_data(self, logits, target, weight, smoothing, ignore);
        self.push(
            vec![1],
            data,
            Op::CrossEntropy {
                logits,
                target,
                weight,
                smoothing,
                ignore,
            },
        )
    }

    /// Multi-layer, optionally bidirectional LSTM over `x (N, T, input)`, returning the output
    /// `(N, T, hidden * directions)` (the final hidden state is not put on the tape, because the Python
    /// models only read `output`). Weights are the `(4*H, in)` / `(4*H, H)` / `(4*H,)` blocks in
    /// PyTorch's `weight_ih_l{layer}[R]`, `weight_hh_l{layer}[R]`, `bias_ih_l{layer}[R]`,
    /// `bias_hh_l{layer}[R]` order, forward directions first: `4 * directions` blocks per layer.
    ///
    /// Mirrors `torch.nn.LSTM(..., batch_first=True)`. `nn.LSTM`'s inter-layer `dropout` is not part of
    /// this operation; the [`crate::tensor::Lstm`] layer applies [`crate::tensor::Graph::dropout`] between per-layer calls.
    pub fn lstm(
        &mut self,
        x: NodeId,
        weights: &[NodeId],
        hidden: usize,
        layers: usize,
        bidirectional: bool,
    ) -> NodeId {
        let (shape, data) = ops::lstm_data(self, x, weights, hidden, layers, bidirectional);
        self.push(
            shape,
            data,
            Op::Lstm {
                x,
                w: weights.to_vec(),
                hidden,
                layers,
                bidirectional,
            },
        )
    }
}

#[cfg(test)]
mod tests;
