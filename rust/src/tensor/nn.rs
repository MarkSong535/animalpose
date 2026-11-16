//! The layer objects the models crate assembles into the animalpose networks.
//!
//! A layer owns its tensors *by checkpoint name*, so a state dict drops straight in: the names are the
//! Python module paths of `src/animalpose/models/backbone.py`, `pose/model.py`, `detector/model.py`,
//! `temporal/model.py` and `behavior/model.py`. Nothing is stored on the graph between steps: each
//! forward pushes the layer's own copies with [`Graph::param`], tagged with the parameter's slot (its
//! index in the model's flat parameter list, assigned by [`Layer::assign_slots`]), and the step's
//! gradients come back through [`Graph::param_grads`].
//!
//! The name mapping (a layer named `p` below produces exactly these keys):
//!
//! | Rust layer | Python state-dict keys |
//! | --- | --- |
//! | [`conv_bn_act("p", ..)`] | `p.0.weight` (conv, `bias=False`), `p.1.weight`, `p.1.bias`, `p.1.running_mean`, `p.1.running_var` |
//! | [`Conv2dLayer::new("p", ..)`] | `p.weight`, `p.bias` |
//! | [`Conv1dLayer::new("p", ..)`] | `p.weight`, `p.bias` |
//! | [`Dense::new("p", ..)`] | `p.weight` (`out, in`), `p.bias` |
//! | [`BatchNorm::new("p", ..)`] | `p.weight`, `p.bias`, `p.running_mean`, `p.running_var` |
//! | [`Lstm::new("p", ..)`] | `p.weight_ih_l{L}`, `p.weight_hh_l{L}`, `p.bias_ih_l{L}`, `p.bias_hh_l{L}`, then the same four with a `_reverse` suffix, per layer |
//! | [`Sequential`] / [`Residual`] / [`Parallel`] | the children's keys, in child order |
//!
//! So `ConvBNAct(3, 32, 3, 2)` inside `TinyResNet.stages[0][0]` is
//! `conv_bn_act("stages.0.0", 3, 32, 3, 2, 1, ..)`, whose keys are `stages.0.0.0.weight` and
//! `stages.0.0.1.weight` / `.bias` / `.running_mean` / `.running_var`, exactly like the Python
//! checkpoint. `num_batches_tracked` is not modelled: the Rust batch norm always uses a fixed momentum,
//! like PyTorch's default.
//!
//! ```
//! use animalpose::tensor::{conv_bn_act, ConvBnActOpts, Ctx, Graph, Layer};
//!
//! let layer = conv_bn_act("stages.0.0", 3, 4, 3, 1, 1, ConvBnActOpts::default());
//! assert_eq!(
//!     layer.params().join(","),
//!     "stages.0.0.0.weight,stages.0.0.1.weight,stages.0.0.1.bias"
//! );
//! let mut g = Graph::new();
//! let x = g.constant(0.5, &[1, 3, 8, 8]);
//! let y = layer.forward(&mut g, x, &mut Ctx::new(true, 0));
//! assert_eq!(g.shape(y).as_slice(), [1, 4, 8, 8]);
//! ```

use crate::tensor::ops::{self, SplitMix};
use crate::tensor::{BnMode, Graph, NodeId, Shape};
use std::cell::RefCell;

/// How a forward pass behaves, mirroring `nn.Module.train()` / `nn.Module.eval()`: batch norm uses the
/// batch statistics and updates its running buffers while training, dropout is active, and both use the
/// stored values while evaluating.
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    /// `true` while training.
    pub train: bool,
    /// Seed for stochastic layers, so a training run reproduces itself.
    pub seed: u64,
}

impl Ctx {
    /// A context for the given mode.
    pub fn new(train: bool, seed: u64) -> Self {
        Self { train, seed }
    }

    /// An evaluation context (`nn.Module.eval()`); dropout is off and batch norm uses running statistics.
    pub fn eval() -> Self {
        Self {
            train: false,
            seed: 0,
        }
    }

    /// A training context (`nn.Module.train()`).
    pub fn train(seed: u64) -> Self {
        Self { train: true, seed }
    }
}

/// One named tensor owned by a layer: a trainable parameter or a buffer such as a batch-norm running
/// statistic.
#[derive(Clone, Debug)]
pub struct Slot {
    /// Checkpoint key, e.g. `stages.0.0.1.running_mean`.
    pub name: String,
    /// Shape in PyTorch order: a conv weight is `(out, in, kH, kW)`, a linear weight `(out, in)`.
    pub shape: Shape,
    /// Row-major values.
    pub data: Vec<f32>,
    /// `true` for parameters the optimiser updates, `false` for buffers.
    pub trainable: bool,
    /// Index in the model's flat parameter list, set by [`Layer::assign_slots`].
    pub slot: usize,
}

impl Slot {
    /// Pushes the parameter onto `g` as a trainable node tagged with its slot.
    pub(crate) fn push(&self, g: &mut Graph) -> NodeId {
        g.param(self.slot, &self.shape, &self.data)
    }
}

/// A layer of the network: it runs on a tape node and knows its own parameters by name.
pub trait Layer {
    /// Runs the layer on `x`, pushing its parameters onto the tape first.
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId;

    /// The names of the trainable parameters, in slot order; for a composite this is its children's
    /// list, concatenated, which is exactly the order [`Layer::assign_slots`] numbers.
    fn params(&self) -> &[String];

    /// Calls `f` for every slot this layer (or its children) owns: parameters first, then buffers.
    fn visit(&self, f: &mut dyn FnMut(&Slot));

    /// Mutable counterpart of [`Layer::visit`].
    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot));

    /// Names of the non-trainable checkpoint entries, i.e. the batch-norm running statistics.
    fn buffers(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.visit(&mut |s| {
            if !s.trainable {
                names.push(s.name.clone());
            }
        });
        names
    }

    /// Numbers every trainable slot from `base` in [`Layer::params`] order; returns the next free index.
    fn assign_slots(&mut self, base: usize) -> usize {
        let mut next = base;
        self.visit_mut(&mut |s| {
            if s.trainable {
                s.slot = next;
                next += 1;
            }
        });
        next
    }

    /// Copies checkpoint `data` into the slot with this name; `false` when the name is unknown or the
    /// number of values does not match.
    fn load(&mut self, name: &str, data: &[f32]) -> bool {
        let mut hit = false;
        self.visit_mut(&mut |s| {
            if s.name == name && s.data.len() == data.len() {
                s.data.copy_from_slice(data);
                hit = true;
            }
        });
        hit
    }

    /// Values of the named slot, for dumping a checkpoint back out.
    fn values(&self, name: &str) -> Option<Vec<f32>> {
        let mut found = None;
        self.visit(&mut |s| {
            if s.name == name {
                found = Some(s.data.clone());
            }
        });
        found
    }

    /// `(name, shape)` of every trainable slot, for building an optimiser's parameter list.
    fn shapes(&self) -> Vec<(String, Shape)> {
        let mut out = Vec::new();
        self.visit(&mut |s| {
            if s.trainable {
                out.push((s.name.clone(), s.shape.clone()));
            }
        });
        out
    }
}

/// Slots owned by a leaf layer: parameters and buffers in one list, borrow-checked so a training
/// forward pass may update the running statistics through `&self`.
struct Slots(RefCell<Vec<Slot>>);

impl Slots {
    fn new(slots: Vec<Slot>) -> Self {
        Self(RefCell::new(slots))
    }

    /// Pushes slot `i` onto the tape.
    fn push(&self, g: &mut Graph, i: usize) -> NodeId {
        self.0.borrow()[i].push(g)
    }

    /// Values of slot `i`, cloned out of the borrow.
    fn data(&self, i: usize) -> Vec<f32> {
        self.0.borrow()[i].data.clone()
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        let held = self.0.borrow();
        for s in held.iter() {
            f(s);
        }
    }

    fn visit_mut(&self, f: &mut dyn FnMut(&mut Slot)) {
        let mut held = self.0.borrow_mut();
        for s in held.iter_mut() {
            f(s);
        }
    }

    /// The trainable names, in slot order.
    fn names(&self) -> Vec<String> {
        self.0
            .borrow()
            .iter()
            .filter(|s| s.trainable)
            .map(|s| s.name.clone())
            .collect()
    }
}

/// A slot filled with one value, e.g. `nn.BatchNorm`'s `gamma = 1` / `beta = 0`.
fn filled(name: &str, shape: &[usize], value: f32, trainable: bool) -> Slot {
    Slot {
        name: name.to_string(),
        shape: shape.to_vec(),
        data: vec![value; Graph::numel(shape)],
        trainable,
        slot: 0,
    }
}

/// A slot filled with `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, the bound PyTorch uses for both
/// `nn.Linear` (`init.kaiming_uniform_` with `a = sqrt(5)`) and `nn.Conv2d`.
///
/// `// differs from Python:` the numbers come from [`SplitMix`] seeded by the slot name, not from
/// PyTorch's RNG; loading a checkpoint replaces them, and the shapes and bounds are identical.
fn uniform(name: &str, shape: &[usize], fan_in: usize) -> Slot {
    let limit = 1.0 / (fan_in.max(1) as f32).sqrt();
    let mut rng = SplitMix::new(name_seed(name));
    let data = (0..Graph::numel(shape))
        .map(|_| rng.symmetric(limit))
        .collect();
    Slot {
        name: name.to_string(),
        shape: shape.to_vec(),
        data,
        trainable: true,
        slot: 0,
    }
}

/// Hashes a checkpoint name into a deterministic seed, so the same layer always starts identically.
fn name_seed(name: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.as_bytes() {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Square convolution geometry: `nn.Conv2d`'s `kernel_size`, `stride`, `padding` and `dilation`.
#[derive(Clone, Copy, Debug)]
pub struct ConvGeom {
    /// Square kernel edge.
    pub k: usize,
    /// Square stride.
    pub stride: usize,
    /// Square padding.
    pub pad: usize,
    /// Square dilation.
    pub dilation: usize,
}

impl ConvGeom {
    /// Explicit geometry.
    pub fn new(k: usize, stride: usize, pad: usize, dilation: usize) -> Self {
        Self {
            k,
            stride,
            pad,
            dilation,
        }
    }

    /// The "same" padding every animalpose convolution uses, `dilation * (k - 1) / 2` (`k // 2` for a
    /// dilated `3x3` is what `ConvBNAct` passes).
    pub fn same(k: usize, stride: usize, dilation: usize) -> Self {
        Self::new(k, stride, dilation * (k - 1) / 2, dilation)
    }
}

/// Options for [`conv_bn_act`], defaulting to Python's `ConvBNAct`: no conv bias, `eps = 1e-5`,
/// `momentum = 0.1`, ReLU after the normalisation.
#[derive(Clone, Copy, Debug)]
pub struct ConvBnActOpts {
    /// Give the convolution a bias (PyTorch's `ConvBNAct` uses `bias=False`).
    pub bias: bool,
    /// `nn.BatchNorm2d`'s `eps`.
    pub eps: f32,
    /// `nn.BatchNorm2d`'s `momentum`.
    pub momentum: f32,
    /// Apply ReLU after the normalisation.
    pub relu: bool,
}

impl Default for ConvBnActOpts {
    fn default() -> Self {
        Self {
            bias: false,
            eps: 1e-5,
            momentum: 0.1,
            relu: true,
        }
    }
}

/// `nn.Conv2d(cin, cout, k, s, k // 2, bias=False)` followed by `nn.BatchNorm2d(cout)` and ReLU, i.e.
/// `ConvBNAct` from `models/backbone.py`.
pub struct ConvBnAct {
    slots: Slots,
    names: Vec<String>,
    geom: ConvGeom,
    norm: BatchNorm,
    relu: bool,
    bias: bool,
}

impl Layer for ConvBnAct {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        let w = self.slots.push(g, CONV_WEIGHT);
        let b = if self.bias {
            Some(self.slots.push(g, CONV_BIAS))
        } else {
            None
        };
        let conv = g.conv2d(x, w, b, self.geom.stride, self.geom.pad, self.geom.dilation);
        // The ReLU rides on the normalisation node (`Graph::batch_norm_relu`): same values and
        // gradients as a separate ReLU, one pass fewer over the activations.
        self.norm.forward_act(g, conv, ctx, self.relu)
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.slots.visit(f);
        self.norm.visit(f);
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        self.slots.visit_mut(f);
        self.norm.visit_mut(f);
    }
}

impl ConvBnAct {
    /// Numbers its own and its batch norm's slots, so the flat parameter order matches `params()`.
    fn renumber(&mut self) {
        self.names = self.slots.names();
        self.norm.assign_slots(self.names.len());
        self.names.extend(self.norm.params().iter().cloned());
    }
}

/// Slot order of the convolution part of a [`ConvBnAct`].
const CONV_WEIGHT: usize = 0;
const CONV_BIAS: usize = 1;

/// `ConvBNAct(cin, cout, k, s)` from `models/backbone.py`: conv (no bias by default) + batch norm +
/// ReLU, with the checkpoint names `{name}.0.weight` and `{name}.1.*`.
pub fn conv_bn_act(
    name: &str,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    opts: ConvBnActOpts,
) -> ConvBnAct {
    let mut slots = vec![uniform(
        &format!("{name}.0.weight"),
        &[cout, cin, k, k],
        cin * k * k,
    )];
    if opts.bias {
        slots.push(uniform(&format!("{name}.0.bias"), &[cout], cin * k * k));
    }
    let mut layer = ConvBnAct {
        slots: Slots::new(slots),
        names: Vec::new(),
        geom: ConvGeom::same(k, stride, dilation),
        norm: BatchNorm::new(&format!("{name}.1"), cout, opts.eps, opts.momentum),
        relu: opts.relu,
        bias: opts.bias,
    };
    layer.renumber();
    layer
}

/// `nn.Conv2d(cin, cout, k, s, p, d)`, optionally with a bias, named `{name}.weight` / `{name}.bias`.
pub struct Conv2dLayer {
    slots: Slots,
    names: Vec<String>,
    geom: ConvGeom,
    bias: bool,
}

impl Conv2dLayer {
    /// A `k x k` convolution; `pad` is explicit so 1x1 projections (`pad = 0`) and the 3x3 stems share
    /// one constructor.
    pub fn new(name: &str, cin: usize, cout: usize, geom: ConvGeom, bias: bool) -> Self {
        let fan_in = cin * geom.k * geom.k;
        let mut slots = vec![uniform(
            &format!("{name}.weight"),
            &[cout, cin, geom.k, geom.k],
            fan_in,
        )];
        if bias {
            slots.push(uniform(&format!("{name}.bias"), &[cout], fan_in));
        }
        let slots = Slots::new(slots);
        let names = slots.names();
        Self {
            slots,
            names,
            geom,
            bias,
        }
    }

    /// The `1x1` projection used by the FPN laterals and the detection heads.
    pub fn pointwise(name: &str, cin: usize, cout: usize, bias: bool) -> Self {
        Self::new(name, cin, cout, ConvGeom::new(1, 1, 0, 1), bias)
    }
}

impl Layer for Conv2dLayer {
    fn forward(&self, g: &mut Graph, x: NodeId, _ctx: &mut Ctx) -> NodeId {
        let w = self.slots.push(g, 0);
        let b = if self.bias {
            Some(self.slots.push(g, 1))
        } else {
            None
        };
        g.conv2d(x, w, b, self.geom.stride, self.geom.pad, self.geom.dilation)
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.slots.visit(f);
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        self.slots.visit_mut(f);
    }
}

/// `nn.Conv1d(cin, cout, k, dilation=d)` over an `(N, C, L)` input (channels first, like PyTorch), with
/// the `padding = dilation * (k - 1) / 2` the behaviour network relies on.
pub struct Conv1dLayer {
    slots: Slots,
    names: Vec<String>,
    k: usize,
    dilation: usize,
    bias: bool,
}

impl Conv1dLayer {
    /// A dilated 1D convolution, `TCNBlock`'s `conv1` being `Conv1dLayer::dilated(name, c, c, 3, d, true)`.
    pub fn new(name: &str, cin: usize, cout: usize, k: usize, dilation: usize, bias: bool) -> Self {
        let fan_in = cin * k;
        let mut slots = vec![uniform(&format!("{name}.weight"), &[cout, cin, k], fan_in)];
        if bias {
            slots.push(uniform(&format!("{name}.bias"), &[cout], fan_in));
        }
        let slots = Slots::new(slots);
        let names = slots.names();
        Self {
            slots,
            names,
            k,
            dilation,
            bias,
        }
    }
}

impl Layer for Conv1dLayer {
    fn forward(&self, g: &mut Graph, x: NodeId, _ctx: &mut Ctx) -> NodeId {
        let w = self.slots.push(g, 0);
        let b = if self.bias {
            Some(self.slots.push(g, 1))
        } else {
            None
        };
        g.conv1d(x, w, b, self.k, self.dilation)
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.slots.visit(f);
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        self.slots.visit_mut(f);
    }
}

/// `nn.Linear(in, out)`: `y = x @ W.T + b`, with the weight stored `(out, in)` like the checkpoint.
pub struct Dense {
    slots: Slots,
    names: Vec<String>,
    bias: bool,
}

impl Dense {
    /// A fully connected layer named `{name}.weight` / `{name}.bias`.
    pub fn new(name: &str, input: usize, out: usize, bias: bool) -> Self {
        let mut slots = vec![uniform(&format!("{name}.weight"), &[out, input], input)];
        if bias {
            slots.push(uniform(&format!("{name}.bias"), &[out], input));
        }
        let slots = Slots::new(slots);
        let names = slots.names();
        Self { slots, names, bias }
    }
}

impl Layer for Dense {
    fn forward(&self, g: &mut Graph, x: NodeId, _ctx: &mut Ctx) -> NodeId {
        let w = self.slots.push(g, 0);
        let wt = g.permute(w, &[1, 0]);
        let mut y = g.matmul(x, wt);
        if self.bias {
            let b = self.slots.push(g, 1);
            y = g.add(y, b);
        }
        y
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.slots.visit(f);
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        self.slots.visit_mut(f);
    }
}

/// `nn.BatchNorm1d` / `nn.BatchNorm2d`: channel-wise normalisation with running statistics.
///
/// The running buffers are updated by a training forward pass (`nn.BatchNorm` does the same) and used
/// verbatim when [`Ctx::train`] is false. `num_batches_tracked` is not stored, which is what PyTorch's
/// default `momentum` implies.
pub struct BatchNorm {
    slots: Slots,
    names: Vec<String>,
    eps: f32,
    momentum: f32,
}

/// Slot order of a batch norm: gamma, beta, running mean, running variance.
const BN_GAMMA: usize = 0;
const BN_BETA: usize = 1;
const BN_MEAN: usize = 2;
const BN_VAR: usize = 3;

impl BatchNorm {
    /// A normalisation layer over `channels` channels.
    pub fn new(name: &str, channels: usize, eps: f32, momentum: f32) -> Self {
        let slots = vec![
            filled(&format!("{name}.weight"), &[channels], 1.0, true),
            filled(&format!("{name}.bias"), &[channels], 0.0, true),
            filled(&format!("{name}.running_mean"), &[channels], 0.0, false),
            filled(&format!("{name}.running_var"), &[channels], 1.0, false),
        ];
        let slots = Slots::new(slots);
        let names = slots.names();
        Self {
            slots,
            names,
            eps,
            momentum,
        }
    }

    /// The Python defaults, `eps = 1e-5` and `momentum = 0.1`.
    pub fn python(name: &str, channels: usize) -> Self {
        Self::new(name, channels, 1e-5, 0.1)
    }

    /// [`Layer::forward`], optionally followed by a ReLU fused into the same node.
    fn forward_act(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx, relu: bool) -> NodeId {
        let gamma = self.slots.push(g, BN_GAMMA);
        let beta = self.slots.push(g, BN_BETA);
        let norm = |g: &mut Graph, mode: BnMode| {
            if relu {
                g.batch_norm_relu(x, gamma, beta, self.eps, mode)
            } else {
                g.batch_norm(x, gamma, beta, self.eps, mode)
            }
        };
        if !ctx.train {
            let (mean, var) = (self.slots.data(BN_MEAN), self.slots.data(BN_VAR));
            return norm(g, BnMode::Eval { mean, var });
        }
        let (mean, var) = ops::batch_stats(g, x);
        let out = norm(g, BnMode::Train);
        // `nn.BatchNorm` refreshes its buffers during the same forward pass it computed the batch
        // statistics in.
        let mut at = 0usize;
        self.slots.visit_mut(&mut |s| {
            if at == BN_MEAN {
                ops::update_running(&mut s.data, &mean, self.momentum);
            } else if at == BN_VAR {
                ops::update_running(&mut s.data, &var, self.momentum);
            }
            at += 1;
        });
        out
    }
}

impl Layer for BatchNorm {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        self.forward_act(g, x, ctx, false)
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.slots.visit(f);
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        self.slots.visit_mut(f);
    }
}

/// `nn.Dropout(p)`: inverted dropout while training, the identity in eval mode.
#[derive(Clone, Copy, Debug)]
pub struct Dropout {
    /// Probability of dropping an element.
    pub p: f32,
}

impl Dropout {
    /// A dropout layer with probability `p`.
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Layer for Dropout {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        if ctx.train && self.p > 0.0 {
            g.dropout(x, self.p, ctx.seed)
        } else {
            x
        }
    }

    fn params(&self) -> &[String] {
        &[]
    }

    fn visit(&self, _f: &mut dyn FnMut(&Slot)) {}

    fn visit_mut(&mut self, _f: &mut dyn FnMut(&mut Slot)) {}
}

/// `nn.Flatten(start_dim)`: folds the trailing axes into one, leaving the leading axes alone.
#[derive(Clone, Copy, Debug)]
pub struct Flatten {
    /// First axis folded into the feature axis.
    pub start_dim: usize,
}

impl Flatten {
    /// A layer flattening from `start_dim` to the last axis.
    pub fn new(start_dim: usize) -> Self {
        Self { start_dim }
    }
}

impl Layer for Flatten {
    fn forward(&self, g: &mut Graph, x: NodeId, _ctx: &mut Ctx) -> NodeId {
        g.flatten(x, self.start_dim)
    }

    fn params(&self) -> &[String] {
        &[]
    }

    fn visit(&self, _f: &mut dyn FnMut(&Slot)) {}

    fn visit_mut(&mut self, _f: &mut dyn FnMut(&mut Slot)) {}
}

/// The four blocks of one LSTM direction, named like `nn.LSTM`'s state dict:
/// `{prefix}.weight_ih_l{layer}[_reverse]`, `weight_hh`, `bias_ih`, `bias_hh`, in that order.
pub struct LstmCell {
    slots: Slots,
    names: Vec<String>,
    input: usize,
    hidden: usize,
    reverse: bool,
}

impl LstmCell {
    /// The blocks of layer `layer` (direction `reverse`) of an `nn.LSTM` named `prefix`.
    pub fn new(prefix: &str, layer: usize, input: usize, hidden: usize, reverse: bool) -> Self {
        let suffix = if reverse { "_reverse" } else { "" };
        let gate = 4 * hidden;
        let slot = |which: &str, fan_in: usize| {
            uniform(
                &format!("{prefix}.{which}_l{layer}{suffix}"),
                &[gate, fan_in],
                fan_in,
            )
        };
        let bias = |which: &str| {
            filled(
                &format!("{prefix}.{which}_l{layer}{suffix}"),
                &[gate],
                0.0,
                true,
            )
        };
        let slots = vec![
            slot("weight_ih", input),
            slot("weight_hh", hidden),
            bias("bias_ih"),
            bias("bias_hh"),
        ];
        let slots = Slots::new(slots);
        let names = slots.names();
        Self {
            slots,
            names,
            input,
            hidden,
            reverse,
        }
    }

    /// Input width this cell was built for.
    pub fn input(&self) -> usize {
        self.input
    }

    /// Hidden width, i.e. `nn.LSTM`'s `hidden_size`.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// `true` for the reverse direction.
    pub fn reverse(&self) -> bool {
        self.reverse
    }

    /// Pushes the four blocks onto `g` in [`Graph::lstm`]'s order.
    fn push_blocks(&self, g: &mut Graph) -> Vec<NodeId> {
        (0..4).map(|i| self.slots.push(g, i)).collect()
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.slots.visit(f);
    }

    fn visit_mut(&self, f: &mut dyn FnMut(&mut Slot)) {
        self.slots.visit_mut(f);
    }
}

/// `nn.LSTM(hidden, hidden, layers, batch_first=True, bidirectional=..., dropout=...)` from
/// `temporal/model.py`: `layers` stacked [`LstmCell`] pairs, with `nn.Dropout` between them.
pub struct Lstm {
    cells: Vec<LstmCell>,
    names: Vec<String>,
    hidden: usize,
    layers: usize,
    bidirectional: bool,
    dropout: f32,
}

impl Lstm {
    /// Builds the cells of a `layers`-deep LSTM; `input` is the width of the first layer (`hidden` for
    /// every later one). `nn.LSTM` only drops out *between* layers, so `dropout` is ignored for a
    /// single-layer stack, exactly as `TemporalLSTM` configures it.
    pub fn new(
        prefix: &str,
        input: usize,
        hidden: usize,
        layers: usize,
        bidirectional: bool,
        dropout: f32,
    ) -> Self {
        let mut cells = Vec::new();
        let width = dir_width(hidden, bidirectional);
        for l in 0..layers {
            let in_w = if l == 0 { input } else { width };
            cells.push(LstmCell::new(prefix, l, in_w, hidden, false));
            if bidirectional {
                cells.push(LstmCell::new(prefix, l, in_w, hidden, true));
            }
        }
        let names = cells.iter().flat_map(|c| c.names.iter().cloned()).collect();
        let dropout = if layers > 1 { dropout } else { 0.0 };
        Self {
            cells,
            names,
            hidden,
            layers,
            bidirectional,
            dropout,
        }
    }

    /// `nn.LSTM`'s `hidden_size`.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Width of the output: `hidden * 2` for a bidirectional stack (`TemporalLSTM`'s `out`).
    pub fn out_width(&self) -> usize {
        dir_width(self.hidden, self.bidirectional)
    }
}

impl Layer for Lstm {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        let dir = usize::from(self.bidirectional) + 1;
        let mut out = x;
        for l in 0..self.layers {
            let w: Vec<NodeId> = self.cells[l * dir..(l + 1) * dir]
                .iter()
                .flat_map(|c| c.push_blocks(g))
                .collect();
            out = g.lstm(out, &w, self.hidden, 1, self.bidirectional);
            if l + 1 < self.layers && self.dropout > 0.0 && ctx.train {
                out = g.dropout(out, self.dropout, ctx.seed + l as u64);
            }
        }
        out
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        for c in &self.cells {
            c.visit(f);
        }
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        for c in self.cells.iter_mut() {
            c.visit_mut(f);
        }
    }
}

/// A stack of layers run in order, i.e. `nn.Sequential`.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    names: Vec<String>,
}

impl Sequential {
    /// An empty stack.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            names: Vec::new(),
        }
    }

    /// A stack of `layers`, mirroring `nn.Sequential(*layers)`.
    pub fn of(layers: Vec<Box<dyn Layer>>) -> Self {
        let mut out = Self {
            layers,
            names: Vec::new(),
        };
        out.renumber();
        out
    }

    /// Appends one layer (`Sequential::new().with(a).with(b)`).
    #[must_use]
    pub fn with(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self.renumber();
        self
    }

    fn renumber(&mut self) {
        self.names = self
            .layers
            .iter()
            .flat_map(|l| l.params().iter().cloned())
            .collect();
        self.assign_slots(0);
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sequential {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        let mut out = x;
        for l in &self.layers {
            out = l.forward(g, out, ctx);
        }
        out
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        for l in &self.layers {
            l.visit(f);
        }
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        for l in self.layers.iter_mut() {
            l.visit_mut(f);
        }
    }
}

/// `ResBlock` from `models/backbone.py`: `F.relu(main(x) + skip(x))`, the skip being the identity when
/// it is `None` (stride 1, channels unchanged).
pub struct Residual {
    main: Box<dyn Layer>,
    skip: Option<Box<dyn Layer>>,
    names: Vec<String>,
}

impl Residual {
    /// A residual block with an explicit skip branch (a `1x1` conv + batch norm in Python).
    pub fn new(main: Box<dyn Layer>, skip: Box<dyn Layer>) -> Self {
        let mut out = Self {
            main,
            skip: Some(skip),
            names: Vec::new(),
        };
        out.renumber();
        out
    }

    /// A residual block whose skip is the identity.
    pub fn identity(main: Box<dyn Layer>) -> Self {
        let mut out = Self {
            main,
            skip: None,
            names: Vec::new(),
        };
        out.renumber();
        out
    }

    fn renumber(&mut self) {
        let mut names = self.main.params().to_vec();
        if let Some(skip) = &self.skip {
            names.extend(skip.params().iter().cloned());
        }
        self.names = names;
        self.assign_slots(0);
    }
}

impl Layer for Residual {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        let body = self.main.forward(g, x, ctx);
        let skip = match &self.skip {
            Some(l) => l.forward(g, x, ctx),
            None => x,
        };
        let sum = g.add(body, skip);
        g.relu(sum)
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        self.main.visit(f);
        if let Some(skip) = &self.skip {
            skip.visit(f);
        }
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        self.main.visit_mut(f);
        if let Some(skip) = &mut self.skip {
            skip.visit_mut(f);
        }
    }
}

/// Runs every branch on the same input and merges the results: `add` when `axis` is `None`, `concat`
/// along `axis` otherwise. The FPN of `models/backbone.py` merges its lateral and upsampled paths this
/// way.
pub struct Parallel {
    branches: Vec<Box<dyn Layer>>,
    axis: Option<usize>,
    names: Vec<String>,
}

impl Parallel {
    /// Branches summed element-wise (`axis = None`).
    pub fn add(branches: Vec<Box<dyn Layer>>) -> Self {
        Self::build(branches, None)
    }

    /// Branches concatenated along `axis`.
    pub fn concat(branches: Vec<Box<dyn Layer>>, axis: usize) -> Self {
        Self::build(branches, Some(axis))
    }

    fn build(branches: Vec<Box<dyn Layer>>, axis: Option<usize>) -> Self {
        let mut out = Self {
            branches,
            axis,
            names: Vec::new(),
        };
        out.names = out
            .branches
            .iter()
            .flat_map(|l| l.params().iter().cloned())
            .collect();
        out.assign_slots(0);
        out
    }
}

impl Layer for Parallel {
    fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> NodeId {
        let parts: Vec<NodeId> = self.branches.iter().map(|l| l.forward(g, x, ctx)).collect();
        match self.axis {
            Some(axis) => g.concat(&parts, axis),
            None => {
                let mut out = parts[0];
                for p in &parts[1..] {
                    out = g.add(out, *p);
                }
                out
            }
        }
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        for l in &self.branches {
            l.visit(f);
        }
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        for l in self.branches.iter_mut() {
            l.visit_mut(f);
        }
    }
}

/// Output width of one LSTM layer: `hidden`, or `hidden * 2` when the two directions are concatenated.
fn dir_width(hidden: usize, bidirectional: bool) -> usize {
    hidden * (usize::from(bidirectional) + 1)
}
