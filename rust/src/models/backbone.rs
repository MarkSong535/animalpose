//! The encoder + FPN of `src/animalpose/models/backbone.py`.
//!
//! [`Backbone`] normalises an RGB image with the ImageNet constants, runs an encoder that emits feature
//! maps at strides 4, 8, 16 and 32, and merges them top-down into one stride-4 map. The layer names are
//! the Python module paths (`backbone.encoder.stem.0.0.weight`, `backbone.fpn.lateral.2.weight`,
//! `backbone.encoder.layer2.0.downsample.0.weight`), so a Python state dict loads by name.
//!
//! Every convolution + normalisation step in the animalpose encoders is the same `ConvBNAct`, so all of
//! them are built with [`conv_bn_act`] and, where the Python module numbers its submodules differently,
//! renamed with [`crate::models::layers::rename_slots`]:
//!
//! | Python module | checkpoint keys | Rust |
//! | --- | --- | --- |
//! | `Normalize` buffers | `backbone.norm.mean`, `backbone.norm.std` | [`Normalize::new`] |
//! | `TinyResNet.stem[i]` (`ConvBNAct`) | `backbone.encoder.stem.{i}.0.weight`, `...{i}.1.*` | `conv_bn_act("backbone.encoder.stem.{i}")` |
//! | `ResBlock.conv1` (`ConvBNAct`) | `...stages.{s}.{b}.conv1.0.weight`, `...conv1.1.*` | `conv_bn_act("...conv1")` |
//! | `ResBlock.conv2` (`Conv2d` + `BatchNorm2d`, no ReLU) | `...stages.{s}.{b}.conv2.0.weight`, `...conv2.1.*` | `conv_bn_act("...conv2", relu = false)` |
//! | `ResBlock.skip` (`Conv2d(cin, cout, 1, s)` + BN) | `...stages.{s}.{b}.skip.0.weight`, `...skip.1.*` | `conv_bn_act("...skip", k = 1, relu = false)` |
//! | `FPN.lateral[i]` (`nn.Conv2d(c, out, 1)`, with bias) | `backbone.fpn.lateral.{i}.weight` / `.bias` | [`Conv2dLayer::pointwise`] |
//! | `FPN.smooth` (`ConvBNAct`) | `backbone.fpn.smooth.0.weight`, `...smooth.1.*` | `conv_bn_act("backbone.fpn.smooth")` |
//! | `TorchvisionResNet.stem` (`net.conv1`, `net.bn1`, relu, max pool) | `backbone.encoder.stem.0.weight`, `backbone.encoder.stem.1.*` | `conv_bn_act` + [`torch_step_names`] (7x7, stride 2, padding 3) |
//! | `BasicBlock` of `stages{s}` (`net.layer{i}`) | `...stages.{s}.{b}.conv{1,2}.weight`, `...bn{1,2}.*`, `...downsample.{0,1}.*` | [`torch_basic_block`] |
//!
//! `num_batches_tracked` is not stored, which is what PyTorch's default momentum implies;
//! [`crate::tensor::BatchNorm`] documents the same deviation.
//!
//! ```
//! use animalpose::models::backbone::Backbone;
//! use animalpose::models::config::BackboneConfig;
//! use animalpose::tensor::{Ctx, Graph};
//!
//! let cfg: BackboneConfig =
//!     serde_json::from_str(r#"{"name": "tiny", "width": 8, "out_channels": 16}"#).unwrap();
//! let mut backbone = Backbone::new(&cfg).unwrap();
//! animalpose::models::layers::assign_slots(&mut backbone, 0);
//! let mut g = Graph::new();
//! let x = g.constant(0.5, &[1, 3, 64, 64]);
//! let (feat, deep) = backbone.forward(&mut g, x, &mut Ctx::eval()).unwrap();
//! assert_eq!(g.shape(feat).as_slice(), [1, 16, 16, 16]);
//! assert_eq!(g.shape(deep).as_slice(), [1, 64, 2, 2]);
//! ```

use crate::models::config::BackboneConfig;
use crate::models::layers::{rename_slots, LayerTree, Params};
use crate::tensor::{
    conv_bn_act, Conv2dLayer, ConvBnAct, ConvBnActOpts, Ctx, Graph, Layer, NodeId, Residual,
    Sequential,
};
use anyhow::{bail, Result};

/// `IMAGENET_MEAN` of `models/backbone.py`.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// `IMAGENET_STD` of `models/backbone.py`.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The `blocks` argument of `TinyResNet.__init__`, which the configs do not change.
pub const TINY_BLOCKS: [usize; 4] = [1, 2, 2, 1];

/// The stage widths of a `TorchvisionResNet("resnet18")` encoder, `TorchvisionResNet.channels`.
pub const RESNET18_CHANNELS: [usize; 4] = [64, 128, 256, 512];

/// `Normalize` from `models/backbone.py`: maps RGB float in `[0, 1]` to ImageNet-normalised input.
pub struct Normalize {
    mean: Params,
    std: Params,
}

impl Normalize {
    /// The normalisation buffers, named `{prefix}.mean` / `{prefix}.std` (Python's `Backbone.norm`
    /// registers them as `(1, 3, 1, 1)` tensors holding [`IMAGENET_MEAN`] and [`IMAGENET_STD`]).
    pub fn new(prefix: &str) -> Self {
        Self {
            mean: Params::buffer(
                &format!("{prefix}.mean"),
                &[1, 3, 1, 1],
                IMAGENET_MEAN.to_vec(),
            ),
            std: Params::buffer(
                &format!("{prefix}.std"),
                &[1, 3, 1, 1],
                IMAGENET_STD.to_vec(),
            ),
        }
    }

    /// `(x - mean) / std`.
    ///
    /// `// differs from Python:` the division by the per-channel `std` is a multiplication by its
    /// reciprocal, because the tape has no division node; the two agree to the last bit of `f32`.
    pub fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let mean = self.mean.forward(g, x, &mut Ctx::eval());
        let shifted = g.sub(x, mean);
        let inv: Vec<f32> = self.std.values().iter().map(|s| 1.0 / s).collect();
        let shape = self.std.shape().to_vec();
        let inv = g.from_data(inv, &shape);
        g.mul(shifted, inv)
    }
}

impl LayerTree for Normalize {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        f(&self.mean);
        f(&self.std);
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        f(&mut self.mean);
        f(&mut self.std);
    }
}

/// A `ConvBNAct` with no ReLU after the normalisation: the `conv2` / `skip` / `downsample` branches,
/// which in Python are bare `Conv2d` + `BatchNorm2d` pairs inside a `nn.Sequential`.
fn conv_bn(name: &str, cin: usize, cout: usize, k: usize, stride: usize) -> ConvBnAct {
    conv_bn_act(
        name,
        cin,
        cout,
        k,
        stride,
        1,
        ConvBnActOpts {
            relu: false,
            ..ConvBnActOpts::default()
        },
    )
}

/// Renames the two numbered steps of a [`ConvBnAct`] built under `built_as` (`{built_as}.0.*` and
/// `{built_as}.1.*`) to the names the Python module gives its convolution and its normalisation:
/// torchvision's own `conv1` / `bn1` inside a `BasicBlock`, or the `stem.0` / `stem.1` step indices that
/// `TorchvisionResNet`'s `nn.Sequential` gives `net.conv1` and `net.bn1`.
fn torch_step_names(layer: &mut ConvBnAct, built_as: &str, conv: &str, bn: &str) {
    let (conv_prefix, bn_prefix) = (format!("{built_as}.0."), format!("{built_as}.1."));
    let (conv, bn) = (conv.to_string(), bn.to_string());
    let n = rename_slots(layer, &|name: &str| {
        if let Some(rest) = name.strip_prefix(conv_prefix.as_str()) {
            return Some(format!("{conv}.{rest}"));
        }
        name.strip_prefix(bn_prefix.as_str())
            .map(|rest| format!("{bn}.{rest}"))
    });
    debug_assert_eq!(n, 5, "{built_as} has five slots");
}

/// `ResBlock` from `models/backbone.py`: `F.relu(conv2(conv1(x)) + skip(x))`, the shortcut being the
/// identity exactly when Python uses one (`stride == 1 && cin == cout`).
pub fn res_block(prefix: &str, cin: usize, cout: usize, stride: usize) -> Box<dyn Layer> {
    let main = Sequential::of(vec![
        Box::new(conv_bn_act(
            &format!("{prefix}.conv1"),
            cin,
            cout,
            3,
            stride,
            1,
            ConvBnActOpts::default(),
        )),
        Box::new(conv_bn(&format!("{prefix}.conv2"), cout, cout, 3, 1)),
    ]);
    if stride == 1 && cin == cout {
        return Box::new(Residual::identity(Box::new(main)));
    }
    let skip = conv_bn(&format!("{prefix}.skip"), cin, cout, 1, stride);
    Box::new(Residual::new(Box::new(main), Box::new(skip)))
}

/// torchvision's `BasicBlock`, as `TorchvisionResNet` names it (`backbone.encoder.stages.{s}.{b}`):
/// `conv1`/`bn1` (with the block's stride), `conv2`/`bn2`, and a `downsample.{0,1}` shortcut whenever the
/// shapes change.
pub fn torch_basic_block(prefix: &str, cin: usize, cout: usize, stride: usize) -> Box<dyn Layer> {
    let mut conv1 = conv_bn_act(prefix, cin, cout, 3, stride, 1, ConvBnActOpts::default());
    torch_step_names(
        &mut conv1,
        prefix,
        &format!("{prefix}.conv1"),
        &format!("{prefix}.bn1"),
    );
    let mut conv2 = conv_bn(prefix, cout, cout, 3, 1);
    torch_step_names(
        &mut conv2,
        prefix,
        &format!("{prefix}.conv2"),
        &format!("{prefix}.bn2"),
    );
    let main = Sequential::of(vec![Box::new(conv1), Box::new(conv2)]);
    if stride == 1 && cin == cout {
        return Box::new(Residual::identity(Box::new(main)));
    }
    let mut skip = conv_bn(prefix, cin, cout, 1, stride);
    torch_step_names(
        &mut skip,
        prefix,
        &format!("{prefix}.downsample.0"),
        &format!("{prefix}.downsample.1"),
    );
    Box::new(Residual::new(Box::new(main), Box::new(skip)))
}

/// `TinyResNet` from `models/backbone.py`: two `ConvBNAct`s (stride 2 then 1), then four stages whose
/// first block strides by 2, emitting maps at strides 4, 8, 16 and 32.
pub struct TinyResNet {
    stem: Sequential,
    stages: Vec<Sequential>,
    channels: Vec<usize>,
}

impl TinyResNet {
    /// `TinyResNet(width, blocks)` with the default `blocks = (1, 2, 2, 1)`; `prefix` is the Python
    /// path of the encoder, `backbone.encoder`.
    pub fn new(prefix: &str, width: usize) -> Self {
        Self::with_blocks(prefix, width, &TINY_BLOCKS)
    }

    /// The same with an explicit block count per stage.
    pub fn with_blocks(prefix: &str, width: usize, blocks: &[usize]) -> Self {
        let chans: Vec<usize> = (0..4).map(|i| width << i).collect();
        let stem = Sequential::of(vec![
            Box::new(conv_bn_act(
                &format!("{prefix}.stem.0"),
                3,
                width,
                3,
                2,
                1,
                ConvBnActOpts::default(),
            )),
            Box::new(conv_bn_act(
                &format!("{prefix}.stem.1"),
                width,
                width,
                3,
                1,
                1,
                ConvBnActOpts::default(),
            )),
        ]);
        let mut stages = Vec::new();
        let mut cin = width;
        for (stage, (&c, &n)) in chans.iter().zip(blocks).enumerate() {
            let mut layers: Vec<Box<dyn Layer>> = Vec::new();
            for block in 0..n {
                // The first block of *every* stage halves the resolution, so with the stride-2 stem
                // the stage outputs land at strides 4, 8, 16 and 32.
                let stride = if block == 0 { 2 } else { 1 };
                layers.push(res_block(
                    &format!("{prefix}.stages.{stage}.{block}"),
                    cin,
                    c,
                    stride,
                ));
                cin = c;
            }
            stages.push(Sequential::of(layers));
        }
        Self {
            stem,
            stages,
            channels: chans,
        }
    }

    /// `TinyResNet.channels`, the stage widths `[w, 2w, 4w, 8w]`.
    pub fn channels(&self) -> &[usize] {
        &self.channels
    }

    /// The stage outputs, the `feats` list of `TinyResNet.forward`.
    pub fn forward_feats(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> Vec<NodeId> {
        let mut x = self.stem.forward(g, x, ctx);
        let mut feats = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            x = stage.forward(g, x, ctx);
            feats.push(x);
        }
        feats
    }
}

impl LayerTree for TinyResNet {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        f(&self.stem);
        for stage in &self.stages {
            f(stage);
        }
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        f(&mut self.stem);
        for stage in &mut self.stages {
            f(stage);
        }
    }
}

/// `TorchvisionResNet("resnet18")` from `models/backbone.py`: a 7x7 stride-2 conv with padding 3, its
/// batch norm, ReLU and a 3x3 stride-2 max pool, then the four `layer{i}` stages of two
/// [`torch_basic_block`]s each.
pub struct ResNet18 {
    stem: ConvBnAct,
    stages: Vec<Sequential>,
    channels: Vec<usize>,
}

impl ResNet18 {
    /// The encoder under `prefix`; `width` replaces torchvision's fixed 64 channels, which only the
    /// tests need (a real resnet18 checkpoint is 44 MB of weights).
    pub fn new(prefix: &str, width: usize) -> Self {
        // `net.conv1` (7x7, stride 2, padding 3), `net.bn1`, `net.relu`: one `ConvBNAct`.
        let mut stem = conv_bn_act(prefix, 3, width, 7, 2, 1, ConvBnActOpts::default());
        torch_step_names(
            &mut stem,
            prefix,
            &format!("{prefix}.stem.0"),
            &format!("{prefix}.stem.1"),
        );
        // `[64, 128, 256, 512]` for torchvision's own `width = 64`.
        let chans: Vec<usize> = (0..4).map(|i| width << i).collect();
        let mut stages = Vec::new();
        let mut cin = width;
        for (stage, &c) in chans.iter().enumerate() {
            // `layer1` runs at the stem's stride 4; every later stage halves the resolution again.
            // `TorchvisionResNet` collects the layers in a `ModuleList` named `stages`.
            let stride = if stage == 0 { 1 } else { 2 };
            let name = format!("{prefix}.stages.{stage}");
            stages.push(Sequential::of(vec![
                torch_basic_block(&format!("{name}.0"), cin, c, stride),
                torch_basic_block(&format!("{name}.1"), c, c, 1),
            ]));
            cin = c;
        }
        Self {
            stem,
            stages,
            channels: chans,
        }
    }

    /// `TorchvisionResNet.channels`.
    pub fn channels(&self) -> &[usize] {
        &self.channels
    }

    /// `TorchvisionResNet.forward`: `stem` (conv, norm, relu, max pool) then `stages`, i.e. `layer1..4`.
    pub fn forward_feats(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> Vec<NodeId> {
        let x = self.stem.forward(g, x, ctx);
        let mut x = g.max_pool2d(x, 3, 2, 1); // `net.maxpool`
        let mut feats = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            x = stage.forward(g, x, ctx);
            feats.push(x);
        }
        feats
    }
}

impl LayerTree for ResNet18 {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        f(&self.stem);
        for stage in &self.stages {
            f(stage);
        }
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        f(&mut self.stem);
        for stage in &mut self.stages {
            f(stage);
        }
    }
}

/// `FPN` from `models/backbone.py`: 1x1 laterals, nearest-neighbour top-down upsampling and a sum, then
/// a `ConvBNAct` smoother.
pub struct FPN {
    lateral: Vec<Conv2dLayer>,
    smooth: ConvBnAct,
}

impl FPN {
    /// `FPN(in_channels, out_channels)`; `prefix` is `backbone.fpn`.
    pub fn new(prefix: &str, in_channels: &[usize], out_channels: usize) -> Self {
        Self {
            lateral: in_channels
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    Conv2dLayer::pointwise(&format!("{prefix}.lateral.{i}"), c, out_channels, true)
                })
                .collect(),
            smooth: conv_bn_act(
                &format!("{prefix}.smooth"),
                out_channels,
                out_channels,
                3,
                1,
                1,
                ConvBnActOpts::default(),
            ),
        }
    }

    /// `FPN.forward(feats)`: the merged stride-4 map.
    pub fn forward(&self, g: &mut Graph, feats: &[NodeId], ctx: &mut Ctx) -> NodeId {
        let last = feats.len() - 1;
        let mut x = self.lateral[last].forward(g, feats[last], ctx);
        for i in (0..last).rev() {
            let lateral = self.lateral[i].forward(g, feats[i], ctx);
            let shape = g.shape(feats[i]).clone();
            let (h, w) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            let up = g.upsample_nearest(x, h, w);
            x = g.add(lateral, up);
        }
        self.smooth.forward(g, x, ctx)
    }
}

impl LayerTree for FPN {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        for lateral in &self.lateral {
            f(lateral);
        }
        f(&self.smooth);
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        for lateral in &mut self.lateral {
            f(lateral);
        }
        f(&mut self.smooth);
    }
}

/// Which encoder a checkpoint holds: [`TinyResNet`] or `TorchvisionResNet`.
pub enum Encoder {
    /// `TinyResNet`, the encoder animalpose trains from scratch.
    Tiny(TinyResNet),
    /// `TorchvisionResNet("resnet18")`.
    Resnet18(ResNet18),
}

impl Encoder {
    /// The encoder's stage widths, `TinyResNet.channels` / `TorchvisionResNet.channels`.
    pub fn channels(&self) -> &[usize] {
        match self {
            Self::Tiny(e) => e.channels(),
            Self::Resnet18(e) => e.channels(),
        }
    }

    /// The stage outputs, at strides 4, 8, 16 and 32.
    pub fn forward_feats(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> Vec<NodeId> {
        match self {
            Self::Tiny(e) => e.forward_feats(g, x, ctx),
            Self::Resnet18(e) => e.forward_feats(g, x, ctx),
        }
    }
}

impl LayerTree for Encoder {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        match self {
            Self::Tiny(e) => e.each_layer(f),
            Self::Resnet18(e) => e.each_layer(f),
        }
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        match self {
            Self::Tiny(e) => e.each_layer_mut(f),
            Self::Resnet18(e) => e.each_layer_mut(f),
        }
    }
}

/// `Backbone` from `models/backbone.py`: [`Normalize`] + encoder + [`FPN`].
pub struct Backbone {
    norm: Normalize,
    encoder: Encoder,
    fpn: FPN,
    out_channels: usize,
    deep_channels: usize,
    stride: usize,
}

impl Backbone {
    /// `build_backbone(cfg)`.
    pub fn new(cfg: &BackboneConfig) -> Result<Self> {
        let out_channels = cfg.out_channels();
        let encoder = match cfg.name() {
            "tiny" => Encoder::Tiny(TinyResNet::new("backbone.encoder", cfg.width())),
            "resnet18" => Encoder::Resnet18(ResNet18::new("backbone.encoder", RESNET18_CHANNELS[0])),
            other => bail!(
                "unsupported backbone name {other:?}: this port builds `tiny` and `resnet18` encoders"
            ),
        };
        let in_ch = encoder.channels().to_vec();
        let deep_channels = *in_ch
            .last()
            .ok_or_else(|| anyhow::anyhow!("the encoder has no stages"))?;
        Ok(Self {
            norm: Normalize::new("backbone.norm"),
            encoder,
            fpn: FPN::new("backbone.fpn", &in_ch, out_channels),
            out_channels,
            deep_channels,
            stride: 4,
        })
    }

    /// `Backbone.out_channels`, the FPN map width.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// `Backbone.deep_channels`, the last encoder stage's width.
    pub fn deep_channels(&self) -> usize {
        self.deep_channels
    }

    /// `Backbone.stride`, the FPN map's downsampling factor.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The encoder behind this backbone.
    pub fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// `Backbone.forward(x)` -> `(stride-4 map, deepest map)`.
    pub fn forward(&self, g: &mut Graph, x: NodeId, ctx: &mut Ctx) -> Result<(NodeId, NodeId)> {
        let x = self.norm.forward(g, x);
        let feats = self.encoder.forward_feats(g, x, ctx);
        let deep = *feats
            .last()
            .ok_or_else(|| anyhow::anyhow!("the encoder produced no feature maps"))?;
        Ok((self.fpn.forward(g, &feats, ctx), deep))
    }
}

impl LayerTree for Backbone {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        self.norm.each_layer(f);
        self.encoder.each_layer(f);
        self.fpn.each_layer(f);
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        self.norm.each_layer_mut(f);
        self.encoder.each_layer_mut(f);
        self.fpn.each_layer_mut(f);
    }
}
