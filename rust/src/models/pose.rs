//! `src/animalpose/pose/model.py`: the top-down keypoint network.
//!
//! Heat maps come from *keypoint queries* — every canonical keypoint owns a learned embedding and its
//! map is the dot product of that embedding with the shared feature map, done as a `1x1` convolution
//! ([`PoseNet::forward`]). Locations are decoded elsewhere (see `crate::algos::decode`) as the peak
//! plus a learned sub-pixel offset; a pooled appearance embedding is emitted for re-identification.
//!
//! The checkpoint names this network owns, on top of the `backbone.*` keys of [`crate::models::backbone`]:
//!
//! | Python module | checkpoint key | Rust construction |
//! | --- | --- | --- |
//! | `PoseNet.heat_feat` (`ConvBNAct`, `ConvBNAct`, `Conv2d`) | `heat_feat.{0,1}.0.weight`, `...{0,1}.1.*`, `heat_feat.2.weight` / `.bias` | [`conv_bn_act`] + [`Conv2dLayer::pointwise`] |
//! | `PoseNet.off_feat` | `off_feat.0.0.weight`, `off_feat.0.1.*`, `off_feat.1.weight` / `.bias` | as above |
//! | `PoseNet.heat_queries` / `heat_bias` / `off_queries` (`nn.Parameter`) | `heat_queries`, `heat_bias`, `off_queries` | [`Params`] |
//! | `PoseNet.embed` (`Linear`, `ReLU`, `Linear`) | `embed.0.weight` / `.bias`, `embed.2.weight` / `.bias` | [`Dense`] + [`Relu`] |
//! | `PoseNet.context` (optional FiLM MLP) | `context.0.*`, `context.2.*` | [`Dense`] + [`Relu`] |

use crate::models::backbone::Backbone;
use crate::models::config::ModelConfig;
use crate::models::layers::{LayerTree, LoadReport, Params, Relu};
use crate::models::weights::Weights;
use crate::tensor::{
    conv_bn_act, Conv2dLayer, ConvBnActOpts, Ctx, Dense, Graph, Layer, NodeId, Sequential,
};
use anyhow::{ensure, Result};

/// The output names of `PoseNet.forward`, in dict order: `heat`, `offset`, `embedding`.
pub const OUTPUTS: [&str; 3] = ["heat", "offset", "embedding"];

/// The input names of `PoseNet.forward`: one RGB crop, `(B, 3, S, S)` float in `[0, 1]`.
pub const INPUTS: [&str; 1] = ["crops"];

/// `PoseNet` from `src/animalpose/pose/model.py`.
pub struct PoseNet {
    backbone: Backbone,
    heat_feat: Sequential,
    off_feat: Sequential,
    heat_queries: Params,
    heat_bias: Params,
    off_queries: Params,
    embed: Sequential,
    context: Option<Sequential>,
    num_keypoints: usize,
    channels: usize,
}

impl PoseNet {
    /// `PoseNet(num_keypoints, backbone, embed_dim, context)`.
    ///
    /// `// differs from Python:` the parameters start at their Python values except that
    /// `heat_queries` is zeroed rather than `torch.randn(K, C) / sqrt(C)` (PyTorch's RNG is not
    /// reproduced here); [`PoseNet::load_weights`] replaces every one of them.
    pub fn new(cfg: &ModelConfig) -> Result<Self> {
        let backbone = Backbone::new(&cfg.backbone())?;
        let c = backbone.out_channels();
        let deep = backbone.deep_channels();
        let k = cfg.num_keypoints();
        let heat_feat = Sequential::of(vec![
            tower_conv("heat_feat.0", c, c),
            tower_conv("heat_feat.1", c, c),
            Box::new(Conv2dLayer::pointwise("heat_feat.2", c, c, true)),
        ]);
        let off_feat = Sequential::of(vec![
            tower_conv("off_feat.0", c, c),
            Box::new(Conv2dLayer::pointwise("off_feat.1", c, c, true)),
        ]);
        Ok(Self {
            heat_feat,
            off_feat,
            heat_queries: Params::parameter("heat_queries", &[k, c], vec![0.0f32; k * c]),
            heat_bias: Params::parameter("heat_bias", &[k], vec![-4.0f32; k]),
            off_queries: Params::parameter("off_queries", &[2 * k, c], vec![0.0f32; 2 * k * c]),
            embed: mlp("embed", deep, EMBED_HIDDEN, cfg.embed_dim()),
            context: cfg.context().then(|| {
                // `nn.init.zeros_(self.context[-1].weight / .bias)`: FiLM starts as the identity.
                let mut film = mlp("context", deep, EMBED_HIDDEN, 2 * c);
                film.load("context.2.weight", &vec![0.0f32; 2 * c * EMBED_HIDDEN]);
                film.load("context.2.bias", &vec![0.0f32; 2 * c]);
                film
            }),
            backbone,
            num_keypoints: k,
            channels: c,
        })
    }

    /// `PoseNet.num_keypoints`, i.e. `heat_queries.shape[0]`.
    pub fn num_keypoints(&self) -> usize {
        self.num_keypoints
    }

    /// `PoseNet.stride`, the downsampling factor of the heat maps.
    pub fn stride(&self) -> usize {
        self.backbone.stride()
    }

    /// The FPN map width, `heat_feat`'s input and output channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// `PoseNet.forward(crops)`, returning `[heat, offset, embedding]`
    /// (the order of the returned dict).
    pub fn forward(&self, g: &mut Graph, crops: NodeId, ctx: &mut Ctx) -> Result<[NodeId; 3]> {
        let (feat, deep) = self.backbone.forward(g, crops, ctx)?;
        let shape = g.shape(feat);
        ensure!(
            shape.len() == 4 && shape[1] == self.channels,
            "the backbone returned {:?}, expected channels {}",
            shape,
            self.channels
        );
        let batch = shape[0];
        let c = self.channels;
        let k = self.num_keypoints;
        let pooled = g.reduce_mean(deep, &[2, 3]);

        let mut hf = self.heat_feat.forward(g, feat, ctx);
        if let Some(context) = &self.context {
            // `self.context(pooled).view(-1, 2 * c, 1, 1).chunk(2, dim=1)`, then `hf * (1 + scale) + shift`.
            let film = context.forward(g, pooled, ctx);
            let film = g.reshape(film, &[batch, 2 * c, 1, 1]);
            let scale = g.slice(film, 1, 0, c);
            let shift = g.slice(film, 1, c, 2 * c);
            let one = g.constant(1.0, &[1]);
            let gain = g.add(one, scale);
            let scaled = g.mul(hf, gain);
            hf = g.add(scaled, shift);
        }

        let queries = self.heat_queries.forward(g, feat, ctx);
        let queries = g.reshape(queries, &[k, c, 1, 1]);
        let bias = self.heat_bias.forward(g, feat, ctx);
        let heat = g.conv2d(hf, queries, Some(bias), 1, 0, 1);

        let features = self.off_feat.forward(g, feat, ctx);
        let off_queries = self.off_queries.forward(g, feat, ctx);
        let off_queries = g.reshape(off_queries, &[2 * k, c, 1, 1]);
        let offset = g.conv2d(features, off_queries, None, 1, 0, 1);

        let embedding = self.embed.forward(g, pooled, ctx);
        let embedding = g.l2_normalize(embedding, 1);
        Ok([heat, offset, embedding])
    }

    /// `PoseNet.forward` as named outputs, the dict the Python model returns.
    pub fn forward_named(
        &self,
        g: &mut Graph,
        crops: NodeId,
        ctx: &mut Ctx,
    ) -> Result<Vec<(&'static str, NodeId)>> {
        let [heat, offset, embedding] = self.forward(g, crops, ctx)?;
        Ok(vec![
            (OUTPUTS[0], heat),
            (OUTPUTS[1], offset),
            (OUTPUTS[2], embedding),
        ])
    }

    /// `model.load_state_dict(...)`: copies every matching tensor of `weights` in.
    pub fn load_weights(&mut self, weights: &Weights) -> LoadReport {
        crate::models::layers::load_into(self, weights)
    }
}

impl LayerTree for PoseNet {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        self.backbone.each_layer(f);
        f(&self.heat_feat);
        f(&self.off_feat);
        f(&self.heat_queries);
        f(&self.heat_bias);
        f(&self.off_queries);
        f(&self.embed);
        if let Some(context) = &self.context {
            f(context);
        }
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        self.backbone.each_layer_mut(f);
        f(&mut self.heat_feat);
        f(&mut self.off_feat);
        f(&mut self.heat_queries);
        f(&mut self.heat_bias);
        f(&mut self.off_queries);
        f(&mut self.embed);
        if let Some(context) = &mut self.context {
            f(context);
        }
    }
}

/// The hidden width of `PoseNet.embed` / `PoseNet.context`, hardcoded in the Python model.
pub const EMBED_HIDDEN: usize = 128;

/// One `ConvBNAct(c, c, 3)` tower step, named `name`.
fn tower_conv(name: &str, c: usize, cout: usize) -> Box<dyn Layer> {
    Box::new(conv_bn_act(
        name,
        c,
        cout,
        3,
        1,
        1,
        ConvBnActOpts::default(),
    ))
}

/// `nn.Sequential(nn.Linear(a, hidden), nn.ReLU(), nn.Linear(hidden, b))`, numbered `name.0` and
/// `name.2` like the Python module.
fn mlp(name: &str, input: usize, hidden: usize, output: usize) -> Sequential {
    Sequential::of(vec![
        Box::new(Dense::new(&format!("{name}.0"), input, hidden, true)),
        Box::new(Relu),
        Box::new(Dense::new(&format!("{name}.2"), hidden, output, true)),
    ])
}
