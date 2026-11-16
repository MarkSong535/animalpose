//! `src/animalpose/detector/model.py`: the anchor-free, class-agnostic animal detector.
//!
//! [`CenterDetector`] predicts one "animal" centre heat map plus size and sub-cell offset regressors;
//! an optional species head classifies each detection for routing without affecting localisation. Peak
//! extraction lives in `crate::algos::detector_decode` (the Python `decode` of this module).
//!
//! Checkpoint naming, on top of the `backbone.*` keys of [`crate::models::backbone`]; `name` is one of
//! `heat`, `size`, `offset`, `species`:
//!
//! | Python module | checkpoint key | Rust construction |
//! | --- | --- | --- |
//! | `_head(name, ..)[0]` (`ConvBNAct`) | `{name}.0.0.weight`, `{name}.0.1.*` | [`conv_bn_act`] |
//! | `_head(name, ..)[1]` (`nn.Conv2d(cin, cout, 1)`) | `{name}.1.weight`, `{name}.1.bias` | [`Conv2dLayer::pointwise`] |

use crate::models::backbone::Backbone;
use crate::models::config::ModelConfig;
use crate::models::layers::{LayerTree, LoadReport};
use crate::models::weights::Weights;
use crate::tensor::{
    conv_bn_act, Conv2dLayer, ConvBnActOpts, Ctx, Graph, Layer, NodeId, Sequential,
};
use anyhow::Result;

/// `CenterDetector.forward`'s output names: `heat`, `size`, `offset`, then `species` when the head
/// exists.
pub const OUTPUTS: [&str; 4] = ["heat", "size", "offset", "species"];

/// The input names of `CenterDetector.forward`: one RGB image, `(B, 3, H, W)`, `H`/`W` multiples of 32.
pub const INPUTS: [&str; 1] = ["images"];

/// The heat-map bias prior of `CenterDetector.__init__`, `-ln((1 - p) / p)` for `p = 0.01`.
pub fn heat_bias_prior(prior: f32) -> f32 {
    -((1.0 - prior) / prior).ln()
}

/// `_head(cin, cout, bias)` from `detector/model.py`: a `ConvBNAct(cin, cin, 3)` then a `1x1`
/// convolution whose bias starts at `bias`.
pub fn head(prefix: &str, cin: usize, cout: usize, bias: f32) -> Sequential {
    let mut pointwise = Conv2dLayer::pointwise(&format!("{prefix}.1"), cin, cout, true);
    // `nn.init.constant_(head[-1].bias, bias)`; a checkpoint load overwrites it anyway.
    let values = vec![bias; cout];
    pointwise.load(&format!("{prefix}.1.bias"), &values);
    Sequential::of(vec![
        Box::new(conv_bn_act(
            &format!("{prefix}.0"),
            cin,
            cin,
            3,
            1,
            1,
            ConvBnActOpts::default(),
        )),
        Box::new(pointwise),
    ])
}

/// `CenterDetector` from `src/animalpose/detector/model.py`.
pub struct CenterDetector {
    backbone: Backbone,
    heat: Sequential,
    size: Sequential,
    offset: Sequential,
    species: Option<Sequential>,
    num_species: usize,
    channels: usize,
}

impl CenterDetector {
    /// `CenterDetector(backbone, num_species)`.
    pub fn new(cfg: &ModelConfig) -> Result<Self> {
        let backbone = Backbone::new(&cfg.backbone())?;
        let c = backbone.out_channels();
        let num_species = cfg.num_species();
        Ok(Self {
            heat: head("heat", c, 1, heat_bias_prior(0.01)),
            size: head("size", c, 2, 0.0),
            offset: head("offset", c, 2, 0.0),
            species: (num_species > 0).then(|| head("species", c, num_species, 0.0)),
            backbone,
            num_species,
            channels: c,
        })
    }

    /// `CenterDetector.num_species`; `0` means the species head is absent.
    pub fn num_species(&self) -> usize {
        self.num_species
    }

    /// `CenterDetector.stride`, the stride of the prediction maps.
    pub fn stride(&self) -> usize {
        self.backbone.stride()
    }

    /// The FPN map width.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The output names of [`CenterDetector::forward`]: `heat`, `size`, `offset`, plus `species` when
    /// the species head exists.
    pub fn output_names(&self) -> &'static [&'static str] {
        if self.species.is_some() {
            &OUTPUTS
        } else {
            &OUTPUTS[..3]
        }
    }

    /// `CenterDetector.forward(images)`, the raw training maps in dict order.
    pub fn forward(
        &self,
        g: &mut Graph,
        images: NodeId,
        ctx: &mut Ctx,
    ) -> Result<Vec<(&'static str, NodeId)>> {
        let (feat, _) = self.backbone.forward(g, images, ctx)?;
        let mut out = vec![
            (OUTPUTS[0], self.heat.forward(g, feat, ctx)),
            (OUTPUTS[1], self.size.forward(g, feat, ctx)),
            (OUTPUTS[2], self.offset.forward(g, feat, ctx)),
        ];
        if let Some(species) = &self.species {
            out.push((OUTPUTS[3], species.forward(g, feat, ctx)));
        }
        Ok(out)
    }

    /// `model.load_state_dict(...)`: copies every matching tensor of `weights` in.
    pub fn load_weights(&mut self, weights: &Weights) -> LoadReport {
        crate::models::layers::load_into(self, weights)
    }
}

impl LayerTree for CenterDetector {
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer)) {
        self.backbone.each_layer(f);
        f(&self.heat);
        f(&self.size);
        f(&self.offset);
        if let Some(species) = &self.species {
            f(species);
        }
    }

    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer)) {
        self.backbone.each_layer_mut(f);
        f(&mut self.heat);
        f(&mut self.size);
        f(&mut self.offset);
        if let Some(species) = &mut self.species {
            f(species);
        }
    }
}
