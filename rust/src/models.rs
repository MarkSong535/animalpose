//! The animalpose networks in Rust: the detector, pose, temporal and behaviour models of
//! `src/animalpose/`, loadable from the same checkpoints as Python.
//!
//! [`crate::models::Model`] is built from a [`crate::models::config::ModelConfig`] — the `model:` block of the YAML configs, which is
//! also the `model_cfg` dict of `src/animalpose/train/checkpoint.py` and the `"animalpose"` metadata
//! entry of a `.safetensors` file — and filled with [`crate::models::weights::Weights`] read from that file. Weight
//! names are never renamed: a layer owns its tensors under the Python module path that produced them
//! (the table in [`crate::tensor::Layer`] covers the shared layers, each module here documents its
//! own), so `animalpose export safetensors` is the only bridge that is ever needed.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `train/checkpoint.py:build_model(kind, model_cfg)` | [`crate::models::Model::build`] |
//! | `train/checkpoint.py:load_model(path)` | [`crate::models::load`] + [`crate::models::Model::build`] |
//! | `models/backbone.py:Backbone` / `build_backbone` | [`crate::models::backbone::Backbone`] |
//! | `models/backbone.py:TinyResNet` / `FPN` / `Normalize` / `ResBlock` | [`crate::models::backbone::TinyResNet`] / [`crate::models::backbone::FPN`] / [`crate::models::backbone::Normalize`] / [`crate::models::backbone::res_block`] |
//! | `pose/model.py:PoseNet` | [`crate::models::pose::PoseNet`] |
//! | `detector/model.py:CenterDetector` / `_head` | [`crate::models::detector::CenterDetector`] / [`crate::models::detector::head`] |
//! | `behavior/model.py:BehaviorTCN` / `EgocentricFeatures` / `TCNBlock` | [`crate::models::behavior::BehaviorTCN`] / [`crate::models::behavior::EgocentricFeatures`] / [`crate::models::behavior::TcnBlock`] |
//! | `temporal/model.py:TemporalLSTM` (`normalizer`, `kernel_prior`, `linear_prior`) | [`crate::models::temporal::TemporalLSTM`] ([`crate::models::temporal::normalizer`], [`crate::models::temporal::kernel_prior`], [`crate::models::temporal::linear_prior`]) |
//! | `torch.load(...)["state_dict"]` | [`crate::models::weights::Weights::read`] |
//!
//! Not modelled, and reported as such rather than silently ignored: the export wrappers
//! (`PoseExport`, `DetectorExport`, `TemporalExport`, `BehaviorExport`) and the decode functions,
//! which live in `algos` (the pose / detector targets and losses are in [`crate::models::losses`]);
//! the encoders of `torchvision` other than `resnet18`; and the
//! `blocks` argument of `TinyResNet`, which no config changes.
//!
//! ```
//! use animalpose::models::config::ModelConfig;
//! use animalpose::models::weights::Weights;
//! use animalpose::models::{Model, ModelKind};
//!
//! let cfg: ModelConfig = serde_json::from_str(
//!     r#"{"backbone": {"name": "tiny", "width": 8, "out_channels": 16},
//!         "num_keypoints": 5, "embed_dim": 8}"#,
//! )
//! .unwrap();
//! let model = Model::build(ModelKind::Pose, &cfg, &Weights::new()).unwrap();
//! assert_eq!(model.output_names(), ["heat", "offset", "embedding"]);
//! ```

pub mod backbone;
pub mod behavior;
pub mod config;
pub mod detector;
pub mod layers;
pub mod losses;
pub mod pose;
pub mod temporal;
pub mod torch_pt;
pub mod weights;

use crate::models::config::ModelConfig;
use crate::models::layers::LoadReport;
use crate::models::weights::{Weights, CONFIG_KEY, KIND_KEY};
use crate::tensor::{Ctx, Graph, NodeId};
use anyhow::{Context, Result};
use std::path::Path;
use std::str::FromStr;

/// Which network a config describes: the `kind` entry of a checkpoint, i.e. the key of `BUILDERS` in
/// `train/checkpoint.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelKind {
    /// `crate::models::pose`, the top-down keypoint network (`pose/model.py`).
    Pose,
    /// `crate::models::detector`, the anchor-free animal detector (`detector/model.py`).
    Detector,
    /// `crate::models::behavior`, the frame-wise behaviour classifier (`behavior/model.py`).
    Behavior,
    /// `crate::models::temporal`, the keypoint-track smoother (`temporal/model.py`).
    Temporal,
    /// An arena detector (`scripts/train_arena.py`): the pose network over the whole frame, one
    /// point per arena landmark, its arena definition in the config's `arena` entry
    /// ([`crate::assay::arena_model`]).
    Arena,
}

impl ModelKind {
    /// The name this kind is stored under (`"pose"`, `"detector"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pose => "pose",
            Self::Detector => "detector",
            Self::Behavior => "behavior",
            Self::Temporal => "temporal",
            Self::Arena => "arena",
        }
    }
}

impl FromStr for ModelKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "pose" => Ok(Self::Pose),
            "detector" => Ok(Self::Detector),
            "behavior" => Ok(Self::Behavior),
            "temporal" => Ok(Self::Temporal),
            "arena" => Ok(Self::Arena),
            other => anyhow::bail!(
                "unknown model kind {other:?}: expected pose, detector, behavior, temporal or arena"
            ),
        }
    }
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The four animalpose networks.
pub enum Model {
    /// `PoseNet`.
    Pose(Box<pose::PoseNet>),
    /// `CenterDetector`.
    Detector(Box<detector::CenterDetector>),
    /// `BehaviorTCN`.
    Behavior(Box<behavior::BehaviorTCN>),
    /// `TemporalLSTM`.
    Temporal(Box<temporal::TemporalLSTM>),
}

impl Model {
    /// `build_model(kind, model_cfg)` followed by `model.load_state_dict(state_dict)`.
    ///
    /// A non-empty checkpoint must fill every slot of the network, as `load_state_dict` insists; an
    /// empty [`crate::models::weights::Weights`] builds the network at its initialisation values, which is how training from
    /// scratch starts.
    pub fn build(kind: ModelKind, cfg: &ModelConfig, weights: &Weights) -> Result<Self> {
        let mut model = match kind {
            ModelKind::Pose | ModelKind::Arena => Self::Pose(Box::new(pose::PoseNet::new(cfg)?)),
            ModelKind::Detector => Self::Detector(Box::new(detector::CenterDetector::new(cfg)?)),
            ModelKind::Behavior => Self::Behavior(Box::new(behavior::BehaviorTCN::new(cfg))),
            ModelKind::Temporal => Self::Temporal(Box::new(temporal::TemporalLSTM::new(cfg)?)),
        };
        if !weights.is_empty() {
            model.load_weights(weights).ensure_complete()?;
        }
        Ok(model)
    }

    /// [`crate::models::Model::build`] with the kind the config carries, i.e. the `"kind"` metadata entry a
    /// checkpoint stores next to its model config (see [`crate::models::load`]).
    pub fn from_config(cfg: &ModelConfig, weights: &Weights) -> Result<Self> {
        let kind = cfg
            .extra
            .get(KIND_KEY)
            .and_then(serde_json::Value::as_str)
            .context(
                "the model config has no \"kind\" key: pass the kind to `crate::models::Model::build`",
            )?;
        Self::build(ModelKind::from_str(kind)?, cfg, weights)
    }

    /// Which network this is.
    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Pose(_) => ModelKind::Pose,
            Self::Detector(_) => ModelKind::Detector,
            Self::Behavior(_) => ModelKind::Behavior,
            Self::Temporal(_) => ModelKind::Temporal,
        }
    }

    /// The inputs of [`crate::models::Model::forward`], in order: `crops` / `images` for the image networks,
    /// `coords, conf, observed` for the temporal one, `coords, mask` for the behaviour one.
    pub fn input_names(&self) -> &'static [&'static str] {
        match self {
            Self::Pose(_) => &pose::INPUTS,
            Self::Detector(_) => &detector::INPUTS,
            Self::Behavior(_) => &behavior::INPUTS,
            Self::Temporal(_) => &temporal::INPUTS,
        }
    }

    /// The outputs of [`crate::models::Model::forward`], in the order the Python `forward` returns them.
    pub fn output_names(&self) -> &'static [&'static str] {
        match self {
            Self::Pose(_) => &pose::OUTPUTS,
            Self::Detector(m) => m.output_names(),
            Self::Behavior(_) => &behavior::OUTPUTS,
            Self::Temporal(_) => &temporal::OUTPUTS,
        }
    }

    /// Runs the network, mirroring the matching Python `forward`. `inputs` are [`crate::models::Model::input_names`]
    /// in order; the result is [`crate::models::Model::output_names`] in order.
    pub fn forward(&self, ctx: &mut Ctx, g: &mut Graph, inputs: &[NodeId]) -> Result<Vec<NodeId>> {
        Ok(self
            .forward_named(ctx, g, inputs)?
            .into_iter()
            .map(|(_, n)| n)
            .collect())
    }

    /// [`crate::models::Model::forward`] with the output names attached, i.e. the dict the Python models return.
    pub fn forward_named(
        &self,
        ctx: &mut Ctx,
        g: &mut Graph,
        inputs: &[NodeId],
    ) -> Result<Vec<(&'static str, NodeId)>> {
        let expected = self.input_names().len();
        anyhow::ensure!(
            inputs.len() == expected,
            "{} takes {} input(s), got {}",
            self.kind(),
            expected,
            inputs.len()
        );
        match self {
            Self::Pose(m) => {
                let [heat, offset, embedding] = m.forward(g, inputs[0], ctx)?;
                Ok(vec![
                    (pose::OUTPUTS[0], heat),
                    (pose::OUTPUTS[1], offset),
                    (pose::OUTPUTS[2], embedding),
                ])
            }
            Self::Detector(m) => m.forward(g, inputs[0], ctx),
            Self::Behavior(m) => Ok(vec![(
                behavior::OUTPUTS[0],
                m.forward(g, inputs[0], inputs[1], ctx)?,
            )]),
            Self::Temporal(m) => Ok(m.forward(g, inputs[0], inputs[1], inputs[2], ctx)?),
        }
    }

    /// `model.load_state_dict(...)`: copies every matching tensor of `crate::models::weights` into the network.
    pub fn load_weights(&mut self, weights: &Weights) -> LoadReport {
        match self {
            Self::Pose(m) => m.load_weights(weights),
            Self::Detector(m) => m.load_weights(weights),
            Self::Behavior(m) => m.load_weights(weights),
            Self::Temporal(m) => m.load_weights(weights),
        }
    }

    /// The trainable parameter names, in slot order.
    pub fn param_names(&self) -> Vec<String> {
        match self {
            Self::Pose(m) => layers::param_names(&**m),
            Self::Detector(m) => layers::param_names(&**m),
            Self::Behavior(m) => layers::param_names(&**m),
            Self::Temporal(m) => layers::param_names(&**m),
        }
    }

    /// Every tensor name the network owns — trainable slots and buffers alike — i.e. the key set of
    /// the matching Python `state_dict()`.
    pub fn slot_names(&self) -> Vec<String> {
        match self {
            Self::Pose(m) => layers::slot_names(&**m),
            Self::Detector(m) => layers::slot_names(&**m),
            Self::Behavior(m) => layers::slot_names(&**m),
            Self::Temporal(m) => layers::slot_names(&**m),
        }
    }

    /// Numbers every trainable slot from `base`, so a training step can read its gradients back with
    /// [`crate::tensor::Graph::param_grads`].
    pub fn assign_slots(&mut self, base: usize) -> usize {
        match self {
            Self::Pose(m) => layers::assign_slots(&mut **m, base),
            Self::Detector(m) => layers::assign_slots(&mut **m, base),
            Self::Behavior(m) => layers::assign_slots(&mut **m, base),
            Self::Temporal(m) => layers::assign_slots(&mut **m, base),
        }
    }
}

/// Reads a checkpoint — a PyTorch `.pt` file written by `animalpose train` or a `.safetensors` file
/// written by `animalpose export safetensors` — as its model config, its kind and its weights (the
/// `kind`, `model_cfg` and `state_dict` entries of a Python checkpoint).
pub fn load(path: &Path) -> Result<(ModelConfig, ModelKind, Weights)> {
    let weights = Weights::read(path)?;
    let config = weights
        .metadata_value(CONFIG_KEY)
        .with_context(|| format!("{} has no {CONFIG_KEY:?} metadata", path.display()))?;
    let kind = weights
        .metadata_value(KIND_KEY)
        .with_context(|| format!("{} has no {KIND_KEY:?} metadata", path.display()))?;
    Ok((
        ModelConfig::from_json_str(config)?,
        ModelKind::from_str(kind)?,
        weights,
    ))
}

/// Writes the counterpart of a Python checkpoint: the weights plus the `"animalpose"` and `"kind"`
/// metadata entries [`crate::models::load`] reads back.
pub fn save(path: &Path, kind: ModelKind, cfg: &ModelConfig, weights: &mut Weights) -> Result<()> {
    weights.set_metadata(KIND_KEY, kind.as_str());
    weights.set_metadata(CONFIG_KEY, cfg.to_json_string()?);
    weights.write(path)
}

#[cfg(test)]
mod tests;
