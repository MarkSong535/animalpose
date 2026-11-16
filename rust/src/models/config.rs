//! Model configuration: the `model:` block of the YAML configs, i.e. the `model_cfg` dict of
//! `src/animalpose/train/checkpoint.py`.
//!
//! Every key and default here is the one `BUILDERS` in that module reads
//! (`backbone: {name, width, out_channels, pretrained}`, `num_keypoints`, `embed_dim`, `context`,
//! `num_species`, `dims`, `hidden`, `layers`, `bidirectional`, `dropout`, `prior_bandwidth`, `prior`,
//! `classes`, `dilations`). Unknown keys are collected in [`ModelConfig::extra`] instead of failing, so
//! a checkpoint written by a newer Python build still loads, and comes back unchanged through
//! [`ModelConfig::to_json`].
//!
//! ```
//! let cfg: animalpose::models::config::ModelConfig = serde_json::from_str(
//!     r#"{"backbone": {"name": "tiny", "width": 8, "out_channels": 16}, "num_keypoints": 5}"#
//! ).unwrap();
//! assert_eq!(cfg.backbone().width(), 8);
//! assert_eq!(cfg.num_keypoints(), 5);
//! assert_eq!(cfg.embed_dim(), 64); // the `PoseNet` default
//! ```

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The `BEHAVIORS` tuple of `src/animalpose/labels.py`, the default `BehaviorTCN.classes`.
pub const BEHAVIORS: [&str; 5] = ["resting", "walking", "rearing", "grooming", "eating"];

/// The default `BehaviorTCN` dilations.
pub const DEFAULT_DILATIONS: [usize; 5] = [1, 2, 4, 8, 16];

/// The `backbone:` sub-dict of `build_backbone` in `models/backbone.py`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BackboneConfig {
    /// `"tiny"` ([`crate::models::backbone::TinyResNet`]) or `"resnet18"` (`TorchvisionResNet`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `TinyResNet(width)`, the stem and stage-1 channel count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    /// Channels of the FPN output map, `Backbone.out_channels`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_channels: Option<usize>,
    /// `TorchvisionResNet(pretrained)`, which this port ignores: weights come from the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pretrained: Option<bool>,
    /// Keys this port does not know about.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl BackboneConfig {
    /// `cfg.get("name", "tiny")`.
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("tiny")
    }

    /// `cfg.get("width", 32)`.
    pub fn width(&self) -> usize {
        self.width.unwrap_or(32)
    }

    /// `cfg.get("out_channels", 64)`.
    pub fn out_channels(&self) -> usize {
        self.out_channels.unwrap_or(64)
    }

    /// `cfg.get("pretrained", False)`.
    pub fn pretrained(&self) -> bool {
        self.pretrained.unwrap_or(false)
    }
}

/// The config of one network, the dict `build_model(kind, model_cfg)` consumes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// The `CenterDetector`/`PoseNet` encoder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backbone: Option<BackboneConfig>,
    /// Keypoints of the canonical vocabulary (`num_keypoints`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_keypoints: Option<usize>,
    /// `PoseNet.embed_dim`, the re-identification embedding width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embed_dim: Option<usize>,
    /// `PoseNet.context`, the whole-crop FiLM branch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<bool>,
    /// `CenterDetector.num_species`; `0` disables the species head.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_species: Option<usize>,
    /// Coordinate width, `dims`: 2 for image tracks, 3 for triangulated ones.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dims: Option<usize>,
    /// `TemporalLSTM.hidden` / `BehaviorTCN.hidden`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden: Option<usize>,
    /// `TemporalLSTM.layers`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<usize>,
    /// `TemporalLSTM.bidirectional`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bidirectional: Option<bool>,
    /// `TemporalLSTM.dropout` / `BehaviorTCN.dropout`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dropout: Option<f32>,
    /// `TemporalLSTM.prior_bandwidth`; `0` disables the interpolation prior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prior_bandwidth: Option<f32>,
    /// `TemporalLSTM.prior`: `"kernel"`, `"linear"` or `"hybrid"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prior: Option<String>,
    /// `BehaviorTCN.classes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<String>>,
    /// `BehaviorTCN.dilations`, one [`crate::models::behavior::TcnBlock`] per entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dilations: Option<Vec<usize>>,
    /// Keys this port does not know about, kept for round-tripping.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// The backbone dict, or `build_backbone`'s defaults when the config has none (`cfg.get("backbone")`
    /// returns `None` for the behaviour and temporal networks).
    pub fn backbone(&self) -> BackboneConfig {
        self.backbone.clone().unwrap_or_default()
    }

    /// `cfg.get("num_keypoints", 25)`.
    pub fn num_keypoints(&self) -> usize {
        self.num_keypoints.unwrap_or(25)
    }

    /// `cfg.get("embed_dim", 64)`.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim.unwrap_or(64)
    }

    /// `cfg.get("context", False)`.
    pub fn context(&self) -> bool {
        self.context.unwrap_or(false)
    }

    /// `cfg.get("num_species", 0)`.
    pub fn num_species(&self) -> usize {
        self.num_species.unwrap_or(0)
    }

    /// `cfg.get("dims", 2)`.
    pub fn dims(&self) -> usize {
        self.dims.unwrap_or(2)
    }

    /// `cfg.get("hidden", default)`; `BUILDERS` defaults it to 128 for the temporal network and 96 for
    /// the behaviour one, so the caller passes the kind-specific default.
    pub fn hidden_or(&self, default: usize) -> usize {
        self.hidden.unwrap_or(default)
    }

    /// `cfg.get("hidden", 128)`, the `TemporalLSTM` default.
    pub fn hidden(&self) -> usize {
        self.hidden_or(128)
    }

    /// `cfg.get("layers", 2)`.
    pub fn layers(&self) -> usize {
        self.layers.unwrap_or(2)
    }

    /// `cfg.get("bidirectional", True)`.
    pub fn bidirectional(&self) -> bool {
        self.bidirectional.unwrap_or(true)
    }

    /// `cfg.get("dropout", 0.1)`.
    pub fn dropout(&self) -> f32 {
        self.dropout.unwrap_or(0.1)
    }

    /// `cfg.get("prior_bandwidth", 0.0)`.
    pub fn prior_bandwidth(&self) -> f32 {
        self.prior_bandwidth.unwrap_or(0.0)
    }

    /// `cfg.get("prior", "kernel")`.
    pub fn prior(&self) -> &str {
        self.prior.as_deref().unwrap_or("kernel")
    }

    /// `tuple(cfg.get("classes", BEHAVIORS))`.
    pub fn classes(&self) -> Vec<String> {
        self.classes
            .clone()
            .unwrap_or_else(|| BEHAVIORS.iter().map(|s| (*s).to_string()).collect())
    }

    /// `cfg.get("dilations", (1, 2, 4, 8, 16))`.
    ///
    /// `build_model` does not forward the key to `BehaviorTCN`, so a checkpoint's stack is always the
    /// default; the accessor serves configs that state a stack of their own.
    pub fn dilations(&self) -> Vec<usize> {
        self.dilations
            .clone()
            .unwrap_or_else(|| DEFAULT_DILATIONS.to_vec())
    }

    /// Parses one config object, tolerating unknown keys.
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        ensure!(
            value.is_object(),
            "a model config must be a JSON object, got {value}"
        );
        Ok(serde_json::from_value(value.clone())?)
    }

    /// Parses a config from a JSON string, e.g. the `"animalpose"` metadata entry of a checkpoint.
    pub fn from_json_str(text: &str) -> Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    /// The config back as JSON, the form stored in the checkpoint metadata.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    /// The config as the compact JSON string the `"animalpose"` metadata key holds.
    pub fn to_json_string(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}
