//! Tests.
//!
//! Two kinds of check:
//!
//! * **torch fixtures** — `animalpose fixtures models` writes one JSON file per network under
//!   `tests/fixtures/models/` (config, inputs, outputs and the whole state dict, plus the same weights
//!   as a `.safetensors` file). Each fixture is rebuilt from its config, filled with those weights
//!   through the production `safetensors` path, and its outputs are compared with the dumped ones at a
//!   `2e-4` absolute tolerance. When the directory is absent — a fresh clone, or a CI job without torch
//!   — the test prints a line and passes.
//! * **structural checks** — output shapes, the unsupported-backbone error and the weight round-trip;
//!   these need nothing but this crate and run on every commit.

use super::*;
use crate::models::config::ModelConfig;
use crate::models::weights::Weights;
use crate::tensor::{Ctx, Graph, NodeId};
use serde::{Deserialize, Deserializer};
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Absolute tolerance against the torch fixtures.
const TOL: f32 = 2e-4;

/// One tensor of a fixture.
#[derive(Debug, Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    #[serde(deserialize_with = "data_of")]
    data: Vec<f32>,
}

/// Reads the flat values of a fixture tensor: `null` stands for NaN, which strict JSON cannot write.
fn data_of<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = Vec::<Option<f32>>::deserialize(deserializer)?;
    Ok(values
        .into_iter()
        .map(|value| value.unwrap_or(f32::NAN))
        .collect())
}

/// One dumped torch reference run: the config, the inputs, the outputs and the state dict.
///
/// A fixture from a trained checkpoint leaves [`Fixture::weights`] empty — its tensors are tens of
/// megabytes, which live only in the sibling `.safetensors` — and names that file instead.
#[derive(Debug, Deserialize)]
struct Fixture {
    config: serde_json::Value,
    inputs: BTreeMap<String, Tensor>,
    outputs: BTreeMap<String, Tensor>,
    #[serde(default)]
    weights: BTreeMap<String, Tensor>,
}

impl Fixture {
    /// The largest absolute difference between a fixture tensor and what the tape produced.
    fn error(&self, name: &str, got: &[f32], shape: &[usize]) -> f32 {
        let want = self.outputs.get(name).unwrap_or_else(|| {
            panic!(
                "the fixture has no output {name}, has {:?}",
                self.outputs.keys()
            )
        });
        assert_eq!(
            want.shape, shape,
            "output {name} has shape {shape:?}, the fixture wants {:?}",
            want.shape
        );
        assert_eq!(
            want.data.len(),
            got.len(),
            "output {name} has the wrong length"
        );
        want.data
            .iter()
            .zip(got)
            // A NaN reference (an unobserved coordinate) only matches a NaN of ours.
            .fold(0.0f32, |acc, (&a, &b)| {
                acc.max(if a.is_nan() {
                    if b.is_nan() {
                        0.0
                    } else {
                        f32::INFINITY
                    }
                } else if b.is_nan() {
                    f32::INFINITY
                } else {
                    (a - b).abs()
                })
            })
    }
}

/// The directory the model fixtures live in: `$ANIMALPOSE_FIXTURES/models` (or the variable itself when
/// it names the model directory), else the repository's `tests/fixtures/models`, found by walking up
/// from the crate directory `cargo test` runs in.
fn fixtures_dir() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("ANIMALPOSE_FIXTURES") {
        let base = PathBuf::from(dir);
        let nested = base.join("models");
        if nested.is_dir() {
            return Some(nested);
        }
        if base.is_dir() {
            return Some(base);
        }
    }
    let mut at = env::current_dir().ok()?;
    for _ in 0..6 {
        let candidate = at.join("tests").join("fixtures").join("models");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !at.pop() {
            break;
        }
    }
    None
}

/// Every fixture file in `dir`, in name order, skipping the manifest.
fn fixtures(dir: &Path) -> Vec<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "json")
                && path.file_stem().is_some_and(|stem| stem != "manifest")
        })
        .collect();
    files.sort();
    files
}

/// The model kind of a fixture, from its `<kind>_<name>` file stem.
fn kind_of(path: &Path) -> ModelKind {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let kind = stem.split('_').next().unwrap_or("");
    ModelKind::from_str(kind).unwrap_or_else(|err| panic!("{stem}: {err}"))
}

/// Fills a model from a fixture's JSON weights.
fn json_weights(fx: &Fixture) -> Weights {
    let mut weights = Weights::new();
    for (name, tensor) in &fx.weights {
        weights.insert(name.clone(), &tensor.shape, tensor.data.clone());
    }
    weights
}

/// Runs one fixture: build from the config, load the weights the way production does, forward the
/// inputs and compare every output with torch.
fn check(path: &Path) {
    let text = std::fs::read_to_string(path).expect("fixture is not readable");
    let fx: Fixture = serde_json::from_str(&text).expect("fixture is not valid JSON");
    let kind = kind_of(path);
    let cfg = ModelConfig::from_json(&fx.config).expect("fixture config is not a model config");

    // The `.safetensors` sibling is the production loading path; when it is absent (it is git-ignored,
    // and only the JSON is guaranteed to be checked in) fall back to the JSON weights.
    let sidecar = path.with_extension("safetensors");
    if !sidecar.is_file() && fx.weights.is_empty() {
        println!(
            "skipping {}: its weights are only in the absent {}",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
            sidecar.file_name().and_then(|s| s.to_str()).unwrap_or("?")
        );
        return;
    }
    let weights = if sidecar.is_file() {
        let (loaded_cfg, loaded_kind, weights) =
            load(&sidecar).unwrap_or_else(|err| panic!("{}: {err}", sidecar.display()));
        assert_eq!(loaded_kind, kind, "{}: wrong kind", sidecar.display());
        assert_eq!(
            loaded_cfg,
            cfg,
            "{}: the metadata config differs from the fixture config",
            sidecar.display()
        );
        let mut worst = 0.0f32;
        for (name, tensor) in &fx.weights {
            let got = weights
                .data(name)
                .unwrap_or_else(|| panic!("the safetensors file has no {name}"));
            assert_eq!(
                weights.shape(name).unwrap(),
                tensor.shape.as_slice(),
                "{name}: wrong shape"
            );
            worst = worst.max(
                tensor
                    .data
                    .iter()
                    .zip(got)
                    .fold(0.0f32, |acc, (&a, &b)| acc.max((a - b).abs())),
            );
        }
        assert!(
            worst <= 1e-6,
            "safetensors round trip of {} deviates by {worst}",
            sidecar.display()
        );
        weights
    } else {
        eprintln!(
            "note: {} is missing, using the JSON weights",
            sidecar.display()
        );
        json_weights(&fx)
    };

    let mut model = Model::build(kind, &cfg, &Weights::new()).expect("cannot build the model");
    let report = model.load_weights(&weights);
    assert!(
        report.missing.is_empty(),
        "weights missing: {:?}",
        report.missing
    );

    let mut g = Graph::new();
    let mut ctx = Ctx::eval();
    let inputs: Vec<NodeId> = model
        .input_names()
        .iter()
        .map(|name| {
            let tensor = fx
                .inputs
                .get(*name)
                .unwrap_or_else(|| panic!("the fixture has no input {name}"));
            g.from_data(tensor.data.clone(), &tensor.shape)
        })
        .collect();
    let outs = model
        .forward(&mut ctx, &mut g, &inputs)
        .expect("forward failed");
    assert_eq!(outs.len(), model.output_names().len());

    let mut worst = 0.0f32;
    let mut errors: Vec<(&str, f32)> = Vec::new();
    for (name, node) in model.output_names().iter().zip(&outs) {
        let err = fx.error(name, g.value(*node), g.shape(*node));
        errors.push((name, err));
        worst = worst.max(err);
    }
    // The behaviour fixtures also dump `EgocentricFeatures`, so a failure says whether the hand-written
    // feature block or the convolution stack above it drifted.
    if kind == ModelKind::Behavior && fx.outputs.contains_key("features") {
        let Model::Behavior(network) = &model else {
            unreachable!("a behavior fixture builds a BehaviorTCN")
        };
        let coords = fx.inputs["coords"].shape.clone();
        let got = network.features().compute(
            &fx.inputs["coords"].data,
            &fx.inputs["mask"].data,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
        );
        let err = fx.error(
            "features",
            &got,
            &[coords[0], coords[1], network.features().out_dim()],
        );
        errors.push(("features", err));
        worst = worst.max(err);
    }
    println!(
        "{}: {}",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
        errors
            .iter()
            .map(|(name, err)| format!("{name} {err:.3e}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert!(worst <= TOL, "max abs error {worst} exceeds {TOL}");
}

#[test]
fn models_match_torch() {
    let Some(dir) = fixtures_dir() else {
        println!("no torch fixtures found, skipping model parity");
        return;
    };
    let files = fixtures(&dir);
    assert!(!files.is_empty(), "{} has no fixtures", dir.display());
    for path in &files {
        check(path);
    }
}

#[test]
fn weights_survive_a_round_trip() {
    let mut weights = Weights::new();
    weights.insert(
        "heat_queries",
        &[2, 3],
        vec![1.5, -0.25, 0.0, 4.0, 0.125, -7.0],
    );
    weights.insert("norm.std", &[3], vec![0.229, 0.224, 0.225]);
    weights.set_metadata(KIND_KEY, "pose");
    let bytes = weights.to_bytes().unwrap();
    let back = Weights::from_bytes(&bytes).unwrap();
    assert_eq!(back.len(), 2);
    assert_eq!(
        back.data("heat_queries").unwrap(),
        weights.data("heat_queries").unwrap()
    );
    assert_eq!(back.shape("heat_queries").unwrap(), [2, 3]);
    assert_eq!(back.metadata_value(KIND_KEY), Some("pose"));
    assert_eq!(back.metadata_value(CONFIG_KEY), None);
}

#[test]
fn checkpoint_metadata_round_trips_through_disk() {
    let dir = std::env::temp_dir().join("animalpose-models-checkpoint");
    let path = dir.join("pose_tiny.safetensors");
    let cfg: ModelConfig =
        serde_json::from_str(r#"{"num_keypoints": 5, "backbone": {"width": 8}, "extra_key": 3}"#)
            .unwrap();
    let mut weights = Weights::new();
    weights.insert("heat_bias", &[5], vec![-4.0; 5]);
    save(&path, ModelKind::Pose, &cfg, &mut weights).unwrap();
    let (loaded_cfg, kind, loaded) = load(&path).unwrap();
    assert_eq!(kind, ModelKind::Pose);
    assert_eq!(loaded_cfg.num_keypoints(), 5);
    assert_eq!(loaded_cfg.backbone().width(), 8);
    // Unknown keys survive, so a newer Python build still loads.
    assert_eq!(
        loaded_cfg.extra.get("extra_key").unwrap(),
        &serde_json::json!(3)
    );
    assert_eq!(loaded.data("heat_bias").unwrap(), vec![-4.0; 5]);
    let model = Model::build(kind, &loaded_cfg, &Weights::new()).unwrap();
    assert_eq!(model.kind(), ModelKind::Pose);
    assert!(model.slot_names().contains(&"heat_bias".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn forward_shapes_match_the_python_models() {
    let cfg: ModelConfig = serde_json::from_str(
        r#"{"backbone": {"name": "tiny", "width": 8, "out_channels": 16},
            "num_keypoints": 5, "embed_dim": 8, "dims": 2, "hidden": 8, "layers": 2,
            "bidirectional": false, "dropout": 0.0, "classes": ["a", "b"], "dilations": [1, 2],
            "num_species": 2}"#,
    )
    .unwrap();
    let mut g = Graph::new();
    let mut ctx = Ctx::eval();

    let image = g.constant(0.4, &[1, 3, 64, 64]);
    let pose = Model::build(ModelKind::Pose, &cfg, &Weights::new()).unwrap();
    let outs = pose.forward(&mut ctx, &mut g, &[image]).unwrap();
    assert_eq!(g.shape(outs[0]).as_slice(), [1, 5, 16, 16]); // heat
    assert_eq!(g.shape(outs[1]).as_slice(), [1, 10, 16, 16]); // offset
    assert_eq!(g.shape(outs[2]).as_slice(), [1, 8]); // embedding

    let detector = Model::build(ModelKind::Detector, &cfg, &Weights::new()).unwrap();
    let outs = detector.forward(&mut ctx, &mut g, &[image]).unwrap();
    assert_eq!(
        detector.output_names(),
        ["heat", "size", "offset", "species"]
    );
    assert_eq!(g.shape(outs[0]).as_slice(), [1, 1, 16, 16]);
    assert_eq!(g.shape(outs[3]).as_slice(), [1, 2, 16, 16]);

    let coords = g.constant(1.0, &[2, 16, 5, 2]);
    let mask = g.constant(1.0, &[2, 16, 5]);
    let behavior = Model::build(ModelKind::Behavior, &cfg, &Weights::new()).unwrap();
    let outs = behavior
        .forward(&mut ctx, &mut g, &[coords, mask])
        .expect("behavior forward");
    assert_eq!(behavior.output_names(), ["logits"]);
    assert_eq!(g.shape(outs[0]).as_slice(), [2, 16, 2]);

    let conf = g.constant(0.5, &[2, 16, 5]);
    let temporal = Model::build(ModelKind::Temporal, &cfg, &Weights::new()).unwrap();
    let outs = temporal
        .forward(&mut ctx, &mut g, &[coords, conf, mask])
        .expect("temporal forward");
    assert_eq!(
        temporal.output_names(),
        ["coords", "coords_norm", "log_sigma", "center", "scale"]
    );
    assert_eq!(g.shape(outs[0]).as_slice(), [2, 16, 5, 2]);
    assert_eq!(g.shape(outs[2]).as_slice(), [2, 16, 5]);
    assert_eq!(g.shape(outs[3]).as_slice(), [2, 1, 1, 2]);
    assert_eq!(g.shape(outs[4]).as_slice(), [2, 1, 1, 1]);
}

#[test]
fn unknown_backbones_and_priors_are_reported() {
    let cfg: ModelConfig =
        serde_json::from_str(r#"{"backbone": {"name": "resnet50"}, "num_keypoints": 5}"#).unwrap();
    let err = Model::build(ModelKind::Pose, &cfg, &Weights::new())
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("unsupported backbone"),
        "unexpected error: {err}"
    );
    let cfg: ModelConfig = serde_json::from_str(
        r#"{"num_keypoints": 5, "prior": "spline", "prior_bandwidth": 1.0, "dims": 2}"#,
    )
    .unwrap();
    let err = Model::build(ModelKind::Temporal, &cfg, &Weights::new())
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("unknown temporal prior"),
        "unexpected error: {err}"
    );
}

#[test]
fn a_checkpoint_must_cover_the_network() {
    let cfg: ModelConfig = serde_json::from_str(
        r#"{"backbone": {"name": "tiny", "width": 8, "out_channels": 16},
            "num_keypoints": 5, "embed_dim": 8}"#,
    )
    .unwrap();
    let mut weights = Weights::new();
    weights.insert("heat_bias", &[5], vec![-4.0; 5]);
    weights.insert(
        "backbone.encoder.layer3.0.num_batches_tracked",
        &[1],
        vec![9.0],
    );
    let mut model = Model::build(ModelKind::Pose, &cfg, &Weights::new()).unwrap();
    let report = model.load_weights(&weights);
    assert_eq!(report.loaded, ["heat_bias"]);
    // `num_batches_tracked` is a torch bookkeeping buffer this port does not store: reported, not fatal.
    assert!(
        report
            .ignored
            .iter()
            .any(|name| name.ends_with("num_batches_tracked")),
        "unexpected ignored set: {:?}",
        report.ignored
    );
    assert!(report.missing.iter().any(|name| name == "heat_queries"));
    let err = report.ensure_complete().unwrap_err();
    assert!(
        err.to_string().contains("does not provide"),
        "unexpected error: {err}"
    );

    // A tensor whose element count does not fit the slot is ignored, so the slot keeps its
    // initialisation values rather than being half-filled.
    let mut wrong = Weights::new();
    wrong.insert("heat_bias", &[3], vec![1.0, 2.0, 3.0]);
    let report = model.load_weights(&wrong);
    assert!(report.loaded.is_empty(), "loaded: {:?}", report.loaded);
    assert!(report.ignored.iter().any(|name| name == "heat_bias"));
}

#[test]
fn slot_names_follow_the_python_state_dict() {
    let cfg: ModelConfig = serde_json::from_str(
        r#"{"backbone": {"name": "tiny", "width": 4, "out_channels": 8},
                                 "num_keypoints": 3, "embed_dim": 4}"#,
    )
    .unwrap();
    let model = Model::build(ModelKind::Pose, &cfg, &Weights::new()).unwrap();
    let names = model.slot_names();
    for want in [
        "backbone.norm.mean",
        "backbone.encoder.stem.0.0.weight",
        "backbone.encoder.stem.1.1.weight",
        "backbone.encoder.stages.0.0.conv1.0.weight",
        "backbone.encoder.stages.0.0.conv2.1.bias",
        "backbone.encoder.stages.3.0.skip.0.weight",
        "backbone.fpn.lateral.0.weight",
        "backbone.fpn.smooth.0.weight",
        "heat_feat.0.0.weight",
        "heat_feat.2.bias",
        "off_feat.1.weight",
        "backbone.encoder.stages.0.0.conv1.1.running_mean",
        "heat_queries",
        "heat_bias",
        "off_queries",
        "embed.0.weight",
        "embed.2.bias",
    ] {
        assert!(names.iter().any(|n| n == want), "no parameter named {want}");
    }
    // torch's step counter is the one buffer this port does not store.
    assert!(
        !names.iter().any(|n| n.ends_with("num_batches_tracked")),
        "num_batches_tracked should not be a slot"
    );
}

#[test]
fn torch_backbone_names_are_mapped() {
    let cfg: ModelConfig = serde_json::from_str(
        r#"{"backbone": {"name": "resnet18", "out_channels": 16},
                                 "num_keypoints": 3, "embed_dim": 4}"#,
    )
    .unwrap();
    let model = Model::build(ModelKind::Pose, &cfg, &Weights::new()).unwrap();
    let names = model.slot_names();
    // `TorchvisionResNet` re-wraps torchvision's layers, so the checkpoint's keys are the wrapper's.
    for want in [
        "backbone.encoder.stem.0.weight",
        "backbone.encoder.stem.1.running_mean",
        "backbone.encoder.stages.0.0.conv1.weight",
        "backbone.encoder.stages.0.0.bn1.weight",
        "backbone.encoder.stages.1.0.bn2.bias",
        "backbone.encoder.stages.1.0.downsample.0.weight",
        "backbone.encoder.stages.1.0.downsample.1.weight",
        "backbone.encoder.stages.3.1.conv2.weight",
    ] {
        assert!(names.iter().any(|n| n == want), "no parameter named {want}");
    }
    // `stages.0.0` changes neither the width nor the resolution, so torchvision has no `downsample`.
    assert!(!names
        .iter()
        .any(|n| n.starts_with("backbone.encoder.stages.0.0.downsample")));

    let mut g = Graph::new();
    let x = g.constant(0.3, &[1, 3, 64, 64]);
    let mut ctx = Ctx::eval();
    let outs = model.forward(&mut ctx, &mut g, &[x]).unwrap();
    // The torchvision stem (conv stride 2, max pool stride 2) puts `layer1` at stride 4.
    assert_eq!(g.shape(outs[0]).as_slice(), [1, 3, 16, 16]);
}
