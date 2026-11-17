//! The settings of a pipeline config that the apps edit in a form: thresholds, tracking, smoothing
//! and behavior. Checkpoint paths are left out (profiles choose the models), as are values tied to
//! a trained network, such as the input and crop sizes.
//!
//! Values are read with inheritance (`extends:`) and written into the library's YAML file. The
//! default pipeline is read-only; duplicating it gives an editable config that extends it.

use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value as Json};
use serde_yaml::{Mapping, Value as Yaml};

use super::library::{Library, CONFIG_EXTENSIONS};
use crate::data::config::Config;

/// How a setting is edited.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Kind {
    Bool,
    Int {
        min: i64,
        max: i64,
    },
    /// An integer where 0 is written as `null` ("no limit").
    Limit {
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
        step: f64,
    },
    Choice {
        options: &'static [&'static str],
    },
}

/// One editable setting.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Field {
    /// The dotted config key, e.g. `pipeline.det_thresh`.
    pub(crate) key: &'static str,
    pub(crate) label: &'static str,
    #[serde(flatten)]
    pub(crate) kind: Kind,
    /// The value when the config does not set it: the pipeline's built-in default.
    #[serde(skip)]
    pub(crate) default: Default,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Default {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(&'static str),
}

/// Settings in groups, in the order the forms show them.
pub(crate) const SECTIONS: &[(&str, &[Field])] = &[
    (
        "General",
        &[field(
            "device",
            "Device",
            Kind::Choice {
                options: &["auto", "cpu"],
            },
            Default::Str("auto"),
        )],
    ),
    (
        "Detection",
        &[
            float(
                "pipeline.det_thresh",
                "Detection Threshold",
                0.0,
                1.0,
                0.01,
                0.3,
            ),
            int("pipeline.max_det", "Maximum Detections", 1, 50, 20),
            float("pipeline.merge_iou", "Merge Overlap", 0.0, 1.0, 0.05, 0.5),
        ],
    ),
    (
        "Pose",
        &[
            float(
                "pipeline.kp_conf_thresh",
                "Keypoint Threshold",
                0.0,
                1.0,
                0.01,
                0.3,
            ),
            float(
                "pipeline.crop_padding",
                "Crop Padding",
                1.0,
                2.0,
                0.05,
                1.25,
            ),
            boolean("pipeline.pair_decoding", "Separate Left and Right", true),
            boolean(
                "pipeline.pair_identity",
                "Keep Left and Right Over Time",
                true,
            ),
            float(
                "pipeline.pair_motion_frac",
                "Side Switch Penalty",
                0.0,
                1.0,
                0.01,
                0.1,
            ),
        ],
    ),
    (
        "Background",
        &[
            boolean("pipeline.background_filter", "Background Filter", false),
            int(
                "pipeline.background_frames",
                "Background Frames",
                8,
                512,
                64,
            ),
            float(
                "pipeline.foreground_diff",
                "Foreground Difference",
                0.0,
                255.0,
                1.0,
                25.0,
            ),
            float(
                "pipeline.min_foreground",
                "Minimum Foreground",
                0.0,
                1.0,
                0.01,
                0.08,
            ),
            boolean("pipeline.blob_arena_roi", "Arena Only", true),
            int("pipeline.blob_open_px", "Tail Removal", 0, 32, 5),
        ],
    ),
    (
        "Tracking",
        &[
            field(
                "tracker.max_animals",
                "Maximum Animals",
                Kind::Limit { max: 50 },
                Default::Int(0),
            ),
            float(
                "tracker.high_thresh",
                "Track Threshold",
                0.0,
                1.0,
                0.01,
                0.5,
            ),
            float(
                "tracker.low_thresh",
                "Recovery Threshold",
                0.0,
                1.0,
                0.01,
                0.2,
            ),
            float(
                "tracker.match_thresh",
                "Match Threshold",
                0.0,
                1.0,
                0.01,
                0.8,
            ),
            int("tracker.n_init", "Frames to Confirm", 1, 30, 3),
            int("tracker.max_lost", "Frames to Keep Lost", 0, 600, 90),
            field(
                "pipeline.embedder",
                "Identity",
                Kind::Choice {
                    options: &["model", "color", "none"],
                },
                Default::Str("model"),
            ),
        ],
    ),
    (
        "Smoothing",
        &[int("pipeline.temporal_window", "Window", 8, 512, 64)],
    ),
    (
        "Behavior",
        &[
            float(
                "pipeline.behavior_conf",
                "Keypoint Threshold",
                0.0,
                1.0,
                0.01,
                0.6,
            ),
            int("pipeline.behavior_top_k", "Behaviors per Frame", 1, 10, 3),
            int("pipeline.behavior_smooth", "Smoothing", 1, 61, 5),
            boolean("pipeline.behavior_prior_adapt", "Adapt to Recording", true),
        ],
    ),
    (
        "Camera Drift",
        &[
            boolean("drift.enabled", "Correct Camera Drift", true),
            int("drift.iterations", "Steps per Level", 1, 200, 30),
            int("drift.samples", "Pixels per Level", 1000, 1_000_000, 40_000),
            int("drift.coarsest", "Coarsest Level (px)", 16, 512, 64),
            float("drift.huber", "Robust Weight (σ)", 0.5, 5.0, 0.05, 1.345),
        ],
    ),
    (
        "Experiment",
        &[
            field(
                "assay.experiment",
                "Experiment",
                Kind::Choice {
                    options: &[
                        "none",
                        "open_field",
                        "plus_maze",
                        "three_chamber",
                        "grooming",
                    ],
                },
                Default::Str("none"),
            ),
            float(
                "assay.openfield.center_frac",
                "Centre Size",
                0.1,
                0.9,
                0.05,
                0.5,
            ),
            float(
                "assay.contact.dist_bl",
                "Contact Distance (BL)",
                0.0,
                0.5,
                0.01,
                0.05,
            ),
            float(
                "assay.contact.facing_deg",
                "Facing Angle (°)",
                10.0,
                180.0,
                5.0,
                60.0,
            ),
            float(
                "assay.groom.speed_bl_s",
                "Grooming Speed (BL/s)",
                0.0,
                2.0,
                0.05,
                0.25,
            ),
            float(
                "assay.groom.compact",
                "Grooming Posture",
                0.0,
                1.0,
                0.05,
                0.8,
            ),
            float(
                "assay.groom.min_bout_s",
                "Shortest Grooming (s)",
                0.2,
                10.0,
                0.1,
                1.5,
            ),
            float(
                "assay.rear.nose_z_bl",
                "Rearing Height (BL)",
                0.1,
                2.0,
                0.05,
                0.65,
            ),
            float("assay.maze.wall_ratio", "Wall Contrast", 1.0, 5.0, 0.1, 1.5),
            float(
                "assay.gap.max_fill_s",
                "Longest Filled Gap (s)",
                0.0,
                5.0,
                0.1,
                0.5,
            ),
            float("assay.qa.min_valid3d", "3D Needed", 0.0, 1.0, 0.05, 0.8),
        ],
    ),
];

const fn field(key: &'static str, label: &'static str, kind: Kind, default: Default) -> Field {
    Field {
        key,
        label,
        kind,
        default,
    }
}

const fn boolean(key: &'static str, label: &'static str, default: bool) -> Field {
    field(key, label, Kind::Bool, Default::Bool(default))
}

const fn int(key: &'static str, label: &'static str, min: i64, max: i64, default: i64) -> Field {
    field(key, label, Kind::Int { min, max }, Default::Int(default))
}

const fn float(
    key: &'static str,
    label: &'static str,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
) -> Field {
    field(
        key,
        label,
        Kind::Float { min, max, step },
        Default::Float(default),
    )
}

fn find(key: &str) -> Result<&'static Field> {
    SECTIONS
        .iter()
        .flat_map(|(_, fields)| fields.iter())
        .find(|f| f.key == key)
        .with_context(|| format!("{key} is not an editable setting"))
}

/// The form for config `name` (`None`: the default pipeline): sections of fields with their
/// current values, and whether the file may be changed.
pub(crate) fn form(library: &Library, name: Option<&str>) -> Result<Json> {
    let path = library.config_path(name)?;
    let config = Config::load(&path, &[])?;
    let sections: Vec<Json> = SECTIONS
        .iter()
        .map(|(title, fields)| {
            let fields: Vec<Json> = fields
                .iter()
                .map(|f| {
                    let mut item = serde_json::to_value(f).unwrap_or_default();
                    item["value"] = value(&config, f);
                    item
                })
                .collect();
            json!({"title": title, "fields": fields})
        })
        .collect();
    Ok(json!({"name": name, "editable": name.is_some(), "sections": sections}))
}

/// The value a field has in `config`, or its default.
pub(crate) fn value(config: &Config, field: &Field) -> Json {
    let set = config.get(field.key).filter(|v| !v.is_null());
    match (field.kind, field.default) {
        (Kind::Bool, Default::Bool(d)) => json!(set.map_or(d, |v| v.as_bool())),
        (Kind::Limit { .. }, _) => json!(set.and_then(|v| v.as_i64()).unwrap_or(0)),
        (Kind::Int { .. }, Default::Int(d)) => json!(set.and_then(|v| v.as_i64()).unwrap_or(d)),
        (Kind::Float { .. }, Default::Float(d)) => {
            json!(set.and_then(|v| v.as_f64()).unwrap_or(d))
        }
        // Any device but `cpu` means the GPU when there is one, so `mps` shows as `auto`.
        (Kind::Choice { options }, Default::Str(d)) => json!(set
            .and_then(|v| v.as_str())
            .filter(|v| options.contains(v))
            .unwrap_or(d)),
        _ => Json::Null,
    }
}

/// Sets `key` in library config `name` to `value`, checked against the field.
pub(crate) fn set(library: &Library, name: &str, key: &str, value: &Json) -> Result<()> {
    let field = find(key)?;
    let yaml = match field.kind {
        Kind::Bool => Yaml::Bool(value.as_bool().context("expected true or false")?),
        Kind::Int { min, max } => {
            let n = value.as_f64().context("expected a number")?.round() as i64;
            Yaml::Number(n.clamp(min, max).into())
        }
        Kind::Limit { max } => match value.as_f64().context("expected a number")?.round() as i64 {
            n if n <= 0 => Yaml::Null,
            n => Yaml::Number(n.min(max).into()),
        },
        Kind::Float { min, max, .. } => {
            let x = value.as_f64().context("expected a number")?;
            ensure!(x.is_finite(), "expected a number");
            Yaml::Number(x.clamp(min, max).into())
        }
        Kind::Choice { options } => {
            let choice = value.as_str().context("expected a choice")?;
            ensure!(
                options.contains(&choice),
                "{choice} is not an option for {key}"
            );
            Yaml::String(choice.to_owned())
        }
    };
    let path = library.config_path(Some(name))?;
    let mut root = read(&path)?;
    let mut node = &mut root;
    let parts: Vec<&str> = key.split('.').collect();
    for part in &parts[..parts.len() - 1] {
        let map = node
            .as_mapping_mut()
            .context("the config is not a mapping")?;
        let entry = map
            .entry(Yaml::String((*part).to_owned()))
            .or_insert_with(|| Yaml::Mapping(Mapping::new()));
        if !entry.is_mapping() {
            *entry = Yaml::Mapping(Mapping::new());
        }
        node = entry;
    }
    node.as_mapping_mut()
        .context("the config is not a mapping")?
        .insert(Yaml::String(parts[parts.len() - 1].to_owned()), yaml);
    write(&path, &root)
}

/// A new library config `new_name` that extends `source` (`None`: the default pipeline), so the
/// checkpoints and every setting carry over until changed. Returns the stored file name.
pub(crate) fn duplicate(library: &Library, source: Option<&str>, new_name: &str) -> Result<String> {
    let stem = super::recording::safe_name(new_name.trim());
    let base = library.config_path(source)?.canonicalize()?;
    let dir = library.root.join("configs");
    fs::create_dir_all(&dir)?;
    let mut name = format!("{stem}.yaml");
    for n in 2.. {
        if !dir.join(&name).exists() {
            break;
        }
        name = format!("{stem}-{n}.yaml");
    }
    let mut root = Mapping::new();
    root.insert(
        Yaml::String("extends".into()),
        Yaml::String(base.to_string_lossy().into_owned()),
    );
    write(&dir.join(&name), &Yaml::Mapping(root))?;
    Ok(name)
}

/// Removes library config `name`.
pub(crate) fn delete(library: &Library, name: &str) -> Result<()> {
    let path = library.config_path(Some(name))?;
    let used_by: Vec<String> = library
        .profiles()
        .into_iter()
        .filter(|p| p.config.as_deref() == Some(name))
        .map(|p| p.name)
        .collect();
    if !used_by.is_empty() {
        bail!("{name} is used by {}.", used_by.join(", "));
    }
    ensure!(
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| CONFIG_EXTENSIONS.contains(&e)),
        "{name} is not a config"
    );
    fs::remove_file(&path).with_context(|| format!("cannot delete {name}"))
}

fn read(path: &Path) -> Result<Yaml> {
    let text =
        fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let value: Yaml = serde_yaml::from_str(&text)?;
    Ok(if value.is_null() {
        Yaml::Mapping(Mapping::new())
    } else {
        value
    })
}

fn write(path: &Path, value: &Yaml) -> Result<()> {
    let dir = path.parent().context("the config has no folder")?;
    let mut file = tempfile::NamedTempFile::new_in(dir)?;
    std::io::Write::write_all(file.as_file_mut(), serde_yaml::to_string(value)?.as_bytes())?;
    file.persist(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::predict::PipelineCfg;

    /// The form's defaults are the pipeline's: a config that spells every default out runs the
    /// same as an empty one.
    #[test]
    fn defaults_match_the_pipeline() {
        let mut spelled = Vec::new();
        for (_, fields) in SECTIONS {
            for f in fields.iter() {
                let value = match f.default {
                    Default::Bool(b) => b.to_string(),
                    Default::Int(0) if matches!(f.kind, Kind::Limit { .. }) => "null".into(),
                    Default::Int(n) => n.to_string(),
                    Default::Float(x) => format!("{x:?}"),
                    Default::Str(s) => s.to_owned(),
                };
                spelled.push(format!("{}={value}", f.key));
            }
        }
        let empty = Config::parse("{}", &[]).unwrap();
        let full = Config::parse("{}", &spelled).unwrap();
        assert_eq!(
            format!("{:?}", PipelineCfg::from_config(&empty)),
            format!("{:?}", PipelineCfg::from_config(&full))
        );
        assert_eq!(
            format!("{:?}", crate::cli::predict::tracker_config(&empty)),
            format!("{:?}", crate::cli::predict::tracker_config(&full))
        );
        // The experiment fields, with an experiment chosen so they are read.
        let chosen = ["assay.experiment=open_field".to_owned()];
        let empty = Config::parse("{}", &chosen).unwrap();
        let mut both = spelled.clone();
        both.extend(chosen);
        let full = Config::parse("{}", &both).unwrap();
        assert_eq!(
            crate::assay::AssayConfig::from_config(&empty),
            crate::assay::AssayConfig::from_config(&full)
        );
    }

    #[test]
    fn duplicates_inherit_and_edits_are_written_back() {
        let dir = tempfile::tempdir().unwrap();
        let library = Library::open(dir.path().to_path_buf());
        let base = dir.path().join("base.yaml");
        fs::create_dir_all(dir.path().join("configs")).unwrap();
        fs::write(
            dir.path().join("configs/base.yaml"),
            "checkpoints: {pose: p.pt}\npipeline: {det_thresh: 0.42}\n",
        )
        .unwrap();
        let _ = base;
        let copy = duplicate(&library, Some("base.yaml"), "Open Field").unwrap();
        assert_eq!(copy, "Open-Field.yaml");
        let read_value = |key: &str| {
            let form = form(&library, Some(&copy)).unwrap();
            form["sections"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|s| s["fields"].as_array().unwrap().clone())
                .find(|f| f["key"] == key)
                .unwrap()["value"]
                .clone()
        };
        assert_eq!(read_value("pipeline.det_thresh"), json!(0.42));
        set(&library, &copy, "pipeline.det_thresh", &json!(0.6)).unwrap();
        set(&library, &copy, "tracker.max_animals", &json!(2)).unwrap();
        set(&library, &copy, "pipeline.temporal_window", &json!(10_000)).unwrap();
        assert_eq!(read_value("pipeline.det_thresh"), json!(0.6));
        assert_eq!(read_value("tracker.max_animals"), json!(2));
        assert_eq!(read_value("pipeline.temporal_window"), json!(512));
        set(&library, &copy, "tracker.max_animals", &json!(0)).unwrap();
        assert_eq!(read_value("tracker.max_animals"), json!(0));
        assert!(set(&library, &copy, "checkpoints.pose", &json!("x")).is_err());
        assert!(set(&library, &copy, "pipeline.embedder", &json!("magic")).is_err());
        let loaded = Config::load(library.config_path(Some(&copy)).unwrap(), &[]).unwrap();
        assert_eq!(
            loaded.get("checkpoints.pose").and_then(|v| v.as_str()),
            Some("p.pt")
        );
        assert_eq!(form(&library, None).unwrap()["editable"], json!(false));
        delete(&library, &copy).unwrap();
        assert!(!library.config_path(Some(&copy)).unwrap().exists());
    }
}
