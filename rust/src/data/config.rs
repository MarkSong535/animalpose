//! YAML config loading with dotted command-line overrides (`train.steps=200`) and `extends:`
//! inheritance, where a config merges the one file (or the list of files) it names underneath its
//! own keys before any override runs.
//!
//! Port of `src/animalpose/config.py:load_config`, which every Python entry point
//! (`animalpose train`, `animalpose predict`) uses: read the file with `yaml.safe_load`,
//! then apply each `key.path=value` override by walking (and creating) the nodes of
//! the dotted path and assigning `yaml.safe_load(value)` to the leaf. The same
//! `configs/*.yaml` files are read here, so the keys are those of
//! `configs/behavior.yaml` (`model.classes`, `data.window`, `train.batch_size`,
//! …), `configs/pose.yaml` and `configs/temporal.yaml`.
//!
//! A file may carry a top-level `extends:` naming one path or a list of them; each base resolves
//! relative to the directory of the file that declares it (an absolute path is taken as given), the
//! bases merge left to right and the declaring file's own keys land on top, mappings merging key by
//! key while every other value — a scalar, a string, a list — is replaced wholesale, and the
//! `extends` key itself is dropped so no caller sees it. Inheritance therefore resolves *before*
//! the `key.path=value` overrides, which may target a key that only the bases define. A base that
//! cannot be read, or a file that transitively extends itself, is an error naming the chain.
//!
//! ```
//! use animalpose::data::config::Config;
//!
//! let cfg = Config::parse(
//!     "train: {steps: 1000, lr: 5.0e-4}\ndata: {species: [mouse]}",
//!     &["train.steps=200".to_string(), "data.species=[mouse, rat]".to_string()],
//! )
//! .unwrap();
//! assert_eq!(cfg.usize_at("train.steps", 1), 200);
//! assert_eq!(cfg.f64_at("train.lr", 1.0), 5.0e-4);
//! assert_eq!(cfg.str_list_at("data.species"), ["mouse", "rat"]);
//! ```
//!
//! `// differs from Python:` PyYAML resolves a plain scalar with the YAML 1.1 rules,
//! which need a sign in an exponent (`5.0e-4` is a float there, `1.0e3` is a string);
//! serde_yaml follows YAML 1.2, where both are floats. Every exponent form the
//! `configs/*.yaml` files use (`5.0e-4`, `1.0e-3`) resolves the same way in both.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// One node of a config tree: the `dict` / `list` / scalar a YAML file parses to.
///
/// `// differs from Python:` mappings are sorted (Python dicts keep insertion order),
/// which only changes the order two `model:` keys appear in when a checkpoint's
/// metadata is re-serialised.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// `None` / `null` / an empty scalar (`yaml.safe_load("")`).
    #[default]
    Null,
    /// A YAML boolean.
    Bool(bool),
    /// An integer scalar (`batch_size: 32`).
    Int(i64),
    /// A float scalar (`lr: 5.0e-4`).
    Float(f64),
    /// A string scalar (`init: runs/pose.pt`).
    Str(String),
    /// A sequence (`species: [mouse, rat]`).
    List(Vec<Value>),
    /// A mapping (`train: {steps: 200}`).
    Map(BTreeMap<String, Value>),
}

impl Value {
    /// `true` for `null`, which the configs use to switch a source off (`real_root: null`).
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// The scalar as `f64`; integers widen, as they do when Python passes an `int` where a float is read.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// The scalar as `i64` (a float truncates towards zero, like `int(x)`).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            Self::Bool(v) => Some(i64::from(*v)),
            _ => None,
        }
    }

    /// The scalar as a non-negative integer; a negative value is not a valid count.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64()
            .and_then(|v| u64::try_from(v).ok())
            .map(|v| v as usize)
    }

    /// The scalar as a boolean.
    pub fn as_bool(&self) -> bool {
        match self {
            Self::Bool(v) => *v,
            Self::Int(v) => *v != 0,
            Self::Float(v) => *v != 0.0,
            Self::Null => false,
            Self::Str(v) => !v.is_empty(),
            Self::List(v) => !v.is_empty(),
            Self::Map(v) => !v.is_empty(),
        }
    }

    /// The scalar as a string; numbers are rendered the way `yaml.safe_dump` would.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(v) => Some(v),
            _ => None,
        }
    }

    /// The node as a sequence.
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Self::List(v) => Some(v),
            _ => None,
        }
    }

    /// The node as a mapping.
    pub fn map(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Self::Map(v) => Some(v),
            _ => None,
        }
    }

    /// `node[key]`: a mapping lookup, or a sequence lookup by decimal index.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Map(map) => map.get(key),
            Self::List(items) => key.parse::<usize>().ok().and_then(|i| items.get(i)),
            _ => None,
        }
    }

    /// `cfg["train"]["steps"]` for the dotted key `"train.steps"`; `None` when any step is missing.
    pub fn path(&self, dotted: &str) -> Option<&Value> {
        let mut node = self;
        for part in dotted.split('.') {
            node = node.get(part)?;
        }
        Some(node)
    }

    /// The node as a list of names; a lone string counts as a one-element list (Python configs accept both).
    pub fn str_list(&self) -> Vec<String> {
        match self {
            Self::List(items) => items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            Self::Str(s) => vec![s.clone()],
            _ => Vec::new(),
        }
    }

    /// The node as a list of numbers (`det_size: [480, 640]`).
    pub fn f64_list(&self) -> Vec<f64> {
        match self {
            Self::List(items) => items.iter().filter_map(Value::as_f64).collect(),
            _ => self.as_f64().map(|v| vec![v]).unwrap_or_default(),
        }
    }

    /// The node as JSON, so a typed struct can be deserialized from it.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Null => serde_json::Value::Null,
            Self::Bool(v) => serde_json::Value::Bool(*v),
            Self::Int(v) => serde_json::json!(*v),
            Self::Float(v) => serde_json::json!(*v),
            Self::Str(v) => serde_json::Value::String(v.clone()),
            Self::List(items) => {
                serde_json::Value::Array(items.iter().map(Self::to_json).collect())
            }
            Self::Map(map) => map
                .iter()
                .map(|(k, v)| (k.clone(), v.to_json()))
                .collect::<serde_json::Map<_, _>>()
                .into(),
        }
    }

    /// `node[parts[-1]] = value` after `node = node.setdefault(p, {})` for every leading part.
    ///
    /// Mirrors the override loop of `config.py:load_config`; a scalar on the way is an
    /// error here where Python raises `AttributeError`.
    fn set_path(&mut self, parts: &[&str], value: Value) -> Result<()> {
        ensure!(!parts.is_empty(), "an override needs a key");
        if parts.len() == 1 {
            if !matches!(self, Self::Map(_) | Self::Null) {
                bail!(
                    "cannot set {}/{}: the node is not a mapping",
                    parts[0],
                    value.kind_name()
                );
            }
            if self.is_null() {
                *self = Self::Map(BTreeMap::new());
            }
            if let Self::Map(map) = self {
                map.insert(parts[0].to_string(), value);
            }
            return Ok(());
        }
        if self.is_null() {
            *self = Self::Map(BTreeMap::new());
        }
        let Self::Map(map) = self else {
            bail!(
                "cannot override under {:?}: the node is not a mapping",
                parts[0]
            );
        };
        let entry = map
            .entry(parts[0].to_string())
            .or_insert_with(|| Self::Map(BTreeMap::new()));
        if entry.is_null() {
            *entry = Self::Map(BTreeMap::new());
        }
        entry.set_path(&parts[1..], value)
    }

    /// Name of the node's kind, for error messages.
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "bool",
            Self::Int(_) | Self::Float(_) => "number",
            Self::Str(_) => "string",
            Self::List(_) => "list",
            Self::Map(_) => "mapping",
        }
    }
}

/// A loaded config: the tree plus the file it came from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    value: Value,
    path: Option<String>,
}

impl Config {
    /// `load_config(path, overrides)`: the file with its `extends:` bases merged in, then each
    /// override, so an override may set a key the file only inherits.
    pub fn load(path: impl AsRef<Path>, overrides: &[String]) -> Result<Self> {
        let path = path.as_ref();
        let value = load_tree(path, &mut Vec::new())?;
        let mut cfg = Self {
            value,
            path: Some(path.display().to_string()),
        };
        for item in overrides {
            cfg.apply_override(item)
                .with_context(|| format!("{}: override {item:?}", path.display()))?;
        }
        Ok(cfg)
    }

    /// `load_config` for a config already in memory.
    ///
    /// A string has no directory, so a relative `extends:` there resolves against the current
    /// directory and a base that is not there is an error saying so; an absolute base always works.
    pub fn parse(text: &str, overrides: &[String]) -> Result<Self> {
        let value = merge_bases(parse_tree(text)?, None, &mut Vec::new())?;
        let mut cfg = Self { value, path: None };
        for item in overrides {
            cfg.apply_override(item)?;
        }
        Ok(cfg)
    }

    /// The file this config was read from, if any.
    pub fn source(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// The whole tree.
    pub fn value(&self) -> &Value {
        &self.value
    }

    /// The tree as JSON.
    pub fn json(&self) -> serde_json::Value {
        self.value.to_json()
    }

    /// `cfg.get(key)` for a dotted key: `None` when the key is absent or `null`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.value.path(key).filter(|v| !v.is_null())
    }

    /// The node of a dotted key, `null` and missing reported the same way `cfg.get` reports them.
    pub fn section(&self, key: &str) -> Option<&Value> {
        self.value.path(key)
    }

    /// The whole resolved tree, for a test that compares a file's config end to end.
    pub fn tree(&self) -> &Value {
        &self.value
    }

    /// `cfg.get(key, default)` as a string.
    pub fn str_at(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_string()
    }

    /// A required string, e.g. `checkpoints.pose`.
    pub fn require_str(&self, key: &str) -> Result<String> {
        self.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .with_context(|| format!("{} is missing from {}", key, self.describe()))
    }

    /// `cfg.get(key, default)` as a float (`train.lr`).
    pub fn f64_at(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(Value::as_f64).unwrap_or(default)
    }

    /// `cfg.get(key, default)` as an integer (`train.steps`).
    pub fn usize_at(&self, key: &str, default: usize) -> usize {
        self.get(key).and_then(Value::as_usize).unwrap_or(default)
    }

    /// `cfg.get(key, default)` as a bool (`pipeline.pair_decoding`).
    pub fn bool_at(&self, key: &str, default: bool) -> bool {
        self.get(key).map(Value::as_bool).unwrap_or(default)
    }

    /// `list(cfg.get(key, []))` of names (`data.species`, `data.session_dirs`).
    pub fn str_list_at(&self, key: &str) -> Vec<String> {
        self.get(key).map(Value::str_list).unwrap_or_default()
    }

    /// `cfg.get(key)` as a list of names, distinguishing "absent or `null`" (`None`) from an empty
    /// list: `data/session_groups=None` and a missing key behave alike in Python.
    pub fn opt_str_list_at(&self, key: &str) -> Option<Vec<String>> {
        self.get(key).map(Value::str_list)
    }

    /// `list(cfg.get(key, default))` of numbers (`pipeline.det_size`).
    pub fn f64_list_at(&self, key: &str, default: &[f64]) -> Vec<f64> {
        self.get(key)
            .map(Value::f64_list)
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| default.to_vec())
    }

    /// The `model:` block as JSON, i.e. the dict `train/checkpoint.py:build_model` consumes.
    ///
    /// An absent block becomes `{}`, the way `cfg["model"]` of a config without one would be a
    /// `KeyError` but `cfg.get("model", {})` is the empty dict the builders default from.
    pub fn model_json(&self) -> Result<serde_json::Value> {
        Ok(self
            .get("model")
            .map(Value::to_json)
            .unwrap_or_else(|| serde_json::json!({})))
    }

    /// The training block (`train:`) with the defaults `train/loops.py` reads.
    pub fn train(&self) -> TrainConfig {
        TrainConfig {
            steps: self.usize_at("train.steps", 2000),
            batch_size: self.usize_at("train.batch_size", 32),
            lr: self.f64_at("train.lr", 1e-3),
            weight_decay: self.f64_at("train.weight_decay", 1e-4),
            clip: self.f64_at("train.clip", 5.0),
            seed: self.usize_at("train.seed", 0),
            log_every: self.usize_at("train.log_every", 50),
            workers: self.usize_at("train.workers", 0),
        }
    }

    /// The data block (`data:`) with the defaults the keypoint trainers read.
    pub fn data(&self) -> DataConfig {
        let real_mapping = self.str_at("data.real_mapping", "homecage_dlc");
        let kp_subsets = {
            let list = self.str_list_at("data.kp_subsets");
            if list.is_empty() {
                vec![real_mapping.clone()]
            } else {
                list
            }
        };
        DataConfig {
            window: self.usize_at("data.window", 64),
            species: {
                let list = self.str_list_at("data.species");
                if list.is_empty() {
                    vec!["mouse".to_string(), "rat".to_string()]
                } else {
                    list
                }
            },
            seed: self.usize_at("data.seed", 0),
            pool_sims: self.usize_at("data.pool_sims", 64),
            pool_frames: self.usize_at("data.pool_frames", 600),
            session_dirs: self.str_list_at("data.session_dirs"),
            session_groups: self.opt_str_list_at("data.session_groups"),
            real_root: self
                .get("data.real_root")
                .and_then(Value::as_str)
                .map(str::to_string),
            real_animals: self.opt_str_list_at("data.real_animals"),
            real_fraction: self.f64_at("data.real_fraction", 0.5),
            real_mapping,
            kp_subsets,
            kp_subset_p: self.f64_at("data.kp_subset_p", 0.5),
        }
    }

    /// Applies one `key.path=value` override (`train.steps=200`, `data.species=[mouse, rat]`, `init=`).
    ///
    /// `// differs from Python:` the value is parsed as YAML 1.2 (see the module docs), and a value
    /// that fails to parse is taken literally, which `yaml.safe_load` also does for a plain string.
    pub fn apply_override(&mut self, item: &str) -> Result<()> {
        // `str.partition("=")`: the key runs to the first `=`, the value is everything after it.
        let (key, raw) = split_once(item, '=');
        ensure!(!key.is_empty(), "override {item:?} has no key");
        let value = parse_scalar(raw);
        let parts: Vec<&str> = key.split('.').collect();
        self.value.set_path(&parts, value)
    }

    /// How this config is named in error messages.
    fn describe(&self) -> String {
        self.path
            .clone()
            .unwrap_or_else(|| "the config".to_string())
    }
}

/// `cfg.get("train", {})` with the defaults of `train/loops.py`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// `steps`, which `train_temporal` / `train_behavior` default to 2000 and the image trainers to 1000.
    pub steps: usize,
    /// `batch_size` (`32` for the keypoint trainers).
    pub batch_size: usize,
    /// `lr`, `_optimizer`'s `cfg.get("lr", 1e-3)`.
    pub lr: f64,
    /// `weight_decay`, `_optimizer`'s `cfg.get("weight_decay", 1e-4)`.
    pub weight_decay: f64,
    /// `clip`, the `clip_grad_norm_` threshold of `_loop`.
    pub clip: f64,
    /// `seed`, `torch.manual_seed` and the batch-generator seed.
    pub seed: usize,
    /// `log_every`, `_log`'s cadence (`50` in Python).
    pub log_every: usize,
    /// `workers`, ignored here: batching is single-threaded.
    pub workers: usize,
}

/// `cfg.get("data", {})` for the keypoint (temporal / behavior) trainers.
#[derive(Clone, Debug, PartialEq)]
pub struct DataConfig {
    /// `window`, the number of frames per training sample (`64`).
    pub window: usize,
    /// `species`, `("mouse", "rat")` by default.
    pub species: Vec<String>,
    /// `seed`, offset by `100` for the simulation pool as `_pool` does.
    pub seed: usize,
    /// `pool_sims`, the number of pre-simulated sequences (`64`).
    pub pool_sims: usize,
    /// `pool_frames`, the length of each (`600`).
    pub pool_frames: usize,
    /// `session_dirs`, the labeled `.npz` sessions of `data/behavior_sets.py:load_session_dir`.
    pub session_dirs: Vec<String>,
    /// `session_groups`, the `animal` names those directories are filtered to (`None` keeps all).
    pub session_groups: Option<Vec<String>>,
    /// `real_root`, the home-cage DLC tree `data/tracks.py:load_sessions` reads (`.npz` here).
    pub real_root: Option<String>,
    /// `real_animals`, the recordings `load_sessions` keeps.
    pub real_animals: Option<Vec<String>>,
    /// `real_fraction`, the share of batches taken from real sessions (`0.5`).
    pub real_fraction: f64,
    /// `real_mapping`, the keypoint subset a real track is masked to (`homecage_dlc`).
    pub real_mapping: String,
    /// `kp_subsets`, the mappings `_KeypointSubset` draws its masks from.
    pub kp_subsets: Vec<String>,
    /// `kp_subset_p`, the probability that a batch is masked to one of them (`0.5`).
    pub kp_subset_p: f64,
}

/// `str.partition(sep)` as `(before, after)`; a missing separator leaves `after` empty.
fn split_once(text: &str, sep: char) -> (&str, &str) {
    match text.find(sep) {
        Some(at) => (&text[..at], &text[at + sep.len_utf8()..]),
        None => (text, ""),
    }
}

/// `yaml.safe_load(value)` for the right-hand side of an override.
///
/// A bare `-`, an empty string or an unparsable scalar comes back as a string, which is what
/// PyYAML returns for a plain scalar too (`--override model.backbone.name=tiny`).
fn parse_scalar(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Null;
    }
    serde_yaml::from_str(raw).unwrap_or_else(|_| Value::Str(raw.to_string()))
}

/// The key a config names the files it inherits from with.
const EXTENDS_KEY: &str = "extends";

/// `yaml.safe_load(text) or {}`: an empty file is an empty config.
fn parse_tree(text: &str) -> Result<Value> {
    if text.trim().is_empty() {
        return Ok(Value::Map(BTreeMap::new()));
    }
    serde_yaml::from_str(text).context("config is not valid YAML")
}

/// The tree of `path` with every file it `extends:` merged in and `extends` removed.
///
/// `stack` is the chain of files already being loaded, canonicalised so one file reached under two
/// spellings counts as reached once; meeting a file again is a cycle, not infinite recursion.
fn load_tree(path: &Path, stack: &mut Vec<PathBuf>) -> Result<Value> {
    let seen = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    if stack.contains(&seen) {
        let mut cycle = stack.clone();
        cycle.push(seen);
        bail!(
            "config inheritance cycles: {}",
            cycle
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(" -> ")
        );
    }
    stack.push(seen);
    let loaded = read_own_tree(path, stack);
    stack.pop();
    loaded
}

/// One file's own tree, read from disk and merged with the bases its `extends:` names.
fn read_own_tree(path: &Path, stack: &mut Vec<PathBuf>) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read the config {}", path.display()))?;
    merge_bases(parse_tree(&text)?, Some(path), stack)
}

/// `tree` with the bases its `extends:` names merged underneath, and without that key.
///
/// The bases merge left to right and the tree's own keys land on top. `from` is the file `tree`
/// was read from, which is what a relative base resolves against; `None` is a config parsed from a
/// string, which has no directory of its own.
fn merge_bases(tree: Value, from: Option<&Path>, stack: &mut Vec<PathBuf>) -> Result<Value> {
    let Some(bases) = extends_of(&tree)? else {
        return Ok(tree);
    };
    let mut own = tree;
    if let Value::Map(entries) = &mut own {
        entries.remove(EXTENDS_KEY);
    }
    let mut merged = Value::Map(BTreeMap::new());
    for base in &bases {
        merged = merge_value(&merged, &load_base(base, from, stack)?);
    }
    Ok(merge_value(&merged, &own))
}

/// The `extends:` entries of a tree: `None` when the key is absent, the paths when it is a path or
/// a list of them. `extends: null` and `extends: []` name no base, as an empty Python list does.
fn extends_of(tree: &Value) -> Result<Option<Vec<String>>> {
    let Some(node) = tree.get(EXTENDS_KEY) else {
        return Ok(None);
    };
    let names = match node {
        Value::Null => Vec::new(),
        Value::Str(one) => vec![one.clone()],
        Value::List(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .with_context(|| {
                        format!("an `{EXTENDS_KEY}` entry must be a path, not {}", item.kind_name())
                    })
            })
            .collect::<Result<Vec<_>>>()?,
        other => bail!(
            "`{EXTENDS_KEY}` must be a path or a list of paths, not {}",
            other.kind_name()
        ),
    };
    Ok(Some(names))
}

/// One base of a config, read with its own bases.
fn load_base(name: &str, from: Option<&Path>, stack: &mut Vec<PathBuf>) -> Result<Value> {
    let resolved = base_path(name, from)?;
    let who = match from {
        Some(file) => format!("the config {}", file.display()),
        None => {
            let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            format!(
                "the config parsed from a string, whose `{EXTENDS_KEY}` resolves relative to the \
                 current directory {} because no file declared it",
                dir.display()
            )
        }
    };
    ensure!(
        resolved.is_file(),
        "{who} extends {name:?}, which is not a file at {}",
        resolved.display()
    );
    load_tree(&resolved, stack).with_context(|| format!("{who} extends {}", resolved.display()))
}

/// The path a base names: as given when absolute, else under the directory of the file that
/// declares it, or under the current directory for a config parsed from a string.
fn base_path(name: &str, from: Option<&Path>) -> Result<PathBuf> {
    let named = Path::new(name);
    if named.is_absolute() {
        return Ok(named.to_path_buf());
    }
    match from {
        Some(file) => {
            let dir = match file.parent() {
                Some(dir) if !dir.as_os_str().is_empty() => dir,
                _ => Path::new("."),
            };
            Ok(dir.join(named))
        }
        None => Ok(std::env::current_dir()
            .context("cannot read the current directory to resolve an `extends` path")?
            .join(named)),
    }
}

/// The merge of two config trees: two mappings merge key by key, and any other value — a scalar,
/// a string, a list — the child replaces wholesale, so lists are never concatenated.
fn merge_value(base: &Value, child: &Value) -> Value {
    if let (Value::Map(base_entries), Value::Map(child_entries)) = (base, child) {
        let mut merged = base_entries.clone();
        for (key, value) in child_entries {
            let merged_value = match merged.get(key) {
                Some(previous) if previous.map().is_some() && value.map().is_some() => {
                    merge_value(previous, value)
                }
                _ => value.clone(),
            };
            merged.insert(key.clone(), merged_value);
        }
        return Value::Map(merged);
    }
    child.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strings `src/animalpose/config.py` accepts, with what `yaml.safe_load` makes of them.
    #[test]
    fn overrides_match_load_config() {
        let cases: &[(&str, Value)] = &[
            ("train.steps=200", Value::Int(200)),
            ("train.lr=5.0e-4", Value::Float(5.0e-4)),
            ("train.batch_size=4", Value::Int(4)),
            (
                "data.species=[mouse, rat]",
                Value::List(vec![Value::Str("mouse".into()), Value::Str("rat".into())]),
            ),
            (
                "model.classes=[a, b]",
                Value::List(vec![Value::Str("a".into()), Value::Str("b".into())]),
            ),
            ("device=cpu", Value::Str("cpu".into())),
            ("data.real_root=null", Value::Null),
            ("train.workers=true", Value::Bool(true)),
            ("train.workers=0.5", Value::Float(0.5)),
            ("output=runs/a.pt", Value::Str("runs/a.pt".into())),
            // `init=` partitions to an empty value, which `yaml.safe_load` reads as `None`.
            ("init=", Value::Null),
            // A second `=` is part of the value, as `partition` keeps everything after the first.
            ("pipeline.blob_area=10=20", Value::Str("10=20".into())),
        ];
        for (item, want) in cases {
            let got = Config::parse("output: runs/x.pt", &[])
                .unwrap()
                .with_override(item);
            assert_eq!(got.override_value(item), *want, "override {item}");
        }
    }

    /// Overrides create the path they need, exactly like `node.setdefault(p, {})`.
    #[test]
    fn overrides_create_missing_sections() {
        let cfg =
            Config::parse("device: auto", &["loss.sigma=2.0".into(), "a.b.c=1".into()]).unwrap();
        assert_eq!(cfg.f64_at("loss.sigma", 0.0), 2.0);
        assert_eq!(cfg.usize_at("a.b.c", 0), 1);
        assert_eq!(cfg.str_at("device", ""), "auto");
        assert_eq!(cfg.get("nope"), None);
    }

    /// `yaml.safe_load("") or {}`: an empty file is an empty config, not an error.
    #[test]
    fn empty_file_is_an_empty_config() {
        let cfg = Config::parse("", &[]).unwrap();
        assert!(cfg.value().map().is_some_and(BTreeMap::is_empty));
        let cfg = Config::parse("# only a comment\n", &["train.steps=1".into()]).unwrap();
        assert_eq!(cfg.usize_at("train.steps", 0), 1);
    }

    /// The keys of the shipped configs read here: `configs/behavior.yaml`.
    #[test]
    fn reads_the_shipped_behavior_config() {
        let Some(root) = crate::data::repo_root() else {
            return; // a crate copy outside the repository
        };
        let path = root.join("configs").join("behavior.yaml");
        if !path.exists() {
            return;
        }
        let cfg = Config::load(&path, &[]).unwrap();
        assert_eq!(cfg.str_at("output", ""), "runs/behavior.pt");
        assert_eq!(cfg.data().window, 96);
        assert_eq!(cfg.str_list_at("data.species"), ["mouse", "rat"]);
        // steps, batch size and learning rate are tuned per run, so only their types and signs are checked:
        // a scientific-notation scalar and a flow mapping must still parse as numbers.
        let train = cfg.train();
        assert!(train.steps > 0 && train.batch_size > 0, "{train:?}");
        assert!(train.lr > 0.0 && train.lr < 1.0, "{train:?}");
        assert!(train.weight_decay > 0.0 && train.weight_decay < 1.0, "{train:?}");
        let classes = cfg.str_list_at("model.classes");
        assert_eq!(classes.len(), 5);
        assert_eq!(classes[1], "walking");
        // An override of a flow-mapping value replaces the whole node, as Python does.
        let cfg = Config::load(&path, &["train.batch_size=8".into()]).unwrap();
        assert_eq!(cfg.train().batch_size, 8);
        assert_eq!(cfg.train().steps, train.steps);
    }

    /// A scalar in the middle of an override path is an error, not a silent no-op.
    #[test]
    fn override_through_a_scalar_fails() {
        let err = Config::parse("device: auto", &["device.deeper=1".into()]).unwrap_err();
        assert!(err.to_string().contains("not a mapping"), "{err}");
    }

    /// A scratch directory of config files, named after the test so the tests do not collide.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "animalpose-config-{tag}-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes `text` to `dir/name` (creating subdirectories) and returns the path.
    fn write_config(dir: &Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, text).unwrap();
        path
    }

    /// `extends:` merges mappings key by key and replaces every other value, as `{**base, **child}` does.
    #[test]
    fn a_child_merges_over_its_base() {
        let dir = temp_dir("merge");
        write_config(
            &dir,
            "base.yaml",
            "train: {steps: 100, lr: 0.01}\ndata: {window: 64, species: [mouse, rat]}\ntags: [a, b]\nmodel: tiny\n",
        );
        let child = write_config(
            &dir,
            "child.yaml",
            "extends: base.yaml\ntrain: {lr: 0.001}\ndata: {species: [rat]}\ntags: [c]\nmodel: {name: small}\n",
        );
        let cfg = Config::load(&child, &[]).unwrap();
        assert_eq!(cfg.f64_at("train.lr", 0.0), 0.001, "the child's scalar wins");
        assert_eq!(
            cfg.usize_at("train.steps", 0),
            100,
            "a sibling key of the base survives"
        );
        assert_eq!(cfg.usize_at("data.window", 0), 64);
        assert_eq!(
            cfg.str_list_at("data.species"),
            ["rat"],
            "a list replaces rather than appends"
        );
        assert_eq!(cfg.str_list_at("tags"), ["c"]);
        assert_eq!(
            cfg.model_json().unwrap()["name"], "small",
            "a mapping in the child replaces a scalar of the base"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Several bases merge left to right, and the child still lands on top of all of them.
    #[test]
    fn several_bases_merge_left_to_right() {
        let dir = temp_dir("multi");
        write_config(&dir, "a.yaml", "train: {steps: 1, lr: 1.0}\nfirst: true\n");
        write_config(&dir, "b.yaml", "train: {steps: 2, lr: 2.0}\nsecond: true\n");
        let child = write_config(&dir, "child.yaml", "extends: [a.yaml, b.yaml]\n");
        let cfg = Config::load(&child, &[]).unwrap();
        assert_eq!(cfg.usize_at("train.steps", 0), 2, "the later base wins");
        assert_eq!(cfg.f64_at("train.lr", 0.0), 2.0);
        assert!(cfg.bool_at("first", false) && cfg.bool_at("second", false));
        // A lone `extends:` string is the one-element list, and a base may sit in another directory.
        let child = write_config(
            &dir,
            "child.yaml",
            "extends: base.yaml\ntrain: {steps: 3}\n",
        );
        write_config(&dir, "base.yaml", "extends: [a.yaml, b.yaml]\ntrain: {lr: 3.0}\n");
        let cfg = Config::load(&child, &[]).unwrap();
        assert_eq!(cfg.usize_at("train.steps", 0), 3, "the child outranks its bases");
        assert_eq!(cfg.f64_at("train.lr", 0.0), 3.0);
        assert!(cfg.bool_at("first", false), "a base of a base merges too");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `extends` is a loading instruction, not a config key: no caller of `load` sees it.
    #[test]
    fn extends_is_absent_from_the_resolved_tree() {
        let dir = temp_dir("drop");
        write_config(&dir, "base.yaml", "train: {steps: 5}\nextends_note: kept\n");
        let child = write_config(&dir, "child.yaml", "extends: base.yaml\ndevice: auto\n");
        let grand = write_config(&dir, "grand.yaml", "extends: child.yaml\noutput: runs/g.pt\n");
        for path in &[&child, &grand] {
            let cfg = Config::load(path, &[]).unwrap();
            assert!(
                cfg.value().get(EXTENDS_KEY).is_none(),
                "{} still carries `extends`",
                path.display()
            );
            assert!(cfg.json().get(EXTENDS_KEY).is_none());
        }
        let cfg = Config::load(&grand, &[]).unwrap();
        assert_eq!(cfg.usize_at("train.steps", 0), 5, "the whole chain resolved");
        assert_eq!(cfg.str_at("output", ""), "runs/g.pt");
        assert_eq!(cfg.str_at("extends_note", ""), "kept", "only the key is dropped");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Inheritance resolves before the overrides, which may therefore target an inherited key.
    #[test]
    fn overrides_apply_on_top_of_inherited_keys() {
        let dir = temp_dir("override");
        write_config(
            &dir,
            "base.yaml",
            "train: {steps: 100, lr: 0.01}\ndata: {species: [mouse]}\n",
        );
        let child = write_config(&dir, "child.yaml", "extends: base.yaml\ntrain: {lr: 0.02}\n");
        let cfg = Config::load(
            &child,
            &[
                "train.steps=9".to_string(),
                "data.species=[rat]".to_string(),
                "eval.holdout=0.2".to_string(),
            ],
        )
        .unwrap();
        let train = cfg.train();
        assert_eq!(train.steps, 9, "an inherited key takes the override");
        assert_eq!(train.lr, 0.02, "the child still outranks the base");
        assert_eq!(train.batch_size, 32, "the default of a key no file names");
        assert_eq!(cfg.str_list_at("data.species"), ["rat"]);
        assert_eq!(cfg.f64_at("eval.holdout", 0.0), 0.2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A base that is not there is an error naming both the config and the base it looked for.
    #[test]
    fn a_missing_base_names_both_files() {
        let dir = temp_dir("missing");
        let child = write_config(&dir, "child.yaml", "extends: absent.yaml\n");
        let err = format!("{:#}", Config::load(&child, &[]).unwrap_err());
        assert!(err.contains("child.yaml"), "{err}");
        assert!(err.contains("absent.yaml"), "{err}");
        // Bases resolve against the declaring file, so one under a subdirectory needs that prefix.
        let nested = write_config(&dir, "sub/child.yaml", "extends: ../gone.yaml\n");
        let err = format!("{:#}", Config::load(&nested, &[]).unwrap_err());
        assert!(err.contains("gone.yaml"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file that transitively extends itself fails with the chain, rather than recursing forever.
    #[test]
    fn a_cycle_reports_the_chain() {
        let dir = temp_dir("cycle");
        let self_same = write_config(&dir, "self.yaml", "extends: self.yaml\n");
        let err = format!("{:#}", Config::load(&self_same, &[]).unwrap_err());
        assert!(err.contains("cycles"), "{err}");
        assert!(err.contains("self.yaml"), "{err}");
        let a = write_config(&dir, "a.yaml", "extends: b.yaml\n");
        write_config(&dir, "b.yaml", "extends: a.yaml\n");
        let err = format!("{:#}", Config::load(&a, &[]).unwrap_err());
        assert!(err.contains("cycles"), "{err}");
        // The error lists the cycle path, from the file that started loading back to itself.
        let chain = err.split("cycles: ").nth(1).unwrap_or_default();
        let links = chain.split(" -> ").collect::<Vec<_>>();
        assert_eq!(links.len(), 3, "{err}");
        assert!(links[0].ends_with("a.yaml"), "{err}");
        assert!(links[1].ends_with("b.yaml"), "{err}");
        assert!(links[2].ends_with("a.yaml"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A config parsed from a string has no directory: a relative base is an error that says so,
    /// while an absolute one resolves like any other file.
    #[test]
    fn a_string_config_resolves_an_absolute_base_only() {
        let dir = temp_dir("string");
        let base = write_config(&dir, "base.yaml", "train: {steps: 7}\n");
        let cfg = Config::parse(
            &format!("extends: {}\noutput: runs/x.pt\n", base.display()),
            &["train.steps=8".to_string()],
        )
        .unwrap();
        assert_eq!(cfg.usize_at("train.steps", 0), 8);
        assert!(cfg.value().get(EXTENDS_KEY).is_none());
        // The relative name below is unlikely to exist in whatever directory the tests run in.
        let err = format!(
            "{:#}",
            Config::parse("extends: animalpose-no-such-base.yaml\n", &[])
                .unwrap_err()
                .to_string()
        );
        assert!(err.contains("animalpose-no-such-base.yaml"), "{err}");
        assert!(err.contains("current directory"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An `extends:` that is neither a path nor a list of them is an error, not a key to ignore.
    #[test]
    fn extends_must_name_paths() {
        let dir = temp_dir("malformed");
        let child = write_config(&dir, "child.yaml", "extends: {path: base.yaml}\n");
        let err = format!("{:#}", Config::load(&child, &[]).unwrap_err());
        assert!(err.contains("extends"), "{err}");
        assert!(err.contains("mapping"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    impl Config {
        fn with_override(&self, item: &str) -> Config {
            let mut cfg = self.clone();
            cfg.apply_override(item).unwrap();
            cfg
        }

        fn override_value(&self, item: &str) -> Value {
            let key = item.split('=').next().unwrap();
            self.value.path(key).cloned().unwrap_or(Value::Null)
        }
    }
}
