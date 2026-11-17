//! Files owned by the desktop applications: imported models and configs, and the settings.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use super::recording::{default_stages, Stage};
use crate::cli::predict::{checkpoint_path, default_pipeline_config};
use crate::data::config::{Config, Value};
use crate::models::{self, ModelKind};
use crate::multiview::camera::{load_cameras, Camera};

pub(crate) const MODEL_KINDS: [ModelKind; 5] = [
    ModelKind::Detector,
    ModelKind::Pose,
    ModelKind::Temporal,
    ModelKind::Behavior,
    ModelKind::Arena,
];
pub(crate) const MODEL_EXTENSIONS: [&str; 2] = ["pt", "safetensors"];
pub(crate) const CONFIG_EXTENSIONS: [&str; 2] = ["yaml", "yml"];

#[derive(Default)]
pub(crate) struct FolderImport {
    pub(crate) configs: usize,
    pub(crate) models: usize,
    pub(crate) issues: Vec<String>,
}

pub(crate) struct ConfigImport {
    pub(crate) name: String,
    pub(crate) issues: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Settings {
    /// The profile in use. `config`, `models`, `classify_behavior` and `stages` mirror it.
    pub(crate) profile: Option<String>,
    /// Result folders of recent jobs, newest first.
    pub(crate) recent_results: Vec<PathBuf>,
    pub(crate) recordings: Option<PathBuf>,
    /// Where processed videos are saved; `None` puts the results next to the first video.
    pub(crate) output: Option<PathBuf>,
    pub(crate) config: Option<String>,
    pub(crate) models: BTreeMap<String, String>,
    pub(crate) stages: Vec<Stage>,
    pub(crate) live: bool,
    pub(crate) classify_behavior: bool,
    /// The arena's camera model (a calibration in `cameras/`) for multiview analysis.
    pub(crate) camera_model: Option<String>,
    /// The view each camera records as, by [`super::capture::Source::key`].
    pub(crate) views: BTreeMap<String, String>,
    /// Process the videos in each folder together, as views of one arena.
    pub(crate) multiview: bool,
    /// Sample files installed into the library ([`super::samples`]): bundle path → SHA-256.
    pub(crate) samples: BTreeMap<String, String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            profile: None,
            recent_results: Vec::new(),
            recordings: None,
            output: None,
            config: None,
            models: BTreeMap::new(),
            stages: default_stages(),
            live: true,
            classify_behavior: true,
            camera_model: None,
            views: BTreeMap::new(),
            multiview: false,
            samples: BTreeMap::new(),
        }
    }
}

/// A named experiment setup, such as "Open Field": the pipeline config, the models, whether to
/// classify behavior, the recording stages, and the arena's cameras (their model and the view
/// each one records as). Stored as `profiles/<name>.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Profile {
    pub(crate) name: String,
    pub(crate) config: Option<String>,
    pub(crate) models: BTreeMap<String, String>,
    pub(crate) classify_behavior: bool,
    pub(crate) stages: Vec<Stage>,
    pub(crate) camera_model: Option<String>,
    pub(crate) views: BTreeMap<String, String>,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            config: None,
            models: BTreeMap::new(),
            classify_behavior: true,
            stages: default_stages(),
            camera_model: None,
            views: BTreeMap::new(),
        }
    }
}

impl Profile {
    /// The profile's part of `settings`.
    pub(crate) fn from_settings(name: &str, settings: &Settings) -> Self {
        Self {
            name: name.to_owned(),
            config: settings.config.clone(),
            models: settings.models.clone(),
            classify_behavior: settings.classify_behavior,
            stages: settings.stages.clone(),
            camera_model: settings.camera_model.clone(),
            views: settings.views.clone(),
        }
    }

    /// Copies the profile into `settings` and makes it the one in use.
    pub(crate) fn apply(&self, settings: &mut Settings) {
        settings.profile = Some(self.name.clone());
        settings.config = self.config.clone();
        settings.models = self.models.clone();
        settings.classify_behavior = self.classify_behavior;
        settings.stages = self.stages.clone();
        settings.camera_model = self.camera_model.clone();
        settings.views = self.views.clone();
    }
}

#[derive(Clone)]
pub(crate) struct Library {
    pub(crate) root: PathBuf,
}

impl Library {
    pub(crate) fn open(root: PathBuf) -> Self {
        Self { root }
    }

    pub(crate) fn load(&self) -> Result<Settings> {
        match fs::read(self.root.join("settings.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("cannot read the settings file"),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(Settings::default()),
            Err(e) => Err(e).context("cannot open the settings file"),
        }
    }

    pub(crate) fn save(&self, settings: &Settings) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        let mut file = NamedTempFile::new_in(&self.root)?;
        serde_json::to_writer_pretty(file.as_file_mut(), settings)?;
        file.as_file_mut().sync_all()?;
        file.persist(self.root.join("settings.json"))?;
        Ok(())
    }

    /// The file profile `name` is stored in.
    pub(crate) fn profile_file(&self, name: &str) -> PathBuf {
        self.profile_path(name)
    }

    fn profile_path(&self, name: &str) -> PathBuf {
        self.root
            .join("profiles")
            .join(format!("{}.json", super::recording::safe_name(name)))
    }

    /// Every profile, by name.
    pub(crate) fn profiles(&self) -> Vec<Profile> {
        let mut profiles: Vec<Profile> = list(&self.root.join("profiles"), &["json"])
            .into_iter()
            .filter_map(|file| {
                let bytes = fs::read(self.root.join("profiles").join(file)).ok()?;
                serde_json::from_slice::<Profile>(&bytes).ok()
            })
            .filter(|profile| !profile.name.trim().is_empty())
            .collect();
        profiles.sort_by_key(|p| p.name.to_lowercase());
        profiles
    }

    /// Saves `profile`, replacing the one called `previous` (a rename) when given. A name in use
    /// by another profile is refused.
    pub(crate) fn save_profile(&self, profile: &Profile, previous: Option<&str>) -> Result<()> {
        let name = profile.name.trim();
        ensure!(!name.is_empty(), "A profile needs a name.");
        let path = self.profile_path(name);
        let renamed = previous.is_some_and(|old| self.profile_path(old) != path);
        ensure!(
            !(renamed || previous.is_none()) || !path.exists(),
            "A profile named {name} already exists."
        );
        let dir = self.root.join("profiles");
        fs::create_dir_all(&dir)?;
        let mut profile = profile.clone();
        profile.name = name.to_owned();
        let mut file = NamedTempFile::new_in(&dir)?;
        serde_json::to_writer_pretty(file.as_file_mut(), &profile)?;
        file.persist(&path)?;
        if let Some(old) = previous.filter(|_| renamed) {
            let _ = fs::remove_file(self.profile_path(old));
        }
        Ok(())
    }

    pub(crate) fn delete_profile(&self, name: &str) -> Result<()> {
        fs::remove_file(self.profile_path(name))
            .with_context(|| format!("cannot delete the profile {name}"))
    }

    /// The config file `name` stands for; `None` is the default pipeline.
    pub(crate) fn config_path(&self, name: Option<&str>) -> Result<PathBuf> {
        let default = default_pipeline_config();
        Ok(match name {
            Some(name) => self.root.join("configs").join(filename(Path::new(name))?),
            None if default.is_file() => default,
            None => {
                fs::create_dir_all(&self.root)?;
                let path = self.root.join("default-pipeline.yaml");
                if !path.exists() {
                    fs::write(&path, "device: auto\n")?;
                }
                path
            }
        })
    }

    /// Arena camera models: calibrations (`cams.json`) of a multi-camera rig.
    pub(crate) fn camera_models(&self) -> Vec<String> {
        list(&self.root.join("cameras"), &["json"])
    }

    pub(crate) fn camera_model_path(&self, name: &str) -> Result<PathBuf> {
        Ok(self.root.join("cameras").join(filename(Path::new(name))?))
    }

    /// The cameras of model `name`.
    pub(crate) fn camera_model(&self, name: &str) -> Result<Vec<Camera>> {
        load_cameras(self.camera_model_path(name)?)
    }

    /// Copies a calibration into the library after checking that it reads.
    pub(crate) fn import_camera_model(&self, source: &Path) -> Result<String> {
        let name = filename(source)?;
        ensure!(
            has_extension(source, &["json"]),
            "Select a cams.json camera model."
        );
        let cameras = load_cameras(source)?;
        ensure!(
            cameras.len() > 1,
            "A camera model for multiview needs at least two cameras."
        );
        let dir = self.root.join("cameras");
        fs::create_dir_all(&dir)?;
        for existing in self.camera_models() {
            if same_contents(source, &dir.join(&existing))? {
                return Ok(existing);
            }
        }
        let mut tmp = NamedTempFile::new_in(&dir)?;
        std::io::copy(&mut File::open(source)?, tmp.as_file_mut())?;
        install(tmp, &dir, &name)
    }

    pub(crate) fn configs(&self) -> Vec<String> {
        list(&self.root.join("configs"), &CONFIG_EXTENSIONS)
    }

    pub(crate) fn models(&self, kind: ModelKind) -> Vec<String> {
        list(&self.model_dir(kind), &MODEL_EXTENSIONS)
    }

    fn model_dir(&self, kind: ModelKind) -> PathBuf {
        self.root.join("models").join(kind.as_str())
    }

    pub(crate) fn import_model(&self, source: &Path) -> Result<(ModelKind, String)> {
        let name = filename(source)?;
        ensure!(
            has_extension(source, &MODEL_EXTENSIONS),
            "Select a .pt or .safetensors checkpoint."
        );
        ensure!(source.is_file(), "{} is not a file", source.display());
        fs::create_dir_all(&self.root)?;
        let mut tmp = NamedTempFile::new_in(&self.root)?;
        std::io::copy(&mut File::open(source)?, tmp.as_file_mut())?;
        let (_, kind, _) = models::load(tmp.path())
            .with_context(|| format!("cannot import checkpoint {}", source.display()))?;
        let dir = self.model_dir(kind);
        fs::create_dir_all(&dir)?;
        for name in self.models(kind) {
            if same_contents(tmp.path(), &dir.join(&name))? {
                return Ok((kind, name));
            }
        }
        Ok((kind, install(tmp, &dir, &name)?))
    }

    pub(crate) fn import_config(&self, source: &Path) -> Result<ConfigImport> {
        Importer::new(self, None).config(source)
    }

    pub(crate) fn import_folder(&self, source: &Path) -> Result<FolderImport> {
        let source = source.canonicalize()?;
        ensure!(source.is_dir(), "Choose a folder to import.");
        let mut report = FolderImport::default();
        let (configs, models) = folder_files(&source, &mut report.issues);
        let mut importer = Importer::new(self, Some(&source));
        for model in models {
            if let Err(err) = importer.model(&model) {
                report.issues.push(format!("{}: {err:#}", model.display()));
            }
        }
        let mut saved_configs = BTreeSet::new();
        for config in configs {
            match importer.config(&config) {
                Ok(imported) => {
                    saved_configs.insert(imported.name);
                    report.issues.extend(imported.issues);
                }
                Err(err) => report.issues.push(format!("{}: {err:#}", config.display())),
            }
        }
        report.configs = saved_configs.len();
        report.models = importer
            .models
            .values()
            .map(|(kind, name)| (kind.as_str(), name))
            .collect::<BTreeSet<_>>()
            .len();
        Ok(report)
    }

    pub(crate) fn pipeline(&self, settings: &Settings) -> Result<PipelineFiles> {
        let config = self.config_path(settings.config.as_deref())?;
        let base = Config::load(&config, &[])?;
        let mut overrides = Vec::new();
        for kind in MODEL_KINDS {
            let key = format!("checkpoints.{}", kind.as_str());
            // Optional networks: a missing behavior classifier or arena detector is left out.
            let optional = matches!(kind, ModelKind::Behavior | ModelKind::Arena);
            let behavior = kind == ModelKind::Behavior;
            if behavior && !settings.classify_behavior {
                overrides.push(format!("{key}=null"));
            } else if let Some(name) = settings.models.get(kind.as_str()) {
                let model = self.model_dir(kind).join(filename(Path::new(name))?);
                match model.canonicalize() {
                    Ok(path) => overrides.push(format!("{key}={}", serde_json::to_string(&path)?)),
                    Err(err) if optional && err.kind() == ErrorKind::NotFound => {
                        overrides.push(format!("{key}=null"));
                    }
                    Err(err) => {
                        return Err(err).with_context(|| format!("cannot find {}", model.display()))
                    }
                }
            } else if let Some(reference) = base.get(&key).and_then(Value::as_str) {
                // Configs imported by older builds may still point to the original checkpoints.
                if reference.trim().is_empty() || !checkpoint_path(&base, reference).try_exists()? {
                    overrides.push(format!("{key}=null"));
                }
            }
        }
        Ok(PipelineFiles { config, overrides })
    }
}

/// Each source checkpoint is validated and copied at most once during an import.
struct Importer<'a> {
    library: &'a Library,
    folder: Option<&'a Path>,
    models: BTreeMap<PathBuf, (ModelKind, String)>,
}

impl<'a> Importer<'a> {
    fn new(library: &'a Library, folder: Option<&'a Path>) -> Self {
        Self {
            library,
            folder,
            models: BTreeMap::new(),
        }
    }

    fn model(&mut self, source: &Path) -> Result<(ModelKind, String)> {
        let key = source.canonicalize()?;
        if let Some(imported) = self.models.get(&key) {
            return Ok(imported.clone());
        }
        let imported = self.library.import_model(source)?;
        self.models.insert(key, imported.clone());
        Ok(imported)
    }

    fn config(&mut self, source: &Path) -> Result<ConfigImport> {
        let name = filename(source)?;
        ensure!(
            has_extension(source, &CONFIG_EXTENSIONS),
            "Select a .yaml or .yml config."
        );
        let config = Config::load(source, &[])?;
        // Store inherited settings. A folder import can also link copies of the checkpoints.
        let mut value = config.value().clone();
        let mut issues = Vec::new();
        if let Value::Map(root) = &mut value {
            if let (Some(_), Some(Value::Map(checkpoints))) =
                (self.folder, root.remove("checkpoints"))
            {
                let mut owned = BTreeMap::new();
                for kind in MODEL_KINDS {
                    let key = kind.as_str();
                    let Some(reference) = checkpoints.get(key).and_then(Value::as_str) else {
                        continue;
                    };
                    if reference.trim().is_empty() {
                        continue;
                    }
                    match self.owned_checkpoint(source, &config, reference, kind) {
                        Ok(path) => {
                            owned.insert(key.to_owned(), Value::Str(path));
                        }
                        Err(err) if err.downcast_ref::<std::io::Error>()
                            .is_some_and(|err| err.kind() == ErrorKind::NotFound) => {}
                        Err(err) => issues.push(format!(
                            "{name}: {kind} checkpoint {reference} could not be copied ({err:#}). Select a {kind} model in the app."
                        )),
                    }
                }
                if !owned.is_empty() {
                    root.insert("checkpoints".into(), Value::Map(owned));
                }
            }
        }
        let dir = self.library.root.join("configs");
        fs::create_dir_all(&dir)?;
        let mut temp = NamedTempFile::new_in(&dir)?;
        temp.write_all(serde_yaml::to_string(&value)?.as_bytes())?;
        Ok(ConfigImport {
            name: install(temp, &dir, &name)?,
            issues,
        })
    }

    fn owned_checkpoint(
        &mut self,
        source: &Path,
        config: &Config,
        reference: &str,
        expected: ModelKind,
    ) -> Result<String> {
        let path = source
            .parent()
            .into_iter()
            .chain(self.folder)
            .map(|dir| dir.join(reference))
            .find(|path| path.is_file())
            .unwrap_or_else(|| checkpoint_path(config, reference));
        let (kind, name) = self.model(&path)?;
        ensure!(kind == expected, "the file contains a {kind} model");
        self.library
            .model_dir(kind)
            .join(name)
            .canonicalize()?
            .into_os_string()
            .into_string()
            .map_err(|_| anyhow::anyhow!("the library path is not valid Unicode"))
    }
}

fn folder_files(root: &Path, issues: &mut Vec<String>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut configs = Vec::new();
    let mut models = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(err) => {
                issues.push(format!("{}: {err}", dir.display()));
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(err) => {
                    issues.push(format!("{}: {err}", dir.display()));
                    continue;
                }
            };
            let path = entry.path();
            // Do not follow directory symlinks, which can escape the selection or form cycles.
            match entry.file_type() {
                Ok(ty) if ty.is_dir() => pending.push(path),
                Ok(ty) if ty.is_file() || (ty.is_symlink() && path.is_file()) => {
                    if has_extension(&path, &CONFIG_EXTENSIONS) {
                        configs.push(path);
                    } else if has_extension(&path, &MODEL_EXTENSIONS) {
                        models.push(path);
                    }
                }
                Ok(_) => {}
                Err(err) => issues.push(format!("{}: {err}", path.display())),
            }
        }
    }
    configs.sort();
    models.sort();
    (configs, models)
}

#[derive(Clone)]
pub(crate) struct PipelineFiles {
    pub(crate) config: PathBuf,
    pub(crate) overrides: Vec<String>,
}

fn filename(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(OsStr::to_str)
        .map(str::to_owned)
        .context("a valid filename is required")
}

fn has_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|ext| extensions.iter().any(|e| ext.eq_ignore_ascii_case(e)))
}

fn list(dir: &Path, extensions: &[&str]) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            (path.is_file() && has_extension(&path, extensions))
                .then(|| filename(&path).ok())
                .flatten()
        })
        .collect();
    names.sort();
    names
}

fn install(mut file: NamedTempFile, dir: &Path, name: &str) -> Result<String> {
    file.as_file().sync_all()?;
    for index in 0.. {
        let name = if index == 0 {
            name.to_string()
        } else {
            format!("{index}-{name}")
        };
        let target = dir.join(&name);
        match file.persist_noclobber(&target) {
            Ok(_) => return Ok(name),
            Err(err) if err.error.kind() == ErrorKind::AlreadyExists => {
                file = err.file;
                if same_contents(file.path(), &target)? {
                    return Ok(name);
                }
            }
            Err(err) => return Err(err.into()),
        }
    }
    unreachable!()
}

fn same_contents(left: &Path, right: &Path) -> Result<bool> {
    // Exact comparison with bounded memory, even for large model files.
    match fs::symlink_metadata(right) {
        Ok(metadata) if !metadata.is_file() => return Ok(false),
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(false),
        result => {
            result?;
        }
    }
    let mut left = File::open(left)?;
    let mut right = File::open(right)?;
    let mut remaining = left.metadata()?.len();
    if remaining != right.metadata()?.len() {
        return Ok(false);
    }
    let mut a = [0; 16 * 1024];
    let mut b = [0; 16 * 1024];
    while remaining > 0 {
        let len = remaining.min(a.len() as u64) as usize;
        left.read_exact(&mut a[..len])?;
        right.read_exact(&mut b[..len])?;
        if a[..len] != b[..len] {
            return Ok(false);
        }
        remaining -= len as u64;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_yaml_drops_checkpoint_paths_even_when_the_files_exist() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path().join("Library"));
        let originals = temp.path().join("originals");
        fs::create_dir(&originals).unwrap();
        let weights = originals.join("pose.safetensors");
        fs::copy(
            crate::data::fixtures_dir()
                .unwrap()
                .join("models/pose_tiny.safetensors"),
            &weights,
        )
        .unwrap();
        let source = originals.join("setup.yaml");
        fs::write(
            &source,
            "device: cpu\ncheckpoints: {pose: pose.safetensors, behavior: runs/behavior_real.pt}\n",
        )
        .unwrap();
        let import = library.import_config(&source).unwrap();
        assert!(import.issues.is_empty());
        fs::remove_dir_all(originals).unwrap();

        let path = library.root.join("configs").join(import.name);
        let cfg = Config::load(path, &[]).unwrap();
        assert!(cfg.get("checkpoints").is_none());
        assert_eq!(cfg.require_str("device").unwrap(), "cpu");
        assert!(library.models(ModelKind::Pose).is_empty());
    }

    #[test]
    fn folder_import_reuses_models_and_preserves_distinct_files_with_the_same_name() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path().join("Library"));
        let originals = temp.path().join("bundle");
        let setups = originals.join("project/setups");
        let weights = originals.join("weights");
        let alternatives = originals.join("alternatives");
        for dir in [&setups, &weights, &alternatives] {
            fs::create_dir_all(dir).unwrap();
        }
        let fixtures = crate::data::fixtures_dir().unwrap().join("models");
        fs::copy(
            fixtures.join("detector_tiny.safetensors"),
            weights.join("detector.safetensors"),
        )
        .unwrap();
        let original_pose = weights.join("pose.safetensors");
        fs::copy(fixtures.join("pose_tiny.safetensors"), &original_pose).unwrap();
        fs::copy(&original_pose, weights.join("duplicate.safetensors")).unwrap();
        fs::copy(
            fixtures.join("pose_tiny_context.safetensors"),
            alternatives.join("pose.safetensors"),
        )
        .unwrap();
        fs::write(
            setups.join("base.yaml"),
            "device: cpu\ncheckpoints:\n  detector: weights/detector.safetensors\n  pose: weights/pose.safetensors\n  behavior: runs/behavior_real.pt\n",
        )
        .unwrap();
        fs::write(
            setups.join("experiment.yml"),
            "extends: base.yaml\npipeline: {crop_size: 64}\n",
        )
        .unwrap();
        fs::write(setups.join("invalid.yaml"), "extends: missing.yaml\n").unwrap();
        fs::write(weights.join("invalid.pt"), b"not a checkpoint").unwrap();
        fs::write(originals.join("notes.txt"), "ignored").unwrap();

        for _ in 0..2 {
            let imported = library.import_folder(&originals).unwrap();
            assert_eq!(imported.configs, 2);
            assert_eq!(imported.models, 3);
            assert_eq!(imported.issues.len(), 2, "{:?}", imported.issues);
        }
        assert_eq!(library.configs(), ["base.yaml", "experiment.yml"]);
        assert_eq!(library.models(ModelKind::Pose).len(), 2);
        fs::remove_dir_all(originals).unwrap();

        let config_dir = library.root.join("configs");
        let base = Config::load(config_dir.join("base.yaml"), &[]).unwrap();
        let experiment = Config::load(config_dir.join("experiment.yml"), &[]).unwrap();
        assert!(experiment.get("checkpoints.behavior").is_none());
        let pose = experiment.require_str("checkpoints.pose").unwrap();
        assert_eq!(pose, base.require_str("checkpoints.pose").unwrap());
        assert_eq!(experiment.usize_at("pipeline.crop_size", 0), 64);
        assert_eq!(
            fs::read(pose).unwrap(),
            fs::read(fixtures.join("pose_tiny.safetensors")).unwrap()
        );
        crate::cli::predict::Pipeline::from_config(&experiment).unwrap();
    }

    #[test]
    fn imported_yaml_with_a_stale_path_uses_the_selected_model() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path().join("Library"));
        let source = temp.path().join("setup.yml");
        fs::write(&source, "checkpoints: {pose: absent/pose.safetensors}\n").unwrap();
        let import = library.import_config(&source).unwrap();
        assert!(import.issues.is_empty());
        let fixtures = crate::data::fixtures_dir().unwrap();
        let (kind, name) = library
            .import_model(&fixtures.join("models/pose_tiny.safetensors"))
            .unwrap();
        let mut settings = Settings {
            config: Some(import.name),
            ..Settings::default()
        };
        settings.models.insert(kind.as_str().into(), name);
        let files = library.pipeline(&settings).unwrap();
        let cfg = Config::load(&files.config, &files.overrides).unwrap();
        let pose = cfg.require_str("checkpoints.pose").unwrap();
        assert!(Path::new(&pose).is_file());
    }

    #[test]
    fn behavior_selection_is_optional_and_can_be_disabled() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path().join("Library"));
        let source = temp.path().join("setup.yaml");
        fs::write(&source, "device: cpu\n").unwrap();
        let fixtures = crate::data::fixtures_dir().unwrap();
        let (kind, name) = library
            .import_model(&fixtures.join("models/behavior_tiny.safetensors"))
            .unwrap();
        let mut settings = Settings {
            config: Some(library.import_config(&source).unwrap().name),
            models: BTreeMap::from([(kind.as_str().into(), name.clone())]),
            ..Settings::default()
        };
        let read = |settings: &Settings| {
            let files = library.pipeline(settings).unwrap();
            Config::load(&files.config, &files.overrides).unwrap()
        };
        let path = read(&settings).require_str("checkpoints.behavior").unwrap();
        assert!(Path::new(&path).is_file());

        settings.classify_behavior = false;
        library.save(&settings).unwrap();
        let restored = library.load().unwrap();
        assert!(!restored.classify_behavior);
        assert!(read(&restored).get("checkpoints.behavior").is_none());

        fs::remove_file(library.model_dir(kind).join(name)).unwrap();
        settings.classify_behavior = true;
        assert!(read(&settings).get("checkpoints.behavior").is_none());
    }

    #[test]
    fn imports_configs_and_models_and_restores_a_protocol() {
        let temp = tempfile::tempdir().unwrap();
        let library = Library::open(temp.path().join("Library"));
        let mut settings = library.load().unwrap();
        settings.stages[0].seconds = 17;
        settings.stages[0].wait_after = true;
        fs::write(temp.path().join("base.yaml"), "device: cpu\n").unwrap();
        let source = temp.path().join("setup.yaml");
        fs::write(&source, "extends: base.yaml\npipeline: {crop_size: 64}\n").unwrap();
        settings.config = Some(library.import_config(&source).unwrap().name);
        assert_eq!(
            library.import_config(&source).unwrap().name,
            settings.config.clone().unwrap()
        );
        let fixtures = crate::data::fixtures_dir().unwrap();
        let (kind, name) = library
            .import_model(&fixtures.join("models/pose_tiny.safetensors"))
            .unwrap();
        assert_eq!(kind, ModelKind::Pose);
        settings.models.insert(kind.as_str().into(), name.clone());
        library.save(&settings).unwrap();
        let restored = library.load().unwrap();
        assert_eq!(restored.stages, settings.stages);
        assert_eq!(library.models(kind), [name]);
        let files = library.pipeline(&restored).unwrap();
        let cfg = Config::load(&files.config, &files.overrides).unwrap();
        assert_eq!(cfg.require_str("device").unwrap(), "cpu");
        assert_eq!(cfg.usize_at("pipeline.crop_size", 0), 64);
        assert!(Path::new(&cfg.require_str("checkpoints.pose").unwrap()).is_file());
    }
}
