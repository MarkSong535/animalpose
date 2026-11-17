//! Sample profiles shipped with the apps: experiment setups (config, networks, camera pair) that
//! are installed into the library on launch.
//!
//! The bundle is `samples/` in the repository (built by `scripts/build_samples.py`), copied into
//! `AnimalPose.app/Contents/Resources/samples` by `macos/build-app.sh` and next to the binaries by
//! `scripts/package_release.py`. `manifest.json` lists every file with its SHA-256.
//!
//! Installing never loses a person's work. Each installed file's hash is remembered in the
//! settings (`samples`), and so is the manifest's: a launch with the same bundle does nothing.
//!
//! | In the library | Installed |
//! | --- | --- |
//! | missing, never installed | copied |
//! | missing, installed before | left out (it was deleted on purpose) |
//! | the sample's current contents | nothing to do |
//! | what was installed last time | replaced by the newer sample |
//! | anything else (edited, or a person's own file of that name) | kept |

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use tempfile::NamedTempFile;

use super::library::{Library, Profile, Settings, MODEL_KINDS};
use crate::assay::output::sha256_cached;
use crate::models::{self, ModelKind};

#[derive(Debug, Deserialize)]
pub(crate) struct Manifest {
    pub(crate) version: u32,
    #[serde(default)]
    pub(crate) profiles: Vec<String>,
    pub(crate) files: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Entry {
    /// Relative to the bundle: `models/<kind>/…`, `configs/…`, `cameras/…` or `profiles/…`.
    pub(crate) path: String,
    pub(crate) sha256: String,
}

/// What an install did.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Installed {
    pub(crate) added: Vec<String>,
    pub(crate) updated: Vec<String>,
    /// Edited or deleted in the library since the last install, so left alone.
    pub(crate) kept: Vec<String>,
}

/// The bundle shipped with this binary: `ANIMALPOSE_SAMPLES`, the macOS app's resources, next to
/// the executable, a Linux package's `share/`, the checkout's `samples/`, or the model bundle
/// unzipped into `~/.animalpose`.
pub(crate) fn locate() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("ANIMALPOSE_SAMPLES") {
        candidates.push(dir.into());
    }
    if let Some(exe) = &exe_dir {
        candidates.push(exe.join("../Resources/samples"));
        candidates.push(exe.join("samples"));
        candidates.push(exe.join("../share/animalpose/samples"));
    }
    if let Some(root) = crate::data::repo_root() {
        candidates.push(root.join("samples"));
    }
    // The model bundle (`animalpose-models-*.zip`) unzipped into ~/.animalpose.
    if let Some(home) = std::env::home_dir() {
        candidates.push(home.join(".animalpose/samples"));
    }
    candidates
        .into_iter()
        .find(|dir| dir.join("manifest.json").is_file())
}

pub(crate) fn read_manifest(dir: &Path) -> Result<Manifest> {
    let path = dir.join("manifest.json");
    let bytes = fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not valid", path.display()))?;
    ensure!(
        manifest.version == 1,
        "sample bundle version {} is not supported",
        manifest.version
    );
    Ok(manifest)
}

/// Every file of the bundle is present, is not a git-lfs pointer, and has its listed SHA-256.
pub(crate) fn verify(dir: &Path) -> Result<Manifest> {
    let manifest = read_manifest(dir)?;
    for entry in &manifest.files {
        let path = dir.join(&entry.path);
        ensure!(path.is_file(), "{} is missing", path.display());
        let mut head = [0u8; 24];
        let n = std::io::Read::read(&mut fs::File::open(&path)?, &mut head)?;
        ensure!(
            !head[..n].starts_with(b"version https://git-lfs"),
            "{} is a git-lfs pointer; run `git lfs pull`",
            path.display()
        );
        let actual = sha256_cached(&path)?;
        ensure!(
            actual.eq_ignore_ascii_case(&entry.sha256),
            "{} has SHA-256 {actual}, the manifest lists {}",
            path.display(),
            entry.sha256
        );
    }
    Ok(manifest)
}

/// Where a bundle file goes in the library, and the profile it holds (for `profiles/…`).
fn target(library: &Library, dir: &Path, rel: &str) -> Result<(PathBuf, Option<Profile>)> {
    let parts: Vec<&str> = rel.split('/').collect();
    match parts.as_slice() {
        ["models", kind, name] => {
            let kind: ModelKind = kind.parse()?;
            ensure!(MODEL_KINDS.contains(&kind), "{rel}: unknown model kind");
            Ok((
                library.root.join("models").join(kind.as_str()).join(name),
                None,
            ))
        }
        ["configs", name] => Ok((library.root.join("configs").join(name), None)),
        ["cameras", name] => Ok((library.root.join("cameras").join(name), None)),
        ["profiles", _] => {
            let profile: Profile = serde_json::from_slice(&fs::read(dir.join(rel))?)
                .with_context(|| format!("{rel} is not a profile"))?;
            ensure!(
                !profile.name.trim().is_empty(),
                "{rel}: the profile has no name"
            );
            Ok((library.profile_file(&profile.name), Some(profile)))
        }
        _ => bail!("{rel}: not a sample file"),
    }
}

/// Copies the bundle in `dir` into `library` by the rules in the module docs, recording what was
/// installed in `settings.samples`. The bundle is verified first; nothing is copied if it fails.
pub(crate) fn install(library: &Library, dir: &Path, settings: &mut Settings) -> Result<Installed> {
    // The bundle installed last time, unchanged: nothing to copy (edits and deletions are kept
    // anyway), and nothing to hash beyond the manifest.
    let manifest_sha = crate::cli::download::sha256_file(&dir.join("manifest.json"))?;
    if settings.samples.get(MANIFEST) == Some(&manifest_sha) {
        return Ok(Installed::default());
    }
    let manifest = verify(dir)?;
    let mut report = Installed::default();
    for entry in &manifest.files {
        let (path, _) = target(library, dir, &entry.path)?;
        let key = entry.path.clone();
        let previous = settings.samples.get(&key).cloned();
        let current = if path.is_file() {
            Some(sha256_cached(&path)?)
        } else {
            None
        };
        let replace = match (&current, &previous) {
            (Some(now), _) if now.eq_ignore_ascii_case(&entry.sha256) => {
                settings.samples.insert(key, entry.sha256.clone());
                continue;
            }
            (None, None) => false,
            (None, Some(_)) => {
                report.kept.push(entry.path.clone());
                continue;
            }
            (Some(now), Some(before)) if now.eq_ignore_ascii_case(before) => true,
            (Some(_), _) => {
                report.kept.push(entry.path.clone());
                continue;
            }
        };
        let parent = path.parent().context("a library file has a folder")?;
        fs::create_dir_all(parent)?;
        let mut tmp = NamedTempFile::new_in(parent)?;
        std::io::copy(
            &mut fs::File::open(dir.join(&entry.path))?,
            tmp.as_file_mut(),
        )?;
        tmp.as_file_mut().sync_all()?;
        tmp.persist(&path)?;
        settings.samples.insert(key, entry.sha256.clone());
        if replace {
            report.updated.push(entry.path.clone());
        } else {
            report.added.push(entry.path.clone());
        }
    }
    settings.samples.insert(MANIFEST.to_owned(), manifest_sha);
    Ok(report)
}

/// The key under which [`Settings::samples`] remembers the manifest of the last bundle installed.
const MANIFEST: &str = "manifest.json";

/// Installs the bundle into a fresh library under `scratch` and checks each profile end to end:
/// its config loads, every network loads as its kind and runs on a few frames, the camera model
/// reads and names a top and a side camera, and the assay section parses. Returns one line per
/// profile.
pub(crate) fn check(dir: &Path, scratch: &Path) -> Result<Vec<String>> {
    let manifest = verify(dir)?;
    let library = Library::open(scratch.to_path_buf());
    let mut settings = Settings::default();
    install(&library, dir, &mut settings)?;
    for entry in manifest
        .files
        .iter()
        .filter(|e| e.path.starts_with("models/"))
    {
        let (path, _) = target(&library, dir, &entry.path)?;
        let (_, kind, _) =
            models::load(&path).with_context(|| format!("{} does not load", entry.path))?;
        let folder = entry.path.split('/').nth(1).unwrap_or_default();
        ensure!(
            kind.as_str() == folder,
            "{} is a {kind} network but sits in models/{folder}",
            entry.path
        );
    }
    let mut lines = Vec::new();
    let profiles = library.profiles();
    for name in &manifest.profiles {
        let profile = profiles
            .iter()
            .find(|p| &p.name == name)
            .with_context(|| format!("the profile {name} was not installed"))?;
        for kind in MODEL_KINDS {
            if let Some(model) = profile.models.get(kind.as_str()) {
                ensure!(
                    library.models(kind).contains(model),
                    "{name}: its {kind} network {model} is not in the bundle"
                );
            }
        }
        let mut settings = Settings::default();
        profile.apply(&mut settings);
        let files = library.pipeline(&settings)?;
        let config = crate::data::config::Config::load(&files.config, &files.overrides)
            .with_context(|| format!("{name}: the config does not load"))?;
        let pipeline = crate::cli::predict::Pipeline::from_config(&config)
            .with_context(|| format!("{name}: the networks do not load"))?;
        let frames = test_frames();
        let out = pipeline
            .run(&frames, None, 30.0)
            .with_context(|| format!("{name}: the networks do not run"))?;
        ensure!(
            out.n_frames == frames.len(),
            "{name}: {} of {} frames",
            out.n_frames,
            frames.len()
        );
        let assay = crate::assay::AssayConfig::from_config(&config)
            .with_context(|| format!("{name}: the config has no assay section"))?;
        let arena = match &assay.arena.model {
            Some(path) => {
                let net = crate::assay::arena_model::ArenaModel::load(path)
                    .with_context(|| format!("{name}: the arena detector does not load"))?;
                let grey: Vec<crate::algos::image::Image> =
                    frames.iter().map(crate::assay::arena::to_u8).collect();
                net.detect(&grey)
                    .with_context(|| format!("{name}: the arena detector does not run"))?;
                ensure!(
                    net.def.experiment.is_empty() || net.def.experiment == assay.experiment,
                    "{name}: the arena detector is for {}, the config runs {}",
                    net.def.experiment,
                    assay.experiment
                );
                format!(", arena {}", net.def.name)
            }
            None => String::new(),
        };
        let cameras = match &profile.camera_model {
            Some(model) => {
                let cameras = library
                    .camera_model(model)
                    .with_context(|| format!("{name}: the camera model does not read"))?;
                for view in [&assay.arena.top_view, &assay.arena.side_view] {
                    ensure!(
                        view.is_empty() || cameras.iter().any(|c| &c.name == view),
                        "{name}: the config names camera {view}, the camera model has {:?}",
                        cameras.iter().map(|c| c.name.as_str()).collect::<Vec<_>>()
                    );
                }
                cameras.len()
            }
            None => 0,
        };
        lines.push(format!(
            "{name}: {}{arena} ({} cameras), networks {} — loaded and ran on {} frames",
            assay.experiment,
            cameras,
            MODEL_KINDS
                .iter()
                .filter_map(|k| profile.models.get(k.as_str()).cloned())
                .collect::<Vec<_>>()
                .join(", "),
            out.n_frames
        ));
    }
    Ok(lines)
}

/// A few 640×480 frames with a dark, mouse-sized blob moving across a light floor.
fn test_frames() -> Vec<crate::algos::image::GrayF> {
    (0..4)
        .map(|t| {
            let (w, h) = (640u32, 480u32);
            let (cx, cy) = (280.0 + 6.0 * t as f32, 240.0);
            let pixels = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32, (i / w) as f32);
                    let d = ((x - cx) / 22.0).powi(2) + ((y - cy) / 11.0).powi(2);
                    if d < 1.0 {
                        30.0
                    } else {
                        220.0
                    }
                })
                .collect();
            crate::algos::image::GrayF {
                width: w,
                height: h,
                pixels,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bundle with one config, written by hand.
    fn bundle(dir: &Path, config: &str) {
        fs::create_dir_all(dir.join("configs")).unwrap();
        fs::write(dir.join("configs/a.yaml"), config).unwrap();
        let sha = crate::cli::download::sha256_file(&dir.join("configs/a.yaml")).unwrap();
        fs::write(
            dir.join("manifest.json"),
            format!(
                r#"{{"version": 1, "files": [{{"path": "configs/a.yaml", "sha256": "{sha}"}}]}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn installs_once_updates_untouched_files_and_keeps_edits_and_deletions() {
        let tmp = tempfile::tempdir().unwrap();
        let (dir, lib) = (
            tmp.path().join("bundle"),
            Library::open(tmp.path().join("lib")),
        );
        let mut settings = Settings::default();
        let file = lib.root.join("configs/a.yaml");

        bundle(&dir, "device: auto\n");
        let first = install(&lib, &dir, &mut settings).unwrap();
        assert_eq!(first.added, ["configs/a.yaml"]);
        assert_eq!(
            install(&lib, &dir, &mut settings).unwrap(),
            Installed::default()
        );

        // A newer sample replaces a file nobody edited.
        bundle(&dir, "device: cpu\n");
        let second = install(&lib, &dir, &mut settings).unwrap();
        assert_eq!(second.updated, ["configs/a.yaml"]);
        assert_eq!(fs::read_to_string(&file).unwrap(), "device: cpu\n");

        // An edited file is kept…
        fs::write(&file, "device: gpu\n").unwrap();
        bundle(&dir, "device: auto\n");
        assert_eq!(
            install(&lib, &dir, &mut settings).unwrap().kept,
            ["configs/a.yaml"]
        );
        assert_eq!(fs::read_to_string(&file).unwrap(), "device: gpu\n");

        // …and a deleted one stays deleted: the same bundle is not even looked at again, and a
        // newer one does not bring it back.
        fs::remove_file(&file).unwrap();
        assert_eq!(
            install(&lib, &dir, &mut settings).unwrap(),
            Installed::default()
        );
        bundle(&dir, "device: mps\n");
        assert_eq!(
            install(&lib, &dir, &mut settings).unwrap().kept,
            ["configs/a.yaml"]
        );
        assert!(!file.exists());
    }

    #[test]
    fn a_damaged_bundle_installs_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let (dir, lib) = (
            tmp.path().join("bundle"),
            Library::open(tmp.path().join("lib")),
        );
        bundle(&dir, "device: auto\n");
        fs::write(
            dir.join("configs/a.yaml"),
            "version https://git-lfs.github.com/spec/v1\n",
        )
        .unwrap();
        let err = install(&lib, &dir, &mut Settings::default()).unwrap_err();
        assert!(format!("{err:#}").contains("git-lfs"), "{err:#}");
        assert!(!lib.root.join("configs").exists());
    }

    /// The bundle in the repository: every profile installs, loads and runs.
    #[test]
    fn the_shipped_samples_install_load_and_run() {
        let dir = crate::data::repo_root().unwrap().join("samples");
        if !dir.join("manifest.json").is_file() {
            return;
        }
        let scratch = tempfile::tempdir().unwrap();
        let lines = check(&dir, scratch.path()).unwrap();
        let manifest = read_manifest(&dir).unwrap();
        assert_eq!(lines.len(), manifest.profiles.len());
        assert!(!lines.is_empty());
    }
}
