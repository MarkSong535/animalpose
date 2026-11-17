//! The desktop application's state and actions, independent of any window toolkit. The egui window
//! ([`crate::gui`]) and the native macOS app (through [`crate::ffi`]) both drive one [`Engine`].
//!
//! Each recording sends captured frames in order to an MP4 encoder. Preview uses a separate
//! latest-frame slot, and pose inference pauses during recording. Cameras are not synchronized.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};

pub use super::capture::{list_cameras, Source};
use super::capture::{CaptureControl, Worker};
use super::library::{ConfigImport, FolderImport, Library, PipelineFiles, Profile, Settings};
use super::log::Log;
use super::models::ModelCache;
use super::processing::{self, Control as ProcessingControl, Job as ProcessingJob, Outcome};
use super::recording::{self, Phase, Session};
use super::views;
use crate::algos::image::Bgr;
use crate::cli::predict::{Pipeline, Progress};
use crate::multiview::camera::{load_cameras, Camera};
use crate::multiview::drift::References;

/// A camera model for Multiview: its cameras, and where their reference pictures live (the
/// session drift of each camera is measured against them, [`crate::multiview::drift`]).
#[derive(Clone, Debug)]
pub(crate) struct Rig {
    pub cameras: Vec<Camera>,
    pub references: Option<References>,
}

impl Rig {
    /// The camera model saved at `path`, with its references beside it.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            cameras: load_cameras(path)?,
            references: Some(References::beside(path)),
        })
    }
}

/// Runs the offline pipeline on one recording. Multiple views are processed separately unless
/// a camera calibration file is supplied for 3D fusion.
pub fn analyze(config: &str, views: &[PathBuf], cameras: Option<&str>) -> Result<PathBuf> {
    let rig = cameras.map(|c| Rig::load(Path::new(c))).transpose()?;
    let dir = views
        .first()
        .and_then(|v| v.parent())
        .context("nothing was recorded")?
        .join("analysis");
    process_views(
        &PipelineFiles {
            config: PathBuf::from(config),
            overrides: Vec::new(),
        },
        views,
        rig.as_ref(),
        &dir,
        None,
        &ModelCache::default(),
        None,
        &dir.with_file_name("zones"),
    )?;
    Ok(dir)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn process_views(
    files: &PipelineFiles,
    views: &[PathBuf],
    rig: Option<&Rig>,
    dir: &Path,
    control: Option<&ProcessingControl>,
    models: &ModelCache,
    log: Option<&Log>,
    zones: &Path,
) -> Result<()> {
    anyhow::ensure!(
        crate::data::video::ffmpeg_available(),
        "Install FFmpeg to produce an overlay video."
    );
    let report = |progress| match control {
        Some(control) => control.report(progress),
        None => Ok(()),
    };
    report(Progress::Preparing)?;
    let pipeline = models.get(files)?;
    if let Some(rig) = rig.filter(|_| views.len() > 1) {
        let names: Vec<String> = rig.cameras.iter().map(|c| c.name.clone()).collect();
        let ordered = views::match_cameras(views, &names)?;
        if let Some(control) = control {
            control.begin_clip(display_name(dir));
        }
        fuse(&pipeline, &ordered, rig, dir, report, log)?;
        score(files, dir, None, zones, log);
        if let Some(control) = control {
            control.clip_done()?;
        }
        return Ok(());
    }
    for v in views {
        let out = if views.len() == 1 {
            dir.to_path_buf()
        } else {
            dir.join(v.file_stem().context("video needs a filename")?)
        };
        if let Some(control) = control {
            control.begin_clip(display_name(v));
        }
        predict_one(&pipeline, v, &out, report, log)?;
        score(files, &out, Some(v), zones, log);
        if let Some(control) = control {
            control.clip_done()?;
        }
    }
    Ok(())
}

/// One video through networks that are already loaded. Each call starts a new sequence: a fresh
/// tracker, and temporal windows that begin at this video's first frame.
fn predict_one(
    pipeline: &crate::cli::predict::Pipeline,
    video: &Path,
    out: &Path,
    report: impl FnMut(Progress) -> Result<()>,
    log: Option<&Log>,
) -> Result<()> {
    anyhow::ensure!(video.is_file(), "{} does not exist", video.display());
    let started = std::time::Instant::now();
    crate::cli::predict::run_loaded_with_progress(
        pipeline,
        &video.to_string_lossy(),
        &out.to_string_lossy(),
        None,
        None,
        true,
        report,
        |line| match log {
            Some(log) => log.line(line),
            None => println!("{line}"),
        },
    )?;
    for name in ["tracks.csv", "tracks.npz", "overlay.mp4"] {
        anyhow::ensure!(
            out.join(name).is_file(),
            "{} was not produced",
            out.join(name).display()
        );
    }
    if let Some(log) = log {
        log.line(&format!(
            "{}: done in {:.1} s",
            video.display(),
            started.elapsed().as_secs_f64()
        ));
    }
    Ok(())
}

/// The views of one arena, in the order of the rig's cameras, fused into 3D with networks that are
/// already loaded. Each camera is first corrected for its drift since its reference picture.
/// `views.json` records which video each camera's tracks came from, `cameras.json` the cameras used.
fn fuse(
    pipeline: &Pipeline,
    views: &[PathBuf],
    rig: &Rig,
    out: &Path,
    report: impl FnMut(Progress) -> Result<()>,
    log: Option<&Log>,
) -> Result<()> {
    for view in views {
        anyhow::ensure!(view.is_file(), "{} does not exist", view.display());
    }
    let started = std::time::Instant::now();
    let fused = crate::multiview::pipeline::run_loaded(
        pipeline,
        rig.cameras.clone(),
        views,
        out,
        None,
        rig.references.as_ref(),
        report,
    )?;
    let say = |line: &str| match log {
        Some(log) => log.line(line),
        None => println!("{line}"),
    };
    for entry in &fused.drift {
        say(&entry.line());
    }
    let animals = fused.animals;
    let manifest: Vec<_> = views
        .iter()
        .zip(&fused.cameras)
        .enumerate()
        .map(|(i, (video, camera))| {
            serde_json::json!({
                "view": camera.name,
                "video": video,
                "tracks": format!("view{i}_tracks.csv"),
            })
        })
        .collect();
    std::fs::write(
        out.join("views.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let line = format!(
        "{}: {animals} animals fused across {} views in {:.1} s",
        display_name(out),
        views.len(),
        started.elapsed().as_secs_f64()
    );
    match log {
        Some(log) => log.line(&line),
        None => println!("{line}"),
    }
    Ok(())
}

/// Scores the experiment the config sets up on a finished analysis. A problem (say, plus maze
/// zones that need a review) is logged and returned; the tracking results stay.
fn score(
    files: &PipelineFiles,
    dir: &Path,
    video: Option<&Path>,
    zones: &Path,
    log: Option<&Log>,
) -> Option<String> {
    let say = |line: &str| match log {
        Some(log) => log.line(line),
        None => println!("{line}"),
    };
    match crate::assay::output::run_config(
        dir,
        &files.config,
        &files.overrides,
        video,
        zones,
        false,
    ) {
        Ok(None) => None,
        Ok(Some(report)) => {
            say(&format!("{}: behavior scored", display_name(dir)));
            for note in &report.notes {
                say(&format!("  {note}"));
            }
            None
        }
        Err(err) => {
            let warning = format!("Behavior: {err:#}");
            say(&format!("{}: {warning}", display_name(dir)));
            Some(warning)
        }
    }
}

/// One unit of a processing run: a video, or the views of one arena in camera-model order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Item {
    /// The results folder's name.
    pub(crate) name: String,
    /// What the progress shows.
    pub(crate) label: String,
    pub(crate) views: Vec<PathBuf>,
}

impl Item {
    pub(crate) fn video(video: &Path) -> Self {
        Self {
            name: recording::safe_name(
                video
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("video"),
            ),
            label: display_name(video),
            views: vec![video.to_path_buf()],
        }
    }
}

/// Videos the Process list holds, grouped by folder for Multiview, with the camera each one is.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewGroup {
    pub(crate) title: String,
    /// Index into the list, and the camera of the model that video is.
    pub(crate) videos: Vec<(usize, Option<String>)>,
    /// Why the folder cannot be fused.
    pub(crate) error: Option<String>,
}

/// `parent/name.mp4`: enough to tell apart same-named videos from different folders.
pub(crate) fn display_name(video: &Path) -> String {
    let name = video.file_name().unwrap_or(video.as_os_str());
    match video.parent().and_then(Path::file_name) {
        Some(parent) => Path::new(parent).join(name).display().to_string(),
        None => name.to_string_lossy().into_owned(),
    }
}

/// `base`, or `base-2`, `base-3`, … : the first name not yet taken in `dir`.
fn free_name(dir: &Path, base: &str) -> String {
    let mut name = base.to_owned();
    for n in 2.. {
        if !dir.join(&name).exists() {
            break;
        }
        name = format!("{base}-{n}");
    }
    name
}

/// Where a processing run saves its results: the custom folder from the settings, or else the
/// folder of the first video.
pub(crate) fn output_dir(settings: &Settings, videos: &[PathBuf]) -> Option<PathBuf> {
    settings
        .output
        .clone()
        .or_else(|| videos.first()?.parent().map(Path::to_path_buf))
}

/// Analyzes each video on its own into `target`, with networks from `models`. A single video is
/// saved in a `<name>-analysis` folder; several go to one `animalpose-batch-…` folder with a
/// subfolder per video and a `batch.json` listing where each input went. A video that fails is
/// reported and the batch moves on to the next one; stopping keeps the videos that already finished.
/// The job's log is saved with the results as `animalpose.log`.
#[cfg(test)]
pub(crate) fn process_batch(
    files: &PipelineFiles,
    videos: &[PathBuf],
    target: &Path,
    control: &ProcessingControl,
    models: &ModelCache,
    log: &Log,
) -> Result<Outcome> {
    let items: Vec<Item> = videos.iter().map(|v| Item::video(v)).collect();
    process_items(
        files,
        &items,
        None,
        target,
        control,
        models,
        log,
        &target.join("zones"),
    )
}

/// [`process_batch`] over items: single videos, and with `cameras` the views of an arena, which
/// are fused into 3D.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_items(
    files: &PipelineFiles,
    items: &[Item],
    rig: Option<&Rig>,
    target: &Path,
    control: &ProcessingControl,
    models: &ModelCache,
    log: &Log,
    zones: &Path,
) -> Result<Outcome> {
    let videos = items;
    anyhow::ensure!(!videos.is_empty(), "Choose at least one video.");
    anyhow::ensure!(
        crate::data::video::ffmpeg_available(),
        "Install FFmpeg to produce an overlay video."
    );
    let report = |progress| control.report(progress);
    report(Progress::Preparing)?;
    log.line(&format!(
        "Processing {} item(s) with {}{}",
        videos.len(),
        files.config.display(),
        files
            .overrides
            .iter()
            .map(|o| format!(" {o}"))
            .collect::<String>()
    ));
    let pipeline = models.get(files)?;
    let warnings = std::cell::RefCell::new(Vec::new());
    let run = |item: &Item, dir: &Path| {
        let video = match (item.views.as_slice(), rig) {
            ([video], _) => {
                predict_one(&pipeline, video, dir, report, Some(log))?;
                Some(video.as_path())
            }
            (views, Some(rig)) => {
                fuse(&pipeline, views, rig, dir, report, Some(log))?;
                None
            }
            (_, None) => anyhow::bail!("Choose a camera model to fuse several views."),
        };
        if let Some(warning) = score(files, dir, video, zones, Some(log)) {
            warnings
                .borrow_mut()
                .push(format!("{}: {warning}", item.label));
        }
        Ok(())
    };
    let noun = if rig.is_some() { "items" } else { "videos" };
    if let [item] = videos {
        control.begin_clip(item.label.clone());
        let name = free_name(target, &format!("{}-analysis", item.name));
        let result = processing::save_output(target, &name, Some(&name), |dir| {
            log.start_job(&dir.join("animalpose.log"));
            run(item, dir)
        });
        log.end_job();
        let dir = result?;
        control.advance();
        let mut status = String::from("Analysis finished.");
        for warning in warnings.borrow().iter() {
            status += &format!("\n{warning}");
        }
        return Ok(Outcome {
            status,
            dir: Some(dir),
        });
    }

    let batch = recording::new_output_dir(target, "animalpose-batch")?;
    log.start_job(&batch.join("animalpose.log"));
    let mut used = HashSet::new();
    let mut entries = Vec::new();
    let mut failed = Vec::new();
    let mut stopped = false;
    for item in videos {
        let video = &item.label;
        let base = item.name.clone();
        let mut name = base.clone();
        for n in 2.. {
            if used.insert(name.clone()) {
                break;
            }
            name = format!("{base}-{n}");
        }
        control.begin_clip(item.label.clone());
        let result = processing::save_output(&batch, &name, Some(&name), |dir| run(item, dir));
        let input = match item.views.as_slice() {
            [video] => serde_json::json!(video),
            views => serde_json::json!(views),
        };
        match result {
            Ok(dir) => entries.push(serde_json::json!({"video": input, "output": dir})),
            Err(err) if err.is::<processing::Stopped>() => {
                log.line("Stopped.");
                stopped = true;
                break;
            }
            Err(err) => {
                let error = format!("{err:#}");
                log.line(&format!("{video}: failed: {error}"));
                failed.push(format!("{video}: {error}"));
                entries.push(serde_json::json!({"video": input, "error": error}));
            }
        }
        control.advance();
    }
    log.end_job();
    let done = entries.iter().filter(|e| e.get("output").is_some()).count();
    if entries.is_empty() {
        let _ = std::fs::remove_file(batch.join("animalpose.log"));
        let _ = std::fs::remove_dir(&batch);
        return Err(processing::Stopped.into());
    }
    let manifest = serde_json::json!({"videos": entries, "stopped": stopped});
    std::fs::write(
        batch.join("batch.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .with_context(|| format!("cannot write {}", batch.join("batch.json").display()))?;
    let mut status = if stopped {
        format!("Stopped after {done} of {} {noun}.", videos.len())
    } else {
        format!("Analyzed {done} of {} {noun}.", videos.len())
    };
    if !failed.is_empty() {
        status += &format!("\n{}", failed.join("\n"));
    }
    for warning in warnings.borrow().iter() {
        status += &format!("\n{warning}");
    }
    Ok(Outcome {
        status,
        dir: Some(batch),
    })
}

/// Expands chosen files and folders into the videos to process. Folders are searched recursively
/// in name order; see [`walk_videos`] for what is skipped.
pub(crate) fn collect_videos(
    paths: impl IntoIterator<Item = PathBuf>,
) -> (Vec<PathBuf>, Vec<String>) {
    let mut found = Vec::new();
    let mut issues = Vec::new();
    for path in paths {
        // Typed paths may be relative; results are saved next to the videos, so keep them whole.
        let path = path.canonicalize().unwrap_or(path);
        if path.is_dir() {
            let before = found.len();
            if let Err(err) = walk_videos(&path, &mut found) {
                issues.push(format!("{}: {err:#}", path.display()));
            } else if found.len() == before {
                issues.push(format!("No videos found in {}", path.display()));
            }
        } else if path.is_file() {
            if crate::data::video::is_video(&path) {
                found.push(path);
            } else {
                issues.push(format!("{} is not a supported video", path.display()));
            }
        } else {
            issues.push(format!("{} does not exist", path.display()));
        }
    }
    (found, issues)
}

/// Videos under `dir`. Hidden entries and symbolic links to folders are skipped, as is any folder
/// holding a `tracks.csv`: that is an analysis this app wrote, whose `overlay.mp4` is not footage.
fn walk_videos(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if dir.join("tracks.csv").is_file() {
        return Ok(());
    }
    let mut entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            walk_videos(&path, out)?;
        } else if path.is_file() && crate::data::video::is_video(&path) {
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) struct CaptureSummary {
    pub(crate) status: String,
    pub(crate) clips: BTreeMap<PathBuf, Vec<PathBuf>>,
    /// The recording's folder.
    pub(crate) dir: Option<PathBuf>,
}

pub(crate) fn finish_capture(
    workers: Vec<Worker>,
    session: Option<Arc<Session>>,
    interrupted: bool,
) -> CaptureSummary {
    let reports: Vec<_> = workers
        .into_iter()
        .map(|worker| {
            let name = worker.source.name();
            let mut report = worker.finish();
            if let Some(session) = &session {
                if report.error.is_none() {
                    let missing = session.missing_stages(&report.clips);
                    if report.clips.is_empty() {
                        report.error = Some("no frames were recorded".into());
                    } else if !interrupted && !missing.is_empty() {
                        report.error = Some(format!(
                            "no frames were recorded for {}",
                            missing.join(", ")
                        ));
                    }
                }
            }
            (name, report)
        })
        .collect();
    let mut clips = BTreeMap::<PathBuf, Vec<PathBuf>>::new();
    let mut errors = Vec::new();
    let mut count = 0;
    for (source, report) in &reports {
        if let Some(error) = &report.error {
            errors.push(format!("{source}: {error}"));
        }
        for clip in &report.clips {
            if let Some(parent) = clip.path.parent() {
                clips
                    .entry(parent.to_path_buf())
                    .or_default()
                    .push(clip.path.clone());
                count += clip.frames;
            }
        }
    }
    let mut status = String::new();
    let dir = session.as_ref().map(|s| s.dir.clone());
    if let Some(session) = session {
        let manifest = serde_json::json!({"sources": reports, "interrupted": interrupted});
        let result = (|| -> Result<()> {
            std::fs::write(
                session.dir.join("recording.json"),
                serde_json::to_vec_pretty(&manifest)?,
            )?;
            Ok(())
        })();
        if let Err(err) = result {
            errors.push(format!("cannot save recording details: {err:#}"));
        }
        let verb = if !errors.is_empty() {
            "Recording incomplete"
        } else if interrupted {
            "Recording ended early"
        } else {
            "Recording finished"
        };
        status = format!("{verb}. {count} frames saved.");
    }
    if !errors.is_empty() {
        status += &format!("\n{}", errors.join("\n"));
    }
    CaptureSummary { status, clips, dir }
}

/// What to bring into the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportKind {
    /// A `.pt` / `.safetensors` checkpoint.
    Model,
    /// A pipeline `.yaml`.
    Config,
    /// Every config and checkpoint in a folder.
    Folder,
}

enum Imported {
    Model(crate::models::ModelKind, String),
    Config(ConfigImport),
    Folder(FolderImport),
}

fn append_issues(status: &mut String, issues: &[String]) {
    if !issues.is_empty() {
        if !status.is_empty() {
            status.push_str("\n\n");
        }
        status.push_str(&issues.join("\n"));
    }
}

/// One preview frame, converted for display.
pub(crate) struct Preview {
    /// The capture sequence number and whether the pose overlay is drawn: a new key means a new
    /// image.
    pub(crate) key: (u64, bool),
    /// `[width, height]` of `rgba`.
    pub(crate) size: [usize; 2],
    /// Straight RGBA pixels, `None` when `key` matches the caller's.
    pub(crate) rgba: Option<Vec<u8>>,
    /// Animals found and the analysis time, with the overlay.
    pub(crate) analysis: Option<(usize, f64)>,
}

/// RGBA at most 960 × 720, enough for a preview tile.
pub(crate) fn preview_rgba(frame: &Bgr) -> ([usize; 2], Vec<u8>) {
    let full_width = frame.width as usize;
    let full_height = frame.height as usize;
    let stride = full_width
        .div_ceil(960)
        .max(full_height.div_ceil(720))
        .max(1);
    let width = full_width.div_ceil(stride);
    let height = full_height.div_ceil(stride);
    let mut rgba = Vec::with_capacity(width * height * 4);
    for y in (0..full_height).step_by(stride) {
        for x in (0..full_width).step_by(stride) {
            let i = (y * full_width + x) * 3;
            rgba.extend_from_slice(&[
                frame.pixels[i + 2],
                frame.pixels[i + 1],
                frame.pixels[i],
                255,
            ]);
        }
    }
    ([width, height], rgba)
}

/// The application: camera sources, the video list, the library and the jobs in flight.
pub struct Engine {
    pub(crate) sources: Vec<Source>,
    pub(crate) selected: Vec<bool>,
    pub(crate) library: Library,
    pub(crate) settings: Settings,
    /// Videos queued for processing, in order.
    pub(crate) videos: Vec<PathBuf>,
    /// Plays the first queued video in the preview as if it were a camera.
    pub(crate) playback: bool,
    /// The cameras of the profile's camera model, or why it could not be read.
    pub(crate) cameras: Option<Result<Vec<Camera>, String>>,
    pub(crate) workers: Vec<Worker>,
    pub(crate) control: Arc<CaptureControl>,
    finishing: Option<JoinHandle<CaptureSummary>>,
    pub(crate) processing: Option<ProcessingJob>,
    importing: Option<Receiver<Result<Imported>>>,
    pub(crate) last_recording: BTreeMap<PathBuf, Vec<PathBuf>>,
    pub(crate) status: String,
    /// The folder the last job wrote, shown next to `status`.
    pub(crate) result: Option<PathBuf>,
    /// Every profile; the one in use is `settings.profile`.
    pub(crate) profiles: Vec<Profile>,
    /// Names the next recording's folder; may be empty.
    pub(crate) recording_name: String,
    /// The config open in the config editor (`Some(None)`: the default pipeline) and its form.
    pub(crate) config_form: Option<(Option<String>, serde_json::Value)>,
    pub(crate) models: Arc<ModelCache>,
    pub(crate) log: Arc<Log>,
    /// Called from background threads when something changed, to wake the window.
    notify: Arc<dyn Fn() + Send + Sync>,
}

/// How many result folders the apps list.
const RECENT_RESULTS: usize = 8;

impl Engine {
    /// The library under `~/AnimalPose` and the cameras the system reports.
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> Self {
        let root = std::env::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("AnimalPose");
        let mut engine = Self::with_library(Library::open(root), list_cameras(), notify);
        engine.log.line("AnimalPose started");
        if let Some(dir) = super::samples::locate() {
            engine.install_samples(&dir);
        }
        engine.prewarm();
        engine
    }

    pub(crate) fn with_library(
        library: Library,
        sources: Vec<Source>,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let (settings, status) = match library.load() {
            Ok(settings) => (settings, String::new()),
            Err(err) => (Settings::default(), format!("{err:#}")),
        };
        let log = Arc::new(Log::new(library.root.join("logs")));
        let mut engine = Self {
            selected: vec![false; sources.len()],
            sources,
            library,
            settings,
            videos: Vec::new(),
            playback: false,
            cameras: None,
            workers: Vec::new(),
            control: Arc::new(CaptureControl::default()),
            finishing: None,
            processing: None,
            importing: None,
            last_recording: BTreeMap::new(),
            status,
            result: None,
            profiles: Vec::new(),
            recording_name: String::new(),
            config_form: None,
            models: Arc::new(ModelCache::default()),
            log,
            notify: Arc::new(notify),
        };
        engine.load_profiles();
        engine.load_camera_model();
        engine
    }

    /// Installs the sample profiles shipped in `dir` ([`super::samples`]); a person's own and
    /// edited files are never replaced.
    pub(crate) fn install_samples(&mut self, dir: &Path) {
        match super::samples::install(&self.library, dir, &mut self.settings) {
            Ok(done) => {
                if !done.added.is_empty() || !done.updated.is_empty() {
                    self.log.line(&format!(
                        "Sample profiles from {}: {} files added, {} updated",
                        dir.display(),
                        done.added.len(),
                        done.updated.len()
                    ));
                }
                let _ = self.library.save(&self.settings);
                self.load_profiles();
                // The active profile's calibration may be one of the files just updated.
                self.load_camera_model();
            }
            Err(err) => self
                .log
                .line(&format!("Sample profiles not installed: {err:#}")),
        }
    }

    fn load_camera_model(&mut self) {
        self.cameras = self.settings.camera_model.as_deref().map(|name| {
            self.library
                .camera_model(name)
                .map_err(|err| format!("{err:#}"))
        });
    }

    /// The camera model's cameras, for Multiview.
    pub(crate) fn camera_model(&self) -> Result<&[Camera]> {
        match &self.cameras {
            Some(Ok(cameras)) => Ok(cameras),
            Some(Err(err)) => anyhow::bail!("The camera model cannot be read: {err}"),
            None => anyhow::bail!("Choose the arena's camera model in the profile."),
        }
    }

    /// The camera model with its reference pictures, for Multiview.
    pub(crate) fn rig(&self) -> Result<Rig> {
        let cameras = self.camera_model()?.to_vec();
        let references = self
            .settings
            .camera_model
            .as_deref()
            .and_then(|name| self.library.camera_model_path(name).ok())
            .map(|path| References::beside(&path));
        Ok(Rig {
            cameras,
            references,
        })
    }

    /// The camera names of the model, if it reads.
    pub(crate) fn camera_names(&self) -> Vec<String> {
        self.camera_model()
            .map(|cams| cams.iter().map(|c| c.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Copies a calibration into the library and makes it profile `profile`'s camera model.
    pub(crate) fn import_camera_model(&mut self, path: &Path, profile: &str) {
        let imported = self.library.import_camera_model(path).and_then(|name| {
            let mut next = self
                .profiles
                .iter()
                .find(|p| p.name == profile)
                .cloned()
                .context("That profile no longer exists.")?;
            next.camera_model = Some(name);
            self.save_profile(Some(profile), next)
        });
        if let Err(err) = imported {
            self.status = format!("{err:#}");
        }
    }

    pub(crate) fn set_multiview(&mut self, on: bool) {
        self.settings.multiview = on;
        self.save();
    }

    /// The key a camera's view is saved under: its name, numbered when several cameras share it.
    pub(crate) fn source_key(&self, source: &Source) -> String {
        let name = source.name();
        let Source::Camera { index, .. } = source else {
            return name;
        };
        let twins: Vec<u32> = self
            .sources
            .iter()
            .filter_map(|s| match s {
                Source::Camera { index, name: other } if *other == name => Some(*index),
                _ => None,
            })
            .collect();
        match twins.iter().position(|i| i == index) {
            Some(n) if n > 0 => format!("{name} ({})", n + 1),
            _ => name,
        }
    }

    /// The view each preview source records as.
    pub(crate) fn views(&self) -> Vec<String> {
        self.views_of(self.workers.iter().map(|w| &w.source))
    }

    /// The label each source records as: the name given to it, or its default (see
    /// [`Self::default_view`]).
    pub(crate) fn views_of<'a>(&self, sources: impl Iterator<Item = &'a Source>) -> Vec<String> {
        sources
            .map(|source| {
                self.settings
                    .views
                    .get(&self.source_key(source))
                    .cloned()
                    .unwrap_or_else(|| self.default_view(source))
            })
            .collect()
    }

    /// A camera's label before it is named: `Camera <ID>`, its place in the system's camera list
    /// from 1, so each camera keeps its default whichever cameras are chosen. A video played as a
    /// camera is `Video`.
    pub(crate) fn default_view(&self, source: &Source) -> String {
        match self.sources.iter().position(|s| s == source) {
            Some(i) => views::default_view(i),
            None => "Video".into(),
        }
    }

    /// Names the view preview source `index` records as; blank goes back to the default.
    pub(crate) fn set_view(&mut self, index: usize, view: &str) {
        if let Some(source) = self.workers.get(index).map(|w| w.source.clone()) {
            self.name_view(&source, view);
        }
    }

    pub(crate) fn name_view(&mut self, source: &Source, view: &str) {
        let key = self.source_key(source);
        match view.trim() {
            "" => self.settings.views.remove(&key),
            view => self.settings.views.insert(key, view.to_owned()),
        };
        self.save();
    }

    /// With Multiview, the Process list by folder and the camera each video is.
    pub(crate) fn view_groups(&self) -> Option<Vec<ViewGroup>> {
        if !self.settings.multiview {
            return None;
        }
        let names = self
            .camera_model()
            .map(|cams| cams.iter().map(|c| c.name.clone()).collect::<Vec<_>>());
        let groups = views::by_folder(&self.videos)
            .into_iter()
            .map(|group| {
                let indices: Vec<usize> = group
                    .iter()
                    .filter_map(|v| self.videos.iter().position(|w| w == v))
                    .collect();
                let title = group[0].parent().map(display_name).unwrap_or_default();
                let matched = match (&names, group.len()) {
                    (_, 1) => Ok(None),
                    (Ok(names), _) => views::match_cameras(&group, names).map(Some),
                    (Err(err), _) => Err(anyhow::anyhow!("{err:#}")),
                };
                let (videos, error) = match matched {
                    Ok(Some(ordered)) => (
                        indices
                            .iter()
                            .map(|&i| {
                                let at = ordered.iter().position(|v| *v == self.videos[i]);
                                (i, at.and_then(|at| names.as_ref().ok()?.get(at).cloned()))
                            })
                            .collect(),
                        None,
                    ),
                    Ok(None) => (indices.iter().map(|&i| (i, None)).collect(), None),
                    Err(err) => (
                        indices.iter().map(|&i| (i, None)).collect(),
                        Some(format!("{err:#}")),
                    ),
                };
                ViewGroup {
                    title,
                    videos,
                    error,
                }
            })
            .collect();
        Some(groups)
    }

    /// What Process analyzes: each video on its own, or with Multiview each folder's videos
    /// together as the views of the camera model (a folder with one video stays single-view).
    pub(crate) fn process_items(&self) -> Result<(Vec<Item>, Option<Rig>)> {
        if !self.settings.multiview {
            return Ok((self.videos.iter().map(|v| Item::video(v)).collect(), None));
        }
        let rig = self.rig()?;
        let names: Vec<String> = rig.cameras.iter().map(|c| c.name.clone()).collect();
        let mut items = Vec::new();
        for group in views::by_folder(&self.videos) {
            if let [video] = group.as_slice() {
                items.push(Item::video(video));
                continue;
            }
            let folder = group[0].parent().unwrap_or(Path::new(""));
            let ordered =
                views::match_cameras(&group, &names).with_context(|| display_name(folder))?;
            items.push(Item {
                name: recording::safe_name(
                    &folder
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "views".into()),
                ),
                label: display_name(folder),
                views: ordered,
            });
        }
        Ok((items, Some(rig)))
    }

    /// Reads the profiles and puts the chosen one in use. A library without profiles gets a
    /// "Default" profile made from the current settings.
    fn load_profiles(&mut self) {
        self.profiles = self.library.profiles();
        if self.profiles.is_empty() {
            let name = self
                .settings
                .profile
                .clone()
                .unwrap_or_else(|| "Default".into());
            let profile = Profile::from_settings(&name, &self.settings);
            if let Err(err) = self.library.save_profile(&profile, None) {
                self.status = format!("Could not save the profile: {err:#}");
            }
            self.profiles = vec![profile];
        }
        let active = self
            .settings
            .profile
            .as_ref()
            .and_then(|name| self.profiles.iter().find(|p| &p.name == name))
            .unwrap_or(&self.profiles[0])
            .clone();
        active.apply(&mut self.settings);
        let _ = self.library.save(&self.settings);
    }

    /// Loads the active profile's networks in the background.
    pub(crate) fn prewarm(&self) {
        if let Ok(files) = self.library.pipeline(&self.settings) {
            self.models.prewarm(files);
        }
    }

    /// Puts profile `name` in use.
    pub(crate) fn select_profile(&mut self, name: &str) {
        if !self.idle() {
            return;
        }
        let Some(profile) = self.profiles.iter().find(|p| p.name == name).cloned() else {
            return;
        };
        profile.apply(&mut self.settings);
        self.save();
        self.log.line(&format!("Profile: {name}"));
        self.prewarm();
    }

    /// Saves `profile`, which replaces the one called `previous` (renaming it) or is new.
    pub(crate) fn save_profile(&mut self, previous: Option<&str>, profile: Profile) -> Result<()> {
        anyhow::ensure!(self.idle(), "Wait for the current job to finish.");
        self.library.save_profile(&profile, previous)?;
        let name = profile.name.trim().to_owned();
        let active = self.settings.profile.as_deref();
        let was_active = previous.is_some_and(|old| Some(old) == active);
        self.profiles = self.library.profiles();
        if was_active || previous.is_none() {
            if let Some(saved) = self.profiles.iter().find(|p| p.name == name).cloned() {
                saved.apply(&mut self.settings);
                self.save();
                self.prewarm();
            }
        }
        Ok(())
    }

    /// Deletes profile `name`; the last profile cannot be deleted.
    pub(crate) fn delete_profile(&mut self, name: &str) -> Result<()> {
        anyhow::ensure!(self.idle(), "Wait for the current job to finish.");
        anyhow::ensure!(self.profiles.len() > 1, "Keep at least one profile.");
        self.library.delete_profile(name)?;
        self.profiles = self.library.profiles();
        if self.settings.profile.as_deref() == Some(name) {
            let first = self.profiles[0].clone();
            first.apply(&mut self.settings);
            self.save();
            self.prewarm();
        }
        Ok(())
    }

    /// A free profile name based on `base`: `base`, `base 2`, …
    /// Copies profile `source` as "`source` Copy" (numbered when taken); the profile in use stays.
    /// Returns the copy's name.
    pub(crate) fn duplicate_profile(&mut self, source: &str) -> Result<String> {
        anyhow::ensure!(self.idle(), "Wait for the current job to finish.");
        let mut copy = self
            .profiles
            .iter()
            .find(|p| p.name == source)
            .cloned()
            .with_context(|| format!("there is no profile {source}"))?;
        copy.name = self.new_profile_name(&format!("{source} Copy"));
        self.library.save_profile(&copy, None)?;
        self.profiles = self.library.profiles();
        self.log
            .line(&format!("Profile {source} copied as {}", copy.name));
        Ok(copy.name)
    }

    pub(crate) fn new_profile_name(&self, base: &str) -> String {
        let taken = |name: &str| {
            self.profiles
                .iter()
                .any(|p| recording::safe_name(&p.name) == recording::safe_name(name))
        };
        let mut name = base.to_owned();
        for n in 2.. {
            if !taken(&name) {
                break;
            }
            name = format!("{base} {n}");
        }
        name
    }

    /// Opens config `name` (`None`: the default pipeline) in the config editor.
    pub(crate) fn edit_config(&mut self, name: Option<String>) -> Result<()> {
        let form = super::configs::form(&self.library, name.as_deref())?;
        self.config_form = Some((name, form));
        Ok(())
    }

    /// Changes one setting of the config open in the editor.
    pub(crate) fn set_config_value(&mut self, key: &str, value: &serde_json::Value) -> Result<()> {
        anyhow::ensure!(self.idle(), "Wait for the current job to finish.");
        let Some((Some(name), _)) = &self.config_form else {
            anyhow::bail!("Duplicate the default config to change it.");
        };
        let name = name.clone();
        super::configs::set(&self.library, &name, key, value)?;
        self.edit_config(Some(name))?;
        self.prewarm();
        Ok(())
    }

    /// Copies config `source` as `new_name` and opens the copy in the editor.
    pub(crate) fn duplicate_config(&mut self, source: Option<&str>, new_name: &str) -> Result<()> {
        let name = super::configs::duplicate(&self.library, source, new_name)?;
        self.edit_config(Some(name))
    }

    pub(crate) fn delete_config(&mut self, name: &str) -> Result<()> {
        anyhow::ensure!(self.idle(), "Wait for the current job to finish.");
        super::configs::delete(&self.library, name)?;
        self.edit_config(None)
    }

    /// Lists `dir` first among the recent results.
    fn remember_result(&mut self, dir: PathBuf) {
        self.settings.recent_results.retain(|d| d != &dir);
        self.settings.recent_results.insert(0, dir.clone());
        self.settings.recent_results.truncate(RECENT_RESULTS);
        self.result = Some(dir);
        let _ = self.library.save(&self.settings);
    }

    pub(crate) fn recording_dir(&self) -> PathBuf {
        self.settings
            .recordings
            .clone()
            .unwrap_or_else(|| self.library.root.join("recordings"))
    }

    /// Where processing the queued videos would save its results.
    pub(crate) fn output_dir(&self) -> Option<PathBuf> {
        output_dir(&self.settings, &self.videos)
    }

    /// Writes the settings, and the profile in use with them.
    pub(crate) fn save(&mut self) {
        self.load_camera_model();
        if let Err(err) = self.library.save(&self.settings) {
            self.status = format!("Could not save settings: {err:#}");
            return;
        }
        let Some(name) = self.settings.profile.clone() else {
            return;
        };
        let profile = Profile::from_settings(&name, &self.settings);
        if self.profiles.contains(&profile) {
            return;
        }
        match self.library.save_profile(&profile, Some(&name)) {
            Ok(()) => self.profiles = self.library.profiles(),
            Err(err) => self.status = format!("Could not save the profile: {err:#}"),
        }
    }

    pub(crate) fn session(&self) -> Option<Arc<Session>> {
        self.control.session.lock().unwrap().clone()
    }

    /// A job is saving, analyzing or importing.
    pub(crate) fn busy(&self) -> bool {
        self.processing.is_some() || self.finishing.is_some() || self.importing.is_some()
    }

    pub(crate) fn importing(&self) -> bool {
        self.importing.is_some()
    }

    pub(crate) fn finishing(&self) -> bool {
        self.finishing.is_some()
    }

    /// Nothing runs: sources and settings can change.
    pub(crate) fn idle(&self) -> bool {
        self.workers.is_empty() && !self.busy()
    }

    pub(crate) fn refresh_cameras(&mut self) {
        if !self.workers.is_empty() {
            return;
        }
        self.sources = list_cameras();
        self.selected = vec![false; self.sources.len()];
    }

    pub(crate) fn set_live(&mut self, live: bool) {
        self.settings.live = live;
        for worker in &self.workers {
            worker.live.store(live, Ordering::Relaxed);
        }
        self.save();
    }

    /// Every preview source is open and none has ended.
    pub(crate) fn sources_ready(&self) -> bool {
        !self.workers.is_empty()
            && self
                .workers
                .iter()
                .all(|w| w.ready.load(Ordering::Acquire) && !w.ended.load(Ordering::Acquire))
    }

    pub(crate) fn start_preview(&mut self) {
        if !self.workers.is_empty() || self.finishing.is_some() {
            return;
        }
        let mut chosen: Vec<Source> = self
            .sources
            .iter()
            .zip(&self.selected)
            .filter(|(_, on)| **on)
            .map(|(source, _)| source.clone())
            .collect();
        if let Some(video) = self.videos.first().filter(|_| self.playback) {
            chosen.push(Source::File(video.clone()));
        }
        if chosen.is_empty() {
            self.status = "Select a camera or select a video for preview.".into();
            return;
        }
        self.control = Arc::new(CaptureControl::default());
        let files = match self.library.pipeline(&self.settings) {
            Ok(files) => Some(files),
            Err(err) => {
                self.status = format!("Live pose preview: {err:#}. Raw preview is available.");
                None
            }
        };
        if files.is_some() {
            self.status = "Opening sources…".into();
        }
        self.workers = chosen
            .into_iter()
            .enumerate()
            .map(|(index, source)| {
                Worker::start(
                    source,
                    files.clone(),
                    self.models.clone(),
                    self.settings.live,
                    index,
                    self.control.clone(),
                )
            })
            .collect();
    }

    pub(crate) fn start_recording(&mut self, timed: bool) {
        if !self.sources_ready() || self.session().is_some() {
            return;
        }
        if !crate::data::video::ffmpeg_available() || !crate::data::video::ffprobe_available() {
            self.status = "Install FFmpeg and ffprobe before recording.".into();
            return;
        }
        let stages = if timed {
            self.settings.stages.clone()
        } else {
            Vec::new()
        };
        if timed && stages.is_empty() {
            self.status = "Add at least one protocol stage.".into();
            return;
        }
        let target = self.recording_dir();
        let sources = self
            .workers
            .iter()
            .map(|w| w.source.name())
            .collect::<Vec<_>>();
        match Session::start(
            &target,
            &self.recording_name,
            stages,
            &sources,
            &self.views(),
        ) {
            Ok(session) => {
                self.status.clear();
                self.log.line(&format!(
                    "Recording {} to {}",
                    sources.join(", "),
                    session.dir.display()
                ));
                *self.control.session.lock().unwrap() = Some(session);
            }
            Err(err) => self.status = format!("Could not start recording: {err:#}"),
        }
    }

    /// Moves a timed recording past a stage that waits for Continue.
    pub(crate) fn continue_stage(&self) {
        if let Some(session) = self.session() {
            session.resume();
        }
    }

    /// Stops recording (and the preview) and saves the files.
    pub(crate) fn stop_recording(&mut self) {
        if let Some(session) = self.session() {
            let (phase, _, _) = session.position();
            self.stop_sources(!session.stages.is_empty() && phase != Phase::Finished);
        }
    }

    pub(crate) fn stop_sources(&mut self, interrupted: bool) {
        if self.workers.is_empty() {
            return;
        }
        self.control.stop.store(true, Ordering::Relaxed);
        let session = self.control.session.lock().unwrap().take();
        if let Some(session) = &session {
            session.stop();
            self.status = "Saving MP4 files…".into();
        }
        let workers = std::mem::take(&mut self.workers);
        for worker in &workers {
            worker.latest.ready.notify_all();
        }
        let notify = self.notify.clone();
        self.finishing = Some(std::thread::spawn(move || {
            let summary = finish_capture(workers, session, interrupted);
            notify();
            summary
        }));
    }

    /// Collects finished jobs and imports; ends a recording whose protocol is over.
    pub(crate) fn poll(&mut self) {
        if let Some(session) = self.session() {
            if session.position().0 == Phase::Finished {
                self.stop_sources(false);
            } else if self.workers.iter().any(|w| w.ended.load(Ordering::Acquire)) {
                self.stop_sources(!session.stages.is_empty());
            }
        }
        if self
            .finishing
            .as_ref()
            .is_some_and(|task| task.is_finished())
        {
            match self.finishing.take().unwrap().join() {
                Ok(summary) => {
                    if !summary.clips.is_empty() {
                        self.last_recording = summary.clips;
                    }
                    self.log.line(&summary.status);
                    self.result = summary.dir;
                    self.status = summary.status;
                }
                Err(_) => self.status = "Could not finalize the recording.".into(),
            }
        }
        if self
            .processing
            .as_ref()
            .is_some_and(|job| job.thread.is_finished())
        {
            self.status = match self.processing.take().unwrap().thread.join() {
                Ok(Ok(outcome)) => {
                    if let Some(dir) = outcome.dir {
                        self.remember_result(dir);
                    }
                    outcome.status
                }
                Ok(Err(err)) if err.is::<processing::Stopped>() => err.to_string(),
                Ok(Err(err)) => format!("Video processing failed: {err:#}"),
                Err(_) => "The video processing thread stopped unexpectedly.".into(),
            };
            self.log.line(&self.status);
        }
        self.poll_import();
    }

    /// Queues the videos found in `paths` (files or folders), skipping ones already queued.
    pub(crate) fn add_videos(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        let (found, issues) = collect_videos(paths);
        let mut queued: HashSet<PathBuf> = self.videos.iter().cloned().collect();
        let before = self.videos.len();
        self.videos.extend(
            found
                .into_iter()
                .filter(|video| queued.insert(video.clone())),
        );
        // The list shows what was added; only problems need a message.
        self.status.clear();
        if self.videos.len() == before && issues.is_empty() {
            self.status = "Already in the list.".into();
        }
        append_issues(&mut self.status, &issues);
    }

    pub(crate) fn remove_video(&mut self, index: usize) {
        if index < self.videos.len() {
            self.videos.remove(index);
        }
    }

    /// Analyzes the queued videos into [`Self::output_dir`].
    pub(crate) fn process_videos(&mut self) {
        if !self.workers.is_empty() || self.busy() {
            return;
        }
        let Some(target) = self.output_dir() else {
            self.status = "Add a video or a folder of videos first.".into();
            return;
        };
        let files = match self.library.pipeline(&self.settings) {
            Ok(files) => files,
            Err(err) => {
                self.status = format!("{err:#}");
                return;
            }
        };
        let (items, cameras) = match self.process_items() {
            Ok(plan) => plan,
            Err(err) => {
                self.status = format!("{err:#}");
                return;
            }
        };
        self.status.clear();
        let notify = self.notify.clone();
        let (models, log) = (self.models.clone(), self.log.clone());
        let zones = self.library.root.join("zones");
        self.result = None;
        self.processing = Some(ProcessingJob::start(items.len(), move |control| {
            let result = process_items(
                &files,
                &items,
                cameras.as_ref(),
                &target,
                control,
                &models,
                &log,
                &zones,
            );
            notify();
            result
        }));
    }

    /// Analyzes every stage of the last recording, next to its clips.
    pub(crate) fn analyze_last(&mut self) {
        if !self.workers.is_empty() || self.busy() {
            return;
        }
        let files = match self.library.pipeline(&self.settings) {
            Ok(files) => files,
            Err(err) => {
                self.status = format!("{err:#}");
                return;
            }
        };
        let groups = self.last_recording.clone();
        // A recording of several cameras is fused when the profile has a camera model.
        let rig = self.rig().ok();
        let Some(result_dir) = groups
            .keys()
            .next()
            .and_then(|stage| stage.parent())
            .map(Path::to_path_buf)
        else {
            return;
        };
        let total = groups
            .values()
            .map(|clips| {
                if rig.is_some() && clips.len() > 1 {
                    1
                } else {
                    clips.len()
                }
            })
            .sum();
        self.status.clear();
        let notify = self.notify.clone();
        let (models, log) = (self.models.clone(), self.log.clone());
        let zones = self.library.root.join("zones");
        self.result = None;
        self.processing = Some(ProcessingJob::start(total, move |control| {
            let result = (|| {
                log.line(&format!(
                    "Analyzing the recording in {}",
                    result_dir.display()
                ));
                for (stage, clips) in groups {
                    processing::save_output(&stage, "analysis", Some("analysis"), |dir| {
                        log.start_job(&dir.join("animalpose.log"));
                        let done = process_views(
                            &files,
                            &clips,
                            rig.as_ref(),
                            dir,
                            Some(control),
                            &models,
                            Some(&log),
                            &zones,
                        );
                        log.end_job();
                        done
                    })?;
                }
                Ok(Outcome {
                    status: "Analysis finished.".into(),
                    dir: Some(result_dir),
                })
            })();
            notify();
            result
        }));
    }

    pub(crate) fn pause_or_resume(&self) {
        if let Some(job) = &self.processing {
            job.control.pause_or_resume();
        }
    }

    pub(crate) fn stop_processing(&self) {
        if let Some(job) = &self.processing {
            job.control.stop();
        }
    }

    /// Copies a checkpoint, config or folder into the library in the background.
    pub(crate) fn import(&mut self, kind: ImportKind, path: PathBuf) {
        if self.importing.is_some() {
            return;
        }
        let library = self.library.clone();
        let notify = self.notify.clone();
        let (send, receive) = mpsc::channel();
        self.status = format!("Importing {}…", path.display());
        self.importing = Some(receive);
        std::thread::spawn(move || {
            let result = match kind {
                ImportKind::Model => library
                    .import_model(&path)
                    .map(|(kind, name)| Imported::Model(kind, name)),
                ImportKind::Config => library.import_config(&path).map(Imported::Config),
                ImportKind::Folder => library.import_folder(&path).map(Imported::Folder),
            };
            let _ = send.send(result);
            notify();
        });
    }

    fn poll_import(&mut self) {
        let Some(task) = &self.importing else { return };
        let result = match task.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err(anyhow::anyhow!("Import stopped unexpectedly.")),
        };
        self.importing = None;
        match result {
            Ok(Imported::Model(kind, name)) => {
                self.status = format!("{kind} checkpoint available as {name}");
                self.settings.models.insert(kind.as_str().to_string(), name);
                self.save();
            }
            Ok(Imported::Config(import)) => {
                self.status = format!(
                    "Imported {}. This config uses the models selected in the app.",
                    import.name
                );
                append_issues(&mut self.status, &import.issues);
                self.settings.config = Some(import.name);
                self.save();
            }
            Ok(Imported::Folder(import)) => {
                self.status =
                    if import.configs == 0 && import.models == 0 && import.issues.is_empty() {
                        "No YAML configs or model checkpoints found in that folder.".into()
                    } else {
                        format!(
                        "{} configurations and {} distinct checkpoints available from this folder.",
                        import.configs, import.models
                    )
                    };
                append_issues(&mut self.status, &import.issues);
            }
            Err(err) => self.status = format!("Import failed: {err:#}"),
        }
    }

    /// The newest frame of preview source `index`: with the pose overlay when live analysis runs
    /// and nothing is recording, raw otherwise. Pixels are converted only when the frame differs
    /// from `known`.
    pub(crate) fn preview(&self, index: usize, known: Option<(u64, bool)>) -> Option<Preview> {
        let worker = self.workers.get(index)?;
        let overlay = self.session().is_none() && self.settings.live;
        if overlay {
            let shown = worker.shown.lock().unwrap();
            if let Some(f) = shown.as_ref() {
                let key = (f.sequence, true);
                let (size, rgba) = if known == Some(key) {
                    ([0, 0], None)
                } else {
                    let (size, rgba) = preview_rgba(&f.image);
                    (size, Some(rgba))
                };
                return Some(Preview {
                    key,
                    size,
                    rgba,
                    analysis: Some((f.animals, f.analysis_ms)),
                });
            }
        }
        let slot = worker.latest.frame.lock().unwrap();
        let raw = slot.as_ref()?;
        let key = (raw.sequence, false);
        // Release the capture lock before converting pixels.
        let image = (known != Some(key)).then(|| raw.image.clone());
        drop(slot);
        let (size, rgba) = match image {
            Some(image) => {
                let (size, rgba) = preview_rgba(&image);
                (size, Some(rgba))
            }
            None => ([0, 0], None),
        };
        Some(Preview {
            key,
            size,
            rgba,
            analysis: None,
        })
    }

    /// Stops every job before the window closes.
    pub(crate) fn shutdown(&mut self) {
        if let Some(job) = self.processing.take() {
            job.control.stop();
            let _ = job.thread.join();
        }
        self.stop_sources(true);
        if let Some(task) = self.finishing.take() {
            let _ = task.join();
        }
    }
}
