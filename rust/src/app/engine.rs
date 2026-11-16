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
use super::library::{ConfigImport, FolderImport, Library, PipelineFiles, Settings};
use super::processing::{self, Control as ProcessingControl, Job as ProcessingJob};
use super::recording::{self, Phase, Session};
use crate::algos::image::Bgr;
use crate::cli::predict::Progress;

/// Runs the offline pipeline on one recording. Multiple views are processed separately unless
/// a camera calibration file is supplied for 3D fusion.
pub fn analyze(config: &str, views: &[PathBuf], cameras: Option<&str>) -> Result<PathBuf> {
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
        cameras,
        &dir,
        None,
    )?;
    Ok(dir)
}

pub(crate) fn process_views(
    files: &PipelineFiles,
    views: &[PathBuf],
    cameras: Option<&str>,
    dir: &Path,
    control: Option<&ProcessingControl>,
) -> Result<()> {
    anyhow::ensure!(
        crate::data::video::ffmpeg_available(),
        "Install FFmpeg to produce an overlay video."
    );
    let report = |progress| match control {
        Some(control) => control.report(progress),
        None => Ok(()),
    };
    let config = files.config.to_string_lossy();
    if let Some(cameras) = cameras.filter(|_| views.len() > 1) {
        let views = views
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        crate::multiview::pipeline::run_cli_with_progress(
            &config,
            &views,
            cameras,
            &dir.to_string_lossy(),
            None,
            &files.overrides,
            report,
        )?;
        if let Some(control) = control {
            control.clip_done()?;
        }
        return Ok(());
    }
    report(Progress::Preparing)?;
    let pipeline = crate::cli::predict::load_pipeline(&config, &files.overrides)?;
    for v in views {
        let out = if views.len() == 1 {
            dir.to_path_buf()
        } else {
            dir.join(v.file_stem().context("video needs a filename")?)
        };
        if let Some(control) = control {
            control.begin_clip(display_name(v));
        }
        predict_one(&pipeline, v, &out, report)?;
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
) -> Result<()> {
    anyhow::ensure!(video.is_file(), "{} does not exist", video.display());
    crate::cli::predict::run_loaded_with_progress(
        pipeline,
        &video.to_string_lossy(),
        &out.to_string_lossy(),
        None,
        None,
        true,
        report,
    )?;
    for name in ["tracks.csv", "tracks.npz", "overlay.mp4"] {
        anyhow::ensure!(
            out.join(name).is_file(),
            "{} was not produced",
            out.join(name).display()
        );
    }
    Ok(())
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

/// Analyzes each video on its own, loading the networks once, into `target`. A single video is
/// saved in a `<name>-analysis` folder; several go to one `animalpose-batch-…` folder with a
/// subfolder per video and a `batch.json` listing where each input went. A video that fails is
/// reported and the batch moves on to the next one; stopping keeps the videos that already finished.
pub(crate) fn process_batch(
    files: &PipelineFiles,
    videos: &[PathBuf],
    target: &Path,
    control: &ProcessingControl,
) -> Result<String> {
    anyhow::ensure!(!videos.is_empty(), "Choose at least one video.");
    anyhow::ensure!(
        crate::data::video::ffmpeg_available(),
        "Install FFmpeg to produce an overlay video."
    );
    let report = |progress| control.report(progress);
    report(Progress::Preparing)?;
    let pipeline =
        crate::cli::predict::load_pipeline(&files.config.to_string_lossy(), &files.overrides)?;
    let stem = |video: &Path| {
        recording::safe_name(
            video
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video"),
        )
    };
    if let [video] = videos {
        control.begin_clip(display_name(video));
        let name = free_name(target, &format!("{}-analysis", stem(video)));
        let dir = processing::save_output(target, &name, Some(&name), |dir| {
            predict_one(&pipeline, video, dir, report)
        })?;
        control.advance();
        return Ok(format!("Analysis saved to {}", dir.display()));
    }

    let batch = recording::new_output_dir(target, "animalpose-batch")?;
    let mut used = HashSet::new();
    let mut entries = Vec::new();
    let mut failed = Vec::new();
    let mut stopped = false;
    for video in videos {
        let base = stem(video);
        let mut name = base.clone();
        for n in 2.. {
            if used.insert(name.clone()) {
                break;
            }
            name = format!("{base}-{n}");
        }
        control.begin_clip(display_name(video));
        let result = processing::save_output(&batch, &name, Some(&name), |dir| {
            predict_one(&pipeline, video, dir, report)
        });
        match result {
            Ok(dir) => entries.push(serde_json::json!({"video": video, "output": dir})),
            Err(err) if err.is::<processing::Stopped>() => {
                stopped = true;
                break;
            }
            Err(err) => {
                let error = format!("{err:#}");
                failed.push(format!("{}: {error}", display_name(video)));
                entries.push(serde_json::json!({"video": video, "error": error}));
            }
        }
        control.advance();
    }
    let done = entries.iter().filter(|e| e.get("output").is_some()).count();
    if entries.is_empty() {
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
        format!(
            "Stopped after {done} of {} videos. Finished results are in {}",
            videos.len(),
            batch.display()
        )
    } else {
        format!(
            "Analyzed {done} of {} videos. Results saved to {}",
            videos.len(),
            batch.display()
        )
    };
    if !failed.is_empty() {
        status += &format!("\n\nFailed:\n{}", failed.join("\n"));
    }
    Ok(status)
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
    let mut status = "Preview stopped".to_owned();
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
        status = format!("{verb}. {count} frames saved in {}", session.dir.display());
    }
    if !errors.is_empty() {
        status += &format!("\n{}", errors.join("\n"));
    }
    CaptureSummary { status, clips }
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
    pub(crate) cameras_json: Option<PathBuf>,
    pub(crate) workers: Vec<Worker>,
    pub(crate) control: Arc<CaptureControl>,
    finishing: Option<JoinHandle<CaptureSummary>>,
    pub(crate) processing: Option<ProcessingJob>,
    importing: Option<Receiver<Result<Imported>>>,
    pub(crate) last_recording: BTreeMap<PathBuf, Vec<PathBuf>>,
    pub(crate) status: String,
    /// Called from background threads when something changed, to wake the window.
    notify: Arc<dyn Fn() + Send + Sync>,
}

impl Engine {
    /// The library under `~/AnimalPose` and the cameras the system reports.
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> Self {
        let root = std::env::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("AnimalPose");
        Self::with_library(Library::open(root), list_cameras(), notify)
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
        Self {
            selected: vec![false; sources.len()],
            sources,
            library,
            settings,
            videos: Vec::new(),
            playback: false,
            cameras_json: None,
            workers: Vec::new(),
            control: Arc::new(CaptureControl::default()),
            finishing: None,
            processing: None,
            importing: None,
            last_recording: BTreeMap::new(),
            status,
            notify: Arc::new(notify),
        }
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

    /// Writes the settings to the library.
    pub(crate) fn save(&mut self) {
        if let Err(err) = self.library.save(&self.settings) {
            self.status = format!("Could not save settings: {err:#}");
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
        match Session::start(&target, stages, &sources) {
            Ok(session) => {
                self.status = format!("Recording to {}", session.dir.display());
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
                Ok(Ok(status)) => status,
                Ok(Err(err)) if err.is::<processing::Stopped>() => err.to_string(),
                Ok(Err(err)) => format!("Video processing failed: {err:#}"),
                Err(_) => "The video processing thread stopped unexpectedly.".into(),
            };
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
        let added = self.videos.len() - before;
        self.status = match added {
            0 if issues.is_empty() => "Those videos are already in the list.".into(),
            0 => String::new(),
            1 => format!("Added 1 video ({} in the list).", self.videos.len()),
            n => format!("Added {n} videos ({} in the list).", self.videos.len()),
        };
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
        let videos = self.videos.clone();
        self.status.clear();
        let notify = self.notify.clone();
        self.processing = Some(ProcessingJob::start(videos.len(), move |control| {
            let result = process_batch(&files, &videos, &target, control);
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
        let cameras = self
            .cameras_json
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned());
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
                if cameras.is_some() && clips.len() > 1 {
                    1
                } else {
                    clips.len()
                }
            })
            .sum();
        self.status.clear();
        let notify = self.notify.clone();
        self.processing = Some(ProcessingJob::start(total, move |control| {
            let result = (|| {
                for (stage, clips) in groups {
                    processing::save_output(&stage, "analysis", Some("analysis"), |dir| {
                        process_views(&files, &clips, cameras.as_deref(), dir, Some(control))
                    })?;
                }
                Ok(format!("Analysis saved to {}", result_dir.display()))
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
