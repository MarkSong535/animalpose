//! `animalpose-gui`: pick USB cameras, watch live tracking, record, and analyze the recording.
//!
//! Every selected source gets two threads:
//! * **capture** reads every frame, appends it to the recording while one is running, and posts it as
//!   the latest frame;
//! * **analysis** takes the latest frame, runs [`LiveTracker`] on it (on the GPU with the `mps`
//!   feature) and draws the skeletons, so slow analysis drops displayed frames, never recorded ones.
//!
//! **Analyze** runs the full offline pipeline on the last recording: `predict` for one camera, or
//! the multi-view 3D pipeline when several cameras were recorded and a `cameras.json` is given.
//! Cameras are free-running, not hardware-synchronized.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use eframe::egui;

use crate::algos::image::Bgr;
use crate::cli::overlay::draw_pose;
use crate::cli::predict::{default_pipeline_config, LiveAnimal, LiveTracker, Pipeline};
use crate::data::config::Config;
use crate::data::video::{VideoReader, VideoWriter};

/// Where frames come from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// A camera by its index in the system list.
    Camera { index: u32, name: String },
    /// A video file played at its own frame rate, like a camera (for trying things out).
    File(PathBuf),
}

impl Source {
    pub fn name(&self) -> String {
        match self {
            Self::Camera { name, .. } => name.clone(),
            Self::File(p) => p
                .file_name()
                .map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into()),
        }
    }
}

/// The cameras the system reports.
pub fn list_cameras() -> Vec<Source> {
    nokhwa::query(nokhwa::utils::ApiBackend::Auto)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|c| match c.index() {
            nokhwa::utils::CameraIndex::Index(index) => Some(Source::Camera {
                index: *index,
                name: c.human_name(),
            }),
            nokhwa::utils::CameraIndex::String(_) => None,
        })
        .collect()
}

/// The newest frame of a source, with a sequence number so the analysis sees each frame once.
#[derive(Default)]
struct Latest {
    frame: Mutex<Option<(u64, Bgr)>>,
    ready: Condvar,
}

/// What the analysis thread hands the window.
pub struct Shown {
    pub image: Bgr,
    pub animals: Vec<LiveAnimal>,
    pub analysis_ms: f64,
}

/// One running source: its threads and the state they share with the window.
pub struct Worker {
    pub source: Source,
    stop: Arc<AtomicBool>,
    pub live: Arc<AtomicBool>,
    record: Arc<Mutex<Option<VideoWriter>>>,
    pub shown: Arc<Mutex<Option<Shown>>>,
    pub fps: Arc<Mutex<f64>>,
    pub frames: Arc<AtomicU64>,
    pub error: Arc<Mutex<Option<String>>>,
    threads: Vec<JoinHandle<()>>,
}

impl Worker {
    /// Starts capturing from `source`; the analysis thread loads the pipeline of `config` (each
    /// thread its own: the networks are not shared between threads) and tracks while `live` is set.
    pub fn start(source: Source, config: String, live: bool) -> Self {
        let latest = Arc::new(Latest::default());
        let w = Self {
            source,
            stop: Arc::new(AtomicBool::new(false)),
            live: Arc::new(AtomicBool::new(live)),
            record: Arc::new(Mutex::new(None)),
            shown: Arc::new(Mutex::new(None)),
            fps: Arc::new(Mutex::new(30.0)),
            frames: Arc::new(AtomicU64::new(0)),
            error: Arc::new(Mutex::new(None)),
            threads: Vec::new(),
        };
        let capture = {
            let (source, stop, record, fps, frames, error, latest) = (
                w.source.clone(),
                w.stop.clone(),
                w.record.clone(),
                w.fps.clone(),
                w.frames.clone(),
                w.error.clone(),
                latest.clone(),
            );
            std::thread::spawn(move || {
                let posted = |seq: u64, frame: Bgr| {
                    *latest.frame.lock().unwrap() = Some((seq, frame));
                    latest.ready.notify_all();
                };
                if let Err(err) = capture_loop(&source, &stop, &record, &fps, &frames, posted) {
                    *error.lock().unwrap() = Some(format!("{err:#}"));
                }
                stop.store(true, Ordering::Relaxed);
                latest.ready.notify_all();
            })
        };
        let analysis = {
            let (stop, live_flag, shown, error) = (
                w.stop.clone(),
                w.live.clone(),
                w.shown.clone(),
                w.error.clone(),
            );
            std::thread::spawn(move || {
                let pipe = match Config::load(&config, &[]).and_then(|c| Pipeline::from_config(&c))
                {
                    Ok(p) => Some(p),
                    Err(err) => {
                        *error.lock().unwrap() = Some(format!("cannot load {config}: {err:#}"));
                        None
                    }
                };
                analysis_loop(&latest, &stop, &live_flag, pipe, &shown);
            })
        };
        Self {
            threads: vec![capture, analysis],
            ..w
        }
    }

    /// Begins writing every captured frame to `path`.
    pub fn start_recording(&self, path: &Path) {
        let fps = *self.fps.lock().unwrap();
        *self.record.lock().unwrap() = Some(VideoWriter::new(path, fps));
    }

    /// Closes the recording, if any; returns the frames written.
    pub fn stop_recording(&self) -> Result<usize> {
        match self.record.lock().unwrap().take() {
            Some(w) => w.finish(),
            None => Ok(0),
        }
    }

    pub fn recording(&self) -> bool {
        self.record.lock().unwrap().is_some()
    }

    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Stops both threads and closes any recording.
    pub fn stop(mut self) -> Result<usize> {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        self.stop_recording()
    }
}

/// Reads frames until `stop`, recording each one while a writer is set.
fn capture_loop(
    source: &Source,
    stop: &AtomicBool,
    record: &Mutex<Option<VideoWriter>>,
    fps: &Mutex<f64>,
    frames: &AtomicU64,
    post: impl Fn(u64, Bgr),
) -> Result<()> {
    let mut seq = 0u64;
    let mut emit = |frame: Bgr| -> Result<()> {
        if let Some(w) = record.lock().unwrap().as_mut() {
            w.write(&frame)?;
        }
        seq += 1;
        frames.store(seq, Ordering::Relaxed);
        post(seq, frame);
        Ok(())
    };
    match source {
        Source::Camera { index, .. } => {
            use nokhwa::pixel_format::RgbFormat;
            use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
            let format =
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
            let mut cam = nokhwa::Camera::new(CameraIndex::Index(*index), format)
                .context("cannot open the camera (is camera access allowed for this app?)")?;
            cam.open_stream()
                .context("cannot start the camera stream")?;
            *fps.lock().unwrap() = f64::from(cam.frame_rate().max(1));
            while !stop.load(Ordering::Relaxed) {
                let rgb = cam.frame()?.decode_image::<RgbFormat>()?;
                let (w, h) = rgb.dimensions();
                let mut px = rgb.into_raw();
                for p in px.chunks_exact_mut(3) {
                    p.swap(0, 2);
                }
                emit(Bgr::from_pixels(w, h, px)?)?;
            }
            let _ = cam.stop_stream();
        }
        Source::File(path) => {
            let rate = crate::data::video::probe(path)?.fps.unwrap_or(30.0);
            *fps.lock().unwrap() = rate;
            let period = Duration::from_secs_f64(1.0 / rate);
            let started = Instant::now();
            for (i, frame) in VideoReader::open(path, None)?.enumerate() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // play at the file's own rate, as a camera would deliver it
                if let Some(wait) = (period * i as u32).checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
                emit(frame?)?;
            }
        }
    }
    Ok(())
}

/// Tracks the newest frame whenever one arrives, and hands the drawn result to the window.
fn analysis_loop(
    latest: &Latest,
    stop: &AtomicBool,
    live: &AtomicBool,
    pipe: Option<Pipeline>,
    shown: &Mutex<Option<Shown>>,
) {
    let mut tracker = pipe.map(LiveTracker::new);
    let mut last = 0u64;
    loop {
        let frame = {
            let mut slot = latest.frame.lock().unwrap();
            while !stop.load(Ordering::Relaxed) && slot.as_ref().map_or(true, |f| f.0 == last) {
                slot = latest
                    .ready
                    .wait_timeout(slot, Duration::from_millis(100))
                    .unwrap()
                    .0;
            }
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let (seq, frame) = slot.as_ref().expect("checked above");
            last = *seq;
            frame.clone()
        };
        let t = Instant::now();
        let mut image = frame;
        let mut animals = Vec::new();
        if let (true, Some(tr)) = (live.load(Ordering::Relaxed), tracker.as_mut()) {
            if let Ok(found) = tr.step(&image) {
                for a in &found {
                    let flat: Vec<f64> = a.keypoints.iter().flat_map(|p| [p[0], p[1]]).collect();
                    if flat.len() == 2 * crate::data::synthetic::NUM_KEYPOINTS {
                        draw_pose(&mut image, &flat, Some(&a.conf), None, a.id, "", 0.3);
                    }
                }
                animals = found;
            }
        }
        *shown.lock().unwrap() = Some(Shown {
            image,
            animals,
            analysis_ms: t.elapsed().as_secs_f64() * 1e3,
        });
    }
}

/// Runs the offline pipeline on one recording: `predict` per view, or the 3D pipeline when several
/// views and a `cameras.json` are given. Writes under `<recording>/analysis`.
pub fn analyze(config: &str, views: &[PathBuf], cameras: Option<&str>) -> Result<PathBuf> {
    let dir = views
        .first()
        .and_then(|v| v.parent())
        .context("nothing was recorded")?
        .join("analysis");
    match (views.len(), cameras) {
        (n, Some(cams)) if n > 1 => {
            let views: Vec<String> = views.iter().map(|v| v.display().to_string()).collect();
            crate::multiview::pipeline::run_cli(
                config,
                &views,
                cams,
                &dir.display().to_string(),
                None,
                &[],
            )?;
        }
        _ => {
            for v in views {
                let stem = v
                    .file_stem()
                    .map_or_else(|| "view".into(), |s| s.to_string_lossy().into_owned());
                let out = if views.len() == 1 {
                    dir.clone()
                } else {
                    dir.join(stem)
                };
                crate::cli::predict::run_with(
                    config,
                    &v.display().to_string(),
                    &out.display().to_string(),
                    None,
                    None,
                    &[],
                    true,
                )?;
            }
        }
    }
    Ok(dir)
}

/// The window.
pub struct App {
    sources: Vec<Source>,
    selected: Vec<bool>,
    file: String,
    config: String,
    out_dir: String,
    cameras_json: String,
    live: bool,
    workers: Vec<Worker>,
    textures: Vec<Option<egui::TextureHandle>>,
    recording: Option<Vec<PathBuf>>,
    last_recording: Vec<PathBuf>,
    status: Arc<Mutex<String>>,
}

impl Default for App {
    fn default() -> Self {
        let sources = list_cameras();
        let recordings = std::env::home_dir()
            .map(|home| home.join("AnimalPose").join("recordings"))
            .unwrap_or_else(|| "recordings".into());
        Self {
            selected: vec![false; sources.len()],
            sources,
            file: String::new(),
            config: default_pipeline_config().to_string_lossy().into_owned(),
            out_dir: recordings.to_string_lossy().into_owned(),
            cameras_json: String::new(),
            live: true,
            workers: Vec::new(),
            textures: Vec::new(),
            recording: None,
            last_recording: Vec::new(),
            status: Arc::new(Mutex::new(if crate::tensor::gpu_available() {
                "GPU (Metal) available".into()
            } else {
                "running on the CPU".into()
            })),
        }
    }
}

impl App {
    fn set_status(&self, s: impl Into<String>) {
        *self.status.lock().unwrap() = s.into();
    }

    fn start(&mut self) {
        if let Err(err) = Config::load(&self.config, &[]) {
            self.set_status(format!("cannot load {}: {err:#}", self.config));
            return;
        }
        let mut chosen: Vec<Source> = self
            .sources
            .iter()
            .zip(&self.selected)
            .filter(|(_, on)| **on)
            .map(|(s, _)| s.clone())
            .collect();
        if !self.file.trim().is_empty() {
            chosen.push(Source::File(PathBuf::from(self.file.trim())));
        }
        if chosen.is_empty() {
            self.set_status("select a camera (or give a video file) first");
            return;
        }
        self.workers = chosen
            .into_iter()
            .map(|s| Worker::start(s, self.config.clone(), self.live))
            .collect();
        self.textures = (0..self.workers.len()).map(|_| None).collect();
        self.set_status(format!("running {} source(s)", self.workers.len()));
    }

    fn stop(&mut self) {
        self.stop_recording();
        for w in self.workers.drain(..) {
            let _ = w.stop();
        }
        self.textures.clear();
        self.set_status("stopped");
    }

    fn start_recording(&mut self) {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let dir = PathBuf::from(&self.out_dir).join(format!("rec-{stamp}"));
        let paths: Vec<PathBuf> = (0..self.workers.len())
            .map(|i| dir.join(format!("cam{i}.mp4")))
            .collect();
        for (w, p) in self.workers.iter().zip(&paths) {
            w.start_recording(p);
        }
        self.set_status(format!("recording to {}", dir.display()));
        self.recording = Some(paths);
    }

    fn stop_recording(&mut self) {
        if let Some(paths) = self.recording.take() {
            let mut msg = Vec::new();
            for w in &self.workers {
                match w.stop_recording() {
                    Ok(n) => msg.push(format!("{}: {n} frames", w.source.name())),
                    Err(err) => msg.push(format!("{}: {err:#}", w.source.name())),
                }
            }
            self.set_status(format!("recorded {}", msg.join(", ")));
            self.last_recording = paths;
        }
    }

    fn analyze(&self) {
        let (config, views, status) = (
            self.config.clone(),
            self.last_recording.clone(),
            self.status.clone(),
        );
        let cams = Some(self.cameras_json.trim().to_string()).filter(|c| !c.is_empty());
        *status.lock().unwrap() = format!("analyzing {} view(s)…", views.len());
        std::thread::spawn(move || {
            let msg = match analyze(&config, &views, cams.as_deref()) {
                Ok(dir) => format!("analysis written to {}", dir.display()),
                Err(err) => format!("analysis failed: {err:#}"),
            };
            *status.lock().unwrap() = msg;
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let running = !self.workers.is_empty();
        egui::SidePanel::left("controls")
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.heading("animalpose");
                ui.add_enabled_ui(!running, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Cameras");
                        if ui.small_button("refresh").clicked() {
                            self.sources = list_cameras();
                            self.selected = vec![false; self.sources.len()];
                        }
                    });
                    for (s, on) in self.sources.iter().zip(self.selected.iter_mut()) {
                        ui.checkbox(on, s.name());
                    }
                    ui.label("or a video file");
                    ui.text_edit_singleline(&mut self.file);
                    ui.label("Pipeline config");
                    ui.text_edit_singleline(&mut self.config);
                    ui.label("Recordings folder");
                    ui.text_edit_singleline(&mut self.out_dir);
                });
                ui.label("cameras.json (multi-view 3D)");
                ui.text_edit_singleline(&mut self.cameras_json);
                if ui
                    .checkbox(&mut self.live, "live tracking overlay")
                    .changed()
                {
                    for w in &self.workers {
                        w.live.store(self.live, Ordering::Relaxed);
                    }
                }
                ui.separator();
                if !running {
                    if ui.button("▶ Start").clicked() {
                        self.start();
                    }
                } else {
                    if ui.button("■ Stop").clicked() {
                        self.stop();
                    }
                    if self.recording.is_none() {
                        if ui.button("● Record").clicked() {
                            self.start_recording();
                        }
                    } else if ui.button("■ Stop recording").clicked() {
                        self.stop_recording();
                    }
                }
                let can_analyze = !self.last_recording.is_empty() && self.recording.is_none();
                if ui
                    .add_enabled(can_analyze, egui::Button::new("Analyze last recording"))
                    .clicked()
                {
                    self.analyze();
                }
                ui.separator();
                ui.label(self.status.lock().unwrap().clone());
                for w in &self.workers {
                    if let Some(err) = w.error.lock().unwrap().as_ref() {
                        ui.colored_label(egui::Color32::RED, format!("{}: {err}", w.source.name()));
                    }
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.workers.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Select one or more cameras and press Start.")
                });
                return;
            }
            let n = self.workers.len();
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = n.div_ceil(cols);
            let cell = egui::vec2(
                ui.available_width() / cols as f32 - 8.0,
                ui.available_height() / rows as f32 - 28.0,
            );
            egui::Grid::new("views").show(ui, |ui| {
                for (i, w) in self.workers.iter().enumerate() {
                    ui.vertical(|ui| {
                        let shown = w.shown.lock().unwrap();
                        let caption = match shown.as_ref() {
                            Some(s) => format!(
                                "{}  ·  {} animal(s)  ·  {:.0} ms{}",
                                w.source.name(),
                                s.animals.len(),
                                s.analysis_ms,
                                if w.recording() { "  ·  ● REC" } else { "" }
                            ),
                            None => format!("{}  ·  starting…", w.source.name()),
                        };
                        ui.label(caption);
                        if let Some(s) = shown.as_ref() {
                            let rgb: Vec<u8> = s
                                .image
                                .pixels
                                .chunks_exact(3)
                                .flat_map(|p| [p[2], p[1], p[0]])
                                .collect();
                            let image = egui::ColorImage::from_rgb(
                                [s.image.width as usize, s.image.height as usize],
                                &rgb,
                            );
                            let tex = self.textures[i].get_or_insert_with(|| {
                                ctx.load_texture(
                                    format!("view{i}"),
                                    image.clone(),
                                    egui::TextureOptions::LINEAR,
                                )
                            });
                            tex.set(image, egui::TextureOptions::LINEAR);
                            let aspect = s.image.height as f32 / s.image.width.max(1) as f32;
                            let size = if cell.x * aspect <= cell.y {
                                egui::vec2(cell.x, cell.x * aspect)
                            } else {
                                egui::vec2(cell.y / aspect, cell.y)
                            };
                            ui.image((tex.id(), size));
                        }
                    });
                    if (i + 1) % cols == 0 {
                        ui.end_row();
                    }
                }
            });
        });
        ctx.request_repaint_after(Duration::from_millis(15));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop();
    }
}

/// Opens the window.
pub fn run() -> Result<()> {
    // macOS asks the user for camera access once; the answer arrives asynchronously.
    #[cfg(target_os = "macos")]
    nokhwa::nokhwa_initialize(|_| {});
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("animalpose")
            .with_inner_size([1200.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native(
        "animalpose",
        options,
        Box::new(|_| Ok(Box::<App>::default())),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rendered two-mouse clip as an `.mp4`, standing in for a camera.
    fn demo_video(dir: &Path, frames: usize) -> PathBuf {
        let png = dir.join("png");
        crate::data::render::write_demo_frames(&png, frames).unwrap();
        let src = crate::data::frames::Source::open(&png).unwrap();
        let path = dir.join("camera.mp4");
        let mut w = VideoWriter::new(&path, 30.0);
        for f in src.iter(None).unwrap() {
            w.write(&f.unwrap()).unwrap();
        }
        w.finish().unwrap();
        path
    }

    /// Live tracking on a played-back clip, a recording in the middle of it, then the offline
    /// analysis of that recording: the whole app without the window.
    #[test]
    fn live_record_and_analyze() {
        let Some(root) = crate::data::repo_root() else {
            return;
        };
        if !crate::data::video::ffmpeg_available() {
            eprintln!("SKIP: ffmpeg is not installed");
            return;
        }
        let pt = root.join("weights/pose_general.pt");
        if std::fs::read(&pt).map_or(true, |b| crate::models::torch_pt::is_lfs_pointer(&b)) {
            eprintln!("SKIP: weights/*_general.pt are absent");
            return;
        }
        let dir = std::env::temp_dir().join(format!("animalpose-gui-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let video = demo_video(&dir, 90);
        let config = root.join("configs/pipeline.yaml").display().to_string();

        let worker = Worker::start(Source::File(video), config.clone(), true);
        let wait = |cond: &dyn Fn() -> bool| {
            let t = Instant::now();
            while !cond() && t.elapsed() < Duration::from_secs(60) {
                std::thread::sleep(Duration::from_millis(20));
            }
        };
        wait(&|| worker.frames.load(Ordering::Relaxed) >= 15);
        let rec = dir.join("rec").join("cam0.mp4");
        worker.start_recording(&rec);
        wait(&|| worker.stopped());
        assert!(
            worker.error.lock().unwrap().is_none(),
            "{:?}",
            worker.error.lock().unwrap()
        );
        let (animals, ms) = {
            let shown = worker.shown.lock().unwrap();
            let s = shown.as_ref().expect("a frame was shown");
            (s.animals.len(), s.analysis_ms)
        };
        let written = worker.stop().unwrap();
        assert!(written >= 30, "recorded {written} frames");
        assert!(animals >= 1, "the live overlay found no animal");
        eprintln!("live: {animals} animal(s), {ms:.1} ms per frame; recorded {written} frames");

        let out = analyze(&config, &[rec], None).unwrap();
        for f in ["tracks.csv", "tracks.npz", "behavior.csv", "overlay.mp4"] {
            assert!(out.join(f).is_file(), "{f} is missing");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
