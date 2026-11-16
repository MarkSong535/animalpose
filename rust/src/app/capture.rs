//! Camera and file capture, with an independent preview and MP4 encoder.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::library::PipelineFiles;
use super::recording::{self, Phase, Session};
use crate::algos::image::Bgr;
use crate::cli::overlay::draw_pose;
use crate::cli::predict::{LiveTracker, Pipeline};
use crate::data::config::Config;
use crate::data::video::VideoReader;

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

#[derive(Default)]
pub(crate) struct CaptureControl {
    pub(crate) stop: AtomicBool,
    pub(crate) session: Mutex<Option<Arc<Session>>>,
}

pub(crate) struct Captured {
    pub(crate) sequence: u64,
    pub(crate) image: Arc<Bgr>,
}

#[derive(Default)]
pub(crate) struct Latest {
    pub(crate) frame: Mutex<Option<Captured>>,
    pub(crate) ready: Condvar,
}

pub(crate) struct Shown {
    pub(crate) sequence: u64,
    pub(crate) image: Bgr,
    pub(crate) animals: usize,
    pub(crate) analysis_ms: f64,
}

pub(crate) struct Worker {
    pub(crate) source: Source,
    pub(crate) live: Arc<AtomicBool>,
    pub(crate) latest: Arc<Latest>,
    pub(crate) shown: Arc<Mutex<Option<Shown>>>,
    pub(crate) ready: Arc<AtomicBool>,
    pub(crate) ended: Arc<AtomicBool>,
    pub(crate) capture_error: Arc<Mutex<Option<String>>>,
    pub(crate) preview_error: Arc<Mutex<Option<String>>>,
    capture: JoinHandle<recording::Report>,
    analysis: JoinHandle<()>,
}

impl Worker {
    pub(crate) fn start(
        source: Source,
        files: Option<PipelineFiles>,
        live: bool,
        index: usize,
        control: Arc<CaptureControl>,
    ) -> Self {
        let latest = Arc::new(Latest::default());
        let shown = Arc::new(Mutex::new(None));
        let ready = Arc::new(AtomicBool::new(false));
        let ended = Arc::new(AtomicBool::new(false));
        let capture_error = Arc::new(Mutex::new(None));
        let preview_error = Arc::new(Mutex::new(None));
        let live = Arc::new(AtomicBool::new(live));
        let capture = {
            let (source, latest, ready, ended, error, control) = (
                source.clone(),
                latest.clone(),
                ready.clone(),
                ended.clone(),
                capture_error.clone(),
                control.clone(),
            );
            std::thread::spawn(move || {
                let (queue, frames) = std::sync::mpsc::sync_channel(8);
                let encoder = std::thread::spawn(move || recording::encode(index, frames));
                let (mut sequence, mut last_preview) = (0, None);
                let result = capture_loop(&source, &control.stop, |fps, frame| {
                    sequence += 1;
                    let image = Arc::new(frame);
                    let session = control.session.lock().unwrap().clone();
                    if let Some(session) = session {
                        let (phase, elapsed, _) = session.position();
                        if phase != Phase::Finished {
                            queue
                                .send(recording::Frame {
                                    session,
                                    phase,
                                    elapsed,
                                    image: image.clone(),
                                    fps,
                                })
                                .context("the MP4 encoder stopped accepting frames")?;
                        }
                    }
                    let now = Instant::now();
                    // A slow display has no effect on the encoder queue.
                    if last_preview
                        .map_or(true, |t| now.duration_since(t) >= Duration::from_millis(33))
                    {
                        *latest.frame.lock().unwrap() = Some(Captured { sequence, image });
                        last_preview = Some(now);
                        latest.ready.notify_all();
                    }
                    ready.store(true, Ordering::Release);
                    Ok(())
                });
                if let Err(err) = result {
                    *error.lock().unwrap() = Some(format!("{err:#}"));
                }
                ended.store(true, Ordering::Release);
                latest.ready.notify_all();
                drop(queue);
                encoder.join().unwrap_or_else(|_| recording::Report {
                    clips: Vec::new(),
                    error: Some("the recording encoder panicked".to_string()),
                })
            })
        };
        let analysis = {
            let (latest, ended, live, shown, error) = (
                latest.clone(),
                ended.clone(),
                live.clone(),
                shown.clone(),
                preview_error.clone(),
            );
            std::thread::spawn(move || {
                analysis_loop(&latest, &ended, &live, &control, files, &shown, &error)
            })
        };
        Self {
            source,
            live,
            latest,
            shown,
            ready,
            ended,
            capture_error,
            preview_error,
            capture,
            analysis,
        }
    }

    pub(crate) fn finish(self) -> recording::Report {
        let _ = self.analysis.join();
        let mut report = self.capture.join().unwrap_or_else(|_| recording::Report {
            clips: Vec::new(),
            error: Some("the capture thread panicked".to_string()),
        });
        if let Some(error) = self.capture_error.lock().unwrap().take() {
            report.error = Some(match report.error {
                Some(details) => format!("{error}. {details}"),
                None => error,
            });
        }
        report
    }
}

fn capture_loop(
    source: &Source,
    stop: &AtomicBool,
    mut emit: impl FnMut(f64, Bgr) -> Result<()>,
) -> Result<()> {
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
            let fps = f64::from(cam.frame_rate().max(1));
            let result = (|| -> Result<()> {
                while !stop.load(Ordering::Relaxed) {
                    let rgb = cam.frame()?.decode_image::<RgbFormat>()?;
                    let (w, h) = rgb.dimensions();
                    let mut px = rgb.into_raw();
                    for p in px.chunks_exact_mut(3) {
                        p.swap(0, 2);
                    }
                    emit(fps, Bgr::from_pixels(w, h, px)?)?;
                }
                Ok(())
            })();
            let _ = cam.stop_stream();
            result?;
        }
        Source::File(path) => {
            let fps = crate::data::video::probe(path)?.fps.unwrap_or(30.0);
            let started = Instant::now();
            for (i, frame) in VideoReader::open(path, None)?.enumerate() {
                let deadline = Duration::from_secs_f64(i as f64 / fps);
                while !stop.load(Ordering::Relaxed) {
                    let Some(wait) = deadline.checked_sub(started.elapsed()) else {
                        break;
                    };
                    std::thread::sleep(wait.min(Duration::from_millis(20)));
                }
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                emit(fps, frame?)?;
            }
        }
    }
    Ok(())
}

fn analysis_loop(
    latest: &Latest,
    ended: &AtomicBool,
    live: &AtomicBool,
    control: &CaptureControl,
    files: Option<PipelineFiles>,
    shown: &Mutex<Option<Shown>>,
    error: &Mutex<Option<String>>,
) {
    let Some(files) = files else { return };
    let mut tracker = None;
    let mut last = 0;
    while !control.stop.load(Ordering::Relaxed) && !ended.load(Ordering::Acquire) {
        let next = {
            let (slot, _) = latest
                .ready
                .wait_timeout_while(
                    latest.frame.lock().unwrap(),
                    Duration::from_millis(100),
                    |slot| slot.as_ref().map_or(true, |f| f.sequence == last),
                )
                .unwrap();
            slot.as_ref()
                .filter(|f| f.sequence != last)
                .map(|f| (f.sequence, f.image.clone()))
        };
        let Some((sequence, frame)) = next else {
            continue;
        };
        last = sequence;
        if !live.load(Ordering::Relaxed) || control.session.lock().unwrap().is_some() {
            continue;
        }
        let t = Instant::now();
        if tracker.is_none() {
            match Config::load(&files.config, &files.overrides)
                .and_then(|c| Pipeline::from_config(&c))
            {
                Ok(pipeline) => tracker = Some(LiveTracker::new(pipeline)),
                Err(err) => {
                    *error.lock().unwrap() = Some(format!("Live pose preview: {err:#}"));
                    return;
                }
            }
        }
        if control.session.lock().unwrap().is_some() {
            continue;
        }
        let mut image = (*frame).clone();
        let animals = match tracker.as_mut().unwrap().step(&image) {
            Ok(animals) => animals,
            Err(err) => {
                *error.lock().unwrap() = Some(format!("Live pose preview: {err:#}"));
                return;
            }
        };
        for a in &animals {
            let flat: Vec<f64> = a.keypoints.iter().flat_map(|p| [p[0], p[1]]).collect();
            if flat.len() == 2 * crate::data::synthetic::NUM_KEYPOINTS {
                draw_pose(&mut image, &flat, Some(&a.conf), None, a.id, "", 0.3);
            }
        }
        *shown.lock().unwrap() = Some(Shown {
            sequence,
            image,
            animals: animals.len(),
            analysis_ms: t.elapsed().as_secs_f64() * 1e3,
        });
    }
}
