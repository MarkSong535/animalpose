//! The C interface of [`crate::app::Engine`] for the native macOS app (`macos/`).
//!
//! The app owns one engine handle and talks to it with JSON: [`ap_engine_state`] polls the engine
//! and describes everything the window shows, [`ap_engine_command`] performs one action and
//! returns the same description. Preview frames travel as RGBA through [`ap_engine_frame`]. All
//! calls on one handle come from the same (main) thread. Strings and pixel buffers returned here
//! are freed with [`ap_string_free`] and [`ap_frame_free`]. The header is `macos/Sources/CAnimalPose/
//! include/animalpose.h`.

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::app::library::MODEL_KINDS;
use crate::app::processing::Mode;
use crate::app::recording::{Phase, Stage};
use crate::app::{Engine, ImportKind};

/// An engine owned by the app.
pub struct ApEngine {
    engine: Engine,
}

/// One preview frame. `rgba` is null when the frame is the one the caller already has.
#[repr(C)]
pub struct ApFrame {
    pub sequence: u64,
    pub overlay: bool,
    pub width: u32,
    pub height: u32,
    pub rgba: *mut u8,
    pub len: usize,
    /// Animals found with the pose overlay, -1 without it.
    pub animals: i32,
    pub analysis_ms: f64,
}

/// What the app asks for, as `{"cmd": "...", ...}`.
#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    RefreshCameras,
    SelectCamera { index: usize, on: bool },
    SetPlayback { on: bool },
    SetLive { on: bool },
    StartPreview,
    StopPreview,
    Record { timed: bool },
    ContinueStage,
    StopRecording,
    AddVideos { paths: Vec<PathBuf> },
    RemoveVideo { index: usize },
    ClearVideos,
    Process,
    AnalyzeLast,
    PauseOrResume,
    StopProcessing,
    Import { kind: String, path: PathBuf },
    SetCalibration { path: Option<PathBuf> },
    SetOutput { path: Option<PathBuf> },
    SetRecordings { path: Option<PathBuf> },
    SetConfig { name: Option<String> },
    SetModel { kind: String, name: Option<String> },
    SetClassifyBehavior { on: bool },
    SetStages { stages: Vec<Stage> },
    ClearStatus,
}

impl ApEngine {
    fn run(&mut self, command: Command) -> anyhow::Result<()> {
        let engine = &mut self.engine;
        let idle = engine.idle();
        match command {
            Command::RefreshCameras => engine.refresh_cameras(),
            Command::SelectCamera { index, on } => {
                if let Some(selected) = engine.selected.get_mut(index).filter(|_| idle) {
                    *selected = on;
                }
            }
            Command::SetPlayback { on } => engine.playback = on,
            Command::SetLive { on } => engine.set_live(on),
            Command::StartPreview => engine.start_preview(),
            Command::StopPreview => engine.stop_sources(false),
            Command::Record { timed } => engine.start_recording(timed),
            Command::ContinueStage => engine.continue_stage(),
            Command::StopRecording => engine.stop_recording(),
            Command::AddVideos { paths } => engine.add_videos(paths),
            Command::RemoveVideo { index } => engine.remove_video(index),
            Command::ClearVideos => engine.videos.clear(),
            Command::Process => engine.process_videos(),
            Command::AnalyzeLast => engine.analyze_last(),
            Command::PauseOrResume => engine.pause_or_resume(),
            Command::StopProcessing => engine.stop_processing(),
            Command::Import { kind, path } => {
                let kind = match kind.as_str() {
                    "model" => ImportKind::Model,
                    "config" => ImportKind::Config,
                    "folder" => ImportKind::Folder,
                    other => anyhow::bail!("unknown import {other:?}"),
                };
                engine.import(kind, path);
            }
            Command::SetCalibration { path } => engine.cameras_json = path,
            Command::SetOutput { path } => {
                engine.settings.output = path;
                engine.save();
            }
            Command::SetRecordings { path } => {
                engine.settings.recordings = path;
                engine.save();
            }
            Command::SetConfig { name } => {
                engine.settings.config = name;
                engine.save();
            }
            Command::SetModel { kind, name } => {
                kind.parse::<crate::models::ModelKind>()?;
                match name {
                    Some(name) => engine.settings.models.insert(kind, name),
                    None => engine.settings.models.remove(&kind),
                };
                engine.save();
            }
            Command::SetClassifyBehavior { on } => {
                engine.settings.classify_behavior = on;
                engine.save();
            }
            Command::SetStages { stages } => {
                engine.settings.stages = stages;
                engine.save();
            }
            Command::ClearStatus => engine.status.clear(),
        }
        Ok(())
    }

    fn state(&mut self) -> Value {
        let engine = &mut self.engine;
        engine.poll();
        let models: serde_json::Map<String, Value> = MODEL_KINDS
            .iter()
            .map(|kind| {
                (
                    kind.as_str().to_owned(),
                    json!(engine.library.models(*kind)),
                )
            })
            .collect();
        let session = engine.session().map(|session| {
            let (phase, elapsed, remaining) = session.position();
            let stage_fraction = match phase {
                Phase::Timed(i) => {
                    let seconds = f64::from(session.stages[i].seconds);
                    Some(1.0 - remaining.as_secs_f64() / seconds)
                }
                _ => None,
            };
            json!({
                "label": session.label(phase),
                "timed": !session.stages.is_empty(),
                "waiting": matches!(phase, Phase::Waiting(_)),
                "stage_fraction": stage_fraction,
                "remaining": remaining.as_secs_f64(),
                "elapsed": elapsed.as_secs_f64(),
                "dir": session.dir,
            })
        });
        let processing = engine.processing.as_ref().map(|job| {
            let snapshot = job.control.snapshot();
            json!({
                "total": snapshot.total(),
                "fraction": snapshot.fraction(),
                "clip_fraction": snapshot.clip_fraction(),
                "video_label": snapshot.video_label(),
                "stage_label": snapshot.stage_label(),
                "label": snapshot.label(),
                "current": job.control.current(),
                "mode": match snapshot.mode {
                    Mode::Running => "running",
                    Mode::Pausing => "pausing",
                    Mode::Paused => "paused",
                    Mode::Stopping => "stopping",
                },
            })
        });
        let workers: Vec<Value> = engine
            .workers
            .iter()
            .map(|worker| {
                json!({
                    "name": worker.source.name(),
                    "ready": worker.ready.load(Ordering::Acquire),
                    "ended": worker.ended.load(Ordering::Acquire),
                    "capture_error": *worker.capture_error.lock().unwrap(),
                    "preview_error": *worker.preview_error.lock().unwrap(),
                })
            })
            .collect();
        json!({
            "status": engine.status,
            "library_root": engine.library.root,
            "recordings_dir": engine.recording_dir(),
            "output_dir": engine.output_dir(),
            "settings": engine.settings,
            "configs": engine.library.configs(),
            "models": models,
            "sources": engine.sources.iter().zip(&engine.selected).map(|(source, on)| {
                json!({"name": source.name(), "selected": on})
            }).collect::<Vec<_>>(),
            "videos": engine.videos,
            "playback": engine.playback,
            "calibration": engine.cameras_json,
            "workers": workers,
            "sources_ready": engine.sources_ready(),
            "session": session,
            "processing": processing,
            "idle": engine.idle(),
            "busy": engine.busy(),
            "importing": engine.importing(),
            "finishing": engine.finishing(),
            "can_analyze_last": !engine.last_recording.is_empty(),
            "ffmpeg": crate::data::video::ffmpeg_available(),
        })
    }
}

fn to_c(value: &Value) -> *mut c_char {
    CString::new(value.to_string())
        .unwrap_or_default()
        .into_raw()
}

/// Asks macOS for camera access once; call before listing cameras.
#[no_mangle]
pub extern "C" fn ap_camera_init() {
    #[cfg(target_os = "macos")]
    nokhwa::nokhwa_initialize(|_| {});
}

/// A new engine over `~/AnimalPose`. Free it with [`ap_engine_free`].
#[no_mangle]
pub extern "C" fn ap_engine_new() -> *mut ApEngine {
    catch_unwind(|| {
        Box::into_raw(Box::new(ApEngine {
            engine: Engine::new(|| {}),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Stops every job and frees the engine.
///
/// # Safety
/// `engine` is null or came from [`ap_engine_new`] and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn ap_engine_free(engine: *mut ApEngine) {
    if engine.is_null() {
        return;
    }
    let mut engine = Box::from_raw(engine);
    let _ = catch_unwind(AssertUnwindSafe(|| engine.engine.shutdown()));
}

/// Collects finished work and returns the state as JSON.
///
/// # Safety
/// `engine` came from [`ap_engine_new`].
#[no_mangle]
pub unsafe extern "C" fn ap_engine_state(engine: *mut ApEngine) -> *mut c_char {
    let Some(engine) = engine.as_mut() else {
        return std::ptr::null_mut();
    };
    let state = catch_unwind(AssertUnwindSafe(|| engine.state()))
        .unwrap_or_else(|_| json!({"error": "the engine stopped unexpectedly"}));
    to_c(&state)
}

/// Runs one JSON command and returns the state, or `{"error": ...}` for a malformed command.
///
/// # Safety
/// `engine` came from [`ap_engine_new`]; `command` is a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ap_engine_command(
    engine: *mut ApEngine,
    command: *const c_char,
) -> *mut c_char {
    let Some(engine) = engine.as_mut() else {
        return std::ptr::null_mut();
    };
    if command.is_null() {
        return to_c(&json!({"error": "no command"}));
    }
    let text = CStr::from_ptr(command).to_string_lossy();
    let result = catch_unwind(AssertUnwindSafe(|| -> anyhow::Result<Value> {
        let command: Command = serde_json::from_str(&text)?;
        engine.run(command)?;
        Ok(engine.state())
    }));
    let value = match result {
        Ok(Ok(state)) => state,
        Ok(Err(err)) => json!({"error": format!("{err:#}")}),
        Err(_) => json!({"error": "the engine stopped unexpectedly"}),
    };
    to_c(&value)
}

/// The newest frame of preview source `index`. Returns false when there is none yet. Pixels are
/// only returned when the frame differs from (`known_sequence`, `known_overlay`); pass sequence
/// 0 to always get them.
///
/// # Safety
/// `engine` came from [`ap_engine_new`]; `out` points to writable memory for one [`ApFrame`].
#[no_mangle]
pub unsafe extern "C" fn ap_engine_frame(
    engine: *mut ApEngine,
    index: u32,
    known_sequence: u64,
    known_overlay: bool,
    out: *mut ApFrame,
) -> bool {
    let (Some(engine), Some(out)) = (engine.as_mut(), out.as_mut()) else {
        return false;
    };
    let known = (known_sequence > 0).then_some((known_sequence, known_overlay));
    let preview = catch_unwind(AssertUnwindSafe(|| {
        engine.engine.preview(index as usize, known)
    }));
    let Ok(Some(preview)) = preview else {
        return false;
    };
    let (rgba, len) = match preview.rgba {
        Some(pixels) => {
            let mut pixels = pixels.into_boxed_slice();
            let len = pixels.len();
            let ptr = pixels.as_mut_ptr();
            std::mem::forget(pixels);
            (ptr, len)
        }
        None => (std::ptr::null_mut(), 0),
    };
    let (animals, analysis_ms) = preview.analysis.map_or((-1, 0.0), |(n, ms)| (n as i32, ms));
    *out = ApFrame {
        sequence: preview.key.0,
        overlay: preview.key.1,
        width: preview.size[0] as u32,
        height: preview.size[1] as u32,
        rgba,
        len,
        animals,
        analysis_ms,
    };
    true
}

/// Frees the pixels of an [`ApFrame`].
///
/// # Safety
/// `rgba` and `len` came from one [`ap_engine_frame`] call and are freed once.
#[no_mangle]
pub unsafe extern "C" fn ap_frame_free(rgba: *mut u8, len: usize) {
    if !rgba.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(rgba, len)));
    }
}

/// Frees a string returned by this library.
///
/// # Safety
/// `text` is null or came from this library and is freed once.
#[no_mangle]
pub unsafe extern "C" fn ap_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_round_trip_through_json() {
        let dir = tempfile::tempdir().unwrap();
        let library = crate::app::library::Library::open(dir.path().join("library"));
        let mut handle = ApEngine {
            engine: Engine::with_library(library, Vec::new(), || {}),
        };
        let videos = dir.path().join("videos");
        std::fs::create_dir_all(&videos).unwrap();
        std::fs::write(videos.join("a.mp4"), b"").unwrap();
        let command = json!({"cmd": "add_videos", "paths": [videos]}).to_string();
        let command = CString::new(command).unwrap();
        let state = unsafe {
            let text = ap_engine_command(&mut handle, command.as_ptr());
            let value: Value =
                serde_json::from_str(CStr::from_ptr(text).to_str().unwrap()).unwrap();
            ap_string_free(text);
            value
        };
        assert_eq!(state["videos"].as_array().unwrap().len(), 1);
        assert_eq!(
            state["output_dir"].as_str().unwrap(),
            videos.canonicalize().unwrap().to_str().unwrap()
        );
        assert_eq!(state["settings"]["output"], Value::Null);
        handle
            .run(Command::SetOutput {
                path: Some(dir.path().join("out")),
            })
            .unwrap();
        assert_eq!(
            handle.state()["output_dir"].as_str().unwrap(),
            dir.path().join("out").to_str().unwrap()
        );
        let bad = CString::new(r#"{"cmd": "fly"}"#).unwrap();
        let reply = unsafe {
            let text = ap_engine_command(&mut handle, bad.as_ptr());
            let value: Value =
                serde_json::from_str(CStr::from_ptr(text).to_str().unwrap()).unwrap();
            ap_string_free(text);
            value
        };
        assert!(reply["error"].is_string());
        let mut frame = std::mem::MaybeUninit::<ApFrame>::uninit();
        assert!(!unsafe { ap_engine_frame(&mut handle, 0, 0, false, frame.as_mut_ptr()) });
    }
}
