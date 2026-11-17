//! The engine's file handling end to end: recording, offline analysis, batches and output folders.

use std::path::PathBuf;
use std::sync::Arc;

use super::capture::{CaptureControl, Source, Worker};
use super::engine::{collect_videos, finish_capture, output_dir, process_batch, process_views};
use super::library::{Library, PipelineFiles, Settings};
use super::log::Log;
use super::models::ModelCache;
use super::processing::{self, Job as ProcessingJob};
use super::recording::Session;
use crate::algos::image::Bgr;
use crate::data::video::{ffmpeg_available, ffprobe_available, mp4_frames, VideoWriter};

#[test]
fn preview_and_offline_processing_leave_every_recorded_frame_intact() {
    if !ffmpeg_available() || !ffprobe_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let fixtures = crate::data::fixtures_dir().unwrap().join("pipeline");
    let video = dir.path().join("selected.mp4");
    let frames = crate::data::frames::Source::open(fixtures.join("frames")).unwrap();
    let mut writer = VideoWriter::new(&video, 30.0);
    for frame in frames.iter(None).unwrap() {
        writer.write(&frame.unwrap()).unwrap();
    }
    let count = writer.finish().unwrap();
    assert!(count > 0);

    let session = Session::start(
        &dir.path().join("target"),
        "",
        Vec::new(),
        &["test source".into()],
        &["View 1".into()],
    )
    .unwrap();
    let control = Arc::new(CaptureControl::default());
    *control.session.lock().unwrap() = Some(session.clone());
    let missing = PipelineFiles {
        config: dir.path().join("missing.yaml"),
        overrides: Vec::new(),
    };
    let worker = Worker::start(
        Source::File(video),
        Some(missing),
        Arc::new(ModelCache::default()),
        true,
        0,
        control.clone(),
    );
    let latest = worker.latest.clone();
    let preview_error = worker.preview_error.clone();
    let summary = finish_capture(vec![worker], Some(session.clone()), false);
    assert!(
        latest.frame.lock().unwrap().is_some(),
        "raw preview remained available"
    );
    assert!(
        preview_error.lock().unwrap().is_none(),
        "preview tried to load a model while recording"
    );
    assert!(
        summary.status.starts_with("Recording finished"),
        "{}",
        summary.status
    );
    let clips = summary.clips.values().next().unwrap();
    assert_eq!(mp4_frames(&clips[0]).unwrap(), count);
    assert!(clips[0].starts_with(dir.path().join("target")));
    assert!(session.dir.join("recording.json").is_file());

    let out = dir.path().join("processed");
    let library = Library::open(dir.path().join("library"));
    let config_dir = library.root.join("configs");
    std::fs::create_dir_all(&config_dir).unwrap();
    let base = fixtures.join("pipeline.yaml").canonicalize().unwrap();
    let missing = dir.path().join("missing-behavior.pt");
    // An older library config can retain a checkpoint path left over from the source machine.
    std::fs::write(
        config_dir.join("legacy.yaml"),
        format!(
            "extends: {}\ncheckpoints:\n  behavior: {}\n",
            serde_json::to_string(&base).unwrap(),
            serde_json::to_string(&missing).unwrap()
        ),
    )
    .unwrap();
    let settings = Settings {
        config: Some("legacy.yaml".into()),
        ..Settings::default()
    };
    let files = library.pipeline(&settings).unwrap();
    process_views(
        &files,
        clips,
        None,
        &out,
        None,
        &ModelCache::default(),
        None,
        &dir.path().join("zones"),
    )
    .unwrap();
    assert!(out.join("tracks.csv").is_file());
    assert!(out.join("tracks.npz").is_file());
    assert_eq!(mp4_frames(&out.join("overlay.mp4")).unwrap(), count);
    assert!(!out.join("behavior.csv").exists());

    let canceled = dir.path().join("canceled");
    let err = processing::save_output(&canceled, "overlay-test", None, |partial| {
        use crate::cli::predict::{run_with_progress, Progress};

        run_with_progress(
            &files.config.to_string_lossy(),
            &clips[0].to_string_lossy(),
            &partial.to_string_lossy(),
            None,
            None,
            &files.overrides,
            true,
            |progress| {
                if matches!(progress, Progress::Overlay { done: 1, .. }) {
                    return Err(processing::Stopped.into());
                }
                Ok(())
            },
        )
    })
    .unwrap_err();
    assert!(err.is::<processing::Stopped>());
    assert_eq!(std::fs::read_dir(canceled).unwrap().count(), 0);
    assert_eq!(mp4_frames(&clips[0]).unwrap(), count);
}

#[test]
fn folders_expand_to_their_videos_but_not_earlier_analyses() {
    let dir = tempfile::tempdir().unwrap();
    // Chosen paths come back canonical (`/var` is `/private/var` on macOS).
    let root = &dir.path().canonicalize().unwrap();
    for file in [
        "b.mp4",
        "a.MOV",
        "notes.txt",
        "day2/cam0.mp4",
        ".hidden/skip.mp4",
        "day2/analysis/overlay.mp4",
        "day2/analysis/tracks.csv",
    ] {
        let path = root.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }
    let (found, issues) = collect_videos([
        root.to_path_buf(),
        root.join("notes.txt"),
        root.join("missing.mp4"),
        root.join(".hidden"),
    ]);
    let found: Vec<_> = found
        .iter()
        .map(|p| p.strip_prefix(root).unwrap().to_path_buf())
        .collect();
    assert_eq!(
        found,
        [
            PathBuf::from("a.MOV"),
            PathBuf::from("b.mp4"),
            PathBuf::from("day2/cam0.mp4"),
            PathBuf::from(".hidden/skip.mp4"),
        ]
    );
    assert_eq!(issues.len(), 2, "{issues:?}");
}

/// Each video of a batch is its own sequence: the second one's tracks match processing it
/// alone, so neither the tracker nor the temporal model's windows carry over from the first.
/// The fixture clip is shorter than one temporal window, the start-of-video case.
#[test]
fn batch_videos_are_independent_and_failures_do_not_stop_the_batch() {
    if !ffmpeg_available() || !ffprobe_available() {
        return;
    }
    let Some(fixtures) = crate::data::fixtures_dir().map(|f| f.join("pipeline")) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let frames = crate::data::frames::Source::open(fixtures.join("frames")).unwrap();
    let frames: Vec<_> = frames.iter(None).unwrap().map(Result::unwrap).collect();
    let write = |name: &str, frames: &mut dyn Iterator<Item = &Bgr>| {
        let path = dir.path().join(name);
        let mut writer = VideoWriter::new(&path, 10.0);
        for frame in frames {
            writer.write(frame).unwrap();
        }
        writer.finish().unwrap();
        path
    };
    let first = write("first.mp4", &mut frames.iter());
    let second = write("second.mp4", &mut frames.iter().rev());
    let files = PipelineFiles {
        config: fixtures.join("pipeline.yaml"),
        overrides: Vec::new(),
    };

    let alone = dir.path().join("alone");
    process_views(
        &files,
        std::slice::from_ref(&second),
        None,
        &alone,
        None,
        &ModelCache::default(),
        None,
        &dir.path().join("zones"),
    )
    .unwrap();
    let models = Arc::new(ModelCache::default());
    let log = Arc::new(Log::new(dir.path().join("logs")));

    let target = dir.path().join("processed");
    let videos = vec![first, dir.path().join("missing.mp4"), second.clone()];
    let job = {
        let (files, videos, target) = (files.clone(), videos.clone(), target.clone());
        let (models, log) = (models.clone(), log.clone());
        ProcessingJob::start(videos.len(), move |control| {
            process_batch(&files, &videos, &target, control, &models, &log)
        })
    };
    let outcome = job.thread.join().unwrap().unwrap();
    let status = outcome.status;
    assert!(status.starts_with("Analyzed 2 of 3 videos"), "{status}");
    assert!(status.contains("missing.mp4"), "{status}");
    assert!(outcome.dir.unwrap().join("animalpose.log").is_file());
    let snapshot = job.control.snapshot();
    assert_eq!(snapshot.fraction(), 1.0);
    assert_eq!(snapshot.video_label(), "3 of 3 videos done");

    let batch = std::fs::read_dir(&target)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(batch.join("batch.json")).unwrap()).unwrap();
    assert_eq!(manifest["videos"].as_array().unwrap().len(), 3);
    assert!(manifest["videos"][1]["error"].is_string());
    for name in ["first", "second"] {
        assert!(batch.join(name).join("overlay.mp4").is_file());
    }
    assert_eq!(
        std::fs::read_to_string(batch.join("second/tracks.csv")).unwrap(),
        std::fs::read_to_string(alone.join("tracks.csv")).unwrap(),
    );

    // A single video is saved as `<name>-analysis` in the target, never over an earlier result.
    let single = dir.path().join("single");
    for expected in ["second-analysis", "second-analysis-2"] {
        let (files, video, single_target) = (files.clone(), second.clone(), single.clone());
        let (models, log) = (models.clone(), log.clone());
        let job = ProcessingJob::start(1, move |control| {
            process_batch(&files, &[video], &single_target, control, &models, &log)
        });
        let status = job.thread.join().unwrap().unwrap().status;
        assert!(
            single.join(expected).join("tracks.csv").is_file(),
            "{status}"
        );
    }
}

#[test]
fn results_go_next_to_the_first_video_unless_a_folder_is_chosen() {
    let videos = [
        PathBuf::from("/data/day1/cam0.mp4"),
        PathBuf::from("/data/day2/cam0.mp4"),
    ];
    let mut settings = Settings::default();
    assert_eq!(
        output_dir(&settings, &videos),
        Some(PathBuf::from("/data/day1"))
    );
    assert_eq!(output_dir(&settings, &[]), None);
    settings.output = Some(PathBuf::from("/results"));
    assert_eq!(
        output_dir(&settings, &videos),
        Some(PathBuf::from("/results"))
    );
    assert_eq!(output_dir(&settings, &[]), Some(PathBuf::from("/results")));
}

#[test]
fn profiles_hold_the_config_models_and_stages() {
    use super::library::Profile;
    use super::recording::Stage;
    use super::Engine;

    let dir = tempfile::tempdir().unwrap();
    let library = Library::open(dir.path().join("library"));
    let mut engine = Engine::with_library(library.clone(), Vec::new(), || {});
    let names = |e: &Engine| {
        e.profiles
            .iter()
            .map(|p| p.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&engine), ["Default"]);
    assert_eq!(engine.settings.profile.as_deref(), Some("Default"));

    let open_field = Profile {
        name: "Open Field".into(),
        classify_behavior: false,
        stages: vec![Stage {
            name: String::new(),
            seconds: 600,
            wait_after: false,
        }],
        ..Profile::default()
    };
    engine.save_profile(None, open_field).unwrap();
    assert_eq!(names(&engine), ["Default", "Open Field"]);
    assert_eq!(engine.settings.profile.as_deref(), Some("Open Field"));
    assert!(!engine.settings.classify_behavior);
    assert_eq!(engine.settings.stages[0].seconds, 600);

    // Changing a setting changes the profile in use.
    engine.settings.stages[0].seconds = 900;
    engine.save();
    let reopened = Engine::with_library(library.clone(), Vec::new(), || {});
    assert_eq!(reopened.settings.profile.as_deref(), Some("Open Field"));
    assert_eq!(reopened.settings.stages[0].seconds, 900);

    engine.select_profile("Default");
    assert!(engine.settings.classify_behavior);
    assert_eq!(engine.new_profile_name("Open Field"), "Open Field 2");
    // A copy holds everything but the name, and the profile in use stays.
    let active = engine.settings.profile.clone();
    assert_eq!(
        engine.duplicate_profile("Open Field").unwrap(),
        "Open Field Copy"
    );
    assert_eq!(
        engine.duplicate_profile("Open Field").unwrap(),
        "Open Field Copy 2"
    );
    let original = engine
        .profiles
        .iter()
        .find(|p| p.name == "Open Field")
        .unwrap()
        .clone();
    let copy = engine
        .profiles
        .iter()
        .find(|p| p.name == "Open Field Copy")
        .unwrap()
        .clone();
    assert_eq!(
        copy,
        Profile {
            name: "Open Field Copy".into(),
            ..original
        }
    );
    assert_eq!(engine.settings.profile, active);
    assert!(engine.duplicate_profile("Missing").is_err());
    engine.delete_profile("Open Field Copy").unwrap();
    engine.delete_profile("Open Field Copy 2").unwrap();

    let mut renamed = engine.profiles[1].clone();
    renamed.name = "Arena".into();
    engine.save_profile(Some("Open Field"), renamed).unwrap();
    assert_eq!(names(&engine), ["Arena", "Default"]);
    let mut clash = engine.profiles[0].clone();
    clash.name = "Default".into();
    assert!(engine.save_profile(Some("Arena"), clash).is_err());

    engine.delete_profile("Arena").unwrap();
    assert!(engine.delete_profile("Default").is_err());
}

/// Two batches with the same profile load the networks once; editing the config loads them again.
/// Each job's log is saved with its results, which are remembered.
#[test]
fn networks_are_loaded_once_and_jobs_leave_a_log() {
    use super::Engine;

    if !ffmpeg_available() || !ffprobe_available() {
        return;
    }
    let Some(fixtures) = crate::data::fixtures_dir().map(|f| f.join("pipeline")) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let frames = crate::data::frames::Source::open(fixtures.join("frames")).unwrap();
    let video = dir.path().join("clip.mp4");
    let mut writer = VideoWriter::new(&video, 10.0);
    for frame in frames.iter(None).unwrap() {
        writer.write(&frame.unwrap()).unwrap();
    }
    writer.finish().unwrap();

    let library = Library::open(dir.path().join("library"));
    std::fs::create_dir_all(library.root.join("configs")).unwrap();
    let base = fixtures.join("pipeline.yaml").canonicalize().unwrap();
    std::fs::write(
        library.root.join("configs/fixture.yaml"),
        format!("extends: {}\n", serde_json::to_string(&base).unwrap()),
    )
    .unwrap();
    let mut engine = Engine::with_library(library, Vec::new(), || {});
    engine.settings.config = Some("fixture.yaml".into());
    engine.save();
    engine.videos = vec![video];
    let run = |engine: &mut Engine| {
        engine.process_videos();
        let start = std::time::Instant::now();
        while engine.processing.is_some() {
            assert!(start.elapsed().as_secs() < 120, "processing timed out");
            std::thread::sleep(std::time::Duration::from_millis(20));
            engine.poll();
        }
        assert_eq!(engine.status, "Analysis finished.");
        engine.result.clone().unwrap()
    };
    let first = run(&mut engine);
    let second = run(&mut engine);
    assert_eq!(engine.models.loads(), 1);
    assert!(first.join("animalpose.log").is_file());
    let log = std::fs::read_to_string(second.join("animalpose.log")).unwrap();
    assert!(log.contains("tracks"), "{log}");
    assert_eq!(engine.settings.recent_results, [second.clone(), first]);
    assert!(engine.log.dir().read_dir().unwrap().count() >= 1);

    engine.edit_config(Some("fixture.yaml".into())).unwrap();
    engine
        .set_config_value("tracker.max_lost", &serde_json::json!(12))
        .unwrap();
    run(&mut engine);
    assert_eq!(engine.models.loads(), 2);
}

/// Multiview: a folder's videos are matched to the camera model by name and fused with the
/// networks the job already loaded; a folder that does not match is refused before anything runs.
#[test]
fn multiview_fuses_each_folder_with_the_camera_model() {
    use super::Engine;
    use crate::multiview::camera::{dumps_cameras, Camera};

    if !ffmpeg_available() || !ffprobe_available() {
        return;
    }
    let Some(fixtures) = crate::data::fixtures_dir().map(|f| f.join("pipeline")) else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let session = dir.path().join("session").join("02-Experiment");
    std::fs::create_dir_all(&session).unwrap();
    let frames = crate::data::frames::Source::open(fixtures.join("frames")).unwrap();
    let mut views = Vec::new();
    for name in ["Side.mp4", "Top.mp4"] {
        let video = session.join(name);
        let mut writer = VideoWriter::new(&video, 10.0);
        for frame in frames.iter(None).unwrap() {
            writer.write(&frame.unwrap()).unwrap();
        }
        writer.finish().unwrap();
        views.push(video);
    }
    let rig: Vec<Camera> = [("top", 90.0, -90.0), ("side", 8.0, -90.0)]
        .iter()
        .map(|(name, el, az)| {
            Camera::from(&crate::data::synthetic::camera_from_view(
                100.0,
                *el,
                *az,
                (96, 96),
                1.6,
                name,
                0.9,
            ))
        })
        .collect();
    let model = dir.path().join("arena.json");
    std::fs::write(&model, dumps_cameras(&rig)).unwrap();

    let library = Library::open(dir.path().join("library"));
    std::fs::create_dir_all(library.root.join("configs")).unwrap();
    let base = fixtures.join("pipeline.yaml").canonicalize().unwrap();
    std::fs::write(
        library.root.join("configs/fixture.yaml"),
        format!("extends: {}\n", serde_json::to_string(&base).unwrap()),
    )
    .unwrap();
    let mut engine = Engine::with_library(library, Vec::new(), || {});
    engine.settings.config = Some("fixture.yaml".into());
    engine.import_camera_model(&model, "Default");
    assert_eq!(engine.settings.camera_model.as_deref(), Some("arena.json"));
    assert_eq!(engine.camera_names(), ["top", "side"]);
    let profile = engine
        .profiles
        .iter()
        .find(|p| p.name == "Default")
        .unwrap();
    assert_eq!(profile.camera_model.as_deref(), Some("arena.json"));
    engine.set_multiview(true);
    engine.videos = views.clone();

    let groups = engine.view_groups().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].error, None);
    assert_eq!(
        groups[0].videos,
        [(0, Some("side".to_owned())), (1, Some("top".to_owned()))]
    );

    engine.process_videos();
    let start = std::time::Instant::now();
    while engine.processing.is_some() {
        assert!(start.elapsed().as_secs() < 300, "processing timed out");
        std::thread::sleep(std::time::Duration::from_millis(20));
        engine.poll();
    }
    assert_eq!(engine.status, "Analysis finished.");
    let out = engine.result.clone().unwrap();
    assert_eq!(out, session.join("02-Experiment-analysis"));
    assert!(out.join("animals3d.npz").is_file());
    assert!(out.join("view1_tracks.csv").is_file());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("views.json")).unwrap()).unwrap();
    assert_eq!(manifest[0]["view"], "top");
    assert!(manifest[0]["video"].as_str().unwrap().ends_with("Top.mp4"));
    assert_eq!(engine.models.loads(), 1);
    // The first session fused with the model becomes each camera's reference picture.
    let drift: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(out.join("camera_drift.json")).unwrap()).unwrap();
    assert_eq!(drift.len(), 2);
    let references = engine.library.root.join("cameras/arena.references");
    for (entry, name) in drift.iter().zip(["top", "side"]) {
        assert_eq!(entry["camera"], name);
        assert_eq!(entry["status"], "reference saved", "{entry}");
        assert!(references.join(format!("{name}.png")).is_file());
    }
    assert!(out.join("cameras.json").is_file());

    std::fs::rename(&views[1], session.join("Front.mp4")).unwrap();
    engine.videos = vec![views[0].clone(), session.join("Front.mp4")];
    assert!(engine.view_groups().unwrap()[0].error.is_some());
    engine.process_videos();
    assert!(engine.processing.is_none());
    assert!(engine.status.contains("top, side"), "{}", engine.status);
}

/// Each camera keeps the view it was given, also when two cameras share a name.
#[test]
fn cameras_keep_their_views() {
    use super::Engine;

    let dir = tempfile::tempdir().unwrap();
    let cameras = vec![
        Source::Camera {
            index: 0,
            name: "USB Camera".into(),
        },
        Source::Camera {
            index: 1,
            name: "USB Camera".into(),
        },
    ];
    let mut engine = Engine::with_library(Library::open(dir.path().to_path_buf()), cameras, || {});
    let sources = engine.sources.clone();
    assert_eq!(engine.views_of(sources.iter()), ["Camera 1", "Camera 2"]);
    // A camera keeps its ID-based default whichever cameras are chosen.
    assert_eq!(engine.views_of(sources[1..].iter()), ["Camera 2"]);
    engine.name_view(&sources[1], "Top");
    assert_eq!(engine.views_of(sources.iter()), ["Camera 1", "Top"]);
    assert_eq!(engine.views_of(sources[1..].iter()), ["Top"]);
    let profile = engine
        .profiles
        .iter()
        .find(|p| p.name == "Default")
        .unwrap();
    assert_eq!(
        profile.views.get("USB Camera (2)").map(String::as_str),
        Some("Top")
    );
    engine.name_view(&sources[1], " ");
    assert_eq!(engine.views_of(sources.iter()), ["Camera 1", "Camera 2"]);
}
