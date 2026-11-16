//! The engine's file handling end to end: recording, offline analysis, batches and output folders.

use std::path::PathBuf;
use std::sync::Arc;

use super::capture::{CaptureControl, Source, Worker};
use super::engine::{collect_videos, finish_capture, output_dir, process_batch, process_views};
use super::library::{Library, PipelineFiles, Settings};
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
        Vec::new(),
        &["test source".into()],
    )
    .unwrap();
    let control = Arc::new(CaptureControl::default());
    *control.session.lock().unwrap() = Some(session.clone());
    let missing = PipelineFiles {
        config: dir.path().join("missing.yaml"),
        overrides: Vec::new(),
    };
    let worker = Worker::start(Source::File(video), Some(missing), true, 0, control.clone());
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
    process_views(&files, clips, None, &out, None).unwrap();
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
    process_views(&files, std::slice::from_ref(&second), None, &alone, None).unwrap();

    let target = dir.path().join("processed");
    let videos = vec![first, dir.path().join("missing.mp4"), second.clone()];
    let job = {
        let (files, videos, target) = (files.clone(), videos.clone(), target.clone());
        ProcessingJob::start(videos.len(), move |control| {
            process_batch(&files, &videos, &target, control)
        })
    };
    let status = job.thread.join().unwrap().unwrap();
    assert!(status.starts_with("Analyzed 2 of 3 videos"), "{status}");
    assert!(status.contains("missing.mp4"), "{status}");
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
        let job = ProcessingJob::start(1, move |control| {
            process_batch(&files, &[video], &single_target, control)
        });
        let status = job.thread.join().unwrap().unwrap();
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
