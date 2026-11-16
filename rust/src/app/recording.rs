//! Stage timing and MP4 encoding for the desktop recorder.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{mpsc::Receiver, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use crate::algos::image::Bgr;
use crate::data::video::{mp4_frames, VideoWriter};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Stage {
    pub(crate) name: String,
    pub(crate) seconds: u32,
    pub(crate) wait_after: bool,
}

pub(crate) fn default_stages() -> Vec<Stage> {
    vec![
        Stage {
            name: "Accustomization".into(),
            seconds: 60,
            wait_after: false,
        },
        Stage {
            name: "Experiment".into(),
            seconds: 300,
            wait_after: false,
        },
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    Manual,
    Timed(usize),
    Waiting(usize),
    Finished,
}

struct Clock {
    phase: Phase,
    deadline: Duration,
}

impl Clock {
    fn new(stages: &[Stage]) -> Self {
        match stages.first() {
            Some(first) => Self {
                phase: Phase::Timed(0),
                deadline: Duration::from_secs(first.seconds.into()),
            },
            None => Self {
                phase: Phase::Manual,
                deadline: Duration::ZERO,
            },
        }
    }

    fn advance(&mut self, stages: &[Stage], elapsed: Duration) {
        while let Phase::Timed(index) = self.phase {
            if elapsed < self.deadline {
                break;
            }
            if index + 1 == stages.len() {
                self.phase = Phase::Finished;
            } else if stages[index].wait_after {
                self.phase = Phase::Waiting(index);
            } else {
                self.phase = Phase::Timed(index + 1);
                self.deadline += Duration::from_secs(stages[index + 1].seconds.into());
            }
        }
    }

    fn resume(&mut self, stages: &[Stage], elapsed: Duration) {
        self.advance(stages, elapsed);
        if let Phase::Waiting(index) = self.phase {
            self.phase = Phase::Timed(index + 1);
            self.deadline = elapsed + Duration::from_secs(stages[index + 1].seconds.into());
        }
    }
}

pub(crate) struct Session {
    pub(crate) dir: PathBuf,
    pub(crate) stages: Vec<Stage>,
    started: Instant,
    clock: Mutex<Clock>,
}

impl Session {
    pub(crate) fn start(
        target: &Path,
        stages: Vec<Stage>,
        sources: &[String],
    ) -> Result<Arc<Self>> {
        ensure!(
            stages
                .iter()
                .all(|s| !s.name.trim().is_empty() && s.seconds > 0),
            "Every stage needs a name and a duration greater than zero."
        );
        let dir = new_output_dir(target, "recording")?;
        let clock = Clock::new(&stages);
        let plan = serde_json::json!({"stages": stages, "sources": sources});
        fs::write(dir.join("session.json"), serde_json::to_vec_pretty(&plan)?)?;
        Ok(Arc::new(Self {
            dir,
            stages,
            started: Instant::now(),
            clock: Mutex::new(clock),
        }))
    }

    // Decide the stage once for every frame, before it enters the encoder queue.
    pub(crate) fn position(&self) -> (Phase, Duration, Duration) {
        let mut clock = self.clock.lock().unwrap();
        let elapsed = self.started.elapsed();
        clock.advance(&self.stages, elapsed);
        (clock.phase, elapsed, clock.deadline.saturating_sub(elapsed))
    }

    pub(crate) fn resume(&self) {
        self.clock
            .lock()
            .unwrap()
            .resume(&self.stages, self.started.elapsed());
    }

    pub(crate) fn stop(&self) {
        self.clock.lock().unwrap().phase = Phase::Finished;
    }

    pub(crate) fn label(&self, phase: Phase) -> String {
        match phase {
            Phase::Manual => "Manual recording".to_string(),
            Phase::Timed(i) => self.stages[i].name.clone(),
            Phase::Waiting(i) => format!("Waiting after {}", self.stages[i].name),
            Phase::Finished => "Finished".to_string(),
        }
    }

    fn phase_dir(&self, phase: Phase) -> PathBuf {
        let folder = match phase {
            Phase::Manual => "manual".to_owned(),
            Phase::Timed(i) => format!("{:02}-{}", i + 1, safe_name(&self.stages[i].name)),
            Phase::Waiting(i) => {
                format!("{:02}-{}-waiting", i + 1, safe_name(&self.stages[i].name))
            }
            Phase::Finished => unreachable!("finished phases cannot receive frames"),
        };
        self.dir.join(folder)
    }

    pub(crate) fn missing_stages(&self, clips: &[Clip]) -> Vec<&str> {
        self.stages
            .iter()
            .enumerate()
            .filter_map(|(i, stage)| {
                let expected = self.phase_dir(Phase::Timed(i));
                (!clips
                    .iter()
                    .any(|clip| clip.path.parent() == Some(expected.as_path())))
                .then_some(stage.name.as_str())
            })
            .collect()
    }
}

pub(crate) fn safe_name(name: &str) -> String {
    let name: String = name
        .chars()
        .take(48)
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let name = name.trim_matches('-');
    if name.is_empty() {
        "untitled".into()
    } else {
        name.to_string()
    }
}

pub(crate) fn new_output_dir(target: &Path, prefix: &str) -> Result<PathBuf> {
    fs::create_dir_all(target).with_context(|| format!("cannot create {}", target.display()))?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    for suffix in 0.. {
        let dir = target.join(format!("{prefix}-{stamp}-{suffix}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(err).with_context(|| format!("cannot create {}", dir.display()))
            }
        }
    }
    unreachable!()
}

pub(crate) struct Frame {
    pub(crate) session: Arc<Session>,
    pub(crate) phase: Phase,
    pub(crate) elapsed: Duration,
    pub(crate) image: Arc<Bgr>,
    pub(crate) fps: f64,
}

#[derive(Serialize)]
pub(crate) struct Clip {
    pub(crate) path: PathBuf,
    pub(crate) frames: usize,
    stage: String,
    first_frame_seconds: f64,
    last_frame_seconds: f64,
    fps: f64,
}

#[derive(Default, Serialize)]
pub(crate) struct Report {
    pub(crate) clips: Vec<Clip>,
    pub(crate) error: Option<String>,
}

struct ClipWriter {
    phase: Phase,
    clip: Clip,
    partial: PathBuf,
    writer: VideoWriter,
}

impl ClipWriter {
    fn new(frame: &Frame, camera: usize) -> Self {
        let path = frame
            .session
            .phase_dir(frame.phase)
            .join(format!("camera-{:02}.mp4", camera + 1));
        let partial = path.with_extension("partial.mp4");
        Self {
            phase: frame.phase,
            writer: VideoWriter::for_recording(&partial, frame.fps),
            partial,
            clip: Clip {
                path,
                frames: 0,
                stage: frame.session.label(frame.phase),
                first_frame_seconds: frame.elapsed.as_secs_f64(),
                last_frame_seconds: frame.elapsed.as_secs_f64(),
                fps: frame.fps,
            },
        }
    }

    fn finish(self) -> Result<Clip> {
        let frame_count = self.writer.finish()?;
        let actual = mp4_frames(&self.partial)?;
        ensure!(
            actual == frame_count && frame_count == self.clip.frames,
            "{} contains {} frames, expected {}",
            self.partial.display(),
            actual,
            self.clip.frames
        );
        fs::rename(&self.partial, &self.clip.path)?;
        Ok(self.clip)
    }
}

/// The capture thread waits for this queue when encoding falls behind. The preview has a
/// separate latest-frame slot and can skip frames without affecting the queue.
pub(crate) fn encode(camera: usize, frames: Receiver<Frame>) -> Report {
    let mut report = Report::default();
    let result = (|| -> Result<()> {
        let mut open: Option<ClipWriter> = None;
        for frame in frames {
            if open.as_ref().is_some_and(|clip| clip.phase != frame.phase) {
                report.clips.push(open.take().unwrap().finish()?);
            }
            let clip = open.get_or_insert_with(|| ClipWriter::new(&frame, camera));
            if let Err(err) = clip.writer.write(&frame.image) {
                let clip = open.take().unwrap();
                let details = clip.writer.finish();
                return Err(err).context(format!(
                    "cannot record {}: {details:?}",
                    clip.partial.display()
                ));
            }
            clip.clip.frames += 1;
            clip.clip.last_frame_seconds = frame.elapsed.as_secs_f64();
        }
        if let Some(clip) = open {
            report.clips.push(clip.finish()?);
        }
        Ok(())
    })();
    if let Err(err) = result {
        report.error = Some(format!("{err:#}"));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn countdown_keeps_deadlines_and_records_the_wait_until_continue() {
        let mut stages = default_stages();
        stages[0].seconds = 2;
        stages[1].seconds = 3;
        stages[1].wait_after = true;
        stages.push(Stage {
            name: "Experiment 2".into(),
            seconds: 4,
            wait_after: false,
        });
        let mut clock = Clock::new(&stages);
        clock.advance(&stages, Duration::from_secs(3));
        assert_eq!(clock.phase, Phase::Timed(1));
        assert_eq!(clock.deadline, Duration::from_secs(5));
        clock.advance(&stages, Duration::from_secs(10));
        assert_eq!(clock.phase, Phase::Waiting(1));
        clock.resume(&stages, Duration::from_secs(10));
        assert_eq!(clock.phase, Phase::Timed(2));
        clock.advance(&stages, Duration::from_secs(13));
        assert_eq!(clock.phase, Phase::Timed(2));
        clock.advance(&stages, Duration::from_secs(14));
        assert_eq!(clock.phase, Phase::Finished);
    }

    #[test]
    fn encodes_every_queued_frame_in_its_own_stage() {
        if !crate::data::video::ffmpeg_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let session = Session::start(dir.path(), default_stages(), &["source".into()]).unwrap();
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        let encoder = std::thread::spawn(move || encode(0, receiver));
        for (phase, count) in [
            (Phase::Timed(0), 13),
            (Phase::Waiting(0), 4),
            (Phase::Timed(1), 17),
        ] {
            for i in 0..count {
                sender
                    .send(Frame {
                        session: session.clone(),
                        phase,
                        elapsed: Duration::from_millis(i),
                        image: Arc::new(Bgr::filled(64, 48, [(16 + i * 8) as u8; 3])),
                        fps: 25.0,
                    })
                    .unwrap();
            }
        }
        drop(sender);
        let report = encoder.join().unwrap();
        assert_eq!(report.error, None);
        assert_eq!(
            report
                .clips
                .iter()
                .map(|clip| clip.frames)
                .collect::<Vec<_>>(),
            [13, 4, 17]
        );
        assert!(report.clips[0]
            .path
            .ends_with("01-Accustomization/camera-01.mp4"));
        assert!(report.clips[1]
            .path
            .ends_with("01-Accustomization-waiting/camera-01.mp4"));
        assert!(report.clips[2]
            .path
            .ends_with("02-Experiment/camera-01.mp4"));
        for clip in report.clips {
            assert_eq!(
                crate::data::video::probe(&clip.path).unwrap().frames,
                clip.frames
            );
            let frames: Vec<_> = crate::data::video::VideoReader::open(&clip.path, None)
                .unwrap()
                .collect::<Result<_>>()
                .unwrap();
            assert_eq!(frames.len(), clip.frames);
            for (i, frame) in frames.iter().enumerate() {
                assert!(frame.pixel(32, 24)[1].abs_diff((16 + i * 8) as u8) <= 3);
            }
        }
    }
}
