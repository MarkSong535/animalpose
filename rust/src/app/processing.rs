use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use thiserror::Error;

use crate::cli::predict::Progress;

#[derive(Debug, Error)]
#[error("Video processing stopped.")]
pub(crate) struct Stopped;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Running,
    Pausing,
    Paused,
    Stopping,
}

#[derive(Clone, Copy)]
pub(crate) struct Snapshot {
    pub(crate) mode: Mode,
    progress: Progress,
    completed: usize,
    total: usize,
}

impl Snapshot {
    /// Progress through the current video, 0 to 1.
    pub(crate) fn clip_fraction(self) -> f32 {
        let fraction = |done: usize, total: usize| {
            if total == 0 {
                0.0
            } else {
                (done as f32 / total as f32).min(1.0)
            }
        };
        if self.completed >= self.total {
            return 1.0;
        }
        match self.progress {
            Progress::Preparing => 0.0,
            Progress::Tracking { done, total } => 0.05 + 0.75 * fraction(done, total),
            Progress::Refining => 0.80,
            Progress::WritingTracks => 0.85,
            Progress::Overlay { done, total } => 0.88 + 0.10 * fraction(done, total),
            Progress::FinishingOverlay => 0.98,
        }
    }

    /// Progress through every video of the job, 0 to 1.
    pub(crate) fn fraction(self) -> f32 {
        let clip = if self.completed >= self.total {
            0.0
        } else {
            self.clip_fraction()
        };
        ((self.completed as f32 + clip) / self.total.max(1) as f32).min(1.0)
    }

    /// The number of videos in this job.
    pub(crate) fn total(self) -> usize {
        self.total
    }

    /// `Video 3 of 10`, counting the one in progress.
    pub(crate) fn video_label(self) -> String {
        if self.completed >= self.total {
            return format!("{} of {} videos done", self.total, self.total);
        }
        format!("Video {} of {}", self.completed + 1, self.total)
    }

    /// What the current video is doing, with its frame count where there is one.
    pub(crate) fn stage_label(self) -> String {
        if self.completed >= self.total {
            return "Saving results…".into();
        }
        let frames = |stage, done, total| {
            if total == 0 || done > total {
                format!("{stage} · {done} frames")
            } else {
                format!("{stage} · {done} / {total} frames")
            }
        };
        match self.progress {
            Progress::Preparing => "Preparing video and models…".into(),
            Progress::Tracking { done, total } => frames("Analyzing", done, total),
            Progress::Refining => "Refining tracks…".into(),
            Progress::WritingTracks => "Saving coordinates…".into(),
            Progress::Overlay { done, total } => frames("Writing overlay", done, total),
            Progress::FinishingOverlay => "Finalizing overlay…".into(),
        }
    }

    pub(crate) fn label(self) -> String {
        let stage = self.stage_label();
        if self.total > 1 && self.completed < self.total {
            format!("{} · {stage}", self.video_label())
        } else {
            stage
        }
    }
}

pub(crate) struct Control {
    state: Mutex<Snapshot>,
    changed: Condvar,
    current: Mutex<String>,
}

impl Control {
    fn new(total: usize) -> Self {
        Self {
            state: Mutex::new(Snapshot {
                mode: Mode::Running,
                progress: Progress::Preparing,
                completed: 0,
                total,
            }),
            changed: Condvar::new(),
            current: Mutex::new(String::new()),
        }
    }

    /// Names the video now being processed.
    pub(crate) fn begin_clip(&self, name: impl Into<String>) {
        *self.current.lock().unwrap() = name.into();
    }

    /// The name given to [`Self::begin_clip`], empty before the first video.
    pub(crate) fn current(&self) -> String {
        self.current.lock().unwrap().clone()
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        *self.state.lock().unwrap()
    }

    pub(crate) fn pause_or_resume(&self) {
        let mut state = self.state.lock().unwrap();
        state.mode = match state.mode {
            Mode::Running => Mode::Pausing,
            Mode::Pausing | Mode::Paused => Mode::Running,
            Mode::Stopping => return,
        };
        self.changed.notify_all();
    }

    pub(crate) fn stop(&self) {
        self.state.lock().unwrap().mode = Mode::Stopping;
        self.changed.notify_all();
    }

    pub(crate) fn report(&self, progress: Progress) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.progress = progress;
        self.checkpoint(state).map(|_| ())
    }

    /// Counts the current video as finished, without stopping here: the next video's first
    /// report is where a pause or stop takes effect.
    pub(crate) fn advance(&self) {
        let mut state = self.state.lock().unwrap();
        state.completed = (state.completed + 1).min(state.total);
        state.progress = Progress::Preparing;
    }

    pub(crate) fn clip_done(&self) -> Result<()> {
        let mut state = self.checkpoint(self.state.lock().unwrap())?;
        state.completed += 1;
        state.progress = Progress::Preparing;
        Ok(())
    }

    fn checkpoint<'a>(
        &self,
        mut state: MutexGuard<'a, Snapshot>,
    ) -> Result<MutexGuard<'a, Snapshot>> {
        if state.mode == Mode::Pausing {
            state.mode = Mode::Paused;
            self.changed.notify_all();
        }
        let state = self
            .changed
            .wait_while(state, |state| state.mode == Mode::Paused)
            .unwrap();
        if state.mode == Mode::Stopping {
            return Err(Stopped.into());
        }
        Ok(state)
    }
}

pub(crate) struct Job {
    /// Finishes with the status message to show.
    pub(crate) thread: JoinHandle<Result<String>>,
    pub(crate) control: Arc<Control>,
}

impl Job {
    pub(crate) fn start(
        total: usize,
        work: impl FnOnce(&Control) -> Result<String> + Send + 'static,
    ) -> Self {
        let control = Arc::new(Control::new(total));
        let worker_control = control.clone();
        let thread = std::thread::spawn(move || work(&worker_control));
        Self { thread, control }
    }
}

/// A stopped or failed analysis drops its partial files and keeps earlier results.
pub(crate) fn save_output(
    parent: &Path,
    prefix: &str,
    default_name: Option<&str>,
    work: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(parent)?;
    let dir = tempfile::Builder::new()
        .prefix(&format!("{prefix}-"))
        .tempdir_in(parent)?;
    work(dir.path())?;
    if let Some(name) = default_name {
        let dest = parent.join(name);
        if !dest.try_exists()? {
            match std::fs::rename(dir.path(), &dest) {
                Ok(()) => return Ok(dest),
                Err(_) if dest.is_dir() => {}
                Err(err) => {
                    return Err(err)
                        .with_context(|| format!("cannot save analysis to {}", dest.display()));
                }
            }
        }
    }
    Ok(dir.keep())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{self, TryRecvError};
    use std::time::Duration;

    #[test]
    fn processing_can_pause_resume_and_stop_while_paused() {
        let (resumed, observed) = mpsc::channel();
        let (finish, can_finish) = mpsc::channel();
        let job = Job::start(1, move |control| {
            // Wait for the test to request the first pause before reaching a checkpoint.
            can_finish.recv()?;
            control.report(Progress::Tracking { done: 2, total: 5 })?;
            resumed.send(())?;
            can_finish.recv()?;
            control.report(Progress::Overlay { done: 1, total: 5 })?;
            Ok(String::new())
        });
        let wait_for_pause = || {
            let (state, _) = job
                .control
                .changed
                .wait_timeout_while(
                    job.control.state.lock().unwrap(),
                    Duration::from_secs(5),
                    |state| state.mode != Mode::Paused,
                )
                .unwrap();
            assert_eq!(state.mode, Mode::Paused);
            *state
        };

        job.control.pause_or_resume();
        finish.send(()).unwrap();
        let paused = wait_for_pause();
        assert_eq!(paused.label(), "Analyzing · 2 / 5 frames");
        assert_eq!(observed.try_recv(), Err(TryRecvError::Empty));
        job.control.pause_or_resume();
        observed.recv_timeout(Duration::from_secs(5)).unwrap();

        job.control.pause_or_resume();
        finish.send(()).unwrap();
        let paused_overlay = wait_for_pause();
        assert!(paused_overlay.fraction() > paused.fraction());
        assert_eq!(paused_overlay.label(), "Writing overlay · 1 / 5 frames");
        job.control.stop();
        assert!(job.thread.join().unwrap().unwrap_err().is::<Stopped>());
    }

    #[test]
    fn a_batch_reports_the_video_count_and_the_frames_of_the_current_video() {
        let control = Control::new(3);
        control.advance();
        control
            .report(Progress::Tracking {
                done: 50,
                total: 100,
            })
            .unwrap();
        let snapshot = control.snapshot();
        assert_eq!(snapshot.video_label(), "Video 2 of 3");
        assert_eq!(snapshot.stage_label(), "Analyzing · 50 / 100 frames");
        assert!((snapshot.clip_fraction() - 0.425).abs() < 1e-6);
        assert!((snapshot.fraction() - (1.425 / 3.0)).abs() < 1e-6);
        control.advance();
        control.advance();
        control.advance();
        let done = control.snapshot();
        assert_eq!(done.video_label(), "3 of 3 videos done");
        assert_eq!(done.fraction(), 1.0);
    }
}
