//! What the app did, kept as text: a daily log in `~/AnimalPose/logs/<date>.log` and, while a job
//! runs, a copy in the job's result folder (`animalpose.log`). Times are UTC.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) struct Log {
    dir: PathBuf,
    job: Mutex<Option<File>>,
}

impl Log {
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            job: Mutex::new(None),
        }
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    /// Appends `text`, one timestamped line per line.
    pub(crate) fn line(&self, text: &str) {
        let (date, time) = now();
        let lines: String = text
            .lines()
            .map(|line| format!("{date} {time}Z {line}\n"))
            .collect();
        let daily = fs::create_dir_all(&self.dir).and_then(|_| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.dir.join(format!("{date}.log")))
        });
        if let Ok(mut file) = daily {
            let _ = file.write_all(lines.as_bytes());
        }
        if let Some(file) = self.job.lock().unwrap().as_mut() {
            let _ = file.write_all(lines.as_bytes());
        }
    }

    /// Copies later lines into `path` too, until [`Self::end_job`].
    pub(crate) fn start_job(&self, path: &Path) {
        *self.job.lock().unwrap() = OpenOptions::new().create(true).append(true).open(path).ok();
    }

    pub(crate) fn end_job(&self) {
        *self.job.lock().unwrap() = None;
    }

    /// Runs `work` with its lines copied into `path`, closing that file before returning: Windows
    /// cannot rename a folder while a file in it is open, and the result folder is renamed next.
    pub(crate) fn during_job<T>(&self, path: &Path, work: impl FnOnce() -> T) -> T {
        self.start_job(path);
        let result = work();
        self.end_job();
        result
    }
}

/// `("2026-11-16", "17:18:37")` in UTC.
fn now() -> (String, String) {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (y, m, d) = civil(i64::try_from(secs / 86_400).unwrap_or(0));
    let rem = secs % 86_400;
    (
        format!("{y:04}-{m:02}-{d:02}"),
        format!("{:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60),
    )
}

/// Days since 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_job_copies() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_773), (2026, 11, 16));
        assert_eq!(civil(11_016), (2000, 2, 29));
        let dir = tempfile::tempdir().unwrap();
        let log = Log::new(dir.path().join("logs"));
        let job = dir.path().join("job.log");
        log.start_job(&job);
        log.line("first\nsecond");
        log.end_job();
        log.line("third");
        let copy = fs::read_to_string(&job).unwrap();
        assert_eq!(copy.lines().count(), 2);
        assert!(copy.lines().all(|l| l.contains("Z ")));
        let daily = fs::read_dir(dir.path().join("logs")).unwrap().count();
        assert_eq!(daily, 1);

        assert_eq!(log.during_job(&job, || 7), 7);
        assert!(
            log.job.lock().unwrap().is_none(),
            "the job's copy is closed"
        );
    }
}
