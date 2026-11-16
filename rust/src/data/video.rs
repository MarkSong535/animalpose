//! Video files through the `ffmpeg` / `ffprobe` executables: probing, streaming decode to 8-bit
//! BGR frames, packet timestamps, and an mp4 writer.
//!
//! The Python package decodes with `cv2.VideoCapture` and writes with `cv2.VideoWriter`
//! (`src/animalpose/frames.py`, `api.py:write_video`) and already shells out to `ffprobe` for
//! timestamps (`frames.py:frame_timestamps`). Here both directions go through the ffmpeg command
//! line, so the crate keeps building without a C toolchain; `ffmpeg` and `ffprobe` must be on `PATH`
//! (or named by `ANIMALPOSE_FFMPEG` / `ANIMALPOSE_FFPROBE`) only when a video file is actually read
//! or written.
//!
//! `// differs from Python:` frames are converted to BGR by ffmpeg's `swscale` (`-pix_fmt bgr24`),
//! where OpenCV uses its own YUV→BGR conversion; the two agree to within a grey level or two.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::algos::image::Bgr;
use anyhow::{bail, ensure, Context, Result};

/// Video extensions `frames.py:VIDEO_EXT` recognises.
pub const VIDEO_EXT: [&str; 8] = [
    ".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv",
];

/// `true` when `path` has one of [`VIDEO_EXT`].
pub fn is_video(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_lowercase()))
        .unwrap_or_default();
    VIDEO_EXT.contains(&ext.as_str())
}

fn tool(env: &str, default: &str) -> PathBuf {
    std::env::var_os(env).map_or_else(|| PathBuf::from(default), PathBuf::from)
}

fn ffmpeg() -> PathBuf {
    tool("ANIMALPOSE_FFMPEG", "ffmpeg")
}

fn ffprobe() -> PathBuf {
    tool("ANIMALPOSE_FFPROBE", "ffprobe")
}

fn video_command(exe: impl AsRef<Path>) -> Command {
    #[allow(unused_mut)] // macOS and Windows configure the child's environment or console.
    let mut cmd = Command::new(exe.as_ref());
    #[cfg(target_os = "macos")]
    {
        // Finder-launched apps may have only the system directories on PATH.
        let path = std::env::var_os("PATH").unwrap_or_default();
        let paths = std::env::split_paths(&path)
            .chain([PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin")]);
        if let Ok(path) = std::env::join_paths(paths) {
            cmd.env("PATH", path);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// `true` when the `ffmpeg` executable runs.
pub fn ffmpeg_available() -> bool {
    video_command(ffmpeg())
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn passthrough_args(exe: &Path) -> [&'static str; 2] {
    // FFmpeg 4.4 uses -vsync. New releases accept -fps_mode and may remove -vsync.
    let supports_fps_mode = video_command(exe)
        .args(["-hide_banner", "-h", "full"])
        .output()
        .is_ok_and(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|line| line.starts_with("-fps_mode"))
        });
    if supports_fps_mode {
        ["-fps_mode", "passthrough"]
    } else {
        ["-vsync", "0"]
    }
}

fn spawn_error(what: &Path, err: &std::io::Error) -> anyhow::Error {
    anyhow::anyhow!(
        "cannot run {}: {err}; install ffmpeg (https://ffmpeg.org) or set ANIMALPOSE_FFMPEG / \
         ANIMALPOSE_FFPROBE to the executables",
        what.display()
    )
}

/// What `ffprobe` reports about the first video stream.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    /// `cv2.CAP_PROP_FPS`'s counterpart: the stream's average frame rate, `None` when unknown.
    pub fps: Option<f64>,
    /// Frame count from the container, or by counting packets when the container has none.
    pub frames: usize,
}

/// `ffprobe` on the first video stream of `path`.
pub fn probe(path: &Path) -> Result<VideoInfo> {
    ensure!(path.is_file(), "no video file at {}", path.display());
    let exe = ffprobe();
    let out = video_command(&exe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_packets",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| spawn_error(&exe, &e))?;
    ensure!(
        out.status.success(),
        "ffprobe cannot read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("ffprobe printed malformed JSON")?;
    let stream = json["streams"]
        .get(0)
        .with_context(|| format!("{} has no video stream", path.display()))?;
    let int = |key: &str| -> Option<u64> {
        let v = &stream[key];
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    };
    let rate = |key: &str| -> Option<f64> {
        let s = stream[key].as_str()?;
        let (n, d) = s.split_once('/').unwrap_or((s, "1"));
        let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
        (d > 0.0 && n > 0.0).then(|| n / d)
    };
    let width = int("width").context("ffprobe reported no width")?;
    let height = int("height").context("ffprobe reported no height")?;
    let frames = int("nb_frames")
        .filter(|n| *n > 0)
        .or_else(|| int("nb_read_packets"))
        .unwrap_or(0);
    Ok(VideoInfo {
        width: u32::try_from(width)?,
        height: u32::try_from(height)?,
        fps: rate("avg_frame_rate").or_else(|| rate("r_frame_rate")),
        frames: usize::try_from(frames)?,
    })
}

/// Streams the frames of a video as 8-bit BGR images, decoding in an `ffmpeg` child process.
pub struct VideoReader {
    child: Child,
    stdout: BufReader<ChildStdout>,
    width: u32,
    height: u32,
    remaining: Option<usize>,
    path: PathBuf,
}

impl VideoReader {
    /// Opens `path`; at most `max_frames` frames are produced.
    pub fn open(path: &Path, max_frames: Option<usize>) -> Result<Self> {
        let info = probe(path)?;
        Self::open_filtered(path, &info, max_frames, None)
    }

    /// Opens `path` with an optional ffmpeg `-vf` filter (e.g. a `select=` of frame indices).
    fn open_filtered(
        path: &Path,
        info: &VideoInfo,
        max_frames: Option<usize>,
        filter: Option<&str>,
    ) -> Result<Self> {
        let exe = ffmpeg();
        let mut cmd = video_command(&exe);
        cmd.args(["-v", "error", "-nostdin", "-i"]).arg(path);
        if let Some(filter) = filter {
            cmd.args(["-vf", filter]).args(passthrough_args(&exe));
        }
        if let Some(n) = max_frames {
            cmd.args(["-frames:v", &n.to_string()]);
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| spawn_error(&exe, &e))?;
        let stdout = child.stdout.take().context("ffmpeg has no stdout")?;
        Ok(Self {
            child,
            stdout: BufReader::with_capacity(1 << 20, stdout),
            width: info.width,
            height: info.height,
            remaining: max_frames,
            path: path.to_owned(),
        })
    }

    fn read_frame(&mut self) -> Result<Option<Bgr>> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        let n = self.width as usize * self.height as usize * 3;
        let mut pixels = vec![0u8; n];
        let mut filled = 0;
        while filled < n {
            let got = self.stdout.read(&mut pixels[filled..])?;
            if got == 0 {
                self.remaining = Some(0);
                let status = self.child.wait()?;
                ensure!(status.success(), "ffmpeg could not decode {} ({status})", self.path.display());
                ensure!(filled == 0, "ffmpeg ended in the middle of a frame");
                return Ok(None);
            }
            filled += got;
        }
        if let Some(r) = self.remaining.as_mut() {
            *r -= 1;
        }
        Ok(Some(Bgr::from_pixels(self.width, self.height, pixels)?))
    }
}

impl Iterator for VideoReader {
    type Item = Result<Bgr>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_frame().transpose()
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The frames at `indices` (any order; returned sorted, duplicates dropped), with their indices.
pub fn frames_at(path: &Path, indices: &[usize]) -> Result<Vec<(usize, Bgr)>> {
    let mut wanted: Vec<usize> = indices.to_vec();
    wanted.sort_unstable();
    wanted.dedup();
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let info = probe(path)?;
    // One decode pass keeping the selected frames: `select` evaluates `n` (the 0-based input index).
    let expr = wanted
        .iter()
        .map(|i| format!("eq(n\\,{i})"))
        .collect::<Vec<_>>()
        .join("+");
    let reader = VideoReader::open_filtered(path, &info, Some(wanted.len()), Some(&format!("select={expr}")))?;
    let mut out = Vec::with_capacity(wanted.len());
    for (index, frame) in wanted.iter().zip(reader) {
        out.push((*index, frame?));
    }
    Ok(out)
}

/// `frames.py:frame_timestamps` for a video: packet presentation times from `ffprobe`, sorted and
/// shifted to start at 0, at most `max_frames` of them.
pub fn timestamps(path: &Path, max_frames: Option<usize>) -> Result<Vec<f64>> {
    let exe = ffprobe();
    let out = video_command(&exe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| spawn_error(&exe, &e))?;
    ensure!(
        out.status.success(),
        "ffprobe cannot read {}",
        path.display()
    );
    let mut t: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .filter_map(|s| s.trim_end_matches(',').parse::<f64>().ok())
        .collect();
    t.sort_by(f64::total_cmp);
    if let Some(&first) = t.first() {
        for v in &mut t {
            *v -= first;
        }
    }
    if let Some(n) = max_frames {
        t.truncate(n);
    }
    Ok(t)
}

/// `api.py:write_video`: an mp4 written frame by frame through an `ffmpeg` child process.
///
/// `// differs from Python:` `cv2.VideoWriter` encodes `mp4v` (MPEG-4 part 2); this writer uses
/// H.264 (`libx264`, `yuv420p`) when the ffmpeg build has it, which every player opens, and falls
/// back to `mpeg4` otherwise.
pub struct VideoWriter {
    path: PathBuf,
    fps: f64,
    inner: Option<(Child, ChildStdin, u32, u32)>,
    written: usize,
}

impl VideoWriter {
    /// A writer that creates `path` when the first frame arrives (its size fixes the video's).
    pub fn new(path: impl Into<PathBuf>, fps: f64) -> Self {
        Self {
            path: path.into(),
            fps: if fps.is_finite() && fps > 0.0 { fps } else { 30.0 },
            inner: None,
            written: 0,
        }
    }

    fn start(&mut self, width: u32, height: u32) -> Result<()> {
        let exe = ffmpeg();
        let codec = if encoder_available("libx264") {
            vec!["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        } else {
            vec!["-c:v", "mpeg4", "-q:v", "3"]
        };
        // yuv420p needs even dimensions; pad by one pixel when a side is odd.
        let mut child = video_command(&exe)
            .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s"])
            .arg(format!("{width}x{height}"))
            .args(["-r", &format!("{}", self.fps), "-i", "-"])
            .args(codec)
            .args([
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ])
            .arg(&self.path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| spawn_error(&exe, &e))?;
        let stdin = child.stdin.take().context("ffmpeg has no stdin")?;
        self.inner = Some((child, stdin, width, height));
        Ok(())
    }

    /// Appends one frame; every frame must have the first frame's size.
    pub fn write(&mut self, frame: &Bgr) -> Result<()> {
        if self.inner.is_none() {
            if let Some(parent) = self.path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            self.start(frame.width, frame.height)?;
        }
        let (_, stdin, w, h) = self.inner.as_mut().expect("started above");
        ensure!(
            frame.width == *w && frame.height == *h,
            "frame {}x{} differs from the video's {w}x{h}",
            frame.width,
            frame.height
        );
        stdin
            .write_all(&frame.pixels)
            .context("ffmpeg stopped accepting frames")?;
        self.written += 1;
        Ok(())
    }

    /// Flushes and closes the file; returns how many frames were written.
    pub fn finish(mut self) -> Result<usize> {
        if let Some((mut child, stdin, _, _)) = self.inner.take() {
            drop(stdin);
            let mut err = String::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_string(&mut err);
            }
            let status = child.wait()?;
            if !status.success() {
                bail!(
                    "ffmpeg failed writing {}: {}",
                    self.path.display(),
                    err.trim()
                );
            }
        }
        Ok(self.written)
    }
}

/// `true` when `ffmpeg -encoders` lists `name`.
fn encoder_available(name: &str) -> bool {
    video_command(ffmpeg())
        .args(["-hide_banner", "-encoders"])
        .output()
        .is_ok_and(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .any(|w| w == name)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_extensions_are_recognised() {
        assert!(is_video(Path::new("a/b/clip.MP4")));
        assert!(is_video(Path::new("clip.avi")));
        assert!(!is_video(Path::new("frames/0001.png")));
        assert!(!is_video(Path::new("frames")));
    }

    /// Write a clip, probe it, stream it back, pick frames out of it and read its timestamps.
    #[test]
    fn round_trip_through_ffmpeg() {
        if !ffmpeg_available() {
            eprintln!("SKIP: ffmpeg is not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!("animalpose-video-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.mp4");
        let mut writer = VideoWriter::new(&path, 25.0);
        for t in 0..12u8 {
            let mut frame = Bgr::filled(64, 48, [20, 20, 20]);
            let x = 4 + u32::from(t) * 4;
            frame.fill_rect(x, 16, x + 8, 32, [230, 230, 230]);
            writer.write(&frame).unwrap();
        }
        assert_eq!(writer.finish().unwrap(), 12);

        let info = probe(&path).unwrap();
        assert_eq!((info.width, info.height, info.frames), (64, 48, 12));
        assert!((info.fps.unwrap() - 25.0).abs() < 1e-6);

        let frames: Vec<Bgr> = VideoReader::open(&path, None)
            .unwrap()
            .collect::<Result<_>>()
            .unwrap();
        assert_eq!(frames.len(), 12);
        // the bright block moves right by 4 px a frame
        let bright = |f: &Bgr| (0..64).find(|&x| f.pixel(x, 24)[1] > 128).unwrap();
        assert!(bright(&frames[0]).abs_diff(4) <= 1);
        assert!(bright(&frames[5]).abs_diff(24) <= 1);
        assert_eq!(VideoReader::open(&path, Some(3)).unwrap().count(), 3);

        let picked = frames_at(&path, &[9, 2, 2]).unwrap();
        assert_eq!(picked.iter().map(|p| p.0).collect::<Vec<_>>(), [2, 9]);
        assert_eq!(picked[0].1.pixels, frames[2].pixels);
        assert_eq!(picked[1].1.pixels, frames[9].pixels);
        assert!(frames_at(&path, &[12, 99]).unwrap().is_empty());

        let mut broken = VideoReader::open_filtered(&path, &info, None, Some("invalid_filter")).unwrap();
        assert!(broken.next().unwrap().unwrap_err().to_string().contains("could not decode"));
        assert!(broken.next().is_none());

        let t = timestamps(&path, None).unwrap();
        assert_eq!(t.len(), 12);
        assert!((t[1] - 0.04).abs() < 1e-6 && t[0] == 0.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
