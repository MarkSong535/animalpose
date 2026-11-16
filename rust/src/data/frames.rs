//! Frame sources: a video file or a folder of images, streamed as 8-bit BGR frames.
//!
//! Port of `src/animalpose/frames.py`'s `Source`, `open_source`, `iter_frames`, `source_fps`,
//! `sampled_background`, `frame_timestamps`, `timing_report` and `find_sources`. Images are returned
//! as 8-bit BGR whatever their storage: grey frames are replicated to three channels and frames
//! deeper than 8 bits are stretched to `0..=255` by their own min/max (`frames.py:to_bgr8` with no
//! range given, as `iter_frames` calls it). Videos decode through [`crate::data::video`].

use std::path::{Path, PathBuf};

use crate::algos::image::{Bgr, GrayF, ToGray};
use crate::data::images::{estimate_fps, image_files, read_gray, timestamp_of};
use crate::data::video::{self, VideoInfo, VideoReader};
use anyhow::{ensure, Context, Result};
use serde_json::json;

/// One recording: a video file, or a folder of image frames in order.
#[derive(Clone, Debug)]
pub enum Source {
    /// A video file and what `ffprobe` reports about it.
    Video(PathBuf, VideoInfo),
    /// Image frames, in the order `frames.py:image_files` sorts them.
    Images(PathBuf, Vec<PathBuf>),
}

/// The streamed frames of a [`Source`].
pub type FrameIter = Box<dyn Iterator<Item = Result<Bgr>>>;

impl Source {
    /// `frames.py:open_source`: a folder of images, one image file, or a video file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            let files = image_files(path)?;
            ensure!(!files.is_empty(), "no image frames in {}", path.display());
            return Ok(Self::Images(path.to_path_buf(), files));
        }
        ensure!(path.exists(), "no video or frame folder at {}", path.display());
        if video::is_video(path) {
            let info = video::probe(path)?;
            return Ok(Self::Video(path.to_path_buf(), info));
        }
        Ok(Self::Images(path.to_path_buf(), vec![path.to_path_buf()]))
    }

    /// The recording's path.
    pub fn path(&self) -> &Path {
        match self {
            Self::Video(p, _) | Self::Images(p, _) => p,
        }
    }

    /// `"video"` or `"images"`, as `Source.kind`.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Video(..) => "video",
            Self::Images(..) => "images",
        }
    }

    /// `Source.n_frames`.
    pub fn n_frames(&self) -> usize {
        match self {
            Self::Video(_, info) => info.frames,
            Self::Images(_, files) => files.len(),
        }
    }

    /// `frames.py:source_fps`: the container's rate, or the rate of file-name timestamps.
    pub fn fps(&self) -> Option<f64> {
        match self {
            Self::Video(_, info) => info.fps,
            Self::Images(_, files) => estimate_fps(files),
        }
    }

    /// `frames.py:iter_frames`: the first `max_frames` frames, decoded one at a time.
    pub fn iter(&self, max_frames: Option<usize>) -> Result<FrameIter> {
        Ok(match self {
            Self::Video(path, _) => Box::new(VideoReader::open(path, max_frames)?),
            Self::Images(_, files) => {
                let files: Vec<PathBuf> = files
                    .iter()
                    .take(max_frames.unwrap_or(usize::MAX))
                    .cloned()
                    .collect();
                Box::new(files.into_iter().map(|f| read_bgr(&f)))
            }
        })
    }

    /// `Source.frames(indices)`: the frames at `indices`, sorted, with their indices.
    pub fn frames_at(&self, indices: &[usize]) -> Result<Vec<(usize, Bgr)>> {
        self.frames_at_with_checkpoint(indices, || Ok(()))
    }

    fn frames_at_with_checkpoint(
        &self,
        indices: &[usize],
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<Vec<(usize, Bgr)>> {
        match self {
            Self::Video(path, _) => video::frames_at_with_checkpoint(path, indices, checkpoint),
            Self::Images(_, files) => {
                let mut wanted: Vec<usize> = indices.to_vec();
                wanted.sort_unstable();
                wanted.dedup();
                wanted
                    .into_iter()
                    .filter(|i| *i < files.len())
                    .map(|i| {
                        let frame = read_bgr(&files[i])?;
                        checkpoint()?;
                        Ok((i, frame))
                    })
                    .collect()
            }
        }
    }

    /// `frames.py:frame_timestamps`: presentation times in seconds from 0.
    ///
    /// Videos: `ffprobe` packet times. Image folders: times parsed from the file names (wrapping
    /// past midnight), else `index / estimated fps` (or 30 fps).
    pub fn timestamps(&self, max_frames: Option<usize>) -> Result<Vec<f64>> {
        let n = self.n_frames().min(max_frames.unwrap_or(usize::MAX));
        match self {
            Self::Video(path, info) => {
                let t = video::timestamps(path, max_frames)?;
                if !t.is_empty() {
                    return Ok(t);
                }
                let fps = info.fps.unwrap_or(30.0);
                Ok((0..n).map(|i| i as f64 / fps).collect())
            }
            Self::Images(_, files) => {
                let files = &files[..n];
                let stamps: Option<Vec<f64>> = files
                    .iter()
                    .map(|p| timestamp_of(&p.file_name()?.to_string_lossy()))
                    .collect();
                if let Some(mut t) = stamps.filter(|t| !t.is_empty()) {
                    let first = t[0];
                    for v in &mut t {
                        if *v < first {
                            *v += 86_400.0;
                        }
                        *v -= first;
                    }
                    return Ok(t);
                }
                let fps = estimate_fps(files).unwrap_or(30.0);
                Ok((0..n).map(|i| i as f64 / fps).collect())
            }
        }
    }

    /// `frames.py:sampled_background`: the grey median of `n` frames spread over the recording
    /// (`np.linspace(0, total - 1, min(n, total)).astype(int)`).
    pub fn sampled_background(&self, n: usize, max_frames: Option<usize>) -> Result<GrayF> {
        self.sampled_background_with_checkpoint(n, max_frames, || Ok(()))
    }

    pub(crate) fn sampled_background_with_checkpoint(
        &self,
        n: usize,
        max_frames: Option<usize>,
        checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<GrayF> {
        let total = self.n_frames().min(max_frames.unwrap_or(usize::MAX));
        ensure!(total > 0, "{} has no frames", self.path().display());
        let k = n.min(total).max(1);
        let indices: Vec<usize> = (0..k)
            .map(|i| {
                if k == 1 {
                    0
                } else {
                    // `linspace` then `astype(int)` truncates
                    ((i as f64) * ((total - 1) as f64) / ((k - 1) as f64)) as usize
                }
            })
            .collect();
        let frames = self.frames_at_with_checkpoint(&indices, checkpoint)?;
        ensure!(!frames.is_empty(), "no readable frames in {}", self.path().display());
        let grays: Vec<GrayF> = frames.iter().map(|(_, f)| f.to_gray()).collect();
        Ok(pixel_median(&grays))
    }
}

/// `np.median(np.stack(frames), 0)`: an even count averages the two middle values.
pub fn pixel_median(frames: &[GrayF]) -> GrayF {
    let (width, height) = (frames[0].width, frames[0].height);
    let mut window = vec![0.0f32; frames.len()];
    let pixels = (0..frames[0].pixels.len())
        .map(|i| {
            for (j, f) in frames.iter().enumerate() {
                window[j] = f.pixels[i];
            }
            window.sort_by(f32::total_cmp);
            let m = window.len();
            if m % 2 == 1 {
                window[m / 2]
            } else {
                (window[m / 2 - 1] + window[m / 2]) / 2.0
            }
        })
        .collect();
    GrayF {
        width,
        height,
        pixels,
    }
}

/// `frames.py:timing_report`: mean rate, interval spread and dropped-frame gaps (intervals longer
/// than 1.5 median intervals); `variable` when intervals spread by more than 10 % or frames dropped.
pub fn timing_report(times: &[f64]) -> serde_json::Value {
    if times.len() < 2 {
        return json!({"frames": times.len()});
    }
    let dt: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let mut sorted = dt.clone();
    sorted.sort_by(f64::total_cmp);
    let med = percentile_sorted(&sorted, 50.0);
    let gaps = dt.iter().filter(|d| **d > 1.5 * med).count();
    let mean = dt.iter().sum::<f64>() / dt.len() as f64;
    let std = (dt.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dt.len() as f64).sqrt();
    json!({
        "frames": times.len(),
        "duration_s": times[times.len() - 1],
        "mean_fps": 1.0 / mean,
        "interval_ms_p1_p99": [percentile_sorted(&sorted, 1.0) * 1e3, percentile_sorted(&sorted, 99.0) * 1e3],
        "gaps": gaps,
        "longest_gap_s": sorted[sorted.len() - 1],
        "variable": std > 0.1 * med || gaps > 0,
    })
}

/// `np.percentile` (linear interpolation) of an ascending slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let pos = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// `frames.py:find_sources`: every video file and every folder that directly holds image files,
/// anywhere under `root`, in `os.walk` order with sorted names.
pub fn find_sources(root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk(root.as_ref(), &mut out)?;
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot list {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    let has_images = entries.iter().any(|p| {
        p.is_file()
            && p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                crate::data::images::IMAGE_EXT.contains(&format!(".{}", e.to_lowercase()).as_str())
            })
    });
    if has_images {
        out.push(dir.to_path_buf());
    }
    out.extend(entries.iter().filter(|p| p.is_file() && video::is_video(p)).cloned());
    for sub in entries.iter().filter(|p| p.is_dir()) {
        walk(sub, out)?;
    }
    Ok(())
}

/// `frames.py:is_held_out`: about 1 in `every` recordings is held out, by the CRC-32 of its path
/// relative to `root`.
pub fn is_held_out(source: &Path, root: &Path, every: u32) -> bool {
    let cwd = std::env::current_dir().unwrap_or_default();
    let abs = |p: &Path| if p.is_absolute() { p.to_path_buf() } else { cwd.join(p) };
    let (s, r) = (abs(source), abs(root));
    let rel = s.strip_prefix(&r).unwrap_or(&s).to_string_lossy().replace('\\', "/");
    crc32(rel.as_bytes()) % every == 0
}

/// `zlib.crc32`.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for b in bytes {
        crc ^= u32::from(*b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// One image file as 8-bit BGR, `cv2.imread(..., IMREAD_UNCHANGED)` + `frames.py:to_bgr8`.
pub fn read_bgr(path: &Path) -> Result<Bgr> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    if ext == "npy" {
        return Ok(stretch_gray(&read_gray(path)?));
    }
    let decoded = image::ImageReader::open(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("cannot read {}", path.display()))?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?;
    use image::ColorType as C;
    match decoded.color() {
        C::L8 | C::La8 | C::Rgb8 | C::Rgba8 => {
            let rgb = decoded.to_rgb8();
            let (w, h) = rgb.dimensions();
            let mut pixels = rgb.into_raw();
            for px in pixels.chunks_exact_mut(3) {
                px.swap(0, 2);
            }
            Bgr::from_pixels(w, h, pixels)
        }
        _ => {
            // deeper than 8 bits: one min/max stretch over every channel of this frame
            let rgb = decoded.to_rgb32f();
            let (w, h) = rgb.dimensions();
            let raw = rgb.into_raw();
            let (lo, hi) = raw
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
            let scale = 255.0 / (hi - lo).max(1e-6);
            let mut pixels = Vec::with_capacity(raw.len());
            for px in raw.chunks_exact(3) {
                for c in [2, 1, 0] {
                    pixels.push(((px[c] - lo) * scale).clamp(0.0, 255.0) as u8);
                }
            }
            Bgr::from_pixels(w, h, pixels)
        }
    }
}

/// Grey levels to BGR: as stored when they are 8-bit whole numbers, else stretched by min/max.
fn stretch_gray(gray: &GrayF) -> Bgr {
    let eight_bit = gray
        .pixels
        .iter()
        .all(|v| *v == v.round() && (0.0..=255.0).contains(v));
    let (lo, hi) = if eight_bit {
        (0.0, 255.0)
    } else {
        gray.pixels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)))
    };
    let scale = 255.0 / (hi - lo).max(1e-6);
    let mut pixels = Vec::with_capacity(gray.pixels.len() * 3);
    for v in &gray.pixels {
        let g = ((v - lo) * scale).clamp(0.0, 255.0) as u8;
        pixels.extend_from_slice(&[g, g, g]);
    }
    Bgr {
        width: gray.width,
        height: gray.height,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        // zlib.crc32(b"123456789") == 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn timing_report_flags_dropped_frames() {
        let even: Vec<f64> = (0..10).map(|i| f64::from(i) / 10.0).collect();
        let r = timing_report(&even);
        assert_eq!(r["gaps"], 0);
        assert_eq!(r["variable"], false);
        assert!((r["mean_fps"].as_f64().unwrap() - 10.0).abs() < 1e-9);
        let mut dropped = even.clone();
        dropped.remove(5);
        let r = timing_report(&dropped);
        assert_eq!(r["gaps"], 1);
        assert_eq!(r["variable"], true);
        assert_eq!(timing_report(&[0.0])["frames"], 1);
    }

    #[test]
    fn median_averages_the_middle_pair() {
        let f = |v: f32| GrayF::from_pixels(1, 1, vec![v]).unwrap();
        assert_eq!(pixel_median(&[f(1.0), f(5.0), f(3.0), f(9.0)]).pixels, [4.0]);
        assert_eq!(pixel_median(&[f(1.0), f(5.0), f(3.0)]).pixels, [3.0]);
    }

    #[test]
    fn folders_stream_bgr_and_sample_a_background() {
        let dir = std::env::temp_dir().join(format!("animalpose-frames-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for t in 0..5u32 {
            let mut img = image::RgbImage::from_pixel(8, 6, image::Rgb([10, 20, 30]));
            img.put_pixel(t, 2, image::Rgb([200, 0, 0]));
            img.save(dir.join(format!("frame_00_00_00_{:03}.png", t * 100)))
                .unwrap();
        }
        let src = Source::open(&dir).unwrap();
        assert_eq!((src.kind(), src.n_frames()), ("images", 5));
        assert!((src.fps().unwrap() - 10.0).abs() < 1e-9);
        let frames: Vec<Bgr> = src.iter(Some(3)).unwrap().collect::<Result<_>>().unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].pixel(5, 5), [30, 20, 10]); // BGR order
        assert_eq!(frames[1].pixel(1, 2), [0, 0, 200]);
        let t = src.timestamps(None).unwrap();
        assert!((t[4] - 0.4).abs() < 1e-9);
        let bg = src.sampled_background(64, None).unwrap();
        // the red dot visits each column once, so the median is the floor everywhere
        assert!(bg.pixels.iter().all(|v| (v - bg.pixels[0]).abs() < 1e-6));
        assert_eq!(src.frames_at(&[4, 0, 9]).unwrap().len(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
