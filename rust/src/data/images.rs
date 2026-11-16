//! Grayscale frames: reading a folder of images or a `.npy` stack, and the resize/letterbox
//! helpers the network inputs are built from.
//!
//! The Python source is `src/animalpose/frames.py` (loading, frame-rate estimation, 8-bit
//! conversion) and `src/animalpose/data/datasets.py` (`letterbox`, `to_tensor`).

use std::path::{Path, PathBuf};

use crate::algos::image::{Bgr, GrayF, Image, ToGray};
use anyhow::{bail, ensure, Result};

use crate::data::sessions::{read_npy, NpyData};

/// Extensions this module decodes: `frames.py:IMAGE_EXT` plus `.npy`.
pub const IMAGE_EXT: [&str; 7] = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".npy"];

/// The image files of a folder in the order `frames.py:image_files` returns them.
///
/// Sorting is on the file name, as Python's `sorted(...)` over `Path` objects reduces to here for
/// files sharing a directory.
pub fn image_files(folder: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let folder = folder.as_ref();
    let mut files: Vec<PathBuf> = Vec::new();
    let entries = folder
        .read_dir()
        .map_err(|e| anyhow::anyhow!("cannot read frame folder {}: {e}", folder.display()))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| anyhow::anyhow!("cannot read frame folder {}: {e}", folder.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = extension(&path);
        if IMAGE_EXT.contains(&ext.as_str()) {
            files.push(path);
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(files)
}

/// `frames.py:image_files` + `read_frames` for a folder (or one file): the frames as grey levels,
/// and the frame rate estimated from file-name timestamps when one is present, divided by `stride`
/// the way `read_frames` returns `fps / stride`.
///
/// Video files decode through [`crate::data::video`] (ffmpeg), as `cv2.VideoCapture` does in
/// Python, and come back as the grey level of their BGR frames. `.npy` frames are a Rust-only
/// addition, since a `.npy` is what `data/behavior_sets.py:save_session` and the fixture dumps write.
pub fn read_frames(
    path: impl AsRef<Path>,
    max_frames: Option<usize>,
    stride: usize,
) -> Result<(Vec<GrayF>, Option<f64>)> {
    read_frames_with_checkpoint(path, max_frames, stride, || Ok(()))
}

pub(crate) fn read_frames_with_checkpoint(
    path: impl AsRef<Path>,
    max_frames: Option<usize>,
    stride: usize,
    mut checkpoint: impl FnMut() -> Result<()>,
) -> Result<(Vec<GrayF>, Option<f64>)> {
    ensure!(stride > 0, "frame stride must be positive, got {stride}");
    let source = path.as_ref();
    if source.is_file() && crate::data::video::is_video(source) {
        let video = crate::data::frames::Source::open(source)?;
        let mut frames = Vec::new();
        for (t, frame) in video.iter(None)?.enumerate() {
            if max_frames.is_some_and(|m| frames.len() >= m) {
                break;
            }
            let frame = frame?;
            if t % stride == 0 {
                frames.push(frame.to_gray());
            }
            checkpoint()?;
        }
        let fps = video.fps().map(|fps| fps / stride as f64);
        return Ok((frames, fps));
    }
    let files = if source.is_dir() {
        image_files(source)?
    } else {
        ensure!(
            IMAGE_EXT.contains(&extension(source).as_str()),
            "{} is not a video, frame folder, image or .npy stack",
            source.display()
        );
        vec![source.to_path_buf()]
    };
    ensure!(!files.is_empty(), "no frames found in {}", source.display());
    // `image_files(path)[::stride][:max_frames]`: the stride and the cap apply before anything else.
    let selected: Vec<PathBuf> = files
        .iter()
        .step_by(stride)
        .take(max_frames.unwrap_or(usize::MAX))
        .cloned()
        .collect();
    ensure!(!selected.is_empty(), "no frames in {}", source.display());
    // `read_frames` returns `fps / stride`: a folder read every second frame plays twice as slow.
    let fps = estimate_fps(&selected).map(|fps| fps / f64::from(stride as u32));

    // Float sequences are scaled to 8 bits with one range for the whole sequence
    // (`frames.py:read_frames` estimates it from `raw[:: max(1, len(raw) // 50)]`, about 50 frames,
    // and takes the 0.5 / 99.5 percentiles of those).
    let sample_step = (selected.len() / 50).max(1);
    let sampled: Vec<&PathBuf> = selected.iter().step_by(sample_step).collect();
    let (lo, hi) = grey_range(&sampled)?;

    let mut frames = Vec::with_capacity(selected.len());
    for file in &selected {
        let gray = read_gray(file)?;
        frames.push(match lo {
            Some(lo) => scale_to_u8_gray(&gray, lo, hi.unwrap_or(255.0)),
            None => gray,
        });
        checkpoint()?;
    }
    Ok((frames, fps))
}

/// Loads one frame as grey levels in `0..=255`: PNG/JPEG through their decoders, `.npy` through
/// the array reader of [`crate::data::sessions`].
pub fn read_gray(path: impl AsRef<Path>) -> Result<GrayF> {
    let path = path.as_ref();
    match extension(path).as_str() {
        ".npy" => read_npy_gray(path),
        ext if IMAGE_EXT.contains(&ext) => {
            let decoded = image::ImageReader::open(path)
                .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?
                .with_guessed_format()
                .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?
                .decode()
                .map_err(|e| anyhow::anyhow!("cannot decode {}: {e}", path.display()))?;
            use image::ColorType as C;
            let eight_bit = matches!(decoded.color(), C::L8 | C::La8 | C::Rgb8 | C::Rgba8);
            if !eight_bit {
                // 16-bit / float frames keep their raw levels; `read_frames` stretches them with one
                // range for the whole sequence, as `frames.py:read_frames` does.
                let luma = decoded.to_luma16();
                let (width, height) = luma.dimensions();
                let pixels = luma.as_raw().iter().map(|v| f32::from(*v)).collect();
                return GrayF::from_pixels(width, height, pixels);
            }
            let luma = decoded.to_luma8();
            let (width, height) = luma.dimensions();
            let pixels = luma
                .as_raw()
                .iter()
                .map(|v| f32::from(*v))
                .collect::<Vec<f32>>();
            GrayF::from_pixels(width, height, pixels)
        }
        other => bail!("unsupported: {} frame format ({})", path.display(), other),
    }
}

fn read_npy_gray(path: &Path) -> Result<GrayF> {
    let array = read_npy(path)?;
    let (height, width) = match array.shape.as_slice() {
        [h, w] => (*h, *w),
        [frames, h, w] if *frames == 1 => (*h, *w),
        other => bail!(
            "unsupported: {} has shape {other:?}; frame stacks need one 2-D image",
            path.display()
        ),
    };
    let pixels = match &array.data {
        // `uint8` arrays are grey levels as stored; float arrays keep their raw values here and
        // are stretched to 8 bits sequence-wide by [`read_frames`], as `frames.py:read_frames` does.
        NpyData::Bytes(v) => v.iter().map(|v| f32::from(*v)).collect(),
        _ => array
            .floats()
            .ok_or_else(|| anyhow::anyhow!("{} does not hold a numeric image", path.display()))?,
    };
    GrayF::from_pixels(width as u32, height as u32, pixels)
}

/// `frames.py:estimate_fps`: `1 / median(diff(timestamps))` over the first 500 names, where the
/// timestamp is the trailing `H_MM_SS_mmm` group of `frames.py:_timestamp`.
pub fn estimate_fps(files: &[PathBuf]) -> Option<f64> {
    let stamps: Vec<f64> = files
        .iter()
        .take(500)
        .map(|p| timestamp(&p.file_name()?.to_string_lossy()))
        .collect::<Option<Vec<f64>>>()?;
    if stamps.len() < 3 {
        return None;
    }
    let mut gaps: Vec<f64> = stamps.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.retain(|d| *d > 0.0);
    if gaps.is_empty() {
        return None;
    }
    Some(1.0 / median(&mut gaps))
}

/// [`timestamp`] for other modules: the time a frame's file name carries, if any.
pub(crate) fn timestamp_of(name: &str) -> Option<f64> {
    timestamp(name)
}

/// `frames.py:_timestamp`: `h * 3600 + m * 60 + s + ms / 1000` from a trailing
/// `(\d{1,2})_(\d{2})_(\d{2})_(\d{1,3})\D*` group, else `None`.
fn timestamp(name: &str) -> Option<f64> {
    let stem = match name.rfind('.') {
        Some(at) => &name[..at],
        None => name,
    };
    let bytes = stem.as_bytes();
    // Scan backwards for `_dd_dd_...`; the millisecond group may be followed by non-digits.
    let mut i = bytes.len();
    let digits = |at: usize, len: usize| -> Option<(usize, usize)> {
        if at < len {
            return None;
        }
        if bytes[at - len..at].iter().all(u8::is_ascii_digit) {
            Some((at - len, at))
        } else {
            None
        }
    };
    while i > 0 && !bytes[i - 1].is_ascii_digit() {
        i -= 1;
    }
    if i == 0 {
        return None;
    }
    let (ms0, ms1) = digits(i, 3)
        .or_else(|| digits(i, 2))
        .or_else(|| digits(i, 1))?;
    let groups = |end: usize| {
        digits(end.checked_sub(1)?, 2)
            .or_else(|| digits(end.checked_sub(1)?, 1))
            .filter(|(s, _)| *s >= 1 && bytes[*s - 1] == b'_')
    };
    let (s0, s1) = groups(ms0)?;
    let (m0, m1) = groups(s0)?;
    let (h0, h1) = groups(m0)?;
    let number = |a: usize, b: usize| stem[a..b].parse::<f64>().ok();
    Some(
        number(h0, h1)? * 3600.0
            + number(m0, m1)? * 60.0
            + number(s0, s1)?
            + number(ms0, ms1)? / 1000.0,
    )
}

/// The `(lo, hi)` grey range of `frames.py:read_frames`, or `None` when every frame is already
/// 8-bit and no stretch is needed.
///
/// `// differs from Python:` Python asks for the first frame's *dtype* (`raw[0].dtype != uint8`),
/// so a float array that happens to hold whole numbers is stretched there and left alone here,
/// where the pixels themselves decide. Every 8-bit PNG/JPEG folder — the frames both ports are
/// handed in practice — takes the same branch.
fn grey_range(files: &[&PathBuf]) -> Result<(Option<f64>, Option<f64>)> {
    let mut samples: Vec<f64> = Vec::new();
    let mut all_integral = true;
    for file in files {
        let gray = read_gray(file)?;
        for v in &gray.pixels {
            if *v != v.round() {
                all_integral = false;
            }
        }
        samples.extend(gray.pixels.iter().map(|v| f64::from(*v)));
    }
    if all_integral || samples.is_empty() {
        return Ok((None, None));
    }
    let lo = percentile(&mut samples.clone(), 0.5);
    let hi = percentile(&mut samples, 99.5);
    Ok((Some(lo), Some(hi)))
}

/// `frames.py:read_frames`' rescaling, applied to grey levels: `(v - lo) * 255 / max(hi - lo, 1e-6)`
/// clipped to `0..=255` and rounded the way `astype(np.uint8)` truncates.
fn scale_to_u8_gray(gray: &GrayF, lo: f64, hi: f64) -> GrayF {
    let span = (hi - lo).max(1e-6);
    let pixels = gray
        .pixels
        .iter()
        .map(|v| {
            ((f64::from(*v) - lo) * (255.0 / span))
                .clamp(0.0, 255.0)
                .trunc() as f32
        })
        .collect();
    GrayF {
        width: gray.width,
        height: gray.height,
        pixels,
    }
}

/// `np.percentile` with numpy's default linear interpolation.
pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let pos = (values.len() - 1) as f64 * (p / 100.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    values[lo] + frac * (values[hi] - values[lo])
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn extension(path: &Path) -> String {
    path.extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

/// `frames.py:to_bgr8` for a grey frame: the grey level replicated into three channels.
pub fn to_bgr8(gray: &GrayF) -> Bgr {
    let mut pixels = Vec::with_capacity((gray.width * gray.height) as usize * 3);
    for v in &gray.pixels {
        let v = v.round().clamp(0.0, 255.0) as u8;
        pixels.extend_from_slice(&[v, v, v]);
    }
    Bgr {
        width: gray.width,
        height: gray.height,
        pixels,
    }
}

/// `data/datasets.py:to_tensor`: an 8-bit `HWC` image to planar `RGB` floats in `0..=1`.
pub fn to_tensor(image: &Bgr) -> Result<Vec<f32>> {
    let plane = (image.width * image.height) as usize;
    ensure!(
        image.pixels.len() == plane * 3,
        "BGR image {}x{} needs {} samples, got {}",
        image.width,
        image.height,
        plane * 3,
        image.pixels.len()
    );
    let mut out = vec![0.0f32; plane * 3];
    for (i, px) in image.pixels.chunks_exact(3).enumerate() {
        // `img[..., ::-1]` reverses BGR to RGB.
        out[i] = f32::from(px[2]) / 255.0;
        out[plane + i] = f32::from(px[1]) / 255.0;
        out[2 * plane + i] = f32::from(px[0]) / 255.0;
    }
    Ok(out)
}

/// Bilinear resize of a float grey frame to `(width, height)`; the padding-free variant of
/// `data/datasets.py:letterbox`'s scaling step.
///
/// Ported through [`crate::algos::image::resize_linear`], which follows the `cv2.INTER_LINEAR`
/// sample mapping.
pub fn resize_gray(gray: &GrayF, width: u32, height: u32) -> GrayF {
    let pixels =
        crate::algos::image::resize_linear(&gray.pixels, gray.width, gray.height, width, height);
    GrayF {
        width,
        height,
        pixels,
    }
}

/// `data/datasets.py:letterbox` for a float grey frame: scale to fit `size = (H, W)`, keep the
/// aspect ratio, centre on a zero border. Returns the padded frame, the scale and `(pad_x, pad_y)`.
pub fn letterbox_gray(gray: &GrayF, size: (u32, u32)) -> Result<(GrayF, f64, [f64; 2])> {
    let image = u8_from_gray(gray);
    let (padded, scale, pad) = crate::algos::image::letterbox_gray(&image, size)?;
    Ok((padded.to_gray(), scale, pad))
}

/// The 8-bit image of `frames.py:to_bgr8`, kept single channel: what the Python pipeline hands to
/// `cv2.resize` after `_gray`, rounded to grey levels.
pub fn u8_from_gray(gray: &GrayF) -> Image {
    let pixels = gray
        .pixels
        .iter()
        .map(|v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    Image {
        width: gray.width,
        height: gray.height,
        pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_png(path: &Path, w: u32, h: u32, fill: u8) {
        let img = image::GrayImage::from_pixel(w, h, image::Luma([fill]));
        img.save(path).unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("animalpose-images-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sorts_and_reads_png_folders() {
        let dir = temp_dir("sort");
        write_png(&dir.join("frame_10.png"), 4, 3, 10);
        write_png(&dir.join("frame_2.png"), 4, 3, 20);
        std::fs::write(dir.join("notes.txt"), b"not an image").unwrap();
        let files = image_files(&dir).unwrap();
        assert_eq!(
            files
                .iter()
                .map(|p| p.file_name().unwrap().into())
                .collect::<Vec<std::ffi::OsString>>(),
            vec!["frame_10.png", "frame_2.png"]
        );
        let (frames, fps) = read_frames(&dir, None, 1).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!((frames[0].width, frames[0].height), (4, 3));
        // Lexicographic, as Python's `sorted` is: "frame_10.png" precedes "frame_2.png".
        assert_eq!(frames[0].pixels[0], 10.0);
        assert_eq!(frames[1].pixels[0], 20.0);
        assert_eq!(fps, None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn honours_max_frames_and_stride() {
        let dir = temp_dir("stride");
        for i in 0..6 {
            write_png(&dir.join(format!("f{i:02}.png")), 2, 2, i);
        }
        let (frames, _) = read_frames(&dir, None, 2).unwrap();
        assert_eq!(
            frames.iter().map(|f| f.pixels[0]).collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0]
        );
        let (frames, _) = read_frames(&dir, Some(2), 1).unwrap();
        assert_eq!(frames.len(), 2);
        std::fs::remove_dir_all(&dir).ok();

        // `read_frames` returns `fps / stride`, and estimates that fps from the frames it actually
        // kept: a 10 fps folder read every second frame is timed as 5 fps / 2 = 2.5 fps, the number
        // Python's `fps / stride` gives (its own fps estimate over the kept frames is already 5).
        let dir = temp_dir("stride-fps");
        for i in 0u8..6 {
            let centis = u32::from(i) * 100;
            write_png(
                &dir.join(format!(
                    "img_00_00_{:02}_{:03}.png",
                    centis / 1000,
                    centis % 1000
                )),
                2,
                2,
                i,
            );
        }
        let (frames, fps) = read_frames(&dir, None, 2).unwrap();
        assert_eq!(frames.len(), 3);
        let fps = fps.expect("the names carry timestamps");
        assert!(
            (fps - 2.5).abs() < 1e-5,
            "a stride of 2 over a 10 fps folder gives {fps}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sixteen_bit_tif_sequences_share_one_range() {
        let dir = temp_dir("tif");
        for (i, level) in [1000u16, 3000, 5000].iter().enumerate() {
            let mut img = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_pixel(
                4,
                4,
                image::Luma([*level]),
            );
            img.put_pixel(0, 0, image::Luma([60_000]));
            img.save(dir.join(format!("f{i}.tif"))).unwrap();
        }
        let (frames, _) = read_frames(&dir, None, 1).unwrap();
        assert_eq!(frames.len(), 3);
        // one stretch for the sequence: the darker frame stays darker
        assert!(frames[0].pixel(1, 1) < frames[1].pixel(1, 1));
        assert!(frames[1].pixel(1, 1) < frames[2].pixel(1, 1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn timestamps_drive_frame_rate() {
        // `img1030_13_47_28_113` style names at 10 ms spacing.
        let names: Vec<PathBuf> = (0..5)
            .map(|i| PathBuf::from(format!("img_13_47_28_{:03}.png", 100 + 10 * i)))
            .collect();
        // `1 / median(diff(...))` on timestamps of ~49648 s: the gaps carry ~2e-12 s of float
        // error, the same figure Python's float64 arithmetic produces.
        assert!((estimate_fps(&names).unwrap() - 100.0).abs() < 1e-5);
        assert_eq!(estimate_fps(&[PathBuf::from("frame_001.png")]), None);
    }

    #[test]
    fn timestamp_matches_python_suffix() {
        assert_eq!(
            timestamp("img1030_13_47_28_113.tif"),
            Some(13.0 * 3600.0 + 47.0 * 60.0 + 28.113)
        );
        assert_eq!(timestamp("frame_000123.png"), None);
        assert_eq!(timestamp("a_1_02_03_4"), Some(3723.004));
    }

    #[test]
    fn percentiles_follow_numpy() {
        let mut values = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        assert!((percentile(&mut values, 0.5) - 0.2).abs() < 1e-9);
        assert!((percentile(&mut values, 50.0) - 20.0).abs() < 1e-9);
        assert!((percentile(&mut values, 99.5) - 39.8).abs() < 1e-9);
    }

    #[test]
    fn npy_stack_loads_as_grey() {
        let dir = temp_dir("npy");
        let path = dir.join("stack.npy");
        let array = crate::data::sessions::NpyArray {
            shape: vec![2, 3],
            data: NpyData::Float(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        std::fs::write(&path, crate::data::sessions::encode_npy(&array)).unwrap();
        let gray = read_gray(&path).unwrap();
        assert_eq!((gray.width, gray.height), (3, 2));
        assert_eq!(gray.pixels, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let (frames, _) = read_frames(&path, None, 1).unwrap();
        assert_eq!(frames.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn letterbox_centres_and_reports_scale() {
        let gray = GrayF::from_pixels(8, 4, vec![100.0; 32]).unwrap();
        let (boxed, scale, pad) = letterbox_gray(&gray, (16, 16)).unwrap();
        assert_eq!((boxed.width, boxed.height), (16, 16));
        assert!((scale - 2.0).abs() < 1e-12);
        assert_eq!(pad, [0.0, 4.0]); // a 16x8 result centred in a 16x16 canvas
        assert_eq!(boxed.pixels[3 * 16], 0.0);
        assert_eq!(boxed.pixels[4 * 16], 100.0);
        assert_eq!(boxed.pixels[11 * 16 + 15], 100.0);
        assert_eq!(boxed.pixels[12 * 16], 0.0);
    }

    #[test]
    fn tensor_is_planar_rgb_in_unit_range() {
        let bgr = Bgr::filled(1, 1, [255, 128, 0]);
        assert_eq!(to_tensor(&bgr).unwrap(), vec![0.0, 128.0 / 255.0, 1.0]);
    }

    #[test]
    fn bgr8_replicates_grey() {
        let gray = GrayF::from_pixels(2, 1, vec![10.0, 250.4]).unwrap();
        assert_eq!(to_bgr8(&gray).pixels, vec![10, 10, 10, 250, 250, 250]);
    }
}
