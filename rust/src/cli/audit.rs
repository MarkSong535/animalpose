//! `animalpose data audit`: measure what a recording does to the image and write a camera profile
//! from it — the port of `src/animalpose/cli/data/audit.py`, plus the recording sources of
//! `src/animalpose/frames.py` (`Source`, `find_sources`, `is_held_out`) it and `animalpose eval
//! unlabeled` read.
//!
//! The measurements (animal size from the moving blob, edge transition width per time block, motion
//! smear, noise, contrast, sharpness) go into a dict printed as JSON; `--out` writes the YAML
//! camera profile `data.camera_profile` reads, with the PSF radius calibrated against clean
//! training crops (`--surrogate`) degraded with candidate radii.
//!
//! `// differs from Python:`
//! * Image folders are read here in colour (PNG/JPEG through the `image` crate); any other source
//!   goes through [`crate::data::images::read_frames`], which hands back grey frames, so a video
//!   is reported as `colour: false` and its frames are replicated into three equal channels.
//! * `write_yaml` prints `None` where Python formats a missing frame rate or animal size with
//!   `:.2f` (Python raises `TypeError` there).
//! * Boxes are carried in `f64`; Python's `largest_moving_blob` returns `float32`, so sizes can
//!   differ in the seventh significant digit (the printed values are rounded well above that).

use crate::algos::background::{arena_roi, median_background};
use crate::algos::blob::{largest_moving_blob, BlobOptions};
use crate::algos::geometry::Box4;
use crate::algos::image::{Bgr, GrayF, Image};
use crate::data::images::{estimate_fps, read_frames};
use crate::data::measure::{
    animal_patch, bgr_to_gray, crc32, degrade_defocus, edge_width, laplacian, median, motion_smear,
    noise_sigma, percentile, read_bgr, variance, warp_bgr, PATCH,
};
use crate::data::np_random::NpRng;
use crate::data::pyjson::{float_key, float_repr, py_round, yaml_dump, PyVal, FLOAT_KEY};
use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};

/// `audit.py:BOX_PAD`: `largest_moving_blob` pads a box by 15 % of its longest side on each side.
pub const BOX_PAD: f64 = 1.3;
/// `audit.py:PSF_GRID`: the candidate defocus radii.
pub const PSF_GRID: [f64; 11] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0];
/// `audit.py:SURROGATE_SETS`: `(json, images, species)` of the prepared datasets crops come from.
pub const SURROGATE_SETS: [(&str, &str, &str); 2] = [
    ("crim13/val.json", "crim13/images", "mouse"),
    ("mirror_mouse/side_val.json", "mirror_mouse/images", "mouse"),
];

/// `frames.py:IMAGE_EXT`.
pub const IMAGE_EXT: [&str; 6] = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"];
/// `frames.py:VIDEO_EXT`.
pub const VIDEO_EXT: [&str; 8] = [
    ".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv",
];

fn ext(path: &Path) -> String {
    path.extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

/// `str(Path(p))`: Python drops `.` components and a trailing separator.
pub fn py_path(path: &Path) -> String {
    let parts: PathBuf = path
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect();
    if parts.as_os_str().is_empty() {
        ".".into()
    } else {
        parts.display().to_string()
    }
}

/// One recording, `frames.py:Source`: a folder of image frames or a video file.
pub struct Source {
    /// The folder or file.
    pub path: PathBuf,
    files: Option<Vec<PathBuf>>,
    decoded: std::cell::OnceCell<(Vec<GrayF>, Option<f64>)>,
}

impl Source {
    /// `open_source(path)`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
                .with_context(|| format!("cannot list {}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| IMAGE_EXT.contains(&ext(p).as_str()))
                .collect();
            files.sort();
            Ok(Self::images(path, files))
        } else {
            Ok(Self::video(path))
        }
    }

    fn images(path: PathBuf, files: Vec<PathBuf>) -> Self {
        Self {
            path,
            files: Some(files),
            decoded: std::cell::OnceCell::new(),
        }
    }

    fn video(path: PathBuf) -> Self {
        Self {
            path,
            files: None,
            decoded: std::cell::OnceCell::new(),
        }
    }

    /// The image files of a frame folder (`None` for a video).
    pub fn files(&self) -> Option<&[PathBuf]> {
        self.files.as_deref()
    }

    fn decoded(&self) -> &(Vec<GrayF>, Option<f64>) {
        self.decoded
            .get_or_init(|| match read_frames(&self.path, None, 1) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("{}: {err:#}", self.path.display());
                    (Vec::new(), None)
                }
            })
    }

    /// `Source.n_frames`.
    pub fn n_frames(&self) -> usize {
        match &self.files {
            Some(files) => files.len(),
            None => self.decoded().0.len(),
        }
    }

    /// `Source.frames(indices)`: 8-bit BGR frames at those indices, unreadable ones skipped.
    pub fn frames(&self, indices: impl IntoIterator<Item = usize>) -> Vec<(usize, Bgr)> {
        let mut out = Vec::new();
        match &self.files {
            Some(files) => {
                for i in indices {
                    if let Some(img) = files.get(i).and_then(|f| read_bgr(f).ok()) {
                        out.push((i, img));
                    }
                }
            }
            None => {
                let frames = &self.decoded().0;
                let mut idx: Vec<usize> = indices.into_iter().collect();
                idx.sort_unstable();
                for i in idx {
                    if let Some(g) = frames.get(i) {
                        out.push((i, crate::data::images::to_bgr8(g)));
                    }
                }
            }
        }
        out
    }

    /// `source_fps(path)`: file-name timestamps for a folder, the container rate for a video.
    pub fn fps(&self) -> Option<f64> {
        match &self.files {
            Some(files) => estimate_fps(files),
            None => self.decoded().1,
        }
    }
}

/// `find_sources(root)`: every folder that directly holds image files and every video file under
/// `root`, in the order of `sorted(os.walk(root))`.
pub fn find_sources(root: impl AsRef<Path>) -> Vec<Source> {
    let mut dirs = Vec::new();
    walk(root.as_ref(), &mut dirs);
    dirs.sort_by(|a, b| a.0.to_string_lossy().cmp(&b.0.to_string_lossy()));
    let mut out = Vec::new();
    for (dir, mut names) in dirs {
        names.sort();
        let images: Vec<PathBuf> = names
            .iter()
            .filter(|n| IMAGE_EXT.contains(&ext(Path::new(n)).as_str()))
            .map(|n| dir.join(n))
            .collect();
        if !images.is_empty() {
            out.push(Source::images(dir.clone(), images));
        }
        for n in &names {
            if VIDEO_EXT.contains(&ext(Path::new(n)).as_str()) {
                out.push(Source::video(dir.join(n)));
            }
        }
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<(PathBuf, Vec<String>)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files = Vec::new();
    let mut subdirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else {
            files.push(e.file_name().to_string_lossy().into_owned());
        }
    }
    out.push((dir.to_path_buf(), files));
    for d in subdirs {
        walk(&d, out);
    }
}

/// `frames.py:is_held_out`: about 1 in `every` recordings, by the CRC32 of the path relative to
/// `root`.
pub fn is_held_out(source: &Source, root: &Path, every: u32) -> bool {
    let abs = |p: &Path| -> PathBuf {
        let p = if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_default().join(p)
        };
        p.components().collect()
    };
    let (src, base) = (abs(&source.path), abs(root));
    let rel = src
        .strip_prefix(&base)
        .map(py_path)
        .unwrap_or_else(|_| py_path(&src));
    crc32(rel.as_bytes()) % every == 0
}

/// `np.linspace(start, stop, num).astype(int)`.
pub fn linspace_int(start: f64, stop: f64, num: usize) -> Vec<usize> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start as usize];
    }
    let step = (stop - start) / (num - 1) as f64;
    (0..num)
        .map(|i| {
            if i == num - 1 {
                stop as usize
            } else {
                (i as f64 * step + start) as usize
            }
        })
        .collect()
}

/// `np.array_split(np.arange(n), parts)`.
pub fn array_split(n: usize, parts: usize) -> Vec<std::ops::Range<usize>> {
    let (q, r) = (n / parts, n % parts);
    let mut out = Vec::with_capacity(parts);
    let mut at = 0;
    for i in 0..parts {
        let len = q + usize::from(i < r);
        out.push(at..at + len);
        at += len;
    }
    out
}

fn to_image(g: &GrayF) -> Image {
    Image {
        width: g.width,
        height: g.height,
        pixels: g.pixels.iter().map(|v| *v as u8).collect(),
    }
}

fn blob(g: &Image, bg: &GrayF, diff: f64, roi: &Image) -> Option<Box4> {
    let opts = BlobOptions {
        roi: Some(roi.clone()),
        ..BlobOptions::default()
    };
    largest_moving_blob(g, bg, diff as f32, &opts).0
}

fn q(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        percentile(v, p)
    }
}

/// `audit._displacement`: blob displacement per frame, on consecutive frames in `windows`
/// windows spread over the recording.
fn displacement(
    src: &Source,
    bg: &GrayF,
    roi: &Image,
    n: usize,
    windows: usize,
    window: usize,
    diff: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    for w in 0..windows {
        let s0 = ((w as f64 + 0.5) * n as f64 / windows.max(1) as f64) as usize;
        let frames = src.frames(s0..(s0 + window).min(n));
        let cents: Vec<Option<[f64; 2]>> = frames
            .iter()
            .map(|(_, f)| {
                blob(&to_image(&bgr_to_gray(f)), bg, diff, roi)
                    .map(|b| [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
            })
            .collect();
        for pair in cents.windows(2) {
            if let (Some(a), Some(b)) = (pair[0], pair[1]) {
                out.push((b[0] - a[0]).hypot(b[1] - a[1]));
            }
        }
    }
    out
}

fn stats(v: &[f64], keys: [&str; 3], ps: [f64; 3]) -> PyVal {
    let mut d = PyVal::dict();
    for (k, p) in keys.iter().zip(ps) {
        d.set(*k, q(v, p));
    }
    d
}

/// `audit(src, n_frames, exposure, windows, window, diff)`: the measurement dict of one recording
/// (`{"source", "error": "too few frames"}` below four frames).
pub fn audit(
    src: &Source,
    n_frames: usize,
    exposure: f64,
    windows: usize,
    window: usize,
    diff: f64,
) -> PyVal {
    let source = py_path(&src.path);
    let n = src.n_frames();
    if n < 4 {
        return PyVal::dict()
            .with("source", source)
            .with("error", "too few frames");
    }
    let idx = linspace_int(0.0, (n - 1) as f64, n_frames.min(n));
    let frames: Vec<Bgr> = src.frames(idx).into_iter().map(|(_, f)| f).collect();
    if frames.is_empty() {
        return PyVal::dict()
            .with("source", source)
            .with("error", "too few frames");
    }
    let gs: Vec<GrayF> = frames.iter().map(bgr_to_gray).collect();
    let g8: Vec<Image> = gs.iter().map(to_image).collect();
    let bg = median_background(&g8, g8.len());
    let roi = arena_roi(&bg, 15);
    let boxes: Vec<Option<Box4>> = g8.iter().map(|g| blob(g, &bg, diff, &roi)).collect();
    let (mut sizes, mut noise, mut lap) = (Vec::new(), Vec::new(), Vec::new());
    for (g, b) in gs.iter().zip(&boxes) {
        lap.push(variance(&laplacian(g)));
        noise.push(noise_sigma(g, Some(&roi)));
        if let Some(b) = b {
            sizes.push((b[2] - b[0]).max(b[3] - b[1]) / BOX_PAD);
        }
    }
    let (mut edge, mut spread) = (Vec::new(), Vec::new());
    for part in array_split(frames.len(), 4) {
        let sel: Vec<usize> = part.filter(|i| boxes[*i].is_some()).collect();
        if sel.len() < 3 {
            continue;
        }
        let mut pts = Vec::new();
        let mut scales = Vec::new();
        for i in sel {
            let (p, k) = animal_patch(&frames[i], boxes[i].expect("selected"), PATCH);
            pts.push(p);
            scales.push(k);
        }
        let (w, s) = edge_width(&pts, 2, 95.0);
        let k = scales.iter().sum::<f64>() / scales.len() as f64;
        edge.push(w / k);
        spread.push(s / k);
    }
    let disp = displacement(src, &bg, &roi, n, windows, window, diff);
    let smear = if disp.is_empty() {
        0.0
    } else {
        motion_smear(&disp, exposure)
    };
    let body = (!sizes.is_empty()).then(|| median(&sizes));
    let fps = src.fps();
    let bgv: Vec<f64> = bg.pixels.iter().map(|v| f64::from(*v)).collect();
    let g0: Vec<f64> = gs[0].pixels.iter().map(|v| f64::from(*v)).collect();
    let colour = frames[0].pixels.chunks_exact(3).any(|p| p[0] != p[2]);
    PyVal::dict()
        .with("source", source)
        .with("frames", n)
        .with("fps", PyVal::opt_f64(fps))
        .with(
            "duration_s",
            PyVal::opt_f64(fps.filter(|f| *f != 0.0).map(|f| py_round(n as f64 / f, 1))),
        )
        .with("colour", colour)
        .with(
            "background_grey",
            vec![q(&bgv, 1.0), q(&bgv, 50.0), q(&bgv, 99.0)],
        )
        .with("frame_contrast", vec![q(&g0, 1.0), q(&g0, 99.0)])
        .with(
            "sharpness_lapvar",
            stats(&lap, ["p10", "median", "p90"], [10.0, 50.0, 90.0]),
        )
        .with(
            "noise_sigma",
            stats(&noise, ["p10", "median", "p90"], [10.0, 50.0, 90.0]),
        )
        .with(
            "animal_px",
            PyVal::dict()
                .with("p10", q(&sizes, 10.0))
                .with("median", PyVal::opt_f64(body))
                .with("p90", q(&sizes, 90.0)),
        )
        .with(
            "frames_with_blob",
            boxes.iter().filter(|b| b.is_some()).count(),
        )
        .with(
            "displacement_px",
            PyVal::dict()
                .with("median", q(&disp, 50.0))
                .with("p90", q(&disp, 90.0))
                .with("windows", windows * window),
        )
        .with(
            "edge_width_px",
            PyVal::dict()
                .with("median", py_round(q(&edge, 50.0), 2))
                .with("p10", py_round(q(&edge, 10.0), 2))
                .with("p90", py_round(q(&edge, 90.0), 2))
                .with("silhouette_spread", py_round(q(&spread, 50.0), 2))
                .with("blocks", edge.len()),
        )
        .with("exposure_fraction", exposure)
        .with("motion_smear_px", py_round(smear, 2))
        .with(
            "motion_smear_body_frac",
            PyVal::opt_f64(body.filter(|b| *b != 0.0).map(|b| py_round(smear / b, 3))),
        )
}

/// One prepared COCO keypoint set, the part of `data/datasets.py:CocoKeypoints` a crop needs.
struct CocoSet {
    root: PathBuf,
    /// `(file_name, boxes (xyxy, float32), track ids)` per image with labelled animals, by id.
    images: Vec<(String, Vec<[f32; 4]>, Vec<i64>)>,
}

impl CocoSet {
    fn load(json: &Path, root: &Path) -> Result<Self> {
        let doc: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(json)
                .with_context(|| format!("cannot read {}", json.display()))?,
        )?;
        let empty = Vec::new();
        let mut names = std::collections::BTreeMap::new();
        for im in doc["images"].as_array().unwrap_or(&empty) {
            names.insert(
                im["id"].as_i64().unwrap_or_default(),
                im["file_name"].as_str().unwrap_or_default().to_string(),
            );
        }
        let mut by_image: std::collections::BTreeMap<i64, (Vec<[f32; 4]>, Vec<i64>)> =
            std::collections::BTreeMap::new();
        for a in doc["annotations"].as_array().unwrap_or(&empty) {
            if a.get("num_keypoints").and_then(|v| v.as_i64()).unwrap_or(1) <= 0 {
                continue;
            }
            let b: Vec<f64> = a["bbox"]
                .as_array()
                .unwrap_or(&empty)
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            if b.len() < 4 {
                continue;
            }
            let entry = by_image
                .entry(a["image_id"].as_i64().unwrap_or_default())
                .or_default();
            entry.0.push([
                b[0] as f32,
                b[1] as f32,
                (b[0] + b[2]) as f32,
                (b[1] + b[3]) as f32,
            ]);
            entry
                .1
                .push(a.get("track_id").and_then(|v| v.as_i64()).unwrap_or(-1));
        }
        let images = by_image
            .into_iter()
            .map(|(id, (boxes, ids))| (names.get(&id).cloned().unwrap_or_default(), boxes, ids))
            .collect();
        Ok(Self {
            root: root.to_path_buf(),
            images,
        })
    }

    /// `PoseCropDataset(self, n, crop_size, augment=False)[i]` as an 8-bit BGR crop.
    fn crop(&self, i: usize, crop_size: u32) -> Result<Bgr> {
        let (base, variant) = (i / 4, i % 4);
        let (name, boxes, ids) = &self.images[base % self.images.len()];
        let img = read_bgr(&self.root.join(name))?;
        let mut scene: Vec<i64> = ids.clone();
        scene.sort_unstable();
        scene.dedup();
        let order = NpRng::from_entropy(&[13, base as u128]).permutation(&scene);
        let target = order[(variant / 2).min(order.len() - 1)];
        let j = ids.iter().position(|v| *v == target).unwrap_or(0);
        // `box_xyxy_to_cs` + `crop_affine` on the float32 box, as numpy computes them.
        let [x0, y0, x1, y1] = boxes[j];
        let c = [(x0 + x1) / 2.0, (y0 + y1) / 2.0];
        let (w, h) = ((x1 - x0).max(1.0), (y1 - y0).max(1.0));
        let side = if w > h { w } else { h } * 1.25f32;
        let s = (crop_size as f32 / side) as f64;
        let half = (f64::from(crop_size) - 1.0) / 2.0;
        let (cx, cy) = (f64::from(c[0]), f64::from(c[1]));
        let m = [[s, 0.0, half - s * cx], [0.0, s, half - s * cy]];
        Ok(warp_bgr(&img, m, crop_size))
    }
}

/// `surrogate_crops(root, n, crop_size, sets)`: clean BGR crops from prepared training datasets.
pub fn surrogate_crops(root: &Path, n: usize, crop_size: u32) -> Result<Vec<Bgr>> {
    let mut out = Vec::new();
    for (js, imgs, _species) in SURROGATE_SETS {
        let j = root.join(js);
        if !j.exists() {
            continue;
        }
        let set = CocoSet::load(&j, &root.join(imgs))?;
        if set.images.is_empty() {
            continue;
        }
        for i in 0..n {
            out.push(set.crop(i, crop_size)?);
        }
    }
    Ok(out)
}

/// `calibrate_psf(target_px, crops)`: the PSF radius whose degraded crops measure `target_px` of
/// edge width, and `{"target_edge_width_px", "edge_width_by_psf_px": {radius: width}}`.
pub fn calibrate_psf(target_px: f64, crops: &[Bgr]) -> (f64, PyVal) {
    let mut curve: Vec<(f64, f64)> = Vec::new();
    for r in PSF_GRID {
        let degraded: Vec<GrayF> = crops
            .iter()
            .map(|c| bgr_to_gray(&degrade_defocus(c, r.max(1e-3))))
            .collect();
        curve.push((
            py_round(r, 2),
            py_round(edge_width(&degraded, 2, 95.0).0, 3),
        ));
    }
    let mut best = curve[0];
    for c in &curve {
        if (c.1 - target_px).abs() < (best.1 - target_px).abs() {
            best = *c;
        }
    }
    let table = PyVal::Dict(
        curve
            .iter()
            .map(|(r, w)| (float_key(*r), PyVal::Float(*w)))
            .collect(),
    );
    (
        best.0,
        PyVal::dict()
            .with("target_edge_width_px", py_round(target_px, 3))
            .with("edge_width_by_psf_px", table),
    )
}

fn get_f64(m: &PyVal, a: &str, b: &str) -> Option<f64> {
    m.get(a).and_then(|v| v.get(b)).and_then(PyVal::as_f64)
}

/// Python's `round(x)` (no digits): half to even, an int.
fn round_int(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `profile_from(m, psf_px)`: a training profile (`Profile.to_dict()`, tuples as lists) from the
/// measurements, plus notes on what is assumed.
pub fn profile_from(m: &PyVal, psf_px: Option<f64>) -> (PyVal, Vec<String>) {
    let mut notes = Vec::new();
    let body = get_f64(m, "animal_px", "median").filter(|b| *b != 0.0);
    let edge = get_f64(m, "edge_width_px", "median").unwrap_or(0.0);
    let spread = get_f64(m, "edge_width_px", "silhouette_spread").unwrap_or(0.0);
    let body_frac = m.get("motion_smear_body_frac").and_then(PyVal::as_f64);
    let motion_hi = match (body, body_frac) {
        (Some(_), Some(frac)) => f64::min(0.2, py_round((frac * 1.5).max(0.0), 3)),
        _ => {
            notes.push("no animal found: motion_frac kept at the default 0.2".to_string());
            0.2
        }
    };
    let psf_px = match psf_px {
        Some(p) => p,
        None => {
            notes.push(
                "psf_px not calibrated: rerun with --calibrate (it matches training crops to this \
                 edge width); the uncalibrated width/1.4 is used"
                    .to_string(),
            );
            py_round(edge / 1.4, 2)
        }
    };
    let p90 = get_f64(m, "edge_width_px", "p90").unwrap_or(0.0);
    let colour = matches!(m.get("colour"), Some(PyVal::Bool(true)));
    let contrast = m
        .get("frame_contrast")
        .and_then(PyVal::as_list)
        .map(|c| c[1].as_f64().unwrap_or(0.0) - c[0].as_f64().unwrap_or(0.0))
        .unwrap_or(0.0);
    let n10 = get_f64(m, "noise_sigma", "p10").unwrap_or(0.0);
    let n90 = get_f64(m, "noise_sigma", "p90").unwrap_or(0.0);
    let downscale = match body {
        Some(b) => {
            // `(max(8.0, round(0.8 * body)), round(1.15 * body))`: `round` gives ints, `max`
            // keeps the float 8.0 on a tie.
            let lo = round_int(0.8 * b);
            let lo = if (lo as f64) > 8.0 {
                PyVal::Int(lo)
            } else {
                PyVal::Float(8.0)
            };
            PyVal::List(vec![lo, PyVal::Int(round_int(1.15 * b))])
        }
        None => vec![16.0, 48.0].into(),
    };
    let profile = PyVal::dict()
        .with(
            "psf_px",
            if psf_px > 0.0 {
                PyVal::Float(psf_px)
            } else {
                PyVal::None
            },
        )
        .with("psf_jitter", (p90 / edge.max(1e-6) - 1.0).clamp(0.05, 0.6))
        .with("defocus_frac", vec![0.0, 0.06])
        .with("motion_frac", vec![0.0, motion_hi])
        .with("downscale_to", downscale)
        .with("mono", if colour { 0.0 } else { 1.0 })
        .with("low_contrast", if contrast > 150.0 { 0.0 } else { 1.0 })
        .with("noise", vec![py_round(n10, 1), py_round(n90.max(1.0), 1)])
        .with("jpeg", vec![40.0, 80.0]);
    if spread > 0.5 && spread > 0.4 * edge {
        notes.push(format!(
            "the edge width varies by {spread:.1} px around the silhouette (mean {edge:.1} px): part of \
             it is the animal's own shape, so treat psf_px as an upper bound"
        ));
    }
    let smear = m
        .get("motion_smear_px")
        .and_then(PyVal::as_f64)
        .unwrap_or(0.0);
    if smear > 0.5 * edge {
        notes.push(format!(
            "motion smear ({smear:.1} px) is comparable to the edge width: raise motion_frac if the \
             recording is at a low frame rate"
        ));
    }
    notes.push(
        "jpeg quality is assumed: compressed sources (h264/jpg) are not measurable per frame"
            .into(),
    );
    (profile, notes)
}

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "None".to_string(), |x| format!("{x:.digits$}"))
}

/// `write_yaml(path, m, p, notes, calibration)`: the comment header, then `measured:` (the
/// measurements without `source`, plus `psf_calibration`) and the profile keys.
pub fn yaml_text(
    m: &PyVal,
    profile: &PyVal,
    notes: &[String],
    calibration: Option<&PyVal>,
) -> String {
    let duration = m
        .get("duration_s")
        .and_then(PyVal::as_f64)
        .filter(|d| *d != 0.0);
    let dur = duration.map_or(String::new(), |d| format!(", {d:.0} s"));
    let colour = matches!(m.get("colour"), Some(PyVal::Bool(true)));
    let psf = match profile.get("psf_px") {
        Some(PyVal::Float(v)) => float_repr(*v),
        _ => "None".into(),
    };
    let jitter = profile
        .get("psf_jitter")
        .and_then(PyVal::as_f64)
        .unwrap_or(0.0);
    let mut head = vec![
        format!(
            "# animalpose data audit on {} ({} frames, {} fps{dur}, {}).",
            m.get("source").and_then(PyVal::as_str).unwrap_or(""),
            m.get("frames").and_then(PyVal::as_f64).unwrap_or(0.0),
            fmt_opt(m.get("fps").and_then(PyVal::as_f64), 2),
            if colour { "colour" } else { "greyscale" }
        ),
        "# Feed to training with data.camera_profile=<this file>; see animalpose.data.degrade.Profile."
            .to_string(),
        format!(
            "# Animal ~{} px, edge width {:.1} px, PSF {psf} px (+-{:.0} %), motion smear {:.1} px, \
             noise sigma {:.1}.",
            fmt_opt(get_f64(m, "animal_px", "median"), 0),
            get_f64(m, "edge_width_px", "median").unwrap_or(0.0),
            100.0 * jitter,
            m.get("motion_smear_px").and_then(PyVal::as_f64).unwrap_or(0.0),
            get_f64(m, "noise_sigma", "median").unwrap_or(0.0),
        ),
    ];
    head.extend(notes.iter().map(|n| format!("# assumed: {n}")));
    let mut measured = match m {
        PyVal::Dict(items) => PyVal::Dict(
            items
                .iter()
                .filter(|(k, _)| k != "source")
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    };
    if let Some(c) = calibration {
        measured.set("psf_calibration", c.clone());
    }
    let mut body = PyVal::dict().with("measured", measured);
    if let PyVal::Dict(items) = profile {
        for (k, v) in items {
            body.set(k.clone(), v.clone());
        }
    }
    head.join("\n") + "\n" + &yaml_dump(&body)
}

/// The `audit` subcommand's arguments.
pub fn command() -> Command {
    Command::new("audit")
        .about("Measure what a recording does to the image, and write a camera profile from it.")
        .arg(
            Arg::new("paths")
                .num_args(0..)
                .help("video files or frame folders"),
        )
        .arg(
            Arg::new("root")
                .long("root")
                .help("audit every recording under this folder"),
        )
        .arg(num("frames", "64", "frames sampled per recording"))
        .arg(num("windows", "6", "windows for per-frame displacement"))
        .arg(num("window", "30", "consecutive frames per window"))
        .arg(
            Arg::new("exposure")
                .long("exposure")
                .default_value("0.5")
                .value_parser(clap::value_parser!(f64))
                .help("exposure as a fraction of the frame interval"),
        )
        .arg(
            Arg::new("diff")
                .long("diff")
                .default_value("25.0")
                .value_parser(clap::value_parser!(f64))
                .help("grey levels that count as moving pixels"),
        )
        .arg(
            Arg::new("no_calibrate")
                .long("no-calibrate")
                .action(ArgAction::SetTrue)
                .help("skip matching PSF candidates to training crops"),
        )
        .arg(
            Arg::new("surrogate")
                .long("surrogate")
                .default_value("data/processed")
                .help("prepared datasets to calibrate the PSF on"),
        )
        .arg(num(
            "crops",
            "96",
            "clean crops used for the PSF calibration",
        ))
        .arg(num("crop_size", "192", "crop side in pixels").long("crop-size"))
        .arg(
            Arg::new("out")
                .long("out")
                .help("write this camera profile YAML (single recording)"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .help("write the full measurements here"),
        )
}

fn num(name: &'static str, default: &'static str, help: &'static str) -> Arg {
    Arg::new(name)
        .long(name)
        .default_value(default)
        .value_parser(clap::value_parser!(usize))
        .help(help)
}

/// `audit.run(args)`.
pub fn run(m: &ArgMatches) -> Result<()> {
    let mut sources: Vec<Source> = m
        .get_many::<String>("paths")
        .into_iter()
        .flatten()
        .map(Source::open)
        .collect::<Result<_>>()?;
    if let Some(root) = m.get_one::<String>("root") {
        sources.extend(find_sources(root));
    }
    if sources.is_empty() {
        bail!("give a path or --root");
    }
    let usize_arg = |k: &str| m.get_one::<usize>(k).copied().unwrap_or_default();
    let f64_arg = |k: &str| m.get_one::<f64>(k).copied().unwrap_or_default();
    let surrogate = m
        .get_one::<String>("surrogate")
        .map_or("data/processed", String::as_str);
    let crops = if m.get_flag("no_calibrate") {
        None
    } else {
        Some(surrogate_crops(
            Path::new(surrogate),
            usize_arg("crops"),
            usize_arg("crop_size") as u32,
        )?)
    };
    if crops.as_ref().is_some_and(Vec::is_empty) {
        println!("no surrogate crops under {surrogate}: PSF stays uncalibrated");
    }
    let mut all = Vec::new();
    let single = sources.len() == 1;
    for src in &sources {
        let mut meas = audit(
            src,
            usize_arg("frames"),
            f64_arg("exposure"),
            usize_arg("windows"),
            usize_arg("window"),
            f64_arg("diff"),
        );
        if let Some(err) = meas.get("error") {
            println!(
                "{} {}",
                meas.get("source").and_then(PyVal::as_str).unwrap_or(""),
                err.as_str().unwrap_or("")
            );
            all.push(meas);
            continue;
        }
        let target = get_f64(&meas, "edge_width_px", "median").unwrap_or(0.0);
        let (mut calibration, mut psf) = (None, None);
        if let Some(crops) = crops.as_ref().filter(|c| !c.is_empty()) {
            let (best, cal) = calibrate_psf(target, crops);
            let clean = cal
                .get("edge_width_by_psf_px")
                .and_then(|c| c.get(&format!("{FLOAT_KEY}0.0")))
                .and_then(PyVal::as_f64)
                .unwrap_or(0.0);
            if let Some(e) = meas.get_mut("edge_width_px") {
                e.set("clean_crop_edge_width_px", py_round(clean, 2));
            }
            psf = Some(best);
            calibration = Some(cal);
        }
        let (profile, notes) = profile_from(&meas, psf);
        let mut first = PyVal::dict();
        for k in [
            "source",
            "fps",
            "animal_px",
            "edge_width_px",
            "motion_smear_px",
            "noise_sigma",
            "sharpness_lapvar",
        ] {
            first.set(k, meas.get(k).cloned().unwrap_or(PyVal::None));
        }
        println!("{}", first.dumps());
        let shown = match &profile {
            PyVal::Dict(items) => PyVal::Dict(
                items
                    .iter()
                    .filter(|(k, _)| k != "defocus_frac")
                    .cloned()
                    .collect(),
            ),
            other => other.clone(),
        };
        println!("  profile: {}", shown.dumps());
        if let Some(cal) = &calibration {
            println!(
                "  psf calibration: {}",
                cal.get("edge_width_by_psf_px")
                    .cloned()
                    .unwrap_or(PyVal::None)
                    .dumps()
            );
        }
        for n in &notes {
            println!("  note: {n}");
        }
        if let (Some(out), true) = (m.get_one::<String>("out"), single) {
            std::fs::write(
                out,
                yaml_text(&meas, &profile, &notes, calibration.as_ref()),
            )
            .with_context(|| format!("cannot write {out}"))?;
            println!("  -> {out}");
        }
        all.push(meas);
    }
    if let Some(js) = m.get_one::<String>("json") {
        std::fs::write(js, PyVal::List(all).dumps_indent(1))
            .with_context(|| format!("cannot write {js}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_match_numpy_and_zlib() {
        assert_eq!(linspace_int(0.0, 59.0, 24)[..5], [0, 2, 5, 7, 10]);
        assert_eq!(*linspace_int(0.0, 59.0, 24).last().unwrap(), 59);
        let parts: Vec<usize> = array_split(10, 4).iter().map(|r| r.len()).collect();
        assert_eq!(parts, [3, 3, 2, 2]);
        assert_eq!(crc32(b"session_a/frame.png"), 0x8d13_719a);
        assert_eq!(py_path(Path::new("./a/b/")), "a/b");
    }

    #[test]
    fn profile_from_a_measurement_dict() {
        let m = PyVal::dict()
            .with("source", "rec")
            .with("frames", 60usize)
            .with("fps", 30.0)
            .with("duration_s", 2.0)
            .with("colour", true)
            .with("frame_contrast", vec![40.0, 180.0])
            .with(
                "noise_sigma",
                PyVal::dict()
                    .with("p10", 0.42)
                    .with("median", 0.6)
                    .with("p90", 0.75),
            )
            .with(
                "animal_px",
                PyVal::dict()
                    .with("p10", 30.0)
                    .with("median", 33.75)
                    .with("p90", 36.0),
            )
            .with(
                "edge_width_px",
                PyVal::dict()
                    .with("median", 5.1)
                    .with("p10", 4.6)
                    .with("p90", 5.9)
                    .with("silhouette_spread", 0.3)
                    .with("blocks", 4usize),
            )
            .with("motion_smear_px", 1.2)
            .with("motion_smear_body_frac", 0.036);
        let (p, notes) = profile_from(&m, None);
        // python: profile_from(m, None)[0].to_dict() with the same dict
        assert_eq!(
            p.dumps(),
            r#"{"psf_px": 3.64, "psf_jitter": 0.15686274509803932, "defocus_frac": [0.0, 0.06], "motion_frac": [0.0, 0.054], "downscale_to": [27, 39], "mono": 0.0, "low_contrast": 1.0, "noise": [0.4, 1.0], "jpeg": [40.0, 80.0]}"#
        );
        assert_eq!(notes.len(), 2);
        assert!(notes[0].starts_with("psf_px not calibrated"));
        let blind = m
            .clone()
            .with(
                "animal_px",
                PyVal::dict()
                    .with("p10", 0.0)
                    .with("median", PyVal::None)
                    .with("p90", 0.0),
            )
            .with("motion_smear_body_frac", PyVal::None);
        let (p2, notes2) = profile_from(&blind, Some(2.0));
        assert_eq!(p2.get("motion_frac").unwrap().dumps(), "[0.0, 0.2]");
        assert_eq!(p2.get("downscale_to").unwrap().dumps(), "[16.0, 48.0]");
        assert!(notes2.iter().any(|n| n.contains("no animal found")));
        assert!(!notes2.iter().any(|n| n.contains("psf_px not calibrated")));
        // `write_yaml` of the Python package on the same dict, byte for byte.
        let want = r#"# animalpose data audit on rec (60 frames, 30.00 fps, 2 s, colour).
# Feed to training with data.camera_profile=<this file>; see animalpose.data.degrade.Profile.
# Animal ~34 px, edge width 5.1 px, PSF 3.64 px (+-16 %), motion smear 1.2 px, noise sigma 0.6.
# assumed: psf_px not calibrated: rerun with --calibrate (it matches training crops to this edge width); the uncalibrated width/1.4 is used
# assumed: jpeg quality is assumed: compressed sources (h264/jpg) are not measurable per frame
measured:
  frames: 60
  fps: 30.0
  duration_s: 2.0
  colour: true
  frame_contrast:
  - 40.0
  - 180.0
  noise_sigma:
    p10: 0.42
    median: 0.6
    p90: 0.75
  animal_px:
    p10: 30.0
    median: 33.75
    p90: 36.0
  edge_width_px:
    median: 5.1
    p10: 4.6
    p90: 5.9
    silhouette_spread: 0.3
    blocks: 4
  motion_smear_px: 1.2
  motion_smear_body_frac: 0.036
psf_px: 3.64
psf_jitter: 0.15686274509803932
defocus_frac:
- 0.0
- 0.06
motion_frac:
- 0.0
- 0.054
downscale_to:
- 27
- 39
mono: 0.0
low_contrast: 1.0
noise:
- 0.4
- 1.0
jpeg:
- 40.0
- 80.0
"#;
        assert_eq!(yaml_text(&m, &p, &notes, None), want);
    }
}
