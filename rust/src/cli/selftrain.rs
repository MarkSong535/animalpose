//! `animalpose label`: pseudo-labels for an unlabeled collection of recordings, the port of
//! `src/animalpose/selftrain.py` (`LabelOptions`, `label_videos` and its acceptance rules).
//!
//! Every video file and every folder of image frames under `root` is one recording; about 1 in 5
//! (by the CRC-32 of its relative path, [`crate::cli::sources::is_held_out`]) is held out. Frames
//! are sampled evenly from the others and labeled by the teacher (detector + pose net), keeping
//! only what the teacher is confident and self-consistent about:
//!
//! * detections with score >= `hi` after the static-background filter; a frame with an uncertain
//!   detection (`lo <= score < hi`) is skipped whole;
//! * keypoints with confidence >= `kp_conf` that the mirrored frame reproduces within `flip_tol`
//!   box widths, on moving pixels when there is a background; animals need `min_kps` of them;
//! * with `boxes = blob`, the box is the largest dark moving blob, and a frame whose labeled nose
//!   is closer to the visible tail than the labeled tail base (a head/tail flip) is dropped.
//!
//! Writes `<out>/train.json` (COCO, canonical keypoint names), `<out>/images/<source id>/*.jpg`
//! and `<out>/split.json`, the layout `selftrain.py` writes, and returns the run statistics.
//!
//! `// differs from Python:` frames are grey levels (see [`crate::cli::sources`]), so the JPEGs are
//! written single channel (`cv2.imread` still returns three identical channels); keypoints are
//! averaged in `f64` where the Python teacher returns `float32`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::algos::background::{arena_roi, foreground_fraction, median_background};
use crate::algos::blob::{blob_box, dilate_rect, morph_open, signed_diff, BlobOptions};
use crate::algos::geometry::Box4;
use crate::algos::image::{GrayF, Image};
use crate::cli::pyjson::Json;
use crate::cli::sources::{abspath, find_sources, is_held_out, linspace_int, source_id, Source};
use crate::data::images::u8_from_gray;
use crate::data::synthetic::{index, CANONICAL_NAMES, FLIP_INDEX};

/// `selftrain.py:LabelOptions`: the teacher and the acceptance thresholds, same defaults.
#[derive(Clone, Debug)]
pub struct LabelOptions {
    /// The teacher detector checkpoint.
    pub detector: String,
    /// The teacher pose checkpoint.
    pub pose: String,
    /// Where `train.json`, `split.json` and `images/` go.
    pub out: PathBuf,
    /// Detector input size `(H, W)`.
    pub det_size: (u32, u32),
    /// Pose crop size.
    pub crop_size: u32,
    /// Frames sampled per recording.
    pub per_source: usize,
    /// Detection score that counts as an animal.
    pub hi: f64,
    /// Below this the frame is background; between `lo` and `hi` it is skipped.
    pub lo: f64,
    /// Keypoint confidence to keep.
    pub kp_conf: f64,
    /// Keypoint flip agreement, in box widths.
    pub flip_tol: f64,
    /// Keypoints an animal needs to be labeled.
    pub min_kps: usize,
    /// Frames needed to use the background filter.
    pub min_seq: usize,
    /// `detector` or `blob`.
    pub boxes: String,
    /// Fraction of each recording to label.
    pub time_range: (f64, f64),
    /// Hold out about 1 in 5 recordings.
    pub holdout: bool,
    /// Only recordings whose path contains one of these.
    pub include: Option<Vec<String>>,
    /// Print per-recording progress.
    pub progress: bool,
    /// `PipelineConfig.foreground_diff` of the teacher pipeline.
    pub foreground_diff: f32,
    /// `PipelineConfig.min_foreground` of the teacher pipeline.
    pub min_foreground: f64,
}

impl Default for LabelOptions {
    fn default() -> Self {
        Self {
            detector: "runs/detector_real.pt".into(),
            pose: "runs/pose_real_r18_side.pt".into(),
            out: PathBuf::from("data/processed/unlabeled_pseudo"),
            det_size: (480, 640),
            crop_size: 192,
            per_source: 200,
            hi: 0.5,
            lo: 0.2,
            kp_conf: 0.5,
            flip_tol: 0.05,
            min_kps: 4,
            min_seq: 30,
            boxes: "detector".into(),
            time_range: (0.0, 1.0),
            holdout: true,
            include: None,
            progress: true,
            foreground_diff: 25.0,
            min_foreground: 0.08,
        }
    }
}

/// Keypoints `(N, K)` and confidences `(N, K)` for `N` boxes.
pub type Poses = (Vec<Vec<[f64; 2]>>, Vec<Vec<f64>>);

/// What `label_videos` needs from `PosePipeline`: `detect` and `estimate`.
pub trait Teacher {
    /// Boxes (xyxy, image pixels) and their scores.
    fn detect(&self, frame: &GrayF) -> Result<(Vec<Box4>, Vec<f64>)>;
    /// Keypoints and confidences for each box.
    fn estimate(&self, frame: &GrayF, boxes: &[Box4]) -> Result<Poses>;
}

impl Teacher for crate::cli::predict::Pipeline {
    fn detect(&self, frame: &GrayF) -> Result<(Vec<Box4>, Vec<f64>)> {
        let bgr = crate::data::images::to_bgr8(frame);
        crate::cli::predict::Pipeline::detect(self, &bgr, frame.width, frame.height)
    }

    fn estimate(&self, frame: &GrayF, boxes: &[Box4]) -> Result<Poses> {
        let mut kps = Vec::with_capacity(boxes.len());
        let mut confs = Vec::with_capacity(boxes.len());
        for b in boxes {
            let e = crate::cli::predict::Pipeline::estimate(self, frame, *b)?;
            kps.push(e.keypoints);
            confs.push(e.conf);
        }
        Ok((kps, confs))
    }
}

/// The teacher pipeline of `label_videos`: `PipelineConfig(crop_size, det_size, det_thresh=lo)`.
pub fn load_teacher(opts: &LabelOptions) -> Result<crate::cli::predict::Pipeline> {
    let yaml = format!(
        "checkpoints:\n  detector: {}\n  pose: {}\npipeline:\n  det_size: [{}, {}]\n  crop_size: {}\n  det_thresh: {}\n",
        yaml_str(&opts.detector),
        yaml_str(&opts.pose),
        opts.det_size.0,
        opts.det_size.1,
        opts.crop_size,
        opts.lo
    );
    let cfg = crate::data::config::Config::parse(&yaml, &[])?;
    crate::cli::predict::Pipeline::from_config(&cfg)
}

fn yaml_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// The counters `label_videos` returns, in the order Python's dict holds them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LabelStats {
    /// Recordings found (after `--include`).
    pub sources: usize,
    /// Frames offered to the teacher.
    pub frames: usize,
    /// Frames written.
    pub labeled: usize,
    /// Frames with an uncertain detection.
    pub skipped_uncertain: usize,
    /// Frames with no animal (or no plausible blob).
    pub no_animal: usize,
    /// Animals written.
    pub animals: usize,
    /// Keypoints written.
    pub keypoints: usize,
    /// Frames dropped as a head/tail flip.
    pub head_tail_flip: usize,
    /// Images in `train.json` (`CocoWriter.save`), once saved.
    pub images: Option<usize>,
}

impl LabelStats {
    /// The stats dict as Python's `json.dumps` prints it.
    pub fn to_json(&self) -> Json {
        let mut pairs = vec![
            ("sources", Json::Int(self.sources as i64)),
            ("frames", Json::Int(self.frames as i64)),
            ("labeled", Json::Int(self.labeled as i64)),
            (
                "skipped_uncertain",
                Json::Int(self.skipped_uncertain as i64),
            ),
            ("no_animal", Json::Int(self.no_animal as i64)),
            ("animals", Json::Int(self.animals as i64)),
            ("keypoints", Json::Int(self.keypoints as i64)),
            ("head_tail_flip", Json::Int(self.head_tail_flip as i64)),
        ];
        if let Some(images) = self.images {
            pairs.push(("images", Json::Int(images as i64)));
        }
        Json::obj(pairs)
    }
}

/// One labeled animal for [`CocoWriter`].
#[derive(Clone, Debug)]
pub struct Animal {
    /// Canonical keypoints `(K, 2)`.
    pub keypoints: Vec<[f64; 2]>,
    /// COCO visibility per keypoint (0 or 2).
    pub visibility: Vec<i64>,
    /// xyxy box.
    pub bbox: Box4,
}

/// `data/real.py:CocoWriter`: canonical COCO keypoints, one image entry per labeled frame.
#[derive(Default)]
pub struct CocoWriter {
    images: Vec<Json>,
    annotations: Vec<Json>,
}

impl CocoWriter {
    /// `CocoWriter.add`.
    pub fn add(&mut self, file_name: &str, size_hw: (u32, u32), animals: &[Animal]) {
        let img_id = self.images.len() as i64;
        self.images.push(Json::obj([
            ("id", Json::Int(img_id)),
            ("file_name", Json::str(file_name)),
            ("height", Json::Int(i64::from(size_hw.0))),
            ("width", Json::Int(i64::from(size_hw.1))),
        ]));
        for a in animals {
            let mut flat = Vec::with_capacity(3 * a.keypoints.len());
            for (p, v) in a.keypoints.iter().zip(&a.visibility) {
                if *v > 0 && p[0].is_finite() && p[1].is_finite() {
                    flat.extend([Json::Float(p[0]), Json::Float(p[1]), Json::Int(*v)]);
                } else {
                    flat.extend([Json::Float(0.0), Json::Float(0.0), Json::Int(0)]);
                }
            }
            let [x0, y0, x1, y1] = a.bbox;
            self.annotations.push(Json::obj([
                ("id", Json::Int(self.annotations.len() as i64)),
                ("image_id", Json::Int(img_id)),
                ("category_id", Json::Int(1)),
                ("keypoints", Json::Arr(flat)),
                (
                    "num_keypoints",
                    Json::Int(a.visibility.iter().filter(|v| **v > 0).count() as i64),
                ),
                (
                    "bbox",
                    Json::Arr(vec![
                        Json::Float(x0),
                        Json::Float(y0),
                        Json::Float(x1 - x0),
                        Json::Float(y1 - y0),
                    ]),
                ),
                ("area", Json::Float((x1 - x0) * (y1 - y0))),
                ("iscrowd", Json::Int(0)),
                ("track_id", Json::Int(-1)),
            ]));
        }
    }

    /// `CocoWriter.save`: writes the json; returns `(images, animals)`.
    pub fn save(&self, path: &Path) -> Result<(usize, usize)> {
        let coco = Json::obj([
            ("images", Json::Arr(self.images.clone())),
            ("annotations", Json::Arr(self.annotations.clone())),
            (
                "categories",
                Json::Arr(vec![Json::obj([
                    ("id", Json::Int(1)),
                    ("name", Json::str("animal")),
                    (
                        "keypoints",
                        Json::Arr(CANONICAL_NAMES.iter().map(|n| Json::str(*n)).collect()),
                    ),
                    ("skeleton", Json::Arr(vec![])),
                ])]),
            ),
        ]);
        crate::cli::pyjson::write_text(path, &coco.dumps())?;
        Ok((self.images.len(), self.annotations.len()))
    }
}

/// Python's `round` on a float (half to even), as an integer.
fn round_even(v: f64) -> i64 {
    if v.is_finite() {
        v.round_ties_even() as i64
    } else {
        0
    }
}

/// `selftrain.py:foreground_mask`: pixels that differ from the background by more than `diff`,
/// opened with a 3x3 square and dilated by `grow` to cover the outline.
pub fn foreground_mask(img: &GrayF, bg: &GrayF, diff: f32, grow: u32) -> Image {
    let m = img.abs_diff(bg).threshold_gt(diff);
    let m = morph_open(&m, 3);
    dilate_rect(&m, 2 * grow + 1)
}

/// `selftrain.py:fit_to_foreground`: the box shrunk to the foreground pixels inside it, padded by
/// `pad` of its size; `None` when fewer than 20 pixels are foreground.
pub fn fit_to_foreground(bbox: Box4, fg: &Image, pad: f64) -> Option<Box4> {
    let (w, h) = (i64::from(fg.width), i64::from(fg.height));
    let x0 = round_even(bbox[0]).max(0);
    let y0 = round_even(bbox[1]).max(0);
    let x1 = round_even(bbox[2]).min(w);
    let y1 = round_even(bbox[3]).min(h);
    let (mut lx, mut ly, mut hx, mut hy, mut n) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            if fg.pixel(x as u32, y as u32) != 0 {
                lx = lx.min(x);
                ly = ly.min(y);
                hx = hx.max(x);
                hy = hy.max(y);
                n += 1;
            }
        }
    }
    if n < 20 {
        return None;
    }
    let lo = [lx as f64, ly as f64];
    let hi = [(hx + 1) as f64, (hy + 1) as f64];
    let d = [(hi[0] - lo[0]) * pad, (hi[1] - lo[1]) * pad];
    // The Python box is float32.
    let f = |v: f64| f64::from(v as f32);
    Some([
        f((lo[0] - d[0]).max(0.0)),
        f((lo[1] - d[1]).max(0.0)),
        f((hi[0] + d[0]).min(w as f64)),
        f((hi[1] + d[1]).min(h as f64)),
    ])
}

/// `selftrain.py:tail_centroid`: the centre of the thin dark pixels near `bbox` that a body-sized
/// opening removes (the tail), or `None` when fewer than 15 such pixels.
pub fn tail_centroid(
    img: &GrayF,
    bg: &GrayF,
    bbox: Box4,
    diff: f32,
    open_px: u32,
) -> Option<[f64; 2]> {
    let m = signed_diff(bg, img).threshold_gt(diff);
    let body = dilate_rect(&morph_open(&m, open_px), 3);
    let (bw, bh) = (bbox[2] - bbox[0], bbox[3] - bbox[1]);
    let (w, h) = (i64::from(img.width), i64::from(img.height));
    let x0 = ((bbox[0] - bw / 2.0).max(0.0) as i64).min(w);
    let y0 = ((bbox[1] - bh / 2.0).max(0.0) as i64).min(h);
    let x1 = slice_end(bbox[2] + bw / 2.0, w);
    let y1 = slice_end(bbox[3] + bh / 2.0, h);
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * w + x) as usize;
            if m.pixels[i] != 0 && body.pixels[i] == 0 {
                sx += (x - x0) as f64;
                sy += (y - y0) as f64;
                n += 1;
            }
        }
    }
    if n < 15 {
        return None;
    }
    Some([sx / n as f64 + x0 as f64, sy / n as f64 + y0 as f64])
}

/// A numpy slice end `int(v)` clipped to `0..=len` (a negative end counts from the back).
fn slice_end(v: f64, len: i64) -> i64 {
    let e = v as i64;
    if e < 0 {
        (len + e).max(0)
    } else {
        e.min(len)
    }
}

/// `selftrain.py:background`: the median of up to 64 frames spread over the recording.
fn source_background(src: &mut Source, n_frames: usize) -> GrayF {
    let idx = linspace_int(0.0, (n_frames - 1) as f64, n_frames.min(64));
    let frames: Vec<Image> = src
        .frames(&idx)
        .iter()
        .map(|(_, f)| u8_from_gray(f))
        .collect();
    median_background(&frames, 64)
}

fn mirror(img: &GrayF) -> GrayF {
    let w = img.width as usize;
    let mut pixels = Vec::with_capacity(img.pixels.len());
    for row in img.pixels.chunks_exact(w.max(1)) {
        pixels.extend(row.iter().rev());
    }
    GrayF {
        width: img.width,
        height: img.height,
        pixels,
    }
}

/// `selftrain.py:flip_estimate`: pose on the mirrored frame, mapped back to the original frame and
/// left/right labels.
pub fn flip_estimate<T: Teacher + ?Sized>(
    teacher: &T,
    img: &GrayF,
    boxes: &[Box4],
) -> Result<Poses> {
    let w = f64::from(img.width);
    let fb: Vec<Box4> = boxes
        .iter()
        .map(|b| [w - b[2], b[1], w - b[0], b[3]])
        .collect();
    let (kp, cf) = teacher.estimate(&mirror(img), &fb)?;
    let kp = kp
        .into_iter()
        .map(|row| {
            (0..row.len())
                .map(|k| {
                    let p = row[FLIP_INDEX.get(k).copied().unwrap_or(k)];
                    [w - 1.0 - p[0], p[1]]
                })
                .collect()
        })
        .collect();
    let cf = cf
        .into_iter()
        .map(|row| {
            (0..row.len())
                .map(|k| row[FLIP_INDEX.get(k).copied().unwrap_or(k)])
                .collect()
        })
        .collect();
    Ok((kp, cf))
}

/// `(keypoints, ok, confidences)` per box, as `consistent_keypoints` returns them.
pub type Consistent = (Vec<Vec<[f64; 2]>>, Vec<Vec<bool>>, Vec<Vec<f64>>);

/// `selftrain.py:consistent_keypoints`: `(mean of the two estimates, ok, conf)` where `ok` marks
/// keypoints confident on both and in agreement within `flip_tol` box sizes.
pub fn consistent_keypoints<T: Teacher + ?Sized>(
    teacher: &T,
    img: &GrayF,
    boxes: &[Box4],
    kp_conf: f64,
    flip_tol: f64,
) -> Result<Consistent> {
    let (kp, cf) = teacher.estimate(img, boxes)?;
    let (kf, cff) = flip_estimate(teacher, img, boxes)?;
    let mut mean = Vec::with_capacity(boxes.len());
    let mut ok = Vec::with_capacity(boxes.len());
    for (i, b) in boxes.iter().enumerate() {
        let size = (b[2] - b[0]).max(b[3] - b[1]);
        let k = kp[i].len();
        let mut m = Vec::with_capacity(k);
        let mut o = Vec::with_capacity(k);
        for j in 0..k {
            let (p, q) = (kp[i][j], kf[i][j]);
            let dist = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
            o.push(cf[i][j] >= kp_conf && cff[i][j] >= kp_conf && dist <= flip_tol * size);
            m.push([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0]);
        }
        mean.push(m);
        ok.push(o);
    }
    Ok((mean, ok, cf))
}

fn write_jpeg(path: &Path, img: &GrayF) -> Result<()> {
    let u8img = u8_from_gray(img);
    let file =
        std::fs::File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
    let mut enc =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 92);
    enc.encode(
        &u8img.pixels,
        u8img.width,
        u8img.height,
        image::ExtendedColorType::L8,
    )
    .with_context(|| format!("cannot encode {}", path.display()))
}

/// `selftrain.py:label_videos`: label sampled frames of every recording under `root`.
pub fn label_videos<T: Teacher + ?Sized>(
    root: &Path,
    teacher: &T,
    opts: &LabelOptions,
) -> Result<LabelStats> {
    let out = &opts.out;
    std::fs::create_dir_all(out.join("images"))
        .with_context(|| format!("cannot create {}", out.join("images").display()))?;
    let mut sources = find_sources(root);
    if let Some(include) = &opts.include {
        sources.retain(|s| {
            let p = s.path.to_string_lossy();
            include.iter().any(|k| p.contains(k.as_str()))
        });
        if opts.progress {
            println!("{} recordings match the filter", sources.len());
        }
    }
    let blob = opts.boxes == "blob";
    let diff = opts.foreground_diff;
    let (nose, base) = (index("nose"), index("tail_base"));
    let mut train = Vec::new();
    let mut held = Vec::new();
    let mut writer = CocoWriter::default();
    let mut stats = LabelStats {
        sources: sources.len(),
        ..LabelStats::default()
    };
    for src in &mut sources {
        if opts.holdout && is_held_out(&src.rel, 5) {
            held.push(Json::str(src.rel.clone()));
            continue;
        }
        train.push(Json::str(src.rel.clone()));
        let n = src.n_frames();
        if n == 0 {
            continue;
        }
        let bg = (n >= opts.min_seq).then(|| source_background(src, n));
        if blob && bg.is_none() {
            continue;
        }
        let roi = if blob {
            bg.as_ref().map(|b| arena_roi(b, 15))
        } else {
            None
        };
        let sid = source_id(&src.rel);
        let img_dir = out.join("images").join(&sid);
        std::fs::create_dir_all(&img_dir)
            .with_context(|| format!("cannot create {}", img_dir.display()))?;
        let lo_f = (opts.time_range.0 * (n - 1) as f64) as i64;
        let hi_f = (opts.time_range.1 * (n - 1) as f64) as i64;
        let count = (opts.per_source as i64).min(hi_f - lo_f + 1).max(0) as usize;
        let idx = linspace_int(lo_f as f64, hi_f as f64, count);
        let blob_opts = BlobOptions {
            roi: roi.clone(),
            ..BlobOptions::default()
        };
        let mut areas = None;
        if let (true, Some(bg)) = (blob, bg.as_ref()) {
            // the typical blob size from a quick pass, to reject odd ones
            let step = (idx.len() / 50).max(1);
            let pass: Vec<usize> = idx.iter().step_by(step).copied().collect();
            let mut sizes = Vec::new();
            for (_, im) in src.frames(&pass) {
                if let Some(b) = blob_box(&u8_from_gray(&im), bg, diff, &blob_opts) {
                    sizes.push((b[2] - b[0]) * (b[3] - b[1]) / 1.69); // undo the 15 % padding
                }
            }
            if !sizes.is_empty() {
                let med = crate::algos::blob::median(&sizes);
                if med != 0.0 {
                    areas = Some((0.15 * med, 2.5 * med));
                }
            }
        }
        for (t, img) in src.frames(&idx) {
            stats.frames += 1;
            let gray8 = u8_from_gray(&img);
            let (mut boxes, mut scores) = if blob {
                let bg = bg.as_ref().expect("blob boxes need a background");
                let b = blob_box(&gray8, bg, diff, &blob_opts);
                let plausible = |b: &Box4| match areas {
                    Some((lo, hi)) => {
                        let a = (b[2] - b[0]) * (b[3] - b[1]) / 1.69;
                        lo <= a && a <= hi
                    }
                    None => true,
                };
                match b {
                    Some(b) if plausible(&b) => (vec![b], vec![1.0]),
                    _ => {
                        stats.no_animal += 1;
                        continue;
                    }
                }
            } else {
                teacher.detect(&img)?
            };
            if let Some(bg) = bg.as_ref() {
                let keep: Vec<bool> = boxes
                    .iter()
                    .map(|b| foreground_fraction(&gray8, bg, *b, diff) >= opts.min_foreground)
                    .collect();
                let mut it = keep.iter();
                boxes.retain(|_| *it.next().unwrap_or(&false));
                let mut it = keep.iter();
                scores.retain(|_| *it.next().unwrap_or(&false));
            }
            if scores.iter().any(|s| *s >= opts.lo && *s < opts.hi) {
                stats.skipped_uncertain += 1;
                continue;
            }
            let mut boxes: Vec<Box4> = boxes
                .into_iter()
                .zip(&scores)
                .filter(|(_, s)| **s >= opts.hi)
                .map(|(b, _)| b)
                .collect();
            if boxes.is_empty() {
                stats.no_animal += 1;
                continue;
            }
            let fg = bg.as_ref().map(|bg| foreground_mask(&img, bg, diff, 3));
            if let Some(fg) = fg.as_ref() {
                if !blob {
                    let fitted: Option<Vec<Box4>> = boxes
                        .iter()
                        .map(|b| fit_to_foreground(*b, fg, 0.1))
                        .collect();
                    match fitted {
                        Some(f) => boxes = f,
                        None => continue,
                    }
                }
            }
            let (kp, mut ok, _) =
                consistent_keypoints(teacher, &img, &boxes, opts.kp_conf, opts.flip_tol)?;
            if let Some(fg) = fg.as_ref() {
                let (w, h) = (i64::from(fg.width), i64::from(fg.height));
                for (row, oks) in kp.iter().zip(ok.iter_mut()) {
                    for (p, o) in row.iter().zip(oks.iter_mut()) {
                        let x = round_even(p[0]).clamp(0, w - 1);
                        let y = round_even(p[1]).clamp(0, h - 1);
                        *o &= fg.pixel(x as u32, y as u32) != 0;
                    }
                }
            }
            if blob {
                let bg = bg.as_ref().expect("blob boxes need a background");
                let tail = tail_centroid(&img, bg, boxes[0], diff, 5);
                if let Some(tail) = tail {
                    let d =
                        |p: [f64; 2]| ((p[0] - tail[0]).powi(2) + (p[1] - tail[1]).powi(2)).sqrt();
                    let has = ok[0].len() > nose.max(base);
                    if has && ok[0][nose] && ok[0][base] && d(kp[0][base]) >= d(kp[0][nose]) {
                        stats.head_tail_flip += 1;
                        continue;
                    }
                }
            }
            let animals: Vec<Animal> = (0..boxes.len())
                .filter(|i| ok[*i].iter().filter(|o| **o).count() >= opts.min_kps)
                .map(|i| Animal {
                    keypoints: kp[i].clone(),
                    visibility: ok[i].iter().map(|o| if *o { 2 } else { 0 }).collect(),
                    bbox: boxes[i],
                })
                .collect();
            if animals.len() < boxes.len() {
                // an animal we cannot label well: do not teach it as background
                continue;
            }
            let name = format!("{sid}/{t:07}.jpg");
            write_jpeg(&out.join("images").join(&name), &img)?;
            writer.add(&name, (img.height, img.width), &animals);
            stats.labeled += 1;
            stats.animals += animals.len();
            stats.keypoints += animals
                .iter()
                .map(|a| a.visibility.iter().filter(|v| **v > 0).count())
                .sum::<usize>();
        }
        if opts.progress {
            println!("{} {}", src.rel, stats.to_json().dumps());
        }
    }
    let (images, animals) = writer.save(&out.join("train.json"))?;
    stats.images = Some(images);
    stats.animals = animals;
    let split = Json::obj([
        ("root", Json::str(abspath(root))),
        ("train", Json::Arr(train)),
        ("held_out", Json::Arr(held)),
    ]);
    crate::cli::pyjson::write_text(&out.join("split.json"), &split.dumps_indent(1))?;
    Ok(stats)
}

#[cfg(test)]
mod tests;

/// Keypoints of the canonical vocabulary, for the tests' stand-in teacher.
#[cfg(test)]
pub(crate) const K: usize = crate::data::synthetic::NUM_KEYPOINTS;
