//! The benchmark harnesses of `src/animalpose/eval/benchmark.py`: scoring on held-out real COCO
//! sets ([`eval_coco`]), the temporal and behavior networks on held-out simulations
//! ([`eval_temporal`], [`eval_behavior`]), and the report (`cli/evaluate/benchmark.py:to_markdown`,
//! [`to_markdown`]).
//!
//! `// differs from Python:`
//! * Simulated windows are drawn from [`crate::data::synthetic::Rng`], not numpy's PCG64, so the
//!   temporal and behavior numbers come from different (identically distributed) held-out windows
//!   than Python's at the same seeds.
//! * `eval_detection`, `eval_pose`, `eval_tracking`, `eval_multiview` and `throughput` score
//!   *rendered* frames (`data/synthetic.py:render`), which this crate does not draw; the benchmark
//!   command reports them as unsupported. [`eval_coco`] is the same pose/detection scoring on
//!   images that already exist, and is what `animalpose eval real` runs.

use crate::algos::decode::{
    pose_decode, pose_decode_pairs, HeatMaps, DEFAULT_PAIR_MIN_PEAK, DEFAULT_PAIR_RADIUS,
};
use crate::algos::geometry::{apply_affine, box_xyxy_to_cs, crop_affine, invert_affine, Box4};
use crate::algos::image::Bgr;
use crate::cli::predict::Pipeline;
use crate::data::images::to_tensor;
use crate::data::measure::{read_bgr, warp_bgr};
use crate::data::synthetic::{BILATERAL_PAIRS, NUM_KEYPOINTS};
use crate::data::pyjson::{loads, PyVal};
use crate::data::synthetic::{
    behavior_batch, temporal_batch, Corruption, Rng, SimulationPool, CANONICAL_NAMES, MILD_CORRUPTION,
};
use crate::eval::metrics::{auroc, average_precision, f1_scores, linear_interpolate, pck, rmse};
use crate::models::Model;
use crate::tensor::{Ctx, Graph};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// `eval/benchmark.py:SEED`: evaluation seeds start far from the training ones.
pub const SEED: u64 = 10_000;

/// One image of a COCO keypoint set in canonical order, `data/datasets.py:Annotation`.
pub struct Annotation {
    /// The 8-bit BGR image.
    pub image: Bgr,
    /// xyxy boxes (rounded through `float32` as numpy stores them).
    pub boxes: Vec<Box4>,
    /// `(N, K)` canonical keypoints, unlabeled ones at 0.
    pub keypoints: Vec<Vec<[f64; 2]>>,
    /// `(N, K)` state: 2 visible, 1 occluded-labelled, 0 unlabeled.
    pub state: Vec<Vec<i64>>,
    /// Track ids (`-1` when absent).
    pub ids: Vec<i64>,
}

/// `data/datasets.py:CocoKeypoints` with an explicit `{source name: canonical name}` mapping.
pub struct CocoKeypoints {
    root: PathBuf,
    /// `(file_name, annotations)` of every image with a labelled animal, by image id.
    images: Vec<(String, Vec<PyVal>)>,
    /// Keypoint names per category id.
    cats: Vec<(i64, Vec<String>)>,
    mapping: Vec<(String, String)>,
}

impl CocoKeypoints {
    /// `CocoKeypoints(json_path, image_root, mapping)`; `mapping` pairs dataset names with
    /// canonical ones (`"synthetic"`: every canonical name maps to itself).
    pub fn new(json_path: &Path, image_root: &Path, mapping: Vec<(String, String)>) -> Result<Self> {
        let data = loads(
            &std::fs::read_to_string(json_path)
                .with_context(|| format!("cannot read {}", json_path.display()))?,
        )?;
        let list = |k: &str| data.get(k).and_then(PyVal::as_list).map(<[PyVal]>::to_vec).unwrap_or_default();
        let int = |v: Option<&PyVal>| v.and_then(PyVal::as_f64).unwrap_or(0.0) as i64;
        let cats = list("categories")
            .iter()
            .map(|c| {
                let names = c
                    .get("keypoints")
                    .and_then(PyVal::as_list)
                    .unwrap_or_default()
                    .iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect();
                (int(c.get("id")), names)
            })
            .collect();
        let mut by_image: std::collections::BTreeMap<i64, Vec<PyVal>> = std::collections::BTreeMap::new();
        for a in list("annotations") {
            if a.get("num_keypoints").and_then(PyVal::as_f64).unwrap_or(1.0) > 0.0 {
                by_image.entry(int(a.get("image_id"))).or_default().push(a);
            }
        }
        let names: std::collections::BTreeMap<i64, String> = list("images")
            .iter()
            .map(|im| {
                (
                    int(im.get("id")),
                    im.get("file_name").and_then(PyVal::as_str).unwrap_or("").to_string(),
                )
            })
            .collect();
        Ok(Self {
            root: image_root.to_path_buf(),
            images: by_image
                .into_iter()
                .map(|(id, anns)| (names.get(&id).cloned().unwrap_or_default(), anns))
                .collect(),
            cats,
            mapping,
        })
    }

    /// The identity mapping of the `synthetic` vocabulary.
    pub fn synthetic() -> Vec<(String, String)> {
        CANONICAL_NAMES.iter().map(|n| (n.to_string(), n.to_string())).collect()
    }

    /// `len(source)`.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// No labelled image at all.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// `source.sample(index)`.
    pub fn sample(&self, index: usize) -> Result<Annotation> {
        let (name, anns) = &self.images[index % self.images.len()];
        let image = read_bgr(&self.root.join(name))?;
        let k = CANONICAL_NAMES.len();
        let mut out = Annotation {
            image,
            boxes: Vec::new(),
            keypoints: Vec::new(),
            state: Vec::new(),
            ids: Vec::new(),
        };
        for a in anns {
            let cat = a.get("category_id").and_then(PyVal::as_f64).unwrap_or(0.0) as i64;
            let names = self.cats.iter().find(|c| c.0 == cat).map(|c| c.1.clone()).unwrap_or_default();
            let raw: Vec<f32> = a
                .get("keypoints")
                .and_then(PyVal::as_list)
                .unwrap_or_default()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
                .collect();
            let mut kp = vec![[0.0f64; 2]; k];
            let mut st = vec![0i64; k];
            for (j, n) in names.iter().enumerate() {
                let Some((_, c)) = self.mapping.iter().find(|(s, _)| s == n) else {
                    continue;
                };
                let Some(i) = CANONICAL_NAMES.iter().position(|x| x == c) else {
                    continue;
                };
                let (x, y, v) = (raw[3 * j], raw[3 * j + 1], raw[3 * j + 2]);
                let num = |f: f32| if f.is_finite() { f64::from(f) } else { 0.0 };
                kp[i] = [num(x), num(y)];
                let v = if v.is_finite() { v } else { 0.0 };
                st[i] = if v >= 2.0 {
                    2
                } else if v >= 1.0 {
                    1
                } else {
                    0
                };
            }
            let b: Vec<f64> = a
                .get("bbox")
                .and_then(PyVal::as_list)
                .unwrap_or_default()
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();
            let f32r = |v: f64| f64::from(v as f32);
            out.boxes.push([f32r(b[0]), f32r(b[1]), f32r(b[0] + b[2]), f32r(b[1] + b[3])]);
            out.keypoints.push(kp);
            out.state.push(st);
            out.ids.push(a.get("track_id").and_then(PyVal::as_f64).unwrap_or(-1.0) as i64);
        }
        Ok(out)
    }
}

/// The pose half of `pipeline.py:PosePipeline.estimate` on the colour frame: the crop is cut with
/// [`warp_bgr`] (`cv2.warpAffine`) from the BGR image, as Python cuts it, rather than from the grey
/// frame [`crate::cli::predict::Pipeline::estimate`] works on.
pub struct PoseNet {
    model: Model,
    stride: usize,
    /// `PipelineConfig.crop_size`.
    pub crop_size: u32,
    /// `PipelineConfig.crop_padding`.
    pub crop_padding: f64,
    /// `PipelineConfig.pair_decoding`.
    pub pair_decoding: bool,
}

impl PoseNet {
    /// Loads a pose checkpoint.
    pub fn load(path: &Path, crop_size: u32, crop_padding: f64, pair_decoding: bool) -> Result<Self> {
        let (cfg, kind, weights) =
            crate::models::load(path).with_context(|| format!("cannot read {}", path.display()))?;
        anyhow::ensure!(
            kind == crate::models::ModelKind::Pose,
            "{} holds a {kind} network, not a pose network",
            path.display()
        );
        let model = Model::build(kind, &cfg, &weights)?;
        let stride = match &model {
            Model::Pose(m) => m.stride(),
            _ => 1,
        };
        Ok(Self {
            model,
            stride,
            crop_size,
            crop_padding,
            pair_decoding,
        })
    }

    /// `estimate(image, [box])`: keypoints in image pixels and their confidences.
    pub fn estimate(&self, image: &Bgr, b: Box4) -> Result<(Vec<[f64; 2]>, Vec<f64>)> {
        let (center, side) = box_xyxy_to_cs(b, 1.0, self.crop_padding);
        let m = crop_affine(center, side, self.crop_size, 0.0);
        let crop = warp_bgr(image, m, self.crop_size);
        let size = self.crop_size as usize;
        let mut g = Graph::new();
        let input = g.from_data(to_tensor(&crop)?, &[1, 3, size, size]);
        let mut ctx = Ctx::eval();
        let out = self.model.forward_named(&mut ctx, &mut g, &[input])?;
        let get = |name: &str| -> Result<Vec<f32>> {
            let id = out
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, id)| *id)
                .with_context(|| format!("the pose network returned no `{name}` output"))?;
            Ok(g.to_vec(id))
        };
        let (heat, offset) = (get("heat")?, get("offset")?);
        let cells = size / self.stride;
        let maps = HeatMaps::new(&heat, &offset, NUM_KEYPOINTS, cells, cells)?;
        let stride = self.stride as f64;
        let decoded = if self.pair_decoding {
            pose_decode_pairs(&maps, stride, &BILATERAL_PAIRS, DEFAULT_PAIR_RADIUS, DEFAULT_PAIR_MIN_PEAK)
        } else {
            pose_decode(&maps, stride)
        };
        let back = invert_affine(m);
        Ok((
            decoded.keypoints.iter().map(|p| apply_affine(back, *p)).collect(),
            decoded.conf,
        ))
    }
}

/// `np.linspace(0, n - 1, k).astype(int)`.
fn spread(n: usize, k: usize) -> Vec<usize> {
    crate::cli::audit::linspace_int(0.0, n.saturating_sub(1) as f64, k)
}

/// `eval_coco(pipe, source, max_images, detect)`: pose with ground-truth boxes (PCK, RMSE, median
/// error, per-keypoint PCK), detection AP, and whether confidence separates visible from labelled
/// occluded points (AUROC).
pub fn eval_coco(
    pipe: &Pipeline,
    pose: &PoseNet,
    source: &CocoKeypoints,
    max_images: usize,
    detect: bool,
) -> Result<PyVal> {
    let n = source.len().min(max_images);
    let (mut preds, mut gts, mut vis, mut norms) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut vis_conf, mut occl_conf) = (Vec::new(), Vec::new());
    let (mut det_b, mut det_s, mut det_g) = (Vec::new(), Vec::new(), Vec::new());
    for i in spread(source.len(), n) {
        let ann = source.sample(i)?;
        if ann.boxes.is_empty() {
            continue;
        }
        for (a, b) in ann.boxes.iter().enumerate() {
            let (keypoints, conf) = pose.estimate(&ann.image, *b)?;
            for (k, p) in keypoints.iter().enumerate() {
                preds.push(*p);
                gts.push(ann.keypoints[a][k]);
                let st = ann.state[a][k];
                vis.push(st == 2);
                if st == 2 {
                    vis_conf.push(conf[k]);
                } else if st == 1 {
                    occl_conf.push(conf[k]);
                }
            }
            norms.push((b[2] - b[0]).max(b[3] - b[1]));
        }
        if detect {
            let (b, s) = pipe.detect(&ann.image, ann.image.width, ann.image.height)?;
            det_b.push(b);
            det_s.push(s);
            det_g.push(ann.boxes.clone());
        }
    }
    let k = CANONICAL_NAMES.len();
    let d: Vec<f64> = preds
        .iter()
        .zip(&gts)
        .map(|(p, g)| (p[0] - g[0]).hypot(p[1] - g[1]))
        .collect();
    let mut per_kp = PyVal::dict();
    for kk in 0..k {
        let rows: Vec<usize> = (0..norms.len()).filter(|r| vis[r * k + kk]).collect();
        if rows.len() >= 20 {
            let hits = rows.iter().filter(|r| d[*r * k + kk] <= 0.1 * norms[**r]).count();
            per_kp.set(CANONICAL_NAMES[kk], hits as f64 / rows.len() as f64);
        }
    }
    let dv: Vec<f64> = d.iter().zip(&vis).filter(|(_, v)| **v).map(|(x, _)| *x).collect();
    let mut out = PyVal::dict()
        .with("n_instances", norms.len())
        .with("pck@0.1", pck(&preds, &gts, &vis, &norms, 0.1))
        .with("pck@0.05", pck(&preds, &gts, &vis, &norms, 0.05))
        .with("rmse_px", rmse(&preds, &gts, &vis))
        .with("median_err_px", crate::data::measure::median(&dv))
        .with("median_box_px", crate::data::measure::median(&norms))
        .with("per_keypoint_pck@0.1", per_kp);
    if !occl_conf.is_empty() {
        out.set("conf_auroc_visible_vs_occluded", auroc(&vis_conf, &occl_conf));
    }
    if detect {
        out.set("det_ap50", average_precision(&det_b, &det_s, &det_g, 0.5));
        out.set("det_ap75", average_precision(&det_b, &det_s, &det_g, 0.75));
    }
    Ok(out)
}

fn species() -> Vec<String> {
    vec!["mouse".into(), "rat".into()]
}

fn rms(a: &[[f64; 3]], b: &[[f64; 3]], dims: usize) -> f64 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(p, q)| (0..dims).map(|d| (p[d] - q[d]).powi(2)).sum::<f64>())
        .sum();
    (s / a.len().max(1) as f64).sqrt()
}

/// `eval_temporal(model, dims, batches, window)`: the LSTM against raw observations and against
/// linear interpolation over gaps, on held-out simulated windows.
pub fn eval_temporal(model: &Model, dims: usize, batches: usize, window: usize) -> Result<PyVal> {
    let mut rng = Rng::new(SEED + 3);
    let pool = SimulationPool::new(16, 400, &species(), 3, &mut Rng::new(SEED + 3))?;
    let k = CANONICAL_NAMES.len();
    let (mut raw_o, mut sm_o, mut gt_o) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lin_g, mut sm_g, mut gt_g) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..batches {
        let b = temporal_batch(16, window, dims, &pool, &mut rng, &Corruption::default());
        let mut g = Graph::new();
        let x = g.from_data(b.x.clone(), &[16, window, k, dims]);
        let c = g.from_data(b.conf.clone(), &[16, window, k]);
        let o = g.from_data(b.observed.clone(), &[16, window, k]);
        let mut ctx = Ctx::eval();
        let outs = model.forward_named(&mut ctx, &mut g, &[x, c, o])?;
        let coords = outs
            .iter()
            .find(|(n, _)| *n == "coords")
            .map(|(_, id)| *id)
            .context("the temporal model has no coords output")?;
        let out = g.to_vec(coords);
        let pt = |v: &[f32], i: usize| -> [f64; 3] {
            let mut p = [0.0; 3];
            for d in 0..dims {
                p[d] = f64::from(v[i * dims + d]);
            }
            p
        };
        for bi in 0..16 {
            let base = bi * window * k;
            let obs: Vec<bool> = (0..window * k).map(|i| b.observed[base + i] > 0.0).collect();
            let xs: Vec<f64> = (0..window * k * dims).map(|i| f64::from(b.x[base * dims + i])).collect();
            let lin = linear_interpolate(&xs, &obs, window, k, dims);
            for i in 0..window * k {
                let gi = base + i;
                if obs[i] {
                    raw_o.push(pt(&b.x, gi));
                    sm_o.push(pt(&out, gi));
                    gt_o.push(pt(&b.target, gi));
                } else if (0..dims).all(|d| lin[i * dims + d].is_finite()) {
                    let mut l = [0.0; 3];
                    l[..dims].copy_from_slice(&lin[i * dims..(i + 1) * dims]);
                    lin_g.push(l);
                    sm_g.push(pt(&out, gi));
                    gt_g.push(pt(&b.target, gi));
                }
            }
        }
    }
    let unit = if dims == 2 { "px" } else { "cm" };
    Ok(PyVal::dict()
        .with(format!("observed_raw_rmse_{unit}"), rms(&raw_o, &gt_o, dims))
        .with(format!("observed_smoothed_rmse_{unit}"), rms(&sm_o, &gt_o, dims))
        .with(format!("gap_linear_interp_rmse_{unit}"), rms(&lin_g, &gt_g, dims))
        .with(format!("gap_lstm_rmse_{unit}"), rms(&sm_g, &gt_g, dims))
        .with(
            "note",
            "gap metrics only over gaps that linear interpolation can fill (both ends observed or held)",
        ))
}

/// `eval_behavior(model, dims, batches, window)`: frame-wise F1 of the classifier on held-out
/// simulated windows.
pub fn eval_behavior(model: &Model, classes: &[String], dims: usize, batches: usize, window: usize) -> Result<PyVal> {
    let mut rng = Rng::new(SEED + 4);
    let pool = SimulationPool::new(24, 600, &species(), 3, &mut Rng::new(SEED + 4))?;
    let k = CANONICAL_NAMES.len();
    let (mut preds, mut gts) = (Vec::new(), Vec::new());
    for _ in 0..batches {
        let b = behavior_batch(16, window, dims, &pool, &mut rng, &MILD_CORRUPTION);
        let mut g = Graph::new();
        let x = g.from_data(b.x.clone(), &[16, window, k, dims]);
        let m = g.from_data(b.mask.clone(), &[16, window, k]);
        let mut ctx = Ctx::eval();
        let outs = model.forward_named(&mut ctx, &mut g, &[x, m])?;
        let logits = g.to_vec(outs[0].1);
        let c = classes.len();
        for (row, y) in logits.chunks_exact(c).zip(&b.labels) {
            let mut best = 0;
            for j in 1..c {
                if row[j] > row[best] {
                    best = j;
                }
            }
            preds.push(best);
            gts.push(usize::try_from(*y).unwrap_or(0));
        }
    }
    let r = f1_scores(&preds, &gts, classes.len());
    Ok(PyVal::dict()
        .with("macro_f1", r.macro_f1)
        .with("accuracy", r.accuracy)
        .with("per_class_f1", r.per_class_f1)
        .with(
            "confusion",
            PyVal::List(r.confusion.iter().map(|row| PyVal::from(row.iter().map(|v| PyVal::Int(*v)).collect::<Vec<_>>())).collect()),
        )
        .with("classes", classes.to_vec()))
}

/// `_fmt(v)`: three decimals for floats, `str` otherwise.
fn fmt(v: Option<&PyVal>) -> String {
    match v {
        Some(PyVal::Float(x)) => py_float(*x, 3),
        Some(PyVal::Int(i)) => i.to_string(),
        Some(PyVal::Str(s)) => s.clone(),
        Some(PyVal::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(PyVal::None) | None => "None".into(),
        Some(other) => other.dumps(),
    }
}

/// `f"{x:.{digits}f}"`, with Python's `nan` / `inf` spellings.
pub fn py_float(x: f64, digits: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.into()
    } else {
        format!("{x:.digits$}")
    }
}

fn items(v: Option<&PyVal>) -> Vec<(String, PyVal)> {
    match v {
        Some(PyVal::Dict(d)) => d.clone(),
        _ => Vec::new(),
    }
}

/// `cli/evaluate/benchmark.py:to_markdown(r)`.
pub fn to_markdown(r: &PyVal) -> String {
    let mut lines: Vec<String> = [
        "# Synthetic benchmark",
        "",
        "Held-out procedural rodent scenes (seeds disjoint from training). These numbers show the",
        "components work together; they are not real-video accuracy.",
        "",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if let Some(det) = r.get("detection") {
        lines.extend(["## Detection", "", "| view | AP50 | AP75 |", "|---|---|---|"].map(String::from));
        for (k, v) in items(Some(det)) {
            lines.push(format!("| {k} | {} | {} |", fmt(v.get("ap50")), fmt(v.get("ap75"))));
        }
    }
    if let Some(pose) = r.get("pose") {
        lines.extend(
            [
                "",
                "## Pose (GT boxes, visible keypoints)",
                "",
                "| view | species | PCK@0.1 | PCK@0.05 | RMSE px | conf visible / hidden |",
                "|---|---|---|---|---|---|",
            ]
            .map(String::from),
        );
        for (view, d) in items(Some(pose)) {
            for (sp, m) in items(Some(&d)) {
                lines.push(format!(
                    "| {view} | {sp} | {} | {} | {} | {} / {} |",
                    fmt(m.get("pck@0.1")),
                    fmt(m.get("pck@0.05")),
                    fmt(m.get("rmse_px")),
                    fmt(m.get("conf_visible")),
                    fmt(m.get("conf_hidden"))
                ));
            }
        }
    }
    for (key, title) in [("temporal2d", "Temporal LSTM, 2D"), ("temporal3d", "Temporal LSTM, 3D")] {
        if let Some(t) = r.get(key) {
            lines.extend(["".into(), format!("## {title}"), "".into(), "| metric | value |".into(), "|---|---|".into()]);
            for (k, v) in items(Some(t)) {
                if k != "note" {
                    lines.push(format!("| {k} | {} |", fmt(Some(&v))));
                }
            }
        }
    }
    for (key, title) in [("behavior2d", "Behavior, 2D random views"), ("behavior3d", "Behavior, 3D")] {
        if let Some(m) = r.get(key) {
            lines.extend([
                "".into(),
                format!("## {title}"),
                "".into(),
                format!("macro-F1 {}, accuracy {}", fmt(m.get("macro_f1")), fmt(m.get("accuracy"))),
                "".into(),
                "| class | F1 |".into(),
                "|---|---|".into(),
            ]);
            let classes = m.get("classes").and_then(PyVal::as_list).unwrap_or_default();
            let f1 = m.get("per_class_f1").and_then(PyVal::as_list).unwrap_or_default();
            for (c, f) in classes.iter().zip(f1) {
                lines.push(format!("| {} | {} |", fmt(Some(c)), fmt(Some(f))));
            }
        }
    }
    for (key, v) in items(Some(r)) {
        if !key.starts_with("tracking") {
            continue;
        }
        let mut title = key.replace('_', " ");
        if let Some(first) = title.get(..1) {
            title = first.to_uppercase() + &title[1..].to_lowercase();
        }
        lines.extend([
            "".into(),
            format!("## {title} (full detect+pose+track, 3 mice, 300 frames)"),
            "".into(),
            "| view | MOTA | IDF1 | ID switches | predicted ids (gt 3) |".into(),
            "|---|---|---|---|---|".into(),
        ]);
        for (k, m) in items(Some(&v)) {
            lines.push(format!(
                "| {k} | {} | {} | {} | {} |",
                fmt(m.get("mota")),
                fmt(m.get("idf1")),
                fmt(m.get("id_switches")),
                fmt(m.get("n_pred_ids"))
            ));
        }
    }
    if let Some(mv) = r.get("multiview") {
        lines.extend(
            [
                "",
                "## Multi-camera (3 views, 2 rats, full pipeline + fusion + triangulation)",
                "",
                "| metric | value |",
                "|---|---|",
            ]
            .map(String::from),
        );
        for (k, v) in items(Some(mv)) {
            lines.push(format!("| {k} | {} |", fmt(Some(&v))));
        }
    }
    if let Some(tp) = r.get("throughput").and_then(PyVal::as_list) {
        lines.extend(
            [
                "",
                "## Throughput (320x256 frames)",
                "",
                "| device | detector ms/frame | pose ms/animal | detect+pose+track FPS |",
                "|---|---|---|---|",
            ]
            .map(String::from),
        );
        for t in tp {
            lines.push(format!(
                "| {} | {} | {} | {} |",
                fmt(t.get("device")),
                fmt(t.get("detector_ms_per_frame")),
                fmt(t.get("pose_ms_per_animal")),
                fmt(t.get("detect_pose_track_fps"))
            ));
        }
    }
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_matches_the_python_report() {
        let r = PyVal::dict()
            .with(
                "temporal2d",
                PyVal::dict()
                    .with("observed_raw_rmse_px", 1.23456)
                    .with("gap_lstm_rmse_px", 2.0)
                    .with("note", "x"),
            )
            .with(
                "behavior2d",
                PyVal::dict()
                    .with("macro_f1", 0.5)
                    .with("accuracy", 0.75)
                    .with("per_class_f1", vec![0.25, f64::NAN])
                    .with("classes", vec!["resting", "walking"]),
            )
            .with(
                "tracking_model_embedding",
                PyVal::dict().with(
                    "top",
                    PyVal::dict()
                        .with("mota", 0.9)
                        .with("idf1", 0.8)
                        .with("id_switches", 1.5)
                        .with("n_pred_ids", 3.0),
                ),
            );
        // python: print(to_markdown(r)) for the same dict
        let want = "# Synthetic benchmark\n\nHeld-out procedural rodent scenes (seeds disjoint from training). These numbers show the\ncomponents work together; they are not real-video accuracy.\n\n\n## Temporal LSTM, 2D\n\n| metric | value |\n|---|---|\n| observed_raw_rmse_px | 1.235 |\n| gap_lstm_rmse_px | 2.000 |\n\n## Behavior, 2D random views\n\nmacro-F1 0.500, accuracy 0.750\n\n| class | F1 |\n|---|---|\n| resting | 0.250 |\n| walking | nan |\n\n## Tracking model embedding (full detect+pose+track, 3 mice, 300 frames)\n\n| view | MOTA | IDF1 | ID switches | predicted ids (gt 3) |\n|---|---|---|---|---|\n| top | 0.900 | 0.800 | 1.500 | 3.000 |\n";
        assert_eq!(to_markdown(&r), want);
    }
}
