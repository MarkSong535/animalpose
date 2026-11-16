//! `animalpose eval <check>`: the scoring harnesses of `src/animalpose/cli/evaluate/`.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `cli/evaluate/unlabeled.py` | [`unlabeled`] |
//! | `cli/evaluate/behavior_sessions.py` | [`behavior_sessions`] |
//! | `cli/evaluate/real.py` | [`real`] |
//! | `cli/evaluate/real_tracking.py` | [`real_tracking`] |
//! | `cli/evaluate/benchmark.py` | [`benchmark`] |
//!
//! `rat7m-3d`, `mirror-paws`, `mirror-videos`, `blur`, `crim13-identity`, `real-tracks` and
//! `behavior-loao` are listed so `--help` shows the whole group, and refuse to run by name.
//!
//! Detection runs through [`crate::cli::predict::Pipeline::detect`] on the colour frame, pose through
//! [`crate::eval::benchmark::PoseNet`] on colour crops, as the Python harnesses do. Checkpoints are
//! read with [`crate::models::load`]. `// differs from Python:` `real-tracking` runs the whole
//! [`crate::cli::predict::Pipeline`], which works on grey frames.

use crate::algos::background::{arena_roi, foreground_fraction, median_background, otsu_threshold};
use crate::algos::blob::{blob_box, BlobOptions, BlobProposer};
use crate::algos::geometry::Box4;
use crate::algos::image::{Bgr, GrayF, Image};
use crate::algos::nms::merge_candidates;
use crate::cli::audit::{find_sources, is_held_out, linspace_int, Source};
use crate::cli::predict::Pipeline;
use crate::data::config::Config;
use crate::data::measure::{bgr_to_gray, median};
use crate::data::pyjson::{loads, py_round, PyVal};
use crate::data::sessions::{load_session_dir, IGNORE};
use crate::data::synthetic::FLIP_INDEX;
use crate::eval::behavior::behavior_report;
use crate::eval::benchmark::{
    eval_behavior, eval_coco, eval_temporal, py_float, to_markdown, CocoKeypoints, PoseNet,
};
use crate::models::{Model, ModelKind};
use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::{Path, PathBuf};

/// Checks with no Rust harness, and the Python module that runs them.
const PYTHON_ONLY: [(&str, &str); 7] = [
    (
        "rat7m-3d",
        "cli/evaluate/rat7m_3d.py (multi-view triangulation on Rat 7M video)",
    ),
    (
        "mirror-paws",
        "cli/evaluate/mirror_paws.py (mirror-mouse paw triangulation)",
    ),
    (
        "mirror-videos",
        "cli/evaluate/mirror_videos.py (mirror-mouse video tracking)",
    ),
    ("blur", "cli/evaluate/blur.py (degraded-image sweeps)"),
    (
        "crim13-identity",
        "cli/evaluate/crim13_identity.py (identity over CRIM13 videos)",
    ),
    (
        "real-tracks",
        "cli/evaluate/real_tracks.py (tracks against CRIM13 labels)",
    ),
    (
        "behavior-loao",
        "cli/evaluate/behavior_loao.py (leave-one-animal-out training)",
    ),
];

/// The `eval` command group.
pub fn command() -> Command {
    let mut cmd = Command::new("eval")
        .about("score a model, with or without ground truth")
        .subcommand_required(true)
        .subcommand(real_command())
        .subcommand(unlabeled_command())
        .subcommand(real_tracking_command())
        .subcommand(behavior_sessions_command())
        .subcommand(benchmark_command());
    for (name, module) in PYTHON_ONLY {
        cmd = cmd.subcommand(
            Command::new(name)
                .about(format!("Python only: {module}"))
                .arg(
                    Arg::new("args")
                        .num_args(0..)
                        .trailing_var_arg(true)
                        .allow_hyphen_values(true),
                ),
        );
    }
    cmd
}

/// Runs `animalpose eval <check>`.
pub fn run(m: &ArgMatches) -> Result<()> {
    match m.subcommand() {
        Some(("real", a)) => real(a),
        Some(("unlabeled", a)) => unlabeled(a),
        Some(("real-tracking", a)) => real_tracking(a),
        Some(("behavior-sessions", a)) => behavior_sessions(a),
        Some(("benchmark", a)) => benchmark(a),
        Some((name, _)) => match PYTHON_ONLY.iter().find(|(n, _)| *n == name) {
            Some((_, module)) => bail!(
                "unsupported: `animalpose eval {name}` is run by src/animalpose/{module}, which \
                 has no Rust counterpart; run it from the Python package"
            ),
            None => bail!("unknown eval check {name:?}"),
        },
        None => bail!("animalpose eval needs a check"),
    }
}

// ---------------------------------------------------------------------------------- helpers

fn s<'a>(m: &'a ArgMatches, k: &str) -> Option<&'a str> {
    m.get_one::<String>(k).map(String::as_str)
}

fn strings(m: &ArgMatches, k: &str) -> Vec<String> {
    m.get_many::<String>(k)
        .into_iter()
        .flatten()
        .cloned()
        .collect()
}

/// `--name` with `_` spelled `-` (clap needs the long name as a `'static` string).
fn long_of(name: &'static str) -> &'static str {
    match name {
        "crop_size" => "crop-size",
        "min_seq" => "min-seq",
        "max_images" => "max-images",
        "n_videos" => "n-videos",
        "min_run" => "min-run",
        "w_app" => "w-app",
        other => other,
    }
}

fn opt_f64(name: &'static str, default: &'static str) -> Arg {
    Arg::new(name)
        .long(long_of(name))
        .default_value(default)
        .value_parser(clap::value_parser!(f64))
}

fn opt_usize(name: &'static str, default: &'static str) -> Arg {
    Arg::new(name)
        .long(long_of(name))
        .default_value(default)
        .value_parser(clap::value_parser!(usize))
}

/// argparse's `BooleanOptionalAction`: `--long` and `--no-long`, the last one given wins.
fn bool_pair(
    name: &'static str,
    long: &'static str,
    no_long: &'static str,
    default: bool,
) -> [Arg; 2] {
    [
        Arg::new(name)
            .long(long)
            .action(ArgAction::SetTrue)
            .overrides_with(no_long),
        Arg::new(no_long)
            .long(no_long)
            .overrides_with(name)
            .action(ArgAction::SetTrue)
            .help(if default {
                "turn off (on by default)"
            } else {
                "turn off"
            }),
    ]
}

fn flag_value(m: &ArgMatches, name: &str, default: bool) -> bool {
    let no = match name {
        "prior" => "no-prior",
        _ => "no-pair-decoding",
    };
    if m.get_flag(name) {
        true
    } else if m.get_flag(no) {
        false
    } else {
        default
    }
}

/// `json.loads(out.read_text()) if out.exists() else {}`.
fn read_results(out: &Path) -> Result<PyVal> {
    if out.exists() {
        loads(&std::fs::read_to_string(out)?)
            .with_context(|| format!("{} is not JSON", out.display()))
    } else {
        Ok(PyVal::dict())
    }
}

fn write_results(out: &Path, v: &PyVal) -> Result<()> {
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, v.dumps_indent(2))
        .with_context(|| format!("cannot write {}", out.display()))
}

/// A [`Pipeline`] from checkpoints plus `pipeline:` / `tracker:` keys (a config built in memory).
fn pipeline(
    detector: &str,
    pose: &str,
    pipeline_keys: serde_json::Value,
    tracker_keys: serde_json::Value,
) -> Result<Pipeline> {
    let doc = serde_json::json!({
        "checkpoints": {"detector": detector, "pose": pose},
        "pipeline": pipeline_keys,
        "tracker": tracker_keys,
    });
    Pipeline::from_config(&Config::parse(&doc.to_string(), &[])?)
}

/// `load_model(path)`: a network and its config from a checkpoint.
fn load_model(path: &str) -> Result<(Model, crate::models::config::ModelConfig, ModelKind)> {
    let (cfg, kind, weights) =
        crate::models::load(Path::new(path)).with_context(|| format!("cannot read {path}"))?;
    let model = Model::build(kind, &cfg, &weights)
        .with_context(|| format!("cannot build the {kind} network of {path}"))?;
    Ok((model, cfg, kind))
}

fn stem(p: &str) -> String {
    Path::new(p)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// --------------------------------------------------------------------------------- unlabeled

fn unlabeled_command() -> Command {
    Command::new("unlabeled")
        .about("Label-free checks on held-out recordings of an unlabeled collection.")
        .arg(Arg::new("root").long("root").required(true))
        .arg(
            Arg::new("pairs")
                .long("pairs")
                .num_args(1..)
                .required(true)
                .help("detector.pt:pose.pt"),
        )
        .arg(
            Arg::new("det_size")
                .long("det-size")
                .num_args(2)
                .value_parser(clap::value_parser!(u32))
                .default_values(["480", "640"]),
        )
        .arg(opt_usize("crop_size", "192"))
        .arg(opt_f64("thresh", "0.3"))
        .arg(opt_usize("runs", "5").help("runs of consecutive frames per recording"))
        .arg(
            Arg::new("run_frames")
                .long("run")
                .value_name("RUN")
                .default_value("20")
                .value_parser(clap::value_parser!(usize))
                .help("frames per run"),
        )
        .arg(opt_usize("min_seq", "30"))
        .arg(
            Arg::new("range")
                .long("range")
                .num_args(2)
                .value_parser(clap::value_parser!(f64))
                .default_values(["0.0", "1.0"]),
        )
        .arg(
            Arg::new("all_sources")
                .long("all-sources")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("candidates")
                .long("candidates")
                .num_args(1..)
                .value_parser(["network", "blob"])
                .default_values(["network"])
                .help("detection candidate sources (blob needs a static background)"),
        )
        .arg(
            Arg::new("blob")
                .long("blob")
                .action(ArgAction::SetTrue)
                .help("also score boxes against the moving-blob box"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .default_value("results/unlabeled.json"),
        )
        .arg(
            Arg::new("include")
                .long("include")
                .num_args(1..)
                .help("only recordings whose path contains one of these strings"),
        )
}

/// `unlabeled._iou`.
fn iou(a: Box4, b: Box4) -> f64 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
}

/// `img[:, ::-1]`.
fn mirror(img: &Bgr) -> Bgr {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in (0..w).rev() {
            pixels.extend_from_slice(&img.pixels[(y * w + x) * 3..(y * w + x) * 3 + 3]);
        }
    }
    Bgr {
        width: img.width,
        height: img.height,
        pixels,
    }
}

/// `(keypoints, agreeing, confidences)` of one box.
type Consistent = (Vec<[f64; 2]>, Vec<bool>, Vec<f64>);

/// `selftrain.consistent_keypoints(pipe, img, boxes, kp_conf, flip_tol)` for one box: the keypoints
/// averaged with those found on the mirrored frame, which of them agree within `flip_tol` of the
/// box size, and the confidences.
fn consistent_keypoints(
    pose: &PoseNet,
    img: &Bgr,
    b: Box4,
    kp_conf: f64,
    flip_tol: f64,
) -> Result<Consistent> {
    let (kps, conf) = pose.estimate(img, b)?;
    let w = f64::from(img.width);
    let (fk, fc) = pose.estimate(&mirror(img), [w - b[2], b[1], w - b[0], b[3]])?;
    let size = (b[2] - b[0]).max(b[3] - b[1]);
    let mut kp = Vec::new();
    let mut ok = Vec::new();
    for (k, p) in kps.iter().enumerate() {
        let j = FLIP_INDEX[k];
        let q = [w - 1.0 - fk[j][0], fk[j][1]];
        ok.push(
            conf[k] >= kp_conf
                && fc[j] >= kp_conf
                && (p[0] - q[0]).hypot(p[1] - q[1]) <= flip_tol * size,
        );
        kp.push([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0]);
    }
    Ok((kp, ok, conf))
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn opt_mean(v: &[f64]) -> PyVal {
    if v.is_empty() {
        PyVal::None
    } else {
        PyVal::Float(mean(v))
    }
}

struct UnlabeledArgs {
    thresh: f64,
    runs: usize,
    run_frames: usize,
    min_seq: usize,
    range: [f64; 2],
    blob: bool,
    blob_candidates: bool,
}

/// `unlabeled.evaluate(pipe, sources, args)`.
fn evaluate_unlabeled(
    pipe: &Pipeline,
    pose: &PoseNet,
    sources: &[&Source],
    a: &UnlabeledArgs,
) -> Result<PyVal> {
    let (mut det, mut agree, mut jit, mut conf, mut ious) =
        (vec![], vec![], vec![], vec![], vec![]);
    let mut per = PyVal::dict();
    let cfg = &pipe.cfg;
    for src in sources {
        let n = src.n_frames();
        if n == 0 {
            continue;
        }
        // `selftrain.background`: the median of 64 frames spread over the recording.
        let bg = (n >= a.min_seq).then(|| {
            let frames: Vec<Image> = src
                .frames(linspace_int(0.0, (n - 1) as f64, 64.min(n)))
                .iter()
                .map(|(_, f)| to_u8(&bgr_to_gray(f)))
                .collect();
            median_background(&frames, 64)
        });
        let roi = if a.blob {
            bg.as_ref().map(|b| arena_roi(b, 15))
        } else {
            None
        };
        let mut prop = match (&bg, a.blob_candidates) {
            (Some(b), true) => {
                let mut p = BlobProposer::new(b.clone())
                    .with_diff(cfg.blob_diff.unwrap_or(cfg.foreground_diff))
                    .with_score(cfg.blob_score as f32)
                    .with_open_px(cfg.blob_open_px)
                    .with_calibrate(cfg.blob_calibrate);
                if let Some(area) = cfg.blob_area {
                    p = p.with_area(area);
                }
                if cfg.blob_arena_roi {
                    p = p.with_roi(arena_roi(b, 15));
                }
                Some(p)
            }
            _ => None,
        };
        let (lo, hi) = (
            (a.range[0] * (n - 1) as f64) as usize,
            (a.range[1] * (n - 1) as f64) as usize,
        );
        let starts = linspace_int(
            lo as f64,
            hi.saturating_sub(a.run_frames).max(lo) as f64,
            a.runs,
        );
        let (mut s_det, mut s_agree, mut s_jit, mut s_iou) = (vec![], vec![], vec![], vec![]);
        for s0 in starts {
            let mut prev: Option<(Vec<[f64; 2]>, Vec<bool>)> = None;
            for (_, img) in src.frames(s0..(s0 + a.run_frames).min(n)) {
                let gray = bgr_to_gray(&img);
                let reference = match (&roi, &bg) {
                    (Some(r), Some(b)) => blob_box(
                        &to_u8(&gray),
                        b,
                        cfg.foreground_diff,
                        &BlobOptions {
                            roi: Some(r.clone()),
                            ..BlobOptions::default()
                        },
                    ),
                    _ => None,
                };
                let (mut boxes, mut scores) = pipe.detect(&img, img.width, img.height)?;
                if let Some(p) = prop.as_mut() {
                    let (eb, es) = p.propose(&to_u8(&gray));
                    let es: Vec<f64> = es.iter().map(|v| f64::from(*v)).collect();
                    (boxes, scores) =
                        merge_candidates(&boxes, &scores, &eb, &es, cfg.merge_iou, cfg.max_det);
                }
                if let Some(b) = &bg {
                    let keep: Vec<bool> = boxes
                        .iter()
                        .map(|bx| {
                            foreground_fraction(&to_u8(&gray), b, *bx, cfg.foreground_diff)
                                >= cfg.min_foreground
                        })
                        .collect();
                    let mut it = keep.iter();
                    boxes.retain(|_| *it.next().unwrap_or(&false));
                    let mut it = keep.iter();
                    scores.retain(|_| *it.next().unwrap_or(&false));
                }
                let first = boxes
                    .iter()
                    .zip(&scores)
                    .find(|(_, s)| **s >= a.thresh)
                    .map(|(b, _)| *b);
                s_det.push(f64::from(u8::from(first.is_some())));
                if let Some(r) = reference {
                    let hit = first.is_some_and(|b| iou(b, r) >= 0.5);
                    ious.push(f64::from(u8::from(hit)));
                    s_iou.push(f64::from(u8::from(hit)));
                }
                let Some(b) = first else {
                    prev = None;
                    continue;
                };
                let (kp, ok, cf) = consistent_keypoints(pose, &img, b, 0.0, 0.05)?;
                let confident: Vec<bool> = cf.iter().map(|c| *c >= 0.5).collect();
                let agreeing: Vec<f64> = ok
                    .iter()
                    .zip(&confident)
                    .filter(|(_, c)| **c)
                    .map(|(o, _)| f64::from(u8::from(*o)))
                    .collect();
                if !agreeing.is_empty() {
                    s_agree.push(mean(&agreeing));
                }
                conf.push(mean(&cf));
                let size = (b[2] - b[0]).max(b[3] - b[1]).max(1.0);
                if let Some((pk, pc)) = &prev {
                    let d: Vec<f64> = (0..kp.len())
                        .filter(|k| confident[*k] && pc[*k])
                        .map(|k| (kp[k][0] - pk[k][0]).hypot(kp[k][1] - pk[k][1]))
                        .collect();
                    if !d.is_empty() {
                        s_jit.push(median(&d) / size);
                    }
                }
                prev = Some((kp, confident));
            }
        }
        det.extend(&s_det);
        agree.extend(&s_agree);
        jit.extend(&s_jit);
        let name = src
            .path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        per.set(
            name,
            PyVal::dict()
                .with(
                    "detected",
                    if s_det.is_empty() { 0.0 } else { mean(&s_det) },
                )
                .with("flip_agreement", opt_mean(&s_agree))
                .with(
                    "jitter",
                    if s_jit.is_empty() {
                        PyVal::None
                    } else {
                        PyVal::Float(median(&s_jit))
                    },
                )
                .with("blob_iou50", opt_mean(&s_iou)),
        );
    }
    Ok(PyVal::dict()
        .with("detected", if det.is_empty() { 0.0 } else { mean(&det) })
        .with("flip_agreement", opt_mean(&agree))
        .with("jitter", opt_mean(&jit))
        .with("conf", opt_mean(&conf))
        .with("frames", det.len())
        .with("blob_iou50", opt_mean(&ious))
        .with("per_source", per))
}

fn to_u8(g: &GrayF) -> Image {
    Image {
        width: g.width,
        height: g.height,
        pixels: g.pixels.iter().map(|v| v.clamp(0.0, 255.0) as u8).collect(),
    }
}

/// `eval unlabeled`: `cli/evaluate/unlabeled.py:run`.
pub fn unlabeled(m: &ArgMatches) -> Result<()> {
    let root = PathBuf::from(s(m, "root").context("--root is required")?);
    let all_sources = m.get_flag("all_sources");
    let include = strings(m, "include");
    let sources = find_sources(&root);
    let held: Vec<&Source> = sources
        .iter()
        .filter(|s| all_sources || is_held_out(s, &root, 5))
        .filter(|s| {
            include.is_empty()
                || include
                    .iter()
                    .any(|k| s.path.to_string_lossy().contains(k.as_str()))
        })
        .collect();
    println!("{} held-out recordings", held.len());
    let out = PathBuf::from(s(m, "out").unwrap_or("results/unlabeled.json"));
    let mut allres = read_results(&out)?;
    let det_size: Vec<u32> = m
        .get_many::<u32>("det_size")
        .into_iter()
        .flatten()
        .copied()
        .collect();
    let candidates = strings(m, "candidates");
    let range: Vec<f64> = m
        .get_many::<f64>("range")
        .into_iter()
        .flatten()
        .copied()
        .collect();
    let args = UnlabeledArgs {
        thresh: *m.get_one::<f64>("thresh").unwrap_or(&0.3),
        runs: *m.get_one::<usize>("runs").unwrap_or(&5),
        run_frames: *m.get_one::<usize>("run_frames").unwrap_or(&20),
        min_seq: *m.get_one::<usize>("min_seq").unwrap_or(&30),
        range: [range[0], range[1]],
        blob: m.get_flag("blob"),
        blob_candidates: candidates.iter().any(|c| c == "blob"),
    };
    for pair in strings(m, "pairs") {
        let (det, pose) = pair
            .split_once(':')
            .with_context(|| format!("{pair}: expected detector:pose"))?;
        if !(Path::new(det).exists() && Path::new(pose).exists()) {
            println!("missing {pair}");
            continue;
        }
        let pipe = pipeline(
            det,
            pose,
            serde_json::json!({
                "crop_size": m.get_one::<usize>("crop_size").copied().unwrap_or(192),
                "det_size": det_size,
                "det_thresh": 0.1,
                "detector_candidates": candidates,
            }),
            serde_json::json!({}),
        )?;
        let crop = m.get_one::<usize>("crop_size").copied().unwrap_or(192) as u32;
        let net = PoseNet::load(
            Path::new(pose),
            crop,
            pipe.cfg.crop_padding,
            pipe.cfg.pair_decoding,
        )?;
        let tag = format!("{}+{}", stem(det), stem(pose));
        let res = evaluate_unlabeled(&pipe, &net, &held, &args)?;
        let shown = match &res {
            PyVal::Dict(items) => PyVal::Dict(
                items
                    .iter()
                    .filter(|(k, _)| k != "per_source")
                    .cloned()
                    .collect(),
            ),
            other => other.clone(),
        };
        println!("{tag} {}", shown.dumps());
        allres.set(tag, res);
        write_results(&out, &allres)?;
    }
    Ok(())
}

// ------------------------------------------------------------------------- behavior-sessions

fn behavior_sessions_command() -> Command {
    let [prior, no_prior] = bool_pair("prior", "prior", "no-prior", true);
    Command::new("behavior-sessions")
        .about("Frame-wise behavior scores of one or more classifiers on labeled session files (*.npz).")
        .arg(Arg::new("model").long("model").num_args(1..).required(true))
        .arg(Arg::new("sessions").long("sessions").required(true))
        .arg(opt_f64("conf", "0.6").help("keypoint confidence above which a point is used"))
        .arg(
            Arg::new("classes")
                .long("classes")
                .num_args(0..)
                .help("score these classes for every model (default: the classes in the labels)"),
        )
        .arg(
            Arg::new("prob_smooth")
                .long("prob-smooth")
                .default_value("1,5,15")
                .help("moving-median windows (frames) to score"),
        )
        .arg(prior)
        .arg(no_prior)
        .arg(Arg::new("out").long("out"))
}

/// `behavior_sessions._rnd`: floats in nested dicts rounded to `nd` digits.
fn rnd(v: &PyVal, nd: usize) -> PyVal {
    match v {
        PyVal::Dict(items) => {
            PyVal::Dict(items.iter().map(|(k, x)| (k.clone(), rnd(x, nd))).collect())
        }
        PyVal::Float(x) => PyVal::Float(py_round(*x, nd)),
        other => other.clone(),
    }
}

/// `eval behavior-sessions`: `cli/evaluate/behavior_sessions.py:run`.
pub fn behavior_sessions(m: &ArgMatches) -> Result<()> {
    let smoothing: Vec<usize> = s(m, "prob_smooth")
        .unwrap_or("1,5,15")
        .split(',')
        .filter(|w| !w.trim().is_empty())
        .map(|w| {
            w.trim()
                .parse::<usize>()
                .with_context(|| format!("bad window {w:?}"))
        })
        .collect::<Result<_>>()?;
    let prior = flag_value(m, "prior", true);
    let conf = *m.get_one::<f64>("conf").unwrap_or(&0.6);
    let wanted: Option<Vec<String>> = m
        .get_many::<String>("classes")
        .map(|v| v.cloned().collect());
    let sessions_dir = s(m, "sessions").context("--sessions is required")?;
    for path in strings(m, "model") {
        let (model, cfg, kind) = load_model(&path)?;
        if kind != ModelKind::Behavior {
            bail!("{path} holds a {kind} network, not a behavior classifier");
        }
        let classes = cfg.classes();
        let class_refs: Vec<&str> = classes.iter().map(String::as_str).collect();
        let sessions = load_session_dir(sessions_dir, 15.0, None, &class_refs)?;
        let labels: Vec<i64> = sessions
            .iter()
            .flat_map(|s| s.labels.iter().copied())
            .collect();
        let mut present: Vec<usize> = labels
            .iter()
            .filter(|l| **l != IGNORE)
            .map(|l| usize::try_from(*l).unwrap_or(0))
            .collect();
        present.sort_unstable();
        present.dedup();
        let shared: Vec<String> = match &wanted {
            Some(w) if !w.is_empty() => w.clone(),
            _ => present.iter().map(|c| classes[*c].clone()).collect(),
        }
        .into_iter()
        .filter(|c| classes.contains(c))
        .collect();
        let rep = behavior_report(
            &model,
            &classes,
            &sessions,
            &[("present".to_string(), shared.clone())],
            &smoothing,
            prior,
            4.0,
            conf,
            None,
        )?;
        let f1_of = |r: &[(String, crate::eval::behavior::ProtocolMetrics)]| r[0].1.macro_f1;
        let mut best = &rep[0];
        for r in &rep {
            if f1_of(&r.1) > f1_of(&best.1) {
                best = r;
            }
        }
        let n = labels.len() as f64;
        let mut share = PyVal::dict();
        for c in &present {
            let count = labels.iter().filter(|l| **l == *c as i64).count() as f64;
            share.set(classes[*c].clone(), py_round(count / n, 3));
        }
        let mut variants = PyVal::dict();
        for (k, r) in &rep {
            variants.set(k.clone(), rnd(&r[0].1.to_py(false), 4));
        }
        let raw = &rep[0].1[0].1;
        let per_class_raw = PyVal::Dict(
            raw.per_class_f1
                .iter()
                .map(|(k, v)| (k.clone(), PyVal::Float(py_round(*v, 3))))
                .collect(),
        );
        let best_pair = PyVal::List(vec![
            best.0.clone().into(),
            py_round(f1_of(&best.1), 4).into(),
        ]);
        let res = PyVal::dict()
            .with("model", path.as_str())
            .with("frames", labels.iter().filter(|l| **l != IGNORE).count())
            .with("classes_present", shared.clone())
            .with("label_share", share)
            .with("variants", variants)
            .with("best", best_pair.clone())
            .with("per_class_f1_raw", per_class_raw.clone());
        println!(
            "{} {}",
            Path::new(&path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            PyVal::dict()
                .with("best", best_pair)
                .with("raw", py_round(raw.macro_f1, 4))
                .with("per_class_f1_raw", per_class_raw)
                .dumps()
        );
        if let Some(out) = s(m, "out") {
            let out = PathBuf::from(out);
            let mut allres = read_results(&out)?;
            allres.set(path.clone(), res);
            write_results(&out, &allres)?;
        }
    }
    Ok(())
}

// -------------------------------------------------------------------------------------- real

/// `real.SETS`: `(name, json, images, species)`; `@` reads the image root from a file.
pub const REAL_SETS: [(&str, &str, &str, &str); 6] = [
    (
        "crim13_ood (top, 2 mice)",
        "crim13/val.json",
        "crim13/images",
        "mouse",
    ),
    (
        "rat7m_val (6 cams, 8-32° elev.)",
        "rat7m/val.json",
        "rat7m/images",
        "rat",
    ),
    (
        "mirror_side_ood",
        "mirror_mouse/side_val.json",
        "mirror_mouse/images",
        "mouse",
    ),
    (
        "mirror_bottom_ood",
        "mirror_mouse/bottom_val.json",
        "mirror_mouse/images",
        "mouse",
    ),
    (
        "rat7m_subject2 (unseen rat)",
        "rat7m_s2/val.json",
        "rat7m_s2/images",
        "rat",
    ),
    (
        "irodent (zero-shot, wild rodents)",
        "irodent/test.json",
        "@irodent/IMAGE_ROOT",
        "rat",
    ),
];

fn real_command() -> Command {
    let [pair, no_pair] = bool_pair("pair_decoding", "pair-decoding", "no-pair-decoding", true);
    Command::new("real")
        .about("Evaluate detector + pose checkpoints on the held-out real datasets.")
        .arg(Arg::new("detector").long("detector").required(true))
        .arg(Arg::new("pose").long("pose").required(true))
        .arg(
            Arg::new("data")
                .long("data")
                .default_value("data/processed"),
        )
        .arg(Arg::new("tag").long("tag").default_value("model"))
        .arg(pair)
        .arg(no_pair)
        .arg(opt_usize("crop_size", "128"))
        .arg(opt_usize("max_images", "400"))
        .arg(
            Arg::new("out")
                .long("out")
                .default_value("results/real.json"),
        )
}

fn num(r: &PyVal, k: &str) -> f64 {
    r.get(k).and_then(PyVal::as_f64).unwrap_or(f64::NAN)
}

/// `eval real`: `cli/evaluate/real.py:run`.
pub fn real(m: &ArgMatches) -> Result<()> {
    let pipe = pipeline(
        s(m, "detector").context("--detector")?,
        s(m, "pose").context("--pose")?,
        serde_json::json!({
            "crop_size": m.get_one::<usize>("crop_size").copied().unwrap_or(128),
            "pair_decoding": flag_value(m, "pair_decoding", true),
        }),
        serde_json::json!({}),
    )?;
    let pose = PoseNet::load(
        Path::new(s(m, "pose").context("--pose")?),
        pipe.cfg.crop_size,
        pipe.cfg.crop_padding,
        pipe.cfg.pair_decoding,
    )?;
    let data = PathBuf::from(s(m, "data").unwrap_or("data/processed"));
    let max_images = m.get_one::<usize>("max_images").copied().unwrap_or(400);
    let mut res = PyVal::dict();
    for (name, js, imgs, _species) in REAL_SETS {
        let js_path = data.join(js);
        if !js_path.exists() {
            continue;
        }
        let img_root = match imgs.strip_prefix('@') {
            Some(file) => PathBuf::from(std::fs::read_to_string(data.join(file))?.trim()),
            None => data.join(imgs),
        };
        let src = CocoKeypoints::new(&js_path, &img_root, CocoKeypoints::synthetic())?;
        let r = eval_coco(&pipe, &pose, &src, max_images, true)?;
        println!(
            "{name}: PCK@0.1 {} PCK@0.05 {} median err {}px (box {}px) AP50 {} conf-AUROC {}",
            py_float(num(&r, "pck@0.1"), 3),
            py_float(num(&r, "pck@0.05"), 3),
            py_float(num(&r, "median_err_px"), 1),
            py_float(num(&r, "median_box_px"), 0),
            py_float(num(&r, "det_ap50"), 3),
            py_float(num(&r, "conf_auroc_visible_vs_occluded"), 3),
        );
        res.set(name, r);
    }
    let out = PathBuf::from(s(m, "out").unwrap_or("results/real.json"));
    let mut all = read_results(&out)?;
    all.set(s(m, "tag").unwrap_or("model"), res);
    write_results(&out, &all)
}

// ----------------------------------------------------------------------------- real-tracking

fn real_tracking_command() -> Command {
    Command::new("real-tracking")
        .about("Real two-mouse tracking test on CRIM13 out-of-distribution videos.")
        .arg(Arg::new("detector").long("detector").required(true))
        .arg(Arg::new("pose").long("pose").required(true))
        .arg(
            Arg::new("videos")
                .long("videos")
                .default_value("data/raw/lp_crim13/extracted/crim13/videos_OOD"),
        )
        .arg(opt_usize("n_videos", "6"))
        .arg(opt_usize("frames", "1500"))
        .arg(opt_usize("crop_size", "128"))
        .arg(opt_usize("min_run", "10"))
        .arg(
            Arg::new("configs")
                .long("configs")
                .num_args(1..)
                .default_values(["no_appearance", "color_histogram", "learned_embedding"]),
        )
        .arg(opt_f64("w_app", "0.2").help("appearance weight in the association cost"))
        .arg(
            Arg::new("tag")
                .long("tag")
                .default_value("")
                .help("prefix for result keys"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .default_value("results/real_tracking.json"),
        )
}

/// `real_tracking.coat_brightness(image, box)`: the median grey level of the box's central half.
pub fn coat_brightness(g: &GrayF, b: Box4) -> f64 {
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let (w, h) = ((b[2] - b[0]) / 4.0, (b[3] - b[1]) / 4.0);
    let clip = |v: f64, n: u32| -> usize { (v.max(0.0) as usize).min(n as usize) };
    let (y0, y1) = (
        clip((cy - h).max(0.0), g.height),
        clip(cy + h + 1.0, g.height),
    );
    let (x0, x1) = (
        clip((cx - w).max(0.0), g.width),
        clip(cx + w + 1.0, g.width),
    );
    let mut v = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            v.push(f64::from(g.pixels[y * g.width as usize + x]));
        }
    }
    if v.is_empty() {
        f64::NAN
    } else {
        median(&v)
    }
}

/// `real_tracking.score(result, frames)`: identity swaps by coat brightness, purity, ids, coverage.
pub fn tracking_score(
    tracks: &[(Vec<usize>, Vec<f64>)],
    frames: &[GrayF],
    min_run: usize,
) -> PyVal {
    let vals: Vec<Vec<f64>> = tracks
        .iter()
        .map(|(fr, boxes)| {
            fr.iter()
                .enumerate()
                .map(|(i, t)| {
                    let b = [
                        boxes[i * 4],
                        boxes[i * 4 + 1],
                        boxes[i * 4 + 2],
                        boxes[i * 4 + 3],
                    ];
                    if b.iter().all(|v| v.is_finite()) {
                        coat_brightness(&frames[*t], b)
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();
    let allv: Vec<f64> = vals
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if allv.len() < 10 {
        return PyVal::dict()
            .with("swaps", 0i64)
            .with("purity", f64::NAN)
            .with("n_ids", vals.len())
            .with("coverage", 0.0);
    }
    let img = Image {
        width: 1,
        height: allv.len() as u32,
        pixels: allv.iter().map(|v| *v as u8).collect(),
    };
    let thr = f64::from(otsu_threshold(&img));
    let (mut swaps, mut agree, mut total) = (0usize, 0usize, 0usize);
    for v in &vals {
        let coat: Vec<bool> = v
            .iter()
            .filter(|x| x.is_finite())
            .map(|x| *x > thr)
            .collect();
        if coat.is_empty() {
            continue;
        }
        let major = coat.iter().filter(|c| **c).count() as f64 / coat.len() as f64 >= 0.5;
        let mut run = 0;
        for c in &coat {
            let wrong = *c != major;
            agree += usize::from(!wrong);
            run = if wrong { run + 1 } else { 0 };
            if run == min_run {
                swaps += 1;
            }
        }
        total += coat.len();
    }
    let mut per_frame = vec![0usize; frames.len()];
    for (fr, boxes) in tracks {
        for (i, t) in fr.iter().enumerate() {
            if boxes[i * 4..i * 4 + 4].iter().all(|v| v.is_finite()) {
                per_frame[*t] += 1;
            }
        }
    }
    PyVal::dict()
        .with("swaps", swaps)
        .with("purity", agree as f64 / total.max(1) as f64)
        .with("n_ids", vals.len())
        .with(
            "coverage",
            per_frame.iter().filter(|c| **c >= 2).count() as f64 / frames.len().max(1) as f64,
        )
}

/// `eval real-tracking`: `cli/evaluate/real_tracking.py:run`.
pub fn real_tracking(m: &ArgMatches) -> Result<()> {
    let det = s(m, "detector").context("--detector")?;
    let pose = s(m, "pose").context("--pose")?;
    let n_videos = m.get_one::<usize>("n_videos").copied().unwrap_or(6);
    let max_frames = m.get_one::<usize>("frames").copied().unwrap_or(1500);
    let min_run = m.get_one::<usize>("min_run").copied().unwrap_or(10);
    let w_app = m.get_one::<f64>("w_app").copied().unwrap_or(0.2);
    let mut videos: Vec<PathBuf> = std::fs::read_dir(s(m, "videos").unwrap_or("."))
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "mp4"))
                .collect()
        })
        .unwrap_or_default();
    videos.sort();
    videos.truncate(n_videos);
    let wanted = strings(m, "configs");
    let mut res = PyVal::dict();
    for (name, embedder) in [
        ("no_appearance", "none"),
        ("color_histogram", "color"),
        ("learned_embedding", "model"),
    ] {
        if !wanted.iter().any(|w| w == name) {
            continue;
        }
        if embedder == "color" {
            println!(
                "{name}: unsupported: the HSV colour-histogram embedder \
                 (tracking/tracker.py:ColorHistogramEmbedder) is not ported"
            );
            continue;
        }
        let pipe = pipeline(
            det,
            pose,
            serde_json::json!({"crop_size": m.get_one::<usize>("crop_size").copied().unwrap_or(128), "embedder": embedder}),
            serde_json::json!({"max_animals": 2, "w_app": w_app}),
        )?;
        let mut per = Vec::new();
        for v in &videos {
            let (frames, fps) = crate::data::images::read_frames(v, Some(max_frames), 1)?;
            let out = pipe.run(&frames, None, fps.unwrap_or(30.0))?;
            let tracks: Vec<(Vec<usize>, Vec<f64>)> = out
                .tracks
                .values()
                .map(|t| (t.frames.clone(), t.boxes.clone()))
                .collect();
            per.push(tracking_score(&tracks, &frames, min_run));
        }
        let col = |k: &str| -> Vec<f64> { per.iter().map(|p| num(p, k)).collect() };
        let purity: Vec<f64> = col("purity").into_iter().filter(|v| !v.is_nan()).collect();
        let entry = PyVal::dict()
            .with("swaps_total", col("swaps").iter().sum::<f64>() as i64)
            .with(
                "purity_mean",
                if purity.is_empty() {
                    f64::NAN
                } else {
                    mean(&purity)
                },
            )
            .with("ids_per_video", mean(&col("n_ids")))
            .with("coverage_mean", mean(&col("coverage")))
            .with("videos", per.len())
            .with("frames_per_video", max_frames);
        println!("{name} {}", entry.dumps());
        res.set(name, entry);
    }
    let out = PathBuf::from(s(m, "out").unwrap_or("results/real_tracking.json"));
    let mut prev = read_results(&out)?;
    let tag = s(m, "tag").unwrap_or("");
    if let PyVal::Dict(items) = res {
        for (k, v) in items {
            prev.set(
                format!("{tag}{k}@w_app={}", crate::data::pyjson::float_repr(w_app)),
                v,
            );
        }
    }
    write_results(&out, &prev)
}

// --------------------------------------------------------------------------------- benchmark

fn benchmark_command() -> Command {
    Command::new("benchmark")
        .about("Evaluate trained checkpoints on held-out synthetic data (results/benchmark.json and .md).")
        .arg(Arg::new("runs").long("runs").default_value("runs"))
        .arg(Arg::new("out").long("out").default_value("results"))
        .arg(Arg::new("quick").long("quick").action(ArgAction::SetTrue).help("fewer samples"))
        .arg(Arg::new("devices").long("devices").num_args(1..).default_values(["auto", "cpu"]))
}

/// `runs/<name>.safetensors` (or `.pt`) when present.
fn find_checkpoint(runs: &Path, name: &str) -> Option<PathBuf> {
    ["safetensors", "pt"]
        .iter()
        .map(|e| runs.join(format!("{name}.{e}")))
        .find(|p| p.exists())
}

/// `eval benchmark`: `cli/evaluate/benchmark.py:run`, for the stages that need no rendered frames.
pub fn benchmark(m: &ArgMatches) -> Result<()> {
    let runs = PathBuf::from(s(m, "runs").unwrap_or("runs"));
    let out = PathBuf::from(s(m, "out").unwrap_or("results"));
    std::fs::create_dir_all(&out)?;
    let mut r = PyVal::dict();
    for stage in ["detection", "pose", "tracking", "multiview", "throughput"] {
        println!(
            "{stage}... unsupported: scored on frames drawn by src/animalpose/data/synthetic.py:render, \
             which is not ported"
        );
    }
    for (key, name, dims) in [
        ("temporal2d", "temporal", 2),
        ("temporal3d", "temporal3d", 3),
    ] {
        if let Some(path) = find_checkpoint(&runs, name) {
            println!("temporal {dims}d...");
            let (model, _, _) = load_model(&path.to_string_lossy())?;
            r.set(key, eval_temporal(&model, dims, 8, 64)?);
        }
    }
    for (key, name, dims) in [
        ("behavior2d", "behavior", 2),
        ("behavior3d", "behavior3d", 3),
    ] {
        if let Some(path) = find_checkpoint(&runs, name) {
            println!("behavior {dims}d...");
            let (model, cfg, _) = load_model(&path.to_string_lossy())?;
            r.set(key, eval_behavior(&model, &cfg.classes(), dims, 8, 128)?);
        }
    }
    std::fs::write(out.join("benchmark.json"), r.dumps_indent(2))?;
    let md = to_markdown(&r);
    std::fs::write(out.join("benchmark.md"), &md)?;
    println!("{md}");
    Ok(())
}
