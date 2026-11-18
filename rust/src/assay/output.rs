//! Runs an experiment setup on an analysis folder and writes its results into `<folder>/assay/`:
//!
//! * `behavior_v3.csv`: one row per frame (`frame, time, scored, location, behavior`, then each
//!   module's columns);
//! * `behavior_summary_v3.csv`: a header line with the code version, config hash and model
//!   hashes, then one row for the video;
//! * `run_manifest.json`: the same hashes, the inputs' sizes and times, notes and timing.
//!
//! Every file is written as `<name>.tmp` and renamed. The folder is built as `assay.staging` and
//! swapped in only when complete; a previous `assay` is moved to `assay.backup-<time>`, never
//! deleted. A folder whose manifest matches the code, config, models and inputs is skipped unless
//! forced. The motion feature, the one costly measurement, is cached by video and parameters.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use serde_json::{json, Value};

use super::arena::{
    arena_mask, find_arena, floor_from_mask, median_u8, render, Layout, Pt, Shape, METHOD,
};
use super::arena_model::{self, ArenaModel};
use super::config::AssayConfig;
use super::features::{self, Features};
use super::input::{self, Input};
use super::maze;
use super::modules::{self, Column, Context, Module};
use super::motion;
use super::temporal;
use crate::algos::image::Image;

/// The hex SHA-256 of a file, computed once per path, size and modification time.
type HashCache = Option<HashMap<(PathBuf, u64, SystemTime), String>>;

/// How long ago a file must have been modified before its hash is remembered. A file written
/// again within the clock's resolution keeps its time (Windows, quick writes of the same length),
/// so a hash is cached only once a later write would show (Git's "racily clean" rule).
const SETTLED: Duration = Duration::from_secs(2);

pub fn sha256_cached(path: &Path) -> Result<String> {
    static CACHE: Mutex<HashCache> = Mutex::new(None);
    let meta = fs::metadata(path).with_context(|| format!("cannot read {}", path.display()))?;
    let key = (path.to_path_buf(), meta.len(), meta.modified()?);
    if let Some(hit) = CACHE.lock().unwrap().as_ref().and_then(|c| c.get(&key)) {
        return Ok(hit.clone());
    }
    let hash = crate::cli::download::sha256_file(path)?;
    let settled = SystemTime::now()
        .duration_since(key.2)
        .is_ok_and(|age| age >= SETTLED);
    if settled {
        CACHE
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .insert(key, hash.clone());
    }
    Ok(hash)
}

/// Writes `bytes` to `path` through `<path>.tmp` and a rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, bytes).with_context(|| format!("cannot write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("cannot write {}", path.display()))
}

fn stamp(path: &Path) -> Value {
    stamp_as(path, &path.display().to_string())
}

/// `path`'s size and modification time, recorded under `name`.
fn stamp_as(path: &Path, name: &str) -> Value {
    match fs::metadata(path) {
        Ok(m) => json!({
            "path": name,
            "size": m.len(),
            "mtime": m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs_f64()),
        }),
        Err(_) => json!({"path": name, "missing": true}),
    }
}

/// How to run.
pub struct Options<'a> {
    /// Where layouts are cached (`<rig>.json` and `<rig>.png`).
    pub zones_dir: &'a Path,
    /// The top view's video, for the motion feature and the background (single-view folders).
    pub video: Option<&'a Path>,
    /// Checkpoints by kind, hashed into the manifest.
    pub models: Vec<(String, PathBuf)>,
    /// The resolved pipeline config, hashed into the manifest.
    pub config: Value,
    /// Redo a folder whose manifest is current.
    pub force: bool,
}

/// The result of one run.
#[derive(Clone, Debug)]
pub struct Report {
    pub dir: PathBuf,
    /// False when the results were current and the run was skipped.
    pub ran: bool,
    pub notes: Vec<String>,
}

fn key(dir: &Path, opts: &Options) -> Result<Value> {
    let mut inputs = Vec::new();
    for name in [
        "tracks.npz",
        "animals3d.npz",
        "identities.json",
        "cameras.json",
        "views.json",
    ] {
        if dir.join(name).exists() {
            inputs.push(stamp_as(&dir.join(name), name));
        }
    }
    for entry in fs::read_dir(dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("view") && name.ends_with("_tracks.npz") {
            inputs.push(stamp_as(&entry.path(), &name));
        }
    }
    inputs.sort_by_key(|v| v["path"].to_string());
    if let Some(v) = opts.video {
        inputs.push(stamp(v));
    }
    let mut models = serde_json::Map::new();
    for (kind, path) in &opts.models {
        models.insert(kind.clone(), json!(sha256_cached(path).ok()));
    }
    let config = serde_json::to_vec(&opts.config)?;
    Ok(json!({
        "code": format!("animalpose {}", env!("CARGO_PKG_VERSION")),
        "config_sha256": sha256_bytes(&config),
        "models_sha256": models,
        "inputs": inputs,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use crate::cli::download::{Hasher, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finish().iter().map(|b| format!("{b:02x}")).collect()
}

/// The layout a module needs (`floor`, `plus_maze` or `zones`), from the first source that has it:
///
/// 1. a hand-made `zones/<rig>.json` (`"source": "manual"`);
/// 2. the setup's arena detector (`checkpoints.arena`), run on this video and saved as
///    `zones/<rig>.<folder>-<video>.json` with a picture;
/// 3. an automatic layout of the setup found earlier with the current [`METHOD`];
/// 4. classical vision on the background (the plus maze, or a floor), cached for the setup.
///
/// `zones` (named regions) come only from 1 or 2; without them the module reports it has none.
fn layout_for(
    need: &str,
    input: &Input,
    cfg: &AssayConfig,
    opts: &Options,
    notes: &mut Vec<String>,
) -> Result<Option<Layout>> {
    let rig = super::recording_safe(&cfg.arena.rig);
    let file = opts.zones_dir.join(format!("{rig}.json"));
    let cached = Layout::load(&file)?;
    if let Some(layout) = cached.as_ref().filter(|l| l.source == "manual") {
        return Ok(Some(layout.clone()));
    }
    let video = input.top_video.as_deref().or(opts.video);
    if let (Some(model), Some(video)) = (&cfg.arena.model, video) {
        let layout = detect_arena(model, video, input, cfg, opts, notes)?;
        if provides(&layout, need) {
            return Ok(Some(layout));
        }
        notes.push(format!(
            "arena: the detector's layout has no {need}; finding it another way"
        ));
    }
    if need == "zones" {
        return Ok(None);
    }
    match cached {
        Some(layout) if layout.method >= METHOD => return Ok(Some(layout)),
        Some(_) => notes.push(format!(
            "zones of {rig} found again with the current method"
        )),
        None => {}
    }
    let video = video.context("finding the arena needs the top view's video")?;
    let bg = motion::background(video, cfg.maze.bg_frames)?;
    let track = top_track(input);
    let layout = match need {
        "plus_maze" => maze::detect(&bg, &cfg.maze, &cfg.arena.rig, &track),
        _ => {
            let (mask, floor, finder, doubts) = match find_arena(&bg, &track, Shape::Box) {
                Some(found) => (found.mask, found.floor, found.method, found.doubts),
                None => (
                    arena_mask(&bg),
                    None,
                    "largest",
                    vec![if track.len() < 20 {
                        "no track to find the arena by; the largest region was used".to_owned()
                    } else {
                        "no region encloses the track; the largest region was used".to_owned()
                    }],
                ),
            };
            let floor = floor
                .or_else(|| floor_from_mask(&mask))
                .context("no arena floor found in the background")?;
            for doubt in &doubts {
                notes.push(format!("arena: {doubt}"));
            }
            Layout {
                rig: cfg.arena.rig.clone(),
                source: "auto".into(),
                method: METHOD,
                status: if doubts.is_empty() {
                    "ok"
                } else {
                    "needs_review"
                }
                .into(),
                reason: doubts.join("; "),
                finder: finder.into(),
                image_size: [bg.width, bg.height],
                floor: floor.to_vec(),
                ..Layout::default()
            }
        }
    };
    let png = save_layout(&layout, &file, &bg, &track)?;
    notes.push(format!(
        "zones found automatically; check {} by eye",
        png.display()
    ));
    Ok(Some(layout))
}

/// Whether `layout` has what a module needs.
fn provides(layout: &Layout, need: &str) -> bool {
    let has = |kind: &str| layout.zones.iter().any(|z| z.kind == kind);
    match need {
        "plus_maze" => has("center") && has("open_arm") && has("closed_arm"),
        "floor" => layout.floor.len() == 4,
        _ => !layout.zones.is_empty() || layout.floor.len() == 4,
    }
}

/// The arena of `video` by the detector at `model` ([`super::arena_model`]), saved with a picture.
fn detect_arena(
    model: &Path,
    video: &Path,
    input: &Input,
    cfg: &AssayConfig,
    opts: &Options,
    notes: &mut Vec<String>,
) -> Result<Layout> {
    if let Some(expected) = &cfg.arena.model_sha256 {
        let actual = sha256_cached(model)?;
        anyhow::ensure!(
            actual.eq_ignore_ascii_case(expected),
            "the arena checkpoint {} has SHA-256 {actual}, the config expects {expected}",
            model.display()
        );
    }
    let net = ArenaModel::load(model)?;
    let frames = motion::sample(video, arena_model::FRAMES)?;
    let bg = median_u8(&frames).with_context(|| format!("{} has no frames", video.display()))?;
    let layout = net.layout(&net.detect(&frames)?, &cfg.arena.rig, &bg, &cfg.maze);
    // The video's folder and name: Multiview pairs all hold a `top.mp4`.
    let name = |p: Option<&std::ffi::OsStr>| {
        p.map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let video_id = super::recording_safe(&format!(
        "{}-{}",
        name(video.parent().and_then(Path::file_name)),
        name(video.file_stem())
    ));
    let rig = super::recording_safe(&cfg.arena.rig);
    let file = opts.zones_dir.join(format!("{rig}.{video_id}.json"));
    let png = save_layout(&layout, &file, &bg, &top_track(input))?;
    notes.push(match layout.status.as_str() {
        "ok" => format!("arena: {}; picture {}", layout.finder, png.display()),
        _ => format!(
            "arena: {} needs a review ({}); picture {}",
            layout.finder,
            layout.reason,
            png.display()
        ),
    });
    Ok(layout)
}

/// The top view's body positions, at most ~5000 of them.
fn top_track(input: &Input) -> Vec<Pt> {
    match (&input.top, input.index("spine_mid")) {
        (Some(top), Some(spine)) => {
            let k = input.kpts();
            let seen: Vec<Pt> = (top.first..top.first + top.rows)
                .filter_map(|t| top.point(t, spine, k).map(|(p, _)| p))
                .collect();
            let step = (seen.len() / 5000).max(1);
            seen.into_iter().step_by(step).collect()
        }
        _ => Vec::new(),
    }
}

/// Writes `layout` as `file` and its picture next to it (`.png`); returns the picture's path.
fn save_layout(layout: &Layout, file: &Path, bg: &Image, track: &[Pt]) -> Result<PathBuf> {
    if let Some(dir) = file.parent() {
        fs::create_dir_all(dir)?;
    }
    write_atomic(file, &serde_json::to_vec_pretty(layout)?)?;
    let picture = render(bg, layout, track);
    let rgb: Vec<u8> = picture
        .pixels
        .chunks(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    let mut bytes = Vec::new();
    image::RgbImage::from_raw(picture.width, picture.height, rgb)
        .context("cannot draw the zones")?
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )?;
    let png = file.with_extension("png");
    write_atomic(&png, &bytes)?;
    Ok(png)
}

fn motion_cached(
    input: &Input,
    cfg: &AssayConfig,
    cache: &Path,
) -> Result<Option<Vec<Option<f64>>>> {
    let (Some(video), Some(top)) = (&input.top_video, &input.top) else {
        return Ok(None);
    };
    let k = json!({"video": stamp(video), "motion": cfg.motion, "frames": input.n_frames});
    if let Ok(bytes) = fs::read(cache) {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            if v["key"] == k {
                return Ok(serde_json::from_value(v["values"].clone()).ok());
            }
        }
    }
    let values = motion::measure(video, top, input.n_frames, input.fps, &cfg.motion)?;
    write_atomic(
        cache,
        &serde_json::to_vec(&json!({"key": k, "values": values}))?,
    )?;
    Ok(Some(values))
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

/// Runs `cfg` on the analysis folder `dir`.
pub fn run_dir(dir: &Path, cfg: &AssayConfig, opts: &Options) -> Result<Report> {
    let started = Instant::now();
    let out = dir.join("assay");
    let key = key(dir, opts)?;
    if !opts.force {
        if let Ok(bytes) = fs::read(out.join("run_manifest.json")) {
            let old: Value = serde_json::from_slice(&bytes).unwrap_or_default();
            if old["key"] == key && old["status"] == "ok" {
                return Ok(Report {
                    dir: out,
                    ran: false,
                    notes: Vec::new(),
                });
            }
        }
    }
    let mut input = input::read_dir(dir, cfg, opts.video)?;
    if input.top_video.is_none() {
        input.top_video = opts.video.map(Path::to_path_buf);
    }
    let mut notes = Vec::new();
    let pose = features::pose(&input, cfg)?;
    if input.world.is_some() && !pose.dims3 {
        notes.push(format!(
            "only {:.0}% of frames have 3D (needs {:.0}%): scored in 2D",
            100.0 * pose.valid3d,
            100.0 * cfg.min_valid3d
        ));
    }
    let mut feats: Features = features::compute(&pose, &input.names, cfg, input.fps)?;
    let modules: Vec<Box<dyn Module>> = cfg
        .modules
        .iter()
        .map(|m| modules::create(m).with_context(|| format!("unknown assay module {m:?}")))
        .collect::<Result<_>>()?;
    let staging = dir.join("assay.staging");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    if cfg.runs("grooming") && input.fps >= cfg.groom.min_motion_fps {
        let cache = out.join("motion_cache.json");
        let staged = staging.join("motion_cache.json");
        if cache.is_file() {
            fs::copy(&cache, &staged)?;
        }
        if let Some(m) = motion_cached(&input, cfg, &staged)? {
            feats.motion = m;
        }
    }
    let mut layout = None;
    for m in &modules {
        let Some(kind) = m.needs_layout() else {
            continue;
        };
        let needed = match kind {
            "floor" => !pose.dims3 && input.rig.is_none(),
            _ => true,
        };
        if needed && layout.is_none() {
            layout = layout_for(kind, &input, cfg, opts, &mut notes)?;
        }
    }
    let ctx = Context {
        input: &input,
        cfg,
        pose: &pose,
        features: &feats,
        layout: layout.as_ref(),
    };
    let mut columns: Vec<Column> = Vec::new();
    let mut states: Vec<(String, Vec<Option<bool>>)> = Vec::new();
    let mut location = None;
    let mut summary: Vec<(String, Value)> = Vec::new();
    for m in &modules {
        let o = m
            .run(&ctx)
            .with_context(|| format!("assay module {}", m.name()))?;
        columns.extend(o.columns);
        states.extend(o.states);
        if location.is_none() {
            location = o.location;
        }
        summary.extend(o.summary);
        notes.extend(o.notes);
    }
    let n = input.n_frames;
    let fps = input.fps;
    let mut ordered: Vec<(&str, &[Option<bool>])> = Vec::new();
    for p in &cfg.priority {
        if let Some((name, s)) = states.iter().find(|(name, _)| name == p) {
            ordered.push((name, s));
        }
    }
    for (name, s) in &states {
        if !cfg.priority.contains(name) {
            ordered.push((name, s));
        }
    }
    let behavior = temporal::resolve(&ordered, n);
    let spine = input.index("spine_mid");
    let scored: Vec<bool> = (0..n).map(|t| pose.at(t, spine).is_some()).collect();

    // Per-frame table.
    let mut csv = String::from("frame,time,scored,location,behavior");
    for c in &columns {
        csv.push(',');
        csv.push_str(&csv_field(c.name()));
    }
    csv.push('\n');
    for t in 0..n {
        let loc = location
            .as_ref()
            .map(|l: &Vec<Option<String>>| l[t].clone().unwrap_or_else(|| "unknown".into()))
            .unwrap_or_default();
        csv.push_str(&format!(
            "{t},{:.4},{},{},{}",
            t as f64 / fps,
            u8::from(scored[t]),
            loc,
            behavior[t]
        ));
        for c in &columns {
            csv.push(',');
            csv.push_str(&csv_field(&c.cell(t)));
        }
        csv.push('\n');
    }
    write_atomic(&staging.join("behavior_v3.csv"), csv.as_bytes())?;

    // Summary row.
    let mut row: Vec<(String, Value)> = vec![
        (
            "folder".into(),
            json!(dir.file_name().map(|n| n.to_string_lossy().into_owned())),
        ),
        ("experiment".into(), json!(cfg.experiment)),
        ("frames".into(), json!(n)),
        ("fps".into(), json!(fps)),
        (
            "scored_s".into(),
            json!(scored.iter().filter(|s| **s).count() as f64 / fps),
        ),
        ("dims".into(), json!(if pose.dims3 { "3d" } else { "2d" })),
        ("valid3d".into(), json!(pose.valid3d)),
        ("body_length".into(), json!(feats.bl)),
    ];
    row.extend(summary.iter().cloned());
    let cell = |v: &Value| match v {
        Value::Null => String::new(),
        Value::String(s) => csv_field(s),
        Value::Number(x) => x.as_f64().map_or_else(
            || x.to_string(),
            |f| {
                if f.fract() == 0.0 && f.abs() < 1e15 {
                    format!("{f:.0}")
                } else {
                    format!("{f:.4}")
                }
            },
        ),
        other => csv_field(&other.to_string()),
    };
    let models = key["models_sha256"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| {
                    format!(
                        "{k}:{}",
                        v.as_str().map_or("missing", |s| &s[..12.min(s.len())])
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let mut text = format!(
        "# {} config_sha256:{} models {}\n",
        key["code"].as_str().unwrap_or_default(),
        key["config_sha256"].as_str().unwrap_or_default(),
        models
    );
    text.push_str(
        &row.iter()
            .map(|(k, _)| csv_field(k))
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    text.push_str(
        &row.iter()
            .map(|(_, v)| cell(v))
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    write_atomic(&staging.join("behavior_summary_v3.csv"), text.as_bytes())?;

    let manifest = json!({
        "key": key,
        "status": "ok",
        "experiment": cfg.experiment,
        "modules": cfg.modules,
        "parameters": cfg,
        "notes": notes,
        "summary": row.iter().cloned().collect::<serde_json::Map<String, Value>>(),
        "seconds": started.elapsed().as_secs_f64(),
    });
    write_atomic(
        &staging.join("run_manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;

    // Swap the finished folder in; keep the one it replaces.
    if out.exists() {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        fs::rename(&out, dir.join(format!("assay.backup-{stamp}")))?;
    }
    fs::rename(&staging, &out)?;
    Ok(Report {
        dir: out,
        ran: true,
        notes,
    })
}

/// Runs the experiment a pipeline config sets up (`assay.experiment`) on `dir`; `None` when the
/// config sets none.
pub fn run_config(
    dir: &Path,
    config: &Path,
    overrides: &[String],
    video: Option<&Path>,
    zones_dir: &Path,
    force: bool,
) -> Result<Option<Report>> {
    let cfg = crate::data::config::Config::load(config, overrides)?;
    let Some(assay) = AssayConfig::from_config(&cfg) else {
        return Ok(None);
    };
    let opts = Options {
        zones_dir,
        video,
        models: crate::cli::predict::checkpoints(&cfg),
        config: cfg.json(),
        force,
    };
    run_dir(dir, &assay, &opts).map(Some)
}
