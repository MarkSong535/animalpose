//! `animalpose behavior plot`: the ethogram figure and the annotated clip of a run, the port of
//! `src/animalpose/behavior/plot.py`.
//!
//! `// differs from Python:`
//! * `ethogram.png` is drawn with [`crate::cli::raster`] instead of matplotlib: the same three
//!   panels (stacked probabilities with a legend, the top-1 colour band, the time-share bars), the
//!   same palette and the same `1820 x 910` size, but a bitmap font and no anti-aliasing.
//! * The clip is `behavior_clip.mp4`, encoded by `ffmpeg` ([`crate::data::video::VideoWriter`]);
//!   without ffmpeg it falls back to numbered PNG frames in `behavior_clip/`. Each frame is the
//!   zoomed crop with the skeleton, then the 300 px panel with the time and the top-`k` behaviors.
//!   The crop is scaled bilinearly where Python uses `INTER_CUBIC`, and frames are grey levels.
//! * The behaviors are ordered by mean probability with a stable sort (`np.argsort` is a quicksort,
//!   which leaves ties in an arbitrary order).

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};

use crate::algos::image::resize_linear;
use crate::cli::clusters::NpzFile;
use crate::cli::pyjson::Json;
use crate::cli::raster::{hex, Canvas, Rgb};
use crate::cli::sources::Source;
use crate::data::sessions::NpyData;
use crate::data::synthetic::{index, NUM_KEYPOINTS};

/// `plot.py:PALETTE`.
pub const PALETTE: [&str; 12] = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
];

/// `CANONICAL.edge_index`.
pub const EDGES: [(usize, usize); 25] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (7, 13),
    (7, 14),
    (13, 15),
    (15, 17),
    (14, 16),
    (16, 18),
    (9, 19),
    (9, 20),
    (19, 21),
    (21, 23),
    (20, 22),
    (22, 24),
];

const BLACK: Rgb = [0, 0, 0];
const WHITE: Rgb = [255, 255, 255];

fn argmax_first(row: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

/// `plot.py:ethogram`: draws the figure to `out` and returns the time share of each drawn
/// behavior (`(name, share)` in drawing order). `full` is `(T, names.len())`.
pub fn ethogram(
    out: &Path,
    names: &[String],
    full: &[f64],
    times: &[f64],
    modelled: &[bool],
    heldout_from: Option<f64>,
) -> Result<Vec<(String, f64)>> {
    let c = names.len();
    let t = times.len();
    ensure!(
        t > 0 && full.len() == t * c,
        "the probabilities do not match the frames"
    );
    let mean: Vec<f64> = (0..c)
        .map(|j| (0..t).map(|i| full[i * c + j]).sum::<f64>() / t as f64)
        .collect();
    let mut order: Vec<usize> = (0..c).collect();
    order.sort_by(|a, b| mean[*b].total_cmp(&mean[*a]));
    let used: Vec<usize> = order
        .into_iter()
        .filter(|i| {
            modelled.get(*i).copied().unwrap_or(false)
                && (0..t)
                    .map(|r| full[r * c + *i])
                    .fold(f64::NEG_INFINITY, f64::max)
                    > 0.05
        })
        .take(12)
        .collect();
    let color = |i: usize| -> Rgb {
        used.iter()
            .position(|u| *u == i)
            .map(|j| hex(PALETTE[j % PALETTE.len()]))
            .unwrap_or(hex("#dddddd"))
    };
    let top: Vec<usize> = (0..t)
        .map(|i| argmax_first(&full[i * c..(i + 1) * c]))
        .collect();
    // dt = np.diff(times, append=times[-1] + np.median(np.diff(times)))
    let diffs: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let step = crate::algos::kinematics::nan_median(&diffs);
    let mut dt = diffs.clone();
    dt.push(if step.is_nan() { 0.0 } else { step });
    let total: f64 = dt.iter().sum();
    let share: Vec<(String, f64)> = used
        .iter()
        .map(|i| {
            // a fold from +0.0: an empty `Sum` of floats is -0.0, which would print as `-0.0`
            let s = (0..t)
                .filter(|r| top[*r] == *i)
                .fold(0.0, |acc, r| acc + dt[r]);
            (names[*i].clone(), s / total)
        })
        .collect();

    // 14 x 7 inches at 130 dpi
    let (w, h) = (1820i64, 910i64);
    let mut img = Canvas::new(w as u32, h as u32, WHITE);
    let (x0, x1) = (150i64, 1520i64);
    let (p0, p0b) = (60i64, 500i64);
    let (p1, p1b) = (530i64, 600i64);
    let (p2, p2b) = (690i64, 870i64);
    let (t0, t1) = (times[0], times[t - 1]);
    let span = if t1 > t0 { t1 - t0 } else { 1.0 };
    let xof = |time: f64| x0 + ((time - t0) / span * (x1 - x0) as f64).round() as i64;
    let title = "Behavior probabilities over time (time axis in seconds from frame timestamps)";
    img.text(
        (x0 + x1) / 2 - Canvas::text_width(title, 2) / 2,
        26,
        title,
        2,
        BLACK,
    );

    // panel 0: stacked probabilities, interpolated at every pixel column
    for x in x0..x1 {
        let time = t0 + (x - x0) as f64 / (x1 - x0) as f64 * span;
        let r = times.partition_point(|v| *v < time).min(t - 1);
        let (ra, rb) = if r > 0 && times[r] > time {
            (r - 1, r)
        } else {
            (r, r)
        };
        let f = if rb > ra {
            (time - times[ra]) / (times[rb] - times[ra])
        } else {
            0.0
        };
        let mut acc = 0.0;
        for i in &used {
            let v = full[ra * c + i] * (1.0 - f) + full[rb * c + i] * f;
            let (lo, hi) = (acc, acc + v);
            acc = hi;
            let ya = p0b - (lo.clamp(0.0, 1.0) * (p0b - p0) as f64).round() as i64;
            let yb = p0b - (hi.clamp(0.0, 1.0) * (p0b - p0) as f64).round() as i64;
            img.fill_rect(x, yb, x + 1, ya, color(*i));
        }
    }
    img.outline(x0, p0, x1, p0b, BLACK);
    for (v, label) in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")] {
        let y = p0b - (v * (p0b - p0) as f64).round() as i64;
        img.fill_rect(x0 - 8, y, x0, y + 1, BLACK);
        img.text(
            x0 - 12 - Canvas::text_width(label, 2),
            y + 7,
            label,
            2,
            BLACK,
        );
    }
    // the y label, written top to bottom
    let ylabel = "probability";
    let top_y = (p0 + p0b) / 2 - 8 * ylabel.len() as i64;
    for (j, ch) in ylabel.chars().enumerate() {
        img.text(30, top_y + 16 * (j as i64 + 1), &ch.to_string(), 2, BLACK);
    }
    for (j, i) in used.iter().enumerate() {
        let y = p0 + 10 + j as i64 * 30;
        img.fill_rect(x1 + 20, y, x1 + 44, y + 16, color(*i));
        img.text(x1 + 54, y + 15, &names[*i], 2, BLACK);
    }

    // panel 1: the top-1 band
    let mut s = 0;
    while s < t {
        let mut e = s + 1;
        while e < t && top[e] == top[s] {
            e += 1;
        }
        let end = if e < t { times[e] } else { times[t - 1] };
        img.fill_rect(
            xof(times[s]),
            p1,
            xof(end).max(xof(times[s]) + 1),
            p1b,
            color(top[s]),
        );
        s = e;
    }
    img.outline(x0, p1, x1, p1b, BLACK);
    img.text(
        x0 - 12 - Canvas::text_width("top 1", 2),
        (p1 + p1b) / 2 + 7,
        "top 1",
        2,
        BLACK,
    );
    let ticks = nice_ticks(t0, t1);
    for v in &ticks {
        let x = xof(*v);
        img.fill_rect(x, p1b, x + 1, p1b + 8, BLACK);
        let label = format_tick(*v);
        img.text(
            x - Canvas::text_width(&label, 2) / 2,
            p1b + 26,
            &label,
            2,
            BLACK,
        );
    }
    img.text((x0 + x1) / 2 - 40, p1b + 52, "time (s)", 2, BLACK);
    if let Some(hf) = heldout_from {
        let x = xof(hf);
        img.dashed_vline(x, p0, p0b, 2, BLACK);
        img.dashed_vline(x, p1, p1b, 2, BLACK);
        img.text(x, p0 - 4, " held out", 2, BLACK);
    }

    // panel 2: the time each behavior is the most likely one
    let mut ranked = share.clone();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let longest = ranked.first().map(|r| r.1).unwrap_or(0.0).max(1e-9) * 100.0;
    let n = ranked.len().max(1) as i64;
    let row = (p2b - p2) / n;
    for (j, (name, v)) in ranked.iter().enumerate() {
        let y = p2 + j as i64 * row;
        let len = (v * 100.0 / (longest * 1.05) * (x1 - x0) as f64).round() as i64;
        let col = names
            .iter()
            .position(|m| m == name)
            .map(color)
            .unwrap_or(BLACK);
        img.fill_rect(x0, y + row / 6, x0 + len, y + row - row / 6, col);
        img.text(
            x0 - 12 - Canvas::text_width(name, 2),
            y + row / 2 + 7,
            name,
            2,
            BLACK,
        );
    }
    img.outline(x0, p2, x1, p2b, BLACK);
    let xlabel = "time as most likely behavior (%)";
    img.text(
        (x0 + x1) / 2 - Canvas::text_width(xlabel, 2) / 2,
        p2b + 30,
        xlabel,
        2,
        BLACK,
    );
    img.save_png(out)?;
    Ok(share)
}

fn nice_ticks(t0: f64, t1: f64) -> Vec<f64> {
    let span = (t1 - t0).max(1e-9);
    let raw = span / 8.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10.0 * mag);
    let mut v = (t0 / step).ceil() * step;
    let mut out = Vec::new();
    while v <= t1 + 1e-9 && out.len() < 50 {
        out.push(v);
        v += step;
    }
    out
}

fn format_tick(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.1}")
    }
}

/// `catalog.py:top_k` for any width: the `k` most likely columns per row (stable on ties).
fn top_k(full: &[f64], width: usize, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let rows = full.len() / width.max(1);
    let mut idx = Vec::with_capacity(rows);
    let mut probs = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &full[r * width..(r + 1) * width];
        let mut order: Vec<usize> = (0..width).collect();
        order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
        order.truncate(k);
        probs.push(order.iter().map(|i| row[*i]).collect());
        idx.push(order);
    }
    (idx, probs)
}

/// The arrays of a run's `tracks.npz` the plots read.
pub struct RunTracks {
    /// Presentation time of every frame of the recording.
    pub times: Vec<f64>,
    /// Frames of track 0.
    pub frames: Vec<i64>,
    /// Track 0 keypoints `(T, K)`.
    pub kp: Vec<[f64; 2]>,
    /// Track 0 confidences `(T, K)`.
    pub conf: Vec<f64>,
    /// Track 0 catalogue probabilities `(T, names.len())`.
    pub full: Vec<f64>,
    /// The catalogue names.
    pub names: Vec<String>,
}

impl RunTracks {
    /// Reads `tracks.npz`.
    pub fn read(path: &Path) -> Result<Self> {
        let z = NpzFile::read(path)?;
        let (kp, conf, n) = crate::cli::clusters::track0(&z, path)?;
        let names = match z.arrays.get("behaviors").map(|a| &a.data) {
            Some(NpyData::Text(v)) => v.clone(),
            _ => anyhow::bail!(
                "{} has no behaviors (run predict with a behavior model)",
                path.display()
            ),
        };
        let full = z.floats("track0_behavior_full")?;
        ensure!(
            full.len() == n * names.len(),
            "track0_behavior_full does not match the behaviors"
        );
        let frames = z
            .arrays
            .get("track0_frames")
            .with_context(|| format!("{} has no track0_frames", path.display()))?
            .ints()
            .context("track0_frames is not numeric")?;
        Ok(Self {
            times: z.floats("times")?,
            frames,
            kp,
            conf,
            full,
            names,
        })
    }
}

/// `plot.py:clip`: a window of the video zoomed on the animal, with its skeleton and the top-`k`
/// behaviors, written to `out`: an `.mp4` path is encoded with ffmpeg, any other path is a folder
/// of numbered PNG frames. Returns the frames written.
#[allow(clippy::too_many_arguments)]
pub fn clip(
    out: &Path,
    video: &Path,
    z: &RunTracks,
    start_s: f64,
    seconds: f64,
    k: usize,
    scale: u32,
    half: u32,
) -> Result<usize> {
    let times = &z.times;
    let first = times.partition_point(|v| *v < start_s);
    let last = times.partition_point(|v| *v < start_s + seconds);
    let width = z.names.len();
    let (idx_k, p_k) = top_k(&z.full, width, k);
    let mut src = Source::open(video);
    let size = 2 * half * scale;
    let panel = 300u32;
    let mut writer = (out.extension().and_then(|e| e.to_str()) == Some("mp4"))
        .then(|| crate::data::video::VideoWriter::new(out, clip_fps(times, start_s, seconds)));
    if writer.is_none() {
        std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    }
    let mid = index("spine_mid");
    let mut centre: Option<[f64; 2]> = None;
    let mut written = 0usize;
    for (f, time) in times.iter().enumerate().take(last).skip(first) {
        let Some((_, img)) = src.frames(&[f]).into_iter().next() else {
            break;
        };
        let i = z.frames.partition_point(|v| *v < f as i64);
        if i >= z.frames.len() || z.frames[i] != f as i64 {
            continue;
        }
        let p = z.kp[i * NUM_KEYPOINTS + mid];
        let c = (p[0].is_finite() && p[1].is_finite()).then_some(p);
        centre = match (centre, c) {
            (Some(old), Some(new)) => {
                Some([0.85 * old[0] + 0.15 * new[0], 0.85 * old[1] + 0.15 * new[1]])
            }
            (None, new) => new,
            (old, None) => old,
        };
        let (cx, cy) = match centre {
            Some(cc) => (nan0(cc[0]) as i64, nan0(cc[1]) as i64),
            None => (i64::from(img.width) / 2, i64::from(img.height) / 2),
        };
        let side = 2 * half;
        let mut crop = Vec::with_capacity((side * side) as usize);
        for v in 0..i64::from(side) {
            for u in 0..i64::from(side) {
                let x = (cx - i64::from(half) + u).clamp(0, i64::from(img.width) - 1);
                let y = (cy - i64::from(half) + v).clamp(0, i64::from(img.height) - 1);
                crop.push(img.pixel(x as u32, y as u32));
            }
        }
        let view = resize_linear(&crop, side, side, size, size);
        let mut frame = Canvas::new(size + panel, size, WHITE);
        for y in 0..size {
            for x in 0..size {
                let g = view[(y * size + x) as usize].round().clamp(0.0, 255.0) as u8;
                frame.put(i64::from(x), i64::from(y), [g, g, g]);
            }
        }
        let origin = [(cx - i64::from(half)) as f64, (cy - i64::from(half)) as f64];
        let pts: Vec<[f64; 2]> = (0..NUM_KEYPOINTS)
            .map(|j| {
                let q = z.kp[i * NUM_KEYPOINTS + j];
                [
                    (q[0] - origin[0]) * f64::from(scale),
                    (q[1] - origin[1]) * f64::from(scale),
                ]
            })
            .collect();
        let ok: Vec<bool> = (0..NUM_KEYPOINTS)
            .map(|j| {
                z.conf[i * NUM_KEYPOINTS + j] > 0.3
                    && pts[j][0].is_finite()
                    && pts[j][1].is_finite()
            })
            .collect();
        for (a, b) in EDGES {
            if ok[a] && ok[b] {
                frame.line(pts[a], pts[b], 2.0, [40, 160, 255]);
            }
        }
        for j in (0..NUM_KEYPOINTS).filter(|j| ok[*j]) {
            frame.circle(pts[j][0], pts[j][1], 4.0, [230, 40, 40]);
        }
        let px = i64::from(size);
        frame.text(px + 12, 34, &format!("t = {time:7.2} s"), 3, BLACK);
        for r in 0..k.min(idx_k.get(i).map_or(0, Vec::len)) {
            let y = 90 + r as i64 * 70;
            let (name, p) = (&z.names[idx_k[i][r]], p_k[i][r]);
            frame.text(px + 12, y, &format!("{}. {name}", r + 1), 2, BLACK);
            frame.fill_rect(
                px + 12,
                y + 12,
                px + 12 + (260.0 * p) as i64 + 1,
                y + 33,
                [60, 120, 180],
            );
            frame.text(px + 16, y + 28, &crate::cli::pyjson::fixed(p, 2), 2, WHITE);
        }
        match writer.as_mut() {
            Some(w) => {
                let mut bgr = frame.pixels.clone();
                for px in bgr.chunks_exact_mut(3) {
                    px.swap(0, 2);
                }
                w.write(&crate::algos::image::Bgr::from_pixels(
                    frame.width,
                    frame.height,
                    bgr,
                )?)?;
            }
            None => frame.save_png(&out.join(format!("{written:06}.png")))?,
        }
        written += 1;
    }
    if let Some(w) = writer {
        w.finish()?;
    }
    Ok(written)
}

fn nan0(v: f64) -> f64 {
    crate::algos::kinematics::nan_to_num(v)
}

/// What `plot_run` returns: the shares, the clip folder and the frames written to it.
pub type PlotRun = (Vec<(String, f64)>, PathBuf, usize);

/// `plot.py:plot_run`: `ethogram.png` and the clip frames next to a run's `tracks.npz`; returns
/// the shares and the clip folder.
pub fn plot_run(
    run: &Path,
    video: &Path,
    clip_start: f64,
    clip_seconds: f64,
    top: usize,
    heldout_from: Option<f64>,
) -> Result<PlotRun> {
    let z = RunTracks::read(&run.join("tracks.npz"))?;
    let catalog_path = run.join("behavior_catalog.json");
    let catalog = Json::parse(
        &std::fs::read_to_string(&catalog_path)
            .with_context(|| format!("cannot read {}", catalog_path.display()))?,
    )?;
    let modelled: Vec<bool> = match &catalog {
        Json::Arr(items) => items
            .iter()
            .map(|d| matches!(d.get("modelled"), Some(Json::Bool(true))))
            .collect(),
        _ => anyhow::bail!("{} is not a list", catalog_path.display()),
    };
    let times: Vec<f64> = z
        .frames
        .iter()
        .map(|f| z.times.get(*f as usize).copied().unwrap_or(f64::NAN))
        .collect();
    let share = ethogram(
        &run.join("ethogram.png"),
        &z.names,
        &z.full,
        &times,
        &modelled,
        heldout_from,
    )?;
    let dir = if crate::data::video::ffmpeg_available() {
        run.join("behavior_clip.mp4")
    } else {
        run.join("behavior_clip")
    };
    let n = clip(&dir, video, &z, clip_start, clip_seconds, top, 3, 60)?;
    Ok((share, dir, n))
}

/// The frame rate the clip would be encoded at: `1 / median(diff(times[first:last + 1]))`.
pub fn clip_fps(times: &[f64], start_s: f64, seconds: f64) -> f64 {
    let first = times.partition_point(|v| *v < start_s);
    let last = times.partition_point(|v| *v < start_s + seconds);
    let window = &times[first.min(times.len())..(last + 1).min(times.len())];
    let d: Vec<f64> = window.windows(2).map(|w| w[1] - w[0]).collect();
    1.0 / crate::algos::kinematics::nan_median(&d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::clusters::tests::{temp, write_tracks, write_video, FPS, MOVE, T};

    fn catalogue() -> (Vec<String>, Vec<bool>, Vec<f64>) {
        let cat = crate::cli::catalog::Catalog::load();
        let names = cat.names.clone();
        let c = names.len();
        let mut modelled = vec![false; c];
        let col = |n: &str| names.iter().position(|m| m == n).unwrap();
        for n in ["rest", "walk", "self_groom"] {
            modelled[col(n)] = true;
        }
        let mut full = vec![0.0; T * c];
        for t in 0..T {
            let after = t as f64 / FPS > MOVE as f64 / FPS;
            full[t * c + col("rest")] = if after { 0.8 } else { 0.1 };
            full[t * c + col("walk")] = if after { 0.1 } else { 0.8 };
            full[t * c + col("self_groom")] = 0.1;
        }
        (names, modelled, full)
    }

    #[test]
    fn plot_reports_the_time_share_of_each_behavior() {
        let dir = temp("plot");
        let (names, modelled, full) = catalogue();
        let run = dir.join("run");
        std::fs::create_dir_all(&run).unwrap();
        write_tracks(&run.join("tracks.npz"), &names, &full);
        let describe = crate::cli::catalog::Catalog::load().describe(&modelled);
        std::fs::write(
            run.join("behavior_catalog.json"),
            serde_json::to_string(&describe).unwrap(),
        )
        .unwrap();
        let video = dir.join("video");
        write_video(&video);
        let (share, clip_dir, n) = plot_run(&run, &video, 0.5, 0.5, 2, Some(2.0)).unwrap();
        // top-1 is walk up to frame MOVE (inclusive: times > MOVE / FPS is strict), rest after
        let walk = (MOVE + 1) as f64 / T as f64;
        let got: std::collections::BTreeMap<String, f64> = share.iter().cloned().collect();
        assert_eq!(got.len(), 3);
        assert!((got["walk"] - walk).abs() < 1e-9 && (got["rest"] - (1.0 - walk)).abs() < 1e-9);
        assert_eq!(got["self_groom"], 0.0);
        let png = image::open(run.join("ethogram.png")).unwrap().to_rgb8();
        assert!(png.height() > 500 && png.pixels().any(|p| p.0.iter().any(|v| *v < 250)));
        assert_eq!(n, 15); // 0.5 s at 30 fps
        if clip_dir.extension().is_some_and(|e| e == "mp4") {
            let info = crate::data::video::probe(&clip_dir).unwrap();
            assert_eq!((info.width, info.height, info.frames), (660, 360, 15));
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let first = image::open(clip_dir.join("000000.png")).unwrap().to_rgb8();
        assert_eq!((first.width(), first.height()), (660, 360));
        // the skeleton is drawn over the grey crop, the panel is white with coloured bars
        let colored = |img: &image::RgbImage, x0: u32, x1: u32| {
            img.enumerate_pixels()
                .filter(|(x, _, p)| {
                    *x >= x0 && *x < x1 && {
                        let (mx, mn) = (p.0.iter().max().unwrap(), p.0.iter().min().unwrap());
                        mx - mn > 40
                    }
                })
                .count()
        };
        assert!(colored(&first, 0, 360) > 200 && colored(&first, 360, 660) > 100);
        assert!(
            (clip_fps(
                &(0..T).map(|t| t as f64 / FPS).collect::<Vec<_>>(),
                0.5,
                0.5
            ) - FPS)
                .abs()
                < 1e-6
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ethogram_shares_follow_the_frame_times() {
        let dir = temp("ethogram");
        let (names, modelled, full) = catalogue();
        let times: Vec<f64> = (0..T).map(|t| t as f64 / FPS).collect();
        let share = ethogram(&dir.join("e.png"), &names, &full, &times, &modelled, None).unwrap();
        let total: f64 = share.iter().map(|s| s.1).sum();
        assert!((total - 1.0).abs() < 1e-9);
        // the first name drawn is the most probable on average: walk, for 61 of the 120 frames
        assert_eq!(share[0].0, "walk");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
