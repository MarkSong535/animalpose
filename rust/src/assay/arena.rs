//! Arena geometry: named zones, the floor, and the layout file that caches them per camera setup.
//!
//! A layout is found once per setup (from the calibration, or by classical vision on the median
//! background, see [`super::maze`]) and saved as `zones/<rig>.json` with a picture to check by
//! eye. A file with `"source": "manual"` is a hand-edited layout and always wins.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::algos::blob::{morph_close, morph_open};
use crate::algos::image::{GrayF, Image};
use crate::multiview::camera::Camera;

pub type Pt = [f64; 2];

/// A closed polygon.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Polygon(pub Vec<Pt>);

impl Polygon {
    /// Even-odd containment.
    pub fn contains(&self, p: Pt) -> bool {
        let v = &self.0;
        let mut inside = false;
        let mut j = v.len().wrapping_sub(1);
        for i in 0..v.len() {
            let (a, b) = (v[i], v[j]);
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Distance from `p` to the polygon's outline.
    pub fn edge_distance(&self, p: Pt) -> f64 {
        let v = &self.0;
        (0..v.len())
            .map(|i| segment_distance(p, v[i], v[(i + 1) % v.len()]))
            .fold(f64::INFINITY, f64::min)
    }

    /// Rectangle with corners `a`..`d` in order.
    pub fn quad(c: [Pt; 4]) -> Self {
        Self(c.to_vec())
    }
}

fn segment_distance(p: Pt, a: Pt, b: Pt) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    ((p[0] - a[0] - t * dx).powi(2) + (p[1] - a[1] - t * dy).powi(2)).sqrt()
}

/// A named region and what kind of place it is (`center`, `open_arm`, `closed_arm`, …).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub name: String,
    pub kind: String,
    pub polygon: Polygon,
}

/// The cached geometry of one camera setup, in top-image pixels.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layout {
    pub rig: String,
    /// How an automatic layout was found ([`METHOD`]); an older one is found again.
    pub method: u32,
    /// `auto` (found by the code) or `manual` (edited by hand; never overwritten).
    pub source: String,
    /// `ok`, or `needs_review` when the code could not decide.
    pub status: String,
    pub reason: String,
    pub image_size: [u32; 2],
    /// The floor's corners in the top image, in order around the floor.
    pub floor: Vec<Pt>,
    pub zones: Vec<Zone>,
    /// How an automatic arena was found: `walls` or `brightness` ([`find_arena`]).
    pub finder: String,
    /// Wall score of each arm (plus maze).
    pub wall_scores: std::collections::BTreeMap<String, f64>,
}

impl Layout {
    pub fn zone(&self, p: Pt) -> Option<&Zone> {
        self.zones.iter().find(|z| z.polygon.contains(p))
    }

    /// Inside any zone.
    pub fn covers(&self, p: Pt) -> bool {
        self.zones.iter().any(|z| z.polygon.contains(p))
    }

    /// Distance from `p` to the nearest zone, 0 inside one.
    pub fn outside_by(&self, p: Pt) -> f64 {
        if self.covers(p) {
            return 0.0;
        }
        self.zones
            .iter()
            .map(|z| z.polygon.edge_distance(p))
            .fold(f64::INFINITY, f64::min)
    }

    pub fn load(path: &Path) -> Result<Option<Self>> {
        match std::fs::read(path) {
            Ok(bytes) => {
                Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
                    format!("cannot read {}", path.display())
                })?))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("cannot open {}", path.display())),
        }
    }
}

/// The current way automatic layouts are found; a cached layout from an older method is found
/// again. 1 took the largest bright region, which can be the desk around the arena. 2 took the
/// smallest bright region holding the track, which merges with a table as bright as the floor.
/// 3 ([`find_arena`]) also builds the arena from the wall-enclosed regions the animal walks in,
/// checks the box shape, and avoids regions that run off the image.
pub const METHOD: u32 = 3;

/// The component of `mask` containing pixel `seed` (8-connected), as a 0/1 mask.
fn component_at(mask: &Image, seed: usize) -> Image {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let mut out = vec![0u8; w * h];
    if mask.pixels[seed] == 0 {
        return Image {
            width: mask.width,
            height: mask.height,
            pixels: out,
        };
    }
    let mut stack = vec![seed];
    out[seed] = 1;
    while let Some(i) = stack.pop() {
        let (x, y) = ((i % w) as i64, (i / w) as i64);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if mask.pixels[j] != 0 && out[j] == 0 {
                    out[j] = 1;
                    stack.push(j);
                }
            }
        }
    }
    Image {
        width: mask.width,
        height: mask.height,
        pixels: out,
    }
}

/// The arena's shape, for checking candidate regions: a box fills most of its oriented rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// A rectangular box (open field, three-chamber, a grooming container).
    Box,
    /// Anything else (a plus maze).
    Any,
}

/// An arena found in the background, and how sure the finder is about it.
#[derive(Clone, Debug)]
pub struct Found {
    /// The arena, walls included, as a 0/1 mask.
    pub mask: Image,
    /// `walls` (regions enclosed by edges that the animal walks in) or `brightness`.
    pub method: &'static str,
    /// For a [`Shape::Box`], the floor's corners in order around it ([`box_floor`]).
    pub floor: Option<[Pt; 4]>,
    /// Empty when the arena is certain; otherwise why it should be checked by eye.
    pub doubts: Vec<String>,
}

/// The arena around the animal ([`find_arena`] for any shape), or `None` without a usable track.
pub fn arena_around(bg: &Image, track: &[Pt]) -> Option<Image> {
    find_arena(bg, track, Shape::Any).map(|found| found.mask)
}

/// The arena as the smallest region of the background that encloses the animal's track.
///
/// A table, a sheet or a bench around the arena is often as bright as the arena floor and joins it
/// in a plain threshold, but the animal never walks on it. So the candidates are:
///
/// * **walls** — the background's edges (gradient above a quantile, widened by a few pixels) cut it
///   into regions; the arena is the union of the regions the track visits, with the walls between
///   them closed and the holes (cups) filled. Partitions with doorways (three-chamber) join up.
/// * **brightness** — over grey-level thresholds of both polarities, the region around the track's
///   median position, holes filled.
///
/// A candidate must hold ≥ [`TRACK_INSIDE`] of the track within [`TRACK_SLACK`] of its edge (a
/// mouse rearing over a wall, and the odd stray detection, land outside the floor), cover ≤ 95% of
/// the image and, for a [`Shape::Box`], fill ≥ 80% of its oriented rectangle. The smallest valid candidate wins: anything that also holds the table is
/// larger. Candidates that run off the image (tables usually do) are used only when nothing else
/// fits, and the result then says so in [`Found::doubts`].
pub fn find_arena(bg: &Image, track: &[Pt], shape: Shape) -> Option<Found> {
    find_arena_with(bg, track, shape, TRACK_INSIDE)
}

/// Share of the track a candidate must hold. Real tracks put 2–3% of positions past the floor's
/// edge (rearing over walls, stray detections): the 3 Chamber box held 96.7–97.6%.
pub const TRACK_INSIDE: f64 = 0.95;
/// How far past a candidate's edge a position still counts as inside, as a share of the width.
pub const TRACK_SLACK: f64 = 0.03;

fn find_arena_with(bg: &Image, track: &[Pt], shape: Shape, inside: f64) -> Option<Found> {
    let (w, h) = (bg.width as usize, bg.height as usize);
    if track.len() < 20 || w < 8 || h < 8 {
        return None;
    }
    let mut candidates: Vec<(Image, &'static str)> = Vec::new();
    wall_candidates(bg, track, &mut candidates);
    brightness_candidates(bg, track, &mut candidates);

    let slack = ((w as f64 * TRACK_SLACK).round() as u32).max(3) | 1;
    let mut valid: Vec<(usize, bool, Image, &'static str)> = Vec::new();
    for (mask, method) in candidates {
        let area = mask.pixels.iter().filter(|p| **p > 0).count();
        if area == 0 || area as f64 > 0.95 * (w * h) as f64 {
            continue;
        }
        if shape == Shape::Box && rect_fill(&mask) < 0.8 {
            continue;
        }
        if share_inside(&fast_dilate(&mask, slack), track) < inside {
            continue;
        }
        valid.push((area, border_share(&mask) > 0.01, mask, method));
    }
    // Inside the image first, then the smallest.
    valid.sort_by_key(|(area, off_image, _, _)| (*off_image, *area));
    let (_, off_image, mask, method) = valid.into_iter().next()?;
    let floor = (shape == Shape::Box)
        .then(|| box_floor(bg, &mask).or_else(|| floor_from_mask(&mask)))
        .flatten();
    let mut doubts = Vec::new();
    if off_image {
        doubts.push(
            "the arena runs off the image, so it may include the table; check the zones picture"
                .to_owned(),
        );
    }
    // The animal explores most of a box; a region far larger than the track is suspect.
    if shape == Shape::Box {
        if let (Some(arena), Some(walked)) = (floor, track_box(track)) {
            let (a, t) = (side_lengths(&arena), side_lengths(&walked));
            if t[0] < 0.5 * a[0] || t[1] < 0.5 * a[1] {
                doubts.push(format!(
                    "the animal walked only {:.0}% × {:.0}% of the arena's extent; check that it is not the table",
                    100.0 * t[0] / a[0],
                    100.0 * t[1] / a[1]
                ));
            }
        }
    }
    Some(Found {
        mask,
        method,
        floor,
        doubts,
    })
}

/// Regions enclosed by the background's edges that the track visits, over a few edge strengths
/// and wall widths.
fn wall_candidates(bg: &Image, track: &[Pt], out: &mut Vec<(Image, &'static str)>) {
    let (w, h) = (bg.width as usize, bg.height as usize);
    let (gx, gy) = sobel(&super::maze::blur5(bg));
    let magnitude: Vec<f64> = gx.iter().zip(&gy).map(|(a, b)| a.hypot(*b)).collect();
    let mut sorted = magnitude.clone();
    sorted.sort_by(f64::total_cmp);
    let scale = (w as f64 / 640.0).max(0.25);
    let close = ((w as f64 * 0.03).round() as u32).max(3) | 1;
    let min_hits = (track.len() / 200).max(3);
    for q in [0.80, 0.88, 0.94] {
        let t = sorted[((sorted.len() - 1) as f64 * q) as usize];
        if t <= 0.0 {
            continue;
        }
        let edges = Image {
            width: bg.width,
            height: bg.height,
            pixels: magnitude.iter().map(|m| u8::from(*m > t)).collect(),
        };
        for r in [1.0, 2.0, 3.0, 5.0] {
            let k = (2.0 * (r * scale).round() + 1.0) as u32;
            let walls = fast_dilate(&edges, k);
            let free = Image {
                width: bg.width,
                height: bg.height,
                pixels: walls.pixels.iter().map(|p| u8::from(*p == 0)).collect(),
            };
            let (labels, n) = label(&free);
            let mut hits = vec![0usize; n + 1];
            for p in track {
                let (x, y) = (p[0].round() as i64, p[1].round() as i64);
                if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                    hits[labels[y as usize * w + x as usize] as usize] += 1;
                }
            }
            let chosen: Vec<bool> = hits
                .iter()
                .enumerate()
                .map(|(l, c)| l > 0 && *c >= min_hits)
                .collect();
            if !chosen.iter().any(|c| *c) {
                continue;
            }
            let walked = Image {
                width: bg.width,
                height: bg.height,
                pixels: labels
                    .iter()
                    .map(|l| u8::from(chosen[*l as usize]))
                    .collect(),
            };
            // Give back the wall band, join regions across partitions, fill cups.
            let region = fill_holes(&fast_close(&fast_dilate(&walked, k), close));
            out.push((region, "walls"));
        }
    }
}

/// The region around the track's median position over grey-level thresholds of both polarities.
fn brightness_candidates(bg: &Image, track: &[Pt], out: &mut Vec<(Image, &'static str)>) {
    let (w, h) = (bg.width as usize, bg.height as usize);
    let mut xs: Vec<f64> = track.iter().map(|p| p[0]).collect();
    let mut ys: Vec<f64> = track.iter().map(|p| p[1]).collect();
    let seed = [
        crate::multiview::dense::median(&mut xs).clamp(0.0, (w - 1) as f64) as usize,
        crate::multiview::dense::median(&mut ys).clamp(0.0, (h - 1) as f64) as usize,
    ];
    let seed = seed[1] * w + seed[0];
    for t in (8..=248).step_by(8) {
        for bright in [true, false] {
            let mask = Image {
                width: bg.width,
                height: bg.height,
                pixels: bg
                    .pixels
                    .iter()
                    .map(|p| u8::from((*p > t) == bright))
                    .collect(),
            };
            let region = fill_holes(&component_at(&morph_open(&mask, 3), seed));
            if region.pixels.iter().any(|p| *p > 0) {
                out.push((region, "brightness"));
            }
        }
    }
}

/// Sobel gradients `(gx, gy)`.
fn sobel(img: &Image) -> (Vec<f64>, Vec<f64>) {
    let (w, h) = (img.width as i64, img.height as i64);
    let at = |x: i64, y: i64| {
        f64::from(img.pixels[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize])
    };
    let mut gx = vec![0.0; (w * h) as usize];
    let mut gy = vec![0.0; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            gx[i] = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            gy[i] = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
        }
    }
    (gx, gy)
}

/// The floor of a box arena: oriented along its walls (the dominant edge direction around the
/// region, modulo 90°), spanning the region, with each side moved out to the first straight wall
/// edge within 12% of the box size. The animal may not have walked right up to every wall, and
/// the region's own outline is ragged; the walls are straight.
pub fn box_floor(bg: &Image, mask: &Image) -> Option<[Pt; 4]> {
    let (w, h) = (bg.width as usize, bg.height as usize);
    let (gx, gy) = sobel(&super::maze::blur5(bg));
    let mag: Vec<f64> = gx.iter().zip(&gy).map(|(a, b)| a.hypot(*b)).collect();
    let mut sorted = mag.clone();
    sorted.sort_by(f64::total_cmp);
    let strong = sorted[((sorted.len() - 1) as f64 * 0.9) as usize];
    // Edges in a band around the region's outline vote for the wall direction.
    let band = ((w as f64 * 0.03).round() as u32).max(3) | 1;
    let (outer, inner) = (fast_dilate(mask, band), fast_erode(mask, band));
    let mut votes = [0.0f64; 90];
    for i in 0..w * h {
        if outer.pixels[i] > 0 && inner.pixels[i] == 0 && mag[i] > strong {
            let deg = gy[i].atan2(gx[i]).to_degrees().rem_euclid(90.0);
            votes[(deg as usize).min(89)] += mag[i];
        }
    }
    let smooth = |k: usize| (0..5).map(|d| votes[(k + 88 + d) % 90]).sum::<f64>();
    let peak = (0..90).max_by(|a, b| smooth(*a).total_cmp(&smooth(*b)))?;
    // Sub-degree: the vote-weighted mean around the peak.
    let (mut sum, mut weight) = (0.0, 0.0);
    for d in -2i64..=2 {
        let k = (peak as i64 + d).rem_euclid(90) as usize;
        sum += (peak as f64 + d as f64) * votes[k];
        weight += votes[k];
    }
    let theta = if weight > 0.0 {
        sum / weight
    } else {
        peak as f64
    }
    .to_radians();
    let (u, v) = ([theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]);
    // The region's extent along both axes (0.5%–99.5%, so a spur does not count).
    let pts: Vec<(f64, f64)> = (0..w * h)
        .filter(|i| mask.pixels[*i] > 0)
        .map(|i| {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            (x * u[0] + y * u[1], x * v[0] + y * v[1])
        })
        .collect();
    if pts.len() < 16 {
        return None;
    }
    let quantiles = |vals: &mut Vec<f64>| {
        vals.sort_by(f64::total_cmp);
        let at = |q: f64| vals[((vals.len() - 1) as f64 * q) as usize];
        (at(0.005), at(0.995))
    };
    let (mut su, mut sv): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
    let ((mut u0, mut u1), (mut v0, mut v1)) = (quantiles(&mut su), quantiles(&mut sv));
    // Edge strength across each axis, summed over the middle 80% of the other: a wall is a peak.
    let profile = |a: Pt, b: Pt, b0: f64, b1: f64| -> (f64, Vec<f64>) {
        let corners = [
            [0.0, 0.0],
            [w as f64, 0.0],
            [0.0, h as f64],
            [w as f64, h as f64],
        ];
        let lo = corners
            .iter()
            .map(|c| c[0] * a[0] + c[1] * a[1])
            .fold(f64::INFINITY, f64::min)
            .floor();
        let hi = corners
            .iter()
            .map(|c| c[0] * a[0] + c[1] * a[1])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut p = vec![0.0; (hi - lo) as usize + 2];
        let (m0, m1) = (b0 + 0.1 * (b1 - b0), b1 - 0.1 * (b1 - b0));
        for i in 0..w * h {
            let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
            let t = x * b[0] + y * b[1];
            if t < m0 || t > m1 {
                continue;
            }
            let s = x * a[0] + y * a[1];
            p[(s - lo) as usize] += (gx[i] * a[0] + gy[i] * a[1]).abs();
        }
        (lo, p)
    };
    let snap = |lo: f64, p: &[f64], from: f64, to: f64, outward: f64| -> f64 {
        let reach = 0.12 * (to - from).abs();
        let idx = |s: f64| ((s - lo).round().max(0.0) as usize).min(p.len() - 1);
        let (a, b) = if outward > 0.0 {
            (idx(to - 0.03 * (to - from).abs()), idx(to + reach))
        } else {
            (idx(to - reach), idx(to + 0.03 * (to - from).abs()))
        };
        let top = p[a.min(b)..=a.max(b)].iter().cloned().fold(0.0, f64::max);
        if top <= 0.0 {
            return to;
        }
        let start = idx(to);
        let mut i = start;
        loop {
            let local = (i == 0 || p[i] >= p[i - 1]) && (i + 1 >= p.len() || p[i] >= p[i + 1]);
            if local && p[i] >= 0.5 * top {
                return lo + i as f64;
            }
            let next = i as i64 + outward as i64;
            if next < a.min(b) as i64 || next > a.max(b) as i64 {
                return to;
            }
            i = next as usize;
        }
    };
    let (lo_u, pu) = profile(u, v, v0, v1);
    let (lo_v, pv) = profile(v, u, u0, u1);
    let (nu0, nu1) = (snap(lo_u, &pu, u1, u0, -1.0), snap(lo_u, &pu, u0, u1, 1.0));
    let (nv0, nv1) = (snap(lo_v, &pv, v1, v0, -1.0), snap(lo_v, &pv, v0, v1, 1.0));
    (u0, u1, v0, v1) = (nu0.min(u0), nu1.max(u1), nv0.min(v0), nv1.max(v1));
    let at = |a: f64, b: f64| [a * u[0] + b * v[0], a * u[1] + b * v[1]];
    Some([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)])
}

/// 8-connected labels of a 0/1 mask: 0 for background, 1..=n for the components.
fn label(mask: &Image) -> (Vec<u32>, usize) {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let mut labels = vec![0u32; w * h];
    let mut n = 0u32;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if mask.pixels[start] == 0 || labels[start] != 0 {
            continue;
        }
        n += 1;
        labels[start] = n;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask.pixels[j] != 0 && labels[j] == 0 {
                        labels[j] = n;
                        stack.push(j);
                    }
                }
            }
        }
    }
    (labels, n as usize)
}

/// Dilation with a `k × k` square in O(pixels), as two running-window passes.
fn fast_dilate(mask: &Image, k: u32) -> Image {
    window(mask, k, true)
}

fn fast_erode(mask: &Image, k: u32) -> Image {
    window(mask, k, false)
}

fn fast_close(mask: &Image, k: u32) -> Image {
    fast_erode(&fast_dilate(mask, k), k)
}

/// One square-window pass: `any` set in the window (dilate) or `all` set (erode; outside the
/// image counts as set, as in OpenCV's default border for erosion).
fn window(mask: &Image, k: u32, any: bool) -> Image {
    let (w, h) = (mask.width as usize, mask.height as usize);
    if k <= 1 || w == 0 || h == 0 {
        return mask.clone();
    }
    let r = (k as usize - 1) / 2;
    let pass = |src: &[u8], len: usize, lines: usize, stride: usize, step: usize| -> Vec<u8> {
        let mut dst = vec![0u8; src.len()];
        for line in 0..lines {
            let base = line * stride;
            let mut prefix = vec![0usize; len + 1];
            for i in 0..len {
                prefix[i + 1] = prefix[i] + usize::from(src[base + i * step] != 0);
            }
            for i in 0..len {
                let (lo, hi) = (i.saturating_sub(r), (i + r + 1).min(len));
                let set = prefix[hi] - prefix[lo];
                let window = 2 * r + 1;
                let out_of_image = window - (hi - lo);
                dst[base + i * step] = if any {
                    u8::from(set > 0)
                } else {
                    u8::from(set + out_of_image == window)
                };
            }
        }
        dst
    };
    let rows = pass(&mask.pixels, w, h, w, 1);
    let both = pass(&rows, h, w, 1, w);
    Image {
        width: mask.width,
        height: mask.height,
        pixels: both,
    }
}

/// Share of `track` positions on the mask.
fn share_inside(mask: &Image, track: &[Pt]) -> f64 {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let inside = track
        .iter()
        .filter(|p| {
            let (x, y) = (p[0] as i64, p[1] as i64);
            x >= 0
                && y >= 0
                && (x as usize) < w
                && (y as usize) < h
                && mask.pixels[y as usize * w + x as usize] > 0
        })
        .count();
    inside as f64 / track.len().max(1) as f64
}

/// Share of the image border that the mask covers.
fn border_share(mask: &Image) -> f64 {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let edge: Vec<usize> = (0..w)
        .flat_map(|x| [x, (h - 1) * w + x])
        .chain((0..h).flat_map(|y| [y * w, y * w + w - 1]))
        .collect();
    edge.iter().filter(|i| mask.pixels[**i] > 0).count() as f64 / edge.len() as f64
}

/// Area of the mask over the area of its oriented rectangle.
fn rect_fill(mask: &Image) -> f64 {
    let area = mask.pixels.iter().filter(|p| **p > 0).count() as f64;
    match floor_from_mask(mask) {
        Some(c) => area / polygon_area(&c).max(1.0),
        None => 0.0,
    }
}

fn polygon_area(c: &[Pt]) -> f64 {
    let n = c.len();
    (0..n)
        .map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1])
        .sum::<f64>()
        .abs()
        / 2.0
}

/// The two side lengths of a quad in order around it.
fn side_lengths(c: &[Pt; 4]) -> [f64; 2] {
    let len = |a: Pt, b: Pt| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    [
        (len(c[0], c[1]) + len(c[3], c[2])) / 2.0,
        (len(c[1], c[2]) + len(c[0], c[3])) / 2.0,
    ]
}

/// The oriented box of the track's 2nd–98th percentile extent, in the same orientation scheme as
/// [`floor_from_mask`] (a stray detection does not stretch it).
fn track_box(track: &[Pt]) -> Option<[Pt; 4]> {
    if track.len() < 20 {
        return None;
    }
    let n = track.len() as f64;
    let c = [
        track.iter().map(|p| p[0]).sum::<f64>() / n,
        track.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in track {
        let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (u, v) = ([theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]);
    let range = |a: Pt| {
        let mut s: Vec<f64> = track
            .iter()
            .map(|p| (p[0] - c[0]) * a[0] + (p[1] - c[1]) * a[1])
            .collect();
        s.sort_by(f64::total_cmp);
        let at = |q: f64| s[((s.len() - 1) as f64 * q) as usize];
        (at(0.02), at(0.98))
    };
    let ((u0, u1), (v0, v1)) = (range(u), range(v));
    let at = |a: f64, b: f64| [c[0] + a * u[0] + b * v[0], c[1] + a * u[1] + b * v[1]];
    Some([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)])
}

/// The homography mapping `src` onto `dst` (four point pairs), row-major 3×3.
pub fn homography(src: &[Pt], dst: &[Pt]) -> Result<[[f64; 3]; 3]> {
    if src.len() != 4 || dst.len() != 4 {
        bail!("a homography needs four point pairs");
    }
    let mut a = Vec::with_capacity(64);
    let mut b = Vec::with_capacity(8);
    for (s, d) in src.iter().zip(dst) {
        a.extend_from_slice(&[s[0], s[1], 1.0, 0.0, 0.0, 0.0, -d[0] * s[0], -d[0] * s[1]]);
        b.push(d[0]);
        a.extend_from_slice(&[0.0, 0.0, 0.0, s[0], s[1], 1.0, -d[1] * s[0], -d[1] * s[1]]);
        b.push(d[1]);
    }
    let h =
        crate::multiview::dense::solve(&a, &b, 8).context("the floor corners are degenerate")?;
    Ok([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
}

pub fn apply(h: &[[f64; 3]; 3], p: Pt) -> Pt {
    let w = h[2][0] * p[0] + h[2][1] * p[1] + h[2][2];
    [
        (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w,
        (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w,
    ]
}

/// Maps top-image pixels to normalised floor coordinates `(u, v)` in `[0, 1]²`.
#[derive(Clone, Debug)]
pub struct Floor {
    h: [[f64; 3]; 3],
    /// Floor length and width, in the units distances are reported in (box units, or the
    /// corners' pixel lengths without a calibration).
    pub size: [f64; 2],
}

impl Floor {
    pub fn uv(&self, image: Pt) -> Pt {
        apply(&self.h, image)
    }

    /// From image corners in order around the floor.
    pub fn from_corners(corners: &[Pt]) -> Result<Self> {
        let h = homography(corners, &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])?;
        let len = |a: Pt, b: Pt| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        Ok(Self {
            h,
            size: [
                (len(corners[0], corners[1]) + len(corners[3], corners[2])) / 2.0,
                (len(corners[1], corners[2]) + len(corners[0], corners[3])) / 2.0,
            ],
        })
    }

    /// The floor of a calibrated box seen by the top camera, through the plane at `height`: a
    /// point on the animal's back sits above the floor, and projecting it with the floor plane
    /// would push positions near the walls outward.
    pub fn from_camera(top: &Camera, width: f64, height: f64) -> Result<Self> {
        let corners = box_corners(width).map(|[x, y]| top.project(&[x, y, height]));
        let mut floor = Self::from_corners(&corners)?;
        floor.size = [1.0, width];
        Ok(floor)
    }
}

/// The floor corners of the calibrated box: length 1 along x, `width` along y, centred.
pub fn box_corners(width: f64) -> [Pt; 4] {
    let w = width / 2.0;
    [[-0.5, -w], [0.5, -w], [0.5, w], [-0.5, w]]
}

/// Box coordinates `(x, y)` → `(u, v)`.
pub fn box_uv(p: Pt, width: f64) -> Pt {
    [p[0] + 0.5, p[1] / width + 0.5]
}

/// The arena as the largest region of the background, polarity chosen so it covers the image
/// centre, cleaned by a close and an open (kernel ~1% of the width) with its holes filled.
pub fn arena_mask(bg: &Image) -> Image {
    let t = crate::algos::background::otsu_threshold(bg);
    let k = ((bg.width as f64 * 0.01).round() as u32).max(3) | 1;
    let pick = |bright: bool| {
        let mask = Image {
            width: bg.width,
            height: bg.height,
            pixels: bg
                .pixels
                .iter()
                .map(|p| u8::from((*p > t) == bright))
                .collect(),
        };
        let mask = morph_open(&morph_close(&mask, k), k);
        fill_holes(&largest(&mask))
    };
    let (w, h) = (bg.width as usize, bg.height as usize);
    let centre = |m: &Image| m.pixels[(h / 2) * w + w / 2] > 0;
    // The arena covers the centre (holes such as a cup filled) and not most of the border.
    let border = |m: &Image| {
        let edge: Vec<usize> = (0..w)
            .flat_map(|x| [x, (h - 1) * w + x])
            .chain((0..h).flat_map(|y| [y * w, y * w + w - 1]))
            .collect();
        edge.iter().filter(|i| m.pixels[**i] > 0).count() as f64 / edge.len() as f64
    };
    let candidates = [pick(true), pick(false)];
    let good = |m: &Image| centre(m) && border(m) < 0.5;
    match candidates.iter().find(|m| good(m)) {
        Some(m) => m.clone(),
        None => candidates
            .into_iter()
            .find(centre)
            .unwrap_or_else(|| pick(true)),
    }
}

/// The largest 8-connected component of a mask, as a 0/1 mask.
pub fn largest(mask: &Image) -> Image {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let mut label = vec![0u32; w * h];
    let mut best: (u32, usize) = (0, 0);
    let mut next = 0;
    let mut stack = Vec::new();
    for start in 0..w * h {
        if mask.pixels[start] == 0 || label[start] != 0 {
            continue;
        }
        next += 1;
        let mut size = 0;
        label[start] = next;
        stack.push(start);
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask.pixels[j] != 0 && label[j] == 0 {
                        label[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        if size > best.1 {
            best = (next, size);
        }
    }
    Image {
        width: mask.width,
        height: mask.height,
        pixels: label
            .iter()
            .map(|l| u8::from(*l == best.0 && best.0 != 0))
            .collect(),
    }
}

/// Background pixels not reachable from the border become foreground (0/1 mask).
pub fn fill_holes(mask: &Image) -> Image {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for x in 0..w {
        stack.push(x);
        stack.push((h - 1) * w + x);
    }
    for y in 0..h {
        stack.push(y * w);
        stack.push(y * w + w - 1);
    }
    while let Some(i) = stack.pop() {
        if outside[i] || mask.pixels[i] > 0 {
            continue;
        }
        outside[i] = true;
        let (x, y) = (i % w, i / w);
        if x > 0 {
            stack.push(i - 1);
        }
        if x + 1 < w {
            stack.push(i + 1);
        }
        if y > 0 {
            stack.push(i - w);
        }
        if y + 1 < h {
            stack.push(i + w);
        }
    }
    Image {
        width: mask.width,
        height: mask.height,
        pixels: outside.iter().map(|o| u8::from(!*o)).collect(),
    }
}

/// The floor of a rectangular arena in the top image: the oriented box (by PCA) of the arena
/// mask, corners in order around it.
pub fn floor_from_mask(mask: &Image) -> Option<[Pt; 4]> {
    let w = mask.width as usize;
    let pts: Vec<Pt> = mask
        .pixels
        .iter()
        .enumerate()
        .filter(|(_, p)| **p > 0)
        .map(|(i, _)| [(i % w) as f64 + 0.5, (i / w) as f64 + 0.5])
        .collect();
    if pts.len() < 16 {
        return None;
    }
    let n = pts.len() as f64;
    let c = [
        pts.iter().map(|p| p[0]).sum::<f64>() / n,
        pts.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in &pts {
        let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (u, v) = ([theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]);
    let proj = |p: &Pt, a: Pt| (p[0] - c[0]) * a[0] + (p[1] - c[1]) * a[1];
    let range = |a: Pt| {
        pts.iter()
            .map(|p| proj(p, a))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            })
    };
    let ((u0, u1), (v0, v1)) = (range(u), range(v));
    let at = |a: f64, b: f64| [c[0] + a * u[0] + b * v[0], c[1] + a * u[1] + b * v[1]];
    Some([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)])
}

/// Per-pixel median of grey frames (`u8`, so memory stays at one byte per pixel per frame).
pub fn median_u8(frames: &[Image]) -> Option<Image> {
    let first = frames.first()?;
    let n = frames.len();
    let mut out = vec![0u8; first.pixels.len()];
    let mut hist = [0u32; 256];
    for (i, o) in out.iter_mut().enumerate() {
        hist.fill(0);
        for f in frames {
            hist[f.pixels[i] as usize] += 1;
        }
        let mut seen = 0;
        for (v, c) in hist.iter().enumerate() {
            seen += *c as usize;
            if seen * 2 > n {
                *o = v as u8;
                break;
            }
        }
    }
    Some(Image {
        width: first.width,
        height: first.height,
        pixels: out,
    })
}

pub fn to_u8(g: &GrayF) -> Image {
    Image {
        width: g.width,
        height: g.height,
        pixels: g
            .pixels
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    }
}

/// The layout drawn over the background, for a check by eye: zones outlined in colour with their
/// names' initials, the floor in white.
pub fn render(bg: &Image, layout: &Layout, track: &[Pt]) -> crate::algos::image::Bgr {
    let (w, h) = (bg.width as i64, bg.height as i64);
    let mut px: Vec<u8> = bg
        .pixels
        .iter()
        .flat_map(|p| [*p / 2, *p / 2, *p / 2])
        .collect();
    let mut dot = |x: i64, y: i64, c: [u8; 3]| {
        if (0..w).contains(&x) && (0..h).contains(&y) {
            let i = ((y * w + x) * 3) as usize;
            px[i..i + 3].copy_from_slice(&c);
        }
    };
    let mut line = |a: Pt, b: Pt, c: [u8; 3]| {
        let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let (x, y) = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
                dot(x as i64 + dx, y as i64 + dy, c);
            }
        }
    };
    let outline = |poly: &[Pt], c: [u8; 3], line: &mut dyn FnMut(Pt, Pt, [u8; 3])| {
        for i in 0..poly.len() {
            line(poly[i], poly[(i + 1) % poly.len()], c);
        }
    };
    // The animal's track, so the zones can be checked against where it went.
    for p in track {
        line(*p, *p, [0, 200, 230]);
    }
    outline(&layout.floor, [255, 255, 255], &mut line);
    for z in &layout.zones {
        let colour = match z.kind.as_str() {
            "open_arm" => [80, 200, 80],
            "closed_arm" => [60, 60, 230],
            "center" => [0, 200, 255],
            _ => [230, 160, 60],
        };
        outline(&z.polygon.0, colour, &mut line);
    }
    crate::algos::image::Bgr {
        width: bg.width,
        height: bg.height,
        pixels: px,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygons_and_homographies() {
        let sq = Polygon::quad([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        assert!(sq.contains([1.0, 1.0]));
        assert!(!sq.contains([3.0, 1.0]));
        assert!((sq.edge_distance([3.0, 1.0]) - 1.0).abs() < 1e-12);
        let quad = [[10.0, 20.0], [110.0, 30.0], [100.0, 120.0], [5.0, 100.0]];
        let floor = Floor::from_corners(&quad).unwrap();
        for (corner, uv) in quad
            .iter()
            .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        {
            let got = floor.uv(*corner);
            assert!((got[0] - uv[0]).abs() < 1e-9 && (got[1] - uv[1]).abs() < 1e-9);
        }
    }

    /// A bright desk with a walled box on it: the box, which holds the track, is the arena — not
    /// the desk, though it is larger and brighter.
    #[test]
    fn the_arena_is_the_region_that_holds_the_track_not_the_desk() {
        let (w, h) = (200u32, 160u32);
        let mut img = Image {
            width: w,
            height: h,
            pixels: vec![20; (w * h) as usize],
        };
        let mut paint = |x0: u32, x1: u32, y0: u32, y1: u32, v: u8| {
            for y in y0..y1 {
                for x in x0..x1 {
                    img.pixels[(y * w + x) as usize] = v;
                }
            }
        };
        paint(10, 190, 10, 150, 200); // the desk
        paint(70, 140, 50, 100, 90); // the box's walls…
        paint(74, 136, 54, 96, 180); // …and its floor
        let track: Vec<Pt> = (0..200)
            .map(|i| {
                let a = i as f64 * 0.37;
                [105.0 + 28.0 * a.cos(), 75.0 + 18.0 * a.sin()]
            })
            .collect();
        let region = arena_around(&img, &track).unwrap();
        let corners = floor_from_mask(&region).unwrap();
        let xs: Vec<f64> = corners.iter().map(|c| c[0]).collect();
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (lo - 74.0).abs() < 4.0 && (hi - 136.0).abs() < 4.0,
            "{corners:?}"
        );
        // Without a track there is nothing to go by.
        assert!(arena_around(&img, &[]).is_none());
    }

    /// A top view like the three-chamber rigs: a textured table pad as bright as the floor (236 ± 10
    /// against 245, as in the 3 Chambers Video top views),
    /// grey rim walls, two partitions with doorways, two dark cups, a shadow along one wall, and a
    /// mouse that hugs the walls. Returns the image, the track and the floor's inner rectangle.
    fn three_chamber_scene(table_to_border: bool) -> (Image, Vec<Pt>, [f64; 4]) {
        let (w, h) = (320u32, 260u32);
        let mut seed = 12345u64;
        let mut noise = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) % 21) as i32 - 10
        };
        let mut img = Image {
            width: w,
            height: h,
            pixels: vec![40; (w * h) as usize],
        };
        let (t0, t1) = if table_to_border {
            (0, w)
        } else {
            (40, w - 40)
        };
        for y in 30..h - 20 {
            for x in t0..t1 {
                img.pixels[(y * w + x) as usize] = (236 + noise()).clamp(0, 255) as u8;
            }
        }
        let (fx0, fx1, fy0, fy1) = (95u32, 235u32, 80u32, 200u32);
        let mut paint = |x0: u32, x1: u32, y0: u32, y1: u32, v: u8| {
            for y in y0..y1 {
                for x in x0..x1 {
                    img.pixels[(y * w + x) as usize] = v;
                }
            }
        };
        paint(fx0 - 4, fx1 + 4, fy0 - 4, fy1 + 4, 140); // rim walls
        paint(fx0, fx1, fy0, fy1, 245); // floor
        for x in [141u32, 188] {
            paint(x, x + 3, fy0, fy1, 150); // partitions…
            paint(x, x + 3, 128, 152, 245); // …with doorways
        }
        for (cx, cy) in [(118i64, 140i64), (212, 140)] {
            for y in cy - 12..cy + 12 {
                for x in cx - 12..cx + 12 {
                    let d = (((x - cx).pow(2) + (y - cy).pow(2)) as f64).sqrt();
                    if d < 11.0 {
                        img.pixels[(y as u32 * w + x as u32) as usize] =
                            if d < 8.0 { 60 } else { 170 };
                    }
                }
            }
        }
        for y in fy1 - 30..fy1 {
            for x in fx0..fx1 {
                // A shadow darkening toward the bottom wall, as dark as the table pad's texture.
                if img.pixels[(y * w + x) as usize] == 245 {
                    img.pixels[(y * w + x) as usize] = (245 - 2 * (y - (fy1 - 30))) as u8;
                }
            }
        }
        // The mouse visits all three chambers and runs along every wall, 2–3 px from it.
        let mut track = Vec::new();
        for i in 0..600 {
            let s = i as f64 / 600.0;
            let (x0, x1, y0, y1) = (
                fx0 as f64 + 2.5,
                fx1 as f64 - 2.5,
                fy0 as f64 + 2.5,
                fy1 as f64 - 2.5,
            );
            let per = 2.0 * ((x1 - x0) + (y1 - y0));
            let d = s * per;
            let p = if d < x1 - x0 {
                [x0 + d, y0]
            } else if d < (x1 - x0) + (y1 - y0) {
                [x1, y0 + d - (x1 - x0)]
            } else if d < 2.0 * (x1 - x0) + (y1 - y0) {
                [x1 - (d - (x1 - x0) - (y1 - y0)), y1]
            } else {
                [x0, y1 - (d - 2.0 * (x1 - x0) - (y1 - y0))]
            };
            track.push(p);
            track.push([
                fx0 as f64 + 10.0 + s * 120.0,
                140.0 + 30.0 * (s * 40.0).sin(),
            ]);
        }
        (img, track, [fx0 as f64, fx1 as f64, fy0 as f64, fy1 as f64])
    }

    fn extent(c: &[Pt; 4]) -> [f64; 4] {
        let xs = c.iter().map(|p| p[0]);
        let ys = c.iter().map(|p| p[1]);
        [
            xs.clone().fold(f64::INFINITY, f64::min),
            xs.fold(f64::NEG_INFINITY, f64::max),
            ys.clone().fold(f64::INFINITY, f64::min),
            ys.fold(f64::NEG_INFINITY, f64::max),
        ]
    }

    /// The table pad is as bright as the floor and joins it at every grey level above the rim's,
    /// so a threshold alone finds the table. The walls the mouse stays inside find the box.
    #[test]
    fn a_table_as_bright_as_the_floor_is_not_the_arena() {
        for table_to_border in [false, true] {
            let (img, track, floor) = three_chamber_scene(table_to_border);
            let found = find_arena(&img, &track, Shape::Box).expect("an arena");
            let got = extent(&found.floor.unwrap());
            // The box: its floor (a shadowed wall may cost a few pixels) up to its 4 px rim, and nothing
            // of the table (≥ 50 px further out).
            for (g, f, outward) in [
                (got[0], floor[0], -1.0),
                (got[1], floor[1], 1.0),
                (got[2], floor[2], -1.0),
                (got[3], floor[3], 1.0),
            ] {
                let off = (g - f) * outward;
                assert!(
                    (-4.0..=7.0).contains(&off),
                    "table_to_border={table_to_border}: {got:?} vs floor {floor:?} ({})",
                    found.method
                );
            }
            assert!(found.doubts.is_empty(), "{:?}", found.doubts);
            // The cups are inside the arena.
            assert_eq!(found.mask.pixels[(140 * img.width + 118) as usize], 1);
        }
    }

    /// A track in one corner of a large box is suspicious: the region is reported for review.
    #[test]
    fn an_arena_much_larger_than_the_track_is_flagged() {
        let (img, _, floor) = three_chamber_scene(false);
        let track: Vec<Pt> = (0..200)
            .map(|i| {
                let a = i as f64 * 0.3;
                [
                    floor[0] + 15.0 + 8.0 * a.cos(),
                    floor[2] + 15.0 + 8.0 * a.sin(),
                ]
            })
            .collect();
        let found = find_arena(&img, &track, Shape::Box).expect("an arena");
        assert!(!found.doubts.is_empty());
    }

    #[test]
    fn square_windows_match_the_reference_morphology() {
        let mut img = Image {
            width: 23,
            height: 17,
            pixels: vec![0; 23 * 17],
        };
        for (i, p) in img.pixels.iter_mut().enumerate() {
            *p = u8::from((i * 7919) % 11 == 0);
        }
        for k in [1u32, 3, 5, 7] {
            assert_eq!(
                fast_dilate(&img, k).pixels,
                crate::algos::blob::dilate_rect(&img, k).pixels
            );
            assert_eq!(
                fast_erode(&img, k).pixels,
                crate::algos::blob::erode_rect(&img, k).pixels
            );
        }
    }

    /// Checks the finder on a real background: `ARENA_BG` (PNG), `ARENA_TRACK` (CSV with `x,y`
    /// columns, top-image pixels); writes the picture to `ARENA_OUT`.
    #[test]
    #[ignore = "needs a background and a track from a real recording"]
    fn real_background() {
        let bg = image::open(std::env::var("ARENA_BG").unwrap())
            .unwrap()
            .to_luma8();
        let bg = Image {
            width: bg.width(),
            height: bg.height(),
            pixels: bg.into_raw(),
        };
        let text = std::fs::read_to_string(std::env::var("ARENA_TRACK").unwrap()).unwrap();
        let mut lines = text.lines();
        let head: Vec<&str> = lines.next().unwrap().split(',').collect();
        let (xi, yi) = (
            head.iter().position(|c| *c == "x").unwrap(),
            head.iter().position(|c| *c == "y").unwrap(),
        );
        let track: Vec<Pt> = lines
            .filter_map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                Some([f.get(xi)?.parse().ok()?, f.get(yi)?.parse().ok()?])
            })
            .collect();
        if std::env::var_os("ARENA_DEBUG").is_some() {
            let mut walls = Vec::new();
            wall_candidates(&bg, &track, &mut walls);
            let slack = ((bg.width as f64 * 0.02).round() as u32).max(3) | 1;
            for (i, (m, _)) in walls.iter().enumerate() {
                let area = m.pixels.iter().filter(|p| **p > 0).count();
                println!(
                    "walls #{i}: area {area} fill {:.2} inside {:.3} border {:.3} extent {:?}",
                    rect_fill(m),
                    share_inside(&fast_dilate(m, slack), &track),
                    border_share(m),
                    floor_from_mask(m).map(|c| extent(&c))
                );
            }
        }
        let found = find_arena(&bg, &track, Shape::Box).expect("an arena");
        let floor = found.floor.unwrap();
        println!(
            "finder {} floor {floor:?} doubts {:?}",
            found.method, found.doubts
        );
        // What brightness alone (method 2) would have taken, for comparison.
        let mut bright = Vec::new();
        brightness_candidates(&bg, &track, &mut bright);
        let slack = ((bg.width as f64 * 0.02).round() as u32).max(3) | 1;
        if let Some((m, _)) = bright
            .iter()
            .filter(|(m, _)| share_inside(&fast_dilate(m, slack), &track) >= 0.98)
            .min_by_key(|(m, _)| m.pixels.iter().filter(|p| **p > 0).count())
        {
            println!(
                "brightness only: {:?}",
                floor_from_mask(m).map(|c| extent(&c))
            );
        }
        let layout = Layout {
            floor: floor.to_vec(),
            image_size: [bg.width, bg.height],
            ..Layout::default()
        };
        let pic = render(&bg, &layout, &track);
        let rgb: Vec<u8> = pic
            .pixels
            .chunks(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        image::RgbImage::from_raw(pic.width, pic.height, rgb)
            .unwrap()
            .save(std::env::var("ARENA_OUT").unwrap())
            .unwrap();
    }

    #[test]
    fn the_arena_mask_and_its_floor_are_found() {
        let (w, h) = (120u32, 100u32);
        let mut img = Image {
            width: w,
            height: h,
            pixels: vec![30; (w * h) as usize],
        };
        for y in 20..80 {
            for x in 20..100 {
                img.pixels[(y * w + x) as usize] = 200;
            }
        }
        // A dark cup inside the arena is a hole, and holes are filled.
        for y in 45..55 {
            for x in 55..65 {
                img.pixels[(y * w + x) as usize] = 30;
            }
        }
        let mask = arena_mask(&img);
        assert_eq!(mask.pixels[(50 * w + 60) as usize], 1);
        assert_eq!(mask.pixels[(5 * w + 5) as usize], 0);
        let corners = floor_from_mask(&mask).unwrap();
        let xs: Vec<f64> = corners.iter().map(|c| c[0]).collect();
        let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (lo - 20.0).abs() < 1.5 && (hi - 100.0).abs() < 1.5,
            "{corners:?}"
        );
        let frames = vec![
            img.clone(),
            img.clone(),
            Image {
                pixels: vec![0; img.pixels.len()],
                ..img.clone()
            },
        ];
        assert_eq!(median_u8(&frames).unwrap().pixels, img.pixels);
    }
}
