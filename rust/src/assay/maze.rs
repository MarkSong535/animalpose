//! The plus maze, found once per setup from the median background of the empty maze.
//!
//! 1. Clean-up: CLAHE, then a 5×5 Gaussian blur.
//! 2. Maze mask: Otsu threshold with the polarity that covers the image centre, close and open,
//!    largest component, holes filled ([`super::arena::arena_mask`]).
//! 3. Arm axes: the mask's edge orientations cluster into two directions, which must be
//!    90° ± `maze.axis_tol_deg` apart (otherwise the setup is flagged).
//! 4. Each arm's centre line: a least-squares fit to the mask's midpoints across that direction.
//! 5. Arm width `w`: the median width across each axis, the crossing excluded.
//! 6. Centre: a `w × w` square at the crossing of the centre lines. Arms: the four rectangles
//!    beyond it.
//! 7. Open or closed: walls show as strong edges along an arm plus a band of different brightness.
//!    The axis with the higher wall score is closed if it beats the other by `maze.wall_ratio`;
//!    otherwise the layout is `needs_review` and nothing is guessed.

use std::collections::BTreeMap;

use super::arena::{arena_mask, Layout, Polygon, Pt, Zone};
use super::config::Maze;
use crate::algos::image::Image;

/// Contrast-limited adaptive histogram equalisation (`tiles × tiles`, clip limit relative to a
/// flat histogram as in OpenCV), bilinear between tile centres.
pub fn clahe(img: &Image, clip: f64, tiles: usize) -> Image {
    let (w, h) = (img.width as usize, img.height as usize);
    let tiles = tiles.max(1);
    let (tw, th) = (w.div_ceil(tiles), h.div_ceil(tiles));
    let mut luts = vec![[0u8; 256]; tiles * tiles];
    for ty in 0..tiles {
        for tx in 0..tiles {
            let mut hist = [0f64; 256];
            let mut count = 0.0;
            for y in ty * th..((ty + 1) * th).min(h) {
                for x in tx * tw..((tx + 1) * tw).min(w) {
                    hist[img.pixels[y * w + x] as usize] += 1.0;
                    count += 1.0;
                }
            }
            if count == 0.0 {
                continue;
            }
            let limit = (clip * count / 256.0).max(1.0);
            let mut excess = 0.0;
            for v in &mut hist {
                if *v > limit {
                    excess += *v - limit;
                    *v = limit;
                }
            }
            let mut cdf = 0.0;
            for (i, v) in hist.iter().enumerate() {
                cdf += v + excess / 256.0;
                luts[ty * tiles + tx][i] = (cdf / count * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let fy = ((y as f64 + 0.5) / th as f64 - 0.5).clamp(0.0, (tiles - 1) as f64);
        let (y0, wy) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(tiles - 1);
        for x in 0..w {
            let fx = ((x as f64 + 0.5) / tw as f64 - 0.5).clamp(0.0, (tiles - 1) as f64);
            let (x0, wx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(tiles - 1);
            let v = img.pixels[y * w + x] as usize;
            let l = |ty: usize, tx: usize| f64::from(luts[ty * tiles + tx][v]);
            let top = l(y0, x0) * (1.0 - wx) + l(y0, x1) * wx;
            let bottom = l(y1, x0) * (1.0 - wx) + l(y1, x1) * wx;
            out[y * w + x] = (top * (1.0 - wy) + bottom * wy).round() as u8;
        }
    }
    Image {
        width: img.width,
        height: img.height,
        pixels: out,
    }
}

/// 5×5 binomial (Gaussian) blur.
pub fn blur5(img: &Image) -> Image {
    let (w, h) = (img.width as usize, img.height as usize);
    let k = [1.0, 4.0, 6.0, 4.0, 1.0];
    let pass = |src: &[f64], horizontal: bool| -> Vec<f64> {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    let o = i as i64 - 2;
                    let (sx, sy) = if horizontal {
                        ((x as i64 + o).clamp(0, w as i64 - 1) as usize, y)
                    } else {
                        (x, (y as i64 + o).clamp(0, h as i64 - 1) as usize)
                    };
                    acc += kv * src[sy * w + sx];
                }
                out[y * w + x] = acc / 16.0;
            }
        }
        out
    };
    let src: Vec<f64> = img.pixels.iter().map(|p| f64::from(*p)).collect();
    let out = pass(&pass(&src, true), false);
    Image {
        width: img.width,
        height: img.height,
        pixels: out.iter().map(|v| v.round() as u8).collect(),
    }
}

/// Sobel gradients `(gx, gy)`.
pub(crate) fn sobel(values: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let at = |x: i64, y: i64| {
        values[y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize]
    };
    let mut gx = vec![0.0; w * h];
    let mut gy = vec![0.0; w * h];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let i = y as usize * w + x as usize;
            gx[i] = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            gy[i] = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
        }
    }
    (gx, gy)
}

fn circ(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(180.0);
    d.min(180.0 - d)
}

/// The two dominant edge directions of a mask (degrees in `[0, 180)`).
fn edge_directions(mask: &Image) -> Option<(f64, f64)> {
    let (w, h) = (mask.width as usize, mask.height as usize);
    let v: Vec<f64> = mask
        .pixels
        .iter()
        .map(|p| f64::from(u8::from(*p > 0)))
        .collect();
    let (gx, gy) = sobel(&v, w, h);
    let mut hist = [0.0f64; 180];
    for i in 0..w * h {
        let m = gx[i].hypot(gy[i]);
        if m > 0.0 {
            let edge = (gy[i].atan2(gx[i]).to_degrees() + 90.0).rem_euclid(180.0);
            hist[(edge.floor() as usize) % 180] += m;
        }
    }
    let smooth: Vec<f64> = (0..180)
        .map(|i| {
            (-3i64..=3)
                .map(|d| hist[(i as i64 + d).rem_euclid(180) as usize])
                .sum()
        })
        .collect();
    let refine = |peak: usize| {
        let (mut s, mut c) = (0.0, 0.0);
        for d in -4i64..=4 {
            let wgt = hist[(peak as i64 + d).rem_euclid(180) as usize];
            s += wgt * (peak as f64 + d as f64 + 0.5);
            c += wgt;
        }
        (if c > 0.0 { s / c } else { peak as f64 }).rem_euclid(180.0)
    };
    let p1 = (0..180).max_by(|a, b| smooth[*a].total_cmp(&smooth[*b]))?;
    let p2 = (0..180)
        .filter(|i| circ(*i as f64, p1 as f64) >= 45.0)
        .max_by(|a, b| smooth[*a].total_cmp(&smooth[*b]))?;
    (smooth[p2] > 0.0).then(|| (refine(p1), refine(p2)))
}

/// Centre line `r = a + b s` across direction `d` (degrees), the arm's extent `(s_min, s_max)`,
/// and the widths of the narrow slices.
struct Axis {
    d: Pt,
    n: Pt,
    a: f64,
    b: f64,
    s_min: f64,
    s_max: f64,
    widths: Vec<f64>,
}

fn axis(mask: &Image, deg: f64) -> Option<Axis> {
    let w = mask.width as usize;
    let (d, n) = {
        let t = deg.to_radians();
        ([t.cos(), t.sin()], [-t.sin(), t.cos()])
    };
    let mut slices: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for (i, p) in mask.pixels.iter().enumerate() {
        if *p == 0 {
            continue;
        }
        let (x, y) = ((i % w) as f64 + 0.5, (i / w) as f64 + 0.5);
        let s = (x * d[0] + y * d[1]).round() as i64;
        let r = x * n[0] + y * n[1];
        let e = slices.entry(s).or_insert((r, r));
        e.0 = e.0.min(r);
        e.1 = e.1.max(r);
    }
    let all: Vec<f64> = slices.values().map(|(lo, hi)| hi - lo + 1.0).collect();
    let typical = super::features::median(&all)?;
    let narrow: Vec<(f64, f64, f64)> = slices
        .iter()
        .filter(|(_, (lo, hi))| hi - lo + 1.0 <= 1.5 * typical)
        .map(|(s, (lo, hi))| (*s as f64, (lo + hi) / 2.0, hi - lo + 1.0))
        .collect();
    if narrow.len() < 4 {
        return None;
    }
    let m = narrow.len() as f64;
    let (ms, mr) = (
        narrow.iter().map(|p| p.0).sum::<f64>() / m,
        narrow.iter().map(|p| p.1).sum::<f64>() / m,
    );
    let sxx: f64 = narrow.iter().map(|p| (p.0 - ms).powi(2)).sum();
    let sxy: f64 = narrow.iter().map(|p| (p.0 - ms) * (p.1 - mr)).sum();
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    Some(Axis {
        d,
        n,
        a: mr - b * ms,
        b,
        s_min: narrow.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
        s_max: narrow.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max),
        widths: narrow.iter().map(|p| p.2).collect(),
    })
}

impl Axis {
    /// The image point at position `s` along the axis, on its centre line.
    fn point(&self, s: f64) -> Pt {
        let r = self.a + self.b * s;
        [s * self.d[0] + r * self.n[0], s * self.d[1] + r * self.n[1]]
    }

    fn s_of(&self, p: Pt) -> f64 {
        p[0] * self.d[0] + p[1] * self.d[1]
    }

    /// Unit direction of the centre line.
    fn dir(&self) -> Pt {
        let v = [
            self.d[0] + self.b * self.n[0],
            self.d[1] + self.b * self.n[1],
        ];
        let l = v[0].hypot(v[1]);
        [v[0] / l, v[1] / l]
    }
}

fn intersect(p: Pt, u: Pt, q: Pt, v: Pt) -> Option<Pt> {
    let det = u[0] * (-v[1]) - u[1] * (-v[0]);
    if det.abs() < 1e-9 {
        return None;
    }
    let (rx, ry) = (q[0] - p[0], q[1] - p[1]);
    let t = (rx * (-v[1]) - ry * (-v[0])) / det;
    Some([p[0] + t * u[0], p[1] + t * u[1]])
}

fn rect(c: Pt, along: Pt, from: f64, to: f64, half: f64) -> Polygon {
    let across = [-along[1], along[0]];
    let at = |s: f64, r: f64| {
        [
            c[0] + s * along[0] + r * across[0],
            c[1] + s * along[1] + r * across[1],
        ]
    };
    Polygon(vec![
        at(from, -half),
        at(to, -half),
        at(to, half),
        at(from, half),
    ])
}

/// Wall score of an arm: oriented edge strength in two bands along its long edges, plus how much
/// those bands differ in brightness from the arm's floor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wall_score(
    img: &Image,
    gx: &[f64],
    gy: &[f64],
    c: Pt,
    along: Pt,
    from: f64,
    to: f64,
    w: f64,
) -> f64 {
    let width = img.width as usize;
    let across = [-along[1], along[0]];
    let band = 0.15 * w;
    let (lo, hi) = (from.min(to), from.max(to));
    let (mut edge, mut nband) = (0.0, 0.0);
    let (mut band_sum, mut floor): (f64, Vec<f64>) = (0.0, Vec::new());
    let reach = hi.abs().max(lo.abs()) + w;
    let (x0, x1) = (
        (c[0] - reach).floor().max(0.0) as usize,
        ((c[0] + reach).ceil() as usize).min(width),
    );
    let height = img.height as usize;
    let (y0, y1) = (
        (c[1] - reach).floor().max(0.0) as usize,
        ((c[1] + reach).ceil() as usize).min(height),
    );
    for y in y0..y1 {
        for x in x0..x1 {
            let (dx, dy) = (x as f64 + 0.5 - c[0], y as f64 + 0.5 - c[1]);
            let s = dx * along[0] + dy * along[1];
            let r = dx * across[0] + dy * across[1];
            if s < lo || s > hi {
                continue;
            }
            let i = y * width + x;
            let grey = f64::from(img.pixels[i]);
            if (r.abs() - w / 2.0).abs() <= band / 2.0 {
                nband += 1.0;
                band_sum += grey;
                let m = gx[i].hypot(gy[i]);
                // An edge running along the arm has its gradient across it.
                let g_across = (gx[i] * across[0] + gy[i] * across[1]).abs();
                if m > 0.0 && (g_across / m) >= 15f64.to_radians().cos() {
                    edge += m;
                }
            } else if r.abs() < w / 2.0 - band / 2.0 {
                floor.push(grey);
            }
        }
    }
    if nband == 0.0 || floor.len() < 2 {
        return 0.0;
    }
    let mean = floor.iter().sum::<f64>() / floor.len() as f64;
    let sd = (floor.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / floor.len() as f64)
        .sqrt()
        .max(1.0);
    edge / nband + (band_sum / nband - mean).abs() / sd
}

/// The plus maze in the background image; `needs_review` when the axes or the walls are unclear.
/// `track`: the animal's top-view positions; the maze is the smallest region holding them
/// ([`super::arena::arena_around`]), or without a track the largest region at the centre.
pub fn detect(bg: &Image, cfg: &Maze, rig: &str, track: &[Pt]) -> Layout {
    let enhanced = blur5(&clahe(bg, cfg.clahe_clip, cfg.clahe_tiles));
    let found = super::arena::find_arena(&enhanced, track, super::arena::Shape::Any);
    let finder = found.as_ref().map_or("largest", |f| f.method);
    let mask = found.map_or_else(|| arena_mask(&enhanced), |f| f.mask);
    let mut layout = Layout {
        rig: rig.to_owned(),
        source: "auto".into(),
        method: super::arena::METHOD,
        status: "needs_review".into(),
        finder: finder.into(),
        image_size: [bg.width, bg.height],
        ..Layout::default()
    };
    let Some((d1, d2)) = edge_directions(&mask) else {
        layout.reason = "no maze found in the background".into();
        return layout;
    };
    if (circ(d1, d2) - 90.0).abs() > cfg.axis_tol_deg {
        layout.reason = format!("the arms are {:.0}° apart, not 90°", circ(d1, d2));
        return layout;
    }
    let (Some(a1), Some(a2)) = (axis(&mask, d1), axis(&mask, d2)) else {
        layout.reason = "the arms could not be measured".into();
        return layout;
    };
    let mut widths = a1.widths.clone();
    widths.extend(&a2.widths);
    let w = super::features::median(&widths).unwrap_or(0.0);
    let Some(c) = intersect(a1.point(0.0), a1.dir(), a2.point(0.0), a2.dir()) else {
        layout.reason = "the arm centre lines do not cross".into();
        return layout;
    };
    let (e1, e2) = (a1.dir(), a2.dir());
    let mut zones = vec![Zone {
        name: "center".into(),
        kind: "center".into(),
        polygon: Polygon(vec![
            [
                c[0] - e1[0] * w / 2.0 - e2[0] * w / 2.0,
                c[1] - e1[1] * w / 2.0 - e2[1] * w / 2.0,
            ],
            [
                c[0] + e1[0] * w / 2.0 - e2[0] * w / 2.0,
                c[1] + e1[1] * w / 2.0 - e2[1] * w / 2.0,
            ],
            [
                c[0] + e1[0] * w / 2.0 + e2[0] * w / 2.0,
                c[1] + e1[1] * w / 2.0 + e2[1] * w / 2.0,
            ],
            [
                c[0] - e1[0] * w / 2.0 + e2[0] * w / 2.0,
                c[1] - e1[1] * w / 2.0 + e2[1] * w / 2.0,
            ],
        ]),
    }];
    let enhanced_f: Vec<f64> = enhanced.pixels.iter().map(|p| f64::from(*p)).collect();
    let (gx, gy) = sobel(&enhanced_f, bg.width as usize, bg.height as usize);
    let mut scores = Vec::new();
    for (k, a) in [&a1, &a2].iter().enumerate() {
        let sc = a.s_of(c);
        let dir = a.dir();
        let scale = dir[0] * a.d[0] + dir[1] * a.d[1];
        let plus = (a.s_max - sc) / scale;
        let minus = (a.s_min - sc) / scale;
        let mut axis_scores = Vec::new();
        for (j, (from, to)) in [(w / 2.0, plus), (-w / 2.0, minus)].into_iter().enumerate() {
            let name = format!("arm{}{}", k + 1, ["a", "b"][j]);
            let score = wall_score(&enhanced, &gx, &gy, c, dir, from, to, w);
            layout.wall_scores.insert(name.clone(), score);
            axis_scores.push(score);
            zones.push(Zone {
                name,
                kind: String::new(),
                polygon: rect(c, dir, from, to, w / 2.0),
            });
        }
        scores.push(axis_scores.iter().sum::<f64>() / 2.0);
    }
    let closed = usize::from(scores[1] > scores[0]);
    let (hi, lo) = (scores[closed], scores[1 - closed]);
    let decided = hi >= cfg.wall_ratio * lo.max(1e-9);
    for z in zones.iter_mut().skip(1) {
        let axis_index = usize::from(z.name.starts_with("arm2"));
        z.kind = if axis_index == closed {
            "closed_arm"
        } else {
            "open_arm"
        }
        .into();
    }
    layout.zones = zones;
    layout.floor = Vec::new();
    if decided {
        layout.status = "ok".into();
    } else {
        layout.reason = format!(
            "the walls are unclear: closed/open score ratio {:.2} is below {}",
            hi / lo.max(1e-9),
            cfg.wall_ratio
        );
    }
    layout
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plus maze on a dark floor; the vertical arms have bright wall tops along their edges.
    pub fn synthetic(walls: bool) -> Image {
        let (w, h) = (220u32, 200u32);
        let mut img = Image {
            width: w,
            height: h,
            pixels: vec![30; (w * h) as usize],
        };
        let mut paint = |x0: u32, x1: u32, y0: u32, y1: u32, v: u8| {
            for y in y0..y1 {
                for x in x0..x1 {
                    img.pixels[(y * w + x) as usize] = v;
                }
            }
        };
        paint(20, 200, 90, 112, 150);
        paint(99, 121, 10, 190, 150);
        if walls {
            for (y0, y1) in [(10, 90), (112, 190)] {
                paint(99, 103, y0, y1, 240);
                paint(117, 121, y0, y1, 240);
            }
        }
        img
    }

    #[test]
    fn arms_centre_and_closed_arms_are_found() {
        let cfg = crate::assay::config::defaults("plus_maze").maze;
        let layout = detect(&synthetic(true), &cfg, "maze", &[]);
        assert_eq!(layout.status, "ok", "{}", layout.reason);
        let kind = |p: Pt| layout.zone(p).map(|z| z.kind.clone()).unwrap_or_default();
        assert_eq!(kind([110.0, 101.0]), "center");
        assert_eq!(kind([110.0, 40.0]), "closed_arm");
        assert_eq!(kind([110.0, 170.0]), "closed_arm");
        assert_eq!(kind([40.0, 101.0]), "open_arm");
        assert_eq!(kind([185.0, 101.0]), "open_arm");
        assert_eq!(kind([40.0, 40.0]), "");
        // Without walls nothing tells the arms apart: the code asks for a review.
        assert_eq!(
            detect(&synthetic(false), &cfg, "maze", &[]).status,
            "needs_review"
        );
    }
}
