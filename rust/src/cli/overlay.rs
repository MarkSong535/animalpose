//! Overlay drawing for inspection videos: port of `src/animalpose/viz.py` (`draw_pose`,
//! `draw_frame`) and `api.py:write_video`'s frame loop.
//!
//! `// differs from Python:` OpenCV's `cv2.line(..., LINE_AA)` / `cv2.circle` rasterisers are
//! replaced by a Wu line and a coverage-weighted disc, and `cv2.putText`'s Hershey font by a 5x7
//! bitmap font; colours, positions, radii and which points are drawn hollow are the same.

use crate::algos::image::Bgr;
use crate::cli::predict::VideoOut;
use crate::data::synthetic::{index, NUM_KEYPOINTS};

/// `viz.py:_PALETTE`, used as BGR exactly as OpenCV receives it.
const PALETTE: [[u8; 3]; 6] = [
    [66, 135, 245],
    [245, 66, 102],
    [66, 245, 135],
    [245, 200, 66],
    [190, 66, 245],
    [66, 230, 245],
];

/// `skeleton/vocabulary.py:_EDGES`.
const EDGES: [(&str, &str); 25] = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_ear", "head_center"),
    ("right_ear", "head_center"),
    ("head_center", "neck"),
    ("neck", "spine_upper"),
    ("spine_upper", "spine_mid"),
    ("spine_mid", "spine_lower"),
    ("spine_lower", "tail_base"),
    ("tail_base", "tail_mid"),
    ("tail_mid", "tail_end"),
    ("spine_upper", "left_shoulder"),
    ("spine_upper", "right_shoulder"),
    ("left_shoulder", "left_front_knee"),
    ("left_front_knee", "left_front_paw"),
    ("right_shoulder", "right_front_knee"),
    ("right_front_knee", "right_front_paw"),
    ("spine_lower", "left_hip"),
    ("spine_lower", "right_hip"),
    ("left_hip", "left_hind_knee"),
    ("left_hind_knee", "left_hind_paw"),
    ("right_hip", "right_hind_knee"),
    ("right_hind_knee", "right_hind_paw"),
];

/// `viz.py:color`.
pub fn color(track_id: i64) -> [u8; 3] {
    PALETTE[track_id.rem_euclid(PALETTE.len() as i64) as usize]
}

/// Blends `col` into one pixel with weight `alpha`.
fn blend(img: &mut Bgr, x: i64, y: i64, col: [u8; 3], alpha: f64) {
    if alpha <= 0.0 || x < 0 || y < 0 || x >= i64::from(img.width) || y >= i64::from(img.height) {
        return;
    }
    let a = alpha.min(1.0);
    let at = (y as usize * img.width as usize + x as usize) * 3;
    for (px, c) in img.pixels[at..at + 3].iter_mut().zip(col) {
        let old = f64::from(*px);
        *px = (old + (f64::from(c) - old) * a).round().clamp(0.0, 255.0) as u8;
    }
}

/// An anti-aliased one-pixel line (Xiaolin Wu).
pub fn line(img: &mut Bgr, a: [f64; 2], b: [f64; 2], col: [u8; 3]) {
    let (mut x0, mut y0, mut x1, mut y1) = (a[0], a[1], b[0], b[1]);
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    if steep {
        std::mem::swap(&mut x0, &mut y0);
        std::mem::swap(&mut x1, &mut y1);
    }
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
        std::mem::swap(&mut y0, &mut y1);
    }
    let dx = x1 - x0;
    let gradient = if dx.abs() < 1e-12 {
        1.0
    } else {
        (y1 - y0) / dx
    };
    let mut plot = |x: i64, y: i64, c: f64| {
        if steep {
            blend(img, y, x, col, c);
        } else {
            blend(img, x, y, col, c);
        }
    };
    let (xs, xe) = (x0.round() as i64, x1.round() as i64);
    for x in xs..=xe {
        let y = y0 + gradient * (x as f64 - x0);
        let yf = y.floor();
        let frac = y - yf;
        plot(x, yf as i64, 1.0 - frac);
        plot(x, yf as i64 + 1, frac);
    }
}

/// A disc of radius `r` (`filled`) or a one-pixel ring, anti-aliased by pixel coverage.
pub fn circle(img: &mut Bgr, c: [f64; 2], r: f64, col: [u8; 3], filled: bool) {
    let reach = (r + 1.5).ceil() as i64;
    let (cx, cy) = (c[0].round() as i64, c[1].round() as i64);
    for y in cy - reach..=cy + reach {
        for x in cx - reach..=cx + reach {
            let d = ((x as f64 - c[0]).powi(2) + (y as f64 - c[1]).powi(2)).sqrt();
            let alpha = if filled {
                r + 0.5 - d
            } else {
                1.0 - (d - r).abs()
            };
            blend(img, x, y, col, alpha.clamp(0.0, 1.0));
        }
    }
}

/// 5x7 glyphs, one byte per row, bit 4 the leftmost column.
fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_lowercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        'b' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E],
        'd' => [0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        'f' => [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],
        'g' => [0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'h' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
        'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        'j' => [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
        'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        'n' => [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        'q' => [0x00, 0x00, 0x0D, 0x13, 0x0F, 0x01, 0x01],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        's' => [0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E],
        't' => [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
        'v' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'w' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        _ => [0; 7],
    }
}

/// Text whose bottom-left corner sits at `(x, baseline)`, like `cv2.putText`'s origin.
pub fn text(img: &mut Bgr, x: i64, baseline: i64, s: &str, col: [u8; 3]) {
    for (i, ch) in s.chars().enumerate() {
        let rows = glyph(ch);
        let left = x + i as i64 * 6;
        for (r, bits) in rows.iter().enumerate() {
            for c in 0..5 {
                if bits & (0x10 >> c) != 0 {
                    blend(img, left + c, baseline - 6 + r as i64, col, 1.0);
                }
            }
        }
    }
}

/// `viz.py:draw_pose` for one animal: skeleton, joints (hollow where imputed) and a label.
#[allow(clippy::too_many_arguments)]
pub fn draw_pose(
    img: &mut Bgr,
    kpts: &[f64],
    conf: Option<&[f64]>,
    imputed: Option<&[f64]>,
    track_id: i64,
    label: &str,
    thresh: f64,
) {
    let col = color(track_id);
    let ok: Vec<bool> = (0..NUM_KEYPOINTS)
        .map(|k| {
            let finite = kpts[2 * k].is_finite() && kpts[2 * k + 1].is_finite();
            let c = conf.map_or(f64::INFINITY, |c| if c[k].is_nan() { 0.0 } else { c[k] });
            finite && c >= thresh
        })
        .collect();
    let p = |k: usize| [kpts[2 * k].round(), kpts[2 * k + 1].round()];
    for (a, b) in EDGES {
        let (a, b) = (index(a), index(b));
        if ok[a] && ok[b] {
            line(img, p(a), p(b), col);
        }
    }
    for k in (0..NUM_KEYPOINTS).filter(|k| ok[*k]) {
        let hollow = imputed.is_some_and(|m| m[k] > 0.5);
        circle(img, p(k), 2.0, col, !hollow);
    }
    if ok.iter().any(|v| *v) {
        let (mut x, mut y) = (f64::INFINITY, f64::INFINITY);
        for k in (0..NUM_KEYPOINTS).filter(|k| ok[*k]) {
            x = x.min(p(k)[0]);
            y = y.min(p(k)[1]);
        }
        let label = format!("{track_id} {label}");
        text(img, x as i64, (y as i64 - 4).max(10), label.trim(), col);
    }
}

/// `viz.py:draw_frame`: every track of `result` at frame `t`.
pub fn draw_frame(image: &Bgr, result: &VideoOut, t: usize) -> Bgr {
    let mut out = image.clone();
    for (tid, tr) in &result.tracks {
        let Some(first) = tr.frames.first() else {
            continue;
        };
        let Some(i) = t.checked_sub(*first).filter(|i| *i < tr.frames.len()) else {
            continue;
        };
        let k = NUM_KEYPOINTS;
        let best = tr.smoothed.as_deref().unwrap_or(&tr.keypoints);
        let conf = tr.smoothed_conf.as_deref().unwrap_or(&tr.conf);
        let imputed = tr.imputed.as_deref().map(|m| &m[i * k..(i + 1) * k]);
        let label = tr
            .behavior
            .as_ref()
            .and_then(|b| usize::try_from(b[i]).ok())
            .and_then(|c| result.behaviors.get(c))
            .map_or("", String::as_str);
        draw_pose(
            &mut out,
            &best[i * k * 2..(i + 1) * k * 2],
            Some(&conf[i * k..(i + 1) * k]),
            imputed,
            *tid,
            label,
            0.3,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_draw_where_asked() {
        let mut img = Bgr::filled(32, 32, [0, 0, 0]);
        line(&mut img, [2.0, 2.0], [20.0, 2.0], [255, 0, 0]);
        assert_eq!(img.pixel(10, 2), [255, 0, 0]);
        assert_eq!(img.pixel(10, 5), [0, 0, 0]);
        circle(&mut img, [16.0, 16.0], 2.0, [0, 255, 0], true);
        assert_eq!(img.pixel(16, 16), [0, 255, 0]);
        let mut ring = Bgr::filled(32, 32, [0, 0, 0]);
        circle(&mut ring, [16.0, 16.0], 2.0, [0, 255, 0], false);
        assert_eq!(ring.pixel(16, 16), [0, 0, 0]);
        assert_eq!(ring.pixel(18, 16), [0, 255, 0]);
        text(&mut img, 2, 28, "1 walk", [0, 0, 255]);
        assert!(img.pixels.chunks(3).any(|p| p == [0, 0, 255]));
    }

    #[test]
    fn palette_cycles_like_python() {
        assert_eq!(color(0), [66, 135, 245]);
        assert_eq!(color(7), color(1));
    }
}
