//! A small RGB raster for the figures `animalpose behavior plot` writes: filled rectangles, thick
//! lines, filled circles and a 5x7 bitmap font, saved as PNG through the `image` crate.
//!
//! `// differs from Python:` the Python figures come from matplotlib (`ethogram.png`) and
//! `cv2.line` / `cv2.circle` / `cv2.putText` with anti-aliasing (the clip); this canvas draws the
//! same elements without anti-aliasing, and its font has upper-case glyphs only (lower-case text
//! is drawn in capitals).

use std::path::Path;

use anyhow::{Context, Result};

/// An RGB colour.
pub type Rgb = [u8; 3];

/// `#rrggbb` to RGB (black for a malformed string).
pub fn hex(s: &str) -> Rgb {
    let s = s.trim_start_matches('#');
    let c = |i: usize| u8::from_str_radix(s.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0);
    [c(0), c(2), c(4)]
}

/// An RGB image.
#[derive(Clone, Debug)]
pub struct Canvas {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGB triplets.
    pub pixels: Vec<u8>,
}

impl Canvas {
    /// A canvas filled with `color`.
    pub fn new(width: u32, height: u32, color: Rgb) -> Self {
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..width * height {
            pixels.extend_from_slice(&color);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// The colour at `(x, y)`.
    pub fn get(&self, x: u32, y: u32) -> Rgb {
        let i = ((y * self.width + x) * 3) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }

    /// Sets `(x, y)` when it lies inside the canvas.
    pub fn put(&mut self, x: i64, y: i64, color: Rgb) {
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return;
        }
        let i = ((y as u32 * self.width + x as u32) * 3) as usize;
        self.pixels[i..i + 3].copy_from_slice(&color);
    }

    /// Fills `[x0, x1) x [y0, y1)`, clipped to the canvas.
    pub fn fill_rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, color: Rgb) {
        for y in y0.max(0)..y1.min(i64::from(self.height)) {
            for x in x0.max(0)..x1.min(i64::from(self.width)) {
                self.put(x, y, color);
            }
        }
    }

    /// A one-pixel rectangle outline around `[x0, x1) x [y0, y1)`.
    pub fn outline(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, color: Rgb) {
        self.fill_rect(x0, y0, x1, y0 + 1, color);
        self.fill_rect(x0, y1 - 1, x1, y1, color);
        self.fill_rect(x0, y0, x0 + 1, y1, color);
        self.fill_rect(x1 - 1, y0, x1, y1, color);
    }

    /// A filled disc of radius `r`.
    pub fn circle(&mut self, cx: f64, cy: f64, r: f64, color: Rgb) {
        let (x0, x1) = ((cx - r).floor() as i64, (cx + r).ceil() as i64);
        let (y0, y1) = ((cy - r).floor() as i64, (cy + r).ceil() as i64);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2) <= r * r {
                    self.put(x, y, color);
                }
            }
        }
    }

    /// A line `thickness` pixels wide (round caps).
    pub fn line(&mut self, a: [f64; 2], b: [f64; 2], thickness: f64, color: Rgb) {
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let steps = (len.ceil() as usize).max(1) * 2;
        let r = (thickness / 2.0).max(0.5);
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            self.circle(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), r, color);
        }
    }

    /// A vertical dashed line at `x` from `y0` to `y1`.
    pub fn dashed_vline(&mut self, x: i64, y0: i64, y1: i64, width: i64, color: Rgb) {
        let mut y = y0;
        while y < y1 {
            self.fill_rect(
                x - width / 2,
                y,
                x - width / 2 + width,
                (y + 8).min(y1),
                color,
            );
            y += 13;
        }
    }

    /// `text` with its lower-left corner at `(x, y)` (as `cv2.putText` places it), each font pixel
    /// drawn as a `scale x scale` square. Returns the width drawn.
    pub fn text(&mut self, x: i64, y: i64, text: &str, scale: u32, color: Rgb) -> i64 {
        let s = i64::from(scale.max(1));
        let top = y - 7 * s;
        let mut at = x;
        for ch in text.chars() {
            let rows = glyph(ch);
            for (r, bits) in rows.iter().enumerate() {
                for c in 0..5 {
                    if bits & (0b10000 >> c) != 0 {
                        let px = at + c * s;
                        let py = top + r as i64 * s;
                        self.fill_rect(px, py, px + s, py + s, color);
                    }
                }
            }
            at += 6 * s;
        }
        at - x
    }

    /// The width `text` takes at `scale`.
    pub fn text_width(text: &str, scale: u32) -> i64 {
        text.chars().count() as i64 * 6 * i64::from(scale.max(1))
    }

    /// Writes the canvas as an RGB PNG.
    pub fn save_png(&self, path: &Path) -> Result<()> {
        let buf = image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .context("canvas buffer size")?;
        buf.save(path)
            .with_context(|| format!("cannot write {}", path.display()))
    }
}

/// The 5x7 glyph of `ch`, one row per entry, bit 4 the leftmost column.
fn glyph(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
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
        '.' => [0, 0, 0, 0, 0, 0x0C, 0x0C],
        ',' => [0, 0, 0, 0, 0x0C, 0x04, 0x08],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        '+' => [0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0],
        '_' => [0, 0, 0, 0, 0, 0, 0x1F],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '=' => [0, 0, 0x1F, 0, 0x1F, 0, 0],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '/' => [0, 0x01, 0x02, 0x04, 0x08, 0x10, 0],
        ' ' => [0; 7],
        _ => [0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_land_where_asked() {
        let mut c = Canvas::new(40, 20, [255, 255, 255]);
        c.fill_rect(2, 3, 6, 5, hex("#4e79a7"));
        assert_eq!(c.get(2, 3), [0x4e, 0x79, 0xa7]);
        assert_eq!(c.get(6, 3), [255, 255, 255]);
        c.circle(20.0, 10.0, 3.0, [0, 0, 0]);
        assert_eq!((c.get(20, 10), c.get(20, 14)), ([0, 0, 0], [255, 255, 255]));
        let w = c.text(0, 19, "Ab1", 1, [9, 9, 9]);
        assert_eq!(w, Canvas::text_width("Ab1", 1));
        assert_eq!(c.get(0, 13), [9, 9, 9]); // the left stroke of A
        c.line([30.0, 2.0], [38.0, 2.0], 2.0, [1, 2, 3]);
        assert_eq!(c.get(34, 2), [1, 2, 3]);
    }
}
