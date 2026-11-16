//! Image containers and the little image processing the blob and background
//! code needs (greyscale conversion, linear resize, letterboxing).
//!
//! // differs from Python: OpenCV works on `uint8` arrays of shape `(H, W)` or
//! // `(H, W, 3)`; here the same data lives in the flat [`Image`]/[`Bgr`]
//! // buffers below, and every routine that OpenCV implemented in C is
//! // re-implemented in this crate.

use anyhow::{bail, ensure, Result};

/// 8-bit single-channel image: the `(H, W)` `uint8` arrays of the Python code.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Image {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major pixels, length `width * height`.
    pub pixels: Vec<u8>,
}

/// 8-bit BGR image: the `(H, W, 3)` `uint8` arrays of the Python code.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Bgr {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major `B, G, R` triplets, length `3 * width * height`.
    pub pixels: Vec<u8>,
}

/// Grey levels as floating point: the output of `animalpose.pipeline._gray`
/// and of `animalpose.pipeline.median_background`.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct GrayF {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major grey levels, length `width * height`.
    pub pixels: Vec<f32>,
}

/// Anything that can be turned into floating point grey levels.
pub trait ToGray {
    /// Port of `animalpose.pipeline._gray`.
    fn to_gray(&self) -> GrayF;
}

impl Image {
    /// Builds an image of `width * height` pixels filled with `value`.
    pub fn filled(width: u32, height: u32, value: u8) -> Self {
        Self {
            width,
            height,
            pixels: vec![value; (width * height) as usize],
        }
    }

    /// Wraps a row-major buffer, checking its length.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        ensure!(
            pixels.len() == (width * height) as usize,
            "image {}x{} needs {} pixels, got {}",
            width,
            height,
            width * height,
            pixels.len()
        );
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Pixel at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> u8 {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Overwrites the pixel at `(x, y)`.
    pub fn set_pixel(&mut self, x: u32, y: u32, value: u8) {
        self.pixels[(y * self.width + x) as usize] = value;
    }

    /// Paints `value` into the half-open rectangle `x0..x1, y0..y1`, clipped to
    /// the image. Mirrors the `img[y0:y1, x0:x1] = value` slices of the tests.
    pub fn fill_rect(&mut self, x0: u32, y0: u32, x1: u32, y1: u32, value: u8) {
        for y in y0..self.height.min(y1) {
            for x in x0..self.width.min(x1) {
                self.pixels[(y * self.width + x) as usize] = value;
            }
        }
    }

    /// `true` when both images have the same size.
    pub fn same_size(&self, other: &Image) -> bool {
        self.width == other.width && self.height == other.height
    }
}

impl Bgr {
    /// Builds a BGR image of `width * height` pixels filled with `value`.
    pub fn filled(width: u32, height: u32, value: [u8; 3]) -> Self {
        let mut pixels = Vec::with_capacity((3 * width * height) as usize);
        for _ in 0..width * height {
            pixels.extend_from_slice(&value);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Wraps a row-major `B, G, R` buffer, checking its length.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        ensure!(
            pixels.len() == (3 * width * height) as usize,
            "bgr image {}x{} needs {} bytes, got {}",
            width,
            height,
            3 * width * height,
            pixels.len()
        );
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Pixel triplet `(B, G, R)` at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = 3 * (y * self.width + x) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }

    /// Overwrites the pixel triplet at `(x, y)`.
    pub fn set_pixel(&mut self, x: u32, y: u32, value: [u8; 3]) {
        let i = 3 * (y * self.width + x) as usize;
        self.pixels[i..i + 3].copy_from_slice(&value);
    }

    /// Paints `value` into the half-open rectangle `x0..x1, y0..y1`, clipped to
    /// the image.
    pub fn fill_rect(&mut self, x0: u32, y0: u32, x1: u32, y1: u32, value: [u8; 3]) {
        for y in y0..self.height.min(y1) {
            for x in x0..self.width.min(x1) {
                self.set_pixel(x, y, value);
            }
        }
    }
}

impl ToGray for Image {
    fn to_gray(&self) -> GrayF {
        GrayF {
            width: self.width,
            height: self.height,
            pixels: self.pixels.iter().map(|v| *v as f32).collect(),
        }
    }
}

impl ToGray for Bgr {
    /// Port of `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` followed by
    /// `.astype(np.float32)`, with OpenCV's fixed-point weights (1868/9617/4899 over 2^14,
    /// rounded), so a grey pixel replicated to three channels comes back unchanged.
    fn to_gray(&self) -> GrayF {
        let mut pixels = Vec::with_capacity((self.width * self.height) as usize);
        for px in self.pixels.chunks_exact(3) {
            let (b, g, r) = (u32::from(px[0]), u32::from(px[1]), u32::from(px[2]));
            let y = (b * 1868 + g * 9617 + r * 4899 + (1 << 13)) >> 14;
            pixels.push(y as f32);
        }
        GrayF {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

impl GrayF {
    /// Wraps a row-major buffer of grey levels, checking its length.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<f32>) -> Result<Self> {
        ensure!(
            pixels.len() == (width * height) as usize,
            "grey image {}x{} needs {} samples, got {}",
            width,
            height,
            width * height,
            pixels.len()
        );
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Grey level at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> f32 {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Casts to `uint8` the way `bg.astype(np.uint8)` does, i.e. truncating
    /// towards zero and saturating at 255.
    pub fn as_u8(&self) -> Image {
        let pixels = self
            .pixels
            .iter()
            .map(|v| v.clamp(0.0, 255.0) as u8)
            .collect();
        Image {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    /// Port of `cv2.absdiff(a, b)` on grey levels.
    pub fn abs_diff(&self, other: &GrayF) -> GrayF {
        let pixels = self
            .pixels
            .iter()
            .zip(other.pixels.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        GrayF {
            width: self.width,
            height: self.height,
            pixels,
        }
    }

    /// Port of `(d > threshold)` producing a `0`/`1` `uint8` mask.
    pub fn threshold_gt(&self, threshold: f32) -> Image {
        let pixels = self
            .pixels
            .iter()
            .map(|v| u8::from(*v > threshold))
            .collect();
        Image {
            width: self.width,
            height: self.height,
            pixels,
        }
    }
}

/// Bilinear resample, the `cv2.INTER_LINEAR` mapping
/// (`src = (dst + 0.5) * scale - 0.5`, clamped to the source).
///
/// // differs from Python: OpenCV evaluates the same mapping with 5-bit
/// // fixed-point weights; the results agree to well under one grey level.
pub fn resize_linear(src: &[f32], width: u32, height: u32, dst_w: u32, dst_h: u32) -> Vec<f32> {
    if dst_w == 0 || dst_h == 0 || width == 0 || height == 0 {
        return vec![0.0; (dst_w * dst_h) as usize];
    }
    let sx = f64::from(width) / f64::from(dst_w);
    let sy = f64::from(height) / f64::from(dst_h);
    let mut out = vec![0.0f32; (dst_w * dst_h) as usize];
    for dy in 0..dst_h {
        let fy = ((f64::from(dy) + 0.5) * sy - 0.5).clamp(0.0, f64::from(height) - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(height as usize - 1);
        let wy = (fy - y0 as f64) as f32;
        for dx in 0..dst_w {
            let fx = ((f64::from(dx) + 0.5) * sx - 0.5).clamp(0.0, f64::from(width) - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(width as usize - 1);
            let wx = (fx - x0 as f64) as f32;
            let a = src[(y0 * width as usize) + x0];
            let b = src[(y0 * width as usize) + x1];
            let c = src[(y1 * width as usize) + x0];
            let d = src[(y1 * width as usize) + x1];
            let top = a + (b - a) * wx;
            let bottom = c + (d - c) * wx;
            out[(dy * dst_w + dx) as usize] = top + (bottom - top) * wy;
        }
    }
    out
}

/// Resize keeping the aspect ratio and pad to `size = (H, W)`.
///
/// Port of `animalpose.data.datasets.letterbox`; returns the padded image, the
/// scale and the `(pad_x, pad_y)` offset of the pasted content.
pub fn letterbox(img: &Bgr, size: (u32, u32)) -> Result<(Bgr, f64, [f64; 2])> {
    if img.width == 0 || img.height == 0 {
        bail!("letterbox: empty image");
    }
    let (h_out, w_out) = (size.0, size.1);
    let scale =
        (f64::from(h_out) / f64::from(img.height)).min(f64::from(w_out) / f64::from(img.width));
    let nh = (f64::from(img.height) * scale).round().max(0.0) as u32;
    let nw = (f64::from(img.width) * scale).round().max(0.0) as u32;
    let py = (h_out - nh.min(h_out)) / 2;
    let px = (w_out - nw.min(w_out)) / 2;
    let mut out = Bgr::filled(w_out, h_out, [0, 0, 0]);
    if nw > 0 && nh > 0 {
        for c in 0..3 {
            let plane: Vec<f32> = img
                .pixels
                .chunks_exact(3)
                .map(|px| f32::from(px[c]))
                .collect();
            let resized = resize_linear(&plane, img.width, img.height, nw, nh);
            for y in 0..nh {
                for x in 0..nw {
                    let i = 3 * ((y + py) * w_out + (x + px)) as usize;
                    let v = resized[(y * nw + x) as usize];
                    out.pixels[i + c] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok((out, scale, [f64::from(px), f64::from(py)]))
}

/// Letterbox for a single-channel frame (the Python helper works on any dtype).
pub fn letterbox_gray(img: &Image, size: (u32, u32)) -> Result<(Image, f64, [f64; 2])> {
    if img.width == 0 || img.height == 0 {
        bail!("letterbox: empty image");
    }
    let (h_out, w_out) = (size.0, size.1);
    let scale =
        (f64::from(h_out) / f64::from(img.height)).min(f64::from(w_out) / f64::from(img.width));
    let nh = (f64::from(img.height) * scale).round().max(0.0) as u32;
    let nw = (f64::from(img.width) * scale).round().max(0.0) as u32;
    let py = (h_out - nh.min(h_out)) / 2;
    let px = (w_out - nw.min(w_out)) / 2;
    let mut out = Image::filled(w_out, h_out, 0);
    if nw > 0 && nh > 0 {
        let gray: Vec<f32> = img.pixels.iter().map(|v| f32::from(*v)).collect();
        let resized = resize_linear(&gray, img.width, img.height, nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let v = resized[(y * nw + x) as usize];
                out.pixels[((y + py) * w_out + (x + px)) as usize] =
                    v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok((out, scale, [f64::from(px), f64::from(py)]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgr_grey_matches_the_opaque_greys_of_the_python_tests() {
        let img = Bgr::filled(4, 3, [200, 200, 200]);
        let gray = img.to_gray();
        assert_eq!(gray.width, 4);
        assert!(gray.pixels.iter().all(|v| *v == 200.0));
        let dark = Bgr::filled(1, 1, [20, 20, 20]);
        assert_eq!(dark.to_gray().pixels[0], 20.0);
    }

    #[test]
    fn float_grey_casts_to_uint8_by_truncation() {
        let g = GrayF::from_pixels(3, 1, vec![199.7, 20.0, 300.0]).unwrap();
        assert_eq!(g.as_u8().pixels, vec![199, 20, 255]);
    }

    #[test]
    fn absdiff_and_threshold_mirror_the_numpy_calls() {
        let a = GrayF::from_pixels(4, 1, vec![100.0, 120.0, 140.0, 90.0]).unwrap();
        let b = GrayF::from_pixels(4, 1, vec![100.0, 100.0, 100.0, 100.0]).unwrap();
        let d = a.abs_diff(&b);
        assert_eq!(d.pixels, vec![0.0, 20.0, 40.0, 10.0]);
        assert_eq!(d.threshold_gt(15.0).pixels, vec![0, 1, 1, 0]);
    }

    #[test]
    fn letterbox_pads_and_reports_scale_and_offset() {
        let img = Bgr::filled(40, 20, [10, 10, 10]);
        let (out, scale, pad) = letterbox(&img, (64, 64)).unwrap();
        assert_eq!(out.width, 64);
        assert_eq!(out.height, 64);
        assert!((scale - 1.6).abs() < 1e-12);
        // Python: s = min(64/20, 64/40) = 1.6, nh = 32, nw = 64, py = 16, px = 0
        assert_eq!(pad, [0.0, 16.0]);
        assert_eq!(out.pixel(0, 0), [0, 0, 0]);
        assert_eq!(out.pixel(32, 16 + 15), [10, 10, 10]);
        assert_eq!(out.pixel(32, 16 + 31), [10, 10, 10]);
    }

    #[test]
    fn letterbox_rejects_empty_images() {
        let img = Bgr::filled(0, 4, [0, 0, 0]);
        assert!(letterbox(&img, (8, 8)).is_err());
    }

    #[test]
    fn linear_resize_preserves_constants_and_blurs_edges() {
        let src = vec![100.0f32; 16];
        let out = resize_linear(&src, 4, 4, 8, 8);
        assert!(out.iter().all(|v| (*v - 100.0).abs() < 1e-5));
        // A hard left/right split stays a hard split, with interpolation on the seam.
        let half: Vec<f32> = (0..4).flat_map(|_| [0.0, 0.0, 255.0, 255.0]).collect();
        let out = resize_linear(&half, 4, 4, 4, 4);
        assert!(out[0] < 1.0 && out[3] > 254.0);
    }
}
