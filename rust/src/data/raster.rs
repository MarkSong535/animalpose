//! The OpenCV calls the renderer, the degradations and the datasets make, re-implemented on a
//! float image: drawing (`cv2.line`, `cv2.circle`, `cv2.fillConvexPoly`, `cv2.convexHull`),
//! resampling (`cv2.resize` with `INTER_LINEAR` / `INTER_AREA`, `cv2.warpAffine`), filtering
//! (`cv2.GaussianBlur`, `cv2.filter2D`, `cv2.dilate`, `cv2.erode`), `np.gradient`,
//! `np.percentile` and a JPEG round trip (`cv2.imencode` + `cv2.imdecode`).
//!
//! [`ImageF`] is the `(H, W, C)` `float32` array of the Python code, row-major and interleaved
//! (BGR order for colour images, as OpenCV keeps them).
//!
//! ## Drawing
//!
//! Every primitive is a shape with a signed distance (negative inside): a disk, a capsule (a line
//! with round caps, which is what `cv2.line` draws for `thickness > 1`) and a convex polygon.
//! `cv2.LINE_8` fills become "pixel centre within `offset` of the shape", `cv2.LINE_AA` fills a
//! coverage ramp `clamp(offset + 0.5 - distance, 0, 1)` blended into the canvas. The offsets were
//! fitted to OpenCV 4 drawing with `shift=2` (pixel counts of filled shapes over several centres):
//!
//! | primitive | `LINE_8` inside when | `LINE_AA` coverage offset |
//! | --- | --- | --- |
//! | `cv2.circle(r, -1)` | `d <= r + 0.2` | `r + 0.5` |
//! | `cv2.line(th > 1)` | `d <= (th + th % 2) / 2 + 0.45` | `(th + th % 2) / 2 + 0.65` |
//! | `cv2.line(th = 1)` | Bresenham between the rounded end points | `0.65` |
//! | `cv2.fillConvexPoly` | `sd <= 0.5` | `sd` ramp offset `0.69` |
//!
//! `// differs from Python:` OpenCV rasterises with fixed-point edge walking, so single boundary
//! pixels differ; filled areas agree to a few percent (the renderer tests pin that against cv2).
//! As in OpenCV, `LINE_AA` only anti-aliases 8-bit images: on a float image (the motion-blur
//! kernel, the food pellets) the hard rule applies, which [`draw_hard_f`] implements.

use anyhow::{ensure, Context, Result};

use crate::algos::image::Bgr;

/// An `(H, W, C)` float image, interleaved row-major: `data[(y * width + x) * channels + c]`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ImageF {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Channels per pixel (3 for BGR, 1 for grey levels and masks).
    pub channels: usize,
    /// The samples, `width * height * channels` of them.
    pub data: Vec<f32>,
}

impl ImageF {
    /// `np.full((h, w, c), value, np.float32)`.
    pub fn new(width: usize, height: usize, channels: usize, value: f32) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![value; width * height * channels],
        }
    }

    /// Wraps samples already laid out as `(H, W, C)`.
    pub fn from_data(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        ensure!(
            data.len() == width * height * channels,
            "image {width}x{height}x{channels} needs {} samples, got {}",
            width * height * channels,
            data.len()
        );
        Ok(Self {
            width,
            height,
            channels,
            data,
        })
    }

    /// `image.astype(np.float32)` of an 8-bit BGR image.
    pub fn from_bgr(image: &Bgr) -> Self {
        Self {
            width: image.width as usize,
            height: image.height as usize,
            channels: 3,
            data: image.pixels.iter().map(|v| f32::from(*v)).collect(),
        }
    }

    /// `np.clip(img, 0, 255).astype(np.uint8)`: clipped, then truncated towards zero.
    pub fn to_bgr_trunc(&self) -> Bgr {
        debug_assert_eq!(self.channels, 3);
        Bgr {
            width: self.width as u32,
            height: self.height as u32,
            pixels: self.data.iter().map(|v| clip_u8_trunc(*v)).collect(),
        }
    }

    /// What OpenCV returns for an 8-bit input: values rounded to the nearest grey level and
    /// saturated (`saturate_cast<uchar>`).
    pub fn to_bgr_round(&self) -> Bgr {
        debug_assert_eq!(self.channels, 3);
        Bgr {
            width: self.width as u32,
            height: self.height as u32,
            pixels: self.data.iter().map(|v| clip_u8_round(*v)).collect(),
        }
    }

    /// Rounds every sample to an 8-bit level in place, keeping the float type (an OpenCV call on a
    /// `uint8` array whose result goes on as `uint8`).
    pub fn quantize(&mut self) {
        for v in self.data.iter_mut() {
            *v = f32::from(clip_u8_round(*v));
        }
    }

    /// Index of sample `(x, y, c)`.
    #[inline]
    pub fn idx(&self, x: usize, y: usize, c: usize) -> usize {
        (y * self.width + x) * self.channels + c
    }

    /// The sample at `(x, y, c)`.
    #[inline]
    pub fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.data[self.idx(x, y, c)]
    }

    /// `img[:, ::-1]`.
    pub fn flip_horizontal(&self) -> Self {
        let mut out = self.clone();
        for y in 0..self.height {
            for x in 0..self.width {
                for c in 0..self.channels {
                    out.data[self.idx(x, y, c)] = self.at(self.width - 1 - x, y, c);
                }
            }
        }
        out
    }

    /// `img.mean(-1, keepdims=True)` repeated over the channels (`np.repeat(..., 3, -1)`).
    pub fn to_mono(&self) -> Self {
        let mut out = self.clone();
        let c = self.channels;
        for px in out.data.chunks_exact_mut(c) {
            let m = px.iter().map(|v| f64::from(*v)).sum::<f64>() / c as f64;
            for v in px.iter_mut() {
                *v = m as f32;
            }
        }
        out
    }

    /// `img.mean()` over every sample.
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|v| f64::from(*v)).sum::<f64>() / self.data.len() as f64
    }

    /// `img.reshape(-1, C).mean(0)`: the mean of each channel.
    pub fn channel_means(&self) -> Vec<f64> {
        let c = self.channels;
        let mut sums = vec![0.0f64; c];
        for px in self.data.chunks_exact(c) {
            for (s, v) in sums.iter_mut().zip(px) {
                *s += f64::from(*v);
            }
        }
        let n = (self.width * self.height).max(1) as f64;
        sums.iter().map(|s| s / n).collect()
    }

    /// `np.clip(img, lo, hi)` in place.
    pub fn clip(&mut self, lo: f32, hi: f32) {
        for v in self.data.iter_mut() {
            *v = v.clamp(lo, hi);
        }
    }
}

/// `np.clip(v, 0, 255).astype(np.uint8)`.
#[inline]
pub fn clip_u8_trunc(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    v.clamp(0.0, 255.0) as u8
}

/// `saturate_cast<uchar>(v)`: round half to even (`cvRound`), then saturate.
#[inline]
pub fn clip_u8_round(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// BGR 8-bit image from a float one by `np.clip(...).astype(np.uint8)`.
pub fn bgr_from(image: &ImageF) -> Bgr {
    image.to_bgr_trunc()
}

// ------------------------------------------------------------------------------------ resampling

/// `cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)`: `src = (dst + 0.5) * scale - 0.5`,
/// clamped to the source.
///
/// `// differs from Python:` OpenCV interpolates 8-bit images with 11-bit fixed-point weights; this
/// is the exact float mapping.
pub fn resize_linear(src: &ImageF, width: usize, height: usize) -> ImageF {
    let c = src.channels;
    let mut out = ImageF::new(width, height, c, 0.0);
    if src.width == 0 || src.height == 0 || width == 0 || height == 0 {
        return out;
    }
    let sx = src.width as f64 / width as f64;
    let sy = src.height as f64 / height as f64;
    let xs: Vec<(usize, usize, f32)> = (0..width)
        .map(|dx| linear_tap(dx, sx, src.width))
        .collect();
    for dy in 0..height {
        let (y0, y1, wy) = linear_tap(dy, sy, src.height);
        for (dx, &(x0, x1, wx)) in xs.iter().enumerate() {
            for ch in 0..c {
                let a = src.at(x0, y0, ch);
                let b = src.at(x1, y0, ch);
                let cc = src.at(x0, y1, ch);
                let d = src.at(x1, y1, ch);
                let top = a + (b - a) * wx;
                let bottom = cc + (d - cc) * wx;
                let i = out.idx(dx, dy, ch);
                out.data[i] = top + (bottom - top) * wy;
            }
        }
    }
    out
}

fn linear_tap(d: usize, scale: f64, n: usize) -> (usize, usize, f32) {
    let f = ((d as f64 + 0.5) * scale - 0.5).clamp(0.0, n as f64 - 1.0);
    let i0 = f.floor() as usize;
    let i1 = (i0 + 1).min(n - 1);
    (i0, i1, (f - i0 as f64) as f32)
}

/// `cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)`: each output pixel averages the source
/// area it covers (fractional pixels weighted by their overlap). Upscaling falls back to
/// [`resize_linear`].
///
/// `// differs from Python:` OpenCV's `INTER_AREA` upscale is a bilinear variant with its own
/// weights; every caller here only shrinks.
pub fn resize_area(src: &ImageF, width: usize, height: usize) -> ImageF {
    if width == 0 || height == 0 || src.width == 0 || src.height == 0 {
        return ImageF::new(width, height, src.channels, 0.0);
    }
    if width > src.width || height > src.height {
        return resize_linear(src, width, height);
    }
    let wx = area_weights(src.width, width);
    let wy = area_weights(src.height, height);
    let c = src.channels;
    // Horizontal pass, then vertical.
    let mut tmp = vec![0.0f64; width * src.height * c];
    for y in 0..src.height {
        for (dx, taps) in wx.iter().enumerate() {
            for &(sx, w) in taps {
                for ch in 0..c {
                    tmp[(y * width + dx) * c + ch] += w * f64::from(src.at(sx, y, ch));
                }
            }
        }
    }
    let mut out = ImageF::new(width, height, c, 0.0);
    for (dy, taps) in wy.iter().enumerate() {
        for dx in 0..width {
            for ch in 0..c {
                let mut acc = 0.0f64;
                for &(sy, w) in taps {
                    acc += w * tmp[(sy * width + dx) * c + ch];
                }
                let i = out.idx(dx, dy, ch);
                out.data[i] = acc as f32;
            }
        }
    }
    out
}

/// Overlap weights of the source pixels each output pixel covers, normalised to sum to one.
fn area_weights(n_src: usize, n_dst: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = n_src as f64 / n_dst as f64;
    (0..n_dst)
        .map(|d| {
            let lo = d as f64 * scale;
            let hi = ((d + 1) as f64 * scale).min(n_src as f64);
            let mut taps = Vec::new();
            let mut s = lo.floor() as usize;
            while (s as f64) < hi && s < n_src {
                let overlap = (hi.min(s as f64 + 1.0) - lo.max(s as f64)).max(0.0);
                if overlap > 1e-12 {
                    taps.push((s, overlap / (hi - lo)));
                }
                s += 1;
            }
            taps
        })
        .collect()
}

/// `cv2.invertAffineTransform`.
pub fn invert_affine(m: &[[f64; 3]; 2]) -> [[f64; 3]; 2] {
    let [[a, b, c], [d, e, f]] = *m;
    let det = a * e - b * d;
    let det = if det.abs() < 1e-300 { 0.0 } else { 1.0 / det };
    let (ia, ib, id, ie) = (e * det, -b * det, -d * det, a * det);
    [
        [ia, ib, -(ia * c + ib * f)],
        [id, ie, -(id * c + ie * f)],
    ]
}

/// `cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)`:
/// `m` maps source pixels to output pixels; every output pixel samples the source bilinearly at
/// the inverse image of its centre, taps outside the source counting as zero.
pub fn warp_affine(src: &ImageF, m: &[[f64; 3]; 2], width: usize, height: usize) -> ImageF {
    let back = invert_affine(m);
    let c = src.channels;
    let mut out = ImageF::new(width, height, c, 0.0);
    let (w, h) = (src.width as i64, src.height as i64);
    for y in 0..height {
        for x in 0..width {
            let sx = back[0][0] * x as f64 + back[0][1] * y as f64 + back[0][2];
            let sy = back[1][0] * x as f64 + back[1][1] * y as f64 + back[1][2];
            if !sx.is_finite() || !sy.is_finite() {
                continue;
            }
            let x0f = sx.floor();
            let y0f = sy.floor();
            if x0f < -2.0 || y0f < -2.0 || x0f > w as f64 + 1.0 || y0f > h as f64 + 1.0 {
                continue;
            }
            let (x0, y0) = (x0f as i64, y0f as i64);
            let (wx, wy) = (sx - x0f, sy - y0f);
            let taps = [
                (x0, y0, (1.0 - wx) * (1.0 - wy)),
                (x0 + 1, y0, wx * (1.0 - wy)),
                (x0, y0 + 1, (1.0 - wx) * wy),
                (x0 + 1, y0 + 1, wx * wy),
            ];
            for ch in 0..c {
                let mut acc = 0.0f64;
                for &(tx, ty, tw) in &taps {
                    if tx >= 0 && ty >= 0 && tx < w && ty < h {
                        acc += tw * f64::from(src.at(tx as usize, ty as usize, ch));
                    }
                }
                let i = out.idx(x, y, ch);
                out.data[i] = acc as f32;
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------- filtering

/// `BORDER_REFLECT_101` (`gfedcb|abcdefgh|gfedcba`), OpenCV's default filter border.
#[inline]
pub fn reflect101(i: i64, n: usize) -> usize {
    let n = n as i64;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * n - 2 - i;
        } else {
            return i as usize;
        }
    }
}

/// `cv2.getGaussianKernel(ksize, sigma)` for `float32` data: the fixed small kernels OpenCV uses
/// when `sigma <= 0` and `ksize <= 7`, else the sampled, normalised Gaussian with
/// `sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8` when `sigma <= 0`.
pub fn gaussian_kernel(ksize: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        match ksize {
            1 => return vec![1.0],
            3 => return vec![0.25, 0.5, 0.25],
            5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
            7 => return vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
            _ => {}
        }
    }
    let sigma = if sigma > 0.0 {
        sigma
    } else {
        0.3 * ((ksize as f64 - 1.0) * 0.5 - 1.0) + 0.8
    };
    let half = (ksize as f64 - 1.0) / 2.0;
    let raw: Vec<f64> = (0..ksize)
        .map(|i| (-(i as f64 - half).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|v| (v / sum) as f32).collect()
}

/// `cv2.GaussianBlur(img, (ksize, ksize), sigma)`; `ksize = 0` derives the size from `sigma` the
/// way OpenCV does for float images (`round(sigma * 4 * 2 + 1) | 1`).
pub fn gaussian_blur(src: &ImageF, ksize: usize, sigma: f64) -> ImageF {
    let ksize = if ksize == 0 {
        ((sigma * 8.0 + 1.0).round_ties_even() as usize) | 1
    } else {
        ksize
    };
    let k = gaussian_kernel(ksize, sigma);
    separable_filter(src, &k, &k)
}

/// A separable correlation with `kx` along x and `ky` along y, `BORDER_REFLECT_101`.
pub fn separable_filter(src: &ImageF, kx: &[f32], ky: &[f32]) -> ImageF {
    let c = src.channels;
    let (w, h) = (src.width, src.height);
    let ax = (kx.len() / 2) as i64;
    let ay = (ky.len() / 2) as i64;
    let mut tmp = ImageF::new(w, h, c, 0.0);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for (i, kv) in kx.iter().enumerate() {
                    let sx = reflect101(x as i64 + i as i64 - ax, w);
                    acc += kv * src.at(sx, y, ch);
                }
                let at = tmp.idx(x, y, ch);
                tmp.data[at] = acc;
            }
        }
    }
    let mut out = ImageF::new(w, h, c, 0.0);
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let mut acc = 0.0f32;
                for (i, kv) in ky.iter().enumerate() {
                    let sy = reflect101(y as i64 + i as i64 - ay, h);
                    acc += kv * tmp.at(x, sy, ch);
                }
                let at = out.idx(x, y, ch);
                out.data[at] = acc;
            }
        }
    }
    out
}

/// `cv2.filter2D(img, -1, kernel)`: correlation with a `(kh, kw)` kernel anchored at its centre,
/// `BORDER_REFLECT_101`. Zero taps are skipped, which keeps the sparse line and disk kernels cheap.
pub fn filter2d(src: &ImageF, kernel: &[f32], kw: usize, kh: usize) -> ImageF {
    let c = src.channels;
    let (w, h) = (src.width, src.height);
    let ax = (kw / 2) as i64;
    let ay = (kh / 2) as i64;
    let taps: Vec<(i64, i64, f32)> = (0..kh)
        .flat_map(|j| (0..kw).map(move |i| (i, j)))
        .filter_map(|(i, j)| {
            let v = kernel[j * kw + i];
            (v != 0.0).then_some((i as i64 - ax, j as i64 - ay, v))
        })
        .collect();
    // Pad once with the reflected border, then add each tap as a shifted copy of whole rows.
    let (pw, ph) = (w + 2 * ax as usize, h + 2 * ay as usize);
    let mut padded = vec![0.0f32; pw * ph * c];
    for py in 0..ph {
        let sy = reflect101(py as i64 - ay, h);
        for px in 0..pw {
            let sx = reflect101(px as i64 - ax, w);
            let (di, si) = ((py * pw + px) * c, (sy * w + sx) * c);
            padded[di..di + c].copy_from_slice(&src.data[si..si + c]);
        }
    }
    let mut out = ImageF::new(w, h, c, 0.0);
    let row = w * c;
    for &(dx, dy, kv) in &taps {
        for y in 0..h {
            let start = ((y as i64 + dy + ay) as usize * pw + (dx + ax) as usize) * c;
            let src_row = &padded[start..start + row];
            let dst_row = &mut out.data[y * row..(y + 1) * row];
            for (o, v) in dst_row.iter_mut().zip(src_row) {
                *o += kv * v;
            }
        }
    }
    out
}

/// `cv2.dilate(mask, np.ones((2r+1, 2r+1)))` on a 0/1 mask; pixels outside the image are ignored
/// (OpenCV's default morphology border).
pub fn dilate(mask: &[u8], width: usize, height: usize, r: usize) -> Vec<u8> {
    morph(mask, width, height, r, true)
}

/// `cv2.erode(mask, np.ones((2r+1, 2r+1)))`; see [`dilate`] for the border.
pub fn erode(mask: &[u8], width: usize, height: usize, r: usize) -> Vec<u8> {
    morph(mask, width, height, r, false)
}

fn morph(mask: &[u8], width: usize, height: usize, r: usize, max: bool) -> Vec<u8> {
    let mut out = vec![0u8; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let mut v = if max { 0u8 } else { u8::MAX };
            for yy in y.saturating_sub(r)..(y + r + 1).min(height) {
                for xx in x.saturating_sub(r)..(x + r + 1).min(width) {
                    let s = mask[yy * width + xx];
                    v = if max { v.max(s) } else { v.min(s) };
                }
            }
            out[y * width + x] = v;
        }
    }
    out
}

/// `np.hypot(*np.gradient(g))` for a 2-D array: central differences inside, one-sided first
/// differences on the edges.
pub fn gradient_magnitude(g: &[f64], width: usize, height: usize) -> Vec<f64> {
    let d = |at: &dyn Fn(usize) -> f64, i: usize, n: usize| -> f64 {
        if n < 2 {
            0.0
        } else if i == 0 {
            at(1) - at(0)
        } else if i == n - 1 {
            at(n - 1) - at(n - 2)
        } else {
            (at(i + 1) - at(i - 1)) / 2.0
        }
    };
    let mut out = vec![0.0f64; g.len()];
    for y in 0..height {
        for x in 0..width {
            let gy = d(&|yy| g[yy * width + x], y, height);
            let gx = d(&|xx| g[y * width + xx], x, width);
            out[y * width + x] = gy.hypot(gx);
        }
    }
    out
}

/// `np.percentile(values, p)` with the default linear interpolation; `NaN` for no values.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = p / 100.0 * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

/// `np.median(values)`.
pub fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

/// `cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1])` for an 8-bit
/// BGR image.
///
/// `// differs from Python:` the pure-Rust encoder of the `image` crate stands in for libjpeg, so
/// the compression artefacts are of the same kind and strength but not the same bytes.
pub fn jpeg_roundtrip(image: &Bgr, quality: u8) -> Result<Bgr> {
    let rgb: Vec<u8> = image
        .pixels
        .chunks_exact(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality.max(1));
    enc.encode(
        &rgb,
        image.width,
        image.height,
        image::ExtendedColorType::Rgb8,
    )
    .context("jpeg encode")?;
    let decoded = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
        .context("jpeg decode")?
        .to_rgb8();
    let pixels = decoded
        .as_raw()
        .chunks_exact(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    Ok(Bgr {
        width: image.width,
        height: image.height,
        pixels,
    })
}

/// `cv2.imread(path, cv2.IMREAD_COLOR)`: any PNG/JPEG as 8-bit BGR.
pub fn read_bgr(path: &std::path::Path) -> Result<Bgr> {
    let decoded = image::ImageReader::open(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("cannot read {}", path.display()))?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?
        .to_rgb8();
    let (width, height) = decoded.dimensions();
    let pixels = decoded
        .as_raw()
        .chunks_exact(3)
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    Ok(Bgr {
        width,
        height,
        pixels,
    })
}

// --------------------------------------------------------------------------------------- drawing

/// A shape to fill, in pixel coordinates (pixel centres at integers, as in OpenCV).
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// `cv2.circle(img, center, r, color, -1)`.
    Disk {
        /// Centre.
        center: [f64; 2],
        /// Radius in pixels.
        radius: f64,
    },
    /// `cv2.line(img, a, b, color, thickness)` with `thickness > 1`: a segment with round caps.
    Capsule {
        /// First end point.
        a: [f64; 2],
        /// Second end point.
        b: [f64; 2],
        /// `cv2.line`'s thickness.
        thickness: i64,
    },
    /// `cv2.fillConvexPoly(img, points, color)`.
    Polygon {
        /// The vertices, in either orientation.
        points: Vec<[f64; 2]>,
    },
}

impl Shape {
    /// `cv2.line(...)` for any thickness (`thickness = 1` is drawn as a Bresenham line by the hard
    /// painter and as a thin capsule by the anti-aliased one).
    pub fn line(a: [f64; 2], b: [f64; 2], thickness: i64) -> Self {
        Self::Capsule {
            a,
            b,
            thickness: thickness.max(1),
        }
    }

    /// `(LINE_8 offset, LINE_AA offset)` of this shape's distance, from the table in the module docs.
    fn offsets(&self) -> (f64, f64) {
        match self {
            Shape::Disk { radius, .. } => (radius + 0.2, radius + 0.5),
            Shape::Capsule { thickness, .. } => {
                let half = if *thickness <= 1 {
                    0.0
                } else {
                    (*thickness + (*thickness & 1)) as f64 / 2.0
                };
                (if *thickness <= 1 { 0.5 } else { half + 0.45 }, half + 0.65)
            }
            Shape::Polygon { .. } => (0.5, 0.69),
        }
    }

    /// Distance of `(x, y)` to the ideal shape (a point, a segment or a polygon), negative inside a
    /// polygon.
    fn distance(&self, x: f64, y: f64) -> f64 {
        match self {
            Shape::Disk { center, .. } => (x - center[0]).hypot(y - center[1]),
            Shape::Capsule { a, b, .. } => segment_distance([x, y], *a, *b),
            Shape::Polygon { points } => polygon_signed_distance(points, [x, y]),
        }
    }

    /// The pixel box `[x0, y0, x1, y1]` (inclusive) the shape and its ramp can touch.
    fn bounds(&self, width: usize, height: usize) -> Option<[usize; 4]> {
        let (_, aa) = self.offsets();
        let pad = aa + 1.0;
        let (lo, hi) = match self {
            Shape::Disk { center, .. } => (*center, *center),
            Shape::Capsule { a, b, .. } => (
                [a[0].min(b[0]), a[1].min(b[1])],
                [a[0].max(b[0]), a[1].max(b[1])],
            ),
            Shape::Polygon { points } => {
                let mut lo = [f64::INFINITY; 2];
                let mut hi = [f64::NEG_INFINITY; 2];
                for p in points {
                    for d in 0..2 {
                        lo[d] = lo[d].min(p[d]);
                        hi[d] = hi[d].max(p[d]);
                    }
                }
                (lo, hi)
            }
        };
        if !(lo[0].is_finite() && lo[1].is_finite() && hi[0].is_finite() && hi[1].is_finite()) {
            return None;
        }
        let x0 = (lo[0] - pad).floor().max(0.0);
        let y0 = (lo[1] - pad).floor().max(0.0);
        let x1 = (hi[0] + pad).ceil().min(width as f64 - 1.0);
        let y1 = (hi[1] + pad).ceil().min(height as f64 - 1.0);
        if x1 < x0 || y1 < y0 || width == 0 || height == 0 {
            return None;
        }
        Some([x0 as usize, y0 as usize, x1 as usize, y1 as usize])
    }
}

fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * ab[0]).hypot(p[1] - a[1] - t * ab[1])
}

/// Signed distance to a convex polygon (negative inside). Degenerate polygons (fewer than three
/// distinct points or no area) are their outline, so they still draw as lines, like OpenCV's.
fn polygon_signed_distance(points: &[[f64; 2]], p: [f64; 2]) -> f64 {
    let n = points.len();
    if n == 0 {
        return f64::INFINITY;
    }
    if n == 1 {
        return (p[0] - points[0][0]).hypot(p[1] - points[0][1]);
    }
    let mut area2 = 0.0;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    let mut outline = f64::INFINITY;
    for i in 0..n {
        outline = outline.min(segment_distance(p, points[i], points[(i + 1) % n]));
    }
    if area2.abs() < 1e-9 {
        return outline;
    }
    let sign = area2.signum();
    let inside = (0..n).all(|i| {
        let (a, b) = (points[i], points[(i + 1) % n]);
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        cross * sign >= 0.0
    });
    if inside {
        -outline
    } else {
        outline
    }
}

/// `cv2.convexHull(points)`: Andrew's monotone chain, counter-clockwise in image coordinates.
pub fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], *p) <= 0.0
        {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], *p) <= 0.0
        {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Bresenham between the rounded end points: `cv2.line(..., 1, cv2.LINE_8)`.
fn bresenham(a: [f64; 2], b: [f64; 2], mut put: impl FnMut(i64, i64)) {
    let (mut x0, mut y0) = (a[0].round_ties_even() as i64, a[1].round_ties_even() as i64);
    let (x1, y1) = (b[0].round_ties_even() as i64, b[1].round_ties_even() as i64);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(x0, y0);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Fills `shape` into a single-channel 8-bit mask with `value` (`cv2.LINE_8`).
pub fn draw_hard(mask: &mut [u8], width: usize, height: usize, shape: &Shape, value: u8) {
    if let Shape::Capsule { a, b, thickness } = shape {
        if *thickness <= 1 {
            bresenham(*a, *b, |x, y| {
                if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                    mask[y as usize * width + x as usize] = value;
                }
            });
            return;
        }
    }
    let Some([x0, y0, x1, y1]) = shape.bounds(width, height) else {
        return;
    };
    let (off, _) = shape.offsets();
    for y in y0..=y1 {
        for x in x0..=x1 {
            if shape.distance(x as f64, y as f64) <= off + 1e-9 {
                mask[y * width + x] = value;
            }
        }
    }
}

/// [`draw_hard`] on a float image: every channel of a covered pixel is set to `color` (what
/// `cv2.LINE_AA` does on a non-8-bit image).
pub fn draw_hard_f(img: &mut ImageF, shape: &Shape, color: &[f32]) {
    let (w, h) = (img.width, img.height);
    let mut mask = vec![0u8; w * h];
    draw_hard(&mut mask, w, h, shape, 1);
    for (i, m) in mask.iter().enumerate() {
        if *m > 0 {
            for (c, v) in color.iter().enumerate().take(img.channels) {
                img.data[i * img.channels + c] = *v;
            }
        }
    }
}

/// Blends `shape` into an 8-bit canvas held as floats (`cv2.LINE_AA`): every touched pixel moves
/// towards `color` by its coverage and is rounded back to a grey level, as OpenCV's 8-bit
/// anti-aliased fill does.
pub fn draw_aa(img: &mut ImageF, shape: &Shape, color: &[f32]) {
    let Some([x0, y0, x1, y1]) = shape.bounds(img.width, img.height) else {
        return;
    };
    let (_, off) = shape.offsets();
    let c = img.channels;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let cov = (off + 0.5 - shape.distance(x as f64, y as f64)).clamp(0.0, 1.0) as f32;
            if cov <= 0.0 {
                continue;
            }
            let base = (y * img.width + x) * c;
            for (ch, col) in color.iter().enumerate().take(c) {
                let v = &mut img.data[base + ch];
                *v = (*v + (col - *v) * cov).round_ties_even();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_kernel_small_sizes_are_opencv_tables() {
        assert_eq!(gaussian_kernel(3, 0.0), vec![0.25, 0.5, 0.25]);
        let k = gaussian_kernel(9, 1.0);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(k[4] > k[3] && k[3] > k[2]);
    }

    #[test]
    fn reflect101_matches_opencv() {
        let n = 5;
        let got: Vec<usize> = (-3..8).map(|i| reflect101(i, n)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1]);
    }

    #[test]
    fn hull_and_polygon_fill() {
        let hull = convex_hull(&[[0.0, 0.0], [10.0, 0.0], [5.0, 5.0], [10.0, 10.0], [0.0, 10.0]]);
        assert_eq!(hull.len(), 4);
        let mut mask = vec![0u8; 20 * 20];
        draw_hard(
            &mut mask,
            20,
            20,
            &Shape::Polygon { points: hull },
            255,
        );
        // 11 x 11 pixel centres lie on or inside the square.
        assert_eq!(mask.iter().filter(|v| **v > 0).count(), 121);
    }

    #[test]
    fn disks_and_lines_cover_their_area() {
        let mut mask = vec![0u8; 100 * 100];
        draw_hard(
            &mut mask,
            100,
            100,
            &Shape::Disk {
                center: [50.0, 50.0],
                radius: 2.0,
            },
            1,
        );
        // cv2.circle(r=2) fills 13 pixels.
        assert_eq!(mask.iter().filter(|v| **v > 0).count(), 13);
        let mut mask = vec![0u8; 100 * 100];
        draw_hard(&mut mask, 100, 100, &Shape::line([10.0, 10.0], [20.0, 13.0], 1), 1);
        assert_eq!(mask.iter().filter(|v| **v > 0).count(), 11);
    }

    #[test]
    fn area_resize_averages_blocks() {
        let img = ImageF::from_data(4, 2, 1, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
            .unwrap();
        let out = resize_area(&img, 2, 1);
        assert_eq!(out.data, vec![5.0, 9.0]);
        let out = resize_area(&img, 3, 2);
        // Source pixels 0, 1 (1/3 of it): (0 + 2/3... ) / (4/3).
        assert!((out.data[0] - (0.0 + 2.0 / 3.0) / (4.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn warp_identity_and_border() {
        let img = ImageF::new(4, 4, 1, 10.0);
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let out = warp_affine(&img, &id, 4, 4);
        assert!(out.data.iter().all(|v| (*v - 10.0).abs() < 1e-6));
        let shift = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]];
        let out = warp_affine(&img, &shift, 4, 4);
        // The first column samples x = -0.5: half its taps are outside.
        assert!((out.at(0, 1, 0) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn numpy_percentile_and_gradient() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0], 95.0) - 3.85).abs() < 1e-12);
        let g = vec![0.0, 1.0, 4.0, 9.0];
        let m = gradient_magnitude(&g, 4, 1);
        assert_eq!(m, vec![1.0, 2.0, 4.0, 5.0]);
    }
}
