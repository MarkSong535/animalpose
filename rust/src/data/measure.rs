//! What a recording does to the image, measured: the measurement half of
//! `src/animalpose/data/degrade.py` (`edge_width`, `motion_smear`, `disk_kernel` and the defocus
//! branch of `degrade`), the measurements of `src/animalpose/cli/data/audit.py` (`noise_sigma`,
//! `animal_patch`, the Laplacian sharpness) and the OpenCV primitives they call.
//!
//! | Python / OpenCV | Rust |
//! | --- | --- |
//! | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | [`bgr_to_gray`] |
//! | `cv2.GaussianBlur(g, (0, 0), sigma)` (float) | [`gaussian_blur`] |
//! | `cv2.Laplacian(g, cv2.CV_32F)` | [`laplacian`] |
//! | `cv2.filter2D(img, -1, kernel)` | [`filter2d`] |
//! | `cv2.warpAffine(..., INTER_LINEAR, BORDER_CONSTANT)` / `geometry.py:warp` | [`warp_bgr`] |
//! | `np.gradient(g)` | [`gradient`] |
//! | `data/degrade.py:edge_width` / `motion_smear` / `disk_kernel` | [`edge_width`] / [`motion_smear`] / [`disk_kernel`] |
//! | `data/degrade.py:degrade` with only defocus on (what `calibrate_psf` runs) | [`degrade_defocus`] |
//! | `cli/data/audit.py:noise_sigma` / `animal_patch` | [`noise_sigma`] / [`animal_patch`] |
//! | `cv2.imread(path, IMREAD_UNCHANGED)` + `frames.py:to_bgr8` | [`read_bgr`] |
//! | `zlib.crc32` | [`crc32`] |
//!
//! The OpenCV numbers are those of the OpenCV the reference environment ships (5.0): its
//! `COLOR_BGR2GRAY` is 15-bit fixed point and its `INTER_LINEAR` warp samples in `float32` with
//! round-to-nearest, both reproduced exactly here.
//!
//! `// differs from Python:` the full `degrade` (motion blur, resolution loss, noise, JPEG) is not
//! here: the port of `data/degrade.py` as a whole lives with the training side of the crate. The
//! PSF calibration of `animalpose data audit` switches every other effect off, so its defocus
//! branch is all it needs.

use crate::algos::blob::{dilate_rect, erode_rect};
use crate::algos::geometry::{box_xyxy_to_cs, crop_affine, invert_affine, Box4};
use crate::algos::image::{Bgr, GrayF, Image};
use anyhow::{Context, Result};
use std::path::Path;

/// `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)`: OpenCV's 15-bit fixed-point weights.
pub fn bgr_to_gray(img: &Bgr) -> GrayF {
    let pixels = img
        .pixels
        .chunks_exact(3)
        .map(|p| {
            let y = (u32::from(p[0]) * 3735 + u32::from(p[1]) * 19235 + u32::from(p[2]) * 9798 + 16384)
                >> 15;
            y.min(255) as f32
        })
        .collect();
    GrayF {
        width: img.width,
        height: img.height,
        pixels,
    }
}

/// OpenCV's `BORDER_REFLECT_101` index (`gfedcb|abcdefgh|gfedcba`).
fn reflect101(i: i64, n: i64) -> usize {
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

/// `cv2.filter2D(src, -1, kernel)` on one float plane: correlation with the kernel centred on each
/// pixel, `BORDER_REFLECT_101` outside.
///
/// OpenCV sums kernels of fewer than 50 taps directly, one fused multiply-add per non-zero tap in
/// row-major order and in `float32` (its NEON path; reproduced here with [`f32::mul_add`]), and
/// larger ones through the DFT, whose result equals the sum taken in `float64`. The two matter
/// because `degrade` truncates the result to 8 bits, where a last-bit difference flips a grey
/// level.
pub fn filter2d(src: &[f32], width: usize, height: usize, kernel: &[f32], kw: usize, kh: usize) -> Vec<f32> {
    let (ax, ay) = (kw / 2, kh / 2);
    let mut out = vec![0.0f32; width * height];
    let cols: Vec<Vec<usize>> = (0..width)
        .map(|x| {
            (0..kw)
                .map(|i| reflect101(x as i64 + i as i64 - ax as i64, width as i64))
                .collect()
        })
        .collect();
    let taps: Vec<(usize, usize, f32)> = (0..kh)
        .flat_map(|j| (0..kw).map(move |i| (j, i)))
        .map(|(j, i)| (j, i, kernel[j * kw + i]))
        .filter(|t| t.2 != 0.0)
        .collect();
    let direct = kw * kh < 50;
    for y in 0..height {
        let rows: Vec<usize> = (0..kh)
            .map(|j| reflect101(y as i64 + j as i64 - ay as i64, height as i64))
            .collect();
        for x in 0..width {
            out[y * width + x] = if direct {
                taps.iter().fold(0.0f32, |acc, (j, i, k)| {
                    k.mul_add(src[rows[*j] * width + cols[x][*i]], acc)
                })
            } else {
                taps.iter()
                    .map(|(j, i, k)| f64::from(*k) * f64::from(src[rows[*j] * width + cols[x][*i]]))
                    .sum::<f64>() as f32
            };
        }
    }
    out
}

/// A separable filter (`cv2.sepFilter2D`): `kx` along rows, then `ky` along columns.
pub fn sep_filter(src: &[f32], width: usize, height: usize, kx: &[f32], ky: &[f32]) -> Vec<f32> {
    let horizontal = filter2d(src, width, height, kx, kx.len(), 1);
    filter2d(&horizontal, width, height, ky, 1, ky.len())
}

/// `cv2.getGaussianKernel(n, sigma, CV_32F)` for `sigma > 0`.
pub fn gaussian_kernel(n: usize, sigma: f64) -> Vec<f32> {
    let scale = -0.5 / (sigma * sigma);
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - (n as f64 - 1.0) / 2.0;
            (scale * x * x).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    k.iter().map(|v| (v / sum) as f32).collect()
}

/// `cv2.GaussianBlur(g, (0, 0), sigma)` for a float image: the kernel size OpenCV derives from
/// `sigma` (`round(8 sigma + 1) | 1`), `BORDER_REFLECT_101`.
pub fn gaussian_blur(g: &GrayF, sigma: f64) -> GrayF {
    let n = (((sigma * 8.0 + 1.0).round_ties_even()) as usize) | 1;
    let k = gaussian_kernel(n, sigma);
    let (w, h) = (g.width as usize, g.height as usize);
    GrayF {
        width: g.width,
        height: g.height,
        pixels: sep_filter(&g.pixels, w, h, &k, &k),
    }
}

/// `cv2.Laplacian(g, cv2.CV_32F)` (`ksize=1`: the 4-neighbour kernel).
pub fn laplacian(g: &GrayF) -> Vec<f32> {
    let k = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    filter2d(&g.pixels, g.width as usize, g.height as usize, &k, 3, 3)
}

/// `np.var` (population variance).
pub fn variance(v: &[f32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len() as f64;
    let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / n;
    v.iter().map(|x| (f64::from(*x) - mean).powi(2)).sum::<f64>() / n
}

/// `np.median`.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

/// `np.percentile` (linear interpolation).
pub fn percentile(values: &[f64], p: f64) -> f64 {
    let mut v = values.to_vec();
    crate::data::images::percentile(&mut v, p)
}

/// `np.std` (population).
pub fn std(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// `noise_sigma(g, roi)`: sensor noise as the robust sigma (`1.4826 * MAD`) of the high-pass
/// residual `g - GaussianBlur(g, 1.2)`, over the arena mask when one is given.
pub fn noise_sigma(g: &GrayF, roi: Option<&Image>) -> f64 {
    let blur = gaussian_blur(g, 1.2);
    let hp: Vec<f64> = g
        .pixels
        .iter()
        .zip(&blur.pixels)
        .enumerate()
        .filter(|(i, _)| roi.map_or(true, |r| r.pixels[*i] > 0))
        .map(|(_, (a, b))| f64::from(a - b))
        .collect();
    let m = median(&hp);
    let dev: Vec<f64> = hp.iter().map(|v| (v - m).abs()).collect();
    1.4826 * median(&dev)
}

/// `np.gradient(g)`: `(d/dy, d/dx)`, central differences inside, one-sided at the edges.
pub fn gradient(g: &GrayF) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (g.width as usize, g.height as usize);
    let p = &g.pixels;
    let mut gy = vec![0.0f32; w * h];
    let mut gx = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            gy[i] = if h < 2 {
                0.0
            } else if y == 0 {
                p[i + w] - p[i]
            } else if y == h - 1 {
                p[i] - p[i - w]
            } else {
                (p[i + w] - p[i - w]) / 2.0
            };
            gx[i] = if w < 2 {
                0.0
            } else if x == 0 {
                p[i + 1] - p[i]
            } else if x == w - 1 {
                p[i] - p[i - 1]
            } else {
                (p[i + 1] - p[i - 1]) / 2.0
            };
        }
    }
    (gy, gx)
}

/// `edge_width(patches, ring, percentile)`: the transition width of a dark subject's silhouette
/// against a brighter surround, in patch pixels, and its spread over the patches.
///
/// The peak gradient across a step of contrast `C` blurred over a width `w` is `C / w`, so per
/// patch the width is `(p95 - p5) / peak`, `peak` being the `percentile`-th gradient magnitude in
/// a `ring`-pixel band around the silhouette (`t > 0.5` of the contrast). Patches with too little
/// contrast (< 8 grey levels), subject (< 30 px) or edge (< 20 px) are skipped; `(0, 0)` when none
/// is left.
pub fn edge_width(patches: &[GrayF], ring: u32, pct: f64) -> (f64, f64) {
    let mut out = Vec::new();
    let k = 2 * ring + 1;
    for g in patches {
        let vals: Vec<f64> = g.pixels.iter().map(|v| f64::from(*v)).collect();
        // `np.percentile` of a float32 array is a float32.
        let hi = percentile(&vals, 95.0) as f32;
        let lo = percentile(&vals, 5.0) as f32;
        if hi - lo < 8.0 {
            continue;
        }
        let mask = Image {
            width: g.width,
            height: g.height,
            pixels: g
                .pixels
                .iter()
                .map(|v| u8::from(((hi - v) / (hi - lo)).clamp(0.0, 1.0) > 0.5))
                .collect(),
        };
        let area: usize = mask.pixels.iter().map(|v| usize::from(*v)).sum();
        if area < 30 {
            continue;
        }
        let dil = dilate_rect(&mask, k);
        let ero = erode_rect(&mask, k);
        let edge: Vec<bool> = dil
            .pixels
            .iter()
            .zip(&ero.pixels)
            .map(|(d, e)| *d & (1 - *e) > 0)
            .collect();
        if edge.iter().filter(|v| **v).count() < 20 {
            continue;
        }
        let (gy, gx) = gradient(g);
        let mags: Vec<f64> = gy
            .iter()
            .zip(&gx)
            .zip(&edge)
            .filter(|(_, e)| **e)
            .map(|((a, b), _)| f64::from(a.hypot(*b)))
            .collect();
        let peak = percentile(&mags, pct) as f32;
        if peak > 1e-6 {
            out.push(f64::from((hi - lo) / peak));
        }
    }
    if out.is_empty() {
        return (0.0, 0.0);
    }
    (median(&out), std(&out))
}

/// `motion_smear(displacement_px, exposure_fraction)`: how far the animal moves during one
/// exposure, the median displacement times the exposure fraction.
pub fn motion_smear(displacement_px: &[f64], exposure_fraction: f64) -> f64 {
    median(displacement_px) * exposure_fraction
}

/// `disk_kernel(radius)`: a normalised disk (`x^2 + y^2 <= r^2 + 0.5`) of side `2 ceil(r) + 1`.
pub fn disk_kernel(radius: f64) -> (Vec<f32>, usize) {
    let r = (radius.ceil() as i64).max(1);
    let n = (2 * r + 1) as usize;
    let mut k = Vec::with_capacity(n * n);
    for y in -r..=r {
        for x in -r..=r {
            k.push(f32::from(u8::from(((x * x + y * y) as f64) <= radius * radius + 0.5)));
        }
    }
    let s: f32 = k.iter().sum();
    (k.iter().map(|v| v / s).collect(), n)
}

/// Splits an interleaved BGR image into three float planes.
fn planes(img: &Bgr) -> [Vec<f32>; 3] {
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for (c, plane) in out.iter_mut().enumerate() {
        *plane = img.pixels.iter().skip(c).step_by(3).map(|v| f32::from(*v)).collect();
    }
    out
}

/// `degrade(image, rng, animal_px, profile=Profile(psf_px=psf, everything else off))`: the
/// defocus disk of radius `max(1, psf)` applied with `cv2.filter2D`, clipped and truncated back
/// to 8 bits. With `psf_jitter = 0` the jitter draw is exactly 1, so no random number matters.
pub fn degrade_defocus(img: &Bgr, psf_px: f64) -> Bgr {
    let (k, n) = disk_kernel(psf_px.max(1.0));
    let (w, h) = (img.width as usize, img.height as usize);
    let filtered: Vec<Vec<f32>> = planes(img)
        .iter()
        .map(|p| filter2d(p, w, h, &k, n, n))
        .collect();
    let mut pixels = vec![0u8; w * h * 3];
    for i in 0..w * h {
        for c in 0..3 {
            pixels[i * 3 + c] = filtered[c][i].clamp(0.0, 255.0) as u8;
        }
    }
    Bgr {
        width: img.width,
        height: img.height,
        pixels,
    }
}

/// `geometry.py:warp(image, m, out_size)` = `cv2.warpAffine(image, m, (out, out), INTER_LINEAR,
/// BORDER_CONSTANT)` on a BGR image: each output pixel samples the inverse map in `float32`,
/// bilinearly, with taps outside the frame read as zero, rounded to the nearest grey level.
pub fn warp_bgr(img: &Bgr, m: [[f64; 3]; 2], out_size: u32) -> Bgr {
    let inv = invert_affine(m);
    let mi: [[f32; 3]; 2] = [
        [inv[0][0] as f32, inv[0][1] as f32, inv[0][2] as f32],
        [inv[1][0] as f32, inv[1][1] as f32, inv[1][2] as f32],
    ];
    let (w, h) = (i64::from(img.width), i64::from(img.height));
    let n = out_size as usize;
    let mut pixels = vec![0u8; n * n * 3];
    for y in 0..n {
        for x in 0..n {
            let (xf, yf) = (x as f32, y as f32);
            let sxf = mi[0][0] * xf + mi[0][1] * yf + mi[0][2];
            let syf = mi[1][0] * xf + mi[1][1] * yf + mi[1][2];
            if !sxf.is_finite() || !syf.is_finite() {
                continue;
            }
            let (fx, fy) = (sxf.floor(), syf.floor());
            let (ax, ay) = (sxf - fx, syf - fy);
            let (sx, sy) = (fx as i64, fy as i64);
            for c in 0..3 {
                let at = |xx: i64, yy: i64| -> f32 {
                    if xx >= 0 && xx < w && yy >= 0 && yy < h {
                        f32::from(img.pixels[((yy * w + xx) * 3) as usize + c])
                    } else {
                        0.0
                    }
                };
                let v = (at(sx, sy) * (1.0 - ax) + at(sx + 1, sy) * ax) * (1.0 - ay)
                    + (at(sx, sy + 1) * (1.0 - ax) + at(sx + 1, sy + 1) * ax) * ay;
                pixels[(y * n + x) * 3 + c] = v.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Bgr {
        width: out_size,
        height: out_size,
        pixels,
    }
}

/// Patch side `animal_patch` resamples to (`audit.py:PATCH`).
pub const PATCH: u32 = 64;

/// `animal_patch(img, box, out)`: `(grey patch, patch px per image px)`, the box plus 40 % context
/// resampled to `out` pixels.
pub fn animal_patch(img: &Bgr, bbox: Box4, out: u32) -> (GrayF, f64) {
    let (center, side) = box_xyxy_to_cs(bbox, 1.0, 1.4);
    let crop = warp_bgr(img, crop_affine(center, side, out, 0.0), out);
    (bgr_to_gray(&crop), f64::from(out) / side)
}

/// `cv2.imread(path, IMREAD_UNCHANGED)` + `frames.py:to_bgr8`: an 8-bit BGR frame (grey images
/// replicated, alpha dropped, deeper images stretched from their min/max).
pub fn read_bgr(path: &Path) -> Result<Bgr> {
    let img = image::ImageReader::open(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?;
    let (w, h) = (img.width(), img.height());
    let rgb: Vec<u8> = match img.color().bytes_per_pixel() / img.color().channel_count().max(1) {
        1 => img.to_rgb8().into_raw(),
        _ => {
            let f = img.to_rgb32f().into_raw();
            let lo = f.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = f.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = 255.0 / (hi - lo).max(1e-6);
            f.iter()
                .map(|v| ((v - lo) * scale).clamp(0.0, 255.0) as u8)
                .collect()
        }
    };
    let mut pixels = Vec::with_capacity(rgb.len());
    for p in rgb.chunks_exact(3) {
        pixels.extend_from_slice(&[p[2], p[1], p[0]]);
    }
    Bgr::from_pixels(w, h, pixels)
}

/// `zlib.crc32`.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for b in bytes {
        crc ^= u32::from(*b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dark disk of radius 12 on a 200-grey floor, blurred by a disk of `psf`.
    fn disk_patch(psf: f64) -> GrayF {
        let n = 64u32;
        let mut img = Bgr::filled(n, n, [200, 200, 200]);
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (f64::from(x) - 31.5, f64::from(y) - 31.5);
                if dx * dx + dy * dy <= 144.0 {
                    img.set_pixel(x, y, [40, 40, 40]);
                }
            }
        }
        bgr_to_gray(&degrade_defocus(&img, psf))
    }

    #[test]
    fn edge_width_grows_with_the_blur() {
        let widths: Vec<f64> = [1.0, 2.0, 4.0, 6.0]
            .iter()
            .map(|r| edge_width(&[disk_patch(*r)], 2, 95.0).0)
            .collect();
        assert!(widths.windows(2).all(|w| w[1] > w[0]), "{widths:?}");
        // A flat patch has no edge to measure.
        let flat = GrayF::from_pixels(64, 64, vec![100.0; 64 * 64]).unwrap();
        assert_eq!(edge_width(&[flat], 2, 95.0), (0.0, 0.0));
    }

    #[test]
    fn kernels_match_opencv() {
        // cv2.getGaussianKernel(11, 1.2, cv2.CV_32F)[:3]
        let k = gaussian_kernel(11, 1.2);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let (d, n) = disk_kernel(1.0);
        assert_eq!(n, 3);
        assert_eq!(d.iter().filter(|v| **v > 0.0).count(), 5); // the plus shape of radius 1
        let (d, n) = disk_kernel(2.5);
        assert_eq!((n, d.iter().filter(|v| **v > 0.0).count()), (7, 21));
    }

    #[test]
    fn gradient_matches_numpy() {
        let g = GrayF::from_pixels(3, 2, vec![0.0, 1.0, 4.0, 2.0, 2.0, 2.0]).unwrap();
        let (gy, gx) = gradient(&g);
        assert_eq!(gy, [2.0, 1.0, -2.0, 2.0, 1.0, -2.0]);
        assert_eq!(gx, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    }
}
