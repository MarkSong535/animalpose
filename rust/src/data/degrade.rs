//! `src/animalpose/data/degrade.py`: image degradations for training on poor recordings — motion
//! and defocus blur, tiny animals, low contrast, monochrome sensors, noise and compression.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `motion_kernel` / `disk_kernel` | [`motion_kernel`] / [`disk_kernel`] |
//! | `Profile` / `DEFAULT_PROFILE` / `Profile.from_dict` / `load_profile` / `DOC_KEYS` | [`Profile`] / [`Profile::default`] / [`Profile::from_value`] / [`load_profile`] / [`DOC_KEYS`] |
//! | `degrade` | [`degrade`] |
//! | `zoom_out` | [`zoom_out`] |
//! | `edge_width` / `motion_smear` | [`edge_width`] / [`motion_smear`] |
//!
//! Strengths are relative to the animal's length in pixels, so one profile serves crops and full
//! frames; [`Profile::psf_px`] is instead a defocus radius in image pixels, the same for every animal
//! in the frame, which is what a fixed lens does.
//!
//! `// differs from Python:` the random stream is [`Rng`] (not numpy's PCG64); the draws happen in
//! the same order with the same distributions. The JPEG step uses the `image` crate's encoder (see
//! [`crate::data::raster::jpeg_roundtrip`]), and the line of [`motion_kernel`] is a Bresenham line
//! (OpenCV's `LINE_AA` falls back to a hard line on a float kernel too, so only single end pixels
//! can differ).

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::algos::image::Bgr;
use crate::data::config::Value;
use crate::data::raster::{
    dilate, draw_hard_f, erode, filter2d, gradient_magnitude, jpeg_roundtrip, median, percentile,
    resize_area, resize_linear, ImageF, Shape,
};
use crate::data::synthetic::Rng;

/// `motion_kernel(length, angle)`: a normalised line kernel of `length` px at `angle` radians.
/// Returns the kernel and its side `n` (`n x n`, odd).
pub fn motion_kernel(length: f64, angle: f64) -> (Vec<f32>, usize) {
    let n = 3usize.max((length.ceil() as usize) | 1);
    let mut k = ImageF::new(n, n, 1, 0.0);
    let c = (n as f64 - 1.0) / 2.0;
    let (dx, dy) = (
        angle.cos() * (length - 1.0) / 2.0,
        angle.sin() * (length - 1.0) / 2.0,
    );
    let a = [(c - dx).round_ties_even(), (c - dy).round_ties_even()];
    let b = [(c + dx).round_ties_even(), (c + dy).round_ties_even()];
    draw_hard_f(&mut k, &Shape::line(a, b, 1), &[1.0]);
    let sum: f32 = k.data.iter().sum();
    let sum = sum.max(1e-6);
    (k.data.iter().map(|v| v / sum).collect(), n)
}

/// `disk_kernel(radius)`: a normalised disk, `x² + y² <= radius² + 0.5`. Returns the kernel and its
/// side.
pub fn disk_kernel(radius: f64) -> (Vec<f32>, usize) {
    let r = 1i64.max(radius.ceil() as i64);
    let n = (2 * r + 1) as usize;
    let mut k = Vec::with_capacity(n * n);
    for y in -r..=r {
        for x in -r..=r {
            let inside = (x * x + y * y) as f64 <= radius * radius + 0.5;
            k.push(if inside { 1.0f32 } else { 0.0 });
        }
    }
    let sum: f32 = k.iter().sum();
    (k.iter().map(|v| v / sum).collect(), n)
}

/// `Profile`: how a camera degrades a recording; ranges are `(low, high)`, sampled per image.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    /// Defocus radius in image pixels; when set, defocus is always applied (jittered by
    /// `psf_jitter`) and `defocus_frac` is unused.
    pub psf_px: Option<f64>,
    /// Relative jitter of `psf_px`.
    pub psf_jitter: f64,
    /// Defocus radius as a fraction of the body length.
    pub defocus_frac: (f64, f64),
    /// Motion smear as a fraction of the body length (encodes the frame rate too).
    pub motion_frac: (f64, f64),
    /// The animal ends up this many pixels long, or `None` to keep the resolution.
    pub downscale_to: Option<(f64, f64)>,
    /// Probability of a monochrome sensor.
    pub mono: f64,
    /// Probability of low contrast plus a gamma change.
    pub low_contrast: f64,
    /// Sensor noise sigma in grey levels.
    pub noise: (f64, f64),
    /// JPEG quality range, or `None` for no compression.
    pub jpeg: Option<(f64, f64)>,
}

impl Default for Profile {
    /// `DEFAULT_PROFILE = Profile()`.
    fn default() -> Self {
        Self {
            psf_px: None,
            psf_jitter: 0.35,
            defocus_frac: (0.0, 0.06),
            motion_frac: (0.0, 0.2),
            downscale_to: Some((16.0, 48.0)),
            mono: 0.5,
            low_contrast: 0.5,
            noise: (2.0, 12.0),
            jpeg: Some((15.0, 70.0)),
        }
    }
}

/// `DOC_KEYS`: keys a profile YAML may carry for documentation (written by `animalpose data audit`).
pub const DOC_KEYS: [&str; 3] = ["measured", "notes", "source"];

/// The keys of `Profile.__dataclass_fields__`.
const PROFILE_KEYS: [&str; 9] = [
    "psf_px",
    "psf_jitter",
    "defocus_frac",
    "motion_frac",
    "downscale_to",
    "mono",
    "low_contrast",
    "noise",
    "jpeg",
];

impl Profile {
    /// `Profile.from_dict(d)`: `None`/null is the default profile, an unknown key is an error.
    pub fn from_value(value: Option<&Value>) -> Result<Self> {
        let Some(map) = value.and_then(Value::map) else {
            if value.is_some_and(|v| !v.is_null()) {
                bail!("a camera profile must be a mapping");
            }
            return Ok(Self::default());
        };
        let unknown: Vec<&String> = map
            .keys()
            .filter(|k| !PROFILE_KEYS.contains(&k.as_str()))
            .collect();
        if !unknown.is_empty() {
            bail!("unknown camera profile keys: {unknown:?}");
        }
        let mut p = Self::default();
        let pair = |key: &str, v: &Value| -> Result<(f64, f64)> {
            let list = v.f64_list();
            if list.len() != 2 {
                bail!("camera profile `{key}` must be a [low, high] pair");
            }
            Ok((list[0], list[1]))
        };
        let num = |key: &str, v: &Value| -> Result<f64> {
            v.as_f64()
                .with_context(|| format!("camera profile `{key}` must be a number"))
        };
        for (key, v) in map {
            match key.as_str() {
                "psf_px" => p.psf_px = if v.is_null() { None } else { Some(num(key, v)?) },
                "psf_jitter" => p.psf_jitter = num(key, v)?,
                "defocus_frac" => p.defocus_frac = pair(key, v)?,
                "motion_frac" => p.motion_frac = pair(key, v)?,
                "downscale_to" => {
                    p.downscale_to = if v.is_null() { None } else { Some(pair(key, v)?) }
                }
                "mono" => p.mono = num(key, v)?,
                "low_contrast" => p.low_contrast = num(key, v)?,
                "noise" => p.noise = pair(key, v)?,
                "jpeg" => p.jpeg = if v.is_null() { None } else { Some(pair(key, v)?) },
                _ => unreachable!("checked above"),
            }
        }
        Ok(p)
    }
}

/// `load_profile(value)`: a profile from a config mapping, a YAML path, or `None` for the default;
/// the documentation keys of [`DOC_KEYS`] are skipped.
pub fn load_profile(value: Option<&Value>) -> Result<Profile> {
    let loaded;
    let value = match value {
        None => return Ok(Profile::default()),
        Some(v) if v.is_null() => return Ok(Profile::default()),
        Some(Value::Str(path)) => {
            let text = std::fs::read_to_string(Path::new(path))
                .with_context(|| format!("cannot read camera profile {path}"))?;
            loaded = serde_yaml::from_str::<Value>(&text)
                .with_context(|| format!("cannot parse camera profile {path}"))?;
            &loaded
        }
        Some(v) => v,
    };
    let filtered = match value.map() {
        Some(map) => Value::Map(
            map.iter()
                .filter(|(k, _)| !DOC_KEYS.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        None if value.is_null() => return Ok(Profile::default()),
        None => bail!("a camera profile must be a mapping or a YAML path"),
    };
    Profile::from_value(Some(&filtered))
}

/// `degrade(image, rng, animal_px, strength, psf_px, profile)`: randomly degrades an 8-bit BGR
/// image whose animal is about `animal_px` long. Each effect is drawn independently; `strength`
/// (0–1) scales how likely and how severe they are; `psf_px` overrides the profile's defocus radius.
pub fn degrade(
    image: &Bgr,
    rng: &mut Rng,
    animal_px: f64,
    strength: f64,
    psf_px: Option<f64>,
    profile: &Profile,
) -> Result<Bgr> {
    let p = profile;
    let psf = psf_px.or(p.psf_px);
    let mut img = ImageF::from_bgr(image);
    let a = animal_px.max(8.0);
    let s = strength.clamp(0.0, 1.0);
    let (w, h) = (img.width, img.height);
    if p.motion_frac.1 > 0.0 && rng.random() < 0.6 * s {
        let length = rng.uniform(3.0, (p.motion_frac.1 * a * s).max(4.0));
        let (k, n) = motion_kernel(length, rng.uniform(0.0, std::f64::consts::PI));
        img = filter2d(&img, &k, n, n);
    }
    if let Some(psf) = psf {
        let r = psf * rng.uniform(1.0 - p.psf_jitter, 1.0 + p.psf_jitter);
        let (k, n) = disk_kernel(r.max(1.0));
        img = filter2d(&img, &k, n, n);
    } else if rng.random() < 0.4 * s {
        let (k, n) = disk_kernel(rng.uniform(1.0, (p.defocus_frac.1 * a * s).max(1.5)));
        img = filter2d(&img, &k, n, n);
    }
    if let Some(range) = p.downscale_to {
        if rng.random() < 0.5 * s {
            let target = rng.uniform(range.0, range.1);
            let f = (target / a).min(1.0);
            if f < 0.95 {
                let small = resize_area(
                    &img,
                    2usize.max((w as f64 * f) as usize),
                    2usize.max((h as f64 * f) as usize),
                );
                img = resize_linear(&small, w, h);
            }
        }
    }
    if rng.random() < p.mono * s {
        img = img.to_mono();
    }
    if rng.random() < p.low_contrast * s {
        let m = img.mean() as f32;
        let k = rng.uniform(0.35, 0.9) as f32;
        for v in img.data.iter_mut() {
            *v = m + (*v - m) * k;
        }
        let gamma = rng.uniform(0.7, 1.5) as f32;
        for v in img.data.iter_mut() {
            *v = 255.0 * (v.clamp(0.0, 255.0) / 255.0).powf(gamma);
        }
    }
    if p.noise.1 > 0.0 && rng.random() < 0.5 * s {
        let sigma = rng.uniform(p.noise.0, p.noise.1) * s;
        for v in img.data.iter_mut() {
            let n = rng.normal(0.0, 1.0);
            *v += (n * sigma * (f64::from(v.clamp(1.0, 255.0)) / 128.0).sqrt()) as f32;
        }
    }
    let mut out = img.to_bgr_trunc();
    if let Some(range) = p.jpeg {
        if rng.random() < 0.4 * s {
            let quality = rng.uniform(range.0, range.1) as u8;
            out = jpeg_roundtrip(&out, quality)?;
        }
    }
    Ok(out)
}

/// `zoom_out(image, boxes, rng, min_scale)`: shrinks the frame into a same-size canvas filled with
/// its mean colour, so animals become small. Returns the new image and boxes.
pub fn zoom_out(
    image: &Bgr,
    boxes: &[[f32; 4]],
    rng: &mut Rng,
    min_scale: f64,
) -> (Bgr, Vec<[f32; 4]>) {
    let (w, h) = (image.width as usize, image.height as usize);
    let f = rng.uniform(min_scale, 1.0);
    let nw = 2usize.max((w as f64 * f) as usize);
    let nh = 2usize.max((h as f64 * f) as usize);
    let x0 = rng.integers(0, (w as i64 - nw as i64 + 1).max(1)) as usize;
    let y0 = rng.integers(0, (h as i64 - nh as i64 + 1).max(1)) as usize;
    let src = ImageF::from_bgr(image);
    // `image.reshape(-1, C).mean(0).astype(image.dtype)`: truncated.
    let fill: Vec<u8> = src
        .channel_means()
        .iter()
        .map(|m| m.clamp(0.0, 255.0) as u8)
        .collect();
    let small = resize_area(&src, nw, nh);
    let mut pixels = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        pixels.extend_from_slice(&fill);
    }
    for y in 0..nh.min(h - y0) {
        for x in 0..nw.min(w - x0) {
            for c in 0..3 {
                pixels[((y + y0) * w + x + x0) * 3 + c] =
                    crate::data::raster::clip_u8_round(small.at(x, y, c));
            }
        }
    }
    let f32f = f as f32;
    let shifted = boxes
        .iter()
        .map(|b| {
            [
                b[0] * f32f + x0 as f32,
                b[1] * f32f + y0 as f32,
                b[2] * f32f + x0 as f32,
                b[3] * f32f + y0 as f32,
            ]
        })
        .collect();
    (
        Bgr {
            width: image.width,
            height: image.height,
            pixels,
        },
        shifted,
    )
}

/// One grey patch for [`edge_width`]: `(width, height, values)`.
pub type Patch = (usize, usize, Vec<f64>);

/// `edge_width(patches, ring, percentile)`: the transition width of the subject silhouette in each
/// grey patch (a dark subject on a brighter surround), in patch pixels, as `(median, std)` over
/// the patches that qualify; `(0, 0)` when none does.
///
/// The peak gradient across a step of contrast `C` blurred to width `w` is `C / w`, so the width
/// follows from the contrast and the 95th-percentile gradient in a thin ring around the silhouette.
pub fn edge_width(patches: &[Patch], ring: usize, pct: f64) -> (f64, f64) {
    let mut out = Vec::new();
    for (w, h, g) in patches {
        let (w, h) = (*w, *h);
        let hi = percentile(g, 95.0);
        let lo = percentile(g, 5.0);
        if hi - lo < 8.0 {
            continue;
        }
        let m: Vec<u8> = g
            .iter()
            .map(|v| u8::from(((hi - v) / (hi - lo)).clamp(0.0, 1.0) > 0.5))
            .collect();
        if m.iter().map(|v| usize::from(*v)).sum::<usize>() < 30 {
            continue;
        }
        let dil = dilate(&m, w, h, ring);
        let ero = erode(&m, w, h, ring);
        let edge: Vec<bool> = dil
            .iter()
            .zip(&ero)
            .map(|(d, e)| *d & (1 - *e) > 0)
            .collect();
        if edge.iter().filter(|v| **v).count() < 20 {
            continue;
        }
        let grad = gradient_magnitude(g, w, h);
        let ring_grad: Vec<f64> = grad
            .iter()
            .zip(&edge)
            .filter(|(_, e)| **e)
            .map(|(v, _)| *v)
            .collect();
        let peak = percentile(&ring_grad, pct);
        if peak > 1e-6 {
            out.push((hi - lo) / peak);
        }
    }
    if out.is_empty() {
        return (0.0, 0.0);
    }
    let mean = out.iter().sum::<f64>() / out.len() as f64;
    let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / out.len() as f64;
    (median(&out), var.sqrt())
}

/// `motion_smear(displacement_px, exposure_fraction)`: the expected smear in pixels, i.e. how far
/// the animal moves during one exposure.
pub fn motion_smear(displacement_px: &[f64], exposure_fraction: f64) -> f64 {
    median(displacement_px) * exposure_fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_kernel_matches_python() {
        // `disk_kernel(1.5)`: r = 2, 5x5, x² + y² <= 2.75 -> 9 ones... plus the (±1, ±1) corners.
        let (k, n) = disk_kernel(1.5);
        assert_eq!(n, 5);
        let ones = k.iter().filter(|v| **v > 0.0).count();
        assert_eq!(ones, 9);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // `disk_kernel(2.0)`: x² + y² <= 4.5 -> 13 taps.
        assert_eq!(disk_kernel(2.0).0.iter().filter(|v| **v > 0.0).count(), 13);
    }

    #[test]
    fn motion_kernel_is_a_normalised_line() {
        // `motion_kernel(5, 0)`: n = 5, a horizontal line from (0, 2) to (4, 2).
        let (k, n) = motion_kernel(5.0, 0.0);
        assert_eq!(n, 5);
        for x in 0..5 {
            assert!((k[2 * 5 + x] - 0.2).abs() < 1e-6);
        }
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // `motion_kernel(6.2, pi / 4)`: the diagonal (1, 1) .. (5, 5), 0.2 each, as cv2 draws it.
        let (k, n) = motion_kernel(6.2, std::f64::consts::FRAC_PI_4);
        assert_eq!(n, 7);
        for (i, v) in k.iter().enumerate() {
            let (x, y) = (i % 7, i / 7);
            let want = if x == y && (1..=5).contains(&x) { 0.2 } else { 0.0 };
            assert!((v - want).abs() < 1e-6, "({x}, {y}) = {v}");
        }
    }

    #[test]
    fn profile_parsing_and_unknown_keys() {
        let v: Value = serde_yaml::from_str("psf_px: 2.0\nnoise: [1, 3]\njpeg: null\nnotes: x").unwrap();
        let p = load_profile(Some(&v)).unwrap();
        assert_eq!(p.psf_px, Some(2.0));
        assert_eq!(p.noise, (1.0, 3.0));
        assert_eq!(p.jpeg, None);
        assert_eq!(p.mono, 0.5);
        let bad: Value = serde_yaml::from_str("psf: 2.0").unwrap();
        assert!(load_profile(Some(&bad)).is_err());
        assert_eq!(load_profile(None).unwrap(), Profile::default());
    }

    fn checker() -> Bgr {
        let (w, h) = (64u32, 48u32);
        let mut pixels = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v = if (x / 8 + y / 8) % 2 == 0 { 40 } else { 200 };
                pixels.extend_from_slice(&[v, v, v / 2]);
            }
        }
        Bgr {
            width: w,
            height: h,
            pixels,
        }
    }

    #[test]
    fn degrade_keeps_the_shape_and_is_deterministic() {
        let img = checker();
        let a = degrade(&img, &mut Rng::new(4), 30.0, 1.0, None, &Profile::default()).unwrap();
        let b = degrade(&img, &mut Rng::new(4), 30.0, 1.0, None, &Profile::default()).unwrap();
        assert_eq!(a, b);
        assert_eq!((a.width, a.height, a.pixels.len()), (64, 48, 64 * 48 * 3));
        // Zero strength leaves the image as it was.
        let same = degrade(&img, &mut Rng::new(4), 30.0, 0.0, None, &Profile::default()).unwrap();
        assert_eq!(same, img);
    }

    #[test]
    fn a_lens_psf_always_blurs() {
        let img = checker();
        let p = Profile {
            psf_px: Some(3.0),
            mono: 0.0,
            low_contrast: 0.0,
            noise: (0.0, 0.0),
            jpeg: None,
            downscale_to: None,
            motion_frac: (0.0, 0.0),
            ..Profile::default()
        };
        let out = degrade(&img, &mut Rng::new(1), 30.0, 1.0, None, &p).unwrap();
        // The step edges soften: grey levels between the two plateaus appear.
        let mid = out
            .pixels
            .chunks_exact(3)
            .filter(|px| px[0] > 60 && px[0] < 180)
            .count();
        assert!(mid > 500, "{mid}");
    }

    #[test]
    fn zoom_out_maps_boxes_with_the_image() {
        let img = checker();
        let (out, boxes) = zoom_out(&img, &[[8.0, 8.0, 24.0, 16.0]], &mut Rng::new(2), 0.35);
        assert_eq!((out.width, out.height), (64, 48));
        let b = boxes[0];
        assert!(b[2] - b[0] <= 16.0 + 1e-4 && b[2] - b[0] >= 16.0 * 0.35 - 1e-4);
        assert!(b[0] >= 0.0 && b[2] <= 64.0);
    }

    #[test]
    fn edge_width_grows_with_blur() {
        // A dark disk on a bright surround, blurred by two different disks.
        let (w, h) = (40usize, 40usize);
        let mut sharp = ImageF::new(w, h, 1, 200.0);
        draw_hard_f(
            &mut sharp,
            &Shape::Disk {
                center: [20.0, 20.0],
                radius: 9.0,
            },
            &[40.0],
        );
        let blur = |r: f64| {
            let (k, n) = disk_kernel(r);
            let out = filter2d(&sharp, &k, n, n);
            (w, h, out.data.iter().map(|v| f64::from(*v)).collect::<Vec<f64>>())
        };
        let (a, _) = edge_width(&[blur(1.0)], 2, 95.0);
        let (b, _) = edge_width(&[blur(3.0)], 2, 95.0);
        // `edge_width` of `data/degrade.py` on the same patches (the disk drawn with the same
        // `d <= 9.2` rule, blurred with `cv2.filter2D(sharp, -1, disk_kernel(r))`).
        assert!((a - 2.23606797749979).abs() < 1e-4, "{a}");
        assert!((b - 4.767568391833185).abs() < 1e-4, "{b}");
        let (m, sd) = edge_width(&[blur(1.0), blur(3.0)], 2, 95.0);
        assert!((m - 3.5018181846664875).abs() < 1e-4 && (sd - 1.2657502071666977).abs() < 1e-4);
        assert_eq!(edge_width(&[(4, 4, vec![100.0; 16])], 2, 95.0), (0.0, 0.0));
        assert_eq!(motion_smear(&[1.0, 2.0, 9.0], 0.5), 1.0);
    }
}
