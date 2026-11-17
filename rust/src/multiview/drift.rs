//! Camera drift: how far each camera has turned since a reference picture of its scene, measured
//! from the videos themselves, and the camera model corrected by it.
//!
//! A calibration holds while the cameras stay put. A knocked mount turns a camera by a fraction
//! of a degree, and every 3D point it helps triangulate moves with it. Each camera's reference is
//! the median background of the first session fused with the camera model, saved beside the model
//! as `<model>.references/<camera>.png`. Every later session's own median background (the animal
//! drops out of a median) is registered against it:
//!
//! * a direct, coarse-to-fine alignment of the two pictures (Gauss-Newton over the camera's
//!   rotation, through the camera's own `K` and lens distortion, so the warp is the one a turned
//!   camera produces rather than a generic image transform);
//! * the lighting's gain and offset solved alongside, so a brighter or dimmer day is not motion;
//! * robust (Huber) weights scaled by the residuals' own median deviation, so a moved object or a
//!   changed cage counts less than the walls and the desk.
//!
//! Nothing is a fixed tolerance. The rotation found is applied when it makes the two pictures
//! agree better than the uncorrected camera did, and is reported either way (`camera_drift.json`
//! and the log). The model is a rotation about the camera's centre, which is what a knock does to
//! a camera on a mount; a camera carried to another place needs a new calibration.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use super::camera::{rodrigues, rotation_to_rvec, Camera};
use super::dense::{cross3, inv3, mul3, mulv3, norm3, transpose3, Mat3, Vec3, EYE3};
use crate::algos::image::GrayF;

/// The `drift:` block of a pipeline config (`pipeline.py:DriftConfig`): how the registration
/// runs, never how far a camera may move.
#[derive(Clone, Debug, PartialEq)]
pub struct DriftCfg {
    /// Correct multiview cameras for drift.
    pub enabled: bool,
    /// The coarsest level of the image pyramid is about this many pixels across.
    pub coarsest: usize,
    /// Gauss-Newton steps per level at most.
    pub iterations: usize,
    /// Pixels per level taking part at most (an even subsample above it).
    pub samples: usize,
    /// Huber's constant, in units of the residuals' robust scale (1.345: 95% efficiency under
    /// Gaussian noise).
    pub huber: f64,
}

impl Default for DriftCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            coarsest: 64,
            iterations: 30,
            samples: 40_000,
            huber: 1.345,
        }
    }
}

impl DriftCfg {
    /// The `drift:` block of a config over the defaults above.
    pub fn from_config(cfg: &crate::data::config::Config) -> Self {
        let d = Self::default();
        Self {
            enabled: cfg.bool_at("drift.enabled", d.enabled),
            coarsest: cfg.usize_at("drift.coarsest", d.coarsest).max(8),
            iterations: cfg.usize_at("drift.iterations", d.iterations),
            samples: cfg.usize_at("drift.samples", d.samples).max(64),
            huber: cfg.f64_at("drift.huber", d.huber),
        }
    }
}

/// One level of an image pyramid: smoothed grey levels and their gradients.
struct Level {
    /// Full-resolution pixels per pixel of this level.
    scale: f64,
    w: usize,
    h: usize,
    img: Vec<f64>,
    gx: Vec<f64>,
    gy: Vec<f64>,
}

impl Level {
    fn new(img: Vec<f64>, w: usize, h: usize, scale: f64) -> Self {
        let img = blur(&img, w, h);
        let (mut gx, mut gy) = (vec![0.0; w * h], vec![0.0; w * h]);
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let i = y * w + x;
                gx[i] = 0.5 * (img[i + 1] - img[i - 1]);
                gy[i] = 0.5 * (img[i + w] - img[i - w]);
            }
        }
        Self {
            scale,
            w,
            h,
            img,
            gx,
            gy,
        }
    }

    /// A full-resolution pixel position in this level's pixels.
    fn at(&self, u: [f64; 2]) -> [f64; 2] {
        [
            (u[0] + 0.5) / self.scale - 0.5,
            (u[1] + 0.5) / self.scale - 0.5,
        ]
    }

    /// Bilinear value and gradient at a position of this level, inside a one-pixel margin.
    fn sample(&self, p: [f64; 2]) -> Option<(f64, f64, f64)> {
        let (x, y) = (p[0], p[1]);
        if !(x >= 1.0 && y >= 1.0 && x < (self.w - 2) as f64 && y < (self.h - 2) as f64) {
            return None;
        }
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let i = y0 * self.w + x0;
        let lerp = |v: &[f64]| {
            let top = v[i] * (1.0 - fx) + v[i + 1] * fx;
            let bottom = v[i + self.w] * (1.0 - fx) + v[i + self.w + 1] * fx;
            top * (1.0 - fy) + bottom * fy
        };
        Some((lerp(&self.img), lerp(&self.gx), lerp(&self.gy)))
    }
}

/// `[1 2 1] / 4` in both directions, edges clamped.
fn blur(img: &[f64], w: usize, h: usize) -> Vec<f64> {
    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let (l, r) = (x.saturating_sub(1), (x + 1).min(w - 1));
            tmp[y * w + x] = 0.25 * img[y * w + l] + 0.5 * img[y * w + x] + 0.25 * img[y * w + r];
        }
    }
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        let (u, d) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            out[y * w + x] = 0.25 * tmp[u * w + x] + 0.5 * tmp[y * w + x] + 0.25 * tmp[d * w + x];
        }
    }
    out
}

/// Levels from full resolution down to about `coarsest` pixels across.
fn pyramid(g: &GrayF, coarsest: usize) -> Vec<Level> {
    let (mut w, mut h) = (g.width as usize, g.height as usize);
    let mut img: Vec<f64> = g.pixels.iter().map(|v| f64::from(*v)).collect();
    let mut levels = vec![Level::new(img.clone(), w, h, 1.0)];
    while w / 2 >= coarsest && h / 2 >= coarsest / 2 {
        let (w2, h2) = (w / 2, h / 2);
        let mut half = vec![0.0; w2 * h2];
        for y in 0..h2 {
            for x in 0..w2 {
                let i = 2 * y * w + 2 * x;
                half[y * w2 + x] = 0.25 * (img[i] + img[i + 1] + img[i + w] + img[i + w + 1]);
            }
        }
        (img, w, h) = (half, w2, h2);
        let scale = levels.last().map_or(1.0, |l| l.scale * 2.0);
        levels.push(Level::new(img.clone(), w, h, scale));
    }
    levels
}

/// The pixel a camera-frame direction lands on, through the camera's lens.
fn project(cam: &Camera, b: Vec3) -> Option<[f64; 2]> {
    if b[2] <= 1e-9 {
        return None;
    }
    let d = cam.distort([b[0] / b[2], b[1] / b[2]]);
    let k = &cam.k;
    Some([
        d[0] * k[0][0] + d[1] * k[0][1] + k[0][2],
        d[0] * k[1][0] + d[1] * k[1][1] + k[1][2],
    ])
}

/// The pixels of `level` of the current picture that take part: each one's viewing direction
/// (through the lens) and grey level.
fn samples(cam: &Camera, level: &Level, most: usize) -> Vec<(Vec3, f64)> {
    let inner = level.w.saturating_sub(2) * level.h.saturating_sub(2);
    let step = ((inner as f64 / most as f64).sqrt().ceil() as usize).max(1);
    let mut out = Vec::new();
    for y in (1..level.h.saturating_sub(1)).step_by(step) {
        for x in (1..level.w.saturating_sub(1)).step_by(step) {
            let u = [
                (x as f64 + 0.5) * level.scale - 0.5,
                (y as f64 + 0.5) * level.scale - 0.5,
            ];
            let n = cam.undistort_point(u);
            out.push(([n[0], n[1], 1.0], level.img[y * level.w + x]));
        }
    }
    out
}

/// Least-squares `cur ≈ a·reference + c` with weights.
fn photometric(pairs: &[(f64, f64)], weights: &[f64]) -> (f64, f64) {
    let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ((x, y), w) in pairs.iter().zip(weights) {
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    let det = sw * sxx - sx * sx;
    if det.abs() < 1e-12 || sw <= 0.0 {
        return (1.0, 0.0);
    }
    let a = (sw * sxy - sx * sy) / det;
    (a, (sy - a * sx) / sw)
}

/// Huber weights for residuals `r`, scaled by their median absolute deviation.
fn huber(r: &[f64], constant: f64) -> Vec<f64> {
    let mut abs: Vec<f64> = r.iter().map(|v| v.abs()).collect();
    abs.sort_by(f64::total_cmp);
    let sigma = 1.4826 * abs.get(abs.len() / 2).copied().unwrap_or(0.0);
    if sigma <= 1e-12 {
        return vec![1.0; r.len()];
    }
    let k = constant * sigma;
    r.iter()
        .map(|v| if v.abs() <= k { 1.0 } else { k / v.abs() })
        .collect()
}

/// Pearson correlation between the current picture and the reference seen through `rd`, over the
/// pixels both show.
fn agreement(cam: &Camera, reference: &Level, current: &[(Vec3, f64)], rd: &Mat3) -> Option<f64> {
    let rt = transpose3(rd);
    let pairs: Vec<(f64, f64)> = current
        .iter()
        .filter_map(|(b, cur)| {
            let u = project(cam, mulv3(&rt, b))?;
            Some((reference.sample(reference.at(u))?.0, *cur))
        })
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 {
        return None;
    }
    let (mx, my) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / n,
        pairs.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in &pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    (sxx > 0.0 && syy > 0.0).then(|| sxy / (sxx * syy).sqrt())
}

/// How a camera turned between its reference picture and now.
#[derive(Clone, Debug)]
pub struct Registration {
    /// `b_now = rotation · b_reference` for a direction in camera coordinates.
    pub rotation: Mat3,
    /// The turn, in degrees.
    pub degrees: f64,
    /// Correlation of the two pictures before and after the turn is taken out.
    pub before: f64,
    pub after: f64,
}

/// Registers `current` against `reference`, both pictures of `cam`'s scene at its resolution.
/// `None` when the two pictures share too few pixels to compare.
pub fn register(
    cam: &Camera,
    reference: &GrayF,
    current: &GrayF,
    cfg: &DriftCfg,
) -> Option<Registration> {
    if !(reference.width == current.width && reference.height == current.height) {
        return None;
    }
    let (refs, curs) = (
        pyramid(reference, cfg.coarsest),
        pyramid(current, cfg.coarsest),
    );
    let mut rd = EYE3;
    for (r, c) in refs.iter().zip(&curs).rev() {
        let pts = samples(cam, c, cfg.samples);
        for _ in 0..cfg.iterations {
            let rt = transpose3(&rd);
            let mut pairs = Vec::with_capacity(pts.len());
            let mut grads = Vec::with_capacity(pts.len());
            for (b, cur) in &pts {
                let b0 = mulv3(&rt, b);
                let Some(u) = project(cam, b0) else { continue };
                let Some((iref, gx, gy)) = r.sample(r.at(u)) else {
                    continue;
                };
                // d(pixel)/d(rotation) by central differences of the lens projection; a turn by
                // δ moves the reference direction by δ × b0.
                let eps = 1e-6;
                let mut g = [0.0; 3];
                for (k, gk) in g.iter_mut().enumerate() {
                    let mut e = [0.0; 3];
                    e[k] = 1.0;
                    let d = cross3(&e, &b0);
                    let plus = [b0[0] + eps * d[0], b0[1] + eps * d[1], b0[2] + eps * d[2]];
                    let minus = [b0[0] - eps * d[0], b0[1] - eps * d[1], b0[2] - eps * d[2]];
                    let (Some(p), Some(m)) = (project(cam, plus), project(cam, minus)) else {
                        continue;
                    };
                    let du = [(p[0] - m[0]) / (2.0 * eps), (p[1] - m[1]) / (2.0 * eps)];
                    *gk = (gx * du[0] + gy * du[1]) / r.scale;
                }
                pairs.push((iref, *cur));
                grads.push(g);
            }
            if pairs.len() < 64 {
                return None;
            }
            let (a, c0) = photometric(&pairs, &vec![1.0; pairs.len()]);
            let resid: Vec<f64> = pairs.iter().map(|(x, y)| y - a * x - c0).collect();
            let weights = huber(&resid, cfg.huber);
            let (a, c0) = photometric(&pairs, &weights);
            let mut h = [[0.0; 3]; 3];
            let mut rhs = [0.0; 3];
            for (((x, y), g), w) in pairs.iter().zip(&grads).zip(&weights) {
                let res = y - a * x - c0;
                let j = [a * g[0], a * g[1], a * g[2]];
                for p in 0..3 {
                    rhs[p] += w * j[p] * res;
                    for q in 0..3 {
                        h[p][q] += w * j[p] * j[q];
                    }
                }
            }
            let Some(hi) = inv3(&h) else { break };
            let delta = mulv3(&hi, &rhs);
            if !delta.iter().all(|v| v.is_finite()) {
                break;
            }
            rd = mul3(&rd, &rodrigues(&[-delta[0], -delta[1], -delta[2]]));
            if norm3(&delta) < 1e-7 {
                break;
            }
        }
    }
    let fine = samples(cam, &curs[0], cfg.samples);
    let before = agreement(cam, &refs[0], &fine, &EYE3)?;
    let after = agreement(cam, &refs[0], &fine, &rd)?;
    Some(Registration {
        degrees: norm3(&rotation_to_rvec(&rd)).to_degrees(),
        rotation: rd,
        before,
        after,
    })
}

/// `cam` after turning by `rotation` about its centre: `R' = rotation · R`, `t' = rotation · t`.
pub fn turned(cam: &Camera, rotation: &Mat3) -> Camera {
    Camera {
        r: mul3(rotation, &cam.r),
        t: mulv3(rotation, &cam.t),
        ..cam.clone()
    }
}

/// The reference pictures of a camera model, one PNG per camera beside the model:
/// `<model>.references/<camera>.png`.
#[derive(Clone, Debug, PartialEq)]
pub struct References {
    pub dir: PathBuf,
}

impl References {
    /// The references of the camera model saved at `model`.
    pub fn beside(model: &Path) -> Self {
        Self {
            dir: model.with_extension("references"),
        }
    }

    fn path(&self, camera: &str) -> PathBuf {
        let name: String = camera
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        self.dir.join(format!("{name}.png"))
    }

    pub fn load(&self, camera: &str) -> Result<Option<GrayF>> {
        let path = self.path(camera);
        if !path.is_file() {
            return Ok(None);
        }
        let img = image::open(&path)
            .with_context(|| format!("cannot read {}", path.display()))?
            .to_luma8();
        let (width, height) = img.dimensions();
        Ok(Some(GrayF {
            width,
            height,
            pixels: img.into_raw().into_iter().map(f32::from).collect(),
        }))
    }

    pub fn save(&self, camera: &str, picture: &GrayF) -> Result<()> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("cannot create {}", self.dir.display()))?;
        let path = self.path(camera);
        let pixels = picture.as_u8().pixels;
        image::GrayImage::from_raw(picture.width, picture.height, pixels)
            .context("reference picture size")?
            .save(&path)
            .with_context(|| format!("cannot write {}", path.display()))
    }
}

/// What happened to one camera.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Entry {
    pub camera: String,
    /// `corrected`, `unchanged`, `reference saved` or `skipped`.
    pub status: String,
    /// The turn found since the reference, in degrees.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrees: Option<f64>,
    /// Correlation with the reference before and after correcting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<f64>,
    pub reference: PathBuf,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub note: String,
}

impl Entry {
    /// One line for the log.
    pub fn line(&self) -> String {
        let measured = match (self.degrees, self.before, self.after) {
            (Some(d), Some(b), Some(a)) => {
                format!(": turned {d:.3}° since its reference (agreement {b:.3} → {a:.3})")
            }
            _ => String::new(),
        };
        let note = if self.note.is_empty() {
            String::new()
        } else {
            format!(" ({})", self.note)
        };
        format!("{} {}{measured}{note}", self.camera, self.status)
    }
}

/// Corrects each camera for the turn its session's `backgrounds` show against its reference.
/// A camera without a reference keeps its calibration and this session's background becomes its
/// reference.
pub fn correct(
    cameras: &[Camera],
    backgrounds: &[GrayF],
    references: &References,
    cfg: &DriftCfg,
) -> Result<(Vec<Camera>, Vec<Entry>)> {
    let mut out = Vec::with_capacity(cameras.len());
    let mut entries = Vec::with_capacity(cameras.len());
    for (cam, bg) in cameras.iter().zip(backgrounds) {
        let mut entry = Entry {
            camera: cam.name.clone(),
            status: "unchanged".into(),
            degrees: None,
            before: None,
            after: None,
            reference: references.path(&cam.name),
            note: String::new(),
        };
        let size = (bg.width as usize, bg.height as usize);
        let mut used = cam.clone();
        if size != cam.size {
            entry.status = "skipped".into();
            entry.note = format!(
                "the video is {}×{}, the camera model {}×{}",
                size.0, size.1, cam.size.0, cam.size.1
            );
        } else {
            match references.load(&cam.name)? {
                None => {
                    references.save(&cam.name, bg)?;
                    entry.status = "reference saved".into();
                    entry.note = "later sessions are measured against this one".into();
                }
                Some(reference) if (reference.width, reference.height) != (bg.width, bg.height) => {
                    entry.status = "skipped".into();
                    entry.note = format!(
                        "the reference is {}×{}; delete it to take a new one",
                        reference.width, reference.height
                    );
                }
                Some(reference) => match register(cam, &reference, bg, cfg) {
                    None => entry.note = "the pictures share too little to compare".into(),
                    Some(reg) => {
                        entry.degrees = Some(reg.degrees);
                        entry.before = Some(reg.before);
                        entry.after = Some(reg.after);
                        if reg.after > reg.before {
                            used = turned(cam, &reg.rotation);
                            entry.status = "corrected".into();
                        } else {
                            entry.note = "correcting did not improve the agreement".into();
                        }
                    }
                },
            }
        }
        out.push(used);
        entries.push(entry);
    }
    Ok((out, entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A camera looking straight down from 2 units above a textured floor.
    fn camera() -> Camera {
        let k = [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]];
        let r = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        Camera::new(k, r, [0.0, 0.0, 2.0], &[-0.15, 0.01], (320, 240), "top")
    }

    /// The floor's texture at a world point: smooth blobs and a few edges, like a desk and walls.
    fn floor(x: f64, y: f64) -> f64 {
        let blobs = 60.0 * (3.1 * x).sin() * (2.3 * y).cos() + 40.0 * (7.0 * x + 5.0 * y).sin();
        let edge = if (x - 0.3).abs() < 0.2 && (y + 0.1).abs() < 0.3 {
            50.0
        } else {
            0.0
        };
        120.0 + blobs + edge
    }

    /// What `cam` sees of the floor, lit with `gain` and `offset`.
    fn picture(cam: &Camera, gain: f64, offset: f64) -> GrayF {
        let (w, h) = cam.size;
        let rt = transpose3(&cam.r);
        let c = cam.center();
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let n = cam.undistort_point([x as f64, y as f64]);
                let d = mulv3(&rt, &[n[0], n[1], 1.0]);
                let s = -c[2] / d[2];
                let v = floor(c[0] + s * d[0], c[1] + s * d[1]);
                pixels.push((gain * v + offset).clamp(0.0, 255.0) as f32);
            }
        }
        GrayF {
            width: w as u32,
            height: h as u32,
            pixels,
        }
    }

    #[test]
    fn a_turned_camera_is_measured_and_corrected() {
        let cam = camera();
        let reference = picture(&cam, 1.0, 0.0);
        let knock = rodrigues(&[0.004, -0.007, 0.010]);
        let moved = turned(&cam, &knock);
        // A different day: dimmer, and the camera turned by about 0.7°.
        let now = picture(&moved, 0.85, 12.0);
        let reg = register(&cam, &reference, &now, &DriftCfg::default()).unwrap();
        let error = norm3(&rotation_to_rvec(&mul3(&transpose3(&knock), &reg.rotation)));
        assert!(
            error.to_degrees() < 0.02,
            "off by {:.4}°",
            error.to_degrees()
        );
        assert!((reg.degrees - 0.7).abs() < 0.05, "{}", reg.degrees);
        assert!(reg.after > reg.before);
        // The corrected camera puts a floor point where the turned camera sees it.
        let fixed = turned(&cam, &reg.rotation);
        let p = [0.2, -0.15, 0.0];
        let (a, b) = (fixed.project(&p), moved.project(&p));
        assert!((a[0] - b[0]).hypot(a[1] - b[1]) < 0.2);
    }

    #[test]
    fn a_changed_scene_does_not_pull_the_camera() {
        let cam = camera();
        let reference = picture(&cam, 1.0, 0.0);
        let knock = rodrigues(&[-0.006, 0.003, 0.005]);
        let mut now = picture(&turned(&cam, &knock), 1.0, 0.0);
        // Something new in a sixth of the picture: a cup put down, a cage moved.
        let w = now.width as usize;
        for y in 30..110 {
            for x in 180..260 {
                now.pixels[y * w + x] = if (x / 8 + y / 8) % 2 == 0 {
                    250.0
                } else {
                    10.0
                };
            }
        }
        let reg = register(&cam, &reference, &now, &DriftCfg::default()).unwrap();
        let error = norm3(&rotation_to_rvec(&mul3(&transpose3(&knock), &reg.rotation)));
        assert!(
            error.to_degrees() < 0.05,
            "off by {:.4}°",
            error.to_degrees()
        );
        assert!(reg.after > reg.before);
    }

    #[test]
    fn a_camera_that_stayed_put_stays_put() {
        let cam = camera();
        let reference = picture(&cam, 1.0, 0.0);
        let reg = register(
            &cam,
            &reference,
            &picture(&cam, 1.1, -5.0),
            &DriftCfg::default(),
        )
        .unwrap();
        assert!(reg.degrees < 0.01, "{}", reg.degrees);
    }

    #[test]
    fn the_first_session_becomes_the_reference() {
        let dir = std::env::temp_dir().join(format!("drift-{}", std::process::id()));
        let refs = References::beside(&dir.join("rig.json"));
        let cam = camera();
        let bg = picture(&cam, 1.0, 0.0);
        let (cams, entries) = correct(
            std::slice::from_ref(&cam),
            std::slice::from_ref(&bg),
            &refs,
            &DriftCfg::default(),
        )
        .unwrap();
        assert_eq!(entries[0].status, "reference saved");
        assert_eq!(cams[0], cam);
        assert!(refs.dir.join("top.png").is_file());

        let knock = rodrigues(&[0.0, 0.006, -0.004]);
        let now = picture(&turned(&cam, &knock), 1.0, 0.0);
        let (cams, entries) = correct(
            std::slice::from_ref(&cam),
            &[now],
            &refs,
            &DriftCfg::default(),
        )
        .unwrap();
        assert_eq!(entries[0].status, "corrected", "{:?}", entries[0]);
        let error = norm3(&rotation_to_rvec(&mul3(
            &transpose3(&knock),
            &mul3(&cams[0].r, &transpose3(&cam.r)),
        )));
        assert!(error.to_degrees() < 0.05, "{}", error.to_degrees());
        assert!(entries[0].line().starts_with("top corrected: turned"));

        let small = GrayF {
            width: 160,
            height: 120,
            pixels: vec![0.0; 160 * 120],
        };
        let (cams, entries) = correct(
            std::slice::from_ref(&cam),
            &[small],
            &refs,
            &DriftCfg::default(),
        )
        .unwrap();
        assert_eq!(entries[0].status, "skipped");
        assert_eq!(cams[0], cam);
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Measures real recordings: `DRIFT_VIDEOS` (`:`-separated, the first is the reference) with a
/// plain camera (`f` = the width, centred, no distortion) when no model is at hand.
#[cfg(test)]
mod real {
    use super::*;

    #[test]
    #[ignore = "needs DRIFT_VIDEOS"]
    fn drift_over_a_day() {
        let videos = std::env::var("DRIFT_VIDEOS").unwrap();
        let videos: Vec<&str> = videos.split(':').collect();
        let bg = |v: &str| {
            let img = crate::assay::motion::background(Path::new(v), 64).unwrap();
            GrayF {
                width: img.width,
                height: img.height,
                pixels: img.pixels.iter().map(|p| f32::from(*p)).collect(),
            }
        };
        let reference = bg(videos[0]);
        let (w, h) = (f64::from(reference.width), f64::from(reference.height));
        let cam = Camera::new(
            [[w, 0.0, w / 2.0], [0.0, w, h / 2.0], [0.0, 0.0, 1.0]],
            EYE3,
            [0.0; 3],
            &[],
            (reference.width as usize, reference.height as usize),
            "cam",
        );
        for v in &videos[1..] {
            let started = std::time::Instant::now();
            let reg = register(&cam, &reference, &bg(v), &DriftCfg::default()).unwrap();
            let r = rotation_to_rvec(&reg.rotation);
            println!(
                "{}: {:.3}° (pan {:.3}, tilt {:.3}, roll {:.3}) agreement {:.4} -> {:.4} in {:.2} s",
                Path::new(v).file_name().unwrap().to_string_lossy(),
                reg.degrees,
                r[1].to_degrees(),
                r[0].to_degrees(),
                r[2].to_degrees(),
                reg.before,
                reg.after,
                started.elapsed().as_secs_f64()
            );
        }
    }
}
