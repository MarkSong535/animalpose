//! Pinhole camera with Brown-Conrady distortion (OpenCV convention): the port of
//! `src/animalpose/multiview/camera.py`.
//!
//! The JSON written by [`Camera::to_json`] / [`dumps_cameras`] is the `Camera.to_dict()` layout
//! (`name`, `K`, `R`, `t`, `dist`, `size`), formatted like `json.dumps(..., indent=2)`, so a
//! `cams.json` written by either implementation is read by the other.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use super::dense::{cross3, dot3, inv3, mul3, mulv3, norm3, skew, transpose3, Mat3, Vec3, EYE3};

/// `rodrigues`: axis-angle `(3,)` -> rotation matrix.
pub fn rodrigues(rvec: &Vec3) -> Mat3 {
    let theta = norm3(rvec);
    if theta < 1e-12 {
        return EYE3;
    }
    let k = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta];
    let kx = skew(&k);
    let kx2 = mul3(&kx, &kx);
    let (s, c) = (theta.sin(), 1.0 - theta.cos());
    let mut out = EYE3;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += s * kx[i][j] + c * kx2[i][j];
        }
    }
    out
}

/// `rotation_to_rvec`: rotation matrix -> axis-angle `(3,)`.
pub fn rotation_to_rvec(r: &Mat3) -> Vec3 {
    let cos = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos.acos();
    if theta < 1e-12 {
        return [0.0; 3];
    }
    if std::f64::consts::PI - theta < 1e-6 {
        // Near 180 degrees: recover the axis from the symmetric part.
        let mut b = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i][j] = (r[i][j] + EYE3[i][j]) / 2.0;
            }
        }
        let axis: Vec<f64> = (0..3).map(|i| b[i][i].max(0.0).sqrt()).collect();
        // `np.argmax`: the first maximum.
        let mut i = 0;
        for j in 1..3 {
            if axis[j] > axis[i] {
                i = j;
            }
        }
        let d = b[i][i].sqrt();
        return [
            b[0][i] / d * theta,
            b[1][i] / d * theta,
            b[2][i] / d * theta,
        ];
    }
    let w = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let f = theta / (2.0 * theta.sin());
    [w[0] * f, w[1] * f, w[2] * f]
}

/// `Camera`: world -> camera `x_cam = R @ X + t`; image `K @ distort(x_cam / z)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// Intrinsics `K`.
    pub k: Mat3,
    /// World -> camera rotation `R`.
    pub r: Mat3,
    /// World -> camera translation `t`.
    pub t: Vec3,
    /// `(k1, k2, p1, p2, k3)`, zero-padded as `__post_init__` does.
    pub dist: [f64; 5],
    /// `(width, height)`.
    pub size: (usize, usize),
    /// `Camera.name`.
    pub name: String,
}

impl Camera {
    /// `Camera(K, R, t, dist, size, name)`: `dist` may hold fewer than five coefficients.
    pub fn new(k: Mat3, r: Mat3, t: Vec3, dist: &[f64], size: (usize, usize), name: &str) -> Self {
        let mut d = [0.0; 5];
        for (slot, v) in d.iter_mut().zip(dist) {
            *slot = *v;
        }
        Self {
            k,
            r,
            t,
            dist: d,
            size,
            name: name.to_string(),
        }
    }

    /// `Camera.P = K @ [R | t]`.
    pub fn p(&self) -> [[f64; 4]; 3] {
        let mut rt = [[0.0; 4]; 3];
        for i in 0..3 {
            rt[i][..3].copy_from_slice(&self.r[i]);
            rt[i][3] = self.t[i];
        }
        let mut out = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..4 {
                out[i][j] = (0..3).map(|k| self.k[i][k] * rt[k][j]).sum();
            }
        }
        out
    }

    /// `Camera.center = -R.T @ t`.
    pub fn center(&self) -> Vec3 {
        let c = mulv3(&transpose3(&self.r), &self.t);
        [-c[0], -c[1], -c[2]]
    }

    /// `Camera.to_camera`: `X @ R.T + t` for one point.
    pub fn to_camera(&self, x: &Vec3) -> Vec3 {
        let c = mulv3(&self.r, x);
        [c[0] + self.t[0], c[1] + self.t[1], c[2] + self.t[2]]
    }

    /// `Camera._distort` for one normalised point.
    pub fn distort(&self, xn: [f64; 2]) -> [f64; 2] {
        let [k1, k2, p1, p2, k3] = self.dist;
        let (x, y) = (xn[0], xn[1]);
        let r2 = x * x + y * y;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        [
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        ]
    }

    /// `Camera.project(X, return_depth=True)` for one point: pixel and camera depth.
    pub fn project_with_depth(&self, x: &Vec3) -> ([f64; 2], f64) {
        let xc = self.to_camera(x);
        let z = xc[2];
        let zz = if z.abs() < 1e-9 { 1e-9 } else { z };
        let xd = self.distort([xc[0] / zz, xc[1] / zz]);
        let k = &self.k;
        (
            [
                xd[0] * k[0][0] + xd[1] * k[0][1] + k[0][2],
                xd[0] * k[1][0] + xd[1] * k[1][1] + k[1][2],
            ],
            z,
        )
    }

    /// `Camera.project` for one point.
    pub fn project(&self, x: &Vec3) -> [f64; 2] {
        self.project_with_depth(x).0
    }

    /// `Camera.project` over many points.
    pub fn project_many(&self, xs: &[Vec3]) -> Vec<[f64; 2]> {
        xs.iter().map(|x| self.project(x)).collect()
    }

    /// `Camera.undistort_points(uv, iters)` for one pixel: normalised, undistorted coordinates.
    pub fn undistort_point_iters(&self, uv: [f64; 2], iters: usize) -> [f64; 2] {
        let k2 = [[self.k[0][0], self.k[0][1]], [self.k[1][0], self.k[1][1]]];
        let det = k2[0][0] * k2[1][1] - k2[0][1] * k2[1][0];
        let inv = [
            [k2[1][1] / det, -k2[0][1] / det],
            [-k2[1][0] / det, k2[0][0] / det],
        ];
        let (u, v) = (uv[0] - self.k[0][2], uv[1] - self.k[1][2]);
        let xd = [inv[0][0] * u + inv[0][1] * v, inv[1][0] * u + inv[1][1] * v];
        if self.dist.iter().all(|d| *d == 0.0) {
            return xd;
        }
        let [k1, k2c, p1, p2, k3] = self.dist;
        let mut xn = xd;
        for _ in 0..iters {
            // Fixed-point inversion, as in cv2.undistortPoints.
            let (x, y) = (xn[0], xn[1]);
            let r2 = x * x + y * y;
            let radial = 1.0 + k1 * r2 + k2c * r2 * r2 + k3 * r2 * r2 * r2;
            let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
            let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
            xn = [(xd[0] - dx) / radial, (xd[1] - dy) / radial];
        }
        xn
    }

    /// `Camera.undistort_points(uv)` (10 iterations) for one pixel.
    pub fn undistort_point(&self, uv: [f64; 2]) -> [f64; 2] {
        self.undistort_point_iters(uv, 10)
    }

    /// `Camera.undistort_points` over many pixels.
    pub fn undistort_points(&self, uv: &[[f64; 2]]) -> Vec<[f64; 2]> {
        uv.iter().map(|p| self.undistort_point(*p)).collect()
    }

    /// `undistort_points(uv) @ K[:2, :2].T + K[:2, 2]`: back to (undistorted) pixels.
    pub fn undistort_pixel(&self, uv: [f64; 2]) -> [f64; 2] {
        let xn = self.undistort_point(uv);
        let k = &self.k;
        [
            xn[0] * k[0][0] + xn[1] * k[0][1] + k[0][2],
            xn[0] * k[1][0] + xn[1] * k[1][1] + k[1][2],
        ]
    }

    /// `Camera.in_image(uv, margin)` for one pixel.
    pub fn in_image(&self, uv: [f64; 2], margin: f64) -> bool {
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        uv[0] >= -margin && uv[0] < w + margin && uv[1] >= -margin && uv[1] < h + margin
    }

    /// `Camera.look_at(eye, target, up, focal, size, dist, name)` (world z-up, image y-down).
    pub fn look_at(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        focal: f64,
        size: (usize, usize),
        dist: Option<&[f64]>,
        name: &str,
    ) -> Self {
        let mut fwd = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let n = norm3(&fwd);
        fwd = [fwd[0] / n, fwd[1] / n, fwd[2] / n];
        let un = norm3(&up);
        let up = if dot3(&fwd, &[up[0] / un, up[1] / un, up[2] / un]).abs() > 0.999 {
            [0.0, 1.0, 0.0] // looking straight down: pick image "up" = +y
        } else {
            up
        };
        let mut right = cross3(&fwd, &up);
        let n = norm3(&right);
        right = [right[0] / n, right[1] / n, right[2] / n];
        let down = cross3(&fwd, &right);
        let r = [right, down, fwd];
        let re = mulv3(&r, &eye);
        let t = [-re[0], -re[1], -re[2]];
        let (w, h) = size;
        let k = [
            [focal, 0.0, (w as f64 - 1.0) / 2.0],
            [0.0, focal, (h as f64 - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ];
        Self::new(k, r, t, dist.unwrap_or(&[0.0; 5]), size, name)
    }

    /// `Camera.to_dict()`.
    pub fn to_py(&self) -> PyJson {
        let mat = |m: &Mat3| {
            PyJson::List(
                m.iter()
                    .map(|row| PyJson::List(row.iter().map(|v| PyJson::Float(*v)).collect()))
                    .collect(),
            )
        };
        let vec = |v: &[f64]| PyJson::List(v.iter().map(|x| PyJson::Float(*x)).collect());
        PyJson::Dict(vec![
            ("name".into(), PyJson::Str(self.name.clone())),
            ("K".into(), mat(&self.k)),
            ("R".into(), mat(&self.r)),
            ("t".into(), vec(&self.t)),
            ("dist".into(), vec(&self.dist)),
            (
                "size".into(),
                PyJson::List(vec![
                    PyJson::Int(self.size.0 as i64),
                    PyJson::Int(self.size.1 as i64),
                ]),
            ),
        ])
    }

    /// `Camera.to_dict()` as a `serde_json` value.
    pub fn to_json(&self) -> Value {
        serde_json::from_str(&self.to_py().dumps(None)).expect("our own JSON parses")
    }

    /// `Camera.from_dict(d)`.
    pub fn from_json(d: &Value) -> Result<Self> {
        let mat = |key: &str| -> Result<Mat3> {
            let rows = d
                .get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("camera entry has no {key} matrix"))?;
            ensure!(rows.len() == 3, "camera {key} must be 3x3");
            let mut out = [[0.0; 3]; 3];
            for (i, row) in rows.iter().enumerate() {
                let row = numbers(row).with_context(|| format!("camera {key} row {i}"))?;
                ensure!(row.len() == 3, "camera {key} must be 3x3");
                out[i].copy_from_slice(&row);
            }
            Ok(out)
        };
        let k = mat("K")?;
        let r = mat("R")?;
        let t = numbers(d.get("t").context("camera entry has no t")?)?;
        ensure!(
            t.len() == 3,
            "camera t must have 3 entries, got {}",
            t.len()
        );
        let dist = match d.get("dist") {
            Some(v) => numbers(v)?,
            None => vec![0.0; 5],
        };
        ensure!(
            dist.len() <= 5,
            "camera dist holds at most 5 coefficients, got {}",
            dist.len()
        );
        let size = match d.get("size") {
            Some(v) => {
                let s = numbers(v)?;
                ensure!(s.len() == 2, "camera size must be [width, height]");
                (s[0] as usize, s[1] as usize)
            }
            None => (640, 480),
        };
        let name = d.get("name").and_then(Value::as_str).unwrap_or("cam");
        Ok(Self::new(k, r, [t[0], t[1], t[2]], &dist, size, name))
    }
}

impl From<&crate::data::synthetic::Camera> for Camera {
    /// The simulator's distortion-free camera (`data/synthetic.py:camera_from_view` returns a full
    /// `Camera` in Python).
    fn from(c: &crate::data::synthetic::Camera) -> Self {
        let k = [[c.focal, 0.0, c.cx], [0.0, c.focal, c.cy], [0.0, 0.0, 1.0]];
        Self::new(k, c.r, c.t, &[0.0; 5], (c.width, c.height), &c.name)
    }
}

/// `essential_matrix`: `E` with `x_b^T E x_a = 0` for undistorted normalised coordinates.
pub fn essential_matrix(cam_a: &Camera, cam_b: &Camera) -> Mat3 {
    let r = mul3(&cam_b.r, &transpose3(&cam_a.r));
    let ra = mulv3(&r, &cam_a.t);
    let t = [cam_b.t[0] - ra[0], cam_b.t[1] - ra[1], cam_b.t[2] - ra[2]];
    mul3(&skew(&t), &r)
}

/// `K^-1`, which every caller here can rely on (a camera matrix is invertible).
pub fn k_inv(cam: &Camera) -> Mat3 {
    inv3(&cam.k).unwrap_or([[f64::NAN; 3]; 3])
}

fn numbers(v: &Value) -> Result<Vec<f64>> {
    let Some(items) = v.as_array() else {
        bail!("expected a list of numbers, got {v}");
    };
    items
        .iter()
        .map(|x| {
            x.as_f64()
                .with_context(|| format!("expected a number, got {x}"))
        })
        .collect()
}

/// `[Camera.from_dict(d) for d in json.loads(Path(path).read_text())]`.
pub fn load_cameras(path: impl AsRef<Path>) -> Result<Vec<Camera>> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))?;
    let Some(items) = value.as_array() else {
        bail!(
            "{} must hold a JSON list of Camera.to_dict() entries",
            path.display()
        );
    };
    items
        .iter()
        .enumerate()
        .map(|(i, d)| Camera::from_json(d).with_context(|| format!("{} entry {i}", path.display())))
        .collect()
}

/// `json.dumps([c.to_dict() for c in cams], indent=2)`.
pub fn dumps_cameras(cams: &[Camera]) -> String {
    PyJson::List(cams.iter().map(Camera::to_py).collect()).dumps(Some(2))
}

/// A JSON value with insertion-ordered keys, printed the way Python's `json.dumps` prints it
/// (float `repr`, `NaN` / `Infinity`, `ensure_ascii`).
#[derive(Clone, Debug, PartialEq)]
pub enum PyJson {
    /// A float, printed with `repr`.
    Float(f64),
    /// An integer.
    Int(i64),
    /// A string.
    Str(String),
    /// A list.
    List(Vec<PyJson>),
    /// A dict, in insertion order.
    Dict(Vec<(String, PyJson)>),
}

impl PyJson {
    /// `json.dumps(self, indent=indent)`.
    pub fn dumps(&self, indent: Option<usize>) -> String {
        let mut out = String::new();
        self.write(&mut out, indent, 0);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize) {
        let newline = |out: &mut String, level: usize| {
            if let Some(n) = indent {
                out.push('\n');
                out.push_str(&" ".repeat(n * level));
            }
        };
        let sep = if indent.is_some() { "," } else { ", " };
        match self {
            PyJson::Float(v) => out.push_str(&py_float_repr(*v, true)),
            PyJson::Int(v) => out.push_str(&v.to_string()),
            PyJson::Str(s) => out.push_str(&py_string(s)),
            PyJson::List(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    newline(out, level + 1);
                    item.write(out, indent, level + 1);
                }
                newline(out, level);
                out.push(']');
            }
            PyJson::Dict(items) => {
                if items.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (key, item)) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    newline(out, level + 1);
                    out.push_str(&py_string(key));
                    out.push_str(": ");
                    item.write(out, indent, level + 1);
                }
                newline(out, level);
                out.push('}');
            }
        }
    }
}

/// `json.dumps(s)` with `ensure_ascii=True`.
fn py_string(s: &str) -> String {
    let quoted = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string());
    let mut out = String::with_capacity(quoted.len());
    for ch in quoted.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut buf = [0u16; 2];
            for unit in ch.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// Python's `repr(float)`: the shortest round-trip digits, positional between `1e-4` and `1e16`,
/// scientific (`1e-05`, `1.5e+16`) outside. With `json`, the special values are spelt as
/// `json.dumps` spells them (`NaN`, `Infinity`).
pub fn py_float_repr(v: f64, json: bool) -> String {
    if v.is_nan() {
        return if json { "NaN" } else { "nan" }.to_string();
    }
    if v.is_infinite() {
        let s = if json { "Infinity" } else { "inf" };
        return if v > 0.0 {
            s.to_string()
        } else {
            format!("-{s}")
        };
    }
    let sci = format!("{v:e}"); // e.g. "-1.2345e-5", "5e0"
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{sign}{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{sign}{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let frac = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}{frac}e{esign}{:02}", &digits[..1], exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        for (v, want) in [
            (1.0, "1.0"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (319.5, "319.5"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (-1.25e-7, "-1.25e-07"),
            (1e16, "1e+16"),
            (123456789012345.0, "123456789012345.0"),
            (1.5e300, "1.5e+300"),
            (0.30000000000000004, "0.30000000000000004"),
        ] {
            assert_eq!(py_float_repr(v, true), want, "{v:e}");
        }
        assert_eq!(py_float_repr(f64::NAN, true), "NaN");
        assert_eq!(py_float_repr(f64::NEG_INFINITY, true), "-Infinity");
    }

    #[test]
    fn rodrigues_round_trips_including_half_turns() {
        assert_eq!(rodrigues(&[0.0; 3]), EYE3);
        assert_eq!(rotation_to_rvec(&EYE3), [0.0; 3]);
        let g = [0.3, -0.7, 1.1];
        let back = rotation_to_rvec(&rodrigues(&g));
        for i in 0..3 {
            assert!((back[i] - g[i]).abs() < 1e-9);
        }
        for axis in [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.0, 0.8],
            [1.0, 1.0, 1.0],
        ] {
            let n = norm3(&axis);
            let rvec = [
                axis[0] / n * std::f64::consts::PI,
                axis[1] / n * std::f64::consts::PI,
                axis[2] / n * std::f64::consts::PI,
            ];
            let rot = rodrigues(&rvec);
            let back = rotation_to_rvec(&rot);
            assert!((norm3(&back) - std::f64::consts::PI).abs() < 1e-6);
            let again = rodrigues(&back);
            for i in 0..3 {
                for j in 0..3 {
                    // `assert_allclose(..., atol=1e-9)` (with its default `rtol=1e-7`): acos
                    // near -1 keeps only ~8 digits of the angle, in Python as here.
                    assert!((again[i][j] - rot[i][j]).abs() <= 1e-9 + 1e-7 * rot[i][j].abs());
                }
            }
        }
    }

    #[test]
    fn distortion_round_trips_and_json_survives() {
        let cam = Camera::look_at(
            [50.0, 40.0, 60.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            600.0,
            (320, 240),
            Some(&[-0.1, 0.01, 0.001, -0.002]),
            "look",
        );
        for i in 0..20 {
            let x = [
                (i as f64 * 1.3).sin() * 20.0,
                (i as f64 * 0.7).cos() * 20.0,
                i as f64 - 10.0,
            ];
            let uv = cam.project(&x);
            let xn = cam.undistort_point(uv);
            let xc = cam.to_camera(&x);
            assert!((xn[0] - xc[0] / xc[2]).abs() < 1e-6 && (xn[1] - xc[1] / xc[2]).abs() < 1e-6);
        }
        let back = Camera::from_json(&cam.to_json()).expect("parses");
        assert_eq!(back, cam);
        let partial = serde_json::json!({"K": [[500, 0, 10], [0, 500, 20], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 5], "dist": [0.1]});
        let c = Camera::from_json(&partial).expect("defaults fill in");
        assert_eq!(
            (c.size, c.name.as_str(), c.dist),
            ((640, 480), "cam", [0.1, 0.0, 0.0, 0.0, 0.0])
        );
    }
}
