//! Camera calibration: the port of `src/animalpose/multiview/calibration.py`.
//!
//! * [`calibrate_dlt`]: the full projection matrix from >= 6 known 3D-2D correspondences
//!   (arena corners, a calibration object), decomposed into `K`, `R`, `t`.
//! * [`solve_extrinsics`]: PnP for a camera with known intrinsics (`cv2.solvePnP`,
//!   `SOLVEPNP_ITERATIVE`).
//! * [`calibrate_intrinsics_points`]: Zhang's planar-target intrinsics from given board
//!   correspondences, the solver behind `calibrate_intrinsics_checkerboard`.
//! * [`bundle_adjust`]: joint refinement of all extrinsics (and optionally the 3D points) from
//!   multi-view 2D observations such as matched animal keypoints.
//!
//! `// differs from Python:` `calibrate_intrinsics_checkerboard` finds the board corners with
//! `cv2.findChessboardCorners` + `cv2.cornerSubPix`; corner detection is not ported, so
//! [`calibrate_intrinsics_checkerboard`] reports itself as unsupported and
//! [`calibrate_intrinsics_points`] takes the detected corners instead. `solvePnP`,
//! `calibrateCamera` and scipy's `least_squares` are replaced by the solvers of
//! [`super::solve`] (same minimum, different iteration path).

use anyhow::{bail, ensure, Result};

use super::camera::{rodrigues, rotation_to_rvec, Camera};
use super::dense::{
    cross3, det3, inv3, mul3, mulv3, norm3, null_vector, qr3, solve, svd, transpose3, Mat3, Vec3,
};
use super::solve::{least_squares_dense, Loss, Tolerances};

/// `_normalize_2d`: the similarity that centres points and scales their mean norm to `sqrt(2)`.
fn normalize_2d(x: &[[f64; 2]]) -> Mat3 {
    let n = x.len() as f64;
    let c = [
        x.iter().map(|p| p[0]).sum::<f64>() / n,
        x.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let mean = x
        .iter()
        .map(|p| (p[0] - c[0]).hypot(p[1] - c[1]))
        .sum::<f64>()
        / n;
    let s = 2f64.sqrt() / mean;
    [[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]]
}

/// `_normalize_3d`: centred, mean norm `sqrt(3)`, as a 4x4 matrix.
fn normalize_3d(x: &[Vec3]) -> [[f64; 4]; 4] {
    let n = x.len() as f64;
    let c: Vec<f64> = (0..3)
        .map(|d| x.iter().map(|p| p[d]).sum::<f64>() / n)
        .collect();
    let mean = x
        .iter()
        .map(|p| norm3(&[p[0] - c[0], p[1] - c[1], p[2] - c[2]]))
        .sum::<f64>()
        / n;
    let s = 3f64.sqrt() / mean;
    [
        [s, 0.0, 0.0, -s * c[0]],
        [0.0, s, 0.0, -s * c[1]],
        [0.0, 0.0, s, -s * c[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn apply2(t: &Mat3, p: [f64; 2]) -> [f64; 2] {
    let q = mulv3(t, &[p[0], p[1], 1.0]);
    [q[0], q[1]]
}

/// `calibrate_dlt`: normalised DLT + RQ decomposition. `X (N, 3)`, `x (N, 2)`, `N >= 6`, not
/// coplanar.
pub fn calibrate_dlt(
    x3: &[Vec3],
    x2: &[[f64; 2]],
    size: (usize, usize),
    name: &str,
) -> Result<Camera> {
    ensure!(
        x3.len() == x2.len(),
        "DLT needs as many 2D points as 3D points"
    );
    if x3.len() < 6 {
        bail!("DLT needs at least 6 correspondences");
    }
    let t2 = normalize_2d(x2);
    let t3 = normalize_3d(x3);
    let mut a = Vec::with_capacity(x3.len() * 24);
    for (p3, p2) in x3.iter().zip(x2) {
        let [u, v] = apply2(&t2, *p2);
        let ph: Vec<f64> = (0..4)
            .map(|i| (0..3).map(|k| t3[i][k] * p3[k]).sum::<f64>() + t3[i][3])
            .collect();
        a.extend([0.0; 4]);
        a.extend(ph.iter().map(|q| -q));
        a.extend(ph.iter().map(|q| v * q));
        a.extend(ph.iter().copied());
        a.extend([0.0; 4]);
        a.extend(ph.iter().map(|q| -u * q));
    }
    let pn = null_vector(&a, 2 * x3.len(), 12);
    let t2i = inv3(&t2).expect("a similarity is invertible");
    // P = inv(T2) @ Pn @ T3
    let mut pt = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            pt[i][j] = (0..4).map(|k| pn[i * 4 + k] * t3[k][j]).sum();
        }
    }
    let mut p = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            p[i][j] = (0..3).map(|k| t2i[i][k] * pt[k][j]).sum();
        }
    }
    let m: Mat3 = [
        [p[0][0], p[0][1], p[0][2]],
        [p[1][0], p[1][1], p[1][2]],
        [p[2][0], p[2][1], p[2][2]],
    ];
    // RQ decomposition via QR of the flipped matrix.
    let flip: Mat3 = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let (q, r) = qr3(&transpose3(&mul3(&flip, &m)));
    let mut k = mul3(&mul3(&flip, &transpose3(&r)), &flip);
    let mut rot = mul3(&flip, &transpose3(&q));
    let sign = |v: f64| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    let d = [sign(k[0][0]), sign(k[1][1]), sign(k[2][2])];
    for row in k.iter_mut() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot *= d[j];
        }
    }
    for (i, row) in rot.iter_mut().enumerate() {
        for slot in row.iter_mut() {
            *slot *= d[i];
        }
    }
    let kf: Vec<f64> = k.iter().flatten().copied().collect();
    let t = solve(&kf, &[p[0][3], p[1][3], p[2][3]], 3).unwrap_or(vec![f64::NAN; 3]);
    let mut t = [t[0], t[1], t[2]];
    if det3(&rot) < 0.0 {
        for row in rot.iter_mut() {
            for slot in row.iter_mut() {
                *slot = -*slot;
            }
        }
        t = [-t[0], -t[1], -t[2]];
    }
    let k22 = k[2][2];
    for row in k.iter_mut() {
        for slot in row.iter_mut() {
            *slot /= k22;
        }
    }
    Ok(Camera::new(k, rot, t, &[0.0; 5], size, name))
}

/// `calibrate_intrinsics_checkerboard`: not ported (see the module docs); always an error naming
/// the replacement.
pub fn calibrate_intrinsics_checkerboard() -> Result<(Mat3, [f64; 5], f64)> {
    bail!(
        "unsupported: checkerboard corner detection (cv2.findChessboardCorners / cornerSubPix in \
         src/animalpose/multiview/calibration.py:calibrate_intrinsics_checkerboard) is not \
         ported; detect the corners elsewhere and pass them to calibrate_intrinsics_points"
    )
}

/// Closest rotation to `m` in the Frobenius norm (`U V^T` of its SVD, with a proper sign).
fn nearest_rotation(m: &Mat3) -> Mat3 {
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    let d = svd(&flat, 3, 3);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| d.u[i * 3 + k] * d.v[j * 3 + k]).sum();
        }
    }
    if det3(&r) < 0.0 {
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] -= 2.0 * d.u[i * 3 + 2] * d.v[j * 3 + 2];
            }
        }
    }
    r
}

/// The normalised DLT homography `H` with `x ~ H [X, Y, 1]` (at least 4 points).
fn homography(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Mat3 {
    let ts = normalize_2d(src);
    let td = normalize_2d(dst);
    let mut a = Vec::with_capacity(src.len() * 18);
    for (s, d) in src.iter().zip(dst) {
        let [x, y] = apply2(&ts, *s);
        let [u, v] = apply2(&td, *d);
        a.extend([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
        a.extend([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]);
    }
    let h = null_vector(&a, 2 * src.len(), 9);
    let hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
    let tdi = inv3(&td).expect("a similarity is invertible");
    mul3(&mul3(&tdi, &hn), &ts)
}

/// An initial `[R | t]` from normalised image coordinates `xn` of world points `x3`:
/// OpenCV's `cvFindExtrinsicCameraParams2` start (a homography for a planar target, the DLT
/// otherwise).
fn initial_pose(x3: &[Vec3], xn: &[[f64; 2]]) -> Result<(Mat3, Vec3)> {
    let n = x3.len() as f64;
    let c: Vec3 = [0, 1, 2].map(|d| x3.iter().map(|p| p[d]).sum::<f64>() / n);
    let mut mm = [[0.0; 3]; 3];
    for p in x3 {
        let q = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for i in 0..3 {
            for j in 0..3 {
                mm[i][j] += q[i] * q[j];
            }
        }
    }
    let flat: Vec<f64> = mm.iter().flatten().copied().collect();
    let eig = svd(&flat, 3, 3);
    let planar = eig.s[2] / eig.s[1].max(f64::MIN_POSITIVE) < 1e-3;
    if planar {
        // Plane frame: rows are the principal directions, the last one the plane normal.
        let mut rp = [[0.0; 3]; 3];
        for (i, row) in rp.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = eig.v[j * 3 + i];
            }
        }
        if det3(&rp) < 0.0 {
            rp[2] = [-rp[2][0], -rp[2][1], -rp[2][2]];
        }
        let src: Vec<[f64; 2]> = x3
            .iter()
            .map(|p| {
                let q = mulv3(&rp, &[p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
                [q[0], q[1]]
            })
            .collect();
        let h = homography(&src, xn);
        let h1 = [h[0][0], h[1][0], h[2][0]];
        let h2 = [h[0][1], h[1][1], h[2][1]];
        let h3 = [h[0][2], h[1][2], h[2][2]];
        let mut lam = 2.0 / (norm3(&h1) + norm3(&h2));
        if h3[2] * lam < 0.0 {
            lam = -lam; // the target must be in front of the camera
        }
        let r1 = h1.map(|v| v * lam);
        let r2 = h2.map(|v| v * lam);
        let r3 = cross3(&r1, &r2);
        let rh = nearest_rotation(&[
            [r1[0], r2[0], r3[0]],
            [r1[1], r2[1], r3[1]],
            [r1[2], r2[2], r3[2]],
        ]);
        let th = h3.map(|v| v * lam);
        let r = mul3(&rh, &rp);
        let rc = mulv3(&r, &c);
        return Ok((r, [th[0] - rc[0], th[1] - rc[1], th[2] - rc[2]]));
    }
    ensure!(
        x3.len() >= 6,
        "DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences"
    );
    let mut a = Vec::with_capacity(x3.len() * 24);
    for (p, q) in x3.iter().zip(xn) {
        let ph = [p[0] - c[0], p[1] - c[1], p[2] - c[2], 1.0];
        let (u, v) = (q[0], q[1]);
        a.extend(ph);
        a.extend([0.0; 4]);
        a.extend(ph.map(|x| -u * x));
        a.extend([0.0; 4]);
        a.extend(ph);
        a.extend(ph.map(|x| -v * x));
    }
    let p = null_vector(&a, 2 * x3.len(), 12);
    let mut m = [[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]];
    let mut t = [p[3], p[7], p[11]];
    if det3(&m) < 0.0 {
        m = m.map(|row| row.map(|v| -v));
        t = t.map(|v| -v);
    }
    let flat: Vec<f64> = m.iter().flatten().copied().collect();
    let scale = svd(&flat, 3, 3).s.iter().sum::<f64>() / 3.0;
    let r = nearest_rotation(&m);
    let t = t.map(|v| v / scale);
    let rc = mulv3(&r, &c);
    Ok((r, [t[0] - rc[0], t[1] - rc[1], t[2] - rc[2]]))
}

/// `solve_extrinsics`: PnP (`cv2.solvePnP(..., flags=SOLVEPNP_ITERATIVE)`) for a camera with known
/// intrinsics `K` and distortion `dist`: a closed-form start, then Levenberg-Marquardt on the
/// pixel reprojection error.
pub fn solve_extrinsics(
    k: &Mat3,
    dist: &[f64],
    x3: &[Vec3],
    x2: &[[f64; 2]],
    size: (usize, usize),
    name: &str,
) -> Result<Camera> {
    ensure!(
        x3.len() == x2.len(),
        "solvePnP needs as many 2D points as 3D points"
    );
    ensure!(
        x3.len() >= 4,
        "solvePnP failed: needs at least 4 correspondences"
    );
    let base = Camera::new(*k, super::dense::EYE3, [0.0; 3], dist, size, name);
    let xn: Vec<[f64; 2]> = x2.iter().map(|p| base.undistort_point(*p)).collect();
    let (r0, t0) = initial_pose(x3, &xn)?;
    let rv = rotation_to_rvec(&r0);
    let x0 = [rv[0], rv[1], rv[2], t0[0], t0[1], t0[2]];
    let residuals = |p: &[f64]| -> Vec<f64> {
        let mut cam = base.clone();
        cam.r = rodrigues(&[p[0], p[1], p[2]]);
        cam.t = [p[3], p[4], p[5]];
        x3.iter()
            .zip(x2)
            .flat_map(|(x, o)| {
                let q = cam.project(x);
                [q[0] - o[0], q[1] - o[1]]
            })
            .collect()
    };
    let tol = Tolerances {
        ftol: 1e-15,
        xtol: 1e-15,
        gtol: 1e-15,
        max_nfev: 200,
    };
    let sol = least_squares_dense(&residuals, &x0, Loss::Linear, 1.0, tol);
    if sol.x.iter().any(|v| !v.is_finite()) {
        bail!("solvePnP failed");
    }
    let mut cam = base;
    cam.r = rodrigues(&[sol.x[0], sol.x[1], sol.x[2]]);
    cam.t = [sol.x[3], sol.x[4], sol.x[5]];
    Ok(cam)
}

/// `cv2.calibrateCamera` on given planar-target correspondences: per view, the board's 3D corners
/// (`z = 0`) and their detected pixels. Returns `(K, dist, rms)` like
/// `calibrate_intrinsics_checkerboard`, which needs at least three views.
///
/// Zhang's method with OpenCV's start (principal point at the image centre, focal lengths from
/// the homographies), then Levenberg-Marquardt over `fx, fy, cx, cy, k1, k2, p1, p2, k3` and every
/// view's pose. `rms` is `sqrt(sum |r|^2 / n_points)`, as OpenCV reports it.
pub fn calibrate_intrinsics_points(
    object_points: &[Vec<Vec3>],
    image_points: &[Vec<[f64; 2]>],
    size: (usize, usize),
) -> Result<(Mat3, [f64; 5], f64)> {
    ensure!(
        object_points.len() == image_points.len(),
        "one list of image points per list of object points"
    );
    if object_points.len() < 3 {
        bail!(
            "checkerboard found in only {} images; need >= 3",
            object_points.len()
        );
    }
    for (o, i) in object_points.iter().zip(image_points) {
        ensure!(
            o.len() == i.len() && o.len() >= 4,
            "every view needs >= 4 matched corners"
        );
        ensure!(
            o.iter().all(|p| p[2] == 0.0),
            "the target must be planar with z = 0"
        );
    }
    let (cx, cy) = ((size.0 as f64 - 1.0) / 2.0, (size.1 as f64 - 1.0) / 2.0);
    // Focal lengths from the homographies, with the principal point at the centre.
    let center: Mat3 = [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]];
    let (mut ata, mut atb) = ([0.0; 4], [0.0; 2]);
    for (o, i) in object_points.iter().zip(image_points) {
        let src: Vec<[f64; 2]> = o.iter().map(|p| [p[0], p[1]]).collect();
        let h = mul3(&center, &homography(&src, i));
        let h1 = [h[0][0], h[1][0], h[2][0]];
        let h2 = [h[0][1], h[1][1], h[2][1]];
        for (row, rhs) in [
            ([h1[0] * h2[0], h1[1] * h2[1]], -h1[2] * h2[2]),
            (
                [h1[0] * h1[0] - h2[0] * h2[0], h1[1] * h1[1] - h2[1] * h2[1]],
                -(h1[2] * h1[2] - h2[2] * h2[2]),
            ),
        ] {
            ata[0] += row[0] * row[0];
            ata[1] += row[0] * row[1];
            ata[3] += row[1] * row[1];
            atb[0] += row[0] * rhs;
            atb[1] += row[1] * rhs;
        }
    }
    ata[2] = ata[1];
    let ab = solve(&ata, &atb, 2).unwrap_or(vec![f64::NAN; 2]);
    ensure!(
        ab[0] > 0.0 && ab[1] > 0.0,
        "calibrateCamera failed: degenerate board views"
    );
    let (fx, fy) = (ab[0].recip().sqrt(), ab[1].recip().sqrt());
    let k0 = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]];
    let mut x0 = vec![fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0];
    let ki = inv3(&k0).expect("positive focal lengths");
    for (o, i) in object_points.iter().zip(image_points) {
        let xn: Vec<[f64; 2]> = i.iter().map(|p| apply2(&ki, *p)).collect();
        let (r, t) = initial_pose(o, &xn)?;
        x0.extend(rotation_to_rvec(&r));
        x0.extend(t);
    }
    let cam_at = |p: &[f64], view: usize| -> Camera {
        let k = [[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]];
        let at = 9 + 6 * view;
        Camera::new(
            k,
            rodrigues(&[p[at], p[at + 1], p[at + 2]]),
            [p[at + 3], p[at + 4], p[at + 5]],
            &p[4..9],
            size,
            "cam",
        )
    };
    let residuals = |p: &[f64]| -> Vec<f64> {
        let mut out = Vec::new();
        for (v, (o, i)) in object_points.iter().zip(image_points).enumerate() {
            let cam = cam_at(p, v);
            for (x, obs) in o.iter().zip(i) {
                let q = cam.project(x);
                out.push(q[0] - obs[0]);
                out.push(q[1] - obs[1]);
            }
        }
        out
    };
    let tol = Tolerances {
        ftol: 1e-15,
        xtol: 1e-15,
        gtol: 1e-15,
        max_nfev: 400,
    };
    let sol = least_squares_dense(&residuals, &x0, Loss::Linear, 1.0, tol);
    let p = &sol.x;
    let k = [[p[0], 0.0, p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]];
    let dist = [p[4], p[5], p[6], p[7], p[8]];
    let n_points = sol.fun.len() / 2;
    let rms = (sol.fun.iter().map(|r| r * r).sum::<f64>() / n_points.max(1) as f64).sqrt();
    Ok((k, dist, rms))
}

/// `reprojection_errors`: `obs (V, N, 2)` (NaN = missing) -> pixel error `(V, N)`.
pub fn reprojection_errors(cameras: &[Camera], x: &[Vec3], obs: &[Vec<[f64; 2]>]) -> Vec<Vec<f64>> {
    cameras
        .iter()
        .zip(obs)
        .map(|(c, o)| {
            x.iter()
                .zip(o)
                .map(|(p, q)| {
                    let uv = c.project(p);
                    (uv[0] - q[0]).hypot(uv[1] - q[1])
                })
                .collect()
        })
        .collect()
}

/// `bundle_adjust`'s keyword arguments, with Python's defaults.
#[derive(Clone, Copy, Debug)]
pub struct BundleOptions {
    /// Keep the first camera fixed (the gauge).
    pub fix_first: bool,
    /// Refine the 3D points too.
    pub refine_points: bool,
    /// The robust loss.
    pub loss: Loss,
    /// Residual scale of the loss, px.
    pub f_scale: f64,
    /// Residual evaluations allowed.
    pub max_nfev: usize,
}

impl Default for BundleOptions {
    fn default() -> Self {
        Self {
            fix_first: true,
            refine_points: true,
            loss: Loss::Huber,
            f_scale: 2.0,
            max_nfev: 200,
        }
    }
}

/// `bundle_adjust`'s info dict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BundleInfo {
    /// RMS reprojection error (px) before.
    pub rmse_before: f64,
    /// RMS reprojection error (px) after.
    pub rmse_after: f64,
    /// Residual evaluations.
    pub nfev: usize,
}

struct Problem<'a> {
    base: &'a [Camera],
    obs: &'a [Vec<[f64; 2]>],
    seen: Vec<(usize, usize)>,
    free: Vec<usize>,
    slot: Vec<Option<usize>>,
    n_points: usize,
    refine: bool,
    fixed_points: &'a [Vec3],
}

impl Problem<'_> {
    fn unpack(&self, p: &[f64]) -> (Vec<Camera>, Vec<Vec3>) {
        let mut cams = self.base.to_vec();
        for (j, &i) in self.free.iter().enumerate() {
            cams[i].r = rodrigues(&[p[6 * j], p[6 * j + 1], p[6 * j + 2]]);
            cams[i].t = [p[6 * j + 3], p[6 * j + 4], p[6 * j + 5]];
        }
        let pts = if self.refine {
            let off = 6 * self.free.len();
            (0..self.n_points)
                .map(|k| [p[off + 3 * k], p[off + 3 * k + 1], p[off + 3 * k + 2]])
                .collect()
        } else {
            self.fixed_points.to_vec()
        };
        (cams, pts)
    }

    fn residuals(&self, cams: &[Camera], pts: &[Vec3]) -> Vec<f64> {
        let mut out = Vec::with_capacity(2 * self.seen.len());
        for &(i, j) in &self.seen {
            let q = cams[i].project(&pts[j]);
            out.push(q[0] - self.obs[i][j][0]);
            out.push(q[1] - self.obs[i][j][1]);
        }
        out
    }
}

fn rmse(res: &[f64]) -> f64 {
    let n = res.len() / 2;
    if n == 0 {
        return f64::NAN;
    }
    (res.iter().map(|r| r * r).sum::<f64>() / n as f64).sqrt()
}

/// `bundle_adjust`: refine extrinsics (and points) to minimise the reprojection error.
///
/// `obs (V, N, 2)` pixel observations, NaN where a point is not seen; `x (N, 3)` the starting
/// points. The first camera fixes the gauge when `fix_first`. The Jacobian is sparse in the same
/// pattern scipy is told about (`jac_sparsity`): each observation depends on its camera's six
/// parameters and its point's three, which the solver exploits through the Schur complement on
/// the point blocks.
pub fn bundle_adjust(
    cameras: &[Camera],
    x: &[Vec3],
    obs: &[Vec<[f64; 2]>],
    opts: BundleOptions,
) -> (Vec<Camera>, Vec<Vec3>, BundleInfo) {
    let v = cameras.len();
    let n = x.len();
    let mut seen = Vec::new();
    for (i, row) in obs.iter().enumerate().take(v) {
        for (j, p) in row.iter().enumerate().take(n) {
            if p[0].is_finite() && p[1].is_finite() {
                seen.push((i, j));
            }
        }
    }
    let free: Vec<usize> = (usize::from(opts.fix_first)..v).collect();
    let mut slot = vec![None; v];
    for (s, &i) in free.iter().enumerate() {
        slot[i] = Some(s);
    }
    let prob = Problem {
        base: cameras,
        obs,
        seen,
        free,
        slot,
        n_points: n,
        refine: opts.refine_points,
        fixed_points: x,
    };
    let mut p: Vec<f64> = Vec::new();
    for &i in &prob.free {
        p.extend(rotation_to_rvec(&cameras[i].r));
        p.extend(cameras[i].t);
    }
    if prob.refine {
        p.extend(x.iter().flatten());
    }
    let (cams0, pts0) = prob.unpack(&p);
    let mut res = prob.residuals(&cams0, &pts0);
    let before = rmse(&res);
    let mut nfev = 1;
    let mut cost = opts.loss.cost(&res, opts.f_scale);
    let tol = Tolerances {
        max_nfev: opts.max_nfev,
        ..Tolerances::default()
    };
    let nf = prob.free.len();
    let f2 = opts.f_scale * opts.f_scale;
    let mut lambda = 1e-3;
    // Observations of each point, for the Schur complement.
    let mut point_obs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (o, &(_, j)) in prob.seen.iter().enumerate() {
        point_obs[j].push(o);
    }
    'outer: while nfev < tol.max_nfev && !p.is_empty() && cost.is_finite() {
        let (cams, pts) = prob.unpack(&p);
        // Central differences of every free camera's six parameters.
        let mut cam_plus: Vec<Vec<(Camera, Camera, f64)>> = Vec::with_capacity(nf);
        for s in 0..nf {
            let mut per = Vec::with_capacity(6);
            for c in 0..6 {
                let h = 1e-6 * p[6 * s + c].abs().max(1.0);
                let mut pp = p[6 * s..6 * s + 6].to_vec();
                pp[c] += h;
                let mut a = cams[prob.free[s]].clone();
                a.r = rodrigues(&[pp[0], pp[1], pp[2]]);
                a.t = [pp[3], pp[4], pp[5]];
                pp[c] -= 2.0 * h;
                let mut b = cams[prob.free[s]].clone();
                b.r = rodrigues(&[pp[0], pp[1], pp[2]]);
                b.t = [pp[3], pp[4], pp[5]];
                per.push((a, b, h));
            }
            cam_plus.push(per);
        }
        let mut u = vec![[[0.0; 6]; 6]; nf];
        let mut gc = vec![[0.0; 6]; nf];
        let mut vb = vec![[[0.0; 3]; 3]; if prob.refine { n } else { 0 }];
        let mut gp = vec![[0.0; 3]; if prob.refine { n } else { 0 }];
        let mut wb = vec![[[0.0; 3]; 6]; prob.seen.len()];
        for (o, &(i, j)) in prob.seen.iter().enumerate() {
            let r = [res[2 * o], res[2 * o + 1]];
            let w = [
                opts.loss.weight(r[0] * r[0] / f2),
                opts.loss.weight(r[1] * r[1] / f2),
            ];
            let mut jc = [[0.0; 6]; 2];
            if let Some(s) = prob.slot[i] {
                for (c, (a, b, h)) in cam_plus[s].iter().enumerate() {
                    let (qa, qb) = (a.project(&pts[j]), b.project(&pts[j]));
                    jc[0][c] = (qa[0] - qb[0]) / (2.0 * h);
                    jc[1][c] = (qa[1] - qb[1]) / (2.0 * h);
                }
                for a in 0..6 {
                    for d in 0..2 {
                        gc[s][a] += w[d] * jc[d][a] * r[d];
                        for b in 0..6 {
                            u[s][a][b] += w[d] * jc[d][a] * jc[d][b];
                        }
                    }
                }
            }
            if prob.refine {
                let mut jp = [[0.0; 3]; 2];
                for c in 0..3 {
                    let h = 1e-6 * pts[j][c].abs().max(1.0);
                    let mut xa = pts[j];
                    xa[c] += h;
                    let mut xb = pts[j];
                    xb[c] -= h;
                    let (qa, qb) = (cams[i].project(&xa), cams[i].project(&xb));
                    jp[0][c] = (qa[0] - qb[0]) / (2.0 * h);
                    jp[1][c] = (qa[1] - qb[1]) / (2.0 * h);
                }
                for a in 0..3 {
                    for d in 0..2 {
                        gp[j][a] += w[d] * jp[d][a] * r[d];
                        for b in 0..3 {
                            vb[j][a][b] += w[d] * jp[d][a] * jp[d][b];
                        }
                    }
                }
                if prob.slot[i].is_some() {
                    for a in 0..6 {
                        for b in 0..3 {
                            wb[o][a][b] = (0..2).map(|d| w[d] * jc[d][a] * jp[d][b]).sum();
                        }
                    }
                }
            }
        }
        let gmax = gc
            .iter()
            .flatten()
            .chain(gp.iter().flatten())
            .fold(0.0f64, |acc, g| acc.max(g.abs()));
        if gmax < tol.gtol {
            break;
        }
        loop {
            // Damped blocks, V^-1 per point.
            let vinv: Vec<Mat3> = vb
                .iter()
                .map(|blk| {
                    let mut d = *blk;
                    for a in 0..3 {
                        d[a][a] += lambda * blk[a][a].max(1e-12);
                    }
                    inv3(&d).unwrap_or([[0.0; 3]; 3])
                })
                .collect();
            let m = 6 * nf;
            let mut s_mat = vec![0.0; m * m];
            let mut rhs = vec![0.0; m];
            for s in 0..nf {
                for a in 0..6 {
                    rhs[6 * s + a] = -gc[s][a];
                    for b in 0..6 {
                        s_mat[(6 * s + a) * m + 6 * s + b] = u[s][a][b];
                    }
                    s_mat[(6 * s + a) * m + 6 * s + a] += lambda * u[s][a][a].max(1e-12);
                }
            }
            if prob.refine && nf > 0 {
                for (j, list) in point_obs.iter().enumerate() {
                    let vi = &vinv[j];
                    // W_o V^-1 for every observation of this point by a free camera.
                    let mut wv: Vec<(usize, usize, [[f64; 3]; 6])> = Vec::new();
                    for &o in list {
                        let Some(s) = prob.slot[prob.seen[o].0] else {
                            continue;
                        };
                        let mut out = [[0.0; 3]; 6];
                        for a in 0..6 {
                            for b in 0..3 {
                                out[a][b] = (0..3).map(|k| wb[o][a][k] * vi[k][b]).sum();
                            }
                            rhs[6 * s + a] += (0..3).map(|k| out[a][k] * gp[j][k]).sum::<f64>();
                        }
                        wv.push((s, o, out));
                    }
                    for (s1, _, wv1) in &wv {
                        for (s2, o2, _) in &wv {
                            for a in 0..6 {
                                for b in 0..6 {
                                    let val: f64 = (0..3).map(|k| wv1[a][k] * wb[*o2][b][k]).sum();
                                    s_mat[(6 * s1 + a) * m + 6 * s2 + b] -= val;
                                }
                            }
                        }
                    }
                }
            }
            let dc = if m > 0 {
                match solve(&s_mat, &rhs, m) {
                    Some(d) => d,
                    None => {
                        lambda *= 10.0;
                        if lambda > 1e16 {
                            break 'outer;
                        }
                        continue;
                    }
                }
            } else {
                Vec::new()
            };
            let mut step = dc.clone();
            if prob.refine {
                for j in 0..n {
                    let mut b = [-gp[j][0], -gp[j][1], -gp[j][2]];
                    for &o in &point_obs[j] {
                        if let Some(s) = prob.slot[prob.seen[o].0] {
                            for (k, bk) in b.iter_mut().enumerate() {
                                *bk -= (0..6).map(|a| wb[o][a][k] * dc[6 * s + a]).sum::<f64>();
                            }
                        }
                    }
                    step.extend(mulv3(&vinv[j], &b));
                }
            }
            let xnorm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            let snorm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
            if snorm < tol.xtol * (tol.xtol + xnorm) {
                break 'outer;
            }
            let p_new: Vec<f64> = p.iter().zip(&step).map(|(a, b)| a + b).collect();
            let (c_new, x_new) = prob.unpack(&p_new);
            let res_new = prob.residuals(&c_new, &x_new);
            nfev += 1;
            let cost_new = opts.loss.cost(&res_new, opts.f_scale);
            if cost_new < cost {
                let old = cost;
                p = p_new;
                res = res_new;
                cost = cost_new;
                lambda = (lambda / 3.0).max(1e-12);
                if old - cost < tol.ftol * old {
                    break 'outer;
                }
                break;
            }
            lambda *= 4.0;
            if nfev >= tol.max_nfev || lambda > 1e16 {
                break 'outer;
            }
        }
    }
    let (cams, pts) = prob.unpack(&p);
    let after = rmse(&res);
    (
        cams,
        pts,
        BundleInfo {
            rmse_before: before,
            rmse_after: after,
            nfev,
        },
    )
}
