//! The small dense linear algebra the multi-view geometry needs, in `f64`: 3x3 helpers, a
//! one-sided Jacobi SVD (`np.linalg.svd`), a Householder QR (`np.linalg.qr`) and a pivoted
//! Gaussian solve (`np.linalg.solve`).
//!
//! `// differs from Python:` numpy calls LAPACK. The SVD here is Hestenes' one-sided Jacobi, which
//! is accurate to working precision for the small (`2V x 4`, `2N x 12`) systems the DLT builds;
//! singular vectors are unique up to sign, and every caller removes the sign (a homogeneous
//! division, or the sign fix of the RQ decomposition).

/// A 3x3 matrix, row-major.
pub type Mat3 = [[f64; 3]; 3];
/// A 3-vector.
pub type Vec3 = [f64; 3];

/// The identity.
pub const EYE3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `a @ b`.
pub fn mul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// `a @ x`.
pub fn mulv3(a: &Mat3, x: &Vec3) -> Vec3 {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

/// `a.T`.
pub fn transpose3(a: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in a.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[j][i] = *v;
        }
    }
    out
}

/// `np.linalg.det`.
pub fn det3(a: &Mat3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// `np.linalg.inv` by the adjugate; `None` for a singular matrix.
pub fn inv3(a: &Mat3) -> Option<Mat3> {
    let det = det3(a);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            // cofactor C[j][i] / det
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            *slot = (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) / det;
        }
    }
    Some(out)
}

/// `a . b`.
pub fn dot3(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `np.cross`.
pub fn cross3(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `np.linalg.norm` of a 3-vector.
pub fn norm3(a: &Vec3) -> f64 {
    dot3(a, a).sqrt()
}

/// `[[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]]`.
pub fn skew(t: &Vec3) -> Mat3 {
    [[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]]
}

/// A thin singular value decomposition `A = U diag(s) V^T` of an `m x n` row-major matrix, with
/// the singular values in decreasing order (as numpy returns them).
#[derive(Clone, Debug)]
pub struct Svd {
    /// Singular values, decreasing.
    pub s: Vec<f64>,
    /// `U`, `m x n` row-major (columns are left singular vectors; zero for a zero singular value).
    pub u: Vec<f64>,
    /// `V`, `n x n` row-major: column `j` is the right singular vector of `s[j]`, i.e. `vt[j]`.
    pub v: Vec<f64>,
    /// Columns of `V`.
    pub n: usize,
}

impl Svd {
    /// `vt[j]`: the right singular vector of the `j`-th largest singular value.
    pub fn vt_row(&self, j: usize) -> Vec<f64> {
        (0..self.n).map(|i| self.v[i * self.n + j]).collect()
    }
}

/// `np.linalg.svd(a)` (thin) by one-sided Jacobi rotations.
pub fn svd(a: &[f64], m: usize, n: usize) -> Svd {
    debug_assert_eq!(a.len(), m * n);
    let mut u = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..80 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (up, uq) = (u[i * n + p], u[i * n + q]);
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let (up, uq) = (u[i * n + p], u[i * n + q]);
                    u[i * n + p] = c * up - s * uq;
                    u[i * n + q] = s * up + c * uq;
                }
                for i in 0..n {
                    let (vp, vq) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * vp - s * vq;
                    v[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let norms: Vec<f64> = (0..n)
        .map(|j| {
            (0..m)
                .map(|i| u[i * n + j] * u[i * n + j])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| norms[*b].total_cmp(&norms[*a]));
    let mut out_u = vec![0.0; m * n];
    let mut out_v = vec![0.0; n * n];
    let mut s = Vec::with_capacity(n);
    for (col, &j) in order.iter().enumerate() {
        let sj = norms[j];
        s.push(sj);
        for i in 0..m {
            out_u[i * n + col] = if sj > 0.0 { u[i * n + j] / sj } else { 0.0 };
        }
        for i in 0..n {
            out_v[i * n + col] = v[i * n + j];
        }
    }
    Svd {
        s,
        u: out_u,
        v: out_v,
        n,
    }
}

/// `np.linalg.svd(a)[2][-1]`: the right singular vector of the smallest singular value, which is
/// the least-squares null vector `argmin |A x|` with `|x| = 1`. With fewer rows than columns the
/// thin decomposition still holds a zero singular value for it.
pub fn null_vector(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let d = svd(a, m, n);
    d.vt_row(n - 1)
}

/// `np.linalg.qr(a)` for a 3x3 matrix by Householder reflections (LAPACK's `geqrf` convention).
pub fn qr3(a: &Mat3) -> (Mat3, Mat3) {
    let mut r = *a;
    let mut q = EYE3;
    for k in 0..2 {
        let norm: f64 = (k..3).map(|i| r[i][k] * r[i][k]).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if r[k][k] > 0.0 { -norm } else { norm };
        let mut v = [0.0; 3];
        for i in k..3 {
            v[i] = r[i][k];
        }
        v[k] -= alpha;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        if vv == 0.0 {
            continue;
        }
        // H = I - 2 v v^T / (v^T v); R <- H R; Q <- Q H.
        for j in 0..3 {
            let d: f64 = (k..3).map(|i| v[i] * r[i][j]).sum::<f64>() * 2.0 / vv;
            for i in k..3 {
                r[i][j] -= d * v[i];
            }
        }
        for row in q.iter_mut() {
            let d: f64 = (k..3).map(|i| row[i] * v[i]).sum::<f64>() * 2.0 / vv;
            for i in k..3 {
                row[i] -= d * v[i];
            }
        }
    }
    for (i, row) in r.iter_mut().enumerate() {
        for slot in row.iter_mut().take(i) {
            *slot = 0.0;
        }
    }
    (q, r)
}

/// `np.linalg.solve(a, b)` for an `n x n` row-major system by Gaussian elimination with partial
/// pivoting; `None` when the matrix is singular.
pub fn solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|i, j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs()))?;
        if m[pivot * n + col] == 0.0 || !m[pivot * n + col].is_finite() {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                m.swap(col * n + j, pivot * n + j);
            }
            x.swap(col, pivot);
        }
        let d = m[col * n + col];
        for i in col + 1..n {
            let f = m[i * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for j in col..n {
                m[i * n + j] -= f * m[col * n + j];
            }
            x[i] -= f * x[col];
        }
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= m[i * n + j] * x[j];
        }
        x[i] = s / m[i * n + i];
    }
    Some(x)
}

/// `np.median` of finite values (the mean of the two middle values for an even count).
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs_and_finds_the_null_vector() {
        // Rank 2 matrix: third column = first + second.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, -1.0, 0.5, -0.5, 2.0, 2.0, 4.0];
        let d = svd(&a, 4, 3);
        for i in 0..4 {
            for j in 0..3 {
                let got: f64 = (0..3)
                    .map(|k| d.u[i * 3 + k] * d.s[k] * d.v[j * 3 + k])
                    .sum();
                assert!((got - a[i * 3 + j]).abs() < 1e-12);
            }
        }
        assert!(d.s[0] >= d.s[1] && d.s[1] >= d.s[2] && d.s[2] < 1e-12);
        let x = null_vector(&a, 4, 3);
        let r = 1.0 / 3f64.sqrt();
        assert!((x[0].abs() - r).abs() < 1e-12 && (x[2].abs() - r).abs() < 1e-12);
        assert!((x[0] - x[1]).abs() < 1e-12 && (x[0] + x[2]).abs() < 1e-12);
    }

    #[test]
    fn qr_and_solve_and_inverse() {
        let a = [[2.0, -1.0, 0.5], [1.0, 3.0, 2.0], [-0.5, 4.0, 1.0]];
        let (q, r) = qr3(&a);
        let back = mul3(&q, &r);
        let qtq = mul3(&transpose3(&q), &q);
        for i in 0..3 {
            for j in 0..3 {
                assert!((back[i][j] - a[i][j]).abs() < 1e-12);
                assert!((qtq[i][j] - EYE3[i][j]).abs() < 1e-12);
            }
        }
        assert!(r[1][0] == 0.0 && r[2][0] == 0.0 && r[2][1] == 0.0);
        let inv = inv3(&a).expect("invertible");
        let id = mul3(&a, &inv);
        for i in 0..3 {
            for j in 0..3 {
                assert!((id[i][j] - EYE3[i][j]).abs() < 1e-12);
            }
        }
        let flat: Vec<f64> = a.iter().flatten().copied().collect();
        let x = solve(&flat, &[1.0, 2.0, 3.0], 3).expect("solvable");
        let y = mulv3(&a, &[x[0], x[1], x[2]]);
        assert!(
            (y[0] - 1.0).abs() < 1e-12 && (y[1] - 2.0).abs() < 1e-12 && (y[2] - 3.0).abs() < 1e-12
        );
        assert_eq!(median(&mut [3.0, 1.0, 2.0, 10.0]), 2.5);
    }
}
