//! Small fixed-size dense `f64` matrices: the linear algebra the Kalman filter
//! needs, without any external crate.
//!
//! Stands in for the `numpy` arrays of `animalpose.tracking.kalman`; only the
//! handful of operations that module uses is implemented.

use anyhow::{bail, ensure, Result};
use std::ops::{Add, Mul, Sub};

/// Dense row-major matrix of `f64`.
///
/// Port of the plain `numpy` arrays used throughout
/// `animalpose.tracking.kalman`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    /// All-zero matrix with `rows` rows and `cols` columns.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Wraps `data` (row-major, length `rows * cols`) in a matrix.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        ensure!(
            data.len() == rows * cols,
            "matrix {}x{} needs {} values, got {}",
            rows,
            cols,
            rows * cols,
            data.len()
        );
        Ok(Self { rows, cols, data })
    }

    /// Identity matrix of size `n` (port of `np.eye(n)`).
    pub fn eye(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.data[i * (n + 1)] = 1.0;
        }
        m
    }

    /// Diagonal matrix from `values` (port of `np.diag(v)`).
    pub fn diag(values: &[f64]) -> Self {
        let mut m = Self::new(values.len(), values.len());
        for (i, v) in values.iter().enumerate() {
            m.data[i * (values.len() + 1)] = *v;
        }
        m
    }

    /// Column matrix `(n, 1)` from `values` (port of a 1-D `numpy` vector).
    pub fn column(values: &[f64]) -> Self {
        Self {
            rows: values.len(),
            cols: 1,
            data: values.to_vec(),
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Entry `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Overwrites entry `(i, j)`.
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.cols + j] = value;
    }

    /// The entries in row-major order.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Entries as a flat vector (port of `np.ndarray.flatten`).
    pub fn to_vec(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Sub-matrix covering rows `r0..r1` and columns `c0..c1`.
    ///
    /// Port of the `a[r0:r1, c0:c1]` slices in `BoxKalman.mahalanobis`.
    pub fn block(&self, r0: usize, c0: usize, r1: usize, c1: usize) -> Mat {
        let mut out = Self::new(r1 - r0, c1 - c0);
        for i in 0..(r1 - r0) {
            for j in 0..(c1 - c0) {
                out.set(i, j, self.get(r0 + i, c0 + j));
            }
        }
        out
    }

    /// Transpose (port of `a.T`).
    pub fn transpose(&self) -> Mat {
        let mut out = Self::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    /// Matrix inverse by Gauss-Jordan elimination with partial pivoting
    /// (port of `np.linalg.inv`).
    pub fn inverse(&self) -> Result<Mat> {
        ensure!(self.rows == self.cols, "inverse needs a square matrix");
        let n = self.rows;
        let mut a = self.data.clone();
        let mut inv = Self::eye(n).data;
        let norm = a.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for col in 0..n {
            let mut piv = col;
            for row in col + 1..n {
                if a[row * n + col].abs() > a[piv * n + col].abs() {
                    piv = row;
                }
            }
            if a[piv * n + col].abs() <= f64::EPSILON * norm {
                bail!("matrix is singular");
            }
            if piv != col {
                swap_rows(&mut a, col, piv, n);
                swap_rows(&mut inv, col, piv, n);
            }
            let p = a[col * n + col];
            for j in 0..n {
                a[col * n + j] /= p;
                inv[col * n + j] /= p;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let f = a[row * n + col];
                if f == 0.0 {
                    continue;
                }
                for j in 0..n {
                    a[row * n + j] -= f * a[col * n + j];
                    inv[row * n + j] -= f * inv[col * n + j];
                }
            }
        }
        Mat::from_row_major(n, n, inv)
    }

    /// Solves `self x = rhs` by Gaussian elimination with partial pivoting
    /// (port of `np.linalg.solve` for a single right-hand side).
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        ensure!(self.rows == self.cols, "solve needs a square matrix");
        ensure!(
            rhs.len() == self.rows,
            "solve: right-hand side has {} entries, expected {}",
            rhs.len(),
            self.rows
        );
        let n = self.rows;
        let mut a = self.data.clone();
        let mut b = rhs.to_vec();
        for col in 0..n {
            let mut piv = col;
            for row in col + 1..n {
                if a[row * n + col].abs() > a[piv * n + col].abs() {
                    piv = row;
                }
            }
            if a[piv * n + col] == 0.0 {
                bail!("matrix is singular");
            }
            if piv != col {
                swap_rows(&mut a, col, piv, n);
                b.swap(piv, col);
            }
            let p = a[col * n + col];
            for j in col..n {
                a[col * n + j] /= p;
            }
            b[col] /= p;
            for row in 0..n {
                if row == col {
                    continue;
                }
                let f = a[row * n + col];
                if f == 0.0 {
                    continue;
                }
                for j in col..n {
                    a[row * n + j] -= f * a[col * n + j];
                }
                b[row] -= f * b[col];
            }
        }
        Ok(b)
    }
}

/// Swaps rows `a` and `b` of a row-major buffer of width `width`.
fn swap_rows(data: &mut [f64], a: usize, b: usize, width: usize) {
    if a == b {
        return;
    }
    for j in 0..width {
        data.swap(a * width + j, b * width + j);
    }
}

impl Add<&Mat> for &Mat {
    type Output = Mat;

    fn add(self, other: &Mat) -> Mat {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "shape mismatch in matrix add"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Sub<&Mat> for &Mat {
    type Output = Mat;

    fn sub(self, other: &Mat) -> Mat {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "shape mismatch in matrix subtract"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Mat {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Mul<&Mat> for &Mat {
    type Output = Mat;

    fn mul(self, other: &Mat) -> Mat {
        assert_eq!(
            self.cols, other.rows,
            "shape mismatch in matrix product: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut out = Mat::new(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    let v = out.get(i, j) + a * other.get(k, j);
                    out.set(i, j, v);
                }
            }
        }
        out
    }
}

impl Mul<f64> for &Mat {
    type Output = Mat;

    fn mul(self, other: f64) -> Mat {
        Mat {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|v| v * other).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn inverse_matches_numpy_on_a_well_conditioned_matrix() {
        // np.linalg.inv(np.array([[4., 3., 0.], [3., 4., 1.], [0., 1., 4.]]))
        let m =
            Mat::from_row_major(3, 3, vec![4.0, 3.0, 0.0, 3.0, 4.0, 1.0, 0.0, 1.0, 4.0]).unwrap();
        let inv = m.inverse().unwrap();
        let expect = [
            0.625,
            -0.4999999999999999,
            0.12499999999999997,
            -0.5,
            0.6666666666666665,
            -0.16666666666666663,
            0.12499999999999997,
            -0.16666666666666663,
            0.29166666666666663,
        ];
        for (a, b) in inv.data().iter().zip(expect) {
            assert!(approx(*a, b), "{a} != {b}");
        }
        let id = &m * &inv;
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx(id.get(i, j), if i == j { 1.0 } else { 0.0 }));
            }
        }
    }

    #[test]
    fn solve_matches_the_matrix_product() {
        let m = Mat::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let x = m.solve(&[5.0, 10.0]).unwrap();
        assert!(approx(x[0], 1.0));
        assert!(approx(x[1], 3.0));
        let b = &m * &Mat::column(&x);
        assert!(approx(b.get(0, 0), 5.0));
        assert!(approx(b.get(1, 0), 10.0));
    }

    #[test]
    fn singular_matrix_is_reported() {
        let m = Mat::from_row_major(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(m.inverse().is_err());
        assert!(m.solve(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn kalman_transition_matrices_multiply_as_in_python() {
        // BoxKalman._F: identity with the position block advanced by velocity.
        let mut f = Mat::eye(8);
        for i in 0..4 {
            f.set(i, i + 4, 1.0);
        }
        let x = Mat::column(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let nx = &f * &x;
        assert!(approx(nx.get(0, 0), 6.0));
        assert!(approx(nx.get(3, 0), 12.0));
        // BoxKalman._H = np.eye(4, 8): the observation picks out the position.
        let mut h = Mat::new(4, 8);
        for i in 0..4 {
            h.set(i, i, 1.0);
        }
        let z = &h * &x;
        assert_eq!(*z.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
