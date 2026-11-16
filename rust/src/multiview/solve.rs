//! Non-linear least squares for calibration: a dense Levenberg-Marquardt for small problems
//! (PnP, intrinsics) and the robust loss functions of `scipy.optimize.least_squares`.
//!
//! `// differs from Python:` `scipy.optimize.least_squares(method="trf", x_scale="jac")` is a
//! trust-region reflective solver, and `cv2.solvePnP` / `cv2.calibrateCamera` run OpenCV's own
//! Levenberg-Marquardt. Both are replaced by a Marquardt-scaled Levenberg-Marquardt with
//! central-difference Jacobians and, for a robust loss, iteratively re-weighted residuals. On a
//! well-posed problem every solver converges to the same minimum (to the stopping tolerance), but
//! the path, and so `nfev`, differs.

use super::dense::solve;

/// `least_squares(loss=...)`: the robust losses scipy offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// `rho(z) = z`.
    Linear,
    /// `rho(z) = z` for `z <= 1`, else `2 sqrt(z) - 1`.
    Huber,
    /// `rho(z) = 2 (sqrt(1 + z) - 1)`.
    SoftL1,
    /// `rho(z) = ln(1 + z)`.
    Cauchy,
    /// `rho(z) = arctan(z)`.
    Arctan,
}

impl Loss {
    /// scipy's loss names.
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        Ok(match name {
            "linear" => Loss::Linear,
            "huber" => Loss::Huber,
            "soft_l1" => Loss::SoftL1,
            "cauchy" => Loss::Cauchy,
            "arctan" => Loss::Arctan,
            other => anyhow::bail!("unknown loss {other:?}"),
        })
    }

    /// `rho(z)`.
    pub fn rho(self, z: f64) -> f64 {
        match self {
            Loss::Linear => z,
            Loss::Huber => {
                if z <= 1.0 {
                    z
                } else {
                    2.0 * z.sqrt() - 1.0
                }
            }
            Loss::SoftL1 => 2.0 * ((1.0 + z).sqrt() - 1.0),
            Loss::Cauchy => z.ln_1p(),
            Loss::Arctan => z.atan(),
        }
    }

    /// `rho'(z)`, the weight of a residual in the re-weighted normal equations.
    pub fn weight(self, z: f64) -> f64 {
        match self {
            Loss::Linear => 1.0,
            Loss::Huber => {
                if z <= 1.0 {
                    1.0
                } else {
                    z.sqrt().recip()
                }
            }
            Loss::SoftL1 => (1.0 + z).sqrt().recip(),
            Loss::Cauchy => (1.0 + z).recip(),
            Loss::Arctan => (1.0 + z * z).recip(),
        }
    }

    /// scipy's cost: `0.5 * f_scale^2 * sum(rho((r / f_scale)^2))`.
    pub fn cost(self, residuals: &[f64], f_scale: f64) -> f64 {
        let f2 = f_scale * f_scale;
        0.5 * f2 * residuals.iter().map(|r| self.rho(r * r / f2)).sum::<f64>()
    }
}

/// Stopping rules shared by the solvers (scipy's defaults).
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    /// Relative cost decrease that ends the run.
    pub ftol: f64,
    /// Relative step size that ends the run.
    pub xtol: f64,
    /// Gradient size that ends the run.
    pub gtol: f64,
    /// Residual evaluations allowed.
    pub max_nfev: usize,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            max_nfev: 200,
        }
    }
}

/// A dense least-squares result.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The parameters.
    pub x: Vec<f64>,
    /// The raw residuals at `x` (`sol.fun`).
    pub fun: Vec<f64>,
    /// Residual evaluations, Jacobian columns excluded.
    pub nfev: usize,
}

/// Minimises `cost(f(x))` by Levenberg-Marquardt with a central-difference Jacobian.
pub fn least_squares_dense(
    f: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    loss: Loss,
    f_scale: f64,
    tol: Tolerances,
) -> Solution {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = f(&x);
    let mut nfev = 1;
    let mut cost = loss.cost(&r, f_scale);
    let mut lambda = 1e-3;
    let m = r.len();
    'outer: while nfev < tol.max_nfev && n > 0 && cost.is_finite() {
        // Jacobian (m x n), column by column.
        let mut jac = vec![0.0; m * n];
        for c in 0..n {
            let h = 1e-6 * x[c].abs().max(1.0);
            let mut xp = x.clone();
            xp[c] += h;
            let rp = f(&xp);
            xp[c] = x[c] - h;
            let rm = f(&xp);
            for i in 0..m {
                jac[i * n + c] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        let f2 = f_scale * f_scale;
        let w: Vec<f64> = r.iter().map(|ri| loss.weight(ri * ri / f2)).collect();
        let mut h = vec![0.0; n * n];
        let mut g = vec![0.0; n];
        for i in 0..m {
            let row = &jac[i * n..(i + 1) * n];
            for a in 0..n {
                g[a] += w[i] * row[a] * r[i];
                for b in a..n {
                    h[a * n + b] += w[i] * row[a] * row[b];
                }
            }
        }
        for a in 0..n {
            for b in 0..a {
                h[a * n + b] = h[b * n + a];
            }
        }
        if g.iter().fold(0.0f64, |acc, v| acc.max(v.abs())) < tol.gtol {
            break;
        }
        let diag: Vec<f64> = (0..n).map(|a| h[a * n + a].max(1e-12)).collect();
        loop {
            let mut damped = h.clone();
            for a in 0..n {
                damped[a * n + a] += lambda * diag[a];
            }
            let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
            let Some(step) = solve(&damped, &neg_g, n) else {
                lambda *= 10.0;
                if lambda > 1e16 {
                    break 'outer;
                }
                continue;
            };
            let xnorm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let snorm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
            if snorm < tol.xtol * (tol.xtol + xnorm) {
                break 'outer;
            }
            let x_new: Vec<f64> = x.iter().zip(&step).map(|(a, b)| a + b).collect();
            let r_new = f(&x_new);
            nfev += 1;
            let cost_new = loss.cost(&r_new, f_scale);
            if cost_new < cost {
                let decrease = cost - cost_new;
                x = x_new;
                r = r_new;
                let old = cost;
                cost = cost_new;
                lambda = (lambda / 3.0).max(1e-12);
                if decrease < tol.ftol * old {
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
    Solution { x, fun: r, nfev }
}
