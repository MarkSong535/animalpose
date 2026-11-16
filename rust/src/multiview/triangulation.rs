//! Confidence-weighted, outlier-robust multi-view triangulation: the port of
//! `src/animalpose/multiview/triangulation.py`.
//!
//! Arrays keep Python's axes: `uv[v][n]` is view `v`'s pixel of point `n` (`(V, N, 2)`, NaN =
//! unseen), `weights[v][n]` / `conf[v][n]` are `(V, N)`.

use super::camera::Camera;
use super::dense::{null_vector, Vec3};

/// `np.nan_to_num` for one value.
pub(crate) fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

fn finite2(p: &[f64; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

/// `triangulate_dlt`: linear triangulation of points seen in several views.
///
/// Points are undistorted first; each view's two DLT rows are scaled by its weight (`None`: all
/// ones). Returns `(N, 3)`, NaN where fewer than two views see the point with a positive weight.
pub fn triangulate_dlt(
    cameras: &[Camera],
    uv: &[Vec<[f64; 2]>],
    weights: Option<&[Vec<f64>]>,
) -> Vec<Vec3> {
    let v = uv.len();
    let n = uv.first().map_or(0, Vec::len);
    let ps: Vec<[[f64; 4]; 3]> = cameras
        .iter()
        .map(|c| {
            let mut p = [[0.0; 4]; 3];
            for i in 0..3 {
                p[i][..3].copy_from_slice(&c.r[i]);
                p[i][3] = c.t[i];
            }
            p
        })
        .collect();
    let mut out = Vec::with_capacity(n);
    let mut a = vec![0.0; 2 * v * 4];
    for j in 0..n {
        let mut n_seen = 0;
        for i in 0..v {
            let w = weights.map_or(1.0, |w| nan_to_num(w[i][j]));
            let p = uv[i][j];
            let seen = finite2(&p) && w > 0.0;
            n_seen += usize::from(seen);
            let xn = cameras[i].undistort_point([nan_to_num(p[0]), nan_to_num(p[1])]);
            let ww = if seen { w } else { 0.0 };
            for c in 0..4 {
                a[(2 * i) * 4 + c] = (xn[0] * ps[i][2][c] - ps[i][0][c]) * ww;
                a[(2 * i + 1) * 4 + c] = (xn[1] * ps[i][2][c] - ps[i][1][c]) * ww;
            }
        }
        if n_seen < 2 {
            out.push([f64::NAN; 3]);
            continue;
        }
        let xh = null_vector(&a, 2 * v, 4);
        let d = if xh[3].abs() < 1e-12 { 1e-12 } else { xh[3] };
        out.push([xh[0] / d, xh[1] / d, xh[2] / d]);
    }
    out
}

/// What [`triangulate_robust`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct Robust {
    /// `X (N, 3)`.
    pub x: Vec<Vec3>,
    /// `err (N,)`: mean reprojection error over the views used; NaN where none was.
    pub err: Vec<f64>,
    /// `used (V, N)`.
    pub used: Vec<Vec<bool>>,
}

/// `triangulate_robust`: triangulation with view selection.
///
/// Views below `conf_thresh` are discarded; if the all-view solution has a view whose
/// reprojection error exceeds `reproj_thresh` px, every view subset (size >= 2) is tried, and the
/// largest subset with all errors under the threshold (lowest mean error on ties) is kept.
pub fn triangulate_robust(
    cameras: &[Camera],
    uv: &[Vec<[f64; 2]>],
    conf: Option<&[Vec<f64>]>,
    conf_thresh: f64,
    reproj_thresh: f64,
) -> Robust {
    let v = uv.len();
    let n = uv.first().map_or(0, Vec::len);
    let conf: Vec<Vec<f64>> = match conf {
        Some(c) => c
            .iter()
            .map(|row| row.iter().map(|x| nan_to_num(*x)).collect())
            .collect(),
        None => vec![vec![1.0; n]; v],
    };
    let ok: Vec<Vec<bool>> = (0..v)
        .map(|i| {
            (0..n)
                .map(|j| finite2(&uv[i][j]) && conf[i][j] >= conf_thresh)
                .collect()
        })
        .collect();
    let masked_uv: Vec<Vec<[f64; 2]>> = (0..v)
        .map(|i| {
            (0..n)
                .map(|j| if ok[i][j] { uv[i][j] } else { [f64::NAN; 2] })
                .collect()
        })
        .collect();
    let masked_w: Vec<Vec<f64>> = (0..v)
        .map(|i| {
            (0..n)
                .map(|j| if ok[i][j] { conf[i][j] } else { 0.0 })
                .collect()
        })
        .collect();
    let mut x = triangulate_dlt(cameras, &masked_uv, Some(&masked_w));
    let mut used = ok.clone();
    let per_view = per_view_errors(cameras, &x, uv);
    let mut err: Vec<f64> = (0..n)
        .map(|j| {
            let e: Vec<f64> = (0..v)
                .filter(|i| used[*i][j])
                .map(|i| per_view[i][j])
                .collect();
            nanmean(&e)
        })
        .collect();
    for j in 0..n {
        // `np.nanmax(np.where(used, e, -inf), axis=0) > reproj_thresh`
        let worst = (0..v)
            .filter(|i| used[*i][j])
            .map(|i| per_view[i][j])
            .filter(|e| !e.is_nan())
            .fold(f64::NEG_INFINITY, f64::max);
        let views: Vec<usize> = (0..v).filter(|i| ok[*i][j]).collect();
        if !(worst > reproj_thresh && views.len() > 2) {
            continue;
        }
        let mut best: Option<(Vec<usize>, f64, Vec3)> = None;
        for size in (2..views.len()).rev() {
            for subset in combinations(&views, size) {
                let cams: Vec<Camera> = subset.iter().map(|i| cameras[*i].clone()).collect();
                let sub_uv: Vec<Vec<[f64; 2]>> = subset.iter().map(|i| vec![uv[*i][j]]).collect();
                let sub_w: Vec<Vec<f64>> = subset.iter().map(|i| vec![conf[*i][j]]).collect();
                let xj = triangulate_dlt(&cams, &sub_uv, Some(&sub_w))[0];
                let e: Vec<f64> = subset
                    .iter()
                    .map(|i| {
                        let p = cameras[*i].project(&xj);
                        (p[0] - uv[*i][j][0]).hypot(p[1] - uv[*i][j][1])
                    })
                    .collect();
                let mean = e.iter().sum::<f64>() / e.len() as f64;
                if e.iter().all(|x| *x <= reproj_thresh)
                    && best.as_ref().map_or(true, |b| mean < b.1)
                {
                    best = Some((subset.clone(), mean, xj));
                }
            }
            if best.is_some() {
                break;
            }
        }
        if let Some((subset, mean, xj)) = best {
            x[j] = xj;
            for (i, row) in used.iter_mut().enumerate() {
                row[j] = subset.contains(&i);
            }
            err[j] = mean;
        }
    }
    Robust { x, err, used }
}

/// `_per_view`: `(V, N)` pixel distance of `project(nan_to_num(X))` to `uv`.
pub(crate) fn per_view_errors(
    cameras: &[Camera],
    x: &[Vec3],
    uv: &[Vec<[f64; 2]>],
) -> Vec<Vec<f64>> {
    cameras
        .iter()
        .zip(uv)
        .map(|(c, obs)| {
            x.iter()
                .zip(obs)
                .map(|(p, o)| {
                    let q = c.project(&[nan_to_num(p[0]), nan_to_num(p[1]), nan_to_num(p[2])]);
                    (q[0] - o[0]).hypot(q[1] - o[1])
                })
                .collect()
        })
        .collect()
}

/// `np.nanmean` of one column; NaN when nothing is finite.
fn nanmean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if finite.is_empty() {
        f64::NAN
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    }
}

/// `itertools.combinations(items, size)`, in the same (lexicographic) order.
pub(crate) fn combinations(items: &[usize], size: usize) -> Vec<Vec<usize>> {
    let n = items.len();
    if size > n {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..size).collect();
    loop {
        out.push(idx.iter().map(|i| items[*i]).collect());
        let Some(i) = (0..size).rev().find(|i| idx[*i] != i + n - size) else {
            return out;
        };
        idx[i] += 1;
        for k in i + 1..size {
            idx[k] = idx[k - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_follow_itertools_order() {
        assert_eq!(
            combinations(&[0, 2, 3, 5], 3),
            vec![vec![0, 2, 3], vec![0, 2, 5], vec![0, 3, 5], vec![2, 3, 5]]
        );
        assert_eq!(combinations(&[1, 2], 2), vec![vec![1, 2]]);
        assert!(combinations(&[1], 2).is_empty());
    }
}
