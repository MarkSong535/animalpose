//! Multi-view left/right disambiguation: the port of `src/animalpose/multiview/symmetry.py`.
//!
//! A single side view often cannot tell the near paw from the far one, so single-view pose swaps
//! bilateral keypoints; another synchronised view usually can. For each bilateral pair, views are
//! decided one at a time (most confident first), keeping or swapping the pair, whichever is more
//! consistent with the views already decided under the symmetric epipolar distance.
//!
//! Geometry is given as pixel-space fundamental matrices `F[(i, j)]` with `x_j^T F x_i = 0`.

use std::collections::BTreeMap;

use super::camera::{essential_matrix, k_inv, Camera};
use super::dense::{mul3, transpose3, Mat3};
use super::triangulation::nan_to_num;
use crate::data::synthetic::BILATERAL_PAIRS;

/// `F_SHARED_X`: corresponding points have equal x (vertical epipolar lines), as in a planar
/// mirror rig whose two views share the horizontal axis: `x_b^T F x_a = u_a - u_b`.
pub const F_SHARED_X: Mat3 = [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

/// `fundamental_from_cameras`: `inv(K_b).T @ E @ inv(K_a)`, for undistorted pixels.
pub fn fundamental_from_cameras(cam_a: &Camera, cam_b: &Camera) -> Mat3 {
    let e = essential_matrix(cam_a, cam_b);
    mul3(&mul3(&transpose3(&k_inv(cam_b)), &e), &k_inv(cam_a))
}

/// `epipolar_distance` for one point pair: the symmetric point-to-epipolar-line distance (px).
pub fn epipolar_distance(f: &Mat3, pa: [f64; 2], pb: [f64; 2]) -> f64 {
    let xa = [pa[0], pa[1], 1.0];
    let xb = [pb[0], pb[1], 1.0];
    // la = xa @ F.T (line in b), lb = xb @ F (line in a)
    let la: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|k| xa[k] * f[i][k]).sum())
        .collect();
    let lb: Vec<f64> = (0..3)
        .map(|j| (0..3).map(|k| xb[k] * f[k][j]).sum())
        .collect();
    let num = (xb[0] * la[0] + xb[1] * la[1] + xb[2] * la[2]).abs();
    0.5 * (num / la[0].hypot(la[1]).max(1e-12) + num / lb[0].hypot(lb[1]).max(1e-12))
}

/// `bilateral_pairs()`: `(left, right)` keypoint indices of the canonical skeleton.
pub fn bilateral_pairs() -> Vec<(usize, usize)> {
    BILATERAL_PAIRS.to_vec()
}

/// Options of [`resolve_left_right`], with Python's defaults.
#[derive(Clone, Debug)]
pub struct ResolveOptions {
    /// Confidence both keypoints of a pair need in a view for it to take part.
    pub conf_thresh: f64,
    /// Minimum epipolar improvement (px) required to swap.
    pub margin: f64,
    /// The pairs to resolve; `None` = [`bilateral_pairs`].
    pub pairs: Option<Vec<(usize, usize)>>,
    /// Fixed anchoring order (most trusted view first) instead of confidence order.
    pub view_priority: Option<Vec<usize>>,
}

impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            conf_thresh: 0.3,
            margin: 0.0,
            pairs: None,
            view_priority: None,
        }
    }
}

/// What [`resolve_left_right`] returns: corrected copies and the `(V, n_pairs)` swaps.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    /// Keypoints `(V, K, 2)` with swapped pairs exchanged.
    pub kpts: Vec<Vec<[f64; 2]>>,
    /// Confidences `(V, K)`, exchanged with them.
    pub conf: Vec<Vec<f64>>,
    /// `swapped[v][p]`.
    pub swapped: Vec<Vec<bool>>,
}

/// `resolve_left_right`: make bilateral labels consistent across views for one animal at one
/// instant. `kpts (V, K, 2)`, `conf (V, K)`; `f` holds the view pairs that have geometry.
pub fn resolve_left_right(
    kpts: &[Vec<[f64; 2]>],
    conf: &[Vec<f64>],
    f: &BTreeMap<(usize, usize), Mat3>,
    opts: &ResolveOptions,
) -> Resolved {
    let mut kpts = kpts.to_vec();
    let mut conf = conf.to_vec();
    let pairs = match &opts.pairs {
        Some(p) if !p.is_empty() => p.clone(),
        _ => bilateral_pairs(),
    };
    let v = kpts.len();
    let mut swapped = vec![vec![false; pairs.len()]; v];
    let fm = |i: usize, j: usize| -> Option<Mat3> {
        if let Some(m) = f.get(&(i, j)) {
            return Some(*m);
        }
        f.get(&(j, i)).map(transpose3)
    };
    for (p, &(lft, rgt)) in pairs.iter().enumerate() {
        let seen: Vec<bool> = (0..v)
            .map(|j| conf[j][lft] >= opts.conf_thresh && conf[j][rgt] >= opts.conf_thresh)
            .collect();
        let order: Vec<usize> = match &opts.view_priority {
            Some(order) => order.clone(),
            None => {
                // `np.argsort(-np.minimum(conf_l, conf_r))`: numpy sorts a handful of views by
                // insertion sort, which keeps ties in view order.
                let key: Vec<f64> = (0..v).map(|j| -conf[j][lft].min(conf[j][rgt])).collect();
                let mut order: Vec<usize> = (0..v).collect();
                order.sort_by(|a, b| key[*a].total_cmp(&key[*b]));
                order
            }
        };
        let mut decided: Vec<usize> = Vec::new();
        for &j in &order {
            if !seen[j] {
                continue;
            }
            let (mut keep, mut swap) = (0.0, 0.0);
            for &i in &decided {
                let Some(fij) = fm(i, j) else {
                    continue;
                };
                let (a_l, a_r) = (kpts[i][lft], kpts[i][rgt]);
                let (b_l, b_r) = (kpts[j][lft], kpts[j][rgt]);
                keep += epipolar_distance(&fij, a_l, b_l) + epipolar_distance(&fij, a_r, b_r);
                swap += epipolar_distance(&fij, a_l, b_r) + epipolar_distance(&fij, a_r, b_l);
            }
            if !decided.is_empty() && swap + opts.margin < keep {
                kpts[j].swap(lft, rgt);
                conf[j].swap(lft, rgt);
                swapped[j][p] = true;
            }
            decided.push(j);
        }
    }
    Resolved {
        kpts,
        conf,
        swapped,
    }
}

/// `resolve_left_right_cameras`: [`resolve_left_right`] for calibrated (possibly distorted)
/// cameras. Points are undistorted to pixels for the decision; the swaps are applied to the
/// original (distorted) keypoints.
pub fn resolve_left_right_cameras(
    kpts: &[Vec<[f64; 2]>],
    conf: &[Vec<f64>],
    cameras: &[Camera],
    opts: &ResolveOptions,
) -> Resolved {
    let und: Vec<Vec<[f64; 2]>> = cameras
        .iter()
        .zip(kpts)
        .map(|(c, k)| {
            k.iter()
                .map(|p| c.undistort_pixel([nan_to_num(p[0]), nan_to_num(p[1])]))
                .collect()
        })
        .collect();
    let mut f = BTreeMap::new();
    for i in 0..cameras.len() {
        for j in i + 1..cameras.len() {
            f.insert((i, j), fundamental_from_cameras(&cameras[i], &cameras[j]));
        }
    }
    let res = resolve_left_right(&und, conf, &f, opts);
    let pairs = match &opts.pairs {
        Some(p) if !p.is_empty() => p.clone(),
        _ => bilateral_pairs(),
    };
    let mut out = kpts.to_vec();
    for (p, &(lft, rgt)) in pairs.iter().enumerate() {
        for (v, row) in out.iter_mut().enumerate() {
            if res.swapped[v][p] {
                row.swap(lft, rgt);
            }
        }
    }
    Resolved {
        kpts: out,
        conf: res.conf,
        swapped: res.swapped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `test_a_view_that_does_not_see_the_pair_cannot_anchor_the_others`.
    #[test]
    fn a_view_that_does_not_see_the_pair_cannot_anchor_the_others() {
        let pair = vec![(0usize, 1usize)];
        let seen = vec![
            vec![[0.0, 5.0], [10.0, 5.0]],
            vec![[10.0, 5.0], [0.0, 5.0]],
            vec![[10.0, 5.0], [0.0, 5.0]],
        ];
        let f = BTreeMap::from([((0, 1), F_SHARED_X)]);
        let opts = ResolveOptions {
            pairs: Some(pair),
            ..ResolveOptions::default()
        };
        let low: Vec<Vec<f64>> = [0.05, 0.9, 0.9].iter().map(|c| vec![*c; 2]).collect();
        let kept = resolve_left_right(&seen, &low, &f, &opts);
        assert!(kept.swapped.iter().flatten().all(|s| !s) && kept.kpts == seen);
        let trusted = vec![vec![0.9; 2]; 3];
        let fixed = resolve_left_right(&seen, &trusted, &f, &opts);
        assert!(fixed.swapped[1][0] && !fixed.swapped[0][0] && !fixed.swapped[2][0]);
        assert_eq!(fixed.kpts[1], vec![[0.0, 5.0], [10.0, 5.0]]);
        // No geometry: nothing changes.
        let none = resolve_left_right(&seen, &trusted, &BTreeMap::new(), &opts);
        assert!(none.swapped.iter().flatten().all(|s| !s) && none.conf == trusted);
    }
}
