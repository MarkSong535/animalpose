//! Cross-view identity fusion: the port of `src/animalpose/multiview/fusion.py`.
//!
//! Each camera is tracked independently, so the same animal has a different track id per view.
//! Two tracks in different views belong to the same animal when their keypoints are
//! epipolar-consistent over the frames both are seen. Every cross-view track pair is scored by the
//! median symmetric epipolar distance (in approximate pixels) and global identities are built by
//! constrained agglomerative clustering ([`fuse_identities`]).

use std::collections::{BTreeMap, BTreeSet};

use super::camera::{essential_matrix, Camera};
use super::dense::{median, Vec3};
use super::symmetry::{resolve_left_right_cameras, ResolveOptions};
use super::triangulation::{nan_to_num, triangulate_robust};

/// `ViewTrack`: one single-view track, dense over `frames`.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewTrack {
    /// The camera index.
    pub view: usize,
    /// The single-view tracker's id.
    pub track_id: i64,
    /// `(T,)` frame indices, increasing.
    pub frames: Vec<usize>,
    /// `(T, K, 2)` pixels, NaN where missing: `keypoints[t][k]`.
    pub keypoints: Vec<Vec<[f64; 2]>>,
    /// `(T, K)` confidences.
    pub conf: Vec<Vec<f64>>,
}

/// `np.intersect1d(a, b, return_indices=True)` for increasing frame lists: the common frames and
/// their first positions in each.
fn intersect(a: &[usize], b: &[usize]) -> Vec<(usize, usize)> {
    let first = |frames: &[usize]| {
        let mut at = BTreeMap::new();
        for (i, f) in frames.iter().enumerate() {
            at.entry(*f).or_insert(i);
        }
        at
    };
    let (fa, fb) = (first(a), first(b));
    fa.iter()
        .filter_map(|(f, ia)| fb.get(f).map(|ib| (*ia, *ib)))
        .collect()
}

/// `epipolar_cost` with Python's defaults (`conf_thresh=0.3`, `min_points=10`).
pub fn epipolar_cost(a: &ViewTrack, b: &ViewTrack, cam_a: &Camera, cam_b: &Camera) -> f64 {
    epipolar_cost_with(a, b, cam_a, cam_b, 0.3, 10)
}

/// `epipolar_cost`: median symmetric epipolar distance (px) over co-visible keypoints; infinite
/// when the tracks share no frame or fewer than `min_points` confident keypoint pairs.
pub fn epipolar_cost_with(
    a: &ViewTrack,
    b: &ViewTrack,
    cam_a: &Camera,
    cam_b: &Camera,
    conf_thresh: f64,
    min_points: usize,
) -> f64 {
    let common = intersect(&a.frames, &b.frames);
    if common.is_empty() {
        return f64::INFINITY;
    }
    let e = essential_matrix(cam_a, cam_b);
    let mut pairs = Vec::new();
    for &(ia, ib) in &common {
        for (k, (pa, pb)) in a.keypoints[ia].iter().zip(&b.keypoints[ib]).enumerate() {
            let ok = a.conf[ia][k] > conf_thresh
                && b.conf[ib][k] > conf_thresh
                && pa[0].is_finite()
                && pa[1].is_finite()
                && pb[0].is_finite()
                && pb[1].is_finite();
            if ok {
                pairs.push((*pa, *pb));
            }
        }
    }
    if pairs.len() < min_points {
        return f64::INFINITY;
    }
    let mut d: Vec<f64> = pairs
        .iter()
        .map(|(pa, pb)| {
            let na = cam_a.undistort_point(*pa);
            let nb = cam_b.undistort_point(*pb);
            let xa = [na[0], na[1], 1.0];
            let xb = [nb[0], nb[1], 1.0];
            // la = xa @ E.T (epipolar line in b), lb = xb @ E (epipolar line in a)
            let la: Vec<f64> = (0..3)
                .map(|i| (0..3).map(|k| xa[k] * e[i][k]).sum())
                .collect();
            let lb: Vec<f64> = (0..3)
                .map(|j| (0..3).map(|k| xb[k] * e[k][j]).sum())
                .collect();
            let num = (xb[0] * la[0] + xb[1] * la[1] + xb[2] * la[2]).abs();
            let d_b = num / la[0].hypot(la[1]) * cam_b.k[0][0];
            let d_a = num / lb[0].hypot(lb[1]) * cam_a.k[0][0];
            0.5 * (d_a + d_b)
        })
        .collect();
    median(&mut d)
}

/// `_overlap`: more than `tolerance` (2) frames in common.
fn overlap(a: &ViewTrack, b: &ViewTrack) -> bool {
    intersect(&a.frames, &b.frames).len() > 2
}

/// Global ids by `(view, track_id)`, `fuse_identities`' return value.
pub type Identities = BTreeMap<(usize, i64), usize>;

/// `fuse_identities`: a global id for every `(view, track_id)` by agglomerative clustering.
///
/// Every single-view track starts as its own cluster; the two clusters with the lowest average
/// cross-view epipolar cost are merged repeatedly, while the cost is below `max_cost` and tracks
/// of the same view stay temporally disjoint (so a track that fragmented in one camera is stitched
/// back through the others). With `max_animals`, merging continues past `max_cost` (still
/// disjoint) until at most that many identities remain. Ids are numbered by total frames covered,
/// most first.
pub fn fuse_identities(
    tracks: &[ViewTrack],
    cameras: &[Camera],
    max_cost: f64,
    max_animals: Option<usize>,
) -> Identities {
    let n = tracks.len();
    let mut pair = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = (&tracks[i], &tracks[j]);
            if a.view != b.view {
                let c = epipolar_cost(a, b, &cameras[a.view], &cameras[b.view]);
                pair[i][j] = c;
                pair[j][i] = c;
            }
        }
    }
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let linkage = |ca: &[usize], cb: &[usize]| -> f64 {
        for &i in ca {
            for &j in cb {
                if tracks[i].view == tracks[j].view && overlap(&tracks[i], &tracks[j]) {
                    return f64::INFINITY;
                }
            }
        }
        let finite: Vec<f64> = ca
            .iter()
            .flat_map(|i| cb.iter().map(|j| pair[*i][*j]))
            .filter(|c| c.is_finite())
            .collect();
        if finite.is_empty() {
            f64::INFINITY
        } else {
            finite.iter().sum::<f64>() / finite.len() as f64
        }
    };
    while clusters.len() > 1 {
        let (mut best, mut bi, mut bj) = (f64::INFINITY, usize::MAX, usize::MAX);
        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let c = linkage(&clusters[i], &clusters[j]);
                if c < best {
                    (best, bi, bj) = (c, i, j);
                }
            }
        }
        let forced = max_animals.is_some_and(|m| clusters.len() > m);
        if bi == usize::MAX || !best.is_finite() || (best > max_cost && !forced) {
            break;
        }
        let merged = clusters.remove(bj);
        clusters[bi].extend(merged);
    }
    // `clusters.sort(key=lambda c: -frames covered)`: a stable sort.
    clusters.sort_by_key(|c| {
        std::cmp::Reverse(c.iter().map(|i| tracks[*i].frames.len()).sum::<usize>())
    });
    let mut out = Identities::new();
    for (g, c) in clusters.iter().enumerate() {
        for &i in c {
            out.insert((tracks[i].view, tracks[i].track_id), g);
        }
    }
    out
}

/// One identity's 3D output, `triangulate_identities`' dict values.
#[derive(Clone, Debug, PartialEq)]
pub struct Animal3d {
    /// `keypoints3d (T, K, 3)`, NaN where fewer than two views were used.
    pub keypoints3d: Vec<Vec<Vec3>>,
    /// `reproj_error (T, K)`.
    pub reproj_error: Vec<Vec<f64>>,
    /// `n_views (T, K)`.
    pub n_views: Vec<Vec<i64>>,
}

/// `triangulate_identities`: per global identity, `(T, K, 3)` keypoints, reprojection error and
/// views used, over `n_frames`. `resolve_left_right` first makes bilateral labels
/// epipolar-consistent across views ([`super::symmetry`]).
pub fn triangulate_identities(
    tracks: &[ViewTrack],
    cameras: &[Camera],
    ids: &Identities,
    n_frames: usize,
    conf_thresh: f64,
    reproj_thresh: f64,
    resolve_left_right: bool,
) -> BTreeMap<usize, Animal3d> {
    let k = tracks
        .iter()
        .find_map(|t| t.keypoints.first().map(Vec::len))
        .unwrap_or(crate::data::synthetic::NUM_KEYPOINTS);
    let v = cameras.len();
    let gids: BTreeSet<usize> = ids.values().copied().collect();
    let mut out = BTreeMap::new();
    for gid in gids {
        // uv (V, T, K, 2), cf (V, T, K)
        let mut uv = vec![vec![vec![[f64::NAN; 2]; k]; n_frames]; v];
        let mut cf = vec![vec![vec![0.0; k]; n_frames]; v];
        for t in tracks {
            if ids.get(&(t.view, t.track_id)) != Some(&gid) {
                continue;
            }
            for (i, &f) in t.frames.iter().enumerate() {
                if f >= n_frames {
                    continue;
                }
                uv[t.view][f].clone_from(&t.keypoints[i]);
                cf[t.view][f] = t.conf[i].iter().map(|c| nan_to_num(*c)).collect();
            }
        }
        if resolve_left_right {
            let opts = ResolveOptions {
                conf_thresh,
                ..ResolveOptions::default()
            };
            for f in 0..n_frames {
                // `np.isfinite(uv[:, f]).all(-1).any(-1)`: views with any finite keypoint.
                let idx: Vec<usize> = (0..v)
                    .filter(|vi| {
                        uv[*vi][f]
                            .iter()
                            .any(|p| p[0].is_finite() && p[1].is_finite())
                    })
                    .collect();
                if idx.len() < 2 {
                    continue;
                }
                let kp: Vec<Vec<[f64; 2]>> = idx
                    .iter()
                    .map(|vi| {
                        uv[*vi][f]
                            .iter()
                            .map(|p| [nan_to_num(p[0]), nan_to_num(p[1])])
                            .collect()
                    })
                    .collect();
                let conf: Vec<Vec<f64>> = idx.iter().map(|vi| cf[*vi][f].clone()).collect();
                let cams: Vec<Camera> = idx.iter().map(|vi| cameras[*vi].clone()).collect();
                let fixed = resolve_left_right_cameras(&kp, &conf, &cams, &opts);
                for (n, &vi) in idx.iter().enumerate() {
                    for kk in 0..k {
                        for d in 0..2 {
                            // `np.where(np.isfinite(uv), fixed, nan)`, element by element.
                            if uv[vi][f][kk][d].is_finite() {
                                uv[vi][f][kk][d] = fixed.kpts[n][kk][d];
                            }
                        }
                    }
                    cf[vi][f].clone_from(&fixed.conf[n]);
                }
            }
        }
        let flat_uv: Vec<Vec<[f64; 2]>> = uv
            .iter()
            .map(|view| view.iter().flatten().copied().collect())
            .collect();
        let flat_cf: Vec<Vec<f64>> = cf
            .iter()
            .map(|view| view.iter().flatten().copied().collect())
            .collect();
        let res = triangulate_robust(
            cameras,
            &flat_uv,
            Some(&flat_cf),
            conf_thresh,
            reproj_thresh,
        );
        let keypoints3d = res.x.chunks(k.max(1)).map(<[Vec3]>::to_vec).collect();
        let reproj_error = res.err.chunks(k.max(1)).map(<[f64]>::to_vec).collect();
        let n_views = (0..n_frames)
            .map(|f| {
                (0..k)
                    .map(|kk| (0..v).filter(|vi| res.used[*vi][f * k + kk]).count() as i64)
                    .collect()
            })
            .collect();
        out.insert(
            gid,
            Animal3d {
                keypoints3d,
                reproj_error,
                n_views,
            },
        );
    }
    out
}
