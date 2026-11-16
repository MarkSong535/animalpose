//! Behavior labels for a new setup from its pose tracks alone: the port of
//! `src/animalpose/behavior/clusters.py`.
//!
//! 1. [`cluster_tracks`]: per-frame kinematic features ([`kinematics`]) averaged over ~0.5 s,
//!    standardised and clustered with k-means fitted on the training range only; example bouts are
//!    saved as montages.
//! 2. [`write_naming`]: each cluster's median kinematics are matched against [`NAME_RULES`].
//! 3. [`label_clusters`]: a `cluster -> behavior` mapping becomes per-frame labels (short bouts
//!    merged into the preceding one) and train/held-out session `.npz` files.
//!
//! `// differs from Python:`
//! * k-means is a port of scikit-learn's `KMeans(k, n_init=10, random_state=0)` (greedy k-means++
//!   with `2 + ln k` local trials, Lloyd iterations to `tol = 1e-4 * mean variance`, the best of 10
//!   inits by inertia), but its random stream is the crate's [`Rng`] rather than numpy's
//!   `RandomState(0)`, so the cluster *numbering* — and, on data without a clear structure, the
//!   partition — can differ from a Python run. The montage bouts are drawn from the same [`Rng`]
//!   instead of `np.random.default_rng(0)`.
//! * Kinematics are computed in `f64` (numpy keeps `float32` tracks in `float32` in places).
//! * Montages are grey (frames are grey levels here), and crops that would run off the padded
//!   frame are clamped to its edge where numpy would return a shorter slice.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

use crate::algos::image::{resize_linear, Image};
use crate::cli::catalog::Catalog;
use crate::cli::pyjson::{py_round, write_text, Json};
use crate::cli::sources::Source;
use crate::data::sessions::{NpyData, Npz, IGNORE};
use crate::data::synthetic::{index, Rng, NUM_KEYPOINTS};

/// The feature names [`kinematics`] returns, in column order.
pub const FEATURES: [&str; 9] = [
    "speed",
    "length",
    "turn",
    "posture",
    "head_motion",
    "bend",
    "speed_sd",
    "length_sd",
    "posture_sd",
];

/// One criterion of a naming rule: `(feature, '<' or '>', threshold)`.
pub type Criterion = (&'static str, char, f64);

/// `NAME_RULES`: `(name, [(feature, op, threshold)])`; the first rule whose criteria all hold wins.
pub const NAME_RULES: &[(&str, &[Criterion])] = &[
    ("other", &[("speed", '>', 8.0)]),
    ("other", &[("speed_sd", '>', 20.0)]),
    ("other", &[("posture_sd", '>', 20.0)]),
    (
        "running",
        &[
            ("speed", '>', 3.0),
            ("length", '>', 0.9),
            ("bend", '<', 1.0),
        ],
    ),
    (
        "grooming",
        &[
            ("bend", '>', 1.0),
            ("head_motion", '>', 0.8),
            ("length", '<', 0.6),
        ],
    ),
    ("turning", &[("turn", '>', 4.0), ("posture", '>', 0.5)]),
    (
        "rearing",
        &[
            ("length", '<', 0.85),
            ("bend", '<', 1.0),
            ("head_motion", '>', 0.4),
            ("posture", '>', 0.5),
        ],
    ),
    (
        "walking",
        &[
            ("speed", '>', 0.4),
            ("length", '>', 0.9),
            ("bend", '<', 1.0),
        ],
    ),
    (
        "sniffing",
        &[
            ("speed", '<', 0.35),
            ("length", '>', 0.95),
            ("bend", '<', 0.4),
        ],
    ),
    ("resting", &[("speed", '<', 0.35), ("length", '<', 0.95)]),
];

// ------------------------------------------------------------------------------------ kinematics

/// `np.gradient(f, t)` (edge_order 1) for one series: second-order central differences with the
/// unequal-spacing weights inside, one-sided at the ends; numpy's uniform formula when every
/// spacing is exactly equal.
pub fn gradient(f: &[f64], t: &[f64]) -> Vec<f64> {
    let n = f.len();
    if n < 2 {
        return vec![f64::NAN; n];
    }
    let dx: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    let mut out = vec![0.0; n];
    if dx.iter().all(|d| *d == dx[0]) {
        let d = dx[0];
        for i in 1..n - 1 {
            out[i] = (f[i + 1] - f[i - 1]) / (2.0 * d);
        }
    } else {
        for i in 1..n - 1 {
            let (dx1, dx2) = (dx[i - 1], dx[i]);
            let a = -dx2 / (dx1 * (dx1 + dx2));
            let b = (dx2 - dx1) / (dx1 * dx2);
            let c = dx1 / (dx2 * (dx1 + dx2));
            out[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
        }
    }
    out[0] = (f[1] - f[0]) / dx[0];
    out[n - 1] = (f[n - 1] - f[n - 2]) / dx[n - 2];
    out
}

/// `np.unwrap(p)` with the default `discont = pi`.
pub fn unwrap(p: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out = p.to_vec();
    let mut correct = 0.0;
    for i in 1..p.len() {
        let dd = p[i] - p[i - 1];
        let mut ddmod = (dd + PI).rem_euclid(2.0 * PI) - PI;
        if ddmod == -PI && dd > 0.0 {
            ddmod = PI;
        }
        let mut ph = ddmod - dd;
        if dd.abs() < PI {
            ph = 0.0;
        }
        correct += ph;
        out[i] = p[i] + correct;
    }
    out
}

/// `clusters.py:_smooth` on one column: an edge-padded moving average of `w` samples.
pub fn smooth(x: &[f64], w: usize) -> Vec<f64> {
    if w <= 1 || x.is_empty() {
        return x.to_vec();
    }
    let n = x.len();
    let left = w / 2;
    let at = |i: isize| x[i.clamp(0, n as isize - 1) as usize];
    let k = 1.0 / w as f64;
    (0..n)
        .map(|i| {
            (0..w)
                .map(|j| at(i as isize + j as isize - left as isize) * k)
                .sum()
        })
        .collect()
}

fn nan_to_num(v: f64) -> f64 {
    crate::algos::kinematics::nan_to_num(v)
}

/// `clusters.py:kinematics`: `(T, 9)` row-major features from keypoints `kp` (`T * K` points) and
/// confidences `conf` (`T * K`); derivatives use `times` (seconds) when given, else `t / fps`.
pub fn kinematics(kp: &[[f64; 2]], conf: &[f64], fps: f64, times: Option<&[f64]>) -> Vec<[f64; 9]> {
    let k = NUM_KEYPOINTS;
    let n = kp.len() / k;
    let t: Vec<f64> = match times {
        Some(ts) => ts[..n.min(ts.len())].to_vec(),
        None => (0..n).map(|i| i as f64 / fps).collect(),
    };
    let body = [
        "neck",
        "spine_upper",
        "spine_mid",
        "spine_lower",
        "tail_base",
        "head_center",
    ]
    .map(index);
    let (nose_i, head_i, tail_i) = (index("nose"), index("head_center"), index("tail_base"));
    let at = |f: usize, j: usize| kp[f * k + j];
    let mut centre = vec![[0.0; 2]; n];
    for (f, c) in centre.iter_mut().enumerate() {
        let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
        for &j in &body {
            let w = if conf[f * k + j] >= 0.3 { 1.0 } else { 0.0 };
            sx += at(f, j)[0] * w;
            sy += at(f, j)[1] * w;
            sw += w;
        }
        let d = f64::max(sw, 1e-6);
        *c = [sx / d, sy / d];
    }
    let length: Vec<f64> = (0..n)
        .map(|f| {
            let (a, b) = (at(f, nose_i), at(f, tail_i));
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        })
        .collect();
    let reference = crate::algos::kinematics::nan_median(&length) + 1e-6;
    let col = |v: &[[f64; 2]], d: usize| v.iter().map(|p| p[d]).collect::<Vec<f64>>();
    let grad2 = |v: &[[f64; 2]]| -> Vec<[f64; 2]> {
        let gx = gradient(&col(v, 0), &t);
        let gy = gradient(&col(v, 1), &t);
        gx.into_iter().zip(gy).map(|(x, y)| [x, y]).collect()
    };
    let norm = |p: [f64; 2]| (p[0] * p[0] + p[1] * p[1]).sqrt();
    let v = grad2(&centre);
    let speed: Vec<f64> = v
        .iter()
        .map(|p| norm([p[0] / reference, p[1] / reference]))
        .collect();
    let ang: Vec<f64> = (0..n)
        .map(|f| {
            let (a, b) = (at(f, nose_i), at(f, tail_i));
            let ax = (a[0] - b[0]) / (length[f] + 1e-6);
            let ay = (a[1] - b[1]) / (length[f] + 1e-6);
            ay.atan2(ax)
        })
        .collect();
    let turn: Vec<f64> = gradient(&unwrap(&ang), &t)
        .into_iter()
        .map(f64::abs)
        .collect();
    let mut posture_sum = vec![0.0; n];
    let mut posture_cnt = vec![0usize; n];
    for j in 0..k {
        let rel: Vec<[f64; 2]> = (0..n)
            .map(|f| {
                let p = at(f, j);
                [
                    (p[0] - centre[f][0]) / reference,
                    (p[1] - centre[f][1]) / reference,
                ]
            })
            .collect();
        for (f, g) in grad2(&rel).into_iter().enumerate() {
            let m = norm(g);
            if !m.is_nan() {
                posture_sum[f] += m;
                posture_cnt[f] += 1;
            }
        }
    }
    let posture: Vec<f64> = posture_sum
        .iter()
        .zip(&posture_cnt)
        .map(|(s, c)| if *c == 0 { f64::NAN } else { s / *c as f64 })
        .collect();
    let nh: Vec<[f64; 2]> = (0..n)
        .map(|f| {
            let (a, b) = (at(f, nose_i), at(f, head_i));
            [(a[0] - b[0]) / reference, (a[1] - b[1]) / reference]
        })
        .collect();
    let head_rel: Vec<f64> = grad2(&nh).into_iter().map(norm).collect();
    let bend: Vec<f64> = (0..n)
        .map(|f| {
            let (a, c, b) = (at(f, nose_i), centre[f], at(f, tail_i));
            let hv = [a[0] - c[0], a[1] - c[1]];
            let tv = [c[0] - b[0], c[1] - b[1]];
            (hv[0] * tv[1] - hv[1] * tv[0])
                .atan2(hv[0] * tv[0] + hv[1] * tv[1])
                .abs()
        })
        .collect();
    let raw: [Vec<f64>; 6] = [
        speed,
        length.iter().map(|l| l / reference).collect(),
        turn,
        posture,
        head_rel,
        bend,
    ];
    let w = ((0.5 * fps).round_ties_even() as i64).max(1) as usize;
    let mut out = vec![[0.0; 9]; n];
    for (c, series) in raw.iter().enumerate() {
        let clean: Vec<f64> = series.iter().map(|v| nan_to_num(*v)).collect();
        for (f, v) in smooth(&clean, w).into_iter().enumerate() {
            out[f][c] = v;
        }
    }
    for (slot, c) in [(6usize, 0usize), (7, 1), (8, 3)] {
        let clean: Vec<f64> = raw[c].iter().map(|v| nan_to_num(*v)).collect();
        let sq: Vec<f64> = clean.iter().map(|v| v * v).collect();
        let m2 = smooth(&sq, 2 * w);
        let m1 = smooth(&clean, 2 * w);
        for f in 0..n {
            out[f][slot] = (m2[f] - m1[f] * m1[f]).max(0.0).sqrt();
        }
    }
    out
}

/// `clusters.py:bouts`: `(start, end, label)` for each run of one label.
pub fn bouts(lab: &[i64]) -> Vec<(usize, usize, i64)> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=lab.len() {
        if i == lab.len() || lab[i] != lab[start] {
            if i > start {
                out.push((start, i, lab[start]));
            }
            start = i;
        }
    }
    out
}

// ------------------------------------------------------------------------------------ naming

fn matches(feat: &Json, crit: &[Criterion]) -> bool {
    crit.iter().all(|(name, op, thr)| {
        let v = feat.get(name).and_then(Json::as_f64).unwrap_or(f64::NAN);
        v.is_finite() && if *op == '<' { v < *thr } else { v > *thr }
    })
}

/// `clusters.py:name_clusters`: `{cluster: {name, share_of_train, features}}`.
pub fn name_clusters(summary: &Json) -> Json {
    let Json::Obj(pairs) = summary else {
        return Json::Obj(vec![]);
    };
    Json::Obj(
        pairs
            .iter()
            .map(|(c, stats)| {
                let name = NAME_RULES
                    .iter()
                    .find(|(_, crit)| matches(stats, crit))
                    .map(|(n, _)| *n)
                    .unwrap_or("other");
                let features = match stats {
                    Json::Obj(fs) => fs
                        .iter()
                        .filter(|(k, _)| k != "share_of_train")
                        .cloned()
                        .collect(),
                    _ => vec![],
                };
                let entry = Json::obj([
                    ("name", Json::str(name)),
                    (
                        "share_of_train",
                        stats.get("share_of_train").cloned().unwrap_or(Json::Null),
                    ),
                    ("features", Json::Obj(features)),
                ]);
                (c.clone(), entry)
            })
            .collect(),
    )
}

/// `clusters.py:naming_agreement`: agreement with hand-named clusters, weighted by share.
pub fn naming_agreement(naming: &Json, human: &Json) -> Json {
    let mut rows = Vec::new();
    if let Json::Obj(pairs) = human {
        for (c, want) in pairs {
            if let Some(entry) = naming.get(c) {
                let share = entry
                    .get("share_of_train")
                    .and_then(Json::as_f64)
                    .unwrap_or(0.0);
                let got = entry
                    .get("name")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string();
                let want = match want {
                    Json::Str(s) => s.clone(),
                    other => other.dumps(),
                };
                rows.push((share, c.clone(), got, want));
            }
        }
    }
    if rows.is_empty() {
        return Json::obj([
            ("n_named_clusters", Json::Int(0)),
            ("weighted_agreement", Json::Float(f64::NAN)),
            ("mismatches", Json::Arr(vec![])),
        ]);
    }
    let total: f64 = rows.iter().map(|r| r.0).sum();
    let agree: f64 = rows
        .iter()
        .map(|(w, _, got, want)| if got == want { *w } else { 0.0 })
        .sum();
    Json::obj([
        ("n_named_clusters", Json::Int(rows.len() as i64)),
        ("weighted_agreement", Json::Float(agree / total.max(1e-9))),
        (
            "mismatches",
            Json::Arr(
                rows.iter()
                    .filter(|(_, _, got, want)| got != want)
                    .map(|(_, c, got, want)| {
                        Json::Str(format!("cluster {c}: rule says {got}, human said {want}"))
                    })
                    .collect(),
            ),
        ),
    ])
}

/// `{n: list(c) for n, c in NAME_RULES}`: one entry per name, the last rule of a repeated name.
pub fn rules_json() -> Json {
    let mut pairs: Vec<(String, Json)> = Vec::new();
    for (name, crit) in NAME_RULES {
        let value = Json::Arr(
            crit.iter()
                .map(|(f, op, thr)| {
                    Json::Arr(vec![
                        Json::str(*f),
                        Json::Str(op.to_string()),
                        Json::Float(*thr),
                    ])
                })
                .collect(),
        );
        match pairs.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = value,
            None => pairs.push((name.to_string(), value)),
        }
    }
    Json::Obj(pairs)
}

/// `clusters.py:write_naming`: name the clusters, optionally score them against `labels.json`,
/// and write `naming.json`.
pub fn write_naming(clusters: &Path, agree_with: Option<&Path>) -> Result<Json> {
    let summary = read_json(&clusters.join("summary.json"))?;
    let naming = name_clusters(&summary);
    let mut report = vec![
        ("naming".to_string(), naming.clone()),
        ("rules".to_string(), rules_json()),
    ];
    if let Some(path) = agree_with {
        let labels = read_json(path)?;
        let human = labels.get("mapping").cloned().unwrap_or(Json::Obj(vec![]));
        report.push(("agreement".to_string(), naming_agreement(&naming, &human)));
    }
    let report = Json::Obj(report);
    write_text(&clusters.join("naming.json"), &report.dumps_indent(1))?;
    Ok(report)
}

fn read_json(path: &Path) -> Result<Json> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    Json::parse(&text).with_context(|| format!("{} is not JSON", path.display()))
}

// ------------------------------------------------------------------------------------ k-means

/// `sklearn.preprocessing.StandardScaler` fitted on `fit`: `(mean, scale)` per column. A column
/// is constant (scale 1) only under scikit-learn's `_is_constant_feature` bound, so a column of
/// rounding noise is still scaled to unit variance, as scikit-learn does.
pub fn standard_scaler(fit: &[[f64; 9]]) -> ([f64; 9], [f64; 9]) {
    let n = fit.len().max(1) as f64;
    let mut mean = [0.0; 9];
    let mut scale = [1.0; 9];
    for c in 0..9 {
        mean[c] = fit.iter().map(|r| r[c]).sum::<f64>() / n;
        // `_incremental_mean_and_var`: the two-pass variance with its correction term.
        let correction: f64 = fit.iter().map(|r| r[c] - mean[c]).sum();
        let squares: f64 = fit.iter().map(|r| (r[c] - mean[c]).powi(2)).sum();
        let var = (squares - correction * correction / n) / n;
        let eps = f64::EPSILON;
        let bound = n * eps * var + (n * mean[c] * eps).powi(2);
        scale[c] = if var <= bound { 1.0 } else { var.sqrt() };
    }
    (mean, scale)
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// A fitted k-means model: `k` centres in the feature space.
#[derive(Clone, Debug)]
pub struct KMeans {
    /// `k` centres of `d` dimensions.
    pub centers: Vec<Vec<f64>>,
    /// Sum of squared distances of the fitted samples to their centre.
    pub inertia: f64,
}

impl KMeans {
    /// `KMeans(k, n_init, random_state=seed).fit(x)`.
    pub fn fit(x: &[Vec<f64>], k: usize, n_init: usize, seed: u64) -> Result<Self> {
        ensure!(k >= 1, "k-means needs at least one cluster");
        ensure!(
            x.len() >= k,
            "n_samples={} should be >= n_clusters={k}",
            x.len()
        );
        let d = x[0].len();
        let mut mean_var = 0.0;
        for c in 0..d {
            let m = x.iter().map(|r| r[c]).sum::<f64>() / x.len() as f64;
            mean_var += x.iter().map(|r| (r[c] - m).powi(2)).sum::<f64>() / x.len() as f64;
        }
        let tol = 1e-4 * mean_var / d as f64;
        let mut master = Rng::new(seed);
        let mut best: Option<(KMeans, Vec<usize>)> = None;
        for _ in 0..n_init.max(1) {
            let mut rng = Rng::new(master.next_u64());
            let init = kmeans_plusplus(x, k, &mut rng);
            let (centers, labels, inertia) = lloyd(x, init, 300, tol);
            let better = match &best {
                None => true,
                Some((b, bl)) => inertia < b.inertia && !same_clustering(&labels, bl, k),
            };
            if better {
                best = Some((KMeans { centers, inertia }, labels));
            }
        }
        Ok(best.expect("at least one init").0)
    }

    /// `KMeans.predict`: the nearest centre of each sample (the lowest index on ties).
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter().map(|r| nearest(r, &self.centers).0).collect()
    }
}

fn nearest(r: &[f64], centers: &[Vec<f64>]) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for (i, c) in centers.iter().enumerate() {
        let d = sq_dist(r, c);
        if d < best.1 {
            best = (i, d);
        }
    }
    best
}

/// `sklearn.cluster._kmeans._kmeans_plusplus`: greedy k-means++ with `2 + ln k` local trials.
fn kmeans_plusplus(x: &[Vec<f64>], k: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let n = x.len();
    let trials = 2 + (k as f64).ln() as usize;
    let first = rng.integers(0, n as i64) as usize;
    let mut centers = vec![x[first].clone()];
    let mut closest: Vec<f64> = x.iter().map(|r| sq_dist(r, &x[first])).collect();
    let mut pot: f64 = closest.iter().sum();
    for _ in 1..k {
        let mut cumsum = Vec::with_capacity(n);
        let mut acc = 0.0;
        for v in &closest {
            acc += v;
            cumsum.push(acc);
        }
        let mut best: Option<(usize, f64, Vec<f64>)> = None;
        for _ in 0..trials {
            let r = rng.random() * pot;
            let cand = cumsum.partition_point(|c| *c < r).min(n - 1);
            let dist: Vec<f64> = x
                .iter()
                .zip(&closest)
                .map(|(row, c)| c.min(sq_dist(row, &x[cand])))
                .collect();
            let p: f64 = dist.iter().sum();
            if best.as_ref().map_or(true, |b| p < b.1) {
                best = Some((cand, p, dist));
            }
        }
        let (cand, p, dist) = best.expect("at least one trial");
        centers.push(x[cand].clone());
        closest = dist;
        pot = p;
    }
    centers
}

/// `_kmeans_single_lloyd`: `(centers, labels, inertia)`.
fn lloyd(
    x: &[Vec<f64>],
    mut centers: Vec<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, f64) {
    let k = centers.len();
    let d = x[0].len();
    let mut labels_old: Option<Vec<usize>> = None;
    let mut strict = false;
    let mut labels = vec![0; x.len()];
    for _ in 0..max_iter {
        let assigned: Vec<(usize, f64)> = x.iter().map(|r| nearest(r, &centers)).collect();
        labels = assigned.iter().map(|a| a.0).collect();
        let mut sums = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for (r, l) in x.iter().zip(&labels) {
            counts[*l] += 1;
            for (s, v) in sums[*l].iter_mut().zip(r) {
                *s += v;
            }
        }
        // `_relocate_empty_clusters_dense`: an empty cluster takes the point farthest from its
        // centre, which leaves its old cluster.
        let empty: Vec<usize> = (0..k).filter(|c| counts[*c] == 0).collect();
        if !empty.is_empty() {
            let mut far: Vec<usize> = (0..x.len()).collect();
            far.sort_by(|a, b| assigned[*b].1.total_cmp(&assigned[*a].1));
            for (slot, c) in empty.iter().enumerate() {
                let i = far[slot.min(far.len() - 1)];
                let old = labels[i];
                counts[old] -= 1;
                for (s, v) in sums[old].iter_mut().zip(&x[i]) {
                    *s -= v;
                }
                sums[*c] = x[i].clone();
                counts[*c] = 1;
            }
        }
        let mut shift = 0.0;
        let new: Vec<Vec<f64>> = (0..k)
            .map(|c| {
                if counts[c] == 0 {
                    centers[c].clone()
                } else {
                    sums[c].iter().map(|s| s / counts[c] as f64).collect()
                }
            })
            .collect();
        for (a, b) in new.iter().zip(&centers) {
            shift += sq_dist(a, b);
        }
        centers = new;
        if labels_old.as_ref() == Some(&labels) {
            strict = true;
            break;
        }
        if shift <= tol {
            break;
        }
        labels_old = Some(labels.clone());
    }
    if !strict {
        labels = x.iter().map(|r| nearest(r, &centers).0).collect();
    }
    let inertia = x
        .iter()
        .zip(&labels)
        .map(|(r, l)| sq_dist(r, &centers[*l]))
        .sum();
    (centers, labels, inertia)
}

/// `_is_same_clustering`: the two labelings are equal up to a permutation of the labels.
fn same_clustering(a: &[usize], b: &[usize], k: usize) -> bool {
    let mut map = vec![usize::MAX; k];
    for (x, y) in a.iter().zip(b) {
        if map[*x] == usize::MAX {
            map[*x] = *y;
        } else if map[*x] != *y {
            return false;
        }
    }
    true
}

// ------------------------------------------------------------------------------------ npz I/O

/// One `.npy` member: numpy's version 1.0 header for `descr` and `shape`, then `body`.
pub(crate) fn npy(descr: &str, shape: &[usize], body: Vec<u8>) -> Vec<u8> {
    let shape = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + body.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    out
}

/// A `float64` array.
pub(crate) fn npy_f64(shape: &[usize], v: &[f64]) -> Vec<u8> {
    npy(
        "<f8",
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

/// A `float32` array.
pub(crate) fn npy_f32(shape: &[usize], v: &[f32]) -> Vec<u8> {
    npy(
        "<f4",
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

/// An `int64` array.
pub(crate) fn npy_i64(shape: &[usize], v: &[i64]) -> Vec<u8> {
    npy(
        "<i8",
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

/// A numpy unicode array (`np.array(str)` for an empty `shape`).
pub(crate) fn npy_str(shape: &[usize], v: &[String]) -> Vec<u8> {
    let chars = v
        .iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0)
        .max(1);
    let mut body = Vec::new();
    for s in v {
        let mut n = 0;
        for ch in s.chars() {
            body.extend_from_slice(&(ch as u32).to_le_bytes());
            n += 1;
        }
        for _ in n..chars {
            body.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    npy(&format!("<U{chars}"), shape, body)
}

/// `np.savez_compressed(path, **members)`, members in the order given.
pub(crate) fn write_npz_members(path: &Path, members: &[(&str, Vec<u8>)]) -> Result<()> {
    use std::io::Write as _;
    use zip::write::SimpleFileOptions;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let file =
        std::fs::File::create(path).with_context(|| format!("cannot write {}", path.display()))?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, bytes) in members {
        zip.start_file(format!("{name}.npy"), options)?;
        zip.write_all(bytes)?;
    }
    zip.finish()?;
    Ok(())
}

/// An `.npz` read twice over: the arrays as [`crate::data::sessions::read_npz`] returns them
/// (floats narrowed to `f32`), and each member's raw `.npy` bytes for [`NpzFile::floats`].
pub(crate) struct NpzFile {
    /// The parsed arrays.
    pub arrays: Npz,
    raw: std::collections::BTreeMap<String, Vec<u8>>,
    path: PathBuf,
}

impl NpzFile {
    /// Reads the archive at `path`.
    pub fn read(path: &Path) -> Result<Self> {
        use std::io::Read as _;
        let file =
            std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let mut raw = std::collections::BTreeMap::new();
        let mut arrays = Npz::new();
        for i in 0..archive.len() {
            let mut member = archive.by_index(i)?;
            if member.is_dir() {
                continue;
            }
            let name = member.name().trim_end_matches(".npy").to_string();
            let mut bytes = Vec::new();
            member.read_to_end(&mut bytes)?;
            arrays.insert(
                name.clone(),
                crate::data::sessions::parse_npy(&bytes)
                    .with_context(|| format!("{}: {name}", path.display()))?,
            );
            raw.insert(name, bytes);
        }
        Ok(Self {
            arrays,
            raw,
            path: path.to_path_buf(),
        })
    }

    /// Whether member `key` exists.
    pub fn has(&self, key: &str) -> bool {
        self.arrays.contains_key(key)
    }

    /// The values of numeric member `key` at full precision (`float64` stays `f64`).
    pub fn floats(&self, key: &str) -> Result<Vec<f64>> {
        let bytes = self
            .raw
            .get(key)
            .with_context(|| format!("{} has no {key}", self.path.display()))?;
        decode_f64(bytes)
            .with_context(|| format!("{}: {key} is not a numeric array", self.path.display()))
    }
}

/// The values of a little-endian numeric `.npy` image as `f64`.
fn decode_f64(bytes: &[u8]) -> Option<Vec<f64>> {
    if bytes.len() < 10 || !bytes.starts_with(b"\x93NUMPY") {
        return None;
    }
    let (len, at) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        _ => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(bytes.get(at..at + len)?).ok()?;
    let descr = header.split("'descr':").nth(1)?.split('\'').nth(1)?;
    let body = &bytes[at + len..];
    let out: Vec<f64> = match descr {
        "<f8" => body
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect(),
        "<f4" => body
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap_or([0; 4]))))
            .collect(),
        "<i8" => body
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f64)
            .collect(),
        "<i4" => body
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes(c.try_into().unwrap_or([0; 4]))))
            .collect(),
        "|u1" | "|b1" => body.iter().map(|b| f64::from(*b)).collect(),
        _ => return None,
    };
    Some(out)
}

/// `track0_smoothed` / `track0_smoothed_conf` of a `tracks.npz`.
pub(crate) fn track0(z: &NpzFile, file: &Path) -> Result<(Vec<[f64; 2]>, Vec<f64>, usize)> {
    let kp = z
        .arrays
        .get("track0_smoothed")
        .with_context(|| format!("{} has no track0_smoothed (no track 0)", file.display()))?;
    ensure!(
        kp.shape.len() == 3 && kp.shape[1] == NUM_KEYPOINTS && kp.shape[2] == 2,
        "{}: track0_smoothed has shape {:?}, expected (T, {NUM_KEYPOINTS}, 2)",
        file.display(),
        kp.shape
    );
    let n = kp.shape[0];
    let raw = z.floats("track0_smoothed")?;
    let points = raw.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let conf = z.floats("track0_smoothed_conf")?;
    ensure!(
        conf.len() == n * NUM_KEYPOINTS,
        "track0_smoothed_conf does not match the keypoints"
    );
    Ok((points, conf, n))
}

// ------------------------------------------------------------------------------------ montages

/// `clusters.py:montage`: a strip of crops (`2 * half` square, scaled by `scale`) around the
/// animal at `frames`; a 10x10 black image when no frame can be read.
pub fn montage(
    src: &mut Source,
    centre: &[[f64; 2]],
    frames: &[usize],
    half: u32,
    scale: u32,
) -> Image {
    let side = 2 * half;
    let out_side = side * scale;
    let mut tiles: Vec<Vec<f32>> = Vec::new();
    for &t in frames {
        let Some((_, img)) = src.frames(&[t]).into_iter().next() else {
            continue;
        };
        let c = centre.get(t).copied().unwrap_or([0.0, 0.0]);
        // `np.nan_to_num(centre[t]).astype(int)` is the top-left of the crop in the padded frame.
        let cx = nan_to_num(c[0]) as i64;
        let cy = nan_to_num(c[1]) as i64;
        let mut crop = Vec::with_capacity((side * side) as usize);
        for v in 0..i64::from(side) {
            for u in 0..i64::from(side) {
                let x = (cx + u - i64::from(half)).clamp(0, i64::from(img.width) - 1);
                let y = (cy + v - i64::from(half)).clamp(0, i64::from(img.height) - 1);
                crop.push(img.pixel(x as u32, y as u32).round().clamp(0.0, 255.0));
            }
        }
        tiles.push(resize_linear(&crop, side, side, out_side, out_side));
    }
    if tiles.is_empty() {
        return Image::filled(10, 10, 0);
    }
    let width = out_side * tiles.len() as u32;
    let mut out = Image::filled(width, out_side, 0);
    for (i, tile) in tiles.iter().enumerate() {
        for y in 0..out_side {
            for x in 0..out_side {
                let v = tile[(y * out_side + x) as usize];
                out.set_pixel(
                    i as u32 * out_side + x,
                    y,
                    v.round().clamp(0.0, 255.0) as u8,
                );
            }
        }
    }
    out
}

pub(crate) fn save_gray_png(path: &Path, img: &Image) -> Result<()> {
    let buf = image::GrayImage::from_raw(img.width, img.height, img.pixels.clone())
        .context("image buffer size")?;
    buf.save(path)
        .with_context(|| format!("cannot write {}", path.display()))
}

// ------------------------------------------------------------------------------------ cluster

/// `clusters.py:cluster_tracks`: cluster the kinematics of `tracks.npz`; writes `clusters.npz`,
/// `summary.json` and `cluster<c>.png` montages to `out`; returns the summary.
pub fn cluster_tracks(
    tracks: &Path,
    video: &Path,
    out: &Path,
    fps: f64,
    train_range: (f64, f64),
    k: usize,
) -> Result<Json> {
    let z = NpzFile::read(tracks)?;
    let (kp, cf, n) = track0(&z, tracks)?;
    let times = if z.has("times") {
        Some(z.floats("times")?)
    } else {
        None
    };
    let feats = kinematics(&kp, &cf, fps, times.as_deref());
    let a = ((train_range.0 * n as f64) as i64).clamp(0, n as i64) as usize;
    let b = ((train_range.1 * n as f64) as i64).clamp(0, n as i64) as usize;
    ensure!(
        b > a,
        "the training range {train_range:?} holds no frame of {n}"
    );
    let (mean, scale) = standard_scaler(&feats[a..b]);
    let scaled: Vec<Vec<f64>> = feats
        .iter()
        .map(|r| (0..9).map(|c| (r[c] - mean[c]) / scale[c]).collect())
        .collect();
    let km = KMeans::fit(&scaled[a..b], k, 10, 0)?;
    let lab: Vec<i64> = km.predict(&scaled).into_iter().map(|l| l as i64).collect();
    std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    let centre: Vec<[f64; 2]> = (0..n)
        .map(|t| kp[t * NUM_KEYPOINTS + index("spine_mid")])
        .collect();
    let mut source = Source::open(video);
    let mut rng = Rng::new(0);
    let train_lab = &lab[a..b];
    let mut summary = Vec::new();
    for c in 0..k as i64 {
        let sel: Vec<usize> = (0..train_lab.len())
            .filter(|i| train_lab[*i] == c)
            .collect();
        let share = sel.len() as f64 / train_lab.len() as f64;
        let all_bouts: Vec<(usize, usize)> = bouts(train_lab)
            .into_iter()
            .filter(|(_, _, l)| *l == c)
            .map(|(s, e, _)| (s, e))
            .collect();
        let mut long: Vec<(usize, usize)> = all_bouts
            .iter()
            .copied()
            .filter(|(s, e)| (e - s) as f64 >= fps)
            .collect();
        if long.is_empty() {
            long = all_bouts
                .iter()
                .copied()
                .filter(|(s, e)| (e - s) as f64 >= 0.3 * fps)
                .collect();
        }
        let mut stats = vec![("share_of_train".to_string(), Json::Float(share))];
        for (i, f) in FEATURES.iter().enumerate() {
            let mut v: Vec<f64> = sel.iter().map(|j| feats[a + j][i]).collect();
            stats.push((f.to_string(), Json::Float(py_round(median(&mut v), 3))));
        }
        summary.push((c.to_string(), Json::Obj(stats)));
        let picks = rng.sample_without_replacement(&(0..long.len()).collect::<Vec<_>>(), 3);
        let rows: Vec<Image> = picks
            .iter()
            .map(|p| {
                let (s, e) = long[*p];
                let lo = (s + a) as f64;
                let hi = (e as f64).min(s as f64 + 2.0 * fps) + a as f64 - 1.0;
                let frames = crate::cli::sources::linspace_int(lo, hi, 6);
                montage(&mut source, &centre, &frames, 45, 2)
            })
            .collect();
        if !rows.is_empty() {
            let wmax = rows.iter().map(|r| r.width).max().unwrap_or(0);
            let height: u32 = rows.iter().map(|r| r.height + 4).sum();
            let mut sheet = Image::filled(wmax, height, 0);
            let mut y0 = 0;
            for r in &rows {
                for y in 0..r.height {
                    for x in 0..r.width {
                        sheet.set_pixel(x, y0 + 2 + y, r.pixel(x, y));
                    }
                }
                y0 += r.height + 4;
            }
            save_gray_png(&out.join(format!("cluster{c}.png")), &sheet)?;
        }
    }
    let flat: Vec<f64> = feats.iter().flat_map(|r| r.iter().copied()).collect();
    write_npz_members(
        &out.join("clusters.npz"),
        &[
            ("labels", npy_i64(&[n], &lab)),
            ("features", npy_f64(&[n, 9], &flat)),
            ("fps", npy_f64(&[], &[fps])),
            (
                "train_range",
                npy_f64(&[2], &[train_range.0, train_range.1]),
            ),
            ("tracks", npy_str(&[], &[tracks.display().to_string()])),
        ],
    )?;
    let summary = Json::Obj(summary);
    write_text(&out.join("summary.json"), &summary.dumps_indent(1))?;
    Ok(summary)
}

/// `np.median` (NaN for an empty selection, as numpy returns with a warning).
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() || v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        (v[m / 2 - 1] + v[m / 2]) / 2.0
    }
}

// ------------------------------------------------------------------------------------ label

/// `clusters.py:parse_mapping`: `["0=resting", "2=grooming"]` -> ordered `(cluster, behavior)`
/// pairs with `dict` semantics (a repeated cluster keeps its first place and its last name).
pub fn parse_mapping(specs: &[String]) -> Result<Vec<(i64, String)>> {
    let mut out: Vec<(i64, String)> = Vec::new();
    for s in specs {
        let parts: Vec<&str> = s.split('=').collect();
        ensure!(
            parts.len() == 2,
            "not enough values to unpack: {s:?} is not cluster=behavior"
        );
        let k: i64 = parts[0]
            .trim()
            .parse()
            .with_context(|| format!("invalid literal for int() with base 10: {:?}", parts[0]))?;
        match out.iter_mut().find(|(c, _)| *c == k) {
            Some(slot) => slot.1 = parts[1].to_string(),
            None => out.push((k, parts[1].to_string())),
        }
    }
    Ok(out)
}

/// `clusters.py:label_clusters`: train/held-out session files from a `cluster -> behavior` map;
/// returns the class shares.
pub fn label_clusters(
    clusters: &Path,
    mapping: &[(i64, String)],
    out: &Path,
    name: &str,
    view: &str,
    min_bout: f64,
) -> Result<Json> {
    let file = clusters.join("clusters.npz");
    let c = NpzFile::read(&file)?;
    let lab = c
        .arrays
        .get("labels")
        .with_context(|| format!("{} has no labels", file.display()))?
        .ints()
        .context("labels is not numeric")?;
    let fps = *c.floats("fps")?.first().context("fps is empty")?;
    let catalog = Catalog::load();
    let mut unknown: Vec<&str> = mapping
        .iter()
        .map(|(_, v)| v.as_str())
        .filter(|v| catalog.canonical(v).is_none())
        .collect();
    unknown.sort_unstable();
    if !unknown.is_empty() {
        let list: Vec<String> = unknown.iter().map(|u| format!("'{u}'")).collect();
        bail!(
            "not in the behavior catalogue (names or synonyms): [{}]",
            list.join(", ")
        );
    }
    let mut names: Vec<String> = Vec::new();
    for (_, v) in mapping {
        if !names.contains(v) {
            names.push(v.clone());
        }
    }
    let mut beh: Vec<i64> = lab
        .iter()
        .map(|x| match mapping.iter().find(|(c, _)| c == x) {
            Some((_, v)) => names.iter().position(|n| n == v).unwrap_or(0) as i64,
            None => IGNORE,
        })
        .collect();
    let min_len = (min_bout * fps).round_ties_even() as i64;
    for (s, e, _) in bouts(&beh.clone()) {
        if ((e - s) as i64) < min_len && s > 0 {
            let prev = beh[s - 1];
            beh[s..e].fill(prev);
        }
    }
    let tracks_path = match c.arrays.get("tracks").map(|t| &t.data) {
        Some(NpyData::Text(v)) if !v.is_empty() => PathBuf::from(&v[0]),
        _ => bail!("{} names no tracks file", file.display()),
    };
    let z = NpzFile::read(&tracks_path)?;
    let (kp, cf, _) = track0(&z, &tracks_path)?;
    let n = lab.len();
    let range = c.floats("train_range")?;
    ensure!(range.len() == 2, "train_range must hold two fractions");
    let a = ((range[0] * n as f64) as usize).min(n);
    let b = ((range[1] * n as f64) as usize).min(n);
    let k = NUM_KEYPOINTS;
    ensure!(
        kp.len() >= n * k,
        "{} is shorter than the clusters",
        tracks_path.display()
    );
    for (part, lo, hi) in [("train", a, b), ("heldout", b, n)] {
        let hi = hi.max(lo);
        let coords: Vec<f32> = kp[lo * k..hi * k]
            .iter()
            .flat_map(|p| [p[0] as f32, p[1] as f32])
            .collect();
        let conf: Vec<f32> = cf[lo * k..hi * k].iter().map(|v| *v as f32).collect();
        let animal = format!("{name}_{part}");
        write_npz_members(
            &out.join(part).join(format!("{animal}.npz")),
            &[
                ("coords", npy_f32(&[hi - lo, k, 2], &coords)),
                ("conf", npy_f32(&[hi - lo, k], &conf)),
                ("labels", npy_i64(&[hi - lo], &beh[lo..hi])),
                ("fps", npy_f64(&[], &[fps])),
                ("animal", npy_str(&[], &[animal.clone()])),
                ("view", npy_str(&[], &[view.to_string()])),
                ("label_names", npy_str(&[names.len()], &names)),
            ],
        )?;
    }
    let mut classes: Vec<i64> = beh.iter().copied().filter(|v| *v != IGNORE).collect();
    classes.sort_unstable();
    classes.dedup();
    let share = Json::Obj(
        classes
            .iter()
            .map(|k| {
                let frac = beh.iter().filter(|v| *v == k).count() as f64 / beh.len() as f64;
                (names[*k as usize].clone(), Json::Float(py_round(frac, 3)))
            })
            .collect(),
    );
    let labels = Json::obj([
        (
            "mapping",
            Json::Obj(
                mapping
                    .iter()
                    .map(|(c, v)| (c.to_string(), Json::str(v.clone())))
                    .collect(),
            ),
        ),
        ("share", share.clone()),
        ("min_bout_s", Json::Float(min_bout)),
    ]);
    std::fs::create_dir_all(out).with_context(|| format!("cannot create {}", out.display()))?;
    write_text(&out.join("labels.json"), &labels.dumps_indent(1))?;
    Ok(share)
}

#[cfg(test)]
pub(crate) mod tests;
