//! Per-frame measurements of the test animal, computed once and shared by every rule.
//!
//! Positions come from the 3D keypoints (box units, z up) when at least `qa.min_valid3d` of the
//! frames have them; otherwise from the top view in pixels, where heights are unknown. Distances
//! are divided by the body length (BL), so the rules read the same in both.

use anyhow::{bail, Result};

use super::config::AssayConfig;
use super::input::Input;
use crate::multiview::dense::Vec3;

/// The test animal's keypoints over the video.
#[derive(Clone, Debug)]
pub struct Pose {
    /// True for box units with heights; false for top-view pixels (z = 0).
    pub dims3: bool,
    pub k: usize,
    /// `(n_frames, K)` positions, NaN where missing.
    pub xyz: Vec<Vec3>,
    /// Share of frames with a 3D `spine_mid`.
    pub valid3d: f64,
}

impl Pose {
    pub fn at(&self, t: usize, k: Option<usize>) -> Option<Vec3> {
        let p = *self.xyz.get(t * self.k + k?)?;
        p.iter().all(|v| v.is_finite()).then_some(p)
    }

    pub fn frames(&self) -> usize {
        self.xyz.len() / self.k.max(1)
    }
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn norm(a: Vec3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

pub fn dist(a: Vec3, b: Vec3) -> f64 {
    norm(sub(a, b))
}

/// Angle between two vectors in degrees.
pub fn angle(a: Vec3, b: Vec3) -> Option<f64> {
    let (na, nb) = (norm(a), norm(b));
    (na > 0.0 && nb > 0.0).then(|| {
        let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
        c.clamp(-1.0, 1.0).acos().to_degrees()
    })
}

/// Linear interpolation over gaps of at most `max_gap` frames, per keypoint.
pub fn fill_gaps(xyz: &mut [Vec3], k: usize, max_gap: usize) {
    let n = xyz.len() / k.max(1);
    for j in 0..k {
        let valid = |t: usize, xyz: &[Vec3]| xyz[t * k + j].iter().all(|v| v.is_finite());
        let mut last: Option<usize> = None;
        for t in 0..n {
            if !valid(t, xyz) {
                continue;
            }
            if let Some(l) = last {
                let gap = t - l - 1;
                if gap > 0 && gap <= max_gap {
                    let (a, b) = (xyz[l * k + j], xyz[t * k + j]);
                    for g in 1..=gap {
                        let w = g as f64 / (gap + 1) as f64;
                        xyz[(l + g) * k + j] = [
                            a[0] + (b[0] - a[0]) * w,
                            a[1] + (b[1] - a[1]) * w,
                            a[2] + (b[2] - a[2]) * w,
                        ];
                    }
                }
            }
            last = Some(t);
        }
    }
}

/// The test animal's pose, gaps up to `gap.max_fill_s` filled.
pub fn pose(input: &Input, cfg: &AssayConfig) -> Result<Pose> {
    let k = input.kpts();
    let n = input.n_frames;
    let spine = input.index("spine_mid");
    let valid3d = match (&input.world, spine) {
        (Some(w), Some(s)) if n > 0 => {
            (0..n)
                .filter(|t| {
                    w.get(t * k + s)
                        .is_some_and(|p| p.iter().all(|v| v.is_finite()))
                })
                .count() as f64
                / n as f64
        }
        _ => 0.0,
    };
    let (dims3, mut xyz) = match &input.world {
        Some(w) if valid3d >= cfg.min_valid3d => {
            let mut w = w.clone();
            w.resize(n * k, [f64::NAN; 3]);
            (true, w)
        }
        _ => {
            let Some(top) = &input.top else {
                bail!("no track of the test animal");
            };
            let mut xyz = vec![[f64::NAN; 3]; n * k];
            for t in 0..n {
                for j in 0..k {
                    if let Some((p, _)) = top.point(t, j, k) {
                        xyz[t * k + j] = [p[0], p[1], 0.0];
                    }
                }
            }
            (false, xyz)
        }
    };
    let max_gap = (cfg.features.max_fill_s * input.fps).round() as usize;
    fill_gaps(&mut xyz, k, max_gap);
    Ok(Pose {
        dims3,
        k,
        xyz,
        valid3d,
    })
}

/// Savitzky–Golay first derivative (quadratic fit; the same weights as a linear fit) over an
/// odd window; `None` where the window is incomplete.
pub fn derivative(values: &[Option<Vec3>], window: usize, fps: f64) -> Vec<Option<Vec3>> {
    let h = (window.max(3) | 1) / 2;
    let norm: f64 = (1..=h).map(|i| 2.0 * (i * i) as f64).sum();
    (0..values.len())
        .map(|t| {
            if t < h || t + h >= values.len() {
                return None;
            }
            let mut d = [0.0; 3];
            for i in 1..=h {
                let (a, b) = (values[t + i]?, values[t - i]?);
                for c in 0..3 {
                    d[c] += i as f64 * (a[c] - b[c]);
                }
            }
            Some(d.map(|v| v / norm * fps))
        })
        .collect()
}

pub fn percentile(values: &[f64], pct: f64) -> Option<f64> {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let pos = (pct / 100.0).clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
}

pub fn median(values: &[f64]) -> Option<f64> {
    percentile(values, 50.0)
}

/// The section-5 features, one value per frame.
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// Body length, in the pose's units.
    pub bl: f64,
    /// Body speed (BL/s).
    pub speed: Vec<Option<f64>>,
    /// Nose-to-tail-base length / BL: about 1 stretched, below 0.8 hunched.
    pub compact: Vec<Option<f64>>,
    /// Nose height / BL; 3D only.
    pub nose_z: Vec<Option<f64>>,
    /// Nearest front paw to the nose / BL.
    pub paw_d: Vec<Option<f64>>,
    /// Fastest front paw relative to the body (BL/s).
    pub paw_speed: Vec<Option<f64>>,
    /// Nose to the nearest shoulder, hip, spine_lower / BL.
    pub nose_body: Vec<Option<f64>>,
    /// Nose to tail_base / BL.
    pub nose_tail: Vec<Option<f64>>,
    /// Nearest front paw to an ear / BL.
    pub paw_ear: Vec<Option<f64>>,
    /// Rate of change of `paw_d` (BL/s); negative when a paw moves toward the nose.
    pub paw_d_rate: Vec<Option<f64>>,
    /// Head direction: nose − mean(ears).
    pub head: Vec<Option<Vec3>>,
    /// Share of outline pixels moving; filled in by [`super::motion`] when the video is at hand.
    pub motion: Vec<Option<f64>>,
}

pub fn compute(pose: &Pose, names: &[String], cfg: &AssayConfig, fps: f64) -> Result<Features> {
    let idx = |n: &str| names.iter().position(|m| m == n);
    let n = pose.frames();
    let at = |t: usize, name: &str| pose.at(t, idx(name));
    let window = ((cfg.features.speed_window_s * fps).round() as usize).max(3) | 1;
    let spine: Vec<Option<Vec3>> = (0..n).map(|t| at(t, "spine_mid")).collect();
    let raw_speed: Vec<Option<f64>> = derivative(&spine, window, fps)
        .into_iter()
        .map(|d| d.map(norm))
        .collect();
    let length: Vec<Option<f64>> = (0..n)
        .map(|t| Some(dist(at(t, "nose")?, at(t, "tail_base")?)))
        .collect();
    // Body length over walking frames: fast, and with the nose low (3D).
    let speeds: Vec<f64> = raw_speed.iter().flatten().copied().collect();
    let fast = percentile(&speeds, cfg.features.walk_speed_pct);
    let heights: Vec<f64> = (0..n).filter_map(|t| at(t, "nose").map(|p| p[2])).collect();
    let low = if pose.dims3 {
        percentile(&heights, cfg.features.walk_nose_pct)
    } else {
        None
    };
    let walking: Vec<f64> = (0..n)
        .filter(|t| fast.is_some_and(|f| raw_speed[*t].is_some_and(|s| s > f)))
        .filter(|t| low.map_or(true, |l| at(*t, "nose").is_some_and(|p| p[2] < l)))
        .filter_map(|t| length[t])
        .collect();
    let all: Vec<f64> = length.iter().flatten().copied().collect();
    let bl = median(&walking).or_else(|| median(&all)).unwrap_or(0.0);
    if bl.is_nan() || bl <= 0.0 {
        bail!("no frame shows both the nose and the tail base, so the body length is unknown");
    }
    let per = |f: &dyn Fn(usize) -> Option<f64>| (0..n).map(f).collect::<Vec<_>>();
    let paws = ["left_front_paw", "right_front_paw"];
    let min_of = |t: usize, from: &str, to: &[&str]| {
        let a = at(t, from)?;
        to.iter()
            .filter_map(|b| at(t, b).map(|b| dist(a, b)))
            .reduce(f64::min)
    };
    let paw_d = per(&|t| min_of(t, "nose", &paws).map(|d| d / bl));
    let rel: Vec<Vec<Option<Vec3>>> = paws
        .iter()
        .map(|p| {
            (0..n)
                .map(|t| Some(sub(at(t, p)?, at(t, "spine_mid")?)))
                .collect()
        })
        .collect();
    let paw_window = ((cfg.features.paw_window_s * fps).round() as usize).max(3) | 1;
    let rel_speed: Vec<Vec<Option<f64>>> = rel
        .iter()
        .map(|r| {
            derivative(r, paw_window, fps)
                .into_iter()
                .map(|d| d.map(norm))
                .collect()
        })
        .collect();
    let paw_d_vec: Vec<Option<Vec3>> = paw_d.iter().map(|d| d.map(|d| [d, 0.0, 0.0])).collect();
    let paw_d_rate = derivative(&paw_d_vec, window, fps)
        .into_iter()
        .map(|d| d.map(|d| d[0]))
        .collect();
    Ok(Features {
        bl,
        speed: raw_speed.iter().map(|s| s.map(|s| s / bl)).collect(),
        compact: length.iter().map(|l| l.map(|l| l / bl)).collect(),
        nose_z: per(&|t| {
            pose.dims3
                .then(|| at(t, "nose").map(|p| p[2] / bl))
                .flatten()
        }),
        paw_d,
        paw_speed: per(&|t| {
            rel_speed
                .iter()
                .filter_map(|s| s[t])
                .reduce(f64::max)
                .map(|s| s / bl)
        }),
        nose_body: per(&|t| {
            min_of(
                t,
                "nose",
                &[
                    "left_shoulder",
                    "right_shoulder",
                    "left_hip",
                    "right_hip",
                    "spine_lower",
                ],
            )
            .map(|d| d / bl)
        }),
        nose_tail: per(&|t| Some(dist(at(t, "nose")?, at(t, "tail_base")?) / bl)),
        paw_ear: per(&|t| {
            paws.iter()
                .filter_map(|p| min_of(t, p, &["left_ear", "right_ear"]))
                .reduce(f64::min)
                .map(|d| d / bl)
        }),
        paw_d_rate,
        head: per_vec(n, &|t| {
            let (l, r) = (at(t, "left_ear")?, at(t, "right_ear")?);
            let mid = [
                (l[0] + r[0]) / 2.0,
                (l[1] + r[1]) / 2.0,
                (l[2] + r[2]) / 2.0,
            ];
            Some(sub(at(t, "nose")?, mid))
        }),
        motion: vec![None; n],
    })
}

fn per_vec(n: usize, f: &dyn Fn(usize) -> Option<Vec3>) -> Vec<Option<Vec3>> {
    (0..n).map(f).collect()
}

/// The median of a feature over a centred window, if at least half the window is known.
pub fn window_median(values: &[Option<f64>], t: usize, half: usize) -> Option<f64> {
    let lo = t.saturating_sub(half);
    let hi = (t + half + 1).min(values.len());
    let known: Vec<f64> = values[lo..hi].iter().flatten().copied().collect();
    (known.len() * 2 >= hi - lo)
        .then(|| median(&known))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_fill_up_to_the_limit_and_derivatives_are_exact_on_lines() {
        let nan = [f64::NAN; 3];
        let mut xyz = vec![
            [0.0; 3],
            nan,
            nan,
            [3.0, 0.0, 0.0],
            nan,
            nan,
            nan,
            [7.0, 0.0, 0.0],
        ];
        fill_gaps(&mut xyz, 1, 2);
        assert_eq!(xyz[1], [1.0, 0.0, 0.0]);
        assert_eq!(xyz[2], [2.0, 0.0, 0.0]);
        assert!(xyz[5][0].is_nan());
        let line: Vec<Option<Vec3>> = (0..9).map(|t| Some([2.0 * t as f64, 0.0, 0.0])).collect();
        let d = derivative(&line, 5, 10.0);
        assert!(d[0].is_none());
        assert!((d[4].unwrap()[0] - 20.0).abs() < 1e-9);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), Some(2.5));
        assert_eq!(
            angle([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).map(|a| a.round()),
            Some(90.0)
        );
    }
}
