//! Per-frame kinematic features of one tracked animal.
//!
//! Port of `animalpose.behavior.catalog.body_kinematics`, the only part of that
//! module that is pure geometry (the catalogue itself stays in Python).

/// Canonical index of `nose`.
pub const NOSE: usize = 0;
/// Canonical index of `head_center`.
pub const HEAD_CENTER: usize = 5;
/// Canonical index of `neck`.
pub const NECK: usize = 6;
/// Canonical index of `spine_upper`.
pub const SPINE_UPPER: usize = 7;
/// Canonical index of `spine_mid`.
pub const SPINE_MID: usize = 8;
/// Canonical index of `spine_lower`.
pub const SPINE_LOWER: usize = 9;
/// Canonical index of `tail_base`.
pub const TAIL_BASE: usize = 10;

/// The `core` spine points whose mean is the body centre, in the order Python
/// averages them.
pub const CORE: [usize; 6] = [
    HEAD_CENTER,
    NECK,
    SPINE_UPPER,
    SPINE_MID,
    SPINE_LOWER,
    TAIL_BASE,
];

/// Confidence at or above which a keypoint counts towards the centre.
pub const CORE_CONF_THRESH: f64 = 0.3;

/// Per-frame centre, unit body axis, body length (px) and speed (body lengths/s).
///
/// Port of the dict `body_kinematics` returns; `length` is the single reference
/// length (the median nose-tail distance) Python stores under that key.
#[derive(Clone, Debug, PartialEq)]
pub struct Kinematics {
    /// Body centre per frame, `(T, 2)`.
    pub centre: Vec<[f64; 2]>,
    /// Unit nose-to-tail axis per frame, `(T, 2)`.
    pub axis: Vec<[f64; 2]>,
    /// Reference body length in pixels (`np.nanmedian` of the nose-tail distance).
    pub length: f64,
    /// Smoothed speed in body lengths per second, `(T,)`.
    pub speed: Vec<f64>,
    /// Unsmoothed velocity in body lengths per second, `(T, 2)`.
    pub velocity: Vec<[f64; 2]>,
    /// Nose per frame, `(T, 2)`.
    pub nose: Vec<[f64; 2]>,
    /// Tail base per frame, `(T, 2)`.
    pub tail: Vec<[f64; 2]>,
    /// `spine_mid` per frame, `(T, 2)`.
    pub body: Vec<[f64; 2]>,
}

/// Per-frame centre, unit body axis, body length (px) and speed (body lengths/s).
///
/// Port of `animalpose.behavior.catalog.body_kinematics`.
///
/// * `kp`: keypoints `(T, K, 2)` row-major, image pixels; unmatched frames hold
///   `NaN` exactly like the Python pipeline.
/// * `conf`: confidence `(T, K)` row-major.
/// * `times`: presentation time per frame in seconds; without it frames are
///   `1 / fps` apart.
///
/// // differs from Python: `round(0.3 * rate)` rounds half away from zero,
/// // where Python's banker's rounding would pick the even neighbour.
pub fn body_kinematics(
    kp: &[f64],
    conf: &[f64],
    frames: usize,
    keypoints: usize,
    fps: f64,
    times: Option<&[f64]>,
) -> Kinematics {
    let at = |t: usize, k: usize| -> [f64; 2] {
        let i = (t * keypoints + k) * 2;
        [kp[i], kp[i + 1]]
    };
    let conf_at = |t: usize, k: usize| -> f64 { conf[t * keypoints + k] };

    // Weighted mean of the core spine points over the confident ones.
    let mut centre = vec![[0.0; 2]; frames];
    for (t, centre_at) in centre.iter_mut().enumerate() {
        let mut sum = [0.0; 2];
        let mut weight = 0.0;
        for &k in &CORE {
            let w = f64::from(conf_at(t, k) >= CORE_CONF_THRESH);
            let p = at(t, k);
            // `np.nan_to_num`: NaN becomes 0, so an unseen point contributes nothing.
            for d in 0..2 {
                sum[d] += nan_to_num(p[d]) * w;
            }
            weight += w;
        }
        let denom = weight.max(1e-6);
        *centre_at = [sum[0] / denom, sum[1] / denom];
    }

    let nose: Vec<[f64; 2]> = (0..frames).map(|t| at(t, NOSE)).collect();
    let tail: Vec<[f64; 2]> = (0..frames).map(|t| at(t, TAIL_BASE)).collect();
    let body: Vec<[f64; 2]> = (0..frames).map(|t| at(t, SPINE_MID)).collect();
    let length: Vec<f64> = (0..frames)
        .map(|t| {
            let dx = nose[t][0] - tail[t][0];
            let dy = nose[t][1] - tail[t][1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    let ref_length = if length.iter().any(|v| v.is_finite()) {
        nan_median(&length)
    } else {
        1.0
    };
    let axis: Vec<[f64; 2]> = (0..frames)
        .map(|t| {
            let denom = nan_to_num(length[t]) + 1e-6;
            [
                nan_to_num(nose[t][0] - tail[t][0]) / denom,
                nan_to_num(nose[t][1] - tail[t][1]) / denom,
            ]
        })
        .collect();

    let (velocity, rate) = if frames > 1 {
        let t: Vec<f64> = match times {
            Some(times) => times.iter().take(frames).copied().collect(),
            None => (0..frames).map(|i| i as f64 / fps).collect(),
        };
        let v = gradient(&centre, &t, ref_length.max(1e-6));
        let dt = median_diff(&t);
        (v, 1.0 / dt.max(1e-6))
    } else {
        (vec![[0.0; 2]; frames], fps)
    };

    // ~0.3 s of speed smoothing.
    let k = (0.3 * rate).round().max(1.0) as usize;
    let speed = convolve_same(
        &velocity
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .collect::<Vec<_>>(),
        k,
    );

    Kinematics {
        centre,
        axis: axis
            .iter()
            .map(|a| [nan_to_num(a[0]), nan_to_num(a[1])])
            .collect(),
        length: ref_length,
        speed,
        velocity,
        nose,
        tail,
        body,
    }
}

/// `np.nan_to_num` for a scalar: `NaN` becomes `0.0`, infinities saturate.
pub fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        -f64::MAX
    } else {
        v
    }
}

/// Median over the finite entries (port of `np.nanmedian` on a 1-D array).
pub fn nan_median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Median of the successive differences of an increasing sequence
/// (`np.median(np.diff(t))`).
pub fn median_diff(times: &[f64]) -> f64 {
    if times.len() < 2 {
        return f64::NAN;
    }
    let d: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    nan_median(&d)
}

/// Derivative of `(T, 2)` samples at times `t`, divided by the reference length
/// (port of `np.gradient(values, t, axis=0) / max(ref, 1e-6)`).
///
/// Second-order accurate central differences in the interior, first-order at
/// the two ends, with the unequal spacing weights numpy uses.
pub fn gradient(values: &[[f64; 2]], t: &[f64], reference: f64) -> Vec<[f64; 2]> {
    let n = values.len().min(t.len());
    let mut out = vec![[0.0; 2]; n];
    if n < 2 {
        return out;
    }
    let scale = 1.0 / reference.max(1e-6);
    for d in 0..2 {
        let f = |i: usize| values[i][d];
        out[0][d] = (f(1) - f(0)) / (t[1] - t[0]);
        out[n - 1][d] = (f(n - 1) - f(n - 2)) / (t[n - 1] - t[n - 2]);
        for i in 1..n - 1 {
            let ds = t[i] - t[i - 1];
            let dd = t[i + 1] - t[i];
            out[i][d] = (ds * ds * f(i + 1) + (dd * dd - ds * ds) * f(i) - dd * dd * f(i - 1))
                / (ds * dd * (ds + dd));
        }
    }
    for v in &mut out {
        v[0] *= scale;
        v[1] *= scale;
    }
    out
}

/// Moving average of `k` samples, `np.convolve(a, np.ones(k) / k, mode="same")`:
/// the full convolution cropped from `(k - 1) / 2`, i.e. the window centred on
/// the current sample and truncated at the ends.
pub fn convolve_same(values: &[f64], k: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let k = k.max(1);
    let start = (k - 1) / 2;
    let mut out = vec![0.0; values.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for j in 0..k {
            let idx = i as isize + start as isize - j as isize;
            if idx >= 0 && (idx as usize) < values.len() {
                sum += values[idx as usize];
            }
        }
        *o = sum / k as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight mouse of length 10 px facing +x, the `_mouse` helper of
    /// `tests/test_behavior_catalog.py`.
    fn mouse(frames: usize, x0: f64, vx: f64) -> (Vec<f64>, Vec<f64>) {
        let k = 25;
        let mut kp = vec![0.0; frames * k * 2];
        let offsets = [
            (NOSE, 5.0),
            (HEAD_CENTER, 3.0),
            (NECK, 2.0),
            (SPINE_UPPER, 1.0),
            (SPINE_MID, 0.0),
            (SPINE_LOWER, -2.0),
            (TAIL_BASE, -5.0),
        ];
        for t in 0..frames {
            let cx = x0 + vx * t as f64;
            for (k, dx) in offsets {
                let i = (t * 25 + k) * 2;
                kp[i] = cx + dx;
                kp[i + 1] = 0.0;
            }
        }
        let conf = vec![1.0; frames * 25];
        (kp, conf)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn kinematics_report_length_axis_and_running_speed() {
        // Reference values from animalpose.behavior.catalog.body_kinematics.
        let (kp, conf) = mouse(30, 0.0, 1.0);
        let kin = body_kinematics(&kp, &conf, 30, 25, 30.0, None);
        assert!(close(kin.length, 10.0, 1e-9));
        assert!(
            close(kin.centre[0][0], -1.0 / 6.0, 1e-9),
            "{:?}",
            kin.centre[0]
        );
        assert!(close(kin.centre[29][0], 28.0 + 5.0 / 6.0, 1e-9));
        assert!(close(kin.axis[0][0], 10.0 / (10.0 + 1e-6), 1e-9));
        assert!(close(kin.axis[0][1], 0.0, 1e-12));
        // 1 px/frame at 30 fps with a 10 px body = 3 body lengths/s.
        for (t, speed) in kin.speed.iter().enumerate().take(25).skip(5) {
            assert!(close(*speed, 3.0, 1e-9), "t={t} {speed}");
        }
        // Edges of the 9-tap moving average truncate: 5 and 8 valid samples.
        assert!(
            close(kin.speed[0], 3.0 * 5.0 / 9.0, 1e-9),
            "{}",
            kin.speed[0]
        );
        assert!(
            close(kin.speed[26], 3.0 * 8.0 / 9.0, 1e-9),
            "{}",
            kin.speed[26]
        );
        assert!(close(kin.velocity[10][0], 3.0, 1e-9));
        assert!(close(kin.velocity[10][1], 0.0, 1e-12));
        assert!(close(kin.nose[0][0], 5.0, 1e-12));
        assert!(close(kin.tail[0][0], -5.0, 1e-12));
        assert!(close(kin.body[0][0], 0.0, 1e-12));
    }

    #[test]
    fn kinematics_use_real_timestamps() {
        // Port of tests/test_behavior_catalog.py::test_kinematics_use_real_timestamps.
        let (kp, conf) = mouse(30, 0.0, 1.0);
        let even = body_kinematics(&kp, &conf, 30, 25, 30.0, None);
        let times: Vec<f64> = (0..30).map(|i| i as f64 * 2.0 / 30.0).collect();
        let slow = body_kinematics(&kp, &conf, 30, 25, 30.0, Some(&times));
        let median = |v: &[f64]| nan_median(&v[5..v.len() - 5]);
        assert!(
            close(median(&slow.speed) * 2.0, median(&even.speed), 1e-6),
            "{} vs {}",
            median(&slow.speed),
            median(&even.speed)
        );
    }

    #[test]
    fn missing_keypoints_and_confidence_mask_the_centre() {
        let (mut kp, conf) = mouse(4, 0.0, 1.0);
        // Wipe the whole pose of frame 2: centre falls back to 0 / 1e-6 = 0.
        kp[2 * 25 * 2..3 * 25 * 2].fill(f64::NAN);
        let kin = body_kinematics(&kp, &conf, 4, 25, 30.0, None);
        assert!(kin.centre[2][0].is_finite());
        assert!(close(kin.centre[2][0], 0.0, 1e-9));
        assert!(kin.axis[2][0].is_finite());
        // A single frame takes the "no gradient" branch.
        let kin = body_kinematics(&kp[..50], &conf[..25], 1, 25, 30.0, None);
        assert_eq!(kin.centre.len(), 1);
        assert_eq!(kin.velocity[0], [0.0, 0.0]);
        assert!(close(kin.speed[0], 0.0, 1e-12));
    }

    #[test]
    fn low_confidence_poses_are_dropped_from_the_centre() {
        let (kp, mut conf) = mouse(6, 0.0, 1.0);
        conf[..6 * 25].fill(0.1); // below CORE_CONF_THRESH
        let kin = body_kinematics(&kp, &conf, 6, 25, 30.0, None);
        for c in &kin.centre {
            assert!(close(c[0], 0.0, 1e-9), "{c:?}");
        }
    }

    #[test]
    fn nan_helpers_match_numpy() {
        assert_eq!(nan_to_num(f64::NAN), 0.0);
        assert!(nan_to_num(f64::INFINITY) > 1e307);
        assert!(nan_median(&[3.0, 1.0, 2.0]) - 2.0 < 1e-12);
        assert!(nan_median(&[3.0, f64::NAN, 1.0, 2.0]) - 2.0 < 1e-12);
        assert!(nan_median(&[4.0, 1.0, 2.0, 3.0]) - 2.5 < 1e-12);
        assert!(nan_median(&[f64::NAN]).is_nan());
        assert!(median_diff(&[0.0, 0.5, 1.0, 2.0]) - 0.5 < 1e-12);
    }

    #[test]
    fn gradient_matches_numpy_on_uneven_spacing() {
        // np.gradient([[t^2, 3t], ...], t, axis=0) for an irregular grid.
        let t = [0.0, 0.1, 0.25, 0.4, 0.5, 0.9];
        let values: Vec<[f64; 2]> = t.iter().map(|v| [v * v, 3.0 * v]).collect();
        let g = gradient(&values, &t, 1.0);
        let expect = [
            [0.10000000000000002, 3.0000000000000004],
            [0.2, 3.0],
            [0.49999999999999994, 3.0],
            [0.8, 3.0],
            [1.0, 3.0],
            [1.4000000000000001, 3.0000000000000004],
        ];
        for (got, want) in g.iter().zip(expect) {
            assert!(close(got[0], want[0], 1e-9), "{got:?} vs {want:?}");
            assert!(close(got[1], want[1], 1e-9), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn convolve_same_matches_numpy_edge_behaviour() {
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let g3 = convolve_same(&a, 3);
        assert!(close(g3[0], 1.0 / 3.0, 1e-12), "{}", g3[0]);
        assert!(close(g3[9], 17.0 / 3.0, 1e-12));
        let g4 = convolve_same(&a, 4);
        assert!(close(g4[0], 0.25, 1e-12));
        assert!(close(g4[9], 6.0, 1e-12));
        let g1 = convolve_same(&a, 1);
        assert_eq!(g1, a);
        assert!(convolve_same(&[], 3).is_empty());
    }
}
