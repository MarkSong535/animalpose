//! Temporal left/right identity for bilateral keypoint pairs.
//!
//! The port of `src/animalpose/temporal/pairs.py:resolve_pair_identity`: a two-state Viterbi per
//! pair and track, whose emission cost is the pose network's left/right margin and whose transition
//! cost is how far each labelled paw would have to jump between frames.

/// `resolve_pair_identity`'s default `margin_weight`.
pub const DEFAULT_MARGIN_WEIGHT: f64 = 1.0;

/// The cost matrix of one transition, `[same, cross]`.
#[derive(Clone, Copy, Debug, Default)]
struct Transition {
    same: f64,
    cross: f64,
}

/// Relabels bilateral pairs consistently over one track.
///
/// * `kpts`: `(frames, keypoints, 2)` decoded keypoints in image pixels, NaN where unobserved;
///   swapped pairs are written back in place.
/// * `margin`: `(frames, pairs.len())` per-frame evidence for the decoded labelling (positive keeps).
/// * `scale`: the animal's size in pixels per frame (`nan_to_num(size, nan = 1.0)` of the tracker).
///
/// Returns the `(frames, pairs.len())` swapped mask, row-major.
///
/// Port of `temporal/pairs.py:resolve_pair_identity` with `margin_weight = 1.0`.
///
/// `// differs from Python:` the two-state Viterbi is the same dynamic program; costs are `f64`
/// where Python's are `float64` too, so the results agree bit for bit given equal inputs.
pub fn resolve_pair_identity(
    kpts: &mut [[f64; 2]],
    margin: &[f64],
    pairs: &[(usize, usize)],
    scale: &[f64],
    motion_frac: f64,
    keypoints: usize,
) -> Vec<bool> {
    let frames = kpts.len() / keypoints.max(1);
    let mut swapped = vec![false; frames * pairs.len()];
    if frames == 0 || pairs.is_empty() {
        return swapped;
    }
    let snapshot = kpts.to_vec();
    let at = |t: usize, k: usize| snapshot[t * keypoints + k];
    for (p, (lft, rgt)) in pairs.iter().enumerate() {
        // `a, b = kpts[:, lft], kpts[:, rgt]`: the decoded paws, kept in the original order so a
        // frame can be swapped exactly once.
        let a: Vec<[f64; 2]> = (0..frames).map(|t| at(t, *lft)).collect();
        let b: Vec<[f64; 2]> = (0..frames).map(|t| at(t, *rgt)).collect();
        let emit: Vec<[f64; 2]> = (0..frames)
            .map(|t| {
                let m = nan_to_num(margin.get(t * pairs.len() + p).copied().unwrap_or(0.0))
                    * DEFAULT_MARGIN_WEIGHT;
                [-m / 2.0, m / 2.0]
            })
            .collect();
        let ok: Vec<bool> = (0..frames)
            .map(|t| a[t].into_iter().all(f64::is_finite) && b[t].into_iter().all(f64::is_finite))
            .collect();
        let trans = transitions(&a, &b, &ok, scale, motion_frac, frames);
        let (back, mut state) = viterbi(&emit, &trans, frames);
        for t in (0..frames).rev() {
            // `swapped[t] = s == 1; s = back[t][s]`: the chain is walked backwards.
            swapped[t * pairs.len() + p] = state == 1;
            state = back[t][state];
        }
        for t in 0..frames {
            if swapped[t * pairs.len() + p] {
                kpts[t * keypoints + *lft] = b[t];
                kpts[t * keypoints + *rgt] = a[t];
            }
        }
    }
    swapped
}

/// Runs the forward pass; returns each frame's best predecessor state and the best final state.
fn viterbi(emit: &[[f64; 2]], trans: &[Transition], frames: usize) -> (Vec<[usize; 2]>, usize) {
    let mut back = vec![[0usize; 2]; frames];
    if frames == 0 {
        return (back, 0);
    }
    let mut cost = emit[0];
    for t in 1..frames {
        let step = trans[t - 1];
        let (stay, swap) = (cost[0], cost[1]);
        // `tot = cost[:, None] + [[same, cross], [cross, same]]`, row = the previous state.
        let tot = [
            [stay + step.same, stay + step.cross],
            [swap + step.cross, swap + step.same],
        ];
        // The index addresses `back`, `tot`, `cost` and `emit` in lockstep, as the vectorised
        // NumPy line above does, so it is not a needless range loop.
        #[allow(clippy::needless_range_loop)]
        for s in 0..2 {
            // `argmin` takes the first minimum, so a tie keeps the "stay" state.
            back[t][s] = usize::from(tot[1][s] < tot[0][s]);
            cost[s] = tot[0][s].min(tot[1][s]) + emit[t][s];
        }
    }
    (back, usize::from(cost[1] < cost[0]))
}

/// The per-transition jump costs, normalised by the motion a frame gap allows.
fn transitions(
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    ok: &[bool],
    scale: &[f64],
    motion_frac: f64,
    frames: usize,
) -> Vec<Transition> {
    // `last = np.maximum.accumulate(np.where(ok, np.arange(t), -1))`: the most recent observed
    // frame at or before each frame, -1 before the first one.
    let mut last = vec![-1i64; frames];
    let mut running = -1i64;
    for t in 0..frames {
        if ok[t] {
            running = t as i64;
        }
        last[t] = running;
    }
    let mut out = Vec::with_capacity(frames.saturating_sub(1));
    for t in 1..frames {
        let prev = last[t - 1].max(0) as usize;
        let gap = if last[t - 1] >= 0 {
            (t - prev) as f64
        } else {
            1.0
        };
        let unit = motion_frac * scale.get(t).copied().unwrap_or(1.0).max(1e-6) * gap;
        let (pa, pb) = (a[prev], b[prev]);
        let same = norm(sub(a[t], pa)) + norm(sub(b[t], pb));
        let cross = norm(sub(a[t], pb)) + norm(sub(b[t], pa));
        let allowed = ok[t];
        out.push(Transition {
            same: nan_to_num(same / unit),
            // `cross[~ok[1:]] = 1e9`: an unobserved frame cannot carry a crossing.
            cross: if allowed {
                nan_to_num(cross / unit)
            } else {
                1e9
            },
        });
    }
    out
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn norm(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// `np.nan_to_num` for one scalar.
fn nan_to_num(v: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    /// A paw pair that crosses, with an unobserved frame in the middle: the labels have to follow
    /// the paws, which is what `temporal/pairs.py:resolve_pair_identity` decides.
    #[test]
    fn a_crossing_pair_is_relabeled_like_python() {
        let mut kpts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [3.0, 0.0],
            [7.0, 0.0],
            [NAN, NAN],
            [NAN, NAN],
            [9.0, 1.0],
            [1.0, 1.0],
        ];
        let margin = [0.0; 4];
        let scale = [100.0; 4];
        let swapped = resolve_pair_identity(&mut kpts, &margin, &[(0, 1)], &scale, 0.1, 2);
        assert_eq!(swapped, [true, true, true, false]);
        let want: [[f64; 2]; 8] = [
            [10.0, 0.0],
            [0.0, 0.0],
            [7.0, 0.0],
            [3.0, 0.0],
            [NAN, NAN],
            [NAN, NAN],
            [9.0, 1.0],
            [1.0, 1.0],
        ];
        for (got, want) in kpts.iter().zip(want) {
            match (got, want) {
                ([x, y], [wx, wy]) if x.is_nan() && y.is_nan() && wx.is_nan() && wy.is_nan() => {}
                _ => assert_eq!(*got, want),
            }
        }
    }

    /// A strong negative margin (the network is sure the decoded labelling is wrong) also flips the
    /// last frame, which the motion term alone would have kept.
    #[test]
    fn margin_overrides_the_motion_cost() {
        let mut kpts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [3.0, 0.0],
            [7.0, 0.0],
            [NAN, NAN],
            [NAN, NAN],
            [9.0, 1.0],
            [1.0, 1.0],
        ];
        let margin = [-5.0, 0.0, 0.0, -5.0];
        let swapped = resolve_pair_identity(&mut kpts, &margin, &[(0, 1)], &[100.0; 4], 0.1, 2);
        assert_eq!(swapped, [true, true, true, true]);
        assert_eq!(kpts[6], [1.0, 1.0]);
        assert_eq!(kpts[7], [9.0, 1.0]);
    }

    #[test]
    fn an_empty_track_swaps_nothing() {
        let mut kpts: Vec<[f64; 2]> = Vec::new();
        assert!(resolve_pair_identity(&mut kpts, &[], &[(0, 1)], &[], 0.1, 2).is_empty());
    }

    #[test]
    fn no_pairs_changes_nothing() {
        let mut kpts = vec![[1.0, 2.0], [3.0, 4.0]];
        let before = kpts.clone();
        assert!(resolve_pair_identity(&mut kpts, &[], &[], &[100.0, 100.0], 0.1, 2).is_empty());
        assert_eq!(kpts, before);
    }
}
