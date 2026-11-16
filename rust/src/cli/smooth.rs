//! Temporal refinement of one track: the port of
//! `src/animalpose/temporal/model.py:smooth_sequence`.

use crate::models::Model;
use crate::tensor::{Ctx, Graph};
use anyhow::{Context, Result};

/// `smooth_sequence`'s `sigma_ref`: the predicted uncertainty that maps to confidence 0.5.
pub const SIGMA_REF: f64 = 0.05;

/// What `smooth_sequence` returns: refined keypoints, their confidence and the imputed mask.
#[derive(Clone, Debug)]
pub struct Smoothed {
    /// Refined coordinates `(frames, keypoints, 2)` in image pixels.
    pub coords: Vec<f64>,
    /// `1 / (1 + sigma / sigma_ref)`, `(frames, keypoints)`.
    pub confidence: Vec<f64>,
    /// `1 - observed`, `(frames, keypoints)`: 1 where the model filled a gap.
    pub imputed: Vec<f64>,
}

/// Runs the temporal network over a whole track with overlapping windows blended by a triangular
/// weight, exactly as `temporal/model.py:smooth_sequence` does.
///
/// * `coords`: `(frames, keypoints, 2)`, NaN where the tracker had no observation.
/// * `conf`: `(frames, keypoints)` confidences of those observations.
/// * `observed`: `(frames, keypoints)` the mask the classifier would have seen.
///
/// The keypoint count comes from `observed` and the coordinate width `D` from `coords`, which
/// holds `frames * keypoints * D` values.
pub fn smooth_sequence(
    model: &Model,
    coords: &[f64],
    conf: &[f64],
    observed: &[f64],
    frames: usize,
    window: usize,
    hop: usize,
) -> Result<Smoothed> {
    anyhow::ensure!(frames > 0, "cannot smooth an empty track");
    let keypoints = observed.len() / frames;
    let window = window.clamp(1, frames);
    // `(T, K, D)`: 2 for image tracks, 3 for triangulated ones (`MultiViewPipeline`).
    let dims = coords.len() / (frames * keypoints.max(1));
    let x: Vec<f32> = coords.iter().map(|v| nan_to_num(*v) as f32).collect();
    let c: Vec<f32> = conf.iter().map(|v| nan_to_num(*v) as f32).collect();
    let o: Vec<f32> = observed.iter().map(|v| *v as f32).collect();

    let starts: Vec<usize> = if frames <= window {
        vec![0]
    } else {
        let mut s: Vec<usize> = (0..=frames - window).step_by(hop.max(1)).collect();
        if s.last().copied() != Some(frames - window) {
            s.push(frames - window);
        }
        s
    };
    // `ramp = min(arange(1, w + 1), arange(w, 0, -1))`: windows fade in and out.
    let ramp: Vec<f64> = (1..=window)
        .map(|i| f64::min(i as f64, (window + 1 - i) as f64))
        .collect();

    let mut acc = vec![0.0f64; frames * keypoints * dims];
    let mut acc_sigma = vec![0.0f64; frames * keypoints];
    let mut wsum = vec![0.0f64; frames];

    // `Graph::matmul` folds away a batch of one (`ops.rs`: `if batch == 1 { vec![m, n] }`), and
    // `lstm` then rejects the 2-D feature map it is handed, so windows go through two at a time.
    //
    // `// differs from Python:` `smooth_sequence` forwards one window at a time, which torch keeps
    // as `(1, window, K, 2)`; a lone window is repeated here and its duplicate output discarded.
    for pair in starts.chunks(2) {
        let batch: Vec<usize> = if pair.len() == 2 {
            vec![pair[0], pair[1]]
        } else {
            vec![pair[0], pair[0]]
        };
        let mut coords_batch = Vec::with_capacity(batch.len() * window * keypoints * dims);
        let mut conf_batch = Vec::with_capacity(batch.len() * window * keypoints);
        let mut obs_batch = Vec::with_capacity(batch.len() * window * keypoints);
        for &start in &batch {
            let end = start + window;
            coords_batch.extend_from_slice(&x[start * keypoints * dims..end * keypoints * dims]);
            conf_batch.extend_from_slice(&c[start * keypoints..end * keypoints]);
            obs_batch.extend_from_slice(&o[start * keypoints..end * keypoints]);
        }
        let mut g = Graph::new();
        let input = g.from_data(coords_batch, &[batch.len(), window, keypoints, dims]);
        let conf_node = g.from_data(conf_batch, &[batch.len(), window, keypoints]);
        let obs_node = g.from_data(obs_batch, &[batch.len(), window, keypoints]);
        let mut ctx = Ctx::eval();
        let out = model
            .forward_named(&mut ctx, &mut g, &[input, conf_node, obs_node])
            .context("`smooth_sequence` needs a temporal checkpoint")?;
        let coords_node = out
            .iter()
            .find(|(name, _)| *name == "coords")
            .map(|(_, node)| *node)
            .context("the temporal model returned no `coords`")?;
        let sigma_node = out
            .iter()
            .find(|(name, _)| *name == "log_sigma")
            .map(|(_, node)| *node)
            .context("the temporal model returned no `log_sigma`")?;
        let predicted = g.to_vec(coords_node);
        let log_sigma = g.to_vec(sigma_node);
        for (row, &start) in batch.iter().enumerate() {
            if row == 1 && pair.len() == 1 {
                continue; // the repeated window, kept only to give the network a batch of two
            }
            for (i, w) in ramp.iter().enumerate() {
                let t = start + i;
                for k in 0..keypoints {
                    for d in 0..dims {
                        acc[(t * keypoints + k) * dims + d] += f64::from(
                            predicted[(row * window + i) * keypoints * dims + k * dims + d],
                        ) * w;
                    }
                    // `acc_s += torch.exp(out["log_sigma"][i]) * ramp`: sigma, not log sigma.
                    acc_sigma[t * keypoints + k] +=
                        f64::from(log_sigma[(row * window + i) * keypoints + k]).exp() * w;
                }
                wsum[t] += w;
            }
        }
    }

    let mut out_coords = vec![0.0f64; acc.len()];
    for t in 0..frames {
        let w = wsum[t].max(1e-12);
        for k in 0..keypoints {
            for d in 0..dims {
                out_coords[(t * keypoints + k) * dims + d] =
                    acc[(t * keypoints + k) * dims + d] / w;
            }
        }
    }
    let mut confidence = vec![0.0f64; frames * keypoints];
    for t in 0..frames {
        let w = wsum[t].max(1e-12);
        for k in 0..keypoints {
            confidence[t * keypoints + k] =
                1.0 / (1.0 + (acc_sigma[t * keypoints + k] / w) / SIGMA_REF);
        }
    }
    let imputed = o.iter().map(|v| 1.0 - f64::from(*v)).collect();
    Ok(Smoothed {
        coords: out_coords,
        confidence,
        imputed,
    })
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

    /// The whole smoothing loop over a real (tiny) checkpoint. A batch of one used to reach
    /// `lstm` as a 2-D feature map, which the tensor crate's `matmul` folds away.
    #[test]
    fn smooths_a_track_with_the_tiny_checkpoint() {
        let Some(fixtures) = crate::data::fixtures_dir() else {
            return;
        };
        let path = fixtures.join("models/temporal_tiny.safetensors");
        let (cfg, kind, weights) = crate::models::load(&path).expect("the fixture loads");
        let model = Model::build(kind, &cfg, &weights).expect("the fixture builds");
        let (frames, keypoints) = (20, 5);
        let coords: Vec<f64> = (0..frames * keypoints * 2)
            .map(|i| f64::from(i as u32 % 17))
            .collect();
        let conf = vec![0.9; frames * keypoints];
        let observed = vec![1.0; frames * keypoints];
        let got = smooth_sequence(&model, &coords, &conf, &observed, frames, 8, 4)
            .expect("smoothing runs");
        assert_eq!(got.coords.len(), frames * keypoints * 2);
        assert_eq!(got.confidence.len(), frames * keypoints);
        assert!(got.coords.iter().all(|v| v.is_finite()));
        assert!(got
            .confidence
            .iter()
            .all(|c| (0.0..=1.0).contains(c) && c.is_finite()));
        assert!(got.imputed.iter().all(|v| *v == 0.0));

        // A gap in the observations comes back imputed.
        let mut masked = observed.clone();
        for k in 0..keypoints {
            masked[9 * keypoints + k] = 0.0;
        }
        let filled = smooth_sequence(&model, &coords, &conf, &masked, frames, 8, 4).expect("runs");
        assert_eq!(filled.imputed[9 * keypoints], 1.0);
        assert_eq!(filled.imputed[8 * keypoints], 0.0);
        assert_eq!(cfg.num_keypoints(), keypoints);
    }
}
