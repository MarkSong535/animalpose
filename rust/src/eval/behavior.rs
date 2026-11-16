//! Behavior metrics over sessions, with post-processing variants from one forward pass — the port
//! of `src/animalpose/eval/behavior.py`.
//!
//! The classifier runs once per session ([`session_logits`]); [`behavior_report`] then scores raw,
//! moving-median smoothed and prior-adapted probabilities from those logits, per session and never
//! across a session boundary. Labels are indices of `label_classes` and are reconciled with the
//! model's classes by name ([`crate::data::sessions::remap_labels`]).

use crate::algos::postprocess::{adapt_prior, median_smooth};
use crate::data::pyjson::PyVal;
use crate::data::sessions::{remap_labels, Session, IGNORE};
use crate::eval::metrics::f1_scores;
use crate::models::Model;
use crate::tensor::{Ctx, Graph};
use anyhow::{Context, Result};

/// `labels.py:SHARED`: the classes every label set and classifier share.
pub const SHARED: [&str; 4] = ["walking", "rearing", "grooming", "eating"];
/// `labels.py:LABELLED`: [`SHARED`] plus `other`, the classes real data labels.
pub const LABELLED: [&str; 5] = ["walking", "rearing", "grooming", "eating", "other"];
/// `labels.py:BEHAVIORS_EXT`: the extended vocabulary session files default to.
pub const BEHAVIORS_EXT: [&str; 6] = ["resting", "walking", "rearing", "grooming", "eating", "other"];

/// `PROTOCOLS`: the two protocols behavior models are reported on.
pub fn protocols() -> Vec<(String, Vec<String>)> {
    vec![
        ("shared4".into(), SHARED.iter().map(|s| s.to_string()).collect()),
        ("labelled5".into(), LABELLED.iter().map(|s| s.to_string()).collect()),
    ]
}

/// `_softmax(z)` over the rows of a `(T, C)` array.
pub fn softmax(z: &[f64], classes: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(z.len());
    for row in z.chunks_exact(classes) {
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        out.extend(e.iter().map(|v| v / s));
    }
    out
}

/// What [`protocol_metrics`] returns, the Python dict.
#[derive(Clone, Debug, PartialEq)]
pub struct ProtocolMetrics {
    /// Macro-F1 over the candidate classes present in the ground truth.
    pub macro_f1: f64,
    /// Share of scored frames predicted right.
    pub accuracy: f64,
    /// `{candidate: F1}`, in candidate order (empty when nothing was scored).
    pub per_class_f1: Vec<(String, f64)>,
    /// `confusion[gt][pred]` over the candidates (`None` when nothing was scored).
    pub confusion: Option<Vec<Vec<i64>>>,
    /// Frames scored.
    pub n_frames: usize,
}

impl ProtocolMetrics {
    /// The Python dict, in its key order (`confusion` dropped when `with_confusion` is false).
    pub fn to_py(&self, with_confusion: bool) -> PyVal {
        let per_class = PyVal::Dict(
            self.per_class_f1
                .iter()
                .map(|(k, v)| (k.clone(), PyVal::Float(*v)))
                .collect(),
        );
        match &self.confusion {
            None => PyVal::dict()
                .with("macro_f1", self.macro_f1)
                .with("accuracy", self.accuracy)
                .with("n_frames", self.n_frames)
                .with("per_class_f1", per_class),
            Some(cm) => {
                let mut d = PyVal::dict()
                    .with("macro_f1", self.macro_f1)
                    .with("accuracy", self.accuracy)
                    .with("per_class_f1", per_class);
                if with_confusion {
                    d.set(
                        "confusion",
                        PyVal::List(
                            cm.iter()
                                .map(|r| PyVal::List(r.iter().map(|v| PyVal::Int(*v)).collect()))
                                .collect(),
                        ),
                    );
                }
                d.with("n_frames", self.n_frames)
            }
        }
    }
}

/// `protocol_metrics(probs, gt, classes, candidates, label_classes)`: frame-wise macro-F1 on
/// labelled frames whose class is a candidate, argmax over the candidates only. `gt` holds indices
/// of `label_classes` when that is given and differs from `classes`.
pub fn protocol_metrics(
    probs: &[f64],
    gt: &[i64],
    classes: &[String],
    candidates: &[String],
    label_classes: Option<&[String]>,
) -> Result<ProtocolMetrics> {
    let n_classes = classes.len();
    let gt: Vec<i64> = match label_classes {
        Some(lc) if lc != classes => {
            let src: Vec<&str> = lc.iter().map(String::as_str).collect();
            let dst: Vec<&str> = classes.iter().map(String::as_str).collect();
            remap_labels(gt, &src, &dst)
        }
        _ => gt.to_vec(),
    };
    let cols: Vec<usize> = candidates
        .iter()
        .map(|c| {
            classes
                .iter()
                .position(|k| k == c)
                .with_context(|| format!("{c:?} is not in list"))
        })
        .collect::<Result<_>>()?;
    let sel: Vec<usize> = (0..gt.len())
        .filter(|&i| gt[i] != IGNORE && cols.contains(&usize::try_from(gt[i]).unwrap_or(usize::MAX)))
        .collect();
    if sel.is_empty() {
        return Ok(ProtocolMetrics {
            macro_f1: f64::NAN,
            accuracy: f64::NAN,
            per_class_f1: Vec::new(),
            confusion: None,
            n_frames: 0,
        });
    }
    let mut pred = Vec::with_capacity(sel.len());
    let mut gt_idx = Vec::with_capacity(sel.len());
    for &i in &sel {
        let row = &probs[i * n_classes..(i + 1) * n_classes];
        let mut best = 0;
        for (j, c) in cols.iter().enumerate() {
            if row[*c] > row[cols[best]] {
                best = j;
            }
        }
        pred.push(best);
        let g = gt[i] as usize;
        gt_idx.push(cols.iter().position(|c| *c == g).unwrap_or(0));
    }
    let r = f1_scores(&pred, &gt_idx, candidates.len());
    Ok(ProtocolMetrics {
        macro_f1: r.macro_f1,
        accuracy: r.accuracy,
        per_class_f1: candidates.iter().cloned().zip(r.per_class_f1).collect(),
        confusion: Some(r.confusion),
        n_frames: sel.len(),
    })
}

/// `{variant: {protocol: metrics}}`, variants in the order Python builds them.
pub type Report = Vec<(String, Vec<(String, ProtocolMetrics)>)>;

/// `behavior_report` on logits that were already computed: `logits[s]` is the `(T_s, C)` output of
/// session `s` and `labels[s]` its labels. Variants are `raw`, `smooth<w>` for every `w > 1`, each
/// also `+prior` when `prior_adapt` is set.
#[allow(clippy::too_many_arguments)]
pub fn behavior_report_from_logits(
    logits: &[Vec<f64>],
    labels: &[Vec<i64>],
    classes: &[String],
    protocols: &[(String, Vec<String>)],
    smoothing: &[usize],
    prior_adapt: bool,
    prior_max_ratio: f64,
    label_classes: Option<&[String]>,
) -> Result<Report> {
    let c = classes.len();
    let mut variants: Vec<(String, Vec<Vec<f64>>)> = vec![("raw".into(), Vec::new())];
    for w in smoothing {
        if *w > 1 && !variants.iter().any(|(n, _)| *n == format!("smooth{w}")) {
            variants.push((format!("smooth{w}"), Vec::new()));
        }
    }
    for z in logits {
        let p = softmax(z, c);
        let frames = p.len() / c.max(1);
        variants[0].1.push(p.clone());
        for w in smoothing {
            if *w > 1 {
                let name = format!("smooth{w}");
                let slot = variants.iter_mut().find(|(n, _)| *n == name).expect("added");
                slot.1.push(median_smooth(&p, frames, c, *w as i64, true));
            }
        }
    }
    let all_labels: Vec<i64> = labels.iter().flatten().copied().collect();
    let mut out = Vec::new();
    for (name, per_session) in &variants {
        let modes: &[bool] = if prior_adapt { &[false, true] } else { &[false] };
        for &use_prior in modes {
            let key = if use_prior {
                format!("{name}+prior")
            } else {
                name.clone()
            };
            let mut rows: Vec<f64> = Vec::new();
            for p in per_session {
                if use_prior {
                    let frames = p.len() / c.max(1);
                    rows.extend(adapt_prior(p, frames, c, None, 25, prior_max_ratio, 1e-4).0);
                } else {
                    rows.extend_from_slice(p);
                }
            }
            let mut per_proto = Vec::new();
            for (proto, cands) in protocols {
                per_proto.push((
                    proto.clone(),
                    protocol_metrics(&rows, &all_labels, classes, cands, label_classes)?,
                ));
            }
            out.push((key, per_proto));
        }
    }
    Ok(out)
}

/// `session_logits(model, sessions, conf_thresh)`: the classifier's `(T, C)` logits per session
/// from the whole track in one pass, keypoints masked where `conf < conf_thresh` or not finite.
pub fn session_logits(model: &Model, sessions: &[Session], conf_thresh: f64) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::with_capacity(sessions.len());
    for s in sessions {
        let (t, k) = (s.frames, s.keypoints);
        let d = if t * k > 0 { s.coords.len() / (t * k) } else { 2 };
        let x: Vec<f32> = s
            .coords
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .collect();
        let m: Vec<f32> = (0..t * k)
            .map(|i| {
                let finite = s.coords[i * d..(i + 1) * d].iter().all(|v| v.is_finite());
                f32::from(u8::from(f64::from(s.conf[i]) >= conf_thresh && finite))
            })
            .collect();
        let mut g = Graph::new();
        let xin = g.from_data(x, &[1, t, k, d]);
        let min = g.from_data(m, &[1, t, k]);
        let mut ctx = Ctx::eval();
        let outs = model.forward_named(&mut ctx, &mut g, &[xin, min])?;
        let logits = outs
            .iter()
            .find(|(name, _)| *name == "logits")
            .map(|(_, id)| *id)
            .context("the behavior model has no logits output")?;
        out.push(g.to_vec(logits).iter().map(|v| f64::from(*v)).collect());
    }
    Ok(out)
}

/// `behavior_report(model, sessions, protocols, smoothing, prior_adapt, prior_max_ratio,
/// conf_thresh, label_classes)`: one forward pass per session, every variant scored.
#[allow(clippy::too_many_arguments)]
pub fn behavior_report(
    model: &Model,
    classes: &[String],
    sessions: &[Session],
    protocols: &[(String, Vec<String>)],
    smoothing: &[usize],
    prior_adapt: bool,
    prior_max_ratio: f64,
    conf_thresh: f64,
    label_classes: Option<&[String]>,
) -> Result<Report> {
    let logits = session_logits(model, sessions, conf_thresh)?;
    let labels: Vec<Vec<i64>> = sessions.iter().map(|s| s.labels.clone()).collect();
    behavior_report_from_logits(
        &logits,
        &labels,
        classes,
        protocols,
        smoothing,
        prior_adapt,
        prior_max_ratio,
        label_classes,
    )
}

/// `protocol_scores(model, sessions, protocols, label_classes, conf_thresh)`: the raw
/// `{protocol: metrics}` (without `confusion`) on the protocols whose classes the model has.
pub fn protocol_scores_from_logits(
    logits: &[Vec<f64>],
    labels: &[Vec<i64>],
    classes: &[String],
    protocols: Option<&[(String, Vec<String>)]>,
    label_classes: Option<&[String]>,
) -> Result<Vec<(String, ProtocolMetrics)>> {
    let all = self::protocols();
    let wanted: Vec<(String, Vec<String>)> = protocols
        .unwrap_or(&all)
        .iter()
        .filter(|(_, cands)| cands.iter().all(|c| classes.contains(c)))
        .cloned()
        .collect();
    let report = behavior_report_from_logits(
        logits,
        labels,
        classes,
        &wanted,
        &[1],
        false,
        4.0,
        label_classes,
    )?;
    Ok(report.into_iter().next().map(|(_, m)| m).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    //! `tests/test_eval_behavior.py`: the scripted-classifier checks, whose numbers are worked out
    //! by hand there.
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    const BEHAVIORS: [&str; 5] = ["resting", "walking", "rearing", "grooming", "eating"];

    fn onehot(pred: &[usize], n: usize) -> Vec<f64> {
        let mut p = vec![0.05; pred.len() * n];
        for (i, c) in pred.iter().enumerate() {
            p[i * n + c] = 0.9;
        }
        p
    }

    /// `_Scripted`: +10 on the scripted class, -10 on the rest.
    fn scripted() -> (Vec<Vec<f64>>, Vec<Vec<i64>>) {
        let pred30: Vec<usize> = [vec![0; 10], vec![1, 1], vec![0; 3], vec![1; 15]].concat();
        let pred6: Vec<usize> = [vec![0], vec![1; 5]].concat();
        let logits = |pred: &[usize]| -> Vec<f64> {
            pred.iter()
                .flat_map(|c| (0..2).map(move |j| if j == *c { 10.0 } else { -10.0 }))
                .collect()
        };
        let gt30: Vec<i64> = [vec![0; 15], vec![1; 15]].concat();
        (vec![logits(&pred30), logits(&pred6)], vec![gt30, vec![0; 6]])
    }

    #[test]
    fn softmax_is_shift_invariant_and_cannot_overflow() {
        let z = [2.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let shifted: Vec<f64> = z.iter().map(|v| v + 500.0).collect();
        let (a, b) = (softmax(&z, 2), softmax(&shifted, 2));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        assert!((a[0] - 1.0 / (1.0 + (-2.0f64).exp())).abs() < 1e-12);
        assert!(softmax(&[500.0; 12], 4).iter().all(|v| (v - 0.25).abs() < 1e-12));
    }

    #[test]
    fn protocol_metrics_scores_labelled_candidate_classes_only_with_hand_checked_arithmetic() {
        let mut probs = onehot(&[1, 2, 2, 1, 4, 0], 5);
        for i in 0..6 {
            probs[i * 5] = 0.9;
        }
        let gt = [1, 1, 2, 2, 3, 0];
        let r = protocol_metrics(&probs, &gt, &s(&BEHAVIORS), &s(&SHARED), None).unwrap();
        assert_eq!(r.n_frames, 5);
        assert_eq!(
            r.confusion.clone().unwrap(),
            vec![vec![1, 1, 0, 0], vec![1, 1, 0, 0], vec![0, 0, 0, 1], vec![0, 0, 0, 0]]
        );
        let f1: Vec<f64> = r.per_class_f1.iter().map(|(_, v)| *v).collect();
        assert_eq!(f1, [0.5, 0.5, 0.0, 0.0]);
        assert!((r.macro_f1 - 1.0 / 3.0).abs() < 1e-12 && (r.accuracy - 0.4).abs() < 1e-12);
        let empty = protocol_metrics(&probs, &[IGNORE; 6], &s(&BEHAVIORS), &s(&SHARED), None).unwrap();
        assert!(empty.macro_f1.is_nan() && empty.accuracy.is_nan());
        assert_eq!(empty.n_frames, 0);
        assert_eq!(
            empty.to_py(true).dumps(),
            r#"{"macro_f1": NaN, "accuracy": NaN, "n_frames": 0, "per_class_f1": {}}"#
        );
    }

    #[test]
    fn frames_outside_the_label_vocabulary_drop_out_of_the_scored_set() {
        let mut probs = onehot(&[1, 2, 2, 1, 4, 0, 0, 0], 6);
        for i in 0..8 {
            probs[i * 6] = 0.9;
        }
        let gt = [1, 1, 2, 2, 3, 0];
        let src: Vec<&str> = BEHAVIORS.to_vec();
        let ext: Vec<i64> = [remap_labels(&gt, &src, &BEHAVIORS_EXT), vec![5, IGNORE]].concat();
        let lc = s(&BEHAVIORS_EXT);
        let narrow =
            protocol_metrics(&probs[..8 * 6], &ext, &s(&BEHAVIORS_EXT), &s(&SHARED), Some(&lc))
                .unwrap();
        assert_eq!(narrow.n_frames, 5);
        let wide =
            protocol_metrics(&probs, &ext, &s(&BEHAVIORS_EXT), &s(&LABELLED), Some(&lc)).unwrap();
        assert_eq!(wide.n_frames, 6);
        assert_eq!(wide.confusion.as_ref().unwrap()[4][0], 1);
        let f1: Vec<f64> = wide.per_class_f1.iter().map(|(_, v)| *v).collect();
        assert_eq!(f1, [0.4, 0.5, 0.0, 0.0, 0.0]);
        assert!((wide.macro_f1 - 0.9 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn report_scores_every_variant_from_one_forward_pass_per_session() {
        let (logits, labels) = scripted();
        let pair = s(&["walking", "rearing"]);
        let protos = vec![("both".to_string(), pair.clone())];
        let rep =
            behavior_report_from_logits(&logits, &labels, &pair, &protos, &[1, 3, 5, 15], true, 4.0, None)
                .unwrap();
        let names: Vec<&str> = rep.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "raw", "raw+prior", "smooth3", "smooth3+prior", "smooth5", "smooth5+prior",
                "smooth15", "smooth15+prior"
            ]
        );
        let cm = |name: &str| rep.iter().find(|(n, _)| n == name).unwrap().1[0].1.clone();
        let raw_cm = vec![vec![14, 7], vec![0, 15]];
        let smooth_cm = vec![vec![16, 5], vec![0, 15]];
        assert_eq!(cm("raw").confusion.unwrap(), raw_cm);
        assert_eq!(cm("smooth3").confusion.unwrap(), raw_cm);
        assert_eq!(cm("smooth5").confusion.unwrap(), smooth_cm);
        assert_eq!(cm("smooth15").confusion.unwrap(), raw_cm);
        let raw = cm("raw");
        assert_eq!(raw.n_frames, 36);
        assert!((raw.accuracy - 29.0 / 36.0).abs() < 1e-12);
        assert!((raw.macro_f1 - (4.0 / 5.0 + 30.0 / 37.0) / 2.0).abs() < 1e-12);
        assert!((cm("smooth5").accuracy - 31.0 / 36.0).abs() < 1e-12);
        let capped =
            behavior_report_from_logits(&logits, &labels, &pair, &protos, &[5], true, 1.0, None)
                .unwrap();
        assert_eq!(capped[2].1, capped[3].1); // smooth5 == smooth5+prior when no prior may move
    }
}
