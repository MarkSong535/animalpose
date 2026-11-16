//! Scoring: the metrics of `src/animalpose/eval/` and the harnesses `animalpose eval` runs on them.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `eval/metrics.py` (`average_precision`, `pck`, `rmse`, `tracking_metrics`, `f1_scores`, `linear_interpolate`, `auroc`) | [`crate::eval::metrics`] |
//! | `eval/behavior.py` (`protocol_metrics`, `behavior_report`, `protocol_scores`, `session_logits`) | [`crate::eval::behavior`] |
//! | `eval/benchmark.py` (`eval_coco`, `eval_temporal`, `eval_behavior`, the synthetic benchmark) | [`crate::eval::benchmark`] |
//!
//! The command-line harnesses (`animalpose eval <check>`) live in `crate::cli::eval_cmd`.

pub mod behavior;
pub mod benchmark;
pub mod metrics;
