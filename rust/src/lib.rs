//! `animalpose` in Rust: the tensors, networks, algorithms, data and command line of the Python
//! package under `../src/animalpose`, in one crate.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`tensor`] | f32 tensors with reverse-mode autodiff, and the layers the networks train |
//! | [`algos`] | pose decoding, non-maximum suppression, Kalman tracking, blob proposals and behavior post-processing |
//! | [`models`] | the detector, pose, temporal and behavior networks, loadable from the same checkpoints as Python |
//! | [`data`] | the procedural simulator, COCO keypoint loading, behavior sessions and degradation profiles |
//! | [`eval`] | detection, pose, tracking and behavior metrics, and the benchmark harnesses |
//! | [`multiview`] | cameras, calibration, triangulation, cross-view identity fusion and the 3D pipeline |
//! | [`cli`] | the command line: `train`, `finetune`, `label`, `predict`, `behavior`, `data`, `eval`, `export`, with the same configs as Python |
//!
//! The modules sit in the dependency order of the table above (rustfmt keeps the declarations
//! alphabetised): `tensor` carries `algos` and `models`, `data` adds the inputs they train on, and
//! `cli` runs them. `src/main.rs` is the executable and calls [`cli::run`], so the whole command
//! line is testable as a library.

#[rustfmt::skip]
pub mod tensor;
#[rustfmt::skip]
pub mod algos;
#[rustfmt::skip]
pub mod models;
#[rustfmt::skip]
pub mod data;
#[rustfmt::skip]
pub mod eval;
#[cfg(feature = "app")]
pub mod app;
pub mod cli;
#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(feature = "gui")]
pub mod gui;
pub mod multiview;
