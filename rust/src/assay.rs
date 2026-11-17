//! The experiment layer ("behavior layer v3"): turns tracked keypoints into the labelled states of
//! an experiment setup, with rules a person can check.
//!
//! Learned models stop at keypoints. Everything after that is geometry, written in body lengths
//! (BL), box units or arena fractions, so no rule depends on the resolution or camera distance:
//!
//! | Stage | Module |
//! | --- | --- |
//! | Inputs: the test animal's 2D track, its 3D keypoints, side-view tracks, cameras | [`input`] |
//! | Arena geometry: floor, zones, cups — a trained arena detector, the calibration or classical vision | [`arena_model`], [`arena`], [`maze`] |
//! | Per-frame features (BL, head direction, speed, compactness, …) | [`features`], [`motion`] |
//! | Rules, one module per behavior or location | [`modules`] |
//! | States over time: hysteresis, dwell, gap merging, minimum bouts, priority | [`temporal`] |
//! | Per-frame CSV, per-video summary, run manifest; atomic, staged, resumable | [`output`] |
//!
//! An experiment setup is a list of modules plus their parameters, all under `assay:` in the
//! pipeline config ([`config`]). The presets (`open_field`, `plus_maze`, `three_chamber`,
//! `grooming`) are lists like any other, so a new setup is a new list, and a new behavior is one
//! more [`modules::Module`].

pub mod arena;
pub mod arena_model;
pub mod config;
pub mod features;
pub mod input;
pub mod maze;
pub mod modules;
pub mod motion;
pub mod output;
pub mod temporal;

pub use config::AssayConfig;
pub use output::{run_dir, Report};

/// A file-name-safe version of `name`: letters, digits, `-` and `_`.
pub(crate) fn recording_safe(name: &str) -> String {
    let s: String = name
        .chars()
        .take(64)
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let s = s.trim_matches('-');
    if s.is_empty() {
        "arena".into()
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests;
