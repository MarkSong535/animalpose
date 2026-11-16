//! Multi-camera 3D: cameras, calibration, triangulation, cross-view identity fusion, left/right
//! disambiguation and the multi-view pipeline — the port of `src/animalpose/multiview/` and of
//! `pipeline.py:MultiViewPipeline`.
//!
//! | Python | Rust |
//! | --- | --- |
//! | `multiview/camera.py:Camera` / `rodrigues` / `rotation_to_rvec` / `essential_matrix` | [`camera::Camera`] / [`camera::rodrigues`] / [`camera::rotation_to_rvec`] / [`camera::essential_matrix`] |
//! | `multiview/triangulation.py:triangulate_dlt` / `triangulate_robust` | [`triangulation::triangulate_dlt`] / [`triangulation::triangulate_robust`] |
//! | `multiview/calibration.py:calibrate_dlt` / `solve_extrinsics` / `bundle_adjust` / `reprojection_errors` | [`calibration::calibrate_dlt`] / [`calibration::solve_extrinsics`] / [`calibration::bundle_adjust`] / [`calibration::reprojection_errors`] |
//! | `multiview/calibration.py:calibrate_intrinsics_checkerboard` | [`calibration::calibrate_intrinsics_points`] (corners given; detection not ported) |
//! | `multiview/fusion.py:ViewTrack` / `epipolar_cost` / `fuse_identities` / `triangulate_identities` | [`fusion::ViewTrack`] / [`fusion::epipolar_cost`] / [`fusion::fuse_identities`] / [`fusion::triangulate_identities`] |
//! | `multiview/symmetry.py:F_SHARED_X` / `fundamental_from_cameras` / `epipolar_distance` / `resolve_left_right` / `resolve_left_right_cameras` | [`symmetry::F_SHARED_X`] / [`symmetry::fundamental_from_cameras`] / [`symmetry::epipolar_distance`] / [`symmetry::resolve_left_right`] / [`symmetry::resolve_left_right_cameras`] |
//! | `pipeline.py:MultiViewPipeline`, `api.py:predict(views=...)` | [`pipeline::MultiViewPipeline`], [`pipeline::run_cli`] |
//! | `cli/data/calibrate.py` | [`crate::cli::calibrate`] |
//!
//! `cams.json` is the same file in both implementations: a JSON list of `Camera.to_dict()`.
//!
//! `// differs from Python:` numpy/LAPACK, scipy's `least_squares` and OpenCV's `solvePnP` /
//! `calibrateCamera` are replaced by the small solvers of [`dense`] and [`solve`]; see those
//! modules for what that changes (nothing at the minimum, the iteration count on the way).
//! Checkerboard corner detection is not ported.

// Small fixed-size matrix code reads closest to the numpy formulas with explicit indices.
#![allow(clippy::needless_range_loop)]

pub mod calibration;
pub mod camera;
pub mod dense;
pub mod fusion;
pub mod npz;
pub mod pipeline;
pub mod solve;
pub mod symmetry;
pub mod triangulation;
