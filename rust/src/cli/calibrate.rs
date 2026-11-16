//! `animalpose data calibrate`: calibrate a multi-camera rig from known 3D landmarks (arena
//! corners, a calibration object) and write `cams.json` for `animalpose predict --views` — the
//! port of `src/animalpose/cli/data/calibrate.py`.
//!
//! Input JSON:
//!
//! ```text
//! {"cameras": [
//!    {"name": "top", "size": [640, 480],
//!     "points3d": [[x, y, z], ...],         # world units (e.g. cm), >= 6, non-coplanar
//!     "points2d": [[u, v], ...],
//!     "K": [[...]], "dist": [...]           # optional: known intrinsics -> PnP instead of DLT
//!    }, ...]}
//! ```
//!
//! `--refine obs.npz` (an `obs` array of shape `(V, N, 2)`, NaN = unseen) refines all extrinsics
//! jointly by bundle adjustment over matched observations, e.g. confident keypoints of a single
//! animal seen by all cameras.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use clap::{Arg, ArgMatches, Command};
use serde_json::Value;

use crate::multiview::calibration::{
    bundle_adjust, calibrate_dlt, reprojection_errors, solve_extrinsics, BundleOptions,
};
use crate::multiview::camera::{dumps_cameras, Camera};
use crate::multiview::dense::{Mat3, Vec3};
use crate::multiview::npz::read_npz_f64;
use crate::multiview::triangulation::triangulate_dlt;

/// `NAME`: the step name under `animalpose data`.
pub const NAME: &str = "calibrate";

/// The `data calibrate` subcommand (`add_arguments`).
pub fn command() -> Command {
    Command::new(NAME)
        .about("calibrate cameras from known 3D landmarks and write cams.json")
        .arg(
            Arg::new("points")
                .required(true)
                .value_name("POINTS")
                .help("rig_points.json: per camera points3d, points2d and optional K / dist"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .default_value("cams.json")
                .value_name("JSON")
                .help("where the Camera.to_dict() list goes"),
        )
        .arg(
            Arg::new("refine")
                .long("refine")
                .value_name("NPZ")
                .help("obs.npz with obs (V, N, 2), NaN = unseen: bundle-adjust the extrinsics"),
        )
}

/// `run(args)`.
pub fn run(m: &ArgMatches) -> Result<()> {
    let points = m
        .get_one::<String>("points")
        .context("`data calibrate` needs the points JSON")?;
    let out = m
        .get_one::<String>("out")
        .context("`data calibrate` needs --out")?;
    calibrate(
        Path::new(points),
        Path::new(out),
        m.get_one::<String>("refine").map(Path::new),
    )?;
    Ok(())
}

fn numbers(v: &Value, what: &str) -> Result<Vec<f64>> {
    let Some(items) = v.as_array() else {
        bail!("{what} must be a list of numbers");
    };
    items
        .iter()
        .map(|x| {
            x.as_f64()
                .with_context(|| format!("{what} holds a non-number {x}"))
        })
        .collect()
}

fn rows<const N: usize>(v: Option<&Value>, what: &str) -> Result<Vec<[f64; N]>> {
    let Some(items) = v.and_then(Value::as_array) else {
        bail!("camera entry has no {what} list");
    };
    items
        .iter()
        .map(|row| {
            let r = numbers(row, what)?;
            ensure!(
                r.len() == N,
                "every {what} entry needs {N} numbers, got {}",
                r.len()
            );
            let mut out = [0.0; N];
            out.copy_from_slice(&r);
            Ok(out)
        })
        .collect()
}

/// Calibrates every camera of `points`, optionally bundle-adjusts with `refine`, writes `out`.
pub fn calibrate(points: &Path, out: &Path, refine: Option<&Path>) -> Result<Vec<Camera>> {
    let text = std::fs::read_to_string(points)
        .with_context(|| format!("cannot read {}", points.display()))?;
    let spec: Value =
        serde_json::from_str(&text).with_context(|| format!("{} is not JSON", points.display()))?;
    let entries = spec
        .get("cameras")
        .and_then(Value::as_array)
        .with_context(|| format!("{} has no \"cameras\" list", points.display()))?;
    let mut cams = Vec::with_capacity(entries.len());
    for c in entries {
        let name = c
            .get("name")
            .and_then(Value::as_str)
            .context("every camera needs a \"name\"")?;
        let x3: Vec<Vec3> = rows::<3>(c.get("points3d"), "points3d")?;
        let x2: Vec<[f64; 2]> = rows::<2>(c.get("points2d"), "points2d")?;
        let size = match c.get("size") {
            Some(s) => {
                let s = numbers(s, "size")?;
                ensure!(s.len() == 2, "size must be [width, height]");
                (s[0] as usize, s[1] as usize)
            }
            None => (640, 480),
        };
        let cam = match c.get("K") {
            Some(k) => {
                let kr = rows::<3>(Some(k), "K")?;
                ensure!(kr.len() == 3, "K must be 3x3");
                let k: Mat3 = [kr[0], kr[1], kr[2]];
                let dist = match c.get("dist") {
                    Some(d) => numbers(d, "dist")?,
                    None => vec![0.0; 5],
                };
                solve_extrinsics(&k, &dist, &x3, &x2, size, name)?
            }
            None => calibrate_dlt(&x3, &x2, size, name)?,
        };
        let err =
            &reprojection_errors(std::slice::from_ref(&cam), &x3, std::slice::from_ref(&x2))[0];
        let mean = err.iter().sum::<f64>() / err.len().max(1) as f64;
        let max = err.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        println!("{name}: mean reprojection error {mean:.2}px (max {max:.2})");
        cams.push(cam);
    }
    if let Some(refine) = refine {
        let (shape, values) = read_npz_f64(refine, "obs")?;
        ensure!(
            shape.len() == 3 && shape[2] == 2 && shape[0] == cams.len(),
            "{} obs must be (V={}, N, 2), got {shape:?}",
            refine.display(),
            cams.len()
        );
        let n = shape[1];
        let obs: Vec<Vec<[f64; 2]>> = (0..shape[0])
            .map(|v| {
                (0..n)
                    .map(|j| [values[(v * n + j) * 2], values[(v * n + j) * 2 + 1]])
                    .collect()
            })
            .collect();
        let x0 = triangulate_dlt(&cams, &obs, None);
        let ok: Vec<usize> = (0..n)
            .filter(|j| x0[*j].iter().all(|v| v.is_finite()))
            .collect();
        let pts: Vec<Vec3> = ok.iter().map(|j| x0[*j]).collect();
        let sub: Vec<Vec<[f64; 2]>> = obs
            .iter()
            .map(|row| ok.iter().map(|j| row[*j]).collect())
            .collect();
        let (refined, _, info) = bundle_adjust(&cams, &pts, &sub, BundleOptions::default());
        println!(
            "bundle adjustment: {:.2}px -> {:.2}px",
            info.rmse_before, info.rmse_after
        );
        cams = refined;
    }
    std::fs::write(out, dumps_cameras(&cams))
        .with_context(|| format!("cannot write {}", out.display()))?;
    println!("wrote {}", out.display());
    Ok(cams)
}
