//! Numeric parity of `animalpose::multiview` with `src/animalpose/multiview`.
//!
//! `multiview_parity.json` holds the Python numbers, written by `multiview_parity.py` next to this
//! file (`PYTHONPATH=src python rust/tests/multiview_parity.py`). The scenes are closed-form
//! (`body_pose` walking around the arena, noise `amp * sin(1.7 i + seed)`), so the larger inputs
//! are rebuilt here rather than stored.

use std::collections::BTreeMap;

use animalpose::data::synthetic::{body_pose, FLIP_INDEX, NUM_KEYPOINTS as K};
use animalpose::multiview::calibration::{
    bundle_adjust, calibrate_dlt, calibrate_intrinsics_checkerboard, calibrate_intrinsics_points,
    reprojection_errors, solve_extrinsics, BundleOptions,
};
use animalpose::multiview::camera::{
    dumps_cameras, essential_matrix, rodrigues, rotation_to_rvec, Camera,
};
use animalpose::multiview::dense::{Mat3, Vec3};
use animalpose::multiview::fusion::{
    epipolar_cost, fuse_identities, triangulate_identities, ViewTrack,
};
use animalpose::multiview::symmetry::{
    epipolar_distance, fundamental_from_cameras, resolve_left_right_cameras, ResolveOptions,
    F_SHARED_X,
};
use animalpose::multiview::triangulation::{triangulate_dlt, triangulate_robust};
use serde_json::Value;

fn reference() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/multiview_parity.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("the reference file exists"))
        .expect("JSON")
}

/// `pose(a, t)` of the generator: one animal's 25 world keypoints.
fn pose(a: usize, t: usize) -> Vec<Vec3> {
    let t = t as f64;
    let params = [0.2, 0.1, 1.0, 0.0, 0.0, 0.0];
    let body = body_pose(&params, 0.4 * t, 0.3 * t, 0.1 * (0.1 * t).sin(), 0.2 * t);
    let h = 0.05 * t + std::f64::consts::PI * a as f64;
    let (c, s) = (h.cos(), h.sin());
    let px = 25.0 * (0.03 * t + 2.0 * a as f64).cos();
    let py = 25.0 * (0.03 * t + 2.0 * a as f64).sin();
    body.iter()
        .map(|b| {
            let b = [b[0] * 20.0, b[1] * 20.0, b[2] * 20.0];
            [c * b[0] - s * b[1] + px, s * b[0] + c * b[1] + py, b[2]]
        })
        .collect()
}

/// `noise(shape, amp, seed)` flattened.
fn noise(n: usize, amp: f64, seed: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (1.7 * i as f64 + seed).sin())
        .collect()
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("a number")
}

/// Reads a nested list of numbers, mapping the generator's `-1e9` / `1e300` back to NaN / inf.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(items) => items.iter().flat_map(flat).collect(),
        other => {
            let x = f(other);
            vec![if x == -1e9 {
                f64::NAN
            } else if x == 1e300 {
                f64::INFINITY
            } else {
                x
            }]
        }
    }
}

fn uv2(v: &Value, views: usize) -> Vec<Vec<[f64; 2]>> {
    let x = flat(v);
    let n = x.len() / (2 * views);
    (0..views)
        .map(|i| {
            (0..n)
                .map(|j| [x[(i * n + j) * 2], x[(i * n + j) * 2 + 1]])
                .collect()
        })
        .collect()
}

fn rows(v: &Value, views: usize) -> Vec<Vec<f64>> {
    let x = flat(v);
    let n = x.len() / views;
    x.chunks(n).map(<[f64]>::to_vec).collect()
}

/// `np.testing.assert_allclose(got, want, rtol, atol)`, NaNs equal.
fn close(got: &[f64], want: &[f64], rtol: f64, atol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() || w.is_infinite() {
            assert!(
                g.is_nan() == w.is_nan() && (g == w || w.is_nan()),
                "{what}[{i}]: {g} vs {w}"
            );
            continue;
        }
        assert!(
            (g - w).abs() <= atol + rtol * w.abs(),
            "{what}[{i}]: {g} vs {w}"
        );
    }
}

fn cameras(r: &Value) -> Vec<Camera> {
    r["cameras"]
        .as_array()
        .expect("cameras")
        .iter()
        .map(|d| Camera::from_json(d).expect("a camera"))
        .collect()
}

fn m3(m: &Mat3) -> Vec<f64> {
    m.iter().flatten().copied().collect()
}

#[test]
fn camera_projection_distortion_and_json() {
    let r = reference();
    let cams = cameras(&r);
    let p = &r["project"];
    let x: Vec<Vec3> = flat(&p["X"])
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    close(
        &x.iter().flatten().copied().collect::<Vec<_>>(),
        &flat(&p["X"]),
        1e-12,
        1e-12,
        "X",
    );
    let rust_pose: Vec<f64> = pose(0, 7).iter().flatten().copied().collect();
    close(&rust_pose, &flat(&p["X"]), 1e-12, 1e-11, "pose(0, 7)");
    let mut uv = Vec::new();
    let mut z = Vec::new();
    for xi in &x {
        let (q, d) = cams[1].project_with_depth(xi);
        uv.extend(q);
        z.push(d);
    }
    close(&uv, &flat(&p["uv"]), 1e-12, 1e-9, "project");
    close(&z, &flat(&p["z"]), 1e-12, 1e-9, "depth");
    let pts: Vec<[f64; 2]> = uv.chunks(2).map(|c| [c[0], c[1]]).collect();
    let und: Vec<f64> = cams[1]
        .undistort_points(&pts)
        .iter()
        .flatten()
        .copied()
        .collect();
    close(&und, &flat(&p["undistorted"]), 1e-10, 1e-12, "undistort");
    let pm: Vec<f64> = cams[2].p().iter().flatten().copied().collect();
    close(&pm, &flat(&p["P"]), 1e-12, 1e-9, "P");
    close(
        &cams[2].center(),
        &flat(&p["center"]),
        1e-12,
        1e-9,
        "center",
    );
    close(
        &m3(&essential_matrix(&cams[0], &cams[1])),
        &flat(&p["E01"]),
        1e-12,
        1e-9,
        "E",
    );

    // `json.dumps([cam.to_dict()], indent=2)`, byte for byte.
    assert_eq!(
        dumps_cameras(std::slice::from_ref(&cams[1])),
        r["camera_json"].as_str().expect("text")
    );

    let rv = &r["rodrigues"];
    let rvec = flat(&rv["rvec"]);
    let rot = rodrigues(&[rvec[0], rvec[1], rvec[2]]);
    close(&m3(&rot), &flat(&rv["R"]), 1e-12, 1e-14, "rodrigues");
    close(
        &rotation_to_rvec(&rot),
        &flat(&rv["back"]),
        1e-10,
        1e-12,
        "rotation_to_rvec",
    );
    let half = flat(&rv["half"]);
    close(
        &rotation_to_rvec(&rodrigues(&[half[0], half[1], half[2]])),
        &flat(&rv["half_back"]),
        1e-6,
        1e-7,
        "half turn",
    );

    let look = Camera::look_at(
        [50.0, 40.0, 60.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        600.0,
        (320, 240),
        Some(&[0.05, -0.01, 0.0, 0.0, 0.002]),
        "look",
    );
    let want = Camera::from_json(&r["look_at"]).expect("camera");
    close(&m3(&look.k), &m3(&want.k), 1e-14, 0.0, "look_at K");
    close(&m3(&look.r), &m3(&want.r), 1e-12, 1e-14, "look_at R");
    close(&look.t, &want.t, 1e-12, 1e-12, "look_at t");
    assert_eq!(
        (look.dist, look.size, look.name.as_str()),
        (want.dist, want.size, "look")
    );
}

#[test]
fn dlt_triangulation_matches_python() {
    let r = reference();
    let cams = cameras(&r);
    let d = &r["dlt"];
    let uv = uv2(&d["uv"], 3);
    let w = rows(&d["w"], 3);
    let x: Vec<f64> = triangulate_dlt(&cams, &uv, Some(&w))
        .iter()
        .flatten()
        .copied()
        .collect();
    close(&x, &flat(&d["X"]), 1e-9, 1e-9, "weighted DLT");
    assert!(
        x[4 * 3].is_nan() && x[3 * 3].is_finite(),
        "one view is not enough, two are"
    );
    let x: Vec<f64> = triangulate_dlt(&cams, &uv, None)
        .iter()
        .flatten()
        .copied()
        .collect();
    close(&x, &flat(&d["X_unweighted"]), 1e-9, 1e-9, "unweighted DLT");
}

#[test]
fn ransac_rejects_the_corrupted_view() {
    let r = reference();
    let mut cams = cameras(&r);
    let rb = &r["robust"];
    cams.push(Camera::from_json(&rb["extra"]).expect("camera"));
    let uv = uv2(&rb["uv"], 4);
    let conf = rows(&rb["conf"], 4);
    let got = triangulate_robust(&cams, &uv, Some(&conf), 0.3, 10.0);
    close(
        &got.x.iter().flatten().copied().collect::<Vec<_>>(),
        &flat(&rb["X"]),
        1e-9,
        1e-9,
        "X",
    );
    close(&got.err, &flat(&rb["err"]), 1e-8, 1e-10, "err");
    let used: Vec<f64> = got
        .used
        .iter()
        .flatten()
        .map(|u| f64::from(u8::from(*u)))
        .collect();
    assert_eq!(used, flat(&rb["used"]));
    assert!(got.used[1][..5].iter().all(|u| !u) && got.used[1][5..].iter().all(|u| *u));
}

#[test]
fn dlt_calibration_and_pnp_recover_the_camera() {
    let r = reference();
    let cams = cameras(&r);
    let c = &r["dlt_calibration"];
    let x3: Vec<Vec3> = flat(&c["X"])
        .chunks(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let x2: Vec<[f64; 2]> = flat(&c["x"]).chunks(2).map(|v| [v[0], v[1]]).collect();
    let est = calibrate_dlt(&x3, &x2, cams[2].size, "angled").expect("calibrates");
    close(&m3(&est.k), &flat(&c["K"]), 1e-8, 1e-8, "K");
    close(&m3(&est.r), &flat(&c["R"]), 1e-8, 1e-10, "R");
    close(&est.t, &flat(&c["t"]), 1e-8, 1e-8, "t");
    let err = reprojection_errors(std::slice::from_ref(&est), &x3, std::slice::from_ref(&x2));
    close(&err[0], &flat(&c["err"]), 1e-7, 1e-9, "reprojection error");
    // Exact correspondences: the camera itself.
    let exact: Vec<[f64; 2]> = x3.iter().map(|p| cams[2].project(p)).collect();
    let est = calibrate_dlt(&x3, &exact, cams[2].size, "angled").expect("calibrates");
    close(&m3(&est.k), &m3(&cams[2].k), 1e-8, 1e-6, "exact K");
    close(&est.center(), &cams[2].center(), 1e-8, 1e-6, "exact center");
    close(
        &m3(&est.k),
        &flat(&c["exact_K"]),
        1e-8,
        1e-6,
        "exact K vs Python",
    );
    assert!(calibrate_dlt(&x3[..5], &x2[..5], (640, 480), "few")
        .unwrap_err()
        .to_string()
        .contains("at least 6 correspondences"));

    // PnP: `cv2.solvePnP(SOLVEPNP_ITERATIVE)` with a distorted lens, non-planar and planar targets.
    let p = &r["pnp"];
    let obs: Vec<[f64; 2]> = flat(&p["x"]).chunks(2).map(|v| [v[0], v[1]]).collect();
    let cam =
        solve_extrinsics(&cams[1].k, &cams[1].dist, &x3, &obs, cams[1].size, "side").expect("PnP");
    close(&m3(&cam.r), &flat(&p["R"]), 1e-6, 1e-7, "PnP R");
    close(&cam.t, &flat(&p["t"]), 1e-6, 1e-6, "PnP t");
    let planar: Vec<Vec3> = flat(&p["planar_X"])
        .chunks(3)
        .map(|v| [v[0], v[1], v[2]])
        .collect();
    let obs: Vec<[f64; 2]> = flat(&p["planar_x"])
        .chunks(2)
        .map(|v| [v[0], v[1]])
        .collect();
    let cam = solve_extrinsics(
        &cams[0].k,
        &cams[0].dist,
        &planar,
        &obs,
        cams[0].size,
        "top",
    )
    .expect("PnP");
    close(
        &m3(&cam.r),
        &flat(&p["planar_R"]),
        1e-6,
        1e-7,
        "planar PnP R",
    );
    close(&cam.t, &flat(&p["planar_t"]), 1e-6, 1e-6, "planar PnP t");
}

#[test]
fn intrinsics_from_board_corners_match_opencv() {
    let r = reference();
    let c = &r["intrinsics"];
    let views = c["obj"].as_array().expect("views").len();
    let obj: Vec<Vec<Vec3>> = c["obj"]
        .as_array()
        .expect("views")
        .iter()
        .map(|v| flat(v).chunks(3).map(|p| [p[0], p[1], p[2]]).collect())
        .collect();
    let img: Vec<Vec<[f64; 2]>> = c["img"]
        .as_array()
        .expect("views")
        .iter()
        .map(|v| flat(v).chunks(2).map(|p| [p[0], p[1]]).collect())
        .collect();
    assert_eq!(views, 6);
    let (k, dist, rms) = calibrate_intrinsics_points(&obj, &img, (520, 400)).expect("calibrates");
    // `cv2.calibrateCamera`: the same minimum of the same reprojection error.
    close(&[rms], &[f(&c["rms"])], 1e-6, 0.0, "rms");
    close(&m3(&k), &flat(&c["K"]), 1e-6, 1e-6, "K");
    close(&dist, &flat(&c["dist"]), 1e-4, 1e-6, "dist");
    assert!(
        calibrate_intrinsics_points(&obj[..2], &img[..2], (520, 400))
            .unwrap_err()
            .to_string()
            .contains("found in only 2 images; need >= 3")
    );
    assert!(calibrate_intrinsics_checkerboard()
        .unwrap_err()
        .to_string()
        .starts_with("unsupported"));
}

/// The bundle-adjustment scene: 1000 points, three views, the first 40 points unseen by view 0.
fn bundle_scene(cams: &[Camera]) -> (Vec<Vec3>, Vec<Vec<[f64; 2]>>) {
    let xb: Vec<Vec3> = (0..2)
        .flat_map(|a| (0..60).step_by(3).flat_map(move |t| pose(a, t)))
        .collect();
    let nz = noise(3 * xb.len() * 2, 0.3, 7.0);
    let mut obs: Vec<Vec<[f64; 2]>> = cams
        .iter()
        .enumerate()
        .map(|(v, c)| {
            xb.iter()
                .enumerate()
                .map(|(j, p)| {
                    let q = c.project(p);
                    let at = (v * xb.len() + j) * 2;
                    [q[0] + nz[at], q[1] + nz[at + 1]]
                })
                .collect()
        })
        .collect();
    for p in obs[0].iter_mut().take(40) {
        *p = [f64::NAN; 2];
    }
    (xb, obs)
}

#[test]
fn bundle_adjustment_reduces_the_reprojection_error_like_python() {
    let r = reference();
    let cams = cameras(&r);
    let b = &r["bundle"];
    let noisy: Vec<Camera> = b["noisy"]
        .as_array()
        .expect("cameras")
        .iter()
        .map(|d| Camera::from_json(d).expect("camera"))
        .collect();
    let (xb, obs) = bundle_scene(&cams);
    let x0 = triangulate_dlt(&noisy, &obs, None);
    close(
        &x0[..5].iter().flatten().copied().collect::<Vec<_>>(),
        &flat(&b["X0_head"]),
        1e-9,
        1e-9,
        "X0",
    );
    let (refined, _pts, info) = bundle_adjust(&noisy, &x0, &obs, BundleOptions::default());
    let want = &b["info"];
    close(
        &[info.rmse_before],
        &[f(&want["rmse_before"])],
        1e-9,
        0.0,
        "rmse before",
    );
    // Same minimum, different path (scipy's TRF vs Levenberg-Marquardt): 0.1146807 px both.
    assert!(
        info.rmse_after < 1.0 && f(&want["rmse_after"]) < 1.0 && info.rmse_after < info.rmse_before
    );
    close(
        &[info.rmse_after],
        &[f(&want["rmse_after"])],
        1e-6,
        0.0,
        "rmse after",
    );
    let python: Vec<Camera> = b["refined"]
        .as_array()
        .expect("cameras")
        .iter()
        .map(|d| Camera::from_json(d).expect("camera"))
        .collect();
    assert_eq!(refined[0], noisy[0], "the first camera fixes the gauge");
    for (got, want) in refined.iter().zip(&python).skip(1) {
        // With the points refined only the first camera fixes the gauge, so the overall scale is
        // free: the two solvers stop ~0.1 mm apart along it, at the same reprojection error.
        close(
            &got.center(),
            &want.center(),
            0.0,
            0.05,
            "refined centre (cm)",
        );
    }

    // Points held fixed: only the extrinsics move.
    let (fixed, pts, info) = bundle_adjust(
        &noisy,
        &xb,
        &obs,
        BundleOptions {
            refine_points: false,
            ..BundleOptions::default()
        },
    );
    let want = &b["info_fixed_points"];
    assert_eq!(pts, xb);
    close(
        &[info.rmse_before],
        &[f(&want["rmse_before"])],
        1e-9,
        0.0,
        "fixed rmse before",
    );
    close(
        &[info.rmse_after],
        &[f(&want["rmse_after"])],
        1e-6,
        0.0,
        "fixed rmse after",
    );
    let python: Vec<Camera> = b["fixed_points"]
        .as_array()
        .expect("cameras")
        .iter()
        .map(|d| Camera::from_json(d).expect("camera"))
        .collect();
    for (got, want) in fixed.iter().zip(&python) {
        close(&m3(&got.r), &m3(&want.r), 0.0, 1e-7, "fixed-points R");
        close(&got.t, &want.t, 0.0, 1e-5, "fixed-points t");
    }
}

/// The fusion scene: two animals, three views, animal 0 fragmented in view 1.
fn fusion_tracks(cams: &[Camera]) -> (Vec<ViewTrack>, Vec<Vec<Vec<Vec3>>>) {
    let n = 60;
    let world: Vec<Vec<Vec<Vec3>>> = (0..2)
        .map(|a| (0..n).map(|t| pose(a, t)).collect())
        .collect();
    let mut tracks = Vec::new();
    for (v, cam) in cams.iter().enumerate() {
        for (a, frames) in world.iter().enumerate() {
            let nz = noise(n * K * 2, 0.5, 10.0 * v as f64 + a as f64);
            let kp: Vec<Vec<[f64; 2]>> = frames
                .iter()
                .enumerate()
                .map(|(t, pts)| {
                    pts.iter()
                        .enumerate()
                        .map(|(k, p)| {
                            let q = cam.project(p);
                            let at = (t * K + k) * 2;
                            [q[0] + nz[at], q[1] + nz[at + 1]]
                        })
                        .collect()
                })
                .collect();
            let track = |id: i64, lo: usize, hi: usize| ViewTrack {
                view: v,
                track_id: id,
                frames: (lo..hi).collect(),
                keypoints: kp[lo..hi].to_vec(),
                conf: vec![vec![1.0; K]; hi - lo],
            };
            if v == 1 && a == 0 {
                tracks.push(track(100, 0, 25));
                tracks.push(track(101, 30, n));
            } else {
                tracks.push(track(((a + v) % 2 + 10 * v) as i64, 0, n));
            }
        }
    }
    (tracks, world)
}

fn ids_of(v: &Value) -> BTreeMap<(usize, i64), usize> {
    v.as_array()
        .expect("ids")
        .iter()
        .map(|row| {
            let row = row.as_array().expect("row");
            (
                (
                    row[0].as_u64().expect("view") as usize,
                    row[1].as_i64().expect("id"),
                ),
                row[2].as_u64().expect("gid") as usize,
            )
        })
        .collect()
}

#[test]
fn fusion_stitches_a_fragmented_track_into_two_animals() {
    let r = reference();
    let cams = cameras(&r);
    let fu = &r["fusion"];
    let (tracks, world) = fusion_tracks(&cams);
    let order: Vec<(usize, i64)> = tracks.iter().map(|t| (t.view, t.track_id)).collect();
    let want_order: Vec<(usize, i64)> = fu["tracks"]
        .as_array()
        .expect("tracks")
        .iter()
        .map(|t| {
            (
                t[0].as_u64().expect("view") as usize,
                t[1].as_i64().expect("id"),
            )
        })
        .collect();
    assert_eq!(order, want_order);
    let costs = rows(&fu["costs"], tracks.len());
    for (i, a) in tracks.iter().enumerate() {
        for (j, b) in tracks.iter().enumerate() {
            if a.view != b.view {
                let c = epipolar_cost(a, b, &cams[a.view], &cams[b.view]);
                close(&[c], &[costs[i][j]], 1e-9, 1e-9, "epipolar cost");
            }
        }
    }
    // Tracks that never share a frame have no epipolar evidence at all.
    let mut later = tracks[4].clone(); // animal 1 in view 1, the whole clip
    later.frames.iter_mut().for_each(|f| *f += 100);
    assert_eq!(later.view, 1);
    assert!(epipolar_cost(&tracks[1], &tracks[4], &cams[0], &cams[1]) < 2.0);
    assert!(epipolar_cost(&tracks[1], &later, &cams[0], &cams[1]).is_infinite());

    let ids = fuse_identities(&tracks, &cams, 15.0, None);
    assert_eq!(ids, ids_of(&fu["ids"]));
    assert_eq!(
        ids.values()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2
    );
    assert!(
        ids[&(1, 100)] == ids[&(1, 101)]
            && ids[&(1, 101)] == ids[&(0, 0)]
            && ids[&(0, 0)] != ids[&(0, 1)]
    );
    assert_eq!(
        fuse_identities(&tracks, &cams, 0.0, Some(2)),
        ids_of(&fu["ids_forced"])
    );

    let out = triangulate_identities(&tracks, &cams, &ids, 60, 0.3, 10.0, false);
    for (g, a) in &out {
        let want = &fu["animals"][g.to_string()];
        let animal = if ids[&(0, 0)] == *g { 0 } else { 1 };
        let mut max_err: f64 = 0.0;
        let mut reproj = Vec::new();
        for (t, frame) in a.keypoints3d.iter().enumerate() {
            for (k, p) in frame.iter().enumerate() {
                let w = world[animal][t][k];
                let e =
                    ((p[0] - w[0]).powi(2) + (p[1] - w[1]).powi(2) + (p[2] - w[2]).powi(2)).sqrt();
                if e.is_finite() {
                    max_err = max_err.max(e);
                }
                if a.reproj_error[t][k].is_finite() {
                    reproj.push(a.reproj_error[t][k]);
                }
            }
        }
        close(
            &[max_err],
            &[f(&want["max_err_cm"])],
            1e-8,
            1e-10,
            "max error",
        );
        assert!(max_err < 0.5, "triangulated to within 5 mm");
        let mean = reproj.iter().sum::<f64>() / reproj.len() as f64;
        close(
            &[mean],
            &[f(&want["mean_reproj"])],
            1e-8,
            1e-10,
            "mean reprojection error",
        );
        let n_views: i64 = a.n_views.iter().flatten().sum();
        assert_eq!(n_views, want["n_views_sum"].as_i64().expect("sum"));
        let f33: Vec<f64> = a.keypoints3d[33].iter().flatten().copied().collect();
        close(&f33, &flat(&want["frame33"]), 1e-9, 1e-9, "frame 33");
        let f27: Vec<f64> = a.keypoints3d[27].iter().flatten().copied().collect();
        close(&f27, &flat(&want["frame27"]), 1e-9, 1e-9, "frame 27");
    }
}

#[test]
fn symmetry_fixes_a_swapped_view() {
    let r = reference();
    let cams = cameras(&r);
    let s = &r["symmetry"];
    let bad = uv2(&s["bad"], 3);
    let conf = rows(&s["conf"], 3);
    let got = resolve_left_right_cameras(&bad, &conf, &cams, &ResolveOptions::default());
    close(
        &got.kpts
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &flat(&s["fixed"]),
        0.0,
        0.0,
        "fixed keypoints",
    );
    let swapped: Vec<f64> = got
        .swapped
        .iter()
        .flatten()
        .map(|b| f64::from(u8::from(*b)))
        .collect();
    assert_eq!(swapped, flat(&s["swapped"]));
    assert!(got.swapped[2].iter().all(|b| *b) && got.swapped[..2].iter().flatten().all(|b| !b));
    // The fix restores the projection: flipping view 2 back.
    let fixed2: Vec<[f64; 2]> = (0..K).map(|k| bad[2][FLIP_INDEX[k]]).collect();
    assert_eq!(got.kpts[2], fixed2);

    let f01 = fundamental_from_cameras(&cams[0], &cams[1]);
    close(&m3(&f01), &flat(&s["F01"]), 1e-9, 1e-15, "F");
    let pa: Vec<[f64; 2]> = flat(&s["pa"]).chunks(2).map(|c| [c[0], c[1]]).collect();
    let pb: Vec<[f64; 2]> = flat(&s["pb"]).chunks(2).map(|c| [c[0], c[1]]).collect();
    let d: Vec<f64> = pa
        .iter()
        .zip(&pb)
        .map(|(a, b)| epipolar_distance(&f01, *a, *b))
        .collect();
    close(&d, &flat(&s["dist"]), 1e-7, 1e-9, "epipolar distance");
    let shared = [
        epipolar_distance(&F_SHARED_X, [5.0, 1.0], [5.0, 90.0]),
        epipolar_distance(&F_SHARED_X, [3.0, 2.0], [7.0, 1.0]),
    ];
    close(&shared, &flat(&s["shared_x"]), 1e-12, 1e-12, "shared x");
    assert!(shared[0] < 1e-9);
}

#[test]
fn left_right_resolution_repairs_a_mirrored_view_before_triangulation() {
    let r = reference();
    let mut two: Vec<Camera> = cameras(&r).into_iter().take(2).collect();
    two[1].dist = [0.0; 5];
    let m = &r["mirror"];
    let n = 24;
    let truth: Vec<Vec<Vec3>> = (0..n).map(|t| pose(0, t)).collect();
    let tracks: Vec<ViewTrack> = two
        .iter()
        .enumerate()
        .map(|(v, c)| ViewTrack {
            view: v,
            track_id: 0,
            frames: (0..n).collect(),
            keypoints: truth
                .iter()
                .map(|pts| {
                    let uv = c.project_many(pts);
                    if v == 1 {
                        (0..K).map(|k| uv[FLIP_INDEX[k]]).collect()
                    } else {
                        uv
                    }
                })
                .collect(),
            conf: vec![vec![if v == 1 { 0.6 } else { 1.0 }; K]; n],
        })
        .collect();
    let ids = BTreeMap::from([((0, 0), 0), ((1, 0), 0)]);
    let max_err = |resolve: bool| {
        let out = triangulate_identities(&tracks, &two, &ids, n, 0.3, 1e9, resolve);
        let a = &out[&0];
        let mut worst: f64 = 0.0;
        for (t, frame) in a.keypoints3d.iter().enumerate() {
            for (k, p) in frame.iter().enumerate() {
                let w = truth[t][k];
                worst = worst.max(
                    ((p[0] - w[0]).powi(2) + (p[1] - w[1]).powi(2) + (p[2] - w[2]).powi(2)).sqrt(),
                );
            }
        }
        (
            worst,
            a.keypoints3d[5]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        )
    };
    let (naive, frame5) = max_err(false);
    let (fixed, _) = max_err(true);
    close(&[naive], &[f(&m["naive_max"])], 1e-8, 1e-10, "naive error");
    close(
        &frame5,
        &flat(&m["naive_frame5"]),
        1e-9,
        1e-9,
        "naive frame 5",
    );
    assert!(naive > 1.0, "two views cannot outvote a mirrored one");
    assert!(
        fixed < 0.1 && f(&m["fixed_max"]) < 0.1,
        "relabelling first recovers the animal: {fixed}"
    );
}
