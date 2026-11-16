//! Ported from `tests/test_behavior_naming.py` and the cluster parts of
//! `tests/test_viz_and_clusters.py` / `tests/test_cli_behavior.py`, with reference numbers from
//! the Python package (`kinematics` and `cluster_tracks` on the same synthetic recording).

#![allow(clippy::excessive_precision)] // reference floats pasted from Python with all their digits

use super::*;
use crate::data::sessions::{read_npz, Session};

pub(crate) const FPS: f64 = 30.0;
pub(crate) const T: usize = 120;
pub(crate) const MOVE: usize = 60;

const OFFSETS: [(&str, (f64, f64)); 25] = [
    ("nose", (20.0, 0.0)),
    ("head_center", (15.0, 0.0)),
    ("neck", (11.0, 0.0)),
    ("spine_upper", (7.0, 0.0)),
    ("spine_mid", (0.0, 0.0)),
    ("spine_lower", (-7.0, 0.0)),
    ("tail_base", (-20.0, 0.0)),
    ("tail_mid", (-28.0, 0.0)),
    ("tail_end", (-35.0, 0.0)),
    ("left_eye", (17.0, 2.0)),
    ("right_eye", (17.0, -2.0)),
    ("left_ear", (13.0, 3.0)),
    ("right_ear", (13.0, -3.0)),
    ("left_shoulder", (7.0, 4.0)),
    ("right_shoulder", (7.0, -4.0)),
    ("left_front_knee", (11.0, 6.0)),
    ("right_front_knee", (11.0, -6.0)),
    ("left_front_paw", (14.0, 7.0)),
    ("right_front_paw", (14.0, -7.0)),
    ("left_hip", (-7.0, 4.0)),
    ("right_hip", (-7.0, -4.0)),
    ("left_hind_knee", (-11.0, 6.0)),
    ("right_hind_knee", (-11.0, -6.0)),
    ("left_hind_paw", (-14.0, 7.0)),
    ("right_hind_paw", (-14.0, -7.0)),
];

/// `body(centre, curl)`: a 40 px animal pointing +x, its rear half curled by `curl` radians
/// (`float32`, as the Python fixture stores it).
pub(crate) fn body(centre: [f64; 2], curl: f64) -> Vec<[f64; 2]> {
    let f = |v: f64| f64::from(v as f32);
    let mut kp = vec![[0.0; 2]; NUM_KEYPOINTS];
    for (name, (a, b)) in OFFSETS {
        kp[index(name)] = [f(centre[0] + a), f(centre[1] + b)];
    }
    if curl != 0.0 {
        let mid = kp[index("spine_mid")];
        let (c, s) = (curl.cos(), curl.sin());
        for name in ["spine_lower", "tail_base", "tail_mid", "tail_end"] {
            let k = index(name);
            let d = [kp[k][0] - mid[0], kp[k][1] - mid[1]];
            kp[k] = [
                f(mid[0] + d[0] * c - d[1] * s),
                f(mid[1] + d[0] * s + d[1] * c),
            ];
        }
    }
    kp
}

/// The recording: runs at 4 px a frame for 2 s, then curls up and stops; the nose wobbles by
/// `0.5 sin(0.7 t)` px (Python: `kp[:, 0, 1] += (0.5 * np.sin(0.7 * np.arange(T))).astype(np.float32)`)
/// so that no feature is constant up to rounding noise, which `StandardScaler` would blow up to
/// unit variance and which no two implementations share.
pub(crate) fn recording_kp() -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut kp = Vec::new();
    for t in 0..T {
        let curl = if t < MOVE { 0.0 } else { 90f64.to_radians() };
        let mut b = body([40.0 + 4.0 * t.min(MOVE - 1) as f64, 120.0], curl);
        let wobble = (0.5 * (0.7 * t as f64).sin()) as f32;
        b[index("nose")][1] = f64::from(b[index("nose")][1] as f32 + wobble);
        kp.extend(b);
    }
    (kp, vec![0.9; T * NUM_KEYPOINTS])
}

pub(crate) fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ap-clusters-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A `tracks.npz` like `predict` writes it, with `track0_behavior_full` over `names`.
pub(crate) fn write_tracks(path: &Path, names: &[String], full: &[f64]) {
    let (kp, conf) = recording_kp();
    let flat: Vec<f32> = kp.iter().flat_map(|p| [p[0] as f32, p[1] as f32]).collect();
    let conf: Vec<f32> = conf.iter().map(|v| *v as f32).collect();
    let times: Vec<f64> = (0..T).map(|t| t as f64 / FPS).collect();
    let frames: Vec<i64> = (0..T as i64).collect();
    write_npz_members(
        path,
        &[
            ("track0_smoothed", npy_f32(&[T, NUM_KEYPOINTS, 2], &flat)),
            ("track0_smoothed_conf", npy_f32(&[T, NUM_KEYPOINTS], &conf)),
            ("track0_frames", npy_i64(&[T], &frames)),
            ("times", npy_f64(&[T], &times)),
            ("track0_behavior_full", npy_f64(&[T, names.len()], full)),
            ("behaviors", npy_str(&[names.len()], names)),
        ],
    )
    .unwrap();
}

/// A dark disc following the animal on a grey floor, as a folder of PNG frames.
pub(crate) fn write_video(dir: &Path) {
    let (kp, _) = recording_kp();
    std::fs::create_dir_all(dir).unwrap();
    for t in 0..T {
        let c = kp[t * NUM_KEYPOINTS + index("spine_mid")];
        let mut img = image::GrayImage::from_pixel(320, 240, image::Luma([150]));
        for y in 0..240u32 {
            for x in 0..320u32 {
                if (f64::from(x) - c[0]).powi(2) + (f64::from(y) - c[1]).powi(2) <= 144.0 {
                    img.put_pixel(x, y, image::Luma([25]));
                }
            }
        }
        img.save(dir.join(format!("f{t:04}.png"))).unwrap();
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

// ------------------------------------------------------------------------------------ naming

fn named() -> Json {
    let rows: [(&str, [f64; 10]); 8] = [
        (
            "0",
            [
                0.390, 0.164, 1.108, 0.265, 0.194, 0.134, 0.131, 0.324, 0.037, 0.236,
            ],
        ),
        (
            "4",
            [
                0.295, 0.148, 0.835, 0.374, 0.235, 0.154, 0.317, 0.236, 0.041, 0.300,
            ],
        ),
        (
            "5",
            [
                0.172, 0.662, 1.083, 0.794, 0.509, 0.329, 0.160, 0.677, 0.154, 0.412,
            ],
        ),
        (
            "3",
            [
                0.097, 0.567, 0.762, 2.018, 0.789, 0.639, 0.351, 0.526, 0.126, 0.522,
            ],
        ),
        (
            "2",
            [
                0.024, 0.615, 0.388, 2.215, 0.839, 1.237, 1.475, 0.522, 0.229, 0.570,
            ],
        ),
        (
            "7",
            [
                0.016, 0.898, 0.510, 7.608, 1.211, 0.821, 0.499, 0.622, 0.248, 0.747,
            ],
        ),
        (
            "1",
            [
                0.004, 37.584, 0.730, 0.387, 38.162, 0.688, 0.939, 94.428, 0.190, 94.000,
            ],
        ),
        (
            "6",
            [
                0.002, 0.676, 0.708, 3.442, 1.206, 0.759, 0.368, 69.425, 0.194, 69.384,
            ],
        ),
    ];
    Json::Obj(
        rows.iter()
            .map(|(c, v)| {
                let mut stats = vec![("share_of_train".to_string(), Json::Float(v[0]))];
                for (i, f) in FEATURES.iter().enumerate() {
                    stats.push((f.to_string(), Json::Float(v[i + 1])));
                }
                (c.to_string(), Json::Obj(stats))
            })
            .collect(),
    )
}

fn human() -> Json {
    Json::obj(
        [
            ("0", "sniffing"),
            ("2", "grooming"),
            ("3", "rearing"),
            ("4", "resting"),
            ("5", "walking"),
            ("7", "turning"),
        ]
        .map(|(c, n)| (c, Json::str(n))),
    )
}

#[test]
fn rule_naming_reproduces_the_hand_named_clusters() {
    let naming = name_clusters(&named());
    let agree = naming_agreement(&naming, &human());
    assert_eq!(agree.get("n_named_clusters"), Some(&Json::Int(6)));
    let w = agree
        .get("weighted_agreement")
        .and_then(Json::as_f64)
        .unwrap();
    assert!(w >= 0.99, "{}", agree.dumps());
}

#[test]
fn rule_naming_rejects_tracking_jumps_and_only_uses_catalogue_names() {
    let naming = name_clusters(&named());
    let name = |c: &str| {
        naming
            .get(c)
            .unwrap()
            .get("name")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(
        (name("1"), name("6")),
        ("other".to_string(), "other".to_string())
    );
    let catalog = Catalog::load();
    let Json::Obj(pairs) = &naming else { panic!() };
    let allowed = [
        "resting", "walking", "running", "sniffing", "grooming", "rearing", "turning", "other",
    ];
    for (_, entry) in pairs {
        let n = entry.get("name").unwrap().as_str().unwrap();
        assert!(
            catalog.canonical(n).is_some() && allowed.contains(&n),
            "{n}"
        );
    }
    assert_eq!(
        naming_agreement(&Json::Obj(vec![]), &Json::Obj(vec![])).get("n_named_clusters"),
        Some(&Json::Int(0))
    );
}

#[test]
fn rules_record_one_entry_per_name() {
    let rules = rules_json();
    let Json::Obj(pairs) = &rules else { panic!() };
    let names: Vec<&str> = pairs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["other", "running", "grooming", "turning", "rearing", "walking", "sniffing", "resting"]
    );
    assert_eq!(
        rules.get("running").unwrap().dumps(),
        r#"[["speed", ">", 3.0], ["length", ">", 0.9], ["bend", "<", 1.0]]"#
    );
    assert_eq!(
        rules.get("other").unwrap().dumps(),
        r#"[["posture_sd", ">", 20.0]]"#
    );
}

// ------------------------------------------------------------------------------------ kinematics

#[test]
fn kinematics_match_the_python_numbers() {
    let (kp, conf) = recording_kp();
    let times: Vec<f64> = (0..T).map(|t| t as f64 / FPS).collect();
    let feats = kinematics(&kp, &conf, FPS, Some(&times));
    // `kinematics(kp, conf, 30.0, times)[0][t]` from the Python package.
    let want: [(usize, [f64; 9]); 5] = [
        (
            0,
            [
                3.496503138955336,
                1.1655258576075238,
                0.19317902823289235,
                0.009006294805599004,
                0.2251573771238327,
                0.008827803284502955,
                0.0,
                3.109938385371364e-05,
                0.003033416473749737,
            ],
        ),
        (
            58,
            [
                2.3304675082368584,
                1.028598900636037,
                1.6733592141419649,
                0.553971016460685,
                0.1685042328511675,
                0.38292114848723147,
                1.722523431189289,
                0.17013098911851565,
                1.0237985933403884,
            ],
        ),
        (
            60,
            [
                1.8642670897094815,
                0.9837760170300801,
                1.68754684664309,
                0.5546423141901067,
                0.1852866755798459,
                0.5013643112060958,
                1.7537484923380975,
                0.17096194613005508,
                1.0238351414148585,
            ],
        ),
        (
            75,
            [
                0.0,
                0.822810137271881,
                0.15781915187835693,
                0.007342384494546664,
                0.1835596123089393,
                0.933183962628592,
                0.4992857731917103,
                0.007361340727834318,
                0.7371148452681984,
            ],
        ),
        (
            119,
            [
                0.0,
                0.8284865061442057,
                0.1205291748046875,
                0.005651335594067385,
                0.14128338396549223,
                0.9159762626384762,
                0.0,
                0.007686206728804832,
                0.002984888643244667,
            ],
        ),
    ];
    for (t, row) in want {
        for c in 0..9 {
            assert!(
                (feats[t][c] - row[c]).abs() < 1e-5,
                "frame {t} {}: {} vs {}",
                FEATURES[c],
                feats[t][c],
                row[c]
            );
        }
    }
    // Without times the same features come from t / fps.
    let plain = kinematics(&kp, &conf, FPS, None);
    assert!(plain
        .iter()
        .zip(&feats)
        .all(|(a, b)| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)));
    // Uneven times go through the non-uniform `np.gradient` weights: a dropped frame (times[50:]
    // shifted by one frame) leaves the per-second speed of the straight run unchanged.
    let mut uneven = times.clone();
    for t in uneven.iter_mut().skip(50) {
        *t += 1.0 / FPS;
    }
    let var = kinematics(&kp, &conf, 1.0, Some(&uneven));
    let base = kinematics(&kp, &conf, 1.0, Some(&times));
    assert!((var[20][0] - base[20][0]).abs() < 1e-9);
    assert!(var[50][0] < base[50][0]); // the gap spreads the same motion over more time
}

#[test]
fn kinematics_measure_speed_in_body_lengths_per_second() {
    let (kp, conf) = recording_kp();
    let feats = kinematics(&kp, &conf, FPS, None);
    assert_eq!(feats.len(), T);
    assert!(feats.iter().all(|r| r.iter().all(|v| v.is_finite())));
    let lengths: Vec<f64> = (0..T)
        .map(|t| {
            let (a, b) = (
                kp[t * NUM_KEYPOINTS + index("nose")],
                kp[t * NUM_KEYPOINTS + index("tail_base")],
            );
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        })
        .collect();
    let mut sorted = lengths.clone();
    let med = median(&mut sorted);
    assert!(feats[..20]
        .iter()
        .all(|r| close(r[0], 4.0 * FPS / med, 0.05)));
    assert!(feats[T - 20..].iter().all(|r| r[0].abs() < 0.05));
    let times: Vec<f64> = (0..T).map(|t| t as f64 / (FPS / 2.0)).collect();
    let half = kinematics(&kp, &conf, FPS, Some(&times));
    assert!(half[..20]
        .iter()
        .zip(&feats)
        .all(|(h, f)| (h[0] - f[0] / 2.0).abs() < 0.05));
    assert!(feats[..20].iter().all(|r| r[1] > 1.1) && feats[T - 20..].iter().all(|r| r[1] < 0.9));
    let bent: f64 = feats[T - 20..].iter().map(|r| r[5]).sum::<f64>() / 20.0;
    assert!(feats[..20].iter().all(|r| r[5] < 0.2) && bent > 0.5);
    assert!(feats[..20].iter().all(|r| r[3].abs() < 0.05)); // posture: a rigid body
    assert!(feats[..20].iter().all(|r| (r[4] - 0.2).abs() < 0.05)); // the wobbling nose
    let real: Vec<f64> = (0..T).map(|t| t as f64 / FPS).collect();
    let one = kinematics(&kp, &conf, 1.0, Some(&real));
    let two = kinematics(&kp, &conf, 2.0, Some(&real));
    assert!(one
        .iter()
        .zip(&two)
        .all(|(a, b)| (0..6).all(|c| (a[c] - b[c]).abs() < 1e-12)));
    assert!(one[MOVE][0] > feats[MOVE][0]); // a one-frame window does not smear the stop
    assert!(one.iter().all(|r| r[6..].iter().all(|v| *v >= 0.0)));
}

#[test]
fn bouts_gradient_and_unwrap_follow_numpy() {
    assert_eq!(
        bouts(&[0, 0, 1, 1, 1, 0]),
        [(0, 2, 0), (2, 5, 1), (5, 6, 0)]
    );
    assert!(bouts(&[]).is_empty());
    // np.gradient([1, 4, 9, 16], [0, 1, 3, 4]) == [3.0, 2.8333333333333335, 5.499999999999999, 7.0]
    let g = gradient(&[1.0, 4.0, 9.0, 16.0], &[0.0, 1.0, 3.0, 4.0]);
    assert_eq!(g, [3.0, 2.8333333333333335, 5.499999999999999, 7.0]);
    // np.unwrap([0, 3, 6.5, 0.2]) == [0.0, 3.0, 0.21681469282041377, 0.2]
    let u = unwrap(&[0.0, 3.0, 6.5, 0.2]);
    assert!(close(u[2], 0.21681469282041377, 1e-12) && close(u[3], 0.2, 1e-12));
    // _smooth([1, 2, 3, 10], 2): edge-padded windows [1,1], [1,2]... -> [1, 1.5, 2.5, 6.5]
    assert_eq!(smooth(&[1.0, 2.0, 3.0, 10.0], 2), [1.0, 1.5, 2.5, 6.5]);
}

// ------------------------------------------------------------------------------------ cluster

fn summary_rows(summary: &Json) -> Vec<(String, Vec<f64>)> {
    let Json::Obj(pairs) = summary else { panic!() };
    pairs
        .iter()
        .map(|(c, v)| {
            let Json::Obj(stats) = v else { panic!() };
            (
                c.clone(),
                stats.iter().map(|(_, x)| x.as_f64().unwrap()).collect(),
            )
        })
        .collect()
}

/// Each Python cluster summary (share + 9 medians) is one of the Rust clusters, whatever its number.
fn assert_same_clusters(summary: &Json, want: &[[f64; 10]]) -> Vec<String> {
    let rows = summary_rows(summary);
    assert_eq!(rows.len(), want.len());
    want.iter()
        .map(|w| {
            let hit = rows
                .iter()
                .find(|(_, r)| r.iter().zip(w).all(|(a, b)| (a - b).abs() < 1e-9))
                .unwrap_or_else(|| panic!("{w:?} not in {}", summary.dumps()));
            hit.0.clone()
        })
        .collect()
}

#[test]
fn clusters_match_the_python_summary_and_montages() {
    let dir = temp("split");
    let tracks = dir.join("tracks.npz");
    write_tracks(&tracks, &["rest".to_string()], &vec![1.0; T]);
    let video = dir.join("video");
    write_video(&video);
    let out = dir.join("clusters");
    let summary = cluster_tracks(&tracks, &video, &out, FPS, (0.0, 0.8), 2).unwrap();
    let rows = summary_rows(&summary);
    assert_eq!(
        rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(),
        ["0", "1"]
    );
    assert!((rows.iter().map(|r| r.1[0]).sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(
        std::fs::read_to_string(out.join("summary.json")).unwrap(),
        summary.dumps_indent(1)
    );
    assert!(out.join("cluster0.png").is_file() && out.join("cluster1.png").is_file());
    // `cluster_tracks("tracks.npz", ..., train_range=(0.0, 0.8), k=2)` in Python: k-means puts the
    // stop (frames 49 to 74) on its own and the steady running and resting together.
    let ids = assert_same_clusters(
        &summary,
        &[
            [
                0.2708333333333333,
                1.515,
                0.949,
                1.673,
                0.554,
                0.182,
                0.595,
                1.585,
                0.154,
                1.024,
            ],
            [
                0.7291666666666666,
                3.497,
                1.166,
                0.157,
                0.007,
                0.182,
                0.017,
                0.0,
                0.0,
                0.003,
            ],
        ],
    );
    let labels = read_npz(out.join("clusters.npz")).unwrap()["labels"]
        .ints()
        .unwrap();
    let runs: Vec<(usize, usize)> = bouts(&labels).iter().map(|(s, e, _)| (*s, *e)).collect();
    assert_eq!(runs, [(0, 49), (49, 75), (75, 120)]);
    assert_eq!(labels[0].to_string(), ids[1]);
    let z = read_npz(out.join("clusters.npz")).unwrap();
    assert_eq!(z["features"].shape, [T, 9]);
    let exact = NpzFile::read(&out.join("clusters.npz")).unwrap();
    assert_eq!(exact.floats("train_range").unwrap(), [0.0, 0.8]);
    assert_eq!(exact.floats("fps").unwrap(), [30.0]);
    // a montage tile is cropped around the animal: dark pixels in the middle of every tile
    let sheet = image::open(out.join(format!("cluster{}.png", ids[1])))
        .unwrap()
        .to_luma8();
    assert_eq!(sheet.height() % 184, 0);
    assert_eq!(sheet.width(), 6 * 180);
    for tile in 0..6 {
        assert!(
            sheet.get_pixel(tile * 180 + 90, 2 + 90)[0] < 90,
            "tile {tile}"
        );
    }
    // a training range too short for any bout shows nothing
    let thin = cluster_tracks(&tracks, &video, &out.join("thin"), FPS, (0.0, 0.05), 2).unwrap();
    assert_eq!(summary_rows(&thin).len(), 2);
    assert!(out.join("thin").join("clusters.npz").is_file());
    assert!(
        !out.join("thin").join("cluster0.png").exists()
            && !out.join("thin").join("cluster1.png").exists()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn three_and_four_clusters_match_the_sklearn_partition() {
    let dir = temp("three");
    let tracks = dir.join("tracks.npz");
    write_tracks(&tracks, &["rest".to_string()], &vec![1.0; T]);
    let out = dir.join("clusters");
    let summary =
        cluster_tracks(&tracks, &dir.join("absent.avi"), &out, FPS, (0.0, 1.0), 3).unwrap();
    assert_same_clusters(
        &summary,
        &[
            [
                0.13333333333333333,
                1.981,
                0.995,
                1.682,
                0.554,
                0.182,
                0.471,
                1.695,
                0.165,
                1.024,
            ],
            [
                0.43333333333333335,
                0.0,
                0.825,
                0.149,
                0.007,
                0.174,
                0.927,
                0.0,
                0.007,
                0.004,
            ],
            [
                0.43333333333333335,
                3.497,
                1.166,
                0.159,
                0.007,
                0.185,
                0.016,
                0.0,
                0.0,
                0.003,
            ],
        ],
    );
    let labels = read_npz(out.join("clusters.npz")).unwrap()["labels"]
        .ints()
        .unwrap();
    let runs: Vec<(usize, usize)> = bouts(&labels).iter().map(|(s, e, _)| (*s, *e)).collect();
    assert_eq!(runs, [(0, 52), (52, 68), (68, 120)]);
    let four = cluster_tracks(
        &tracks,
        &dir.join("absent.avi"),
        &dir.join("four"),
        FPS,
        (0.0, 0.8),
        4,
    )
    .unwrap();
    assert_same_clusters(
        &four,
        &[
            [
                0.5416666666666666,
                3.497,
                1.166,
                0.159,
                0.007,
                0.185,
                0.016,
                0.0,
                0.0,
                0.003,
            ],
            [
                0.15625, 2.097, 1.006, 1.685, 0.555, 0.182, 0.442, 1.695, 0.165, 1.024,
            ],
            [
                0.21875, 0.0, 0.824, 0.157, 0.007, 0.182, 0.93, 0.0, 0.007, 0.004,
            ],
            [
                0.08333333333333333,
                0.0,
                0.824,
                0.158,
                0.007,
                0.178,
                0.928,
                1.363,
                0.122,
                1.024,
            ],
        ],
    );
    let labels = read_npz(dir.join("four/clusters.npz")).unwrap()["labels"]
        .ints()
        .unwrap();
    let runs: Vec<(usize, usize)> = bouts(&labels).iter().map(|(s, e, _)| (*s, *e)).collect();
    assert_eq!(runs, [(0, 52), (52, 67), (67, 75), (75, 120)]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn montage_of_a_missing_video_is_a_black_square() {
    let (kp, _) = recording_kp();
    let centre: Vec<[f64; 2]> = (0..T)
        .map(|t| kp[t * NUM_KEYPOINTS + index("spine_mid")])
        .collect();
    let mut src = Source::open(std::env::temp_dir().join("ap-absent-video.avi"));
    let empty = montage(&mut src, &centre, &[0, 1], 45, 2);
    assert_eq!((empty.width, empty.height), (10, 10));
    assert!(empty.pixels.iter().all(|p| *p == 0));
}

// ------------------------------------------------------------------------------------ name + label

/// Three clusters on the training range, named by rule: `(clusters dir, [walking, resting, running] ids)`.
fn three_named(dir: &Path) -> (PathBuf, Vec<i64>) {
    let tracks = dir.join("tracks.npz");
    write_tracks(&tracks, &["rest".to_string()], &vec![1.0; T]);
    let clusters = dir.join("clusters");
    let summary = cluster_tracks(
        &tracks,
        &dir.join("absent.avi"),
        &clusters,
        FPS,
        (0.0, 0.8),
        3,
    )
    .unwrap();
    // Python `cluster_tracks(..., train_range=(0.0, 0.8), k=3)`
    let ids = assert_same_clusters(
        &summary,
        &[
            [
                0.16666666666666666,
                1.981,
                0.995,
                1.682,
                0.554,
                0.182,
                0.471,
                1.695,
                0.165,
                1.024,
            ],
            [
                0.2916666666666667,
                0.0,
                0.824,
                0.157,
                0.007,
                0.182,
                0.93,
                0.0,
                0.007,
                0.004,
            ],
            [
                0.5416666666666666,
                3.497,
                1.166,
                0.159,
                0.007,
                0.185,
                0.016,
                0.0,
                0.0,
                0.003,
            ],
        ],
    );
    (clusters, ids.iter().map(|c| c.parse().unwrap()).collect())
}

#[test]
fn naming_and_label_files_follow_the_python_run() {
    let dir = temp("label");
    let (clusters, ids) = three_named(&dir);
    let report = write_naming(&clusters, None).unwrap();
    assert!(clusters.join("naming.json").is_file());
    let Json::Obj(keys) = &report else { panic!() };
    assert_eq!(
        keys.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        ["naming", "rules"]
    );
    let naming = report.get("naming").unwrap();
    let name_of = |c: i64| {
        naming
            .get(&c.to_string())
            .unwrap()
            .get("name")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    };
    // Python: {'0': 'walking', '1': 'resting', '2': 'running'}
    assert_eq!(
        ids.iter().map(|c| name_of(*c)).collect::<Vec<_>>(),
        ["walking", "resting", "running"]
    );
    let entry = naming.get("0").unwrap();
    let Json::Obj(fields) = entry.get("features").unwrap() else {
        panic!()
    };
    assert_eq!(
        fields.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        FEATURES
    );

    let mapping: Vec<(i64, String)> = ids
        .iter()
        .zip(["walking", "resting", "running"])
        .map(|(c, n)| (*c, n.to_string()))
        .collect();
    let labels = dir.join("labels");
    let share = label_clusters(&clusters, &mapping, &labels, "synth", "top", 0.5).unwrap();
    // Python `label_clusters(..., min_bout=0.5)`
    assert_eq!(
        share.dumps(),
        r#"{"walking": 0.133, "resting": 0.433, "running": 0.433}"#
    );
    let train = Session::read(labels.join("train/synth_train.npz")).unwrap();
    assert_eq!(train.label_names, ["walking", "resting", "running"]);
    let want: Vec<i64> = [vec![2; 52], vec![0; 16], vec![1; 28]].concat();
    assert_eq!(train.labels, want);
    assert_eq!(
        (
            train.fps,
            train.view.as_str(),
            train.animal.as_str(),
            train.frames
        ),
        (FPS, "top", "synth_train", 96)
    );
    let held = Session::read(labels.join("heldout/synth_heldout.npz")).unwrap();
    assert_eq!(
        (held.animal.as_str(), held.labels.clone()),
        ("synth_heldout", vec![1; 24])
    );
    let written =
        Json::parse(&std::fs::read_to_string(labels.join("labels.json")).unwrap()).unwrap();
    assert_eq!(
        written.get("mapping").unwrap().dumps(),
        format!(
            "{{\"{}\": \"walking\", \"{}\": \"resting\", \"{}\": \"running\"}}",
            ids[0], ids[1], ids[2]
        )
    );
    assert_eq!(written.get("min_bout_s"), Some(&Json::Float(0.5)));
    assert_eq!(written.get("share"), Some(&share));
    // a whole second: the 16-frame stop merges into the run before it
    let longer =
        label_clusters(&clusters, &mapping, &dir.join("long"), "synth", "top", 1.0).unwrap();
    assert_eq!(longer.dumps(), r#"{"resting": 0.433, "running": 0.567}"#);
    let train = Session::read(dir.join("long/train/synth_train.npz")).unwrap();
    assert_eq!(train.labels, [vec![2; 68], vec![1; 28]].concat());
    // the hand naming written above is what `agree_with` scores the rules against
    let agree = write_naming(&clusters, Some(&labels.join("labels.json"))).unwrap();
    let agree = agree.get("agreement").unwrap();
    assert_eq!(
        agree.dumps(),
        r#"{"n_named_clusters": 3, "weighted_agreement": 1.0, "mismatches": []}"#
    );
    let one = Json::obj([(ids[1].to_string(), Json::str("resting"))]);
    assert_eq!(
        naming_agreement(naming, &one).get("weighted_agreement"),
        Some(&Json::Float(1.0))
    );
    let wrong = naming_agreement(
        naming,
        &Json::obj([(ids[2].to_string(), Json::str("grooming"))]),
    );
    assert_eq!(
        wrong.dumps(),
        format!(
            "{{\"n_named_clusters\": 1, \"weighted_agreement\": 0.0, \"mismatches\": [\"cluster {}: rule says running, human said grooming\"]}}",
            ids[2]
        )
    );
    let nobody = naming_agreement(naming, &Json::obj([("7", Json::str("resting"))]));
    assert!(nobody
        .get("weighted_agreement")
        .unwrap()
        .as_f64()
        .unwrap()
        .is_nan());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn label_clusters_rejects_unknown_names_and_merges_short_bouts() {
    let dir = temp("merge");
    let (clusters, ids) = three_named(&dir);
    let err = label_clusters(
        &clusters,
        &[(ids[2], "sidewalking_xyz".into())],
        &dir.join("none"),
        "recording",
        "top",
        0.5,
    )
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "not in the behavior catalogue (names or synonyms): ['sidewalking_xyz']"
    );
    let mapping = vec![
        (ids[2], "running".to_string()),
        (ids[1], "resting".to_string()),
        (ids[0], "walking".to_string()),
    ];
    let share = label_clusters(
        &clusters,
        &mapping,
        &dir.join("merged"),
        "recording",
        "top",
        60.0,
    )
    .unwrap();
    assert_eq!(share.dumps(), r#"{"running": 1.0}"#);
    for part in ["train", "heldout"] {
        let s = Session::read(
            dir.join("merged")
                .join(part)
                .join(format!("recording_{part}.npz")),
        )
        .unwrap();
        assert!(s
            .labels
            .iter()
            .all(|l| s.label_names[*l as usize] == "running"));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn parse_mapping_keeps_dict_semantics() {
    let specs: Vec<String> = ["0=resting", "2=grooming", "0=walking"]
        .map(String::from)
        .to_vec();
    assert_eq!(
        parse_mapping(&specs).unwrap(),
        [(0, "walking".to_string()), (2, "grooming".to_string())]
    );
    assert!(parse_mapping(&["zero=resting".to_string()]).is_err());
    assert!(parse_mapping(&["0".to_string()]).is_err());
}

#[test]
fn kmeans_finds_separated_groups_and_is_deterministic() {
    let mut x = Vec::new();
    for i in 0..30 {
        let j = f64::from(i) * 0.01;
        x.push(vec![j, 0.0]);
        x.push(vec![10.0 + j, 0.0]);
        x.push(vec![0.0, 10.0 + j]);
    }
    let km = KMeans::fit(&x, 3, 10, 0).unwrap();
    let lab = km.predict(&x);
    for g in 0..3 {
        let first = lab[g];
        assert!((0..30).all(|i| lab[3 * i + g] == first));
    }
    assert_ne!(lab[0], lab[1]);
    assert_ne!(lab[1], lab[2]);
    assert_eq!(KMeans::fit(&x, 3, 10, 0).unwrap().predict(&x), lab);
    assert!(KMeans::fit(&x[..2], 3, 10, 0).is_err());
}
