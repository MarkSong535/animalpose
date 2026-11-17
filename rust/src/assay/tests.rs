//! The experiment layer end to end on a scripted animal.

use std::f64::consts::TAU;
use std::path::Path;

use serde_json::Value;

use super::arena::Layout;
use super::config::defaults;
use super::output::{run_dir, Options};
use crate::data::sessions::{NpyArray, NpyData};
use crate::data::synthetic::CANONICAL_NAMES;
use crate::multiview::npz::{write_npz, Member};

const FPS: f64 = 30.0;
const FRAMES: usize = 300;

/// A 40 px animal in a 400 × 400 px open field: it walks along the top edge for 5 s, then sits
/// hunched in the centre and grooms its face (front paws stroking at the nose, 4 Hz) for 5 s.
fn scripted() -> Vec<[f64; 2]> {
    let k = CANONICAL_NAMES.len();
    let idx = |n: &str| CANONICAL_NAMES.iter().position(|m| *m == n).unwrap();
    let mut kp = vec![[f64::NAN; 2]; FRAMES * k];
    for t in 0..FRAMES {
        let walking = t < FRAMES / 2;
        let (p, len) = if walking {
            ([100.0 + 200.0 * t as f64 / (FRAMES / 2) as f64, 40.0], 40.0)
        } else {
            ([200.0, 200.0], 24.0)
        };
        let set = |kp: &mut Vec<[f64; 2]>, name: &str, v: [f64; 2]| kp[t * k + idx(name)] = v;
        let nose = [p[0] + len / 2.0, p[1]];
        set(&mut kp, "spine_mid", p);
        set(&mut kp, "nose", nose);
        set(&mut kp, "tail_base", [p[0] - len / 2.0, p[1]]);
        set(&mut kp, "head_center", [nose[0] - 5.0, p[1]]);
        set(&mut kp, "left_ear", [nose[0] - 6.0, p[1] - 3.0]);
        set(&mut kp, "right_ear", [nose[0] - 6.0, p[1] + 3.0]);
        set(&mut kp, "spine_lower", [p[0] - len / 4.0, p[1]]);
        for (side, sign) in [("left", -1.0), ("right", 1.0)] {
            set(
                &mut kp,
                &format!("{side}_shoulder"),
                [p[0] + len / 4.0, p[1] + sign * 6.0],
            );
            set(
                &mut kp,
                &format!("{side}_hip"),
                [p[0] - len / 4.0, p[1] + sign * 6.0],
            );
            let paw = if walking {
                [p[0] + len / 4.0, p[1] + sign * 8.0]
            } else {
                let stroke = 3.0 * (TAU * 4.0 * t as f64 / FPS).sin();
                [nose[0] - 3.0 + stroke, p[1] + sign * 2.0]
            };
            set(&mut kp, &format!("{side}_front_paw"), paw);
            set(
                &mut kp,
                &format!("{side}_hind_paw"),
                [p[0] - len / 4.0, p[1] + sign * 8.0],
            );
        }
    }
    kp
}

fn write_tracks(dir: &Path) {
    let k = CANONICAL_NAMES.len();
    let kp = scripted();
    let arr = |shape: Vec<usize>, v: Vec<f64>| {
        Member::Npy(NpyArray {
            shape,
            data: NpyData::Float(v.iter().map(|x| *x as f32).collect()),
        })
    };
    let boxes: Vec<f64> = (0..FRAMES)
        .flat_map(|t| {
            let p = kp[t * k + 8];
            [p[0] - 30.0, p[1] - 30.0, p[0] + 30.0, p[1] + 30.0]
        })
        .collect();
    let members = vec![
        (
            "track1_frames".to_owned(),
            Member::Npy(NpyArray {
                shape: vec![FRAMES],
                data: NpyData::Int((0..FRAMES as i64).collect()),
            }),
        ),
        (
            "track1_keypoints".to_owned(),
            arr(
                vec![FRAMES, k, 2],
                kp.iter().flat_map(|p| [p[0], p[1]]).collect(),
            ),
        ),
        (
            "track1_conf".to_owned(),
            arr(vec![FRAMES, k], vec![0.9; FRAMES * k]),
        ),
        ("track1_boxes".to_owned(), arr(vec![FRAMES, 4], boxes)),
        (
            "times".to_owned(),
            arr(vec![FRAMES], (0..FRAMES).map(|t| t as f64 / FPS).collect()),
        ),
        (
            "keypoint_names".to_owned(),
            Member::Npy(NpyArray {
                shape: vec![k],
                data: NpyData::Text(CANONICAL_NAMES.map(str::to_owned).to_vec()),
            }),
        ),
    ];
    write_npz(dir.join("tracks.npz"), &members).unwrap();
}

#[test]
fn an_open_field_run_scores_location_and_grooming_and_resumes() {
    let root = std::env::temp_dir().join(format!("animalpose-assay-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("clip-analysis");
    std::fs::create_dir_all(&dir).unwrap();
    write_tracks(&dir);
    let zones = root.join("zones");
    std::fs::create_dir_all(&zones).unwrap();
    let layout = Layout {
        rig: "open_field".into(),
        source: "manual".into(),
        status: "ok".into(),
        floor: vec![[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]],
        ..Layout::default()
    };
    std::fs::write(
        zones.join("open_field.json"),
        serde_json::to_vec(&layout).unwrap(),
    )
    .unwrap();
    let cfg = defaults("open_field");
    let opts = Options {
        zones_dir: &zones,
        video: None,
        models: Vec::new(),
        config: serde_json::json!({"assay": {"experiment": "open_field"}}),
        force: false,
    };
    let report = run_dir(&dir, &cfg, &opts).unwrap();
    assert!(report.ran);
    assert!(
        report.notes.iter().any(|n| n.contains("rearing: needs 3D")),
        "{:?}",
        report.notes
    );

    let csv = std::fs::read_to_string(dir.join("assay/behavior_v3.csv")).unwrap();
    let rows: Vec<Vec<&str>> = csv.lines().map(|l| l.split(',').collect()).collect();
    assert_eq!(rows.len(), FRAMES + 1);
    let col = |name: &str| {
        rows[0]
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    };
    let (loc, beh, kind) = (col("location"), col("behavior"), col("groom_type"));
    assert_eq!(rows[1 + 50][loc], "periphery");
    assert_eq!(rows[1 + 50][beh], "other");
    assert_eq!(rows[1 + 225][loc], "center");
    assert_eq!(rows[1 + 225][beh], "grooming");
    assert_eq!(rows[1 + 225][kind], "paw_face");
    assert!(col("wall_dist_BL") > 0 && col("grooming_rule") > 0);

    let summary = std::fs::read_to_string(dir.join("assay/behavior_summary_v3.csv")).unwrap();
    let lines: Vec<&str> = summary.lines().collect();
    assert!(lines[0].starts_with("# animalpose ") && lines[0].contains("config_sha256:"));
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("assay/run_manifest.json")).unwrap())
            .unwrap();
    let s = &manifest["summary"];
    let grooming = s["grooming_s"].as_f64().unwrap();
    assert!((3.5..=5.0).contains(&grooming), "grooming {grooming} s");
    assert_eq!(s["grooming_bouts"], 1);
    assert!((s["center_s"].as_f64().unwrap() - 5.0).abs() < 0.5, "{s}");
    assert_eq!(s["dims"], "2d");
    assert!((s["body_length"].as_f64().unwrap() - 40.0).abs() < 1.0);

    // The same inputs and settings: skipped. Forced: redone, and the old results kept.
    assert!(!run_dir(&dir, &cfg, &opts).unwrap().ran);
    let forced = Options {
        force: true,
        ..opts
    };
    assert!(run_dir(&dir, &cfg, &forced).unwrap().ran);
    let backups = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("assay.backup-")
        })
        .count();
    assert_eq!(backups, 1);
    assert!(!dir.join("assay.staging").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// The arena detector in the behavior layer: the shipped grooming detector finds the arena in the
/// video, its layout is saved per video, and the zones module adds floor positions. A synthetic
/// box is not what the detector was trained on, so only the path is checked, not the corners.
#[test]
fn an_arena_detector_supplies_the_layout_and_floor_positions() {
    use crate::algos::image::Bgr;
    use crate::data::video::{ffmpeg_available, VideoWriter};
    let model = crate::data::repo_root()
        .unwrap()
        .join("samples/models/arena/sample-arena-grooming.pt");
    assert!(
        model.is_file(),
        "the sample bundle has no grooming arena detector"
    );
    if !ffmpeg_available() {
        return;
    }
    let root = std::env::temp_dir().join(format!("animalpose-arena-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("clip-analysis");
    std::fs::create_dir_all(&dir).unwrap();
    write_tracks(&dir);
    // A light floor (0..400 px, where the scripted animal walks) with dark walls around it.
    let (w, h) = (640u32, 480u32);
    let mut pixels = vec![40u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = match (x, y) {
                (0..=399, 0..=399) => 225,
                (0..=409, 0..=409) => 120,
                _ => continue,
            };
            pixels[((y * w + x) * 3) as usize..][..3].fill(v);
        }
    }
    let video = root.join("clip.mp4");
    let mut writer = VideoWriter::new(&video, FPS);
    for _ in 0..FRAMES {
        writer
            .write(&Bgr {
                width: w,
                height: h,
                pixels: pixels.clone(),
            })
            .unwrap();
    }
    writer.finish().unwrap();

    let zones = root.join("zones");
    let mut cfg = defaults("grooming");
    cfg.arena.rig = "grooming".into();
    cfg.arena.model = Some(model);
    let opts = Options {
        zones_dir: &zones,
        video: Some(&video),
        models: Vec::new(),
        config: serde_json::json!({"assay": {"experiment": "grooming"}}),
        force: false,
    };
    let report = run_dir(&dir, &cfg, &opts).unwrap();
    assert!(report.ran);
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.starts_with("arena: arena model grooming")),
        "{:?}",
        report.notes
    );
    // Saved per video as `<rig>.<folder>-<video>.json`, with its picture.
    let saved = std::fs::read_dir(&zones)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.starts_with("grooming.") && name.ends_with("-clip.json")
        })
        .expect("the detector's layout is saved");
    let layout = Layout::load(&saved).unwrap().unwrap();
    assert_eq!(layout.source, "model");
    assert_eq!(layout.floor.len(), 4);
    assert!(saved.with_extension("png").is_file());
    let csv = std::fs::read_to_string(dir.join("assay/behavior_v3.csv")).unwrap();
    let header: Vec<&str> = csv.lines().next().unwrap().split(',').collect();
    let floor_columns = ["floor_u", "floor_v", "floor_wall_BL"]
        .iter()
        .all(|c| header.contains(c));
    let skipped = report
        .notes
        .iter()
        .any(|n| n.starts_with("zones: no floor positions"));
    assert!(floor_columns || skipped, "{header:?} {:?}", report.notes);
    let _ = std::fs::remove_dir_all(&root);
}

/// L1: a checkpoint whose SHA-256 differs from `checksums.<kind>` is refused before loading.
#[test]
fn checkpoints_must_match_their_checksums() {
    let file = std::env::temp_dir().join(format!("animalpose-sha-{}.pt", std::process::id()));
    std::fs::write(&file, b"weights").unwrap();
    let sha = crate::cli::download::sha256_file(&file).unwrap();
    let check = |expected: &str| {
        let cfg =
            crate::data::config::Config::parse(&format!("checksums:\n  pose: {expected}\n"), &[])
                .unwrap();
        crate::cli::predict::verify_checksum(&cfg, "checkpoints.pose", &file)
    };
    assert!(check(&sha).is_ok());
    assert!(check(&"0".repeat(64))
        .unwrap_err()
        .to_string()
        .contains("expects"));
    let none = crate::data::config::Config::parse("{}", &[]).unwrap();
    assert!(crate::cli::predict::verify_checksum(&none, "checkpoints.pose", &file).is_ok());
    let _ = std::fs::remove_file(&file);
}

/// Motion inside the outline, measured on an encoded video: a patch that flickers counts, the
/// still background around it does not.
#[test]
fn motion_counts_changing_pixels_inside_the_outline() {
    use crate::algos::image::Bgr;
    use crate::data::video::{ffmpeg_available, VideoWriter};

    if !ffmpeg_available() {
        return;
    }
    let video = std::env::temp_dir().join(format!("animalpose-motion-{}.mp4", std::process::id()));
    let mut writer = VideoWriter::new(&video, 10.0);
    let (w, h) = (64u32, 64u32);
    for t in 0..30 {
        let mut px = vec![40u8; (w * h * 3) as usize];
        // A patch that is only there from frame 20 (so the median background is empty),
        // alternating brightness.
        if t >= 20 {
            let v = if t % 2 == 0 { 200 } else { 120 };
            for y in 20..40 {
                for x in 20..40 {
                    let i = ((y * w + x) * 3) as usize;
                    px[i..i + 3].copy_from_slice(&[v, v, v]);
                }
            }
        }
        writer.write(&Bgr::from_pixels(w, h, px).unwrap()).unwrap();
    }
    writer.finish().unwrap();
    if crate::data::video::probe(&video).is_err() {
        return; // no ffprobe
    }
    let track = super::input::Track2d {
        id: 1,
        first: 0,
        rows: 30,
        keypoints: Vec::new(),
        conf: Vec::new(),
        boxes: vec![[16.0, 16.0, 44.0, 44.0]; 30],
    };
    let mut cfg = defaults("grooming").motion;
    cfg.lag_s = 0.1;
    let m = super::motion::measure(&video, &track, 30, 10.0, &cfg).unwrap();
    assert!(m[0].is_none(), "nothing to compare with yet");
    assert!(m[15].is_none(), "no outline before the patch appears");
    assert!(m[25].unwrap() > 0.5, "{:?}", m[25]);
    let _ = std::fs::remove_file(&video);
}

/// `configs/assay_example.yaml` lists every key at its default.
#[test]
fn the_example_config_spells_out_the_defaults() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/assay_example.yaml");
    let cfg = crate::data::config::Config::load(&path, &[]).unwrap();
    let mut example = super::AssayConfig::from_config(&cfg).unwrap();
    assert_eq!(example.cups.len(), 2);
    example.cups.clear();
    assert_eq!(example, defaults("open_field"));
}
