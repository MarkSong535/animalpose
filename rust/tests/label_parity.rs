//! `animalpose label` against the Python `selftrain.label_videos` on the pipeline fixture: the
//! same frames (two copies of `tests/fixtures/pipeline/frames`), the same teacher (the fixture's
//! detector and pose safetensors, saved back to `.pt` for Python) and the same thresholds give the
//! same statistics, the same image names and the same labels.
//!
//! Python reference (with `opts = dict(det_size=(96, 96), crop_size=64, per_source=8, hi=0.3,
//! lo=0.1, kp_conf=0.0, flip_tol=10.0, min_kps=1, min_seq=8, holdout=False)`):
//! `label_videos("root", LabelOptions(out=..., **opts))` and the same with `boxes="blob"`.

use std::path::{Path, PathBuf};

use animalpose::cli::selftrain::{label_videos, load_teacher, LabelOptions};

fn fixture_root() -> Option<PathBuf> {
    let dir = animalpose::data::fixtures_dir()?.join("pipeline");
    dir.join("pose.safetensors").is_file().then_some(dir)
}

fn copy_frames(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap().flatten() {
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

fn options(fixtures: &Path, out: PathBuf, boxes: &str) -> LabelOptions {
    LabelOptions {
        detector: fixtures.join("detector.safetensors").display().to_string(),
        pose: fixtures.join("pose.safetensors").display().to_string(),
        out,
        det_size: (96, 96),
        crop_size: 64,
        per_source: 8,
        hi: 0.3,
        lo: 0.1,
        kp_conf: 0.0,
        flip_tol: 10.0,
        min_kps: 1,
        min_seq: 8,
        boxes: boxes.to_string(),
        holdout: false,
        progress: false,
        ..LabelOptions::default()
    }
}

#[test]
fn label_matches_python_on_the_pipeline_fixture() {
    let Some(fixtures) = fixture_root() else {
        eprintln!("skipped: tests/fixtures/pipeline is missing");
        return;
    };
    let base = std::env::temp_dir().join(format!("ap-label-parity-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let root = base.join("root");
    for rec in ["rec_a", "rec_b"] {
        copy_frames(&fixtures.join("frames"), &root.join(rec));
    }

    let opts = options(&fixtures, base.join("detector"), "detector");
    let teacher = load_teacher(&opts).unwrap();
    let stats = label_videos(&root, &teacher, &opts).unwrap();
    assert_eq!(
        stats.to_json().dumps(),
        r#"{"sources": 2, "frames": 16, "labeled": 0, "skipped_uncertain": 0, "no_animal": 6, "animals": 0, "keypoints": 0, "head_tail_flip": 0, "images": 0}"#
    );

    let opts = options(&fixtures, base.join("blob"), "blob");
    let stats = label_videos(&root, &teacher, &opts).unwrap();
    assert_eq!(
        stats.to_json().dumps(),
        r#"{"sources": 2, "frames": 16, "labeled": 2, "skipped_uncertain": 0, "no_animal": 14, "animals": 2, "keypoints": 32, "head_tail_flip": 0, "images": 2}"#
    );
    let coco: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(base.join("blob/train.json")).unwrap())
            .unwrap();
    let names: Vec<&str> = coco["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|im| im["file_name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["04cf26af_rec_a/0000004.jpg", "9dc67715_rec_b/0000004.jpg"]
    );
    for ann in coco["annotations"].as_array().unwrap() {
        assert_eq!(ann["bbox"], serde_json::json!([0.0, 0.0, 96.0, 96.0]));
        assert_eq!(ann["num_keypoints"], 16);
        let kp: Vec<f64> = ann["keypoints"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let want = [0.0, 0.0, 0.0, 66.535, 89.666, 2.0, 28.444, 89.681, 2.0];
        for (got, want) in kp.iter().zip(want) {
            assert!((got - want).abs() < 2e-3, "{kp:?}");
        }
    }
    for name in names {
        let img = image::open(base.join("blob/images").join(name)).unwrap();
        assert_eq!((img.width(), img.height()), (96, 96));
    }
    let split: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(base.join("blob/split.json")).unwrap())
            .unwrap();
    assert_eq!(split["train"], serde_json::json!(["rec_a", "rec_b"]));
    assert_eq!(split["held_out"], serde_json::json!([]));
    let _ = std::fs::remove_dir_all(&base);
}
