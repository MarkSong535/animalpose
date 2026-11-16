//! `animalpose eval behavior-sessions` and `animalpose export safetensors|import-safetensors` end to
//! end, on a checkpoint and session files written here; the checks without a Rust harness refuse to
//! run by name.

use animalpose::data::pyjson::{loads, PyVal};
use animalpose::data::sessions::{write_npz, NpyArray, NpyData, IGNORE};
use animalpose::models::config::ModelConfig;
use animalpose::models::weights::Weights;
use animalpose::models::ModelKind;
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("animalpose-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(args: &[&str]) -> anyhow::Result<()> {
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    animalpose::cli::run(&args)
}

fn session(path: &Path, labels: &[i64]) {
    let t = labels.len();
    let k = 25;
    let coords: Vec<f32> = (0..t * k * 2)
        .map(|i| (i % 97) as f32 + (i / 50) as f32)
        .collect();
    let arr = |shape: Vec<usize>, data: NpyData| NpyArray { shape, data };
    let text = |s: &str| arr(vec![], NpyData::Text(vec![s.to_string()]));
    let names: Vec<String> = [
        "resting", "walking", "rearing", "grooming", "eating", "other",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    write_npz(
        path,
        &[
            ("coords", &arr(vec![t, k, 2], NpyData::Float(coords))),
            ("conf", &arr(vec![t, k], NpyData::Float(vec![0.9; t * k]))),
            ("labels", &arr(vec![t], NpyData::Int(labels.to_vec()))),
            ("fps", &arr(vec![], NpyData::Float(vec![15.0]))),
            ("animal", &text("group_a")),
            ("view", &text("top")),
            ("label_names", &arr(vec![names.len()], NpyData::Text(names))),
        ],
    )
    .unwrap();
}

#[test]
fn behavior_sessions_scores_every_variant_of_a_checkpoint() {
    let dir = tmp("eval-sessions");
    let cfg = ModelConfig::from_json_str(r#"{"dims": 2, "hidden": 16}"#).unwrap();
    let model = dir.join("behavior.safetensors");
    animalpose::models::save(&model, ModelKind::Behavior, &cfg, &mut Weights::new()).unwrap();
    std::fs::create_dir_all(dir.join("sessions")).unwrap();
    let labels: Vec<i64> = (0..40)
        .map(|t| if t < 5 { IGNORE } else { 1 + (t / 12) % 2 })
        .collect();
    session(&dir.join("sessions/a.npz"), &labels);
    session(&dir.join("sessions/b.npz"), &labels[..20]);
    let out = dir.join("scores.json");
    run(&[
        "eval",
        "behavior-sessions",
        "--model",
        model.to_str().unwrap(),
        "--sessions",
        dir.join("sessions").to_str().unwrap(),
        "--prob-smooth",
        "1,5",
        "--out",
        out.to_str().unwrap(),
    ])
    .unwrap();
    let all = loads(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let res = all.get(model.to_str().unwrap()).unwrap();
    let keys: Vec<String> = match res {
        PyVal::Dict(d) => d.iter().map(|(k, _)| k.clone()).collect(),
        _ => unreachable!(),
    };
    assert_eq!(
        keys,
        [
            "model",
            "frames",
            "classes_present",
            "label_share",
            "variants",
            "best",
            "per_class_f1_raw"
        ]
    );
    assert_eq!(res.get("frames").unwrap().as_f64(), Some(50.0)); // 35 + 15 labelled frames
    assert_eq!(
        res.get("classes_present").unwrap().dumps(),
        r#"["walking", "rearing"]"#
    );
    let variants: Vec<String> = match res.get("variants").unwrap() {
        PyVal::Dict(d) => d.iter().map(|(k, _)| k.clone()).collect(),
        _ => unreachable!(),
    };
    assert_eq!(variants, ["raw", "raw+prior", "smooth5", "smooth5+prior"]);
    let raw = res.get("variants").unwrap().get("raw").unwrap();
    assert_eq!(raw.get("n_frames").unwrap().as_f64(), Some(50.0));
    assert!(raw.get("confusion").is_none());

    // A second run adds to the same file.
    run(&[
        "eval",
        "behavior-sessions",
        "--model",
        model.to_str().unwrap(),
        "--sessions",
        dir.join("sessions").to_str().unwrap(),
        "--classes",
        "walking",
        "--no-prior",
        "--out",
        out.to_str().unwrap(),
    ])
    .unwrap();
    let all = loads(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let res = all.get(model.to_str().unwrap()).unwrap();
    assert_eq!(
        res.get("classes_present").unwrap().dumps(),
        r#"["walking"]"#
    );
    let n = match res.get("variants").unwrap() {
        PyVal::Dict(d) => d.len(),
        _ => 0,
    };
    assert_eq!(n, 3); // raw, smooth5, smooth15
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_checkpoint_survives_safetensors_and_pt_round_trips() {
    let dir = tmp("export");
    let cfg = ModelConfig::from_json_str(r#"{"dims": 2, "hidden": 16}"#).unwrap();
    let mut w = Weights::new();
    w.insert(
        "blocks.0.conv1.weight",
        &[2, 3],
        vec![0.5, -1.0, 2.0, 0.25, 0.0, 3.5],
    );
    w.insert("head.bias", &[2], vec![1.0, -1.0]);
    w.set_metadata("extra", r#"{"window": 64}"#);
    let original = dir.join("behavior.safetensors");
    animalpose::models::save(&original, ModelKind::Behavior, &cfg, &mut w).unwrap();
    let exported = dir.join("out/behavior.safetensors");
    run(&[
        "export",
        "safetensors",
        "--checkpoint",
        original.to_str().unwrap(),
        "--out",
        exported.to_str().unwrap(),
    ])
    .unwrap();
    let (_, kind, back) = animalpose::models::load(&exported).unwrap();
    assert_eq!(kind, ModelKind::Behavior);
    assert_eq!(back.data("head.bias").unwrap(), [1.0, -1.0]);
    assert_eq!(back.metadata_value("extra"), Some(r#"{"window": 64}"#));
    let pt = dir.join("out/behavior.pt");
    run(&[
        "export",
        "import-safetensors",
        "--safetensors",
        exported.to_str().unwrap(),
        "--out",
        pt.to_str().unwrap(),
        "--check",
        original.to_str().unwrap(),
    ])
    .unwrap();
    let mut z = zip::ZipArchive::new(std::fs::File::open(&pt).unwrap()).unwrap();
    let names: Vec<String> = z.file_names().map(str::to_string).collect();
    for want in [
        "behavior/data.pkl",
        "behavior/data/0",
        "behavior/data/1",
        "behavior/version",
    ] {
        assert!(names.iter().any(|n| n == want), "{names:?}");
    }
    let mut pkl = Vec::new();
    std::io::Read::read_to_end(&mut z.by_name("behavior/data.pkl").unwrap(), &mut pkl).unwrap();
    assert_eq!(&pkl[..2], [0x80, 2]); // pickle protocol 2, what torch.save writes
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn checks_without_a_rust_harness_refuse_by_name() {
    for check in ["rat7m-3d", "mirror-paws", "blur", "behavior-loao"] {
        let err = run(&["eval", check, "--anything", "x"]).unwrap_err();
        assert!(format!("{err}").starts_with("unsupported:"), "{err}");
    }
    let err = run(&["export", "models", "--pose", "p.pt"]).unwrap_err();
    assert!(format!("{err}").contains("torch.onnx"), "{err}");
}
