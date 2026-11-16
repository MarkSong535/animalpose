//! Parity of `sessions.rs` with the `.npz` files numpy writes.
//!
//! `animalpose fixtures pipeline` slices the first 24 frames out of a real
//! `data/behavior_sets.py:save_session` archive and re-saves them with the same members and dtypes
//! (`tests/fixtures/sessions/session.npz`), together with what numpy and the Python loader read back
//! (`session.json`). This test reads that file and compares every step of the behaviour training
//! pipeline's data path: the stored arrays, the label vocabulary, `remap_labels`, and the resample
//! `load_session_dir` does on the way to a training batch.
//!
//! The `float32` arrays are compared value for value at `f32` width: both sides read the same bytes,
//! so a difference means a dtype or a shape was misread, not that a float was rounded differently.
//! A `float64` member is read at `f32` width, which is what training consumes. The test skips when
//! the fixture is absent.

use std::path::PathBuf;

use animalpose::data::sessions::{
    load_session_dir, read_npz, remap_labels, NpyData, Npz, Session, IGNORE,
};
use anyhow::{Context, Result};
use serde_json::Value;

/// The session fixture, or `None` when it was never generated.
fn fixture_dir() -> Option<PathBuf> {
    let dir = animalpose::data::fixtures_dir()?.join("sessions");
    dir.join("session.npz").is_file().then_some(dir)
}

/// The `data` of a `{"shape", "data"}` fixture entry as `f64`.
fn values(entry: &Value) -> Vec<f64> {
    entry["data"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .unwrap_or_default()
}

/// The shape of a fixture array.
fn shape(entry: &Value) -> Vec<usize> {
    entry["shape"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_u64())
                .map(usize::try_from)
                .filter_map(Result::ok)
                .collect()
        })
        .unwrap_or_default()
}

/// The strings of a fixture array.
fn strings(entry: &Value) -> Vec<String> {
    entry
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect()
}

/// A fixture number as `usize`.
fn count(entry: &Value) -> usize {
    usize::try_from(entry.as_u64().unwrap_or(0)).unwrap_or(usize::MAX)
}

/// A fixture array of integers (labels, class indices).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn integers(entry: &Value) -> Result<Vec<i64>> {
    values(entry)
        .into_iter()
        .map(|v| {
            let rounded = v.round();
            (rounded.abs() <= i64::MAX as f64).then_some(rounded as i64)
        })
        .collect::<Option<Vec<i64>>>()
        .context("a fixture integer array holds a value outside i64")
}

/// The numpy dtype name `NpyData` would have been parsed from, as `session.json` records it.
fn dtype_of(data: &NpyData) -> &'static str {
    match data {
        NpyData::Float(_) => "float32",
        NpyData::Int(_) => "int64",
        NpyData::Bytes(_) => "uint8",
        NpyData::Text(_) => "<U",
    }
}

#[test]
fn a_stored_session_reads_back_what_numpy_wrote() {
    if let Err(err) = run().map_err(|err| format!("{err:?}")) {
        panic!("{err}");
    }
}

/// Reads the fixture file and compares it against what Python recorded.
fn run() -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("SKIP: tests/fixtures/sessions is absent (run animalpose fixtures pipeline)");
        return Ok(());
    };
    let npz_path = dir.join("session.npz");
    let json_path = dir.join("session.json");
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(&json_path)
            .with_context(|| format!("cannot read {}", json_path.display()))?,
    )
    .with_context(|| format!("{} is not JSON", json_path.display()))?;

    // ---------------------------------------------------------------- the members on disk
    let archive: Npz = read_npz(&npz_path)?;
    for (name, want_dtype) in fixture["dtypes"].as_object().cloned().unwrap_or_default() {
        let Some(array) = archive.get(&name) else {
            bail_missing(&npz_path, &name);
        };
        let want_dtype = want_dtype.as_str().unwrap_or("");
        let got = dtype_of(&array.data);
        let agrees = match got {
            // `<U6`, `<U12`, ... any unicode width.
            "<U" => want_dtype.starts_with("<U"),
            // `NpyData::Float` holds `f32`: a `float64` member (`fps`, or a session written by an
            // older `save_session`) is read at the width the training path consumes anyway.
            "float32" => want_dtype == "float32" || want_dtype == "float64",
            other => want_dtype == other,
        };
        anyhow::ensure!(
            agrees,
            "{}: member {name} was written as {want_dtype}, read as {got}",
            npz_path.display()
        );
    }
    for name in ["coords", "conf", "boxes"] {
        if !archive.contains_key(name) {
            continue;
        }
        let array = archive
            .get(name)
            .with_context(|| format!("no {name} member"))?;
        anyhow::ensure!(
            array.shape == shape(&fixture[name]),
            "member {name} has shape {:?}, numpy wrote {:?}",
            array.shape,
            shape(&fixture[name])
        );
        let got = array.floats().context("{name} is not numeric")?;
        let want = values(&fixture[name]);
        anyhow::ensure!(
            got.len() == want.len(),
            "member {name} holds {} values, numpy wrote {}",
            got.len(),
            want.len()
        );
        #[allow(clippy::cast_possible_truncation)]
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            // The fixture prints its float32s with `%.9g`, which is the shortest form that reads
            // back as the same `f32`; compare at that width, as `dec` does in Python.
            anyhow::ensure!(
                *a as f32 == *b,
                "member {name} differs at {i}: {b} where numpy stored {a}"
            );
        }
    }

    // ---------------------------------------------------------------- the loaded session
    let session = Session::read(&npz_path)?;
    anyhow::ensure!(
        session.animal == fixture["animal"].as_str().unwrap_or(""),
        "the animal group"
    );
    anyhow::ensure!(
        session.view == fixture["view"].as_str().unwrap_or(""),
        "the camera view"
    );
    anyhow::ensure!(
        session.frames == count(&fixture["frames"]),
        "the session has {} frames, {} were stored",
        session.frames,
        fixture["frames"].as_u64().unwrap_or(0)
    );
    anyhow::ensure!(
        (session.fps - fixture["fps"].as_f64().unwrap_or(-1.0)).abs() < 1e-12,
        "the session runs at {}, numpy stored {}",
        session.fps,
        fixture["fps"].as_f64().unwrap_or(0.0)
    );
    let want_names = strings(&fixture["label_names"]);
    anyhow::ensure!(
        session.label_names == want_names,
        "the label vocabulary stored with the session was not read back"
    );
    anyhow::ensure!(
        session.keypoints == session.coords.len() / session.frames / 2 && session.keypoints > 0,
        "the session reports {} keypoints for {} frames",
        session.keypoints,
        session.frames
    );
    anyhow::ensure!(
        session.labels == integers(&fixture["labels"])?,
        "the stored labels were not read back"
    );

    // ---------------------------------------------------------------- remapping onto another vocabulary
    let class_names = strings(&fixture["classes"]);
    let classes: Vec<&str> = class_names.iter().map(String::as_str).collect();
    let remapped = session.with_classes(&classes);
    anyhow::ensure!(
        remapped.labels == integers(&fixture["remapped"])?,
        "remapping the labels onto {:?} gave {:?}, Python gave {:?}",
        classes,
        remapped.labels,
        integers(&fixture["remapped"])?
    );
    let ignored = integers(&fixture["remapped"])?
        .iter()
        .filter(|l| **l == IGNORE)
        .count();
    anyhow::ensure!(
        remapped.labels.iter().filter(|l| **l == IGNORE).count() == ignored,
        "the vocabulary {:?} leaves {} ignored frames, Python left {ignored}",
        classes,
        remapped.labels.iter().filter(|l| **l == IGNORE).count()
    );

    // The full 133-name vocabulary: every stored label keeps its meaning, and only a name the
    // catalogue lacks turns into `IGNORE`.
    let ext_names = strings(&fixture["ext_classes"]);
    let ext_classes: Vec<&str> = ext_names.iter().map(String::as_str).collect();
    let source_names: Vec<&str> = want_names.iter().map(String::as_str).collect();
    let onto_ext = remap_labels(&session.labels, &source_names, &ext_classes);
    for (from, to) in session.labels.iter().zip(&onto_ext) {
        let name = source_names
            .get(usize::try_from(*from).unwrap_or(usize::MAX))
            .copied();
        match name {
            Some(name) => anyhow::ensure!(
                ext_classes
                    .iter()
                    .position(|c| *c == name)
                    .map(|i| i64::try_from(i).unwrap_or(IGNORE))
                    == Some(*to)
                    || (*to == IGNORE && !ext_classes.contains(&name)),
                "label {name} became class {to} of the full vocabulary"
            ),
            None => anyhow::ensure!(
                *to == IGNORE,
                "label {from} became class {to}, expected {IGNORE}"
            ),
        }
    }

    // ---------------------------------------------------------------- resampling onto the training frame rate
    let loaded = load_session_dir(&dir, 15.0, None, &classes)?;
    anyhow::ensure!(loaded.len() == 1, "{} sessions were found", loaded.len());
    let loaded = &loaded[0];
    anyhow::ensure!(
        loaded.frames == count(&fixture["resampled_frames"]),
        "resampling to 15 fps kept {} frames, `load_session_dir` kept {}",
        loaded.frames,
        fixture["resampled_frames"].as_u64().unwrap_or(0)
    );
    anyhow::ensure!(
        (loaded.fps - fixture["resampled_fps"].as_f64().unwrap_or(-1.0)).abs() < 1e-12,
        "the resampled session runs at {}, `load_session_dir` reports {}",
        loaded.fps,
        fixture["resampled_fps"].as_f64().unwrap_or(0.0)
    );
    anyhow::ensure!(
        loaded.labels == integers(&fixture["resampled_labels"])?,
        "the resampled labels are {:?}, `load_session_dir` gave {:?}",
        loaded.labels,
        integers(&fixture["resampled_labels"])?
    );
    // Every resampled frame is a stride of the stored one, at the rate both sides computed.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let step = (session.fps / loaded.fps).round().max(1.0) as usize;
    anyhow::ensure!(
        loaded.coords == session.stride(step).coords && loaded.conf == session.stride(step).conf,
        "the resampled keypoints are not every {step}-th stored frame"
    );
    println!(
        "session fixture {}: {} frames of {} at {}, {} label names, remapped to {} classes, \
         resampled to {} frames at {} fps, bit-identical to numpy",
        npz_path.display(),
        session.frames,
        session.animal,
        session.view,
        session.label_names.len(),
        classes.len(),
        loaded.frames,
        loaded.fps
    );
    Ok(())
}

/// A member the fixture expects but the archive lacks.
#[allow(clippy::needless_pass_by_value)]
fn bail_missing(path: &std::path::Path, name: &str) -> ! {
    panic!("{} has no {name} member", path.display());
}
