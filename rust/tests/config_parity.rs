//! Parity of `config.rs` with `src/animalpose/config.py:load_config`.
//!
//! `animalpose fixtures pipeline` runs the Python loader over every override string the
//! CLI can be handed (`train.steps=200`, `data.species=[mouse, rat]`, `=1`, …) against
//! `tests/fixtures/config/base.yaml` and records what came back, including the exceptions Python
//! raises. This test feeds the same strings to `Config::parse` and compares the whole tree, not just
//! the leaf, so a path created in the wrong place shows up too.
//!
//! The cases the ports do not read alike are flagged in the fixture with the reason, and the test
//! asserts each of them really does differ: PyYAML resolves plain scalars with the YAML 1.1 rules
//! while `serde_yaml` follows the 1.2 core schema, so `yes`/`off` are booleans there and strings
//! here, `1_000` is 1000 there, `0755` is octal there (493), `10:30` is sexagesimal there (630) and
//! bare `nan`/`inf` are floats there; and `load_config` happily stores `=1` under an empty key,
//! which the CLI refuses. None of those spellings occurs in `configs/*.yaml` or in the argument
//! lists the Python CLI builds, so a command that works in Python works here.

use std::path::PathBuf;

use animalpose::data::config::{Config, Value};
use anyhow::{Context, Result};
use serde_json::{json, Map, Value as Json};

/// The config fixture, or `None` when it was never generated.
fn fixture_dir() -> Option<PathBuf> {
    let dir = animalpose::data::fixtures_dir()?.join("config");
    dir.join("overrides.json").is_file().then_some(dir)
}

/// A `config::Value` in the tagged form the fixture writes, so the two can be compared as JSON.
///
/// Non-finite floats are compared by their `repr`, which the fixture records because strict JSON has
/// no spelling for them.
fn tagged(value: &Value) -> Json {
    match value {
        Value::Null => json!({"kind": "null"}),
        Value::Bool(v) => json!({"kind": "bool", "data": v}),
        Value::Int(v) => json!({"kind": "int", "data": v}),
        Value::Float(v) => {
            let repr = if v.is_nan() {
                "nan".to_string()
            } else if *v == f64::INFINITY {
                "inf".to_string()
            } else if *v == f64::NEG_INFINITY {
                "-inf".to_string()
            } else {
                format!("{v:?}")
            };
            json!({"kind": "float", "data": if v.is_finite() { Json::from(*v) } else { Json::Null }, "repr": repr})
        }
        Value::Str(v) => json!({"kind": "str", "data": v}),
        Value::List(items) => {
            json!({"kind": "list", "data": items.iter().map(tagged).collect::<Vec<Json>>()})
        }
        Value::Map(entries) => {
            let mut object = Map::new();
            for (key, item) in entries {
                object.insert(key.clone(), tagged(item));
            }
            json!({"kind": "map", "data": object})
        }
    }
}

#[test]
fn overrides_resolve_the_way_load_config_resolves_them() {
    if let Err(err) = run().map_err(|err| format!("{err:?}")) {
        panic!("{err}");
    }
}

/// Every fixture case, in order.
fn run() -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("SKIP: tests/fixtures/config is absent (run animalpose fixtures pipeline)");
        return Ok(());
    };
    let path = dir.join("overrides.json");
    let fixture: Json = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("{} is not JSON", path.display()))?;
    let base = fixture["base"].as_str().unwrap_or("");
    let cases = fixture["cases"].as_array().cloned().unwrap_or_default();
    anyhow::ensure!(!cases.is_empty(), "{} lists no cases", path.display());

    let mut agreed = 0;
    let mut documented = Vec::new();
    let flagged = cases
        .iter()
        .filter(|c| c["known_difference"].as_bool().unwrap_or(false))
        .count();
    for case in &cases {
        let item = case["override"].as_str().unwrap_or("").to_string();
        let key = case["key"].as_str().unwrap_or("").to_string();
        let outcome = Config::parse(base, std::slice::from_ref(&item));
        if let Some(want_error) = case["error"].as_str() {
            anyhow::ensure!(
                outcome.is_err(),
                "override {item:?} raised {want_error} in Python and was accepted here as {}",
                tagged(outcome?.section(&key).unwrap_or(&Value::Null))
            );
            agreed += 1;
            continue;
        }
        let want = &case["value"];
        let got = match outcome {
            Ok(cfg) => tagged(cfg.section(&key).unwrap_or(&Value::Null)),
            Err(err) => {
                anyhow::ensure!(
                    case["known_difference"].as_bool().unwrap_or(false),
                    "override {item:?} failed here ({err}) while Python read {want}"
                );
                Json::Null
            }
        };
        if got == *want {
            agreed += 1;
            continue;
        }
        anyhow::ensure!(
            case["known_difference"].as_bool().unwrap_or(false),
            "override {item:?}: Python read {want}, this parser read {got}"
        );
        documented.push(format!(
            "{item:?} ({}): python {want}, rust {got}",
            case["note"].as_str().unwrap_or("no reason recorded")
        ));
    }
    // The flag has to mean something: a case marked as differing must not silently agree again, or
    // the list above is stale.
    anyhow::ensure!(
        documented.len() == flagged,
        "{} of the {} flagged cases actually agree, so the list in the dump script is stale",
        flagged - documented.len(),
        flagged
    );
    println!(
        "config fixture {}: {} override strings agree with load_config, {} differ as documented",
        path.display(),
        agreed,
        documented.len()
    );
    for line in &documented {
        println!("  {line}");
    }
    Ok(())
}

#[test]
fn extends_files_merge_the_way_load_config_merges_them() {
    if let Err(err) = run_extends().map_err(|err| format!("{err:?}")) {
        panic!("{err}");
    }
}

/// Every `extends:` tree in the fixture, in order: the child's own keys on top of its bases, and the
/// bases losing among themselves in the order they are listed.
fn run_extends() -> Result<()> {
    let Some(dir) = fixture_dir() else {
        eprintln!("SKIP: tests/fixtures/config is absent (run animalpose fixtures pipeline)");
        return Ok(());
    };
    let path = dir.join("overrides.json");
    let fixture: Json = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )
    .with_context(|| format!("{} is not JSON", path.display()))?;
    let cases = fixture["extends"].as_array().cloned().unwrap_or_default();
    anyhow::ensure!(
        !cases.is_empty(),
        "{} lists no `extends` cases",
        path.display()
    );

    let mut compared = 0;
    for case in &cases {
        let file = case["file"].as_str().unwrap_or("").to_string();
        let overrides: Vec<String> = case["overrides"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        let outcome = Config::load(dir.join(&file), &overrides);
        if let Some(want_error) = case["error"].as_str() {
            anyhow::ensure!(
                outcome.is_err(),
                "{file} raised {want_error} in Python and was accepted here as {}",
                tagged(outcome?.tree())
            );
            compared += 1;
            continue;
        }
        let want = &case["value"];
        let got = tagged(
            outcome
                .with_context(|| format!("{file} failed here while Python read {want}"))?
                .tree(),
        );
        anyhow::ensure!(
            got == *want,
            "{file}: Python read {want}, this parser read {got}"
        );
        compared += 1;
    }
    println!(
        "config fixture {}: {} extends trees agree with load_config",
        path.display(),
        compared
    );
    Ok(())
}
