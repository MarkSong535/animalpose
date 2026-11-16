//! `animalpose data <step>`: describe a camera and turn raw datasets into training data — the
//! port of `src/animalpose/cli/data/` (`convert`, `audit`, `calibrate`, `download`, `prepare`).
//!
//! Each step is one `Command` built by its own function and one arm in [`run`]; a further step
//! (e.g. `calibrate`) is one more `.subcommand(...)` in [`command`] and one more match arm.

use crate::cli::{audit, calibrate, download};
use crate::data::dlc::{read_dlc_csv, repr_str_list, to_coco};
use crate::data::pyjson::PyVal;
use crate::data::real::{
    convert_coco_keypoints, convert_lp_crim13, convert_mirror_mouse, convert_rat7m, image_root,
    json_files,
};
use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::path::PathBuf;

/// The `data` command group.
pub fn command() -> Command {
    Command::new("data")
        .about("measure a camera profile and prepare datasets")
        .subcommand_required(true)
        .subcommand(audit::command())
        .subcommand(prepare_command())
        .subcommand(convert_command())
        .subcommand(download::command())
        .subcommand(calibrate::command())
}

/// Runs `animalpose data <step>`.
pub fn run(m: &ArgMatches) -> Result<()> {
    match m.subcommand() {
        Some(("audit", a)) => audit::run(a),
        Some((calibrate::NAME, a)) => calibrate::run(a),
        Some(("prepare", a)) => prepare(a),
        Some(("convert", a)) => convert(a),
        Some(("download", a)) => {
            if download::run(a)? {
                Ok(())
            } else {
                bail!("some downloads failed verification")
            }
        }
        Some((other, _)) => bail!("unknown data step {other:?}"),
        None => bail!("animalpose data needs a step"),
    }
}

fn convert_command() -> Command {
    Command::new("convert")
        .about(
            "Convert DeepLabCut labeled data (CollectedData_<scorer>.csv) to COCO keypoint JSON.",
        )
        .arg(Arg::new("csv").required(true))
        .arg(Arg::new("out").long("out").required(true))
        .arg(
            Arg::new("image_root")
                .long("image-root")
                .help("project root to read image sizes from"),
        )
}

/// `data convert`: `cli/data/convert.py:run`.
fn convert(m: &ArgMatches) -> Result<()> {
    let csv = m.get_one::<String>("csv").context("convert needs a CSV")?;
    let out = m.get_one::<String>("out").context("convert needs --out")?;
    let labels = read_dlc_csv(csv)?;
    let root = m.get_one::<String>("image_root").map(PathBuf::from);
    let coco = to_coco(&labels, root.as_deref(), 0.1);
    std::fs::write(out, coco.dumps()).with_context(|| format!("cannot write {out}"))?;
    let count = |k: &str| {
        coco.get(k)
            .and_then(PyVal::as_list)
            .map_or(0, <[PyVal]>::len)
    };
    println!(
        "{} images, {} animals, keypoints: {}",
        count("images"),
        count("annotations"),
        repr_str_list(&labels.bodyparts)
    );
    Ok(())
}

fn flag(name: &'static str, long: &'static str) -> Arg {
    Arg::new(name).long(long).action(ArgAction::SetTrue)
}

fn prepare_command() -> Command {
    Command::new("prepare")
        .about("Convert downloaded public datasets into canonical COCO JSON under data/processed/.")
        .arg(Arg::new("raw").long("raw").default_value("data/raw"))
        .arg(Arg::new("out").long("out").default_value("data/processed"))
        .arg(flag("all", "all"))
        .arg(flag("mirror_mouse", "mirror-mouse"))
        .arg(flag("crim13", "crim13"))
        .arg(flag("rat7m", "rat7m"))
        .arg(flag("rat7m_s2", "rat7m-s2").help("subject 2 day 1, all frames held out"))
        .arg(flag("irodent", "irodent"))
        .arg(
            Arg::new("rat7m_every")
                .long("rat7m-every")
                .default_value("25")
                .value_parser(clap::value_parser!(usize))
                .help("keep every n-th video frame"),
        )
}

/// `data prepare`: `cli/data/prepare.py:run`.
fn prepare(m: &ArgMatches) -> Result<()> {
    let raw = PathBuf::from(
        m.get_one::<String>("raw")
            .map_or("data/raw", String::as_str),
    );
    let out = PathBuf::from(
        m.get_one::<String>("out")
            .map_or("data/processed", String::as_str),
    );
    let all = m.get_flag("all");
    let mut stats = PyVal::dict();
    if all || m.get_flag("mirror_mouse") {
        stats.set(
            "mirror_mouse",
            convert_mirror_mouse(
                &raw.join("lp_mirror_mouse/extracted/mirror-mouse"),
                &out.join("mirror_mouse"),
            )?,
        );
    }
    if all || m.get_flag("crim13") {
        stats.set(
            "crim13",
            convert_lp_crim13(&raw.join("lp_crim13/extracted/crim13"), &out.join("crim13"))?,
        );
    }
    let every = m.get_one::<usize>("rat7m_every").copied().unwrap_or(25);
    if all || m.get_flag("rat7m") {
        stats.set(
            "rat7m",
            convert_rat7m(
                &raw.join("rat7m/mocap-s1-d1.mat"),
                &raw.join("rat7m/s1-d1"),
                &out.join("rat7m"),
                every,
                Some(&[105_000, 122_500]),
            )?,
        );
    }
    if all || m.get_flag("rat7m_s2") {
        stats.set(
            "rat7m_s2",
            convert_rat7m(
                &raw.join("rat7m/mocap-s2-d1.mat"),
                &raw.join("rat7m/s2-d1"),
                &out.join("rat7m_s2"),
                every,
                None,
            )?,
        );
    }
    if all || m.get_flag("irodent") {
        let found = json_files(&raw.join("irodent/extracted"));
        let picked: Vec<&PathBuf> = found
            .iter()
            .filter(|p| {
                let n = p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase();
                n.contains("annotation") || n.contains("coco")
            })
            .collect();
        let ann = picked.first().copied().or(found.first()).with_context(|| {
            format!(
                "no annotation JSON under {}",
                raw.join("irodent/extracted").display()
            )
        })?;
        std::fs::create_dir_all(out.join("irodent"))?;
        let mut s = PyVal::dict().with("source", crate::cli::audit::py_path(ann));
        if let PyVal::Dict(items) =
            convert_coco_keypoints(ann, &out.join("irodent/test.json"), None)?
        {
            for (k, v) in items {
                s.set(k, v);
            }
        }
        stats.set("irodent", s);
        std::fs::write(
            out.join("irodent/IMAGE_ROOT"),
            image_root(ann)?.display().to_string(),
        )?;
    }
    println!("{}", stats.dumps_indent(1));
    Ok(())
}
