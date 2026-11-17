//! `animalpose assay`: the experiment rules on finished analysis folders. Needs no network and no
//! GPU: it reads the saved tracks, so a changed threshold reruns in seconds.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};

pub fn command() -> Command {
    Command::new("assay")
        .about("score an experiment (zones, contact, grooming, …) on analysis folders")
        .arg(
            Arg::new("config")
                .long("config")
                .required(true)
                .help("pipeline config with an `assay:` section"),
        )
        .arg(
            Arg::new("set")
                .long("set")
                .action(ArgAction::Append)
                .help("override a config key, key=value"),
        )
        .arg(
            Arg::new("analysis")
                .long("analysis")
                .required(true)
                .num_args(1..)
                .help("analysis folders (tracks.npz, or a multiview folder)"),
        )
        .arg(
            Arg::new("video")
                .long("video")
                .help("the top view's video, for single-view folders"),
        )
        .arg(
            Arg::new("zones")
                .long("zones")
                .help("where arena layouts are cached [default: next to each folder, zones/]"),
        )
        .arg(
            Arg::new("force")
                .long("force")
                .action(ArgAction::SetTrue)
                .help("redo folders whose results are current"),
        )
}

pub fn run(m: &ArgMatches) -> Result<()> {
    let config = PathBuf::from(m.get_one::<String>("config").expect("required"));
    let overrides: Vec<String> = m
        .get_many::<String>("set")
        .into_iter()
        .flatten()
        .cloned()
        .collect();
    let video = m.get_one::<String>("video").map(PathBuf::from);
    let force = m.get_flag("force");
    let mut failed = Vec::new();
    for dir in m.get_many::<String>("analysis").into_iter().flatten() {
        let dir = Path::new(dir);
        let zones = m
            .get_one::<String>("zones")
            .map(PathBuf::from)
            .unwrap_or_else(|| dir.parent().unwrap_or(Path::new(".")).join("zones"));
        match crate::assay::output::run_config(
            dir,
            &config,
            &overrides,
            video.as_deref(),
            &zones,
            force,
        ) {
            Ok(None) => bail!("{} sets no assay.experiment", config.display()),
            Ok(Some(report)) => {
                let what = if report.ran {
                    "scored"
                } else {
                    "current, skipped"
                };
                println!("{}: {what} -> {}", dir.display(), report.dir.display());
                for note in report.notes {
                    println!("  {note}");
                }
            }
            Err(err) => {
                eprintln!("{}: failed: {err:#}", dir.display());
                failed.push(dir.display().to_string());
            }
        }
    }
    if !failed.is_empty() {
        bail!("{} folder(s) failed: {}", failed.len(), failed.join(", "));
    }
    Ok(())
}
