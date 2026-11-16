//! The `animalpose` command line: the same configs and the same outputs as the Python scripts.
//!
//! `crate::cli::train` — with `--init` for the checkpoint to start from — and `crate::cli::predict`
//! mirror `animalpose train`, `animalpose finetune` and `animalpose predict`, including their
//! argument surfaces: dotted `key.path=value` overrides, `--max-frames`, and an output prefix that
//! gains `tracks.csv`, `behavior.csv` and a checkpoint next to it. `version` prints the version.

pub mod audit;
pub mod calibrate;
pub mod catalog;
pub mod data_cmd;
pub mod download;
pub mod eval_cmd;
pub mod export_cmd;
pub mod overlay;
pub mod pairs;
pub mod predict;
pub mod smooth;
pub mod train;

// `label`, `finetune` and `behavior`: selftrain.py, behavior/clusters.py and behavior/plot.py.
pub mod behavior;
pub mod clusters;
pub mod label;
pub mod plot;
pub mod pyjson;
pub mod raster;
pub mod selftrain;
pub mod sources;

use anyhow::{bail, Context, Result};
use clap::{Arg, Command};
use std::path::Path;

/// The version this binary reports for `animalpose version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The `animalpose` command tree, built with clap's builder API.
pub fn command() -> Command {
    Command::new("animalpose")
        .version(VERSION)
        .about("train and run the animalpose pose, tracking and behavior networks")
        .subcommand(Command::new("version").about("print the version and exit"))
        .subcommand(label::label_command())
        .subcommand(label::finetune_command())
        .subcommand(behavior::command())
        .subcommand(
            Command::new("train")
                .about("train one network from a YAML config (animalpose train)")
                .arg(
                    Arg::new("model")
                        .required(true)
                        .value_parser(["behavior", "temporal", "pose", "detector"])
                        .help("which trainer to run"),
                )
                .arg(
                    Arg::new("config")
                        .required(true)
                        .value_name("CONFIG")
                        .help("the YAML config, e.g. configs/behavior.yaml"),
                )
                .arg(
                    Arg::new("init")
                        .long("init")
                        .value_name("CHECKPOINT")
                        .help("start from this checkpoint, overriding the config's init key"),
                )
                .arg(
                    Arg::new("overrides")
                        .num_args(..)
                        .trailing_var_arg(true)
                        .value_name("KEY=VALUE")
                        .help("dotted overrides, e.g. train.steps=200"),
                ),
        )
        .subcommand(
            Command::new("predict")
                .about(
                    "run detection, pose, tracking and behavior over frames (animalpose predict)",
                )
                .arg(
                    Arg::new("config")
                        .long("config")
                        .default_value(predict::DEFAULT_PIPELINE_CONFIG)
                        .value_name("CONFIG")
                        .help("the pipeline config"),
                )
                .arg(
                    Arg::new("frames")
                        .long("frames")
                        .visible_alias("video")
                        .value_name("DIR")
                        .help("a video file, or a folder of image frames (PNG/JPG/TIF/BMP/.npy)"),
                )
                .arg(
                    Arg::new("no_overlay")
                        .long("no-overlay")
                        .action(clap::ArgAction::SetTrue)
                        .help("skip writing overlay.mp4"),
                )
                .arg(
                    Arg::new("out")
                        .long("out")
                        .default_value("outputs/run")
                        .value_name("DIR")
                        .help("where tracks.csv, behavior.csv and tracks.json go"),
                )
                .arg(
                    Arg::new("max_frames")
                        .long("max-frames")
                        .value_parser(clap::value_parser!(usize))
                        .value_name("N")
                        .help("read at most N frames"),
                )
                .arg(
                    Arg::new("fps")
                        .long("fps")
                        .value_parser(clap::value_parser!(f64))
                        .value_name("FPS")
                        .help("frame rate; default: file-name timestamps, else 30"),
                )
                .arg(
                    Arg::new("views")
                        .long("views")
                        .num_args(1..)
                        .value_name("VIDEO")
                        .help("synchronised views (frame folders), one per --cameras entry"),
                )
                .arg(
                    Arg::new("cameras")
                        .long("cameras")
                        .value_name("JSON")
                        .help("Camera.to_dict() entries, one per --views entry"),
                )
                .arg(
                    Arg::new("synthetic")
                        .long("synthetic")
                        .action(clap::ArgAction::SetTrue)
                        .help("render a two-mouse demo clip instead of reading frames"),
                )
                .arg(
                    Arg::new("overrides")
                        .num_args(..)
                        .trailing_var_arg(true)
                        .value_name("KEY=VALUE")
                        .help("dotted overrides, e.g. pipeline.max_animals=2"),
                ),
        )
        .subcommand(data_cmd::command())
        .subcommand(eval_cmd::command())
        .subcommand(export_cmd::command())
}

/// Runs one invocation; `args` holds the command line after the program name.
///
/// `help` and `--help` print and count as success, as argparse does.
pub fn run(args: &[String]) -> Result<()> {
    let parsed = std::iter::once("animalpose").chain(args.iter().map(String::as_str));
    let matches = match command().try_get_matches_from(parsed) {
        Ok(matches) => matches,
        Err(err)
            if matches!(
                err.kind(),
                clap::error::ErrorKind::DisplayHelp
                    | clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
                    | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            print!("{err}");
            return Ok(());
        }
        Err(err) => return Err(err.into()),
    };
    match matches.subcommand() {
        Some(("label", m)) => label::run_label(m),
        Some(("finetune", m)) => label::run_finetune(m),
        Some(("behavior", m)) => behavior::run(m),
        Some(("version", _)) => {
            println!("animalpose {VERSION}");
            Ok(())
        }
        Some(("train", t)) => {
            let model = t
                .get_one::<String>("model")
                .context("`crate::cli::train` needs the network to build")?;
            let config = t
                .get_one::<String>("config")
                .context("`crate::cli::train` needs a config file")?;
            let mut overrides: Vec<String> = t
                .get_many::<String>("overrides")
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            // `animalpose.api:finetune` is `train` with `init=<checkpoint>` put in front of the
            // overrides, so a trailing `init=` still wins over `--init`.
            if let Some(init) = t.get_one::<String>("init") {
                overrides.insert(0, format!("init={init}"));
            }
            train::run(model, Path::new(config), &overrides)
        }
        Some(("predict", p)) => {
            let config =
                if p.value_source("config") == Some(clap::parser::ValueSource::DefaultValue) {
                    predict::default_pipeline_config()
                        .to_string_lossy()
                        .into_owned()
                } else {
                    p.get_one::<String>("config")
                        .context("predict needs a config")?
                        .clone()
                };
            // `animalpose predict:main`: `--views` runs the multi-view 3D pipeline.
            if let Some(views) = p.get_many::<String>("views") {
                let views: Vec<String> = views.cloned().collect();
                let overrides: Vec<String> = p
                    .get_many::<String>("overrides")
                    .into_iter()
                    .flatten()
                    .cloned()
                    .collect();
                return crate::multiview::pipeline::run_cli(
                    &config,
                    &views,
                    p.get_one::<String>("cameras")
                        .context("--views needs --cameras")?,
                    p.get_one::<String>("out").context("predict needs --out")?,
                    p.get_one::<usize>("max_frames").copied(),
                    &overrides,
                );
            }
            let overrides: Vec<String> = p
                .get_many::<String>("overrides")
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            if p.get_flag("synthetic") {
                // `api.py:predict(synthetic=True)`: a rendered two-mouse clip of `--max-frames`
                // (default 300) frames at 30 fps, written as grey frames and run like a folder.
                // differs from Python: the folder path also builds the median background when the
                // pipeline config asks for one, where Python's synthetic branch passes none.
                let n = p.get_one::<usize>("max_frames").copied().unwrap_or(300);
                let dir = std::env::temp_dir()
                    .join(format!("animalpose-synthetic-{}", std::process::id()));
                crate::data::render::write_demo_frames(&dir, n)?;
                let result = predict::run_with(
                    &config,
                    &dir.to_string_lossy(),
                    p.get_one::<String>("out")
                        .context("`crate::cli::predict` needs --out")?,
                    None,
                    Some(p.get_one::<f64>("fps").copied().unwrap_or(30.0)),
                    &overrides,
                    !p.get_flag("no_overlay"),
                );
                std::fs::remove_dir_all(&dir).ok();
                return result;
            }
            predict::run_with(
                &config,
                p.get_one::<String>("frames").context(
                    "`crate::cli::predict` needs --video (a video file or frame folder)",
                )?,
                p.get_one::<String>("out")
                    .context("`crate::cli::predict` needs --out")?,
                p.get_one::<usize>("max_frames").copied(),
                p.get_one::<f64>("fps").copied(),
                &overrides,
                !p.get_flag("no_overlay"),
            )
        }
        Some(("data", m)) => data_cmd::run(m),
        Some(("eval", m)) => eval_cmd::run(m),
        Some(("export", m)) => export_cmd::run(m),
        None => {
            // argparse prints the usage and exits 0 for a bare `animalpose`; `--help` does too.
            print!("{}", command().render_help());
            Ok(())
        }
        Some(other) => bail!("unknown subcommand {other:?}"),
    }
}
