//! `animalpose label` and `animalpose finetune`: the command-line surfaces of
//! `src/animalpose/cli/label.py` (flags generated from `selftrain.py:LabelOptions`) and of the
//! `finetune` verb of `src/animalpose/cli/train.py`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};

use crate::cli::selftrain::{label_videos, load_teacher, LabelOptions};

/// The `label` subcommand, one flag per `LabelOptions` field with the same names and defaults.
pub fn label_command() -> Command {
    // The defaults are `LabelOptions::default()` spelled out (clap takes static strings); the
    // test below checks the two agree.
    Command::new("label")
        .about("auto-label frames of every recording under --root (animalpose label)")
        .arg(
            Arg::new("root")
                .long("root")
                .required(true)
                .help("folder of videos and image folders, searched recursively"),
        )
        .arg(
            Arg::new("detector")
                .long("detector")
                .default_value("runs/detector_real.pt")
                .help("the teacher detector"),
        )
        .arg(
            Arg::new("pose")
                .long("pose")
                .default_value("runs/pose_real_r18_side.pt")
                .help("the teacher pose model"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .default_value("data/processed/unlabeled_pseudo")
                .help("where train.json, split.json and images/ go"),
        )
        .arg(
            Arg::new("det_size")
                .long("det-size")
                .num_args(2)
                .value_names(["H", "W"])
                .value_parser(clap::value_parser!(u32))
                .default_values(["480", "640"])
                .help("detector input size"),
        )
        .arg(num::<u32>(
            "crop_size",
            "crop-size",
            "192",
            "pose crop size",
        ))
        .arg(num::<usize>(
            "per_source",
            "per-source",
            "200",
            "frames sampled per recording",
        ))
        .arg(num::<f64>(
            "hi",
            "hi",
            "0.5",
            "detection score that counts as an animal",
        ))
        .arg(num::<f64>(
            "lo",
            "lo",
            "0.2",
            "below this the frame is background; between: skip",
        ))
        .arg(num::<f64>(
            "kp_conf",
            "kp-conf",
            "0.5",
            "keypoint confidence to keep",
        ))
        .arg(num::<f64>(
            "flip_tol",
            "flip-tol",
            "0.05",
            "keypoint flip agreement, in box widths",
        ))
        .arg(num::<usize>(
            "min_kps",
            "min-kps",
            "4",
            "keypoints an animal needs to be labelled",
        ))
        .arg(num::<usize>(
            "min_seq",
            "min-seq",
            "30",
            "frames needed to use the background filter",
        ))
        .arg(
            Arg::new("boxes")
                .long("boxes")
                .value_parser(["detector", "blob"])
                .default_value("detector")
                .help("'blob': one animal contrasting with the floor"),
        )
        .arg(
            Arg::new("time_range")
                .long("range")
                .num_args(2)
                .value_names(["FROM", "TO"])
                .value_parser(clap::value_parser!(f64))
                .default_values(["0.0", "1.0"])
                .help("fraction of each recording to label"),
        )
        .arg(
            Arg::new("no_holdout")
                .long("no-holdout")
                .action(ArgAction::SetTrue)
                .help("label every recording, not 4 in 5"),
        )
        .arg(
            Arg::new("include")
                .long("include")
                .num_args(1..)
                .help("only recordings whose path contains one of these"),
        )
        .arg(
            Arg::new("quiet")
                .long("quiet")
                .action(ArgAction::SetTrue)
                .help("print nothing until the summary"),
        )
}

fn num<T>(id: &'static str, long: &'static str, default: &'static str, help: &'static str) -> Arg
where
    T: Clone + Send + Sync + std::str::FromStr + 'static,
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    Arg::new(id)
        .long(long)
        .default_value(default)
        .value_parser(|s: &str| s.parse::<T>())
        .help(help)
}

fn two<T: Copy + Send + Sync + 'static>(m: &ArgMatches, id: &str) -> Result<(T, T)> {
    let v: Vec<T> = m
        .get_many::<T>(id)
        .with_context(|| format!("--{id} needs two values"))?
        .copied()
        .collect();
    Ok((v[0], v[1]))
}

/// `LabelOptions` from the parsed flags (`cli/base.py:build`).
pub fn label_options(m: &ArgMatches) -> Result<LabelOptions> {
    let s = |id: &str| -> Result<String> {
        m.get_one::<String>(id)
            .cloned()
            .with_context(|| format!("--{id} is missing"))
    };
    Ok(LabelOptions {
        detector: s("detector")?,
        pose: s("pose")?,
        out: PathBuf::from(s("out")?),
        det_size: two::<u32>(m, "det_size")?,
        crop_size: *m.get_one::<u32>("crop_size").context("--crop-size")?,
        per_source: *m.get_one::<usize>("per_source").context("--per-source")?,
        hi: *m.get_one::<f64>("hi").context("--hi")?,
        lo: *m.get_one::<f64>("lo").context("--lo")?,
        kp_conf: *m.get_one::<f64>("kp_conf").context("--kp-conf")?,
        flip_tol: *m.get_one::<f64>("flip_tol").context("--flip-tol")?,
        min_kps: *m.get_one::<usize>("min_kps").context("--min-kps")?,
        min_seq: *m.get_one::<usize>("min_seq").context("--min-seq")?,
        boxes: s("boxes")?,
        time_range: two::<f64>(m, "time_range")?,
        holdout: !m.get_flag("no_holdout"),
        include: m
            .get_many::<String>("include")
            .map(|v| v.cloned().collect()),
        progress: !m.get_flag("quiet"),
        ..LabelOptions::default()
    })
}

/// `cli/label.py:_run`: label, then print the stats as one JSON line.
pub fn run_label(m: &ArgMatches) -> Result<()> {
    let root = m.get_one::<String>("root").context("--root is required")?;
    let opts = label_options(m)?;
    let teacher = load_teacher(&opts)?;
    let stats = label_videos(Path::new(root), &teacher, &opts)?;
    println!("{}", stats.to_json().dumps());
    Ok(())
}

/// The `finetune` subcommand: `train` from an existing checkpoint (config key `init`).
pub fn finetune_command() -> Command {
    Command::new("finetune")
        .about("train from an existing checkpoint (config key `init`)")
        .arg(
            Arg::new("model")
                .required(true)
                .value_parser(["behavior", "temporal", "pose", "detector"])
                .help("which trainer to run"),
        )
        .arg(Arg::new("config").required(true).value_name("CONFIG"))
        .arg(
            Arg::new("init")
                .required(true)
                .value_name("INIT")
                .help("the checkpoint to start from, e.g. runs/pose_real.pt"),
        )
        .arg(
            Arg::new("overrides")
                .num_args(..)
                .trailing_var_arg(true)
                .value_name("KEY=VALUE")
                .help("dotted overrides, on top of `init`"),
        )
}

/// `api.py:finetune`: `train` with `init=<checkpoint>` in front of the overrides.
pub fn run_finetune(m: &ArgMatches) -> Result<()> {
    let model = m
        .get_one::<String>("model")
        .context("finetune needs a model")?;
    let config = m
        .get_one::<String>("config")
        .context("finetune needs a config")?;
    let init = m.get_one::<String>("init").context("finetune needs INIT")?;
    let overrides: Vec<String> = std::iter::once(format!("init={init}"))
        .chain(
            m.get_many::<String>("overrides")
                .into_iter()
                .flatten()
                .cloned(),
        )
        .collect();
    crate::cli::train::run(model, Path::new(config), &overrides)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_flags_default_to_label_options() {
        let m = label_command()
            .try_get_matches_from(["label", "--root", "videos"])
            .unwrap();
        let o = label_options(&m).unwrap();
        let d = LabelOptions::default();
        assert_eq!(
            (o.det_size, o.crop_size, o.per_source),
            (d.det_size, d.crop_size, d.per_source)
        );
        assert_eq!(
            (o.hi, o.lo, o.kp_conf, o.flip_tol),
            (d.hi, d.lo, d.kp_conf, d.flip_tol)
        );
        assert_eq!(
            (o.min_kps, o.min_seq, o.boxes.as_str()),
            (4, 30, "detector")
        );
        assert_eq!(
            (o.time_range, o.holdout, o.progress),
            ((0.0, 1.0), true, true)
        );
        assert!(o.include.is_none());
        let m = label_command()
            .try_get_matches_from([
                "label",
                "--root",
                "v",
                "--boxes",
                "blob",
                "--range",
                "0.2",
                "0.6",
                "--no-holdout",
                "--include",
                "my_video",
                "other",
                "--quiet",
                "--det-size",
                "240",
                "320",
            ])
            .unwrap();
        let o = label_options(&m).unwrap();
        assert_eq!(
            (o.boxes.as_str(), o.time_range, o.holdout, o.progress),
            ("blob", (0.2, 0.6), false, false)
        );
        assert_eq!(
            o.include,
            Some(vec!["my_video".to_string(), "other".to_string()])
        );
        assert_eq!(o.det_size, (240, 320));
        assert!(label_command()
            .try_get_matches_from(["label", "--root", "v", "--boxes", "network"])
            .is_err());
    }

    #[test]
    fn finetune_puts_init_before_the_overrides() {
        let m = finetune_command()
            .try_get_matches_from([
                "finetune",
                "pose",
                "configs/pose.yaml",
                "w.pt",
                "train.steps=2",
            ])
            .unwrap();
        assert_eq!(
            m.get_one::<String>("init").map(String::as_str),
            Some("w.pt")
        );
        let rest: Vec<&String> = m.get_many::<String>("overrides").unwrap().collect();
        assert_eq!(rest, ["train.steps=2"]);
    }
}
