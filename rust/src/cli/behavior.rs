//! `animalpose behavior cluster|name|label|plot|sessions`: the port of
//! `src/animalpose/cli/behavior.py`, printing what the Python command prints.
//!
//! `sessions` (the public MIT / ABODe / BORIS dataset builders of `data/behavior_sets.py`) needs
//! their video readers and annotation parsers (pandas pickles, BORIS projects), which are not
//! ported; it is refused by name.

use std::path::Path;

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgMatches, Command};

use crate::cli::clusters::{cluster_tracks, label_clusters, parse_mapping, write_naming, FEATURES};
use crate::cli::pyjson::{fixed, py_round, Json};

/// The `behavior` command tree.
pub fn command() -> Command {
    let f64_arg = |id: &'static str, long: &'static str, default: &'static str| {
        Arg::new(id)
            .long(long)
            .value_parser(clap::value_parser!(f64))
            .default_value(default)
    };
    Command::new("behavior")
        .about("bootstrap behavior labels from pose tracks")
        .subcommand_required(true)
        .subcommand(
            Command::new("cluster")
                .about("cluster the kinematics of a tracks.npz")
                .arg(Arg::new("tracks").long("tracks").required(true))
                .arg(Arg::new("video").long("video").required(true))
                .arg(f64_arg("fps", "fps", "30.0"))
                .arg(
                    Arg::new("train_range")
                        .long("train-range")
                        .num_args(2)
                        .value_parser(clap::value_parser!(f64))
                        .default_values(["0.0", "0.8"]),
                )
                .arg(
                    Arg::new("k")
                        .long("k")
                        .value_parser(clap::value_parser!(usize))
                        .default_value("8"),
                )
                .arg(Arg::new("out").long("out").required(true)),
        )
        .subcommand(
            Command::new("name")
                .about("name the clusters by rule")
                .arg(Arg::new("clusters").long("clusters").required(true))
                .arg(
                    Arg::new("agree_with")
                        .long("agree-with")
                        .help("labels.json of hand-named clusters, to score the rules"),
                ),
        )
        .subcommand(
            Command::new("label")
                .about("write session files from a cluster=behavior mapping")
                .arg(Arg::new("clusters").long("clusters").required(true))
                .arg(
                    Arg::new("map")
                        .long("map")
                        .num_args(1..)
                        .required(true)
                        .help("cluster=behavior, e.g. 0=resting"),
                )
                .arg(f64_arg("min_bout", "min-bout", "0.5").help("seconds"))
                .arg(Arg::new("name").long("name").default_value("recording"))
                .arg(Arg::new("view").long("view").default_value("top"))
                .arg(Arg::new("out").long("out").required(true)),
        )
        .subcommand(
            Command::new("plot")
                .about("ethogram and annotated clip of a run")
                .arg(
                    Arg::new("run_dir")
                        .long("run")
                        .required(true)
                        .help("output folder of `animalpose predict`"),
                )
                .arg(Arg::new("video").long("video").required(true))
                .arg(f64_arg("clip_start", "clip-start", "0.0").help("seconds"))
                .arg(f64_arg("clip_seconds", "clip-seconds", "30.0"))
                .arg(
                    Arg::new("top")
                        .long("top")
                        .value_parser(clap::value_parser!(usize))
                        .default_value("3"),
                )
                .arg(
                    Arg::new("heldout_from")
                        .long("heldout-from")
                        .value_parser(clap::value_parser!(f64))
                        .help("seconds; drawn as a line"),
                ),
        )
        .subcommand(
            Command::new("sessions")
                .about("track and label the public behavior datasets")
                .arg(
                    Arg::new("dataset")
                        .required(true)
                        .value_parser(["mit", "abode", "boris"]),
                )
                .arg(
                    Arg::new("rest")
                        .num_args(..)
                        .trailing_var_arg(true)
                        .allow_hyphen_values(true),
                ),
        )
}

fn req<'a>(m: &'a ArgMatches, id: &str) -> Result<&'a String> {
    m.get_one::<String>(id)
        .with_context(|| format!("--{id} is required"))
}

/// Runs `animalpose behavior <task>`.
pub fn run(m: &ArgMatches) -> Result<()> {
    match m.subcommand() {
        Some(("cluster", c)) => {
            let range: Vec<f64> = c
                .get_many::<f64>("train_range")
                .into_iter()
                .flatten()
                .copied()
                .collect();
            let summary = cluster_tracks(
                Path::new(req(c, "tracks")?),
                Path::new(req(c, "video")?),
                Path::new(req(c, "out")?),
                *c.get_one::<f64>("fps").context("--fps")?,
                (range[0], range[1]),
                *c.get_one::<usize>("k").context("--k")?,
            )?;
            println!("{}", summary.dumps_indent(1));
            Ok(())
        }
        Some(("name", n)) => {
            let report = write_naming(
                Path::new(req(n, "clusters")?),
                n.get_one::<String>("agree_with").map(Path::new),
            )?;
            print!("{}", name_report(&report));
            Ok(())
        }
        Some(("label", l)) => {
            let specs: Vec<String> = l
                .get_many::<String>("map")
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            let share = label_clusters(
                Path::new(req(l, "clusters")?),
                &parse_mapping(&specs)?,
                Path::new(req(l, "out")?),
                req(l, "name")?,
                req(l, "view")?,
                *l.get_one::<f64>("min_bout").context("--min-bout")?,
            )?;
            println!("{}", share.dumps());
            Ok(())
        }
        Some(("plot", p)) => {
            let (share, dir, frames) = crate::cli::plot::plot_run(
                Path::new(req(p, "run_dir")?),
                Path::new(req(p, "video")?),
                *p.get_one::<f64>("clip_start").context("--clip-start")?,
                *p.get_one::<f64>("clip_seconds").context("--clip-seconds")?,
                *p.get_one::<usize>("top").context("--top")?,
                p.get_one::<f64>("heldout_from").copied(),
            )?;
            eprintln!("clip: {frames} frames -> {}", dir.display());
            println!("{}", share_json(&share).dumps());
            Ok(())
        }
        Some(("sessions", s)) => bail!(
            "unsupported: `behavior sessions {}` tracks and labels the public datasets with \
             src/animalpose/data/behavior_sets.py (build_mit / build_abode / build_boris_videos), \
             whose annotation readers (pandas pickles, BORIS projects) are not ported; run it \
             from Python",
            s.get_one::<String>("dataset")
                .map(String::as_str)
                .unwrap_or("?")
        ),
        _ => bail!("behavior needs a task: cluster, name, label, plot or sessions"),
    }
}

/// `{name: round(share, 3)}`, as `_plot` prints it.
pub fn share_json(share: &[(String, f64)]) -> Json {
    Json::Obj(
        share
            .iter()
            .map(|(n, v)| (n.clone(), Json::Float(py_round(*v, 3))))
            .collect(),
    )
}

/// What `_name` prints: one line per cluster (largest share first), the agreement when scored,
/// and the `map:` line that `behavior label --map` takes.
pub fn name_report(report: &Json) -> String {
    let mut out = String::new();
    let Some(Json::Obj(naming)) = report.get("naming") else {
        return out;
    };
    let share = |e: &Json| {
        e.get("share_of_train")
            .and_then(Json::as_f64)
            .unwrap_or(0.0)
    };
    let mut order: Vec<&(String, Json)> = naming.iter().collect();
    order.sort_by(|a, b| (-share(&a.1)).total_cmp(&(-share(&b.1))));
    for (c, entry) in order {
        let name = entry.get("name").and_then(Json::as_str).unwrap_or("");
        let feats = entry.get("features").cloned().unwrap_or(Json::Null);
        let cols: Vec<String> = FEATURES[..6]
            .iter()
            .map(|k| {
                format!(
                    "{k}={}",
                    fixed(feats.get(k).and_then(Json::as_f64).unwrap_or(f64::NAN), 2)
                )
            })
            .collect();
        out.push_str(&format!(
            "cluster {c} -> {name:<9} share {}  {}\n",
            fixed(share(entry), 3),
            cols.join("  ")
        ));
    }
    if let Some(agree) = report.get("agreement") {
        out.push_str(&format!("rule naming vs hand naming: {}\n", agree.dumps()));
    }
    let mut keys: Vec<&(String, Json)> = naming.iter().collect();
    keys.sort_by_key(|(c, _)| c.parse::<i64>().unwrap_or(i64::MAX));
    let pairs: Vec<String> = keys
        .iter()
        .map(|(c, e)| format!("{c}={}", e.get("name").and_then(Json::as_str).unwrap_or("")))
        .collect();
    out.push_str(&format!("map: {}\n", pairs.join(" ")));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::clusters::tests::{temp, write_tracks, T};

    type Pairs = Vec<(String, String)>;

    fn parse_name_report(text: &str) -> (Pairs, Pairs) {
        let mut named = Vec::new();
        let mut mapping = Vec::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("cluster ") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                named.push((parts[0].to_string(), parts[2].to_string()));
            } else if let Some(rest) = line.strip_prefix("map:") {
                mapping = rest
                    .split_whitespace()
                    .map(|p| {
                        let (a, b) = p.split_once('=').unwrap();
                        (a.to_string(), b.to_string())
                    })
                    .collect();
            }
        }
        (named, mapping)
    }

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn cluster_name_and_label_round_trip_through_the_command_line() {
        let dir = temp("cli");
        let tracks = dir.join("tracks.npz");
        write_tracks(&tracks, &["rest".to_string()], &vec![1.0; T]);
        let clusters = dir.join("clusters");
        let s = |p: &Path| p.display().to_string();
        crate::cli::run(&argv(&[
            "behavior",
            "cluster",
            "--tracks",
            &s(&tracks),
            "--video",
            &s(&dir.join("none.avi")),
            "--fps",
            "30",
            "--train-range",
            "0",
            "0.8",
            "--k",
            "3",
            "--out",
            &s(&clusters),
        ]))
        .unwrap();
        crate::cli::run(&argv(&["behavior", "name", "--clusters", &s(&clusters)])).unwrap();
        let report =
            Json::parse(&std::fs::read_to_string(clusters.join("naming.json")).unwrap()).unwrap();
        let text = name_report(&report);
        let (named, mapping) = parse_name_report(&text);
        let mut sorted = named.clone();
        sorted.sort();
        assert_eq!(mapping, sorted); // the map line is the same naming, sorted by cluster
        let names: Vec<&str> = named.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["running", "resting", "walking"]); // largest share first
        assert!(
            text.contains("speed=") && text.contains("posture=") && !text.contains("hand naming")
        );
        let pairs: Vec<String> = mapping.iter().map(|(c, n)| format!("{c}={n}")).collect();
        let out = dir.join("sessions");
        let mut args = argv(&["behavior", "label", "--clusters", &s(&clusters), "--map"]);
        args.extend(pairs);
        args.extend(argv(&[
            "--out",
            &s(&out),
            "--name",
            "synth",
            "--view",
            "top",
            "--min-bout",
            "0.5",
        ]));
        crate::cli::run(&args).unwrap();
        assert!(
            out.join("train/synth_train.npz").is_file()
                && out.join("heldout/synth_heldout.npz").is_file()
        );
        crate::cli::run(&argv(&[
            "behavior",
            "name",
            "--clusters",
            &s(&clusters),
            "--agree-with",
            &s(&out.join("labels.json")),
        ]))
        .unwrap();
        let report =
            Json::parse(&std::fs::read_to_string(clusters.join("naming.json")).unwrap()).unwrap();
        assert!(name_report(&report).contains(
            r#"rule naming vs hand naming: {"n_named_clusters": 3, "weighted_agreement": 1.0, "mismatches": []}"#
        ));
        let err = crate::cli::run(&argv(&[
            "behavior",
            "label",
            "--clusters",
            &s(&clusters),
            "--map",
            "0=sidewalking",
            "--out",
            &s(&dir.join("bad")),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("catalogue"));
        let err =
            crate::cli::run(&argv(&["behavior", "sessions", "mit", "--root", "x"])).unwrap_err();
        assert!(err.to_string().starts_with("unsupported:"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn name_lines_follow_the_python_format() {
        let report = Json::obj([(
            "naming",
            Json::obj([(
                "3",
                Json::obj([
                    ("name", Json::str("running")),
                    ("share_of_train", Json::Float(0.5)),
                    (
                        "features",
                        Json::Obj(
                            FEATURES
                                .iter()
                                .map(|f| (f.to_string(), Json::Float(1.005)))
                                .collect(),
                        ),
                    ),
                ]),
            )]),
        )]);
        assert_eq!(
            name_report(&report),
            "cluster 3 -> running   share 0.500  speed=1.00  length=1.00  turn=1.00  posture=1.00  head_motion=1.00  bend=1.00\nmap: 3=running\n"
        );
    }
}
