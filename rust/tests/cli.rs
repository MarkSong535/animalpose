use animalpose::cli::command;

/// `animalpose predict` has no Rust counterpart for multi-view input, so it fails by name.
#[test]
fn multi_view_is_refused_by_name() {
    let err = animalpose::cli::run(&[
        "predict".to_string(),
        "--views".to_string(),
        "left.mp4".to_string(),
        "right.mp4".to_string(),
    ])
    .expect_err("`--views` needs `--cameras`");
    let text = format!("{err}");
    assert!(text.contains("--views needs --cameras"), "{text}");

    // `--synthetic` renders its clip now; `tests/render_train.rs` runs it end to end.
}

#[test]
fn parses_the_subcommands() {
    let m = command()
        .try_get_matches_from(["animalpose", "version"])
        .expect("`version` is a subcommand");
    let (name, _) = m.subcommand().expect("`version` matched");
    assert_eq!(name, "version");
}

#[test]
fn parses_a_train_invocation() {
    let m = command()
        .try_get_matches_from([
            "animalpose",
            "train",
            "behavior",
            "configs/behavior.yaml",
            "train.steps=200",
        ])
        .expect("`train` is a subcommand");
    let (name, args) = m.subcommand().expect("`train` matched");
    assert_eq!(name, "train");
    assert_eq!(
        args.get_one::<String>("model").map(String::as_str),
        Some("behavior")
    );
    assert_eq!(
        args.get_one::<String>("config").map(String::as_str),
        Some("configs/behavior.yaml")
    );
    assert_eq!(
        args.get_many::<String>("overrides")
            .map(|v| v.cloned().collect::<Vec<_>>())
            .unwrap_or_default(),
        vec!["train.steps=200"]
    );
}

/// `animalpose train --init CHECKPOINT` is `animalpose finetune`, whose overrides carry `init=`.
#[test]
fn parses_a_train_invocation_with_an_init_checkpoint() {
    let m = command()
        .try_get_matches_from([
            "animalpose",
            "train",
            "behavior",
            "configs/behavior.yaml",
            "--init",
            "runs/behavior_mine.safetensors",
            "train.steps=200",
        ])
        .expect("`train --init` is a subcommand");
    let (name, args) = m.subcommand().expect("`train` matched");
    assert_eq!(name, "train");
    assert_eq!(
        args.get_one::<String>("init").map(String::as_str),
        Some("runs/behavior_mine.safetensors")
    );
    assert_eq!(
        args.get_many::<String>("overrides")
            .map(|v| v.cloned().collect::<Vec<_>>())
            .unwrap_or_default(),
        vec!["train.steps=200"]
    );
}
