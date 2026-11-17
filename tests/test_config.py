"""``extends``, dotted overrides and the key check: the three things ``configs/*.yaml`` rely on."""

from pathlib import Path

import pytest

from animalpose.config import TRAIN_SCHEMA, ConfigError, load_config, load_tree, pipeline_schema, validate

PARENT = """
train:
  steps: 30
  lr: 0.5
data:
  species: [rat]
  sources: [{source: coco, weight: 0.5}]
model:
  backbone: {name: tiny, width: 32}
"""


def write(folder: Path, name: str, text: str) -> Path:
    """A config file under ``folder``, creating it."""
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(text, encoding="utf-8")
    return folder / name


def test_a_child_keeps_what_it_does_not_write(tmp_path):
    write(tmp_path, "parent.yaml", PARENT)
    child = write(tmp_path, "child.yaml", "extends: parent.yaml\ntrain:\n  steps: 5\n"
                                         "model:\n  backbone: {width: 64}\n")
    cfg = load_config(child)
    assert cfg["train"] == {"steps": 5, "lr": 0.5}  # steps replaced, lr inherited
    assert cfg["model"]["backbone"] == {"name": "tiny", "width": 64}  # a mapping merges key by key
    assert cfg["data"]["species"] == ["rat"]


def test_a_list_replaces_rather_than_appends(tmp_path):
    """The one rule worth remembering: a child that wants one more source has to write all of them."""
    write(tmp_path, "parent.yaml", PARENT)
    child = write(tmp_path, "child.yaml", "extends: parent.yaml\ndata:\n  sources: [{source: synthetic}]\n")
    assert load_config(child)["data"]["sources"] == [{"source": "synthetic"}]


def test_the_extends_key_is_gone_from_the_result(tmp_path):
    write(tmp_path, "parent.yaml", PARENT)
    child = write(tmp_path, "child.yaml", f"extends: {'parent.yaml'}\ntrain:\n  steps: 1\n")
    assert "extends" not in load_config(child)


def test_several_bases_lose_in_the_order_they_are_written(tmp_path):
    write(tmp_path, "a.yaml", "train:\n  steps: 1\n  lr: 0.1\n")
    write(tmp_path, "b.yaml", "train:\n  steps: 2\n")
    child = write(tmp_path, "child.yaml", "extends: [a.yaml, b.yaml]\n")
    assert load_config(child)["train"] == {"steps": 2, "lr": 0.1}  # the later base wins, a keeps lr
    child = write(tmp_path, "child2.yaml", "extends: [b.yaml, a.yaml]\n")
    assert load_config(child)["train"]["steps"] == 1


def test_bases_resolve_against_the_file_that_names_them(tmp_path):
    (tmp_path / "base").mkdir()
    write(tmp_path, "base/detector.yaml", "train:\n  steps: 40\n")
    child = write(tmp_path, "ablations/side.yaml", "extends: ../base/detector.yaml\ntrain:\n  lr: 0.2\n")
    assert load_config(child)["train"] == {"steps": 40, "lr": 0.2}


def test_an_override_beats_an_inherited_value(tmp_path):
    write(tmp_path, "parent.yaml", PARENT)
    child = write(tmp_path, "child.yaml", "extends: parent.yaml\ntrain:\n  steps: 5\n")
    cfg = load_config(child, ["train.steps=1", "train.seed=0", "data.species=[mouse, rat]"])
    assert cfg["train"] == {"steps": 1, "lr": 0.5, "seed": 0}  # override, inheritance, new key
    assert cfg["data"]["species"] == ["mouse", "rat"]


def test_a_chain_of_three_merges_top_down(tmp_path):
    write(tmp_path, "a.yaml", "train:\n  steps: 1\n  lr: 0.1\n  seed: 3\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\ntrain:\n  lr: 0.2\n")
    write(tmp_path, "c.yaml", "extends: b.yaml\ntrain:\n  seed: 9\n")
    assert load_config(tmp_path / "c.yaml")["train"] == {"steps": 1, "lr": 0.2, "seed": 9}


def test_a_base_that_is_not_there_says_so(tmp_path):
    child = write(tmp_path, "child.yaml", "extends: nowhere.yaml\n")
    with pytest.raises(ConfigError, match="config not found"):
        load_config(child)


def test_a_file_that_is_not_a_mapping_says_so(tmp_path):
    path = write(tmp_path, "list.yaml", "- one\n- two\n")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_two_files_extending_each_other_are_an_error(tmp_path):
    write(tmp_path, "a.yaml", "extends: b.yaml\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="extends itself"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("text", ["extends: 3\n", "extends: {a: b}\n", "extends: []\n"])
def test_extends_has_to_name_paths(tmp_path, text):
    child = write(tmp_path, "child.yaml", text)
    if text == "extends: []\n":  # no base at all is legal: it reads as a plain file
        assert load_config(child) == {}
    else:
        with pytest.raises(ConfigError, match="must name a path"):
            load_config(child)


def test_load_tree_of_a_file_without_extends_is_the_file(tmp_path):
    path = write(tmp_path, "solo.yaml", PARENT)
    assert load_tree(path)["train"]["steps"] == 30


def test_a_typo_is_refused_and_named(tmp_path):
    path = write(tmp_path, "cfg.yaml", "train:\n  stps: 5\n")
    with pytest.raises(ConfigError, match="stps.*did you mean 'steps'"):
        load_config(path, schema=TRAIN_SCHEMA)


def test_a_key_the_trainers_never_read_is_refused(tmp_path):
    path = write(tmp_path, "cfg.yaml", "traun:\n  steps: 5\n")
    with pytest.raises(ConfigError, match="unknown key 'traun'"):
        load_config(path, schema=TRAIN_SCHEMA)


def test_the_pipeline_schema_comes_from_the_dataclasses_that_read_it(tmp_path):
    path = write(tmp_path, "cfg.yaml", "checkpoints:\n  pose: runs/pose.pt\npipeline:\n  det_thresh: 0.4\n")
    cfg = load_config(path, schema=pipeline_schema())
    assert cfg["pipeline"]["det_thresh"] == 0.4
    bad = write(tmp_path, "bad.yaml", "pipeline:\n  det_thesh: 0.4\n")
    with pytest.raises(ConfigError, match="det_thesh.*did you mean 'det_thresh'"):
        load_config(bad, schema=pipeline_schema())


def test_validate_is_quiet_on_a_clean_tree(tmp_path):
    validate({"device": "auto", "train": {"steps": 5}}, {None: {"device", "train"}, "train": {"steps"}})


def test_every_shipped_config_resolves_and_passes_its_check():
    """The configs are documentation: each one has to load, and a child may not inherit a typo."""
    for path in sorted(Path("configs").glob("*.yaml")):
        if path.name in ("real_sources.yaml",) or path.name.startswith("camera_"):
            continue  # a source list and camera profiles are checked where they are read
        pipeline = path.name.startswith("pipeline") or "checkpoints" in load_tree(path)
        schema = pipeline_schema() if pipeline else TRAIN_SCHEMA
        load_config(path, schema=schema)
