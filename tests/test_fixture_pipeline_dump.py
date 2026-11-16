"""What ``animalpose fixtures pipeline`` must emit, pinned by running it.

``rust/tests/pipeline_parity.rs``, ``rust/tests/config_parity.rs`` and ``rust/tests/sessions_parity.rs``
never regenerate the reference: they read ``tests/fixtures/pipeline`` (the frame PNGs, ``pipeline.yaml``,
``fixture.json``, the CSVs), ``tests/fixtures/config/overrides.json`` and ``tests/fixtures/sessions``
exactly as ``src/animalpose/cli/fixtures/pipeline.py`` leaves them. The dump's *shape* is therefore itself
the contract, and this file drives the dumper's own ``run(args)`` to pin it: which files exist; that the
YAML and JSON carry checkout-relative paths both ports resolve the same way; that the PNGs decode to the
8-bit grey arrays and the frame rate the Rust readers take from the file names; that every stage and track
entry carries the arrays its Rust loop indexes, in the ``{"shape", "data"}`` / SHA-256 spellings
``values()`` and ``sha256_f32`` parse, as strict JSON that ``serde_json`` accepts; that the CSVs are CRLF
with key columns that reproduce the recorded arrays; that the emitted config survives
``config.py:load_config`` unchanged (the config-parity fixture depends on it); and that the session ``.npz``
slice keeps the stored dtypes and the stride-aligned labels ``sessions_parity.rs`` compares.

Eight frames -- the default -- are the shortest clip whose tracks reach the five-frame minimum of the
temporal and behaviour stages: a shorter clip loses ``smoothed``, the recorded classifier inputs and
``behavior.csv`` entirely (asserted as a branch below), so the fixture would pin almost nothing. The dump
runs against a throwaway checkout whose ``data/`` holds one *simulated* behavior session, so nothing is
read from the downloaded recordings and nothing is written outside the temp directory. Multi-view is out of
scope: the dumper writes one seeded simulation through one top camera, so no multi-view case exists to pin.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import re
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from animalpose.cli import main
from animalpose.cli.fixtures import pipeline as dumper
from animalpose.config import ConfigError, load_config, pipeline_schema
from animalpose.data.behavior_sets import remap_labels, save_session
from animalpose.data.datasets import letterbox, to_tensor
from animalpose.data.synthetic import simulate
from animalpose.data.tracks import IGNORE, Session
from animalpose.export.safetensors import metadata_for, read_safetensors, write_safetensors
from animalpose.frames import estimate_fps, frame_timestamps, image_files, read_frames
from animalpose.labels import BEHAVIORS_EXT
from animalpose.multiview.symmetry import bilateral_pairs
from animalpose.paths import REPO
from animalpose.pipeline import PipelineConfig
from animalpose.skeleton import CANONICAL
from animalpose.tracking.tracker import TrackerConfig
from animalpose.train.checkpoint import build_model, save_checkpoint

FRAMES = 8
NAME = "pipeline"
KEEP = 24  # `sessions_parity.rs`: the fixture is "the first 24 frames" of a stored session
SESSION_FRAMES, SESSION_FPS = 32, 30.0  # -> 12 frames survive the resample onto 15 fps
KEYPOINTS, N_PAIRS = CANONICAL.num_keypoints, len(bilateral_pairs())
CLASSES = list(dumper.TINY_MODELS["behavior"]["classes"])
KINDS = ("detector", "pose", "temporal", "behavior")
FIXTURE_FILES = {"frames", "pipeline.yaml", "fixture.json", "tracks.csv", "behavior.csv",
                 *(f"{kind}.safetensors" for kind in KINDS)}
# The stage keys `pipeline_parity.rs` indexes; `embeddings` is dumped but only read by the pose net itself.
STAGE_KEYS = {"detector_input", "net_boxes", "net_scores", "blob_boxes", "blob_scores", "boxes", "scores",
              "keypoints", "conf", "embeddings", "pair_margin", "pose_inputs"}
# The track keys of its track loop: the compared arrays plus the three integer ones.
TRACK_KEYS = {"frames", "boxes", "keypoints", "conf", "smoothed", "smoothed_conf", "imputed", "pair_margin",
              "pair_swapped", "behavior", "behavior_probs", "behavior_full"}
# Every top-level name the Rust parity test indexes by name.
FIXTURE_KEYS = {"seed", "frames", "fps", "size", "models", "pipeline", "tracker", "times", "behaviors",
                "background", "background_median", "stages", "tracks", "n_frames", "behavior_inputs"}
SESSION_KEYS = {"source", "frames", "coords", "conf", "labels", "fps", "animal", "view", "label_names",
                "dtypes", "classes", "remapped", "ignored", "resampled_frames", "resampled_fps",
                "resampled_labels", "ext_classes"}

Dump = namedtuple("Dump", "repo root fixture config sessions source doc")


def _args(*argv: str) -> argparse.Namespace:
    """The namespace ``animalpose fixtures pipeline <argv>`` hands to ``run``."""
    parser = argparse.ArgumentParser()
    dumper.add_arguments(parser)
    return parser.parse_args(list(argv))


def _reject(token: str) -> float:
    raise AssertionError(f"serde_json refuses the non-standard JSON token {token}")


def _load(path: Path):
    """A fixture parsed the way ``serde_json`` parses it: no ``NaN``/``Infinity``, no non-finite value."""
    doc = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject)
    json.dumps(doc, allow_nan=False)
    return doc


def _values(entry: dict) -> np.ndarray:
    """A ``{"shape", "data"}`` entry as an array, ``null`` read back as NaN like Rust's ``values()``."""
    shape = [int(d) for d in entry["shape"]]
    assert all(d >= 0 for d in shape), entry["shape"]
    flat = [np.nan if v is None else float(v) for v in entry["data"]]
    assert len(flat) == int(np.prod(shape, dtype=np.int64)), f"{shape} vs {len(flat)} values"
    return np.asarray(flat, np.float64).reshape(shape)


def _hashed(entry: dict) -> None:
    """The ``{"shape", "sha256", "nbytes"}`` spelling of a network input."""
    assert set(entry) == {"shape", "sha256", "nbytes"}
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["sha256"]
    assert entry["nbytes"] == 4 * int(np.prod(entry["shape"], dtype=np.int64))


def _sha_f32(array: np.ndarray) -> str:
    """SHA-256 of the row-major little-endian float32 bytes, as ``sha256_f32`` in the Rust test hashes."""
    return hashlib.sha256(np.ascontiguousarray(array, np.float32).astype("<f4").tobytes()).hexdigest()


def _png(path: Path) -> np.ndarray:
    """A written frame, decoded the way ``images.rs:u8_from_gray`` reads it."""
    grey = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert grey is not None, path
    return grey


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU, spelled out here rather than taken from ``pipeline.py``, to check the merged boxes."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union) if union > 0 else 0.0


def _csv_rows(path: Path) -> list[list[str]]:
    """A CSV the ``csv`` module wrote, split the way ``pipeline_parity.rs:read_csv`` splits it."""
    text = path.read_bytes().decode("utf-8")
    assert text.endswith("\r\n") and "\n" not in "".join(text.split("\r\n")), path  # CRLF, nothing else
    return [row.split(",") for row in text.rstrip("\r\n").split("\r\n")]


def _fake_repo(root: Path) -> tuple[Path, Path]:
    """A checkout stand-in: an empty ``tests/fixtures`` and one simulated session under ``data/``."""
    (root / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    sim = simulate(1, SESSION_FRAMES, "mouse", seed=7)
    coords = (sim.kpts3d[0, :, :, :2] * 4.0).astype(np.float32)
    centre = coords.mean(axis=1)
    session = Session("probe1", "top", coords, np.full((SESSION_FRAMES, KEYPOINTS), 0.7, np.float32),
                      np.resize([0, 1, 2, 3, 4, 5, IGNORE], SESSION_FRAMES).astype(np.int64), SESSION_FPS)
    source = root / "data" / "processed" / "behavior" / "synthetic" / "mouse0.npz"
    save_session(source, session, np.c_[centre - 8.0, centre + 8.0].astype(np.float32), BEHAVIORS_EXT)
    return root, source


def _run_into(repo: Path, *argv: str) -> None:
    """``run(args)`` with the checkout root moved into ``repo``, as the tests must do to stay in tmp."""
    saved, dumper.REPO = dumper.REPO, repo
    try:
        dumper.run(_args(*argv))
    finally:
        dumper.REPO = saved


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    """One full dump (tiny weights, config and session fixtures included) in a throwaway checkout."""
    repo, source = _fake_repo(tmp_path_factory.mktemp("pipeline-repo"))
    _run_into(repo, "--out", "tests/fixtures", "--name", NAME, "--frames", str(FRAMES))
    root = repo / "tests" / "fixtures"
    return Dump(repo, root, root / NAME, root / "config", root / "sessions", source,
                _load(root / NAME / "fixture.json"))


# --------------------------------------------------------------------------------------- written files


def test_the_dump_is_the_file_set_the_three_rust_readers_open(dump):
    """No file a parity test opens may be missing, and no stray one may be written next to them."""
    assert {p.name for p in dump.fixture.iterdir()} == FIXTURE_FILES
    assert {p.name for p in dump.root.iterdir()} == {NAME, "config", "sessions"}
    assert len(list((dump.fixture / "frames").iterdir())) == FRAMES
    assert {p.name for p in dump.config.iterdir()} == {"base.yaml", "overrides.json", *dumper.EXTENDS_FILES}
    assert {p.name for p in dump.sessions.iterdir()} == {"session.npz", "session.json"}
    assert all((dump.fixture / name).stat().st_size for name in FIXTURE_FILES - {"frames"})
    assert set(dump.doc) == FIXTURE_KEYS


def test_the_sidecar_weights_carry_the_architectures_the_run_used(dump):
    """``Pipeline::from_config`` builds the networks from the sidecar metadata, so it must be the run's."""
    expected_extra = {"detector": {"input_size": dump.doc["pipeline"]["det_size"]},
                      "pose": {"crop_size": dump.doc["pipeline"]["crop_size"]},
                      "temporal": {"window": dump.doc["pipeline"]["temporal_window"]},
                      "behavior": {"window": dump.doc["pipeline"]["temporal_window"]}}
    for kind in KINDS:
        tensors, meta = read_safetensors(dump.fixture / f"{kind}.safetensors")
        assert meta["kind"] == kind
        assert json.loads(meta["animalpose"]) == dump.doc["models"][kind] == dumper.TINY_MODELS[kind]
        assert json.loads(meta["extra"]) == expected_extra[kind]
        assert tensors, kind
        assert {str(v.dtype) for v in tensors.values()} <= {"torch.float32", "torch.int64"}
        assert all(torch.isfinite(v).all() for v in tensors.values() if v.is_floating_point()), kind
    # the 25-keypoint vocabulary is the only one the Rust batch builders and the pipeline speak
    assert dump.doc["models"]["pose"]["num_keypoints"] == KEYPOINTS


def test_the_frame_names_carry_the_ten_fps_both_ports_read(dump):
    """The rate is not stored in the PNGs: both readers take it from the ``_HH_MM_SS_mmm`` names."""
    files = image_files(dump.fixture / "frames")
    assert [p.name for p in files] == sorted(p.name for p in files)
    for p in files:
        assert re.fullmatch(r"frame_\d\d_\d\d_\d\d_\d\d\d\.png", p.name), p.name
    assert estimate_fps(files) == pytest.approx(10.0)
    assert frame_timestamps(dump.fixture / "frames") == pytest.approx(np.arange(FRAMES) * 0.1, abs=1e-9)
    assert dump.doc["fps"] == pytest.approx(10.0, abs=1e-9)
    assert dump.doc["frames"] == FRAMES == dump.doc["n_frames"]
    times = np.asarray(dump.doc["times"]["data"], float)
    assert times.shape == (FRAMES,) and np.all(np.diff(times) > 0)
    assert times == pytest.approx(np.arange(FRAMES) * 0.1, abs=1e-9)  # `worst <= 1e-9` in the Rust test


def test_the_frames_are_the_grey_pngs_the_rust_loader_expects(dump):
    """``frames[0].height/width`` and the grey levels behind ``u8_from_gray`` + ``to_bgr8``."""
    height, width = dump.doc["size"]
    files = image_files(dump.fixture / "frames")
    for path in files:
        grey = _png(path)
        assert grey.dtype == np.uint8 and grey.ndim == 2 and grey.shape == (height, width), path.name
        assert int(grey.max()) - int(grey.min()) > 100 and len(np.unique(grey)) > 64  # not a blank canvas
    frames, fps = read_frames(dump.fixture / "frames")
    assert fps == pytest.approx(10.0) and len(frames) == FRAMES
    assert frames[0].shape == (height, width, 3) and frames[0].dtype == np.uint8
    assert np.array_equal(cv2.cvtColor(_png(files[0]), cv2.COLOR_GRAY2BGR), frames[0])


# -------------------------------------------------------------------------------------------- pipeline.yaml


def test_pipeline_yaml_points_at_weight_files_both_ports_resolve(dump):
    """Repository fixtures keep repo-relative paths, so the Rust CLI reads the very same file."""
    cfg = yaml.safe_load((dump.fixture / "pipeline.yaml").read_bytes().decode("utf-8"))
    assert set(cfg) == {"device", "checkpoints", "pipeline", "tracker"}
    assert cfg["device"] == "cpu" and set(cfg["checkpoints"]) == set(KINDS)
    inside = dump.fixture.relative_to(dump.repo).as_posix()
    for kind, rel in cfg["checkpoints"].items():
        assert not Path(rel).is_absolute() and "\\" not in rel, rel
        assert rel == f"{inside}/{kind}.safetensors"
        assert (dump.repo / rel).is_file(), rel  # relative to the checkout root, as `Config::load` resolves it


def test_the_emitted_pipeline_yaml_round_trips_through_the_python_loader(dump):
    """The config-parity fixture depends on ``load_config`` reading this file, keys and all."""
    path = dump.fixture / "pipeline.yaml"
    tree = yaml.safe_load(path.read_bytes().decode("utf-8"))
    assert load_config(path, [], schema=pipeline_schema()) == tree
    bumped = load_config(path, ["pipeline.max_det=1", "tracker.max_animals=1"])
    assert (bumped["pipeline"]["max_det"], bumped["tracker"]["max_animals"]) == (1, 1)
    assert set(bumped["pipeline"]) == set(tree["pipeline"])
    assert bumped["checkpoints"] == tree["checkpoints"]
    with pytest.raises(ConfigError):
        load_config(path, ["pipeline.det_siz=1"], schema=pipeline_schema())  # a typo, not a default
    cfg = PipelineConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in tree["pipeline"].items()},
                         tracker=TrackerConfig(**tree["tracker"]))
    assert cfg.det_size == (96, 96) and cfg.detector_candidates == ("network", "blob")
    assert cfg.background_filter and cfg.pair_identity and cfg.embedder == "model"
    # the run restates the two sections it read, so the Rust `Config::load` of this file sees the same
    assert dump.doc["pipeline"] == tree["pipeline"] and dump.doc["tracker"] == tree["tracker"]


# ------------------------------------------------------------------------------------------------ stages


def test_the_detector_input_hash_is_the_array_the_rust_side_hashes(dump):
    """A SHA-256 over bit-identical float32 bytes: re-derived here from the PNG, not from the run."""
    assert dumper.sha_f32(np.arange(6.0).reshape(2, 3)) == _sha_f32(np.arange(6.0).reshape(2, 3))
    det_size = dump.doc["pipeline"]["det_size"]
    for i, (stage, path) in enumerate(zip(dump.doc["stages"], image_files(dump.fixture / "frames"))):
        entry = stage["detector_input"]
        _hashed(entry)
        grey = _png(path)
        tensor, _, _ = letterbox(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), det_size)
        fed = to_tensor(tensor).numpy()
        assert entry["shape"] == [3, *grey.shape], f"frame {i}"
        assert entry["sha256"] == _sha_f32(fed), f"frame {i} is not the array Python fed the network"


def test_every_stage_names_the_arrays_the_rust_stage_loop_reads(dump):
    """Names, shapes and the merged-box bookkeeping the per-frame Rust comparison indexes by name."""
    max_det, merge_iou = dump.doc["pipeline"]["max_det"], dump.doc["pipeline"]["merge_iou"]
    embed_dim = dump.doc["models"]["pose"]["embed_dim"]
    for i, stage in enumerate(dump.doc["stages"]):
        assert set(stage) == STAGE_KEYS, f"frame {i}"
        _hashed(stage["detector_input"])
        for key in ("net_boxes", "blob_boxes", "boxes"):
            assert _values(stage[key]).shape[1:] == (4,), f"frame {i} {key}"
        net, blob = _values(stage["net_boxes"]), _values(stage["blob_boxes"])
        boxes, scores = _values(stage["boxes"]), _values(stage["scores"])
        assert len(boxes) == len(scores) <= max_det and np.all(np.diff(scores) <= 0), f"frame {i}"
        assert np.isfinite(boxes).all() and np.isfinite(scores).all(), f"frame {i}"
        pool = np.concatenate([net, blob])
        pooled = np.concatenate([_values(stage["net_scores"]), _values(stage["blob_scores"])])
        for box, score in zip(boxes, scores):  # merged candidates, never re-derived or rescaled
            hit = np.flatnonzero((pool == box).all(-1))
            assert len(hit) == 1 and pooled[hit[0]] == score, f"frame {i}"
        for a, b in itertools.combinations(boxes, 2):
            assert _iou(a, b) < merge_iou, f"frame {i}: {a} and {b} should have been merged"
        assert _values(stage["keypoints"]).shape == (len(boxes), KEYPOINTS, 2), f"frame {i}"
        assert _values(stage["conf"]).shape == (len(boxes), KEYPOINTS), f"frame {i}"
        assert _values(stage["embeddings"]).shape == (len(boxes), embed_dim), f"frame {i}"
        assert _values(stage["pair_margin"]).shape == (len(boxes), N_PAIRS), f"frame {i}"
        for key in ("net_boxes", "net_scores", "blob_boxes", "blob_scores", "keypoints", "conf"):
            assert all(v is None or np.isfinite(v) for v in stage[key]["data"]), f"frame {i} {key}"


def test_the_pose_crop_hashes_describe_one_crop_per_merged_box(dump):
    """The warped crop is the one value parity cannot promise byte for byte, so it ships as a digest."""
    crop = dump.doc["pipeline"]["crop_size"]
    hashes = []
    for stage in dump.doc["stages"]:
        assert len(stage["pose_inputs"]) == len(stage["boxes"]["data"]) // 4
        for entry in stage["pose_inputs"]:
            _hashed(entry)
            assert entry["shape"] == [3, crop, crop]
            hashes.append(entry["sha256"])
    assert len(set(hashes)) == len(hashes)  # one distinct crop per box per frame


def test_the_median_background_is_reproducible_from_the_written_pngs(dump):
    """The Rust side hashes its own median; both Python background paths must hash the same."""
    greys = [_png(p) for p in image_files(dump.fixture / "frames")]
    stacked = np.stack([g.astype(np.float32) for g in greys])
    median = np.median(stacked, 0)
    for key in ("background", "background_median"):
        entry = dump.doc[key]
        _hashed(entry)
        assert entry["shape"] == list(median.shape), key
        assert entry["sha256"] == _sha_f32(median), key
    assert dump.doc["background"] == dump.doc["background_median"]


# ------------------------------------------------------------------------------------------------- tracks


def test_the_track_dumps_name_every_array_the_rust_track_loop_compares(dump):
    """Array names, integer spellings and the widths the Rust track loop sizes its rows from."""
    assert dump.doc["behaviors"] and len(dump.doc["behaviors"]) > len(CLASSES)
    for tid, entry in dump.doc["tracks"].items():
        assert str(int(tid)) == tid and int(tid) >= 0  # `id.parse::<i64>()`
        assert set(entry) == TRACK_KEYS, tid
        n = len(entry["frames"]["data"])
        for name in ("frames", "behavior", "pair_swapped"):
            assert all(isinstance(v, int) for v in entry[name]["data"]), f"track {tid} {name}"
        frames = _values(entry["frames"])
        assert list(frames) == list(range(int(frames[0]), int(frames[0]) + n))  # contiguous, as tracked
        for name, width in {"boxes": 4, "conf": KEYPOINTS, "smoothed_conf": KEYPOINTS,
                            "imputed": KEYPOINTS, "pair_margin": N_PAIRS,
                            "behavior_probs": len(CLASSES), "behavior_full": len(dump.doc["behaviors"])}.items():
            assert _values(entry[name]).shape == (n, width), f"track {tid} {name}"
        for name in ("keypoints", "smoothed"):
            assert _values(entry[name]).shape == (n, KEYPOINTS, 2), f"track {tid} {name}"
        assert _values(entry["behavior"]).shape == (n,), tid
        assert set(np.unique(_values(entry["behavior"]))) < set(range(len(dump.doc["behaviors"])))
        assert set(np.unique(_values(entry["pair_swapped"]))) <= {0.0, 1.0}
        assert set(np.unique(_values(entry["imputed"]))) <= {0.0, 1.0}
        assert np.isfinite(_values(entry["smoothed"])).all() and np.isfinite(_values(entry["conf"])).all()
        probs = _values(entry["behavior_probs"])
        assert np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1.0, atol=1e-4), tid
        # `null` is the spelling of an unobserved point, and only the raw stage output carries it
        nulls = {name for name, entry_value in entry.items() if None in entry_value["data"]}
        assert nulls <= {"boxes", "keypoints"}, f"track {tid}: {sorted(nulls)} carry NaN"
        gap = np.flatnonzero(~np.isfinite(_values(entry["boxes"])).all(-1))
        raw = _values(entry["keypoints"])
        assert list(gap) == list(np.flatnonzero(~np.isfinite(raw).all(axis=(-1, -2))))
        assert (_values(entry["imputed"])[gap] == 1.0).all()  # the filled rows are the missing detections


def test_the_classifier_inputs_are_the_track_arrays_the_rust_port_rebuilds(dump):
    """`behavior_input(got, times)` in Rust must land on the very arrays the classifier was handed."""
    inputs = dump.doc["behavior_inputs"]
    assert dump.doc["seed"] == dumper.SEED
    assert {e["track"] for e in inputs} == set(dump.doc["tracks"])  # every track was scored
    for entry in inputs:
        assert set(entry) == {"track", "frames", "coords", "mask", "coords_values", "mask_values"}
        track = dump.doc["tracks"][entry["track"]]
        n = entry["frames"]
        assert n == len(track["frames"]["data"]) > 0, entry["track"]
        assert _values(entry["coords_values"]).shape == (n, KEYPOINTS, 2)
        assert _values(entry["mask_values"]).shape == (n, KEYPOINTS)
        _hashed(entry["coords"])
        _hashed(entry["mask"])
        assert entry["coords"]["shape"] == [n, KEYPOINTS, 2] and entry["mask"]["shape"] == [n, KEYPOINTS]
        assert np.isfinite(_values(entry["coords_values"])).all()  # no NaN reaches the classifier
        assert set(np.unique(_values(entry["mask_values"]))) <= {0.0, 1.0}
        assert entry["coords"]["sha256"] == _sha_f32(_values(entry["coords_values"]))
        assert entry["mask"]["sha256"] == _sha_f32(_values(entry["mask_values"]))
        # the input is the smoothed track at its own frames and the confidence-threshold mask: exactly what
        # `pipeline.behavior_input` recomputes on the Rust side from the arrays above
        assert entry["coords_values"]["data"] == track["smoothed"]["data"], entry["track"]
        observed = np.nan_to_num(_values(track["smoothed_conf"])) >= dump.doc["pipeline"]["behavior_conf"]
        assert (_values(entry["mask_values"]) == observed).all(), entry["track"]


def test_the_csvs_reproduce_the_recorded_arrays_row_by_row(dump):
    """The CSVs are the printed contract: CRLF, fixed headers, and cells the arrays print exactly."""
    rows = _csv_rows(dump.fixture / "tracks.csv")
    assert rows[0] == ["frame", "time_s", "track", "keypoint", "x", "y", "confidence", "imputed", "behavior"]
    times = np.asarray(dump.doc["times"]["data"], float)
    at = 1
    for tid, entry in dump.doc["tracks"].items():
        pts = _values(entry["smoothed"]).reshape(-1, KEYPOINTS, 2)
        conf = _values(entry["smoothed_conf"]).reshape(-1, KEYPOINTS)
        imputed = _values(entry["imputed"]).reshape(-1, KEYPOINTS)
        behavior = _values(entry["behavior"]).astype(int)
        for i, frame in enumerate(_values(entry["frames"]).astype(int)):
            for k, name in enumerate(CANONICAL.names):
                assert rows[at] == [str(int(frame)), f"{times[int(frame)]:.4f}", tid, name,
                                    f"{pts[i, k, 0]:.2f}", f"{pts[i, k, 1]:.2f}", f"{conf[i, k]:.3f}",
                                    str(int(imputed[i, k])), dump.doc["behaviors"][int(behavior[i])]]
                at += 1
    assert at == len(rows)

    top = dump.doc["pipeline"]["behavior_top_k"]
    brows = _csv_rows(dump.fixture / "behavior.csv")
    assert brows[0] == ["frame", "time_s", "track"] + [f"top{i + 1}{s}" for i in range(top)
                                                       for s in ("", "_p")] \
        + [f"p_{name}" for name in dump.doc["behaviors"]]
    assert len(brows) - 1 == sum(len(e["frames"]["data"]) for e in dump.doc["tracks"].values())
    for row in brows[1:]:
        assert row[0].isdigit() and row[2] in dump.doc["tracks"] and float(row[1]) >= 0.0
        assert all(row[3 + 2 * j] in dump.doc["behaviors"] and 0.0 <= float(row[4 + 2 * j]) <= 1.0
                   for j in range(top))
        assert abs(sum(float(v) for v in row[3 + 2 * top:]) - 1.0) < 0.01


# ------------------------------------------------------------------------------------------------- config


def test_the_config_fixture_round_trips_through_the_python_loader(dump):
    """``config_parity.rs`` feeds every recorded string to its own parser: the recording must be current."""
    fixture = _load(dump.config / "overrides.json")
    base = dump.config / "base.yaml"
    assert fixture["base"] == base.read_bytes().decode("utf-8")
    assert [c["override"] for c in fixture["cases"]] == dumper.CONFIG_OVERRIDES
    for case in fixture["cases"]:
        override = case["override"]
        assert case["key"] == override.partition("=")[0]
        assert ("error" in case) != ("value" in case), override
        try:
            cfg = load_config(base, [override])
        except Exception as exc:  # noqa: BLE001
            assert case["error"] == type(exc).__name__, override
            continue
        node = cfg
        for part in case["key"].split("."):
            node = node.get(part) if isinstance(node, dict) else None
        assert dumper.value_entry(node) == case["value"], override
    assert [c["override"] for c in fixture["cases"] if "error" in c] == ["train.steps.deeper=1"]


def test_the_documented_yaml_1_1_spellings_are_the_only_flagged_cases(dump):
    """The Rust test insists every flagged case still differs, so the list must stay exactly true."""
    cases = {c["override"]: c for c in _load(dump.config / "overrides.json")["cases"]}
    flagged = {item: c for item, c in cases.items() if c.get("known_difference")}
    assert set(flagged) == set(dumper.KNOWN_DIFFERENCES)
    for item, case in flagged.items():
        assert case["note"] == dumper.KNOWN_DIFFERENCES[item]
    assert all(not c.get("note") for c in cases.values() if not c.get("known_difference"))
    value = {item: case["value"] for item, case in flagged.items()}
    assert value["train.log_every=yes"] == {"kind": "bool", "data": True}
    assert value["train.log_every=off"] == {"kind": "bool", "data": False}
    assert value["train.steps=1_000"] == {"kind": "int", "data": 1000}
    assert value["train.seed=0755"] == {"kind": "int", "data": 493}  # octal 0o755
    assert value["x.time=10:30"] == {"kind": "int", "data": 630}  # sexagesimal
    assert value["=1"] == {"kind": "int", "data": 1}
    # no shipped config depends on a spelling the two parsers disagree about
    for name in ("yes", "off", "1_000", "0755", "10:30"):
        for path in sorted((REPO / "configs").rglob("*.yaml")):
            text = path.read_bytes().decode("utf-8")
            assert re.search(rf"(?<![\w.:-]){re.escape(name)}(?![\w.])", text) is None, (path, name)


def test_the_extends_trees_are_written_next_to_their_bases(dump):
    """The Rust loader merges the same files from the same directory, so they must be byte-stable."""
    for name, text in dumper.EXTENDS_FILES.items():
        assert (dump.config / name).read_bytes().decode("utf-8") == text, name
    cases = {c["file"]: c for c in _load(dump.config / "overrides.json")["extends"]}
    assert set(cases) == set(dumper.EXTENDS_CASES)
    for name, overrides in sorted(dumper.EXTENDS_CASES.items()):
        assert cases[name]["overrides"] == overrides, name
        if "error" in cases[name]:
            assert cases[name]["error"] == "ConfigError", name
            with pytest.raises(ConfigError):
                load_config(dump.config / name, overrides)
        else:
            assert dumper.value_entry(load_config(dump.config / name, overrides)) == cases[name]["value"], name
    assert {c["file"] for c in cases.values() if "error" in c} == {
        "missing_base.yaml", "cycle_a.yaml", "extends_is_a_number.yaml"}
    child = cases["child.yaml"]["value"]["data"]
    assert "extends" not in child  # the loader removes the key, so no caller ever sees it
    assert child["train"]["data"]["steps"] == {"kind": "int", "data": 5}  # the child wins
    assert child["train"]["data"]["lr"] == {"kind": "float", "data": 0.5, "repr": "0.5"}  # the base is kept
    assert child["data"]["data"]["sources"]["data"][0]["data"]["source"] == {"kind": "str", "data": "synthetic"}
    merged = cases["two_parents.yaml"]["value"]["data"]
    assert merged["train"]["data"]["seed"] == {"kind": "int", "data": 7}  # the later base wins
    assert merged["train"]["data"]["lr"] == {"kind": "float", "data": 0.25, "repr": "0.25"}
    topped = cases["overridden.yaml"]["value"]["data"]  # the dotted overrides land on top of the base
    assert topped["train"]["data"]["steps"] == {"kind": "int", "data": 1}
    assert topped["train"]["data"]["lr"] == {"kind": "float", "data": 0.5, "repr": "0.5"}
    assert topped["data"]["data"]["species"]["data"] == [{"kind": "str", "data": "pig"}]


def test_value_entry_tags_every_value_the_yaml_loader_can_return():
    """``tagged`` in ``config_parity.rs`` has one arm per kind; anything else must be reported, not guessed."""
    assert dumper.value_entry(None) == {"kind": "null"}
    assert dumper.value_entry(True) == {"kind": "bool", "data": True}
    assert dumper.value_entry(3) == {"kind": "int", "data": 3}
    assert dumper.value_entry("mouse") == {"kind": "str", "data": "mouse"}
    assert dumper.value_entry(2.5) == {"kind": "float", "data": 2.5, "repr": "2.5"}
    assert dumper.value_entry([None, 1]) == {
        "kind": "list", "data": [{"kind": "null"}, {"kind": "int", "data": 1}]}
    assert dumper.value_entry({1: None}) == {"kind": "map", "data": {"1": {"kind": "null"}}}
    for number, spelling in ((float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")):
        entry = dumper.value_entry(number)  # strict JSON cannot spell it: the `repr` is the comparison
        assert entry == {"kind": "float", "data": None, "repr": spelling}
        assert json.dumps(entry)
    assert dumper.value_entry(yaml.safe_load("2026-11-03")) == {
        "kind": "unsupported", "data": repr(datetime.date(2026, 11, 3))}


# ----------------------------------------------------------------------------------------------- sessions


def test_the_session_slice_keeps_the_stored_dtypes_and_label_alignment(dump):
    """``sessions_parity.rs`` reads the ``.npz`` itself: members, dtypes, values and the resample stride."""
    stored = np.load(dump.sessions / "session.npz", allow_pickle=False)
    source = np.load(dump.source, allow_pickle=False)
    doc = _load(dump.sessions / "session.json")
    assert set(doc) == SESSION_KEYS
    assert doc["source"] == str(dump.source.relative_to(dump.repo))
    assert not Path(doc["source"]).is_absolute() and (dump.repo / doc["source"]).is_file()
    assert set(doc["dtypes"]) == set(stored.files) == set(source.files)
    assert doc["dtypes"] == {name: str(stored[name].dtype) for name in stored.files}
    assert doc["dtypes"]["labels"] == "int64" and doc["dtypes"]["fps"] == "float64"
    assert all(doc["dtypes"][n] == "float32" for n in ("coords", "conf", "boxes"))
    assert all(doc["dtypes"][n].startswith("<U") for n in ("animal", "view", "label_names"))
    for name in ("coords", "conf", "boxes", "labels"):
        assert stored[name].shape == (KEEP, *source[name].shape[1:]), name
        assert np.array_equal(stored[name], source[name][:KEEP]), name  # a slice, not a re-encode
    assert doc["frames"] == KEEP == stored["coords"].shape[0]
    assert _values(doc["coords"]).shape == (KEEP, KEYPOINTS, 2)
    assert _values(doc["conf"]).shape == (KEEP, KEYPOINTS)
    assert doc["coords"]["data"] and all(v is not None for v in doc["coords"]["data"])
    assert doc["labels"]["data"] == [int(v) for v in stored["labels"]]
    assert doc["fps"] == SESSION_FPS and float(stored["fps"]) == SESSION_FPS
    assert doc["animal"] == str(stored["animal"]) and doc["view"] == str(stored["view"])
    assert doc["label_names"] == [str(n) for n in stored["label_names"]] == list(BEHAVIORS_EXT)
    assert doc["classes"] == CLASSES and doc["ignored"] == IGNORE == -100
    names, classes = doc["label_names"], doc["classes"]
    remapped = [int(v) for v in remap_labels(stored["labels"], names, classes)]
    assert doc["remapped"] == {"shape": [KEEP], "data": remapped}
    assert remapped.count(IGNORE) > 0  # "eating" is in the stored vocabulary and not in the modelled one
    assert list(remap_labels(stored["labels"], names, doc["ext_classes"])) == list(stored["labels"])
    step = round(SESSION_FPS / doc["resampled_fps"])
    assert doc["resampled_fps"] == 15.0 and step == 2
    assert doc["resampled_frames"] == KEEP // step == stored["coords"][::step].shape[0]
    assert doc["resampled_labels"] == {"shape": [doc["resampled_frames"]], "data": remapped[::step]}


def test_a_dump_without_a_downloaded_recording_skips_the_session_fixture(monkeypatch, tmp_path, capsys):
    """CI has no ``data/``: the sessions fixture is then absent, and ``sessions_parity.rs`` skips by name."""
    monkeypatch.setattr(dumper, "REPO", tmp_path / "no-data")
    assert dumper.dump_sessions(tmp_path / "fixtures") is None
    assert "no behavior session .npz found" in capsys.readouterr().out
    assert not (tmp_path / "fixtures" / "sessions").exists()


# ---------------------------------------------------------------------------- determinism and other paths


def test_a_second_dump_from_the_same_seed_reproduces_every_byte(tmp_path_factory, dump):
    """One tolerance budget in the Rust tests: the reference has to be reproducible bit for bit."""
    repo, _ = _fake_repo(tmp_path_factory.mktemp("pipeline-repo-again"))
    _run_into(repo, "--out", "tests/fixtures", "--name", NAME, "--frames", str(FRAMES))
    again = repo / "tests" / "fixtures"
    files = sorted(p.relative_to(dump.root) for p in dump.root.rglob("*") if p.is_file())
    assert files and all((again / rel).is_file() for rel in files)
    differs = []
    for rel in files:
        if rel.suffix == ".safetensors":  # the wheel writes its metadata map in a free order
            first, meta_a = read_safetensors(dump.root / rel)
            second, meta_b = read_safetensors(again / rel)
            if meta_a != meta_b or not all(torch.equal(first[n], second[n]) for n in first):
                differs.append(rel)
        elif (dump.root / rel).read_bytes() != (again / rel).read_bytes():
            differs.append(rel)
    assert differs == []


def test_a_trained_checkpoint_is_copied_or_reexported_next_to_the_fixture(tmp_path):
    """``--detector``/``--pose``: a ``.pt`` is re-exported, a ``.safetensors`` is copied, ``--size`` sets the frame."""
    exported = tmp_path / "exported"
    exported.mkdir()
    det_cfg = {"backbone": dict(dumper.TINY_BACKBONE, width=6, out_channels=12), "num_species": 0}
    pose_cfg = dict(dumper.TINY_MODELS["pose"], embed_dim=12)
    det_pt = exported / "detector_trained.pt"
    save_checkpoint(det_pt, "detector", build_model("detector", det_cfg), det_cfg, {"input_size": [96, 96]})
    pose_st = exported / "pose_trained.safetensors"
    write_safetensors(pose_st, {k: v.contiguous() for k, v in build_model("pose", pose_cfg).state_dict().items()},
                      metadata_for("pose", pose_cfg, {"crop_size": 64}))
    out = tmp_path / "trained"
    dumper.run(_args("--out", str(out), "--name", "trained", "--frames", str(FRAMES), "--size", "48", "64",
                     "--detector", str(det_pt), "--pose", str(pose_st), "--skip", "config", "--skip", "sessions"))
    fixture, doc = out / "trained", _load(out / "trained" / "fixture.json")
    assert doc["size"] == [48, 64] and _png(image_files(fixture / "frames")[0]).shape == (48, 64)
    assert doc["models"]["detector"] == det_cfg and doc["models"]["pose"] == pose_cfg
    assert doc["models"]["temporal"] == dumper.TINY_MODELS["temporal"]  # the rest stay tiny random weights
    model, cfg = dumper.load_checkpoint(det_pt)
    assert cfg == det_cfg and all(torch.isfinite(p).all() for p in model.parameters())
    assert (fixture / "pose.safetensors").read_bytes() == pose_st.read_bytes()  # copied, not rewritten
    pt = torch.load(det_pt, map_location="cpu", weights_only=False)
    tensors, meta = read_safetensors(fixture / "detector.safetensors")
    assert meta["kind"] == "detector" and json.loads(meta["animalpose"]) == det_cfg
    assert json.loads(meta["extra"]) == {"input_size": doc["pipeline"]["det_size"]}
    assert set(tensors) == set(pt["state_dict"])
    assert all(torch.equal(tensors[n], pt["state_dict"][n]) for n in tensors)
    assert not (out / "config").exists() and not (out / "sessions").exists()


def test_the_cli_verb_dumps_an_absolute_out_and_honours_skip(tmp_path, capsys):
    """``animalpose fixtures pipeline --name alt --skip``: outside a checkout the paths stay absolute."""
    out = tmp_path / "fixtures"
    assert main(["fixtures", "pipeline", "--out", str(out), "--name", "alt", "--frames", "5",
                 "--skip", "config", "--skip", "sessions"]) == 0
    assert f"wrote {out / 'alt'}" in capsys.readouterr().out
    fixture = out / "alt"
    assert {p.name for p in fixture.iterdir()} == FIXTURE_FILES
    doc = _load(fixture / "fixture.json")
    assert doc["frames"] == 5 and len(doc["stages"]) == 5 == len(doc["times"]["data"])
    cfg = yaml.safe_load((fixture / "pipeline.yaml").read_bytes().decode("utf-8"))
    assert set(cfg["checkpoints"]) == set(KINDS)
    for kind, path in cfg["checkpoints"].items():
        assert Path(path).is_absolute() and Path(path).is_file(), kind
    assert not (out / "config").exists() and not (out / "sessions").exists()


def test_a_clip_shorter_than_the_temporal_minimum_writes_no_behavior_csv(tmp_path):
    """No track reaches five frames: the temporal and catalogue passes are skipped, and so is the CSV."""
    out = tmp_path / "short"
    dumper.run(_args("--out", str(out), "--name", "short", "--frames", "4", "--skip", "config",
                     "--skip", "sessions"))
    fixture, doc = out / "short", _load(out / "short" / "fixture.json")
    assert doc["frames"] == 4 and len(doc["stages"]) == 4 and doc["tracks"]
    assert "behavior.csv" not in {p.name for p in fixture.iterdir()}
    assert (fixture / "tracks.csv").is_file()
    assert list(doc["behaviors"]) == CLASSES  # the classifier's classes, never the 133-name catalogue
    assert "behavior_inputs" not in doc and "smoothed" not in next(iter(doc["tracks"].values()))
