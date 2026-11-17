"""The documented library surface, driven for real: five verbs over tiny checkpoints and rendered clips.

Every call is made the way the README shows it, on checkpoints trained for a handful of steps from
``TINY`` backbones and on clips rendered by ``animalpose.data.synthetic``. What is checked: that
``train`` returns the network and the history file it writes; that ``finetune`` really starts from the
``init=`` checkpoint it is given; that ``label`` reports the counts its COCO files contain and that the
frames it refuses are absent from them; that ``predict`` leaves behind the ``tracks.npz`` / ``tracks.csv``
/ ``overlay.mp4`` it advertises and returns the tracks it wrote (single recording, streamed file,
rendered clip, multi-camera rig) and honours every argument spelling (path or string, list or tuple
sizes, the default ``out``); that ``write_video`` writes every frame; that ``behavior`` reaches the three
cluster steps over kinematics that match the simulator's own motion; and the lazy exports of
``animalpose/__init__.py`` together with ``python -m animalpose``.
"""

import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

import animalpose
from animalpose import api
from animalpose.behavior import clusters as steps
from animalpose.behavior.catalog import NAMES, canonical
from animalpose.config import ConfigError
from animalpose.data.synthetic import make_cameras, render_video, simulate
from animalpose.selftrain import LabelOptions
from animalpose.skeleton import CANONICAL
from animalpose.train.checkpoint import load_model
from animalpose.viz import color

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
DATA = {"n_sims": 2, "frames": 24, "image_size": [160, 128]}
SEQUENCE = {"window": 32, "pool_sims": 2, "pool_frames": 120}
TRAIN_STEPS = {"detector": 8, "pose": 60, "temporal": 8, "behavior": 8}
KINDS = ("detector", "pose", "temporal", "behavior")
SIZE, FPS, N = (320, 256), 30.0, 60
CSV_HEADER = "frame,time_s,track,keypoint,x,y,confidence,imputed,behavior"


def mouse(seed: int = 7, n: int = N, view: str = "top"):
    """One black mouse walking on a bright floor seen from ``view``: a blob the tracker can follow."""
    sim = simulate(1, n, "mouse", seed=seed, coats=["black"])
    cam = make_cameras(sim.arena_cm, (view,), size=SIZE)[0]
    return sim, cam, render_video(sim, cam, seed=3)


def train_config(kind: str, out: Path, steps: int | None = None) -> dict:
    """A tiny training config for one network, of the shape the ``configs/*.yaml`` files have."""
    model = {"detector": {"backbone": TINY}, "pose": {"backbone": TINY, "embed_dim": 16},
             "temporal": {"dims": 2, "hidden": 16}, "behavior": {"dims": 2, "hidden": 16}}[kind]
    data = {"detector": {**DATA, "input_size": [128, 160]}, "pose": {**DATA, "crop_size": 64},
            "temporal": SEQUENCE, "behavior": SEQUENCE}[kind]
    return {"device": "cpu", "output": str(out / f"{kind}.pt"), "model": model, "data": data,
            "train": {"steps": TRAIN_STEPS[kind] if steps is None else steps, "batch_size": 2, "log_every": 0,
                      "workers": 0}}


def pipeline_config(root: Path, checkpoints: dict, name: str = "pipeline") -> Path:
    """A ``checkpoints`` / ``pipeline`` / ``tracker`` config at the sizes a tiny network needs."""
    cfg = {"device": "cpu", "checkpoints": {k: str(v) for k, v in checkpoints.items()},
           "pipeline": {"det_size": [128, 160], "crop_size": 64, "temporal_window": 32, "det_thresh": 0.05,
                        "detector_candidates": ["blob"], "embedder": "color"},
           "tracker": {"max_animals": 3}}
    path = root / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return path


def decoded(path: Path) -> list[np.ndarray]:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def write(path: Path, images, fps: float = FPS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert api.write_video(path, images, fps) == len(images)
    return path


def painted(frames, rgb, tol: int = 90) -> int:
    """Frames' pixels of one overlay palette colour: proof of which track was drawn on the clip."""
    flat = np.concatenate([f.reshape(-1, 3) for f in frames]).astype(int)
    return int((np.abs(flat - np.array(rgb)).sum(-1) < tol).sum())


def truth_tracks(path: Path, sim, cam, n: int = N):
    """A ``tracks.npz`` of the simulator's own keypoints for animal 0, and those keypoints."""
    frames = render_video(sim, cam, seed=3)[:n]
    kp = np.stack([f.keypoints[0] for f in frames]).astype(np.float32)
    present = np.asarray([f.present[0] for f in frames], bool)
    conf = np.repeat(present[:, None], CANONICAL.num_keypoints, 1).astype(np.float32) * 0.9
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, track0_smoothed=kp, track0_smoothed_conf=conf, track0_frames=np.arange(n),
                        times=np.arange(n) / FPS, keypoint_names=np.array(CANONICAL.names))
    return path, kp


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """The four tiny checkpoints, the recording, and the configs and tracks the tests below reuse."""
    root = tmp_path_factory.mktemp("api-surface")
    ck = {}
    for kind in KINDS:
        cfg = root / f"{kind}.yaml"
        cfg.write_text(yaml.safe_dump(train_config(kind, root)))
        ck[kind] = root / f"{kind}.pt"
        api.train(kind, cfg)
        assert ck[kind].is_file()
    sim, cam, frames = mouse()
    images = [f.image for f in frames]
    write(root / "cage.mp4", images)
    for view, i in (("top", 0), ("angled", 1)):  # the same mouse from two known cameras
        write(root / f"view{i}.mp4", [f.image for f in mouse(view=view)[2]])
    cameras = root / "cams.json"
    cameras.write_text(json.dumps([c.to_dict() for c in make_cameras(sim.arena_cm, ("top", "angled"), size=SIZE)]))
    heads = {k: ck[k] for k in ("detector", "pose")}
    paths, truth = truth_tracks(root / "truth" / "tracks.npz", sim, cam)
    return {"root": root, "ck": ck, "images": images, "video": root / "cage.mp4", "frames": len(images),
            "views": [root / "view0.mp4", root / "view1.mp4"], "cameras": cameras, "kp": frames[0].keypoints[0],
            "tracks": paths, "truth": truth,
            "pipe": pipeline_config(root, ck), "heads": pipeline_config(root / "heads", heads, "heads")}


# ----------------------------------------------------------------------------------------- train


def test_train_returns_the_network_and_the_history_it_writes(tmp_path, capsys):
    spec, cfg = train_config("detector", tmp_path, steps=2), tmp_path / "det.yaml"
    ckpt, log = Path(spec["output"]), Path(spec["output"]).with_suffix(".history.json")
    cfg.write_text(yaml.safe_dump(spec))
    model, history = api.train("detector", cfg, ["train.batch_size=1"])
    assert [h["step"] for h in history] == [1, 2] and np.isfinite([h["total"] for h in history]).all()
    assert next(model.parameters()).device.type == "cpu" and model.stride == 4
    assert f"saved {ckpt} and {log}" in capsys.readouterr().out
    assert json.loads(log.read_text()) == history
    saved = load_model(ckpt)
    assert all(torch.equal(a, b) for a, b in zip(saved.state_dict().values(), model.state_dict().values()))

    cfg.write_text(yaml.safe_dump({k: v for k, v in train_config("detector", tmp_path, 1).items() if k != "output"}))
    api.train("detector", str(cfg))  # a config without ``output`` trains, saves nothing and says nothing
    assert capsys.readouterr().out == ""
    assert sorted(p.name for p in tmp_path.iterdir()) == ["det.yaml", "detector.history.json", "detector.pt"]


def test_train_names_the_models_it_knows_and_rejects_a_typo(tmp_path):
    assert api.MODELS == ("behavior", "detector", "pose", "temporal")
    cfg = tmp_path / "pose.yaml"
    cfg.write_text(yaml.safe_dump(train_config("pose", tmp_path, 1)))
    with pytest.raises(ValueError, match="unknown model 'nope'; choose from behavior, detector, pose, temporal"):
        api.train("nope", cfg)
    with pytest.raises(ConfigError, match="unknown key 'trstin'"):
        api.train("pose", cfg, ["trstin.steps=1"])


# -------------------------------------------------------------------------------------- finetune


def test_finetune_starts_from_the_checkpoint_it_is_given(tmp_path, capsys):
    """A step at lr=0 cannot move a weight, so any weight still differing from the init was never loaded."""
    source_cfg = tmp_path / "init.yaml"
    source_cfg.write_text(yaml.safe_dump(train_config("pose", tmp_path, 2)))
    _, history = api.train("pose", source_cfg)
    assert len(history) == 2
    source = tmp_path / "pose.pt"
    init = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]

    target = tmp_path / "ft.yaml"
    target.write_text(yaml.safe_dump(train_config("pose", tmp_path, 1)))
    fine, _ = api.finetune("pose", target, source, ["train.lr=0", f"output={tmp_path / 'ft.pt'}"])
    started = dict(fine.named_parameters())
    assert set(started) <= set(init) and len(started) > 50
    assert all(torch.equal(started[k], init[k]) for k in started)
    assert f"initialised from {source} (missing 0, unexpected 0, reshaped/skipped [])" in capsys.readouterr().out

    scratch, _ = api.train("pose", target, [f"output={tmp_path / 'scratch.pt'}"])  # same seed, no init
    fresh = dict(scratch.named_parameters())
    assert any(not torch.equal(fresh[k], init[k]) for k in started)  # so the equality above is not vacuous

    def initialised_from(state: dict, name: str) -> str:
        path = tmp_path / f"{name}.pt"
        torch.save({"kind": "pose", "model_cfg": {"backbone": TINY, "embed_dim": 16}, "state_dict": state}, path)
        api.finetune("pose", target, path, ["train.lr=0", f"output={tmp_path / f'{name}-out.pt'}"])
        return capsys.readouterr().out

    dropped = dict(init)
    dropped.pop("heat_queries")
    assert "(missing 1, unexpected 0, reshaped/skipped [])" in initialised_from(dropped, "partial")
    narrowed = dict(init)
    narrowed["heat_bias"] = narrowed["heat_bias"][:1]  # the same network with a narrower head
    assert "reshaped/skipped ['heat_bias']" in initialised_from(narrowed, "narrow")


# ------------------------------------------------------------------------------- pipeline config


def test_build_pipeline_reads_the_three_sections_of_a_pipeline_config(lab):
    pipe = api.build_pipeline(yaml.safe_load(lab["pipe"].read_text()))
    assert pipe.cfg.det_size == (128, 160)  # a list in the YAML, a tuple in the dataclass
    assert pipe.cfg.detector_candidates == ("blob",) and pipe.cfg.embedder == "color"
    assert pipe.cfg.tracker.max_animals == 3 and pipe.cfg.temporal_window == 32
    assert pipe.temporal is not None and pipe.behavior is not None and pipe.device.type == "cpu"

    bare = api.build_pipeline(yaml.safe_load(lab["heads"].read_text()))
    assert bare.temporal is None and bare.behavior is None  # checkpoints the config does not name
    assert bare.cfg.behavior_top_k == 3 and bare.cfg.tracker.max_animals == 3  # keys it omits keep defaults


def test_write_video_writes_every_frame_it_is_given(tmp_path):
    images = [np.full((48, 64, 3), tone, np.uint8) for tone in (40, 120, 200)]
    assert api.write_video(tmp_path / "clip.mp4", images, 10.0) == 3
    back = decoded(tmp_path / "clip.mp4")
    assert len(back) == 3 and back[0].shape == (48, 64, 3)
    assert [float(f.mean()) for f in back] == pytest.approx([40.0, 120.0, 200.0], abs=4.0)
    assert api.write_video(tmp_path / "never.mp4", iter([]), 10.0) == 0
    assert not (tmp_path / "never.mp4").exists()  # no frames is no file, not an empty one


# ----------------------------------------------------------------------------------------- predict


def test_predict_writes_the_tracks_csv_and_overlay_of_one_recording(lab, tmp_path, capsys):
    """With only detector and pose configured there is no temporal model, so the overlay is drawn from
    the keypoints the pose network actually observed."""
    out = tmp_path / "run"
    res = api.predict(lab["video"], config=lab["heads"], out=out, max_frames=24)
    assert res.out == out and res.frames == 24 and res.overlay_frames == 24 and res.tracks == len(res.result.tracks)
    assert res.tracks >= 1 and res.behaviors == ()  # no classifier in the config: no catalogue to report
    assert not (out / "behavior.csv").exists() and not (out / "behavior_catalog.json").exists()

    rows = (out / "tracks.csv").read_text().splitlines()
    assert rows[0] == CSV_HEADER
    assert len(rows) == 1 + 25 * sum(len(t.frames) for t in res.result.tracks.values())
    assert {int(r.split(",")[2]) for r in rows[1:]} == set(res.result.tracks)
    assert [r.split(",")[3] for r in rows[1:26]] == list(CANONICAL.names)  # one row per keypoint, in order

    z = np.load(out / "tracks.npz")
    assert list(z["keypoint_names"]) == list(CANONICAL.names) and list(z["behaviors"]) == []
    track = res.result.tracks[min(res.result.tracks)]
    assert z["track0_keypoints"].shape == track.keypoints.shape
    assert np.array_equal(z["track0_keypoints"], track.keypoints, equal_nan=True)  # NaN gaps and all
    assert "track0_smoothed" not in z.files
    assert np.array_equal(z["times"], res.result.times) and len(z["times"]) == 24
    assert np.isclose(float(rows[26].split(",")[1]), int(rows[26].split(",")[0]) / FPS, atol=1e-3)

    overlay = decoded(out / "overlay.mp4")
    assert len(overlay) == 24 and overlay[0].shape == lab["images"][0].shape
    for tid in res.result.tracks:  # every track drawn in its palette colour, which the clip itself lacks
        assert painted(overlay, color(tid)) > 100 and painted(lab["images"], color(tid)) == 0
    timing = json.loads((out / "timing.json").read_text())
    assert timing["frames"] == 24 and timing["mean_fps"] == pytest.approx(FPS, abs=0.2) and not timing["variable"]
    assert capsys.readouterr().out == ""  # a steady frame rate has nothing to warn about


def test_predict_adds_the_catalogue_when_a_classifier_is_configured(lab, tmp_path):
    out = tmp_path / "full"
    res = api.predict(lab["video"], config=lab["pipe"], out=out, max_frames=24)
    assert res.behaviors == NAMES  # the classifier describes the whole catalogue
    catalog = json.loads((out / "behavior_catalog.json").read_text())
    assert [e["name"] for e in catalog] == list(NAMES)
    assert all(e["definition"] for e in catalog) and any(e["sources"] for e in catalog)
    assert any(e["modelled"] for e in catalog) and any(not e["modelled"] for e in catalog)
    header = (out / "behavior.csv").read_text().splitlines()[0]
    assert header.startswith("frame,time_s,track,top1,top1_p") and "p_rest" in header

    z = np.load(out / "tracks.npz")
    track = res.result.tracks[min(res.result.tracks)]
    assert np.array_equal(z["track0_smoothed"], track.smoothed) and z["track0_imputed"].shape == track.imputed.shape
    assert z["track0_behavior_full"].shape == (len(track.frames), len(NAMES))
    assert z["track0_behavior"].shape == (len(track.frames),)
    assert set(np.unique(z["track0_behavior"])) < set(range(len(NAMES)))  # a tiny classifier names a few
    rows = (out / "tracks.csv").read_text().splitlines()
    assert {r.split(",")[-1] for r in rows[1:]} <= set(NAMES) | {""}  # the winning behavior of every row
    overlay = decoded(out / "overlay.mp4")
    assert len(overlay) == 24 and sum(painted(overlay, color(tid)) for tid in res.result.tracks) == 0
    assert max(float(np.nanmax(t.smoothed_conf)) for t in res.result.tracks.values()) < 0.3
    # a track the temporal model is unsure of is not drawn, so this overlay stays the recording itself
    for got, src in zip(overlay, lab["images"]):  # within what mp4 alone does to a frame
        assert float(np.abs(got.astype(int) - src.astype(int)).mean()) < 8.0


def test_predict_streams_a_recording_named_by_string(lab, tmp_path):
    out = tmp_path / "stream"
    res = api.predict(str(lab["video"]), config=str(lab["pipe"]), out=str(out), overlay=False)
    assert isinstance(res.out, Path) and res.out == out
    assert res.frames == lab["frames"] and res.overlay_frames == 0 and res.result.n_frames == lab["frames"]
    assert not (out / "overlay.mp4").exists()
    assert (out / "tracks.npz").is_file() and (out / "tracks.csv").read_text().count("\n") > 25


def test_predict_takes_the_default_out_and_can_render_its_own_clip(lab, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    res = api.predict(config=lab["pipe"], synthetic=True, max_frames=12, fps=12.0)
    assert res.out == Path("outputs/run") and res.frames == 12 and res.overlay_frames == 12
    assert (tmp_path / "outputs/run/tracks.npz").is_file()
    assert not (tmp_path / "outputs/run/timing.json").exists()  # a rendered clip has no frame timestamps
    rows = (tmp_path / "outputs/run/tracks.csv").read_text().splitlines()
    assert rows[0] == CSV_HEADER
    first, second = rows[1].split(","), rows[26].split(",")
    assert int(first[0]) + 1 == int(second[0])  # the 25 rows later is exactly the next frame
    assert float(second[1]) - float(first[1]) == pytest.approx(1 / FPS, abs=1e-3)  # a rendered clip is 30 fps
    assert max(int(r.split(",")[0]) for r in rows[1:]) < 12  # max_frames really stopped it
    cap = cv2.VideoCapture(str(tmp_path / "outputs/run/overlay.mp4"))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0, abs=0.5)  # fps= is the frame rate of the clip
    cap.release()
    assert capsys.readouterr().out == ""


def test_predict_fuses_the_views_of_a_calibrated_rig(lab, tmp_path):
    out = tmp_path / "rig"
    res = api.predict(config=lab["pipe"], out=out, views=[str(v) for v in lab["views"]], cameras=lab["cameras"],
                      overrides=["tracker.max_animals=1", "pipeline.kp_conf_thresh=0.01"])
    assert res.views == 2 and len(res.per_view) == 2 and res.frames == 0 and res.tracks == 0
    assert res.animals == 1, "the mouse both cameras see has to become one identity"
    z = np.load(out / "animals3d.npz")
    assert sorted(z.files) == [f"animal0_{k}" for k in ("keypoints3d", "n_views", "reproj_error")]
    assert z["animal0_keypoints3d"].shape == (lab["frames"], CANONICAL.num_keypoints, 3)
    assert np.isfinite(z["animal0_keypoints3d"]).any() and int(z["animal0_n_views"].max()) == 2
    finite = np.isfinite(z["animal0_reproj_error"])
    assert finite.any() and (z["animal0_reproj_error"][finite] >= 0).all()
    for v in (0, 1):
        assert (out / f"view{v}_tracks.csv").read_text().splitlines()[0] == CSV_HEADER
    assert not (out / "timing.json").exists()
    # The first session saves each camera's reference; the next is measured against it.
    assert [e.status for e in res.drift] == ["reference saved"] * 2
    assert [d["status"] for d in json.loads((out / "camera_drift.json").read_text())] == ["reference saved"] * 2
    again = api.predict(config=lab["pipe"], out=tmp_path / "again", views=[str(v) for v in lab["views"]],
                        cameras=lab["cameras"], overrides=["tracker.max_animals=1", "drift.iterations=5"])
    assert all(e.status in ("corrected", "unchanged") and e.degrees < 0.05 for e in again.drift), again.drift
    assert len(json.loads((tmp_path / "again" / "cameras.json").read_text())) == 2


# ------------------------------------------------------------------------------------------ label


def test_label_keeps_only_the_frames_it_can_defend(lab, tmp_path):
    raw = tmp_path / "raw"
    write(raw / "caicai" / "cage.mp4", lab["images"])
    write(raw / "caicai" / "cage2.mp4", lab["images"][:35])
    teacher = dict(detector=lab["ck"]["detector"], pose=lab["ck"]["pose"], boxes="blob", kp_conf=0.0, flip_tol=1.0,
                   min_kps=1, progress=False, det_size=(128, 160), crop_size=64)
    stats = api.label(raw, out=tmp_path / "pseudo", per_source=8, holdout=False, **teacher)
    assert stats["sources"] == 2 and stats["frames"] == 16 and stats["labeled"] >= 1
    refused = stats["no_animal"] + stats["skipped_uncertain"] + stats["head_tail_flip"]
    assert stats["labeled"] <= stats["frames"] - refused  # every sampled frame is counted once, kept or refused

    out = tmp_path / "pseudo"
    coco = json.loads((out / "train.json").read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "animal")]
    assert coco["categories"][0]["keypoints"] == list(CANONICAL.names)
    assert stats["labeled"] == stats["images"] == len(coco["images"]) == len(list((out / "images").rglob("*.jpg")))
    assert all((out / "images" / im["file_name"]).is_file() for im in coco["images"])
    assert stats["animals"] == len(coco["annotations"]) == sum(1 for a in coco["annotations"] if a["track_id"] == -1)
    assert stats["keypoints"] == sum(int(v > 0) for a in coco["annotations"] for v in a["keypoints"][2::3])
    for ann in coco["annotations"]:
        seen = np.asarray(ann["keypoints"], float).reshape(-1, 3)[:, 2] > 0
        assert ann["num_keypoints"] == int(seen.sum()) > 0 and ann["category_id"] == 1 and not ann["iscrowd"]
        x, y, w, h = ann["bbox"]
        assert 0 <= x and 0 <= y and w > 0 and h > 0
    split = json.loads((out / "split.json").read_text())
    assert split["train"] == [str(Path("caicai") / name) for name in ("cage.mp4", "cage2.mp4")]
    assert split["held_out"] == []

    kept = api.label(raw, out=tmp_path / "filtered", per_source=4, include=["cage2"], **teacher)
    assert kept["sources"] == 1 and kept["frames"] == 4
    assert json.loads((tmp_path / "filtered/split.json").read_text())["train"] == [str(Path("caicai") / "cage2.mp4")]

    options = LabelOptions(out=str(tmp_path / "given"), detector=str(lab["ck"]["detector"]),
                           pose=str(lab["ck"]["pose"]), boxes="blob", kp_conf=0.0, flip_tol=1.0, min_kps=1,
                           progress=False, det_size=(128, 160), crop_size=64)
    both = api.label(raw, options=options, per_source=3)  # one options object, one field overridden
    assert both["sources"] == 2 and both["frames"] == 6 and (tmp_path / "given/train.json").is_file()
    assert {p.name for p in (tmp_path / "given/images").iterdir()} == {p.name for p in (out / "images").iterdir()}


# --------------------------------------------------------------------------------------- behavior


def test_behavior_dispatches_to_the_three_cluster_steps(lab, tmp_path):
    clusters, labels = tmp_path / "clusters", tmp_path / "labels"
    summary = api.behavior("cluster", tracks=lab["tracks"], video=lab["video"], out=clusters, fps=FPS, k=2)
    assert sorted(summary) == [0, 1] and sum(s["share_of_train"] for s in summary.values()) == pytest.approx(1.0)
    assert all(set(s) >= {"share_of_train", "speed", "length", "turn", "posture", "head_motion", "bend"}
               for s in summary.values())
    assert (clusters / "clusters.npz").is_file() and (clusters / "summary.json").read_text() == json.dumps(summary, indent=1)

    kp = lab["truth"]  # what the simulator actually did, to check the features against
    span = kp[:, CANONICAL.index("nose")] - kp[:, CANONICAL.index("tail_base")]
    body = float(np.median(np.linalg.norm(span, axis=-1)))  # the nose-to-tail length the features normalise by
    walked = np.linalg.norm(np.diff(kp[:, CANONICAL.index("spine_mid")], axis=0), axis=-1) * FPS / body
    fast, slow = max(summary, key=lambda c: summary[c]["speed"]), min(summary, key=lambda c: summary[c]["speed"])
    assert summary[fast]["speed"] == pytest.approx(float(np.median(walked)), rel=0.1)  # body lengths per second
    assert min(s["speed"] for s in summary.values()) >= float(walked.min()) - 0.05
    assert max(s["speed"] for s in summary.values()) <= float(walked.max()) + 0.05
    weighted = sum(s["share_of_train"] * s["speed"] for s in summary.values())
    assert weighted == pytest.approx(float(np.median(walked)), rel=0.2)
    assert summary[fast]["speed"] > summary[slow]["speed"] > 0.4 and summary[fast]["length"] > 0.9
    assert summary[fast]["bend"] < 1.0 and summary[fast]["head_motion"] < 0.8  # straight, unstirred: not grooming

    report = api.behavior("name", clusters=clusters)
    naming = report["naming"]
    assert naming == {str(c): v for c, v in steps.name_clusters(summary).items()}  # keys survive JSON
    assert all(canonical(v["name"]) in NAMES for v in naming.values())  # every rule name is a catalogue class
    assert {v["name"] for v in naming.values()} == {"walking"}  # the mouse walks the whole clip
    assert json.loads((clusters / "naming.json").read_text())["naming"] == naming
    assert naming[str(fast)]["features"]["speed"] == pytest.approx(summary[fast]["speed"])

    share = api.behavior("label", clusters=clusters, mapping={fast: "running", slow: "resting"}, out=labels,
                        name="cage", view="top", min_bout=0.5)
    assert set(share) == {"running", "resting"} and sum(share.values()) == pytest.approx(1.0)
    assert share["running"] == pytest.approx(0.73, abs=0.15)  # the fast cluster is the majority
    held = np.load(labels / "heldout" / "cage_heldout.npz")
    assert set(str(n) for n in held["label_names"][held["labels"]]) == {"running"}  # it was walking at the end
    train = np.load(labels / "train" / "cage_train.npz")
    assert sorted(set(str(n) for n in train["label_names"][train["labels"]].tolist())) == ["resting", "running"]
    assert len(held["labels"]) == 12 and len(train["labels"]) == 48  # 80% train / 20% held out of 60 frames
    assert float(held["fps"]) == FPS and str(held["view"]) == "top" and str(held["animal"]) == "cage_heldout"

    with pytest.raises(ValueError, match="behavior task must be one of cluster, name, label"):
        api.behavior("plot", tracks=lab["tracks"])
    with pytest.raises(ValueError, match="not in the behavior catalogue"):
        api.behavior("label", clusters=clusters, mapping={fast: "teleporting"}, out=tmp_path / "bad")
    assert api.cluster_tracks is steps.cluster_tracks and api.write_naming is steps.write_naming
    assert api.label_clusters is steps.label_clusters and api.behavior.__name__ == "behavior"


# ------------------------------------------------------------------------------- the package itself


def test_the_package_reexports_the_documented_verbs():
    assert animalpose.__version__ == "0.1.0"
    assert all(hasattr(animalpose, name) for name in animalpose.__all__)
    assert callable(animalpose.Skeleton) and callable(animalpose.select_device)
    assert animalpose.CANONICAL is CANONICAL and animalpose.CANONICAL.num_keypoints == 25
    for verb in ("predict", "finetune", "label", "cluster_tracks", "write_naming", "label_clusters"):
        assert getattr(animalpose, verb) is getattr(api, verb), verb  # the verb, reached through the lazy hook
    assert animalpose.__getattr__("train") is api.train  # the verb itself only answers that lookup ...
    assert animalpose.train.__name__ == "animalpose.train"  # ... because once imported, the subpackage owns it
    assert api.behavior.__name__ == "behavior" and not callable(animalpose.behavior)  # the subpackage, again
    with pytest.raises(AttributeError, match="module 'animalpose' has no attribute 'frobnicate'"):
        animalpose.__getattr__("frobnicate")


def test_python_m_animalpose_is_the_command_line(tmp_path, monkeypatch, capsys):
    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["animalpose", *argv])
        buffer = io.StringIO()
        with pytest.raises(SystemExit) as stop, contextlib.redirect_stdout(buffer):
            runpy.run_module("animalpose.__main__", run_name="__main__")
        return stop.value.code, buffer.getvalue()

    code, listing = run("--help")
    assert code == 0 and "train / finetune" in listing
    assert "predict           detect, track and classify a video" in listing
    code, version = run("--version")
    assert code == 0 and version.strip() == f"animalpose {animalpose.__version__}"
    monkeypatch.chdir(tmp_path)
    code, _ = run("predict", "--video", "none.mp4", "--config", str(tmp_path / "missing.yaml"))
    assert code == 1  # a config that is not there is reported, not raised
    assert "error: config not found" in capsys.readouterr().err
    assert not (tmp_path / "outputs").exists()  # it stopped before making the output folder
    code, _ = run("no-such-verb")
    assert code == 2  # argparse rejected the verb
