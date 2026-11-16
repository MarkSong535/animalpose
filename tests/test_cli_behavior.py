"""The rest of the command line, run through ``main(argv)`` on inputs built in this file.

Checked here, against recordings and datasets written into ``tmp_path``: ``behavior cluster`` splitting a
moving stretch from a still one, ``behavior name`` printing the rule names and the map line, ``behavior
label`` turning exactly that map into train/held-out session files, ``behavior plot`` writing an ethogram
whose shares are the frames' own top-1 choices, and ``behavior sessions`` tracking fake MIT / ABODe /
BORIS-labelled recordings with shard filters that decide which videos are done. Then ``predict`` (what it prints,
what it writes, and the input it refuses), ``data convert`` on a DeepLabCut CSV, ``data prepare`` on a fake
download tree (canonical COCO out, the iRodent image root found from the annotation), ``data calibrate``
solving a rig from projected corners, and the error paths ``main`` turns into exit code 1.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from animalpose.api import write_video
from animalpose.behavior.catalog import NAMES, describe
from animalpose.cli import build_parser, main
from animalpose.data.behavior_sets import ABODE_LABELS, MIT_LABELS
from animalpose.data.synthetic import make_cameras, render_video, simulate
from animalpose.data.tracks import IGNORE
from animalpose.labels import BEHAVIORS_EXT
from animalpose.multiview.camera import Camera
from animalpose.skeleton import CANONICAL
from animalpose.train.loops import train_detector, train_pose

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
FPS, T, MOVE = 30.0, 120, 60
OFFSETS = {"nose": (20, 0), "head_center": (15, 0), "neck": (11, 0), "spine_upper": (7, 0), "spine_mid": (0, 0),
           "spine_lower": (-7, 0), "tail_base": (-20, 0), "tail_mid": (-28, 0), "tail_end": (-35, 0),
           "left_eye": (17, 2), "right_eye": (17, -2), "left_ear": (13, 3), "right_ear": (13, -3),
           "left_shoulder": (7, 4), "right_shoulder": (7, -4), "left_front_knee": (11, 6),
           "right_front_knee": (11, -6), "left_front_paw": (14, 7), "right_front_paw": (14, -7),
           "left_hip": (-7, 4), "right_hip": (-7, -4), "left_hind_knee": (-11, 6),
           "right_hind_knee": (-11, -6), "left_hind_paw": (-14, 7), "right_hind_paw": (-14, -7)}


def body(centre, curl: float = 0.0) -> np.ndarray:
    """A 40 px animal pointing +x, its rear half curled by ``curl`` radians when given."""
    kp = np.zeros((CANONICAL.num_keypoints, 2), np.float32)
    for name, (a, b) in OFFSETS.items():
        kp[CANONICAL.index(name)] = np.array(centre, np.float32) + np.array([a, b], np.float32)
    if curl:
        rot = np.array([[np.cos(curl), -np.sin(curl)], [np.sin(curl), np.cos(curl)]])
        for name in ("spine_lower", "tail_base", "tail_mid", "tail_end"):
            k = CANONICAL.index(name)
            kp[k] = kp[CANONICAL.index("spine_mid")] + (kp[k] - kp[CANONICAL.index("spine_mid")]) @ rot.T
    return kp


def movie(path: Path, centres, size=(320, 240)) -> Path:
    """A dark disc following ``centres``: an MJPG avi, so the cluster montages can seek it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
    for c in centres:
        frame = np.full((size[1], size[0], 3), 150, np.uint8)
        cv2.circle(frame, tuple(np.clip(np.round(c), [2, 2], [size[0] - 3, size[1] - 3]).astype(int)), 12, (25, 25, 25), -1)
        vw.write(frame)
    vw.release()
    return path


def recording(root: Path) -> dict:
    """What a ``predict`` run leaves in its folder, for a mouse that runs for 2 s and then curls up.

    The straight stretch covers 4 px a frame, which is 3.5 body lengths/s once the kinematics normalize by
    the median nose-to-tail length; the curled one measures 0.83 of that length at speed 0.
    """
    kp = np.stack([body((40.0 + 4.0 * min(t, MOVE - 1), 120.0), 0.0 if t < MOVE else np.radians(90))
                   for t in range(T)])
    conf = np.full((T, CANONICAL.num_keypoints), 0.9, np.float32)
    times = np.arange(T) / FPS
    modelled = np.zeros(len(NAMES), bool)
    modelled[[NAMES.index(n) for n in ("rest", "walk", "self_groom")]] = True
    full = np.zeros((T, len(NAMES)))
    full[:, NAMES.index("rest")], full[:, NAMES.index("walk")] = np.where(times > MOVE / FPS, 0.8, 0.1), \
        np.where(times > MOVE / FPS, 0.1, 0.8)
    full[:, NAMES.index("self_groom")] = 0.1
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "tracks.npz", track0_smoothed=kp, track0_smoothed_conf=conf,
                        track0_frames=np.arange(T), times=times, track0_behavior_full=full,
                        keypoint_names=np.array(CANONICAL.names), behaviors=np.array(NAMES))
    (run / "behavior_catalog.json").write_text(json.dumps(describe(modelled)))
    video = movie(root / "synth.avi", kp[:, CANONICAL.index("spine_mid")])
    return {"run": run, "tracks": run / "tracks.npz", "video": video, "full": full}


def video_of(path: Path, n: int, size_wh=(160, 128), fps: float = 30.0) -> Path:
    """A moving light bar on a dark floor: enough for a tracker and for the sessions readers.

    Encoded as mp4 and renamed, because the MIT videos are named ``.mpg`` and only ffmpeg picks that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size_wh
    written = path.with_suffix(".mp4")
    vw = cv2.VideoWriter(str(written), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        x = (5 * t) % max(1, w - 16)
        img[h // 3:2 * h // 3, x:x + 16] = 220
        vw.write(img)
    vw.release()
    if written != path:
        written.replace(path)
    return path


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """Tiny detector and pose checkpoints, and the pipeline config every command below is given."""
    out = tmp_path_factory.mktemp("cli-runs")
    data = {"n_sims": 2, "frames": 24, "image_size": [160, 128]}
    train = {"steps": 8, "batch_size": 2, "log_every": 0, "workers": 0}
    train_detector({"device": "cpu", "output": str(out / "det.pt"), "model": {"backbone": TINY},
                    "data": {**data, "input_size": [128, 160]}, "train": train})
    train_pose({"device": "cpu", "output": str(out / "pose.pt"), "model": {"backbone": TINY, "embed_dim": 16},
                "data": {**data, "crop_size": 64}, "train": train})
    cfg = {"device": "cpu", "checkpoints": {"detector": str(out / "det.pt"), "pose": str(out / "pose.pt")},
           "pipeline": {"det_size": [128, 160], "crop_size": 64, "det_thresh": 0.05,
                        "detector_candidates": ["blob"], "embedder": "color"}, "tracker": {"max_animals": 3}}
    (out / "pipeline.yaml").write_text(yaml.safe_dump(cfg))
    sim = simulate(1, 40, "mouse", seed=7, coats=["black"])
    frames = [f.image for f in render_video(sim, make_cameras(sim.arena_cm, ("top",), size=(320, 256))[0], seed=3)]
    clip = out / "cage.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 256))
    for f in frames:
        vw.write(f)
    vw.release()
    return {"det": out / "det.pt", "pose": out / "pose.pt", "pipe": out / "pipeline.yaml", "video": clip}


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """The one fake run the cluster / name / label / plot commands are pointed at."""
    return recording(tmp_path_factory.mktemp("cli-track"))


# ----------------------------------------------------------------------------- behavior cluster / name


def parse_name_report(text: str) -> tuple[dict, dict]:
    """The ``behavior name`` table: per-cluster names and the ``map:`` pairs it prints last."""
    named, mapping = {}, {}
    for line in text.splitlines():
        if line.startswith("cluster "):
            _, cluster, _, name, *_ = line.split()
            named[cluster] = name
        elif line.startswith("map:"):
            mapping = dict(p.split("=") for p in line.removeprefix("map:").split())
    return named, mapping


def test_behavior_cluster_separates_the_moving_stretch_from_the_still_one(clip, tmp_path, capsys):
    out = tmp_path / "clusters"
    argv = ["behavior", "cluster", "--tracks", str(clip["tracks"]), "--video", str(clip["video"]), "--fps", "30",
            "--train-range", "0", "0.8", "--k", "2", "--out", str(out)]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert sorted(summary) == ["0", "1"] and sum(s["share_of_train"] for s in summary.values()) == pytest.approx(1.0)
    assert (out / "summary.json").read_text() == json.dumps(summary, indent=1)
    assert (out / "clusters.npz").is_file() and all((out / f"cluster{c}.png").is_file() for c in summary)

    fast = max(summary, key=lambda c: summary[c]["speed"])
    slow = str(1 - int(fast))
    assert summary[fast]["speed"] > 3.0 and summary[fast]["length"] > 0.9 and summary[fast]["bend"] < 0.1
    assert summary[slow]["speed"] < 0.35 and summary[slow]["length"] < 0.95 and summary[slow]["bend"] > 0.5
    labels = np.load(out / "clusters.npz")["labels"]
    assert set(labels[:30]) == {int(fast)} and set(labels[-30:]) == {int(slow)}  # moving, then still
    assert float(np.load(out / "clusters.npz")["fps"]) == FPS


def test_behavior_name_prints_the_rules_and_the_map_that_round_trips(clip, tmp_path, capsys):
    clusters = tmp_path / "clusters"
    assert main(["behavior", "cluster", "--tracks", str(clip["tracks"]), "--video", str(clip["video"]), "--k", "2",
                 "--out", str(clusters)]) == 0
    summary = json.loads(capsys.readouterr().out)
    fast = max(summary, key=lambda c: summary[c]["speed"])

    assert main(["behavior", "name", "--clusters", str(clusters)]) == 0
    printed = capsys.readouterr().out
    named, mapping = parse_name_report(printed)
    assert named == {str(fast): "running", str(1 - int(fast)): "resting"}  # fast+stretched, still+curled
    assert mapping == named  # the "map:" line is the same naming, ready for ``behavior label``
    assert f"cluster {fast} -> running" in printed and "speed=" in printed and "posture=" in printed
    assert "rule naming vs hand naming" not in printed  # no --agree-with given
    report = json.loads((clusters / "naming.json").read_text())
    assert set(report) == {"naming", "rules"}
    assert all(set(v) == {"name", "share_of_train", "features"} and set(v["features"]) == {"speed", "length", "turn",
               "posture", "head_motion", "bend", "speed_sd", "length_sd", "posture_sd"}
               for v in report["naming"].values())
    assert sorted(report["rules"]) == ["grooming", "other", "rearing", "resting", "running", "sniffing", "turning",
                                       "walking"]
    assert report["rules"]["running"] == [["speed", ">", 3.0], ["length", ">", 0.9], ["bend", "<", 1.0]]


def test_behavior_label_writes_sessions_from_the_printed_map(clip, tmp_path, capsys):
    clusters = tmp_path / "clusters"
    main(["behavior", "cluster", "--tracks", str(clip["tracks"]), "--video", str(clip["video"]), "--k", "2",
          "--out", str(clusters)])
    main(["behavior", "name", "--clusters", str(clusters)])
    mapping = parse_name_report(capsys.readouterr().out)[1]
    pairs = [f"{c}={name}" for c, name in sorted(mapping.items())]

    out = tmp_path / "sessions"
    assert main(["behavior", "label", "--clusters", str(clusters), "--map", *pairs, "--out", str(out),
                 "--name", "synth", "--view", "top", "--min-bout", "0.5"]) == 0
    share = json.loads(capsys.readouterr().out)
    lab = np.load(clusters / "clusters.npz")["labels"]
    ran = int(next(c for c, n in mapping.items() if n == "running"))
    assert share == {"running": pytest.approx(float((lab == ran).mean()), abs=1e-3),
                     "resting": pytest.approx(float((lab != ran).mean()), abs=1e-3)}
    assert 0.4 < share["running"] < 0.6  # it ran for about half of the clip
    written = json.loads((out / "labels.json").read_text())
    assert written["mapping"] == {str(k): v for k, v in mapping.items()} and written["min_bout_s"] == 0.5
    assert written["share"] == share

    train = np.load(out / "train" / "synth_train.npz")
    names = np.array([str(n) for n in train["label_names"]])
    assert str(train["view"]) == "top" and float(train["fps"]) == FPS
    assert sorted(map(str, names)) == ["resting", "running"]
    stopped = int(np.argmax(lab != lab[0]))  # the frame the clusters place the stop
    assert 45 < stopped < 75
    assert (names[train["labels"]][:stopped - 2] == "running").all()
    assert (names[train["labels"]][stopped + 2:] == "resting").all()
    held = np.load(out / "heldout" / "synth_heldout.npz")
    assert set(str(n) for n in held["label_names"][held["labels"]]) == {"resting"}

    # the hand naming written by the step above is what --agree-with scores the rules against
    assert main(["behavior", "name", "--clusters", str(clusters), "--agree-with", str(out / "labels.json")]) == 0
    scored = json.loads(re.search(r"rule naming vs hand naming: (.*)", capsys.readouterr().out).group(1))
    assert scored["n_named_clusters"] == 2 and scored["weighted_agreement"] == 1.0 and scored["mismatches"] == []

    assert main(["behavior", "label", "--clusters", str(clusters), "--map", "0=sidewalking", "--out",
                 str(tmp_path / "bad")]) == 1  # not a name in the catalogue


# ------------------------------------------------------------------------------------ behavior plot


def test_behavior_plot_reports_the_time_share_of_each_behavior(clip, tmp_path, capsys):
    run = tmp_path / "run"
    run.mkdir()
    (run / "tracks.npz").write_bytes((clip["run"] / "tracks.npz").read_bytes())
    (run / "behavior_catalog.json").write_bytes((clip["run"] / "behavior_catalog.json").read_bytes())
    assert main(["behavior", "plot", "--run", str(run), "--video", str(clip["video"]), "--clip-start", "0.5",
                 "--clip-seconds", "0.5", "--top", "2", "--heldout-from", "2.0"]) == 0
    share = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    top = clip["full"].argmax(1)
    assert (top[:MOVE] == NAMES.index("walk")).all() and (top[MOVE + 1:] == NAMES.index("rest")).all()
    expected = {n: float((top == NAMES.index(n)).mean()) for n in ("walk", "rest", "self_groom")}
    assert share == {n: pytest.approx(v, abs=1e-3) for n, v in expected.items()}
    assert sum(share.values()) == pytest.approx(1.0)
    ethogram = cv2.imread(str(run / "ethogram.png"))
    assert ethogram is not None and ethogram.shape[0] > 500 and (ethogram < 250).any()
    cap = cv2.VideoCapture(str(run / "behavior_clip.mp4"))
    ok, first = cap.read()
    fps_read = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert ok and first.shape == (360, 660, 3) and fps_read == pytest.approx(FPS, abs=0.5)


# ------------------------------------------------------------------------------ behavior sessions


def test_behavior_sessions_tracks_and_labels_fake_mit_videos(models, tmp_path, capsys):
    root = tmp_path / "mit"
    for stem in ("S1", "S2", "S3"):
        video_of(root / "video" / f"{stem}.mpg", 12)
    (root / "Annotator_group_1").mkdir(parents=True)
    for stem in ("S1", "S2", "S3"):  # "drink" is not a class of ours, so it becomes "other"
        (root / "Annotator_group_1" / f"{stem}.txt").write_text(
            "frame: 00000001-00000003 drink\nframe: 00000004-00000006 rest\nframe: 00000007-00000012 walk\n")
    out = tmp_path / "sets"
    argv = ["behavior", "sessions", "mit", "--root", str(root), "--out", str(out), "--pose", str(models["pose"]),
            "--detector", str(models["det"]), "--crop-size", "64", "--stride", "2", "--shard", "0", "--nshards", "2"]
    assert main(argv) == 0
    assert capsys.readouterr().out.strip() == f"mit shard 0/2: 2 done -> {out}"
    done = sorted(p.stem for p in (out / "mit").glob("*.npz"))
    assert done == ["S1", "S3"]  # shard 0 of 2 takes every other video of the three
    s1 = np.load(out / "mit" / "S1.npz")
    other, rest, walk = (BEHAVIORS_EXT.index(n) for n in ("other", "resting", "walking"))
    assert [int(v) for v in s1["labels"]] == [other, other, rest, walk, walk, walk]  # every other frame
    assert [str(n) for n in s1["label_names"][[other, rest, walk]]] == ["other", "resting", "walking"]
    assert float(s1["fps"]) == 15.0 and str(s1["view"]) == "side" and str(s1["animal"]) == "S1"
    assert s1["coords"].shape == (6, CANONICAL.num_keypoints, 2) and s1["boxes"].shape == (6, 4)
    assert MIT_LABELS["rest"] == "resting" and MIT_LABELS["drink"] == "other"


def test_behavior_sessions_uses_the_abode_boxes_and_per_second_labels(models, tmp_path, capsys):
    root = tmp_path / "abode"
    mice = ["mouseA", "mouseB"]
    per_snippet = ["Imm"] + ["Feed"] * 5  # one label per second of the snippet
    ann = pd.DataFrame({(m, "GT.Behaviour"): per_snippet * 2 for m in mice},
                       index=pd.MultiIndex.from_product([["12"], ["3"], ["1", "2"], range(6)],
                                                        names=["Cage", "Segment", "Snippet", "BTI"]))
    root.mkdir(parents=True)
    ann.to_pickle(root / "Annotations.df", compression="bz2")
    xywh = np.tile(np.r_[10.0, 20.0, 30.0, 40.0], (12, 1))
    for snip in ("1", "2"):
        video_of(root / "Snippets" / f"12_3_{snip}" / "Video.mp4", 12, fps=25.0)
        ident = pd.DataFrame(np.concatenate([xywh, xywh + np.r_[40.0, 0.0, 0.0, 0.0]], 1),
                             columns=pd.MultiIndex.from_tuples([(m, f) for m in mice for f in "xywh"]))
        ident.to_pickle(root / "Snippets" / f"12_3_{snip}" / "Identities.df", compression="bz2")
    out = tmp_path / "sets"
    argv = ["behavior", "sessions", "abode", "--root", str(root), "--out", str(out), "--pose", str(models["pose"]),
            "--detector", str(models["det"]), "--crop-size", "64", "--stride", "2", "--shard", "1", "--nshards", "2"]
    assert main(argv) == 0
    assert capsys.readouterr().out.strip() == f"abode shard 1/2: 1 done -> {out}"
    files = sorted(p.name for p in (out / "abode").glob("*.npz"))
    assert files == ["12_3_2_mouseA.npz", "12_3_2_mouseB.npz"]  # only the second snippet, both mice

    a = np.load(out / "abode" / "12_3_2_mouseA.npz")
    assert str(a["animal"]) == "cage12" and float(a["fps"]) == 12.5 and str(a["view"]) == "side"
    assert [int(v) for v in a["labels"]] == [BEHAVIORS_EXT.index("resting")] * 6  # the first second was "Imm"
    assert np.allclose(a["boxes"], np.tile([10, 20, 40, 60], (6, 1)))  # the identity box, xywh -> xyxy
    b = np.load(out / "abode" / "12_3_2_mouseB.npz")
    assert np.allclose(b["boxes"], np.tile([50, 20, 80, 60], (6, 1)))  # the second mouse, shifted by +40
    assert ABODE_LABELS["Imm"] == "resting" and ABODE_LABELS["Feed"] == "eating"


def test_behavior_sessions_shards_a_boris_videos_list(models, tmp_path, capsys):
    root = tmp_path / "boris"
    names = ("cage-01.mp4", "cage-02.mp4", "other-03.mp4")
    for name in names:
        video_of(root / name, 10)
    listing = root / "videos.txt"
    listing.write_text("# one video per line, and a comment line that is not\n" +
                       "\n".join(str(root / n) for n in names))
    out = tmp_path / "sets"
    argv = ["behavior", "sessions", "boris", "--videos-list", str(listing), "--out", str(out),
            "--pose", str(models["pose"]), "--detector", str(models["det"]), "--crop-size", "64", "--shard", "1",
            "--nshards", "2"]
    assert main(argv) == 0
    assert capsys.readouterr().out.strip() == f"boris shard 1/2: 1 done -> {out}"
    assert [p.name for p in (out / "boris").glob("*.npz")] == ["cage-02.npz"]  # index 1 of the sorted list
    z = np.load(out / "boris" / "cage-02.npz")
    assert str(z["animal"]) == "cage" and str(z["view"]) == "cage" and float(z["fps"]) == 30.0
    assert (z["labels"] == IGNORE).all()  # no BORIS project was offered, so nothing is labelled
    assert z["coords"].shape == (10, CANONICAL.num_keypoints, 2)  # every frame, stride 1


# ------------------------------------------------------------------------------------------- predict


def test_predict_prints_what_it_wrote_and_refuses_no_input(models, tmp_path, capsys):
    out = tmp_path / "pred"
    assert main(["predict", "--video", str(models["video"]), "--config", str(models["pipe"]), "--out", str(out),
                 "--max-frames", "24"]) == 0
    printed = capsys.readouterr().out.strip()
    frames, tracks, where = re.fullmatch(r"(\d+) frames, (\d+) tracks -> (.*)", printed.splitlines()[-1]).groups()
    assert (int(frames), where) == (24, str(out)) and Path(where).is_dir()
    rows = (out / "tracks.csv").read_text().splitlines()
    body = rows[1:]
    assert rows[0] == "frame,time_s,track,keypoint,x,y,confidence,imputed,behavior"
    assert int(tracks) == len({r.split(",")[2] for r in body}) >= 1
    assert len(body) % CANONICAL.num_keypoints == 0  # one row per (frame, track, keypoint)
    assert max(int(r.split(",")[0]) for r in body) < 24  # --max-frames was honoured
    first_frame, first_time = int(body[0].split(",")[0]), float(body[0].split(",")[1])
    assert first_time == pytest.approx(first_frame / FPS, abs=1e-6)  # times come from the video
    assert (out / "overlay.mp4").is_file() and json.loads((out / "timing.json").read_text())["frames"] == 24

    quiet = tmp_path / "no-overlay"
    assert main(["predict", "--video", str(models["video"]), "--config", str(models["pipe"]), "--out", str(quiet),
                 "--max-frames", "8", "--no-overlay"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("8 frames,")
    assert not (quiet / "overlay.mp4").exists()

    with pytest.raises(SystemExit) as stop:
        main(["predict", "--config", str(models["pipe"]), "--out", str(tmp_path / "nothing")])
    assert "predict needs --video, --views or --synthetic" in str(stop.value)


def test_predict_renders_a_synthetic_demo_and_fuses_a_rig(models, tmp_path, capsys):
    out = tmp_path / "synth"
    assert main(["predict", "--synthetic", "--config", str(models["pipe"]), "--out", str(out), "--max-frames",
                 "16"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith(f"-> {out}") and printed.startswith("16 frames,")
    assert (out / "tracks.npz").is_file() and (out / "overlay.mp4").is_file()

    rig = tmp_path / "rig"
    rig.mkdir()
    sim = simulate(1, 12, "mouse", seed=7)
    cams = make_cameras(sim.arena_cm, ("top", "angled"), size=(320, 256))
    for i, cam in enumerate(cams):
        write_video(rig / f"view{i}.mp4", [f.image for f in render_video(sim, cam, seed=i)], FPS)
    (rig / "cams.json").write_text(json.dumps([c.to_dict() for c in cams]))
    assert main(["predict", "--views", str(rig / "view0.mp4"), str(rig / "view1.mp4"), "--cameras",
                 str(rig / "cams.json"), "--config", str(models["pipe"]), "--out", str(rig / "out")]) == 0
    printed = capsys.readouterr().out.strip()
    animals, views, _ = re.fullmatch(r"(\d+) animals fused across (\d+) views -> (.*)", printed).groups()
    z = np.load(rig / "out" / "animals3d.npz")
    assert int(views) == 2 and int(animals) == len({k.split("_")[0] for k in z.files})
    assert (rig / "out" / "view0_tracks.csv").is_file() and (rig / "out" / "view1_tracks.csv").is_file()


def test_predict_follows_the_timestamps_of_an_irregular_image_folder(models, tmp_path, capsys):
    """A TIF sequence whose names carry the real capture times, with one frame missing."""
    folder = tmp_path / "frames"
    folder.mkdir()
    ms = [0, 33, 66, 100, 133, 166, 200, 300, 333, 366]  # a 100 ms gap where a frame was dropped
    for t in ms:
        img = np.full((128, 160, 3), 40, np.uint8)
        img[40:80, (t // 20) % 140:(t // 20) % 140 + 16] = 220
        cv2.imwrite(str(folder / f"cam_{t // 1000:02d}_{t // 60000:02d}_{t // 1000:02d}_{t % 1000:03d}.tif"), img)
    out = tmp_path / "seq"
    assert main(["predict", "--video", str(folder), "--config", str(models["pipe"]), "--out", str(out),
                 "--no-overlay"]) == 0
    printed = capsys.readouterr().out
    assert "variable frame rate / dropped frames:" in printed
    assert printed.strip().splitlines()[-1].startswith("10 frames,")
    timing = json.loads((out / "timing.json").read_text())
    assert timing["frames"] == 10 and timing["variable"] is True and timing["gaps"] == 1
    assert timing["longest_gap_s"] == pytest.approx(0.1, abs=1e-6)
    by_frame = {int(r.split(",")[0]): float(r.split(",")[1]) for r in (out / "tracks.csv").read_text().splitlines()[1:]}
    assert by_frame[min(by_frame)] == pytest.approx(0.0, abs=1e-9)
    assert by_frame[max(by_frame)] == pytest.approx(0.366, abs=1e-6)  # the file names, not 1/30 per frame


# -------------------------------------------------------------------------------------- data convert


def xy(names):
    """DeepLabCut repeats every bodypart name over its x column and its y column."""
    return [bp for bp in names for _ in (0, 1)]


def dlc_csv(path: Path, multi: bool) -> Path:
    """A DeepLabCut export: four header rows for multi-animal, three for single-animal, and the image
    path split over three columns the way ``labeled-data/<video>/<frame>.png`` exports write it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["nose", "neck", "tail_base"]
    if multi:
        rows = [["scorer"] * 3 + ["Me"] * 12, ["individuals"] * 3 + ["mouse1"] * 6 + ["mouse2"] * 6,
                ["bodyparts"] * 3 + xy(parts) * 2, ["coords"] * 3 + ["x", "y"] * 6,
                ["labeled-data", "vid1", "img0.png", "10", "20", "25", "22", "40", "28",
                 "60", "20", "75", "22", "90", "28"],
                ["labeled-data", "vid1", "img1.png", "12", "22", "27", "24", "42", "30"] + [""] * 6,
                ["labeled-data", "vid1", "img2.png"] + [""] * 12]
    else:
        rows = [["scorer"] + ["Me"] * 6, ["bodyparts"] + xy(("snout", "tailbase", "leftear")),
                ["coords"] + ["x", "y"] * 3,
                ["img0.png", "10", "20", "25", "22", "40", "28"], ["img1.png", "12", "22", "27", "24", "42", "30"]]
    path.write_text("\n".join(",".join(r) for r in rows) + "\n")
    return path


def test_data_convert_turns_a_deeplabcut_csv_into_coco(tmp_path, capsys):
    csv = dlc_csv(tmp_path / "labeled-data" / "vid1" / "CollectedData_Me.csv", multi=True)
    for name in ("img0.png", "img1.png", "img2.png"):  # the images the CSV names, for --image-root
        cv2.imwrite(str(tmp_path / "labeled-data" / "vid1" / name), np.zeros((96, 120, 3), np.uint8))
    out = tmp_path / "vid1.json"
    assert main(["data", "convert", str(csv), "--out", str(out), "--image-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "3 images, 3 animals, keypoints: ['nose', 'neck', 'tail_base']"
    coco = json.loads(out.read_text())
    assert [(c["id"], c["name"]) for c in coco["categories"]] == [(1, "animal")]
    assert coco["categories"][0]["keypoints"] == ["nose", "neck", "tail_base"]  # DLC names, not canonical ones
    assert [im["file_name"] for im in coco["images"]] == [f"labeled-data/vid1/img{i}.png" for i in range(3)]
    assert [(im["height"], im["width"]) for im in coco["images"]] == [(96, 120)] * 3  # read from the files
    assert len(coco["annotations"]) == 3  # both animals of frame 0, one of frame 1, none of the empty frame
    assert [a["track_id"] for a in coco["annotations"]] == [0, 1, 0]
    for ann in coco["annotations"]:
        assert ann["num_keypoints"] == 3 and np.asarray(ann["keypoints"], float).reshape(-1, 3)[:, 2].max() == 2
        assert ann["area"] == pytest.approx(ann["bbox"][2] * ann["bbox"][3])
    first = np.asarray(coco["annotations"][0]["keypoints"], float).reshape(-1, 3)
    assert first[:, :2].tolist() == [[10, 20], [25, 22], [40, 28]]  # the points of the first animal
    assert coco["annotations"][0]["bbox"] == pytest.approx([2.0, 14.2, 46.0, 19.6])  # padded around the points
    assert coco["annotations"][1]["image_id"] == coco["annotations"][0]["image_id"] == 0
    assert coco["annotations"][2]["image_id"] == 1

    single = dlc_csv(tmp_path / "CollectedData_Single.csv", multi=False)
    assert main(["data", "convert", str(single), "--out", str(tmp_path / "single.json")]) == 0
    assert capsys.readouterr().out.strip() == "2 images, 2 animals, keypoints: ['snout', 'tailbase', 'leftear']"
    doc = json.loads((tmp_path / "single.json").read_text())
    assert doc["categories"][0]["keypoints"] == ["snout", "tailbase", "leftear"]
    assert {a["track_id"] for a in doc["annotations"]} == {0} and len(doc["images"]) == 2
    assert "height" not in doc["images"][0]  # --image-root was not offered, so sizes are left out


# -------------------------------------------------------------------------------------- data prepare


def lp_csv(path: Path, bodyparts: list[str], rows: list[list[float]], index: list[str]) -> None:
    cols = pd.MultiIndex.from_product([["Skeleton5"], bodyparts, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    frame = pd.DataFrame(np.asarray(rows, float).reshape(len(index), -1), columns=cols, index=index)
    frame.index.name = "file_name"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def png(path: Path, size_hw=(120, 160), seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.random.default_rng(seed).random((*size_hw, 3)) * 255).astype(np.uint8))
    return path


def fake_raw(root: Path) -> Path:
    """A stand-in for the tree ``animalpose data download`` leaves under data/raw/."""
    crim = root / "lp_crim13" / "extracted" / "crim13"
    parts = ["black_mouse_nose", "black_mouse_top_of_neck", "black_mouse_whisker_pad", "white_mouse_nose",
             "white_mouse_base_tail"]  # two animals; "whisker_pad" has no counterpart of ours
    index = [f"s{a}/frame{b:02d}.png" for a in (1, 2) for b in range(2)]
    rows, y = [], 10
    for _ in index:
        rows.append([20, y, 60, y, 15, y + 5, 40, y + 10, 80, y + 12])
        y += 30
    for split in ("labels_InD.csv", "labels_OOD.csv"):
        lp_csv(crim / split, parts, rows, index)
    for rel in index:
        png(crim / rel)
    mirror = root / "lp_mirror_mouse" / "extracted" / "mirror-mouse"
    mp = [bp + s for bp in ("paw1LH", "paw2LF", "paw3RF", "paw4RH", "tailBase", "tailMid", "nose")
          for s in ("_top", "_bot")]
    mrows, idx = [], []
    for f in range(3):
        row, idx = [], idx + [f"sess/frame{f:02d}.png"]
        for i in range(len(mp) // 2):
            row += [20 + 6 * i, 30 + 4 * i, 20 + 6 * i, 200 + 4 * i]
        mrows.append(row)
        png(mirror / idx[-1], (320, 160))
    lp_csv(mirror / "labels_InD.csv", mp, mrows, idx)
    lp_csv(mirror / "labels_OOD.csv", mp, [mrows[0]], ["other/frame00.png"])
    png(mirror / "other" / "frame00.png", (320, 160), seed=5)
    iro = root / "irodent" / "extracted"
    names = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base", "whisker_pad"]
    ann = {"images": [{"id": 0, "file_name": "img0.png", "height": 120, "width": 160}],
           "categories": [{"id": 1, "name": "mouse", "keypoints": names, "skeleton": []}],
           "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "iscrowd": 0, "num_keypoints": 7,
                            "bbox": [10.0, 10.0, 60.0, 40.0],
                            "keypoints": [v for i in range(len(names)) for v in (20 + 3 * i, 30 + 2 * i, 2)]}]}
    (iro / "annotations").mkdir(parents=True, exist_ok=True)
    (iro / "annotations" / "irodent_coco.json").write_text(json.dumps(ann))
    png(iro / "images" / "img0.png")
    return root


def test_data_prepare_converts_every_fake_download(tmp_path, capsys):
    raw, out = fake_raw(tmp_path / "raw"), tmp_path / "processed"
    assert main(["data", "prepare", "--raw", str(raw), "--out", str(out), "--crim13", "--mirror-mouse",
                 "--irodent"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["crim13"] == {"train": {"images": 4, "animals": 8, "unmapped_bodyparts": ["black_mouse_whisker_pad"]},
                               "val": {"images": 4, "animals": 8, "unmapped_bodyparts": ["black_mouse_whisker_pad"]}}
    assert stats["mirror_mouse"] == {"side_train": {"images": 3, "animals": 3}, "bottom_train": {"images": 3, "animals": 3},
                                     "side_val": {"images": 1, "animals": 1}, "bottom_val": {"images": 1, "animals": 1}}

    doc = json.loads((out / "crim13" / "train.json").read_text())
    assert [(c["id"], c["name"]) for c in doc["categories"]] == [(1, "animal")]
    assert doc["categories"][0]["keypoints"] == list(CANONICAL.names)
    assert len(doc["annotations"]) == 8 and all(a["num_keypoints"] > 0 for a in doc["annotations"])
    assert (out / "crim13" / "images").is_dir() and (out / "mirror_mouse" / "images" / "side").is_dir()

    side = json.loads((out / "mirror_mouse" / "side_train.json").read_text())
    assert [im["file_name"] for im in side["images"]] == [f"side/sess_frame{f:02d}.jpg" for f in range(3)]
    kp = np.asarray(side["annotations"][0]["keypoints"], float).reshape(-1, 3)
    assert {CANONICAL.names[i] for i in np.nonzero(kp[:, 2] > 0)[0]} == {"left_hind_paw", "left_front_paw",
                                                                         "right_front_paw", "right_hind_paw",
                                                                         "tail_base", "tail_mid", "nose"}
    assert (side["images"][0]["height"], side["images"][0]["width"]) == (160, 160)  # the side half of the frame

    iro = json.loads((out / "irodent" / "test.json").read_text())
    assert iro["categories"][0]["keypoints"] == list(CANONICAL.names) and len(iro["annotations"]) == 1
    assert (out / "irodent" / "IMAGE_ROOT").read_text() == str((raw / "irodent" / "extracted" / "images").resolve())
    # the CalMS21 names are canonical already, so the detector settles on the synthetic vocabulary
    assert stats["irodent"]["mapping"] == "synthetic" and stats["irodent"]["unmapped"] == ["whisker_pad"]
    assert len(stats["irodent"]["mapped"]) == 7 and stats["irodent"]["species"] == {"mouse": 1}
    assert stats["irodent"]["images"] == 1 and stats["irodent"]["animals"] == 1


def test_data_prepare_reports_a_missing_download_as_an_error(tmp_path, capsys):
    raw, out = tmp_path / "nothing-here", tmp_path / "processed"
    assert main(["data", "prepare", "--raw", str(raw), "--out", str(out), "--rat7m"]) == 1
    assert "mocap-s1-d1.mat" in capsys.readouterr().err  # the download never happened
    assert main(["data", "prepare", "--raw", str(raw), "--out", str(out), "--rat7m-s2"]) == 1
    assert "mocap-s2-d1.mat" in capsys.readouterr().err  # subject 2 is a separate conversion
    assert not (out / "rat7m").exists()
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data", "prepare", "--all", "--rat7m-every"])  # --rat7m-every wants a value


def irodent_tree(root: Path, file_name: str, images_at: Path | None) -> Path:
    """The one annotation ``data prepare --irodent`` reads, with its images wherever they are."""
    names = ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"]
    ann = {"images": [{"id": 0, "file_name": file_name, "height": 120, "width": 160}],
           "categories": [{"id": 1, "name": "mouse", "keypoints": names, "skeleton": []}],
           "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "iscrowd": 0, "num_keypoints": 7,
                            "bbox": [10.0, 10.0, 60.0, 40.0],
                            "keypoints": [v for i in range(len(names)) for v in (20 + 3 * i, 30 + 2 * i, 2)]}]}
    (root / "irodent" / "extracted" / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "irodent" / "extracted" / "annotations" / "irodent_coco.json").write_text(json.dumps(ann))
    if images_at is not None:
        png(images_at / Path(file_name).name)
    return root


def test_data_prepare_locates_the_images_the_irodent_json_names(tmp_path, capsys):
    """The iRodent JSON stores bare file names; ``IMAGE_ROOT`` has to point at the folder that holds them."""
    raw, out = tmp_path / "raw", tmp_path / "processed"
    ann_dir = raw / "irodent" / "extracted" / "annotations"
    irodent_tree(raw, "img0.png", ann_dir)  # unpacked next to the annotation
    assert main(["data", "prepare", "--raw", str(raw), "--out", str(out), "--irodent"]) == 0
    assert json.loads(capsys.readouterr().out)["irodent"]["images"] == 1
    assert (out / "irodent" / "IMAGE_ROOT").read_text() == str(ann_dir.resolve())

    nested = tmp_path / "unpacked"  # the folder a tar really leaves: images/ beside annotations/
    irodent_tree(nested, "images/img0.png", nested / "irodent" / "extracted" / "images")
    assert main(["data", "prepare", "--raw", str(nested), "--out", str(tmp_path / "p2"), "--irodent"]) == 0
    assert json.loads(capsys.readouterr().out)["irodent"]["images"] == 1
    assert (tmp_path / "p2" / "irodent" / "IMAGE_ROOT").read_text() == str(
        (nested / "irodent" / "extracted").resolve())  # the file_name carries its own folder
    assert (Path((tmp_path / "p2" / "irodent" / "IMAGE_ROOT").read_text()) / "images" / "img0.png").is_file()


# ------------------------------------------------------------------------------------ data calibrate


def test_data_calibrate_solves_a_rig_from_projected_corners(tmp_path, capsys):
    cams = make_cameras(40.0, ("top", "side"), size=(320, 256))
    corners = np.array([[x, y, z] for x in (-20.0, 20.0) for y in (-20.0, 20.0) for z in (0.0, 10.0)], float)
    spec = {"cameras": [{"name": c.name, "size": [320, 256], "points3d": corners.tolist(),
                         "points2d": c.project(corners).tolist()} for c in cams]}
    points, out = tmp_path / "rig.json", tmp_path / "cams.json"
    points.write_text(json.dumps(spec))
    assert main(["data", "calibrate", str(points), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    errors = dict(re.findall(r"(\w+): mean reprojection error ([\d.]+)px", printed))
    assert set(errors) == {"top", "side"} and all(float(v) < 0.05 for v in errors.values())
    assert printed.strip().endswith(f"wrote {out}")
    solved = json.loads(out.read_text())
    assert [c["name"] for c in solved] == ["top", "side"]
    assert set(solved[0]) >= {"name", "K", "R", "t", "dist", "size"}  # what Camera.from_dict reads back
    for entry, cam in zip(solved, cams):
        assert np.abs(Camera.from_dict(entry).project(corners) - cam.project(corners)).max() < 1e-6
        assert Camera.from_dict(entry).size == (320, 256)

    seen = np.stack([c.project(corners) for c in cams]) + np.random.default_rng(3).normal(0, 0.4, (2, 8, 2))
    measured = tmp_path / "measured.json"  # what a survey writes: the arena corners as photographed
    measured.write_text(json.dumps({"cameras": [{"name": c.name, "size": [320, 256], "points3d": corners.tolist(),
                                                 "points2d": seen[i].tolist()} for i, c in enumerate(cams)]}))
    observed = tmp_path / "observed.npz"
    np.savez(observed, obs=seen)
    refined = tmp_path / "cams_refined.json"
    assert main(["data", "calibrate", str(measured), "--out", str(refined), "--refine", str(observed)]) == 0
    before, after = (float(v) for v in re.search(r"bundle adjustment: ([\d.]+)px -> ([\d.]+)px",
                   capsys.readouterr().out).groups())
    assert before > 0.05 and after <= before  # the joint fit has slack to take up, and it takes it
    for i, entry in enumerate(json.loads(refined.read_text())):
        assert np.linalg.norm(Camera.from_dict(entry).project(corners) - seen[i], axis=-1).mean() < 1.0

    known = tmp_path / "rig_known_intrinsics.json"  # with K given, the solver uses PnP instead of DLT
    known.write_text(json.dumps({"cameras": [dict(c, K=cam.K.tolist(), dist=[0.0] * 5)
                                             for c, cam in zip(spec["cameras"], cams)]}))
    solved = tmp_path / "cams_known.json"
    assert main(["data", "calibrate", str(known), "--out", str(solved)]) == 0
    printed = dict(re.findall(r"(\w+): mean reprojection error ([\d.]+)px", capsys.readouterr().out))
    assert set(printed) == {"top", "side"} and all(float(v) < 0.05 for v in printed.values())
    assert all(np.abs(Camera.from_dict(c).project(corners) - cam.project(corners)).max() < 1e-6
               for c, cam in zip(json.loads(solved.read_text()), cams))
