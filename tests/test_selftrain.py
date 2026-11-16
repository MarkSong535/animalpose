"""What the pseudo-labeling writer is allowed to teach: every frame comes from our own simulator or a
hand-drawn arena, the teacher is a tiny trained network, and the COCO json, the split and each of the
acceptance rules (confident, on moving pixels, reproduced under a mirror flip, no head/tail flip) are
asserted on the labels that were actually written."""

import json
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from animalpose.data.synthetic import make_cameras, render, simulate
from animalpose.frames import Source, find_sources, is_held_out
from animalpose.pipeline import PipelineConfig
from animalpose.selftrain import LabelOptions, label_videos, source_id
from animalpose.skeleton import CANONICAL
from animalpose.train.loops import train_detector, train_pose

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
NAMES = CANONICAL.names
CUP = (12, 20, 26, 34)
PAD_X, PAD_Y = 20, 14  # how far the teacher's box overshoots the animal it found
BASE = dict(hi=0.5, lo=0.2, kp_conf=0.4, flip_tol=0.3, min_kps=4, min_seq=8, crop_size=64, det_size=(256, 320))


@pytest.fixture(scope="module")
def teacher(tmp_path_factory):
    """A detector and a pose net trained a few hundred steps on synthetic scenes: real weights, far
    from accurate, exactly what a pseudo-label has to survive."""
    runs = tmp_path_factory.mktemp("selftrain-runs")
    data = {"n_sims": 4, "frames": 40, "image_size": [160, 128]}
    det, _ = train_detector({"device": "cpu", "output": str(runs / "det.pt"), "model": {"backbone": TINY},
                             "data": {**data, "input_size": [128, 160]},
                             "train": {"steps": 250, "batch_size": 4, "log_every": 0, "workers": 0}})
    pose, _ = train_pose({"device": "cpu", "output": str(runs / "pose.pt"),
                          "model": {"backbone": TINY, "embed_dim": 16},
                          "data": {**data, "input_size": [128, 160], "crop_size": 64},
                          "train": {"steps": 150, "batch_size": 8, "log_every": 0, "workers": 0}})
    assert det.stride == 4 and pose.num_keypoints == CANONICAL.num_keypoints
    return str(runs / "det.pt"), str(runs / "pose.pt")


@pytest.fixture(scope="module")
def cages(tmp_path_factory):
    """Four fixed-camera recordings of one dark mouse crossing a bright arena, as frame folders; two
    of them hash to the held-out side of the split."""
    root = tmp_path_factory.mktemp("cages")
    sim = simulate(1, 24, "mouse", seed=11, coats=["black"])
    sim.kpts3d[0, :, :, 0] += np.linspace(-8.0, 10.0, sim.n_frames)[:, None]
    cam = make_cameras(sim.arena_cm, ("top",), size=(320, 256))[0]
    floor = render(sim, cam, 0, np.random.default_rng(0), noise=True).image
    for name in ("cage_a", "cage_b", "cage_d", "cage_e"):
        (root / name).mkdir()
        for t in range(sim.n_frames):
            frame = render(sim, cam, t, np.random.default_rng(t), background=floor, noise=False)
            cv2.imwrite(str(root / name / f"f{t:04d}.png"), frame.image)
    held = {s.path.name for s in find_sources(root) if is_held_out(s, root)}
    assert held == {"cage_d", "cage_e"}, held  # a fixed 1 in 5 by path hash; both sides are present
    return root


def _run(teacher, root, out, **changes):
    detector, pose = teacher or ("", "")
    stats = label_videos(root, LabelOptions(detector=detector, pose=pose, out=out,
                                            **{"progress": False, **BASE, **changes}))
    coco = json.loads((Path(out) / "train.json").read_text())
    per_image = {}
    for ann in coco["annotations"]:
        per_image.setdefault(ann["image_id"], []).append(ann)
    return stats, coco, per_image, Path(out)


def _keypoints(ann):
    return np.asarray(ann["keypoints"], float).reshape(CANONICAL.num_keypoints, 3)


def test_the_written_labels_are_canonical_coco(teacher, cages, tmp_path):
    stats, coco, per_image, out = _run(teacher, cages, tmp_path / "pseudo", per_source=12, boxes="blob")
    assert stats["sources"] == 4 and stats["frames"] == 24 and stats["labeled"] > 3
    assert stats["labeled"] == stats["animals"] == stats["images"] == len(coco["images"])
    assert coco["categories"] == [{"id": 1, "name": "animal", "keypoints": NAMES, "skeleton": []}]
    assert all(a["category_id"] == 1 and a["iscrowd"] == 0 for a in coco["annotations"])
    assert [im["id"] for im in coco["images"]] == list(range(len(coco["images"])))
    assert {a["image_id"] for a in coco["annotations"]} == set(per_image)
    for ann in coco["annotations"]:
        kp = _keypoints(ann)
        assert set(np.unique(kp[:, 2])) <= {0.0, 2.0}  # labeled or unlabeled, never "occluded"
        assert (kp[kp[:, 2] == 0][:, :2] == 0).all()  # a dropped point is written as 0, 0, 0
        kept = kp[kp[:, 2] == 2]
        assert len(kept) >= BASE["min_kps"] and (kept[:, :2] >= 0).all()
        assert (kept[:, 0] < 320).all() and (kept[:, 1] < 256).all()
        x, y, w, h = ann["bbox"]
        assert w > 5 and h > 5 and 0 <= x < x + w <= 320 and 0 <= y < y + h <= 256
        assert ann["area"] == pytest.approx(w * h) and ann["num_keypoints"] == len(kept)
    assert stats["keypoints"] == sum(int((_keypoints(a)[:, 2] == 2).sum()) for a in coco["annotations"])
    for im in coco["images"]:
        on_disk = cv2.imread(str(out / "images" / im["file_name"]))
        assert on_disk is not None and on_disk.shape == (im["height"], im["width"], 3)
        assert im["file_name"].endswith(".jpg") and im["height"] == on_disk.shape[0]
    sids = {Path(im["file_name"]).parent.name for im in coco["images"]}
    split = json.loads((out / "split.json").read_text())
    assert sids == {source_id(cages / rel, cages) for rel in split["train"]}
    assert source_id(cages / "cage_a", cages) == f"{zlib.crc32(b'cage_a'):08x}_cage_a"


def test_a_stricter_teacher_labels_fewer_frames(teacher, cages, tmp_path):
    looser, loose_coco, _, _ = _run(teacher, cages, tmp_path / "loose", per_source=12, boxes="blob")
    assert looser["frames"] == 24 and looser["labeled"] > 3 and looser["keypoints"] > 3
    for tag, change in (("kp_conf", dict(kp_conf=0.9)), ("flip_tol", dict(flip_tol=0.05)),
                        ("min_kps", dict(min_kps=12))):
        stats, coco, _, _ = _run(teacher, cages, tmp_path / tag, per_source=12, boxes="blob", **change)
        assert stats["frames"] == looser["frames"], tag  # the very same frames were offered
        assert stats["labeled"] < looser["labeled"], tag
        assert stats["keypoints"] < looser["keypoints"], tag
        assert len(coco["images"]) == stats["labeled"], tag
        assert len(coco["annotations"]) == stats["animals"], tag
    # The 0.9 confidence cutoff rejects this teacher. The exact number of flip matches varies
    # with the trained weights, so check the reduction for that case above.
    assert json.loads((tmp_path / "kp_conf" / "train.json").read_text())["annotations"] == []
    kept = [_keypoints(a)[:, 2] for a in loose_coco["annotations"]]
    assert all(v.max() == 2 for v in kept) and any(v.min() == 0 for v in kept)  # some points do drop
    assert looser["labeled"] > sum(1 for a in loose_coco["annotations"] if a["num_keypoints"] >= 12)


def test_the_requested_range_labels_the_first_half_of_the_recording(tmp_path, monkeypatch):
    root = tmp_path / "root"
    scene = _flat_arena(root / "cage_a", n=20)
    install_teacher(monkeypatch, score=1.0)
    whole, _, _, _ = _run(None, root, tmp_path / "whole", per_source=20, boxes="detector")
    half, coco, _, _ = _run(None, root, tmp_path / "half", per_source=20, boxes="detector",
                            time_range=(0.0, 0.5))
    assert whole["frames"] == whole["labeled"] == 20 and half["frames"] == half["labeled"] == 10
    assert [int(Path(im["file_name"]).stem) for im in coco["images"]] == list(range(10))  # frames 0 to 9
    for im, ann in zip(coco["images"], coco["annotations"]):
        t = int(Path(im["file_name"]).stem)
        assert ann["bbox"][0] == pytest.approx(scene["x0"] + scene["step"] * t - 6, abs=0.5)


def test_only_confident_detections_teach_an_animal(tmp_path, monkeypatch):
    root = tmp_path / "root"
    scene = _flat_arena(root / "cage_a")
    install_teacher(monkeypatch, score=0.5)
    uncertain, coco, _, out = _run(None, root, tmp_path / "uncertain", per_source=18, boxes="detector",
                                   hi=0.9)
    assert uncertain["frames"] == uncertain["skipped_uncertain"] == 18 and uncertain["labeled"] == 0
    assert coco["images"] == coco["annotations"] == [] and not list((out / "images").rglob("*.jpg"))
    install_teacher(monkeypatch, score=0.5)
    accepted, coco, per_image, _ = _run(None, root, tmp_path / "accepted", per_source=18, boxes="detector",
                                        hi=0.4)
    assert accepted["labeled"] == accepted["animals"] == 18 and accepted["keypoints"] == 25 * 18
    assert accepted["skipped_uncertain"] == accepted["no_animal"] == 0
    assert len(coco["images"]) == 18 and all(len(v) == 1 for v in per_image.values())
    for im, ann in zip(coco["images"], coco["annotations"]):
        x0, y0, x1, y1 = _animal_box(scene, int(Path(im["file_name"]).stem))
        assert ann["num_keypoints"] == 25  # confident and reproduced under the mirror flip
        bx, by, bw, bh = ann["bbox"]
        pad_x, pad_y = 0.1 * (x1 - x0 + 6), 0.1 * (y1 - y0 + 6)  # the box shrunk to the moving pixels
        assert bw < 0.75 * (x1 - x0 + 2 * PAD_X) and bh < 0.75 * (y1 - y0 + 2 * PAD_Y)
        assert bx == pytest.approx(x0 - 3 - pad_x, abs=0.5) and by == pytest.approx(y0 - 3 - pad_y, abs=0.5)
        assert bw == pytest.approx(x1 - x0 + 6 + 2 * pad_x, abs=0.5)
        assert bh == pytest.approx(y1 - y0 + 6 + 2 * pad_y, abs=0.5)
        kp = _keypoints(ann)
        assert kp[0, 2] == 2 and kp[0, 0] == pytest.approx(bx + bw / 2, abs=1.0)
    install_teacher(monkeypatch, score=0.05)
    missed, coco, _, _ = _run(None, root, tmp_path / "missed", per_source=18, boxes="detector", hi=0.4)
    assert missed["no_animal"] == 18 and missed["labeled"] == 0 and coco["annotations"] == []


def test_the_background_filter_decides_what_counts_as_an_animal(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _flat_arena(root / "cage_a")
    install_teacher(monkeypatch, score=1.0, cup=True)
    filtered, coco, per_image, _ = _run(None, root, tmp_path / "filtered", per_source=18, boxes="detector")
    assert filtered["labeled"] == filtered["animals"] == 18  # one animal per frame, never the cup
    assert all(len(v) == 1 for v in per_image.values())
    assert all(a["bbox"][1] > CUP[3] for a in coco["annotations"])  # the static cup got no box at all
    taught, coco, per_image, _ = _run(None, root, tmp_path / "taught", per_source=18, boxes="detector",
                                      min_seq=10 ** 7)
    assert taught["labeled"] == 18 and taught["animals"] == 36  # without a background the cup is an animal
    assert all(len(v) == 2 for v in per_image.values())
    cups = [a["bbox"] for a in coco["annotations"] if a["bbox"][1] < CUP[3]]
    assert len(cups) == 18  # each one is exactly the box the teacher asked for, unshrunk
    assert all(b == pytest.approx([CUP[0] - PAD_X, CUP[1] - PAD_Y, CUP[2] - CUP[0] + 2 * PAD_X,
                                   CUP[3] - CUP[1] + 2 * PAD_Y]) for b in cups)


def test_keypoints_are_rejected_when_off_the_motion_or_off_the_mirror(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _flat_arena(root / "cage_a")
    layout = {"nose": (0.5, -0.1), "spine_lower": (0.72, 0.5)}
    install_teacher(monkeypatch, score=1.0, layout=layout)
    masked, coco_a, per_image_a, _ = _run(None, root, tmp_path / "masked", per_source=18, boxes="detector")
    unmasked, coco_b, per_image_b, _ = _run(None, root, tmp_path / "unmasked", per_source=18,
                                            boxes="detector", min_seq=10 ** 7)
    nose, lower, neck = (CANONICAL.index(n) for n in ("nose", "spine_lower", "neck"))
    for coco, per_image, expect in ((coco_a, per_image_a, 23), (coco_b, per_image_b, 24)):
        assert len(per_image) == 18
        for ann in coco["annotations"]:
            kp = _keypoints(ann)
            assert ann["num_keypoints"] == expect == int((kp[:, 2] == 2).sum())
            # an off-centre midline point is not reproduced by the mirrored estimate
            assert (kp[lower] == [0.0, 0.0, 0]).all()
            assert kp[neck, 2] == 2 and np.isfinite(kp[neck, :2]).all()
            if coco is coco_a:
                assert (kp[nose] == [0.0, 0.0, 0]).all()  # it is confident, but not on a moving pixel
            else:  # with no background nothing rejects it: it floats 10 % of the box above the animal
                by, bh = ann["bbox"][1], ann["bbox"][3]
                assert kp[nose, 2] == 2 and kp[nose, 1] == pytest.approx(by - 0.1 * bh, abs=1.0)
    assert masked["keypoints"] == 23 * 18 and unmasked["keypoints"] == 24 * 18  # one point per frame less
    assert per_image_a[0][0]["keypoints"][:3] == [0.0, 0.0, 0]


def test_a_nose_on_the_tail_side_is_thrown_out_wholesale(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _tailed_arena(root / "cage_a")
    install_teacher(monkeypatch, score=1.0, layout={"nose": (0.5, 0.8), "tail_base": (0.5, 0.2)})
    flipped, coco, _, out = _run(None, root, tmp_path / "flipped", per_source=24, boxes="blob")
    install_teacher(monkeypatch, score=1.0, layout={"nose": (0.5, 0.2), "tail_base": (0.5, 0.8)})
    sound, coco_ok, _, out_ok = _run(None, root, tmp_path / "sound", per_source=24, boxes="blob")
    assert sound["frames"] == flipped["frames"] == 24 and sound["no_animal"] == 0
    assert sound["labeled"] == 24 and sound["head_tail_flip"] == 0 and sound["keypoints"] == 25 * 24
    assert flipped["head_tail_flip"] == 24 and flipped["labeled"] == 0  # the tail is closer to the nose
    assert coco["images"] == coco["annotations"] == [] and len(coco_ok["images"]) == 24
    assert json.loads((out / "split.json").read_text())["train"] == ["cage_a"]
    assert all(_keypoints(a)[[CANONICAL.index("nose"), CANONICAL.index("tail_base")], 2].tolist() == [2, 2]
               for a in coco_ok["annotations"])


def test_one_row_of_a_box_rejects_the_whole_frame(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _flat_arena(root / "cage_a")
    install_teacher(monkeypatch, score=1.0, sliver=True)
    dropped, coco, _, _ = _run(None, root, tmp_path / "sliver", per_source=18, boxes="detector")
    assert dropped["frames"] == 18 and dropped["labeled"] == dropped["animals"] == 0
    assert dropped["no_animal"] == 0 and dropped["keypoints"] == 0  # the frame was never offered
    assert coco["images"] == coco["annotations"] == []


def test_recordings_without_frames_or_without_a_background_are_left_out(tmp_path, monkeypatch):
    root = tmp_path / "root"
    _flat_arena(root / "cage_a", n=6)
    cv2.VideoWriter(str(root / "empty.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 96)).release()
    install_teacher(monkeypatch, score=1.0)
    stats, coco, _, out = _run(None, root, tmp_path / "short", per_source=4, boxes="detector",
                               min_seq=10 ** 7, holdout=False)
    assert stats["sources"] == 2  # the walk lists the video file before it descends into the folder
    assert json.loads((out / "split.json").read_text())["train"] == ["empty.avi", "cage_a"]
    assert stats["frames"] == stats["labeled"] == 4 and stats["keypoints"] == 25 * 4  # only the frames exist
    sid = source_id(root / "cage_a", root)
    assert len(coco["images"]) == 4 and all(im["file_name"].startswith(sid + "/") for im in coco["images"])
    blind, coco_none, _, _ = _run(None, root, tmp_path / "blobless", per_source=4, boxes="blob",
                                  min_seq=10 ** 7, holdout=False)
    assert blind["sources"] == 2 and blind["frames"] == blind["no_animal"] == 0
    assert coco_none["images"] == coco_none["annotations"] == []  # blob boxes need a background to cut out


def test_the_split_and_the_source_filter(teacher, cages, tmp_path, capsys):
    stats, coco, _, out = _run(teacher, cages, tmp_path / "split", per_source=2, boxes="blob")
    split = json.loads((out / "split.json").read_text())
    assert split["root"] == os.path.abspath(cages) and stats["frames"] == 4
    assert sorted(split["train"] + split["held_out"]) == ["cage_a", "cage_b", "cage_d", "cage_e"]
    assert split["train"] == ["cage_a", "cage_b"] and split["held_out"] == ["cage_d", "cage_e"]
    assert all(is_held_out(Source(cages / p, "images"), cages) for p in split["held_out"])
    held = {source_id(cages / p, cages) for p in split["held_out"]}
    assert held.isdisjoint({Path(im["file_name"]).parent.name for im in coco["images"]})
    again, coco_again, _, _ = _run(teacher, cages, tmp_path / "split", per_source=2, boxes="blob")
    assert again["labeled"] == stats["labeled"]  # a relabel overwrites the same image names
    assert [im["file_name"] for im in coco_again["images"]] == [im["file_name"] for im in coco["images"]]
    quiet, _, _, _ = _run(teacher, cages, tmp_path / "filtered", per_source=2, boxes="blob",
                          include=["cage_a"], progress=True)
    printed = capsys.readouterr().out
    assert quiet["sources"] == 1 and "1 recordings match the filter" in printed
    assert "cage_a" in printed and '"labeled"' in printed
    none, coco_none, _, _ = _run(teacher, cages, tmp_path / "none", per_source=2, boxes="blob",
                                 include=["atlas"])
    assert none["sources"] == none["frames"] == 0 and coco_none["images"] == []
    every, _, _, out_all = _run(teacher, cages, tmp_path / "every", per_source=2, boxes="blob",
                                holdout=False)
    split = json.loads((out_all / "split.json").read_text())
    assert split["held_out"] == [] and sorted(split["train"]) == ["cage_a", "cage_b", "cage_d", "cage_e"]
    assert every["sources"] == 4 and every["frames"] == 8


def _animal_box(scene, t):
    """The dark rectangle the scene drew at frame ``t``: (x0, y0, x1, y1)."""
    x = scene["x0"] + scene["step"] * t
    return x, scene["y"], x + scene["w"], scene["y"] + scene["h"]


def _flat_arena(folder: Path, n: int = 18) -> dict:
    """A 24x14 dark body walking right across a bright floor, with a static cup that never moves."""
    folder.mkdir(parents=True, exist_ok=True)
    scene = {"x0": 24, "step": 5, "y": 60, "w": 24, "h": 14}
    for t in range(n):
        img = np.full((96, 160, 3), 200, np.uint8)
        img[CUP[1]:CUP[3], CUP[0]:CUP[2]] = 60
        x = scene["x0"] + scene["step"] * t
        img[scene["y"]:scene["y"] + scene["h"], x:x + scene["w"]] = 30
        cv2.imwrite(str(folder / f"f{t:04d}.png"), img)
    return scene


def _tailed_arena(folder: Path, n: int = 24) -> None:
    """A 30x16 body with a 2 px tail off its lower right corner, inside a walled bright arena."""
    folder.mkdir(parents=True, exist_ok=True)
    for t in range(n):
        img = np.full((120, 320, 3), 90, np.uint8)
        img[20:110, 20:300] = 200
        x = 40 + 5 * t
        img[60:76, x:x + 30] = 30
        img[74:76, x + 30:x + 44] = 30
        cv2.imwrite(str(folder / f"f{t:04d}.png"), img)


class _Teacher:
    """Stand-in for :class:`PosePipeline`: the animal's box from the pixels, keypoints at fixed
    fractions of that box, so every acceptance rule can be switched on and off from the options."""

    score, conf, cup, sliver, layout = 1.0, 0.9, False, False, {}

    def __init__(self, detector, pose, config=None):
        self.cfg = config or PipelineConfig()

    def detect(self, image):
        dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 100
        ys, xs = np.nonzero(dark[50:])
        boxes = [[xs.min() - PAD_X, ys.min() + 50 - PAD_Y, xs.max() + 1 + PAD_X, ys.max() + 1 + 50 + PAD_Y]]
        if self.cup:
            boxes.append([CUP[0] - PAD_X, CUP[1] - PAD_Y, CUP[2] + PAD_X, CUP[3] + PAD_Y])
        if self.sliver:  # one row of the animal: all of it is moving, yet too few pixels to fit a box to
            boxes.append([xs.min(), ys.min() + 50, xs.min() + 19, ys.min() + 51])
        return np.asarray(boxes, np.float32), np.full(len(boxes), self.score, np.float32)

    def estimate(self, image, boxes):
        kp = np.zeros((len(boxes), CANONICAL.num_keypoints, 2), np.float32)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for k, part in enumerate(CANONICAL.keypoints):
                fx, fy = self.layout.get(part.name, ({"left": 0.25, "right": 0.75}.get(part.side, 0.5), 0.5))
                kp[i, k] = (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
        return kp, np.full(kp.shape[:2], self.conf, np.float32), None


def install_teacher(monkeypatch, score=1.0, conf=0.9, cup=False, sliver=False, layout=None) -> None:
    """Point ``label_videos`` at that teacher instead of loading checkpoints."""
    _Teacher.score, _Teacher.conf = score, conf
    _Teacher.cup, _Teacher.sliver, _Teacher.layout = cup, sliver, dict(layout or {})
    monkeypatch.setattr("animalpose.selftrain.PosePipeline", _Teacher)
    monkeypatch.setattr("animalpose.selftrain.load_model", lambda path: path)
