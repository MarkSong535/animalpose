"""Tiny fake on-disk layouts for the real-dataset converters, asserting on the canonical COCO output.

Checks the 25-keypoint vocabulary and the single class-agnostic category, per-instance keypoint
counts, that unmapped source bodyparts are dropped rather than guessed, that a different source
keypoint order still lands on the right canonical names, that coordinates stay valid pixels, and
the documented split rules (mirror-mouse InD -> train / OOD -> val, Rat 7M held-out chunks -> val).
"""

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from animalpose.data.real import (
    MIRROR_MOUSE,
    MIRROR_SPLIT_Y,
    convert_coco_keypoints,
    convert_lp_crim13,
    convert_mirror_mouse,
    convert_rat7m,
    detect_mapping,
    rat7m_cameras,
)
from animalpose.multiview.camera import Camera
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS

RAT7M_MARKERS = {"HeadB": (0.0, 0.10, 0.03), "SpineF": (0.0, 0.06, 0.02), "SpineM": (0.0, 0.0, 0.02),
                 "SpineL": (0.0, -0.06, 0.02), "ShoulderL": (-0.03, 0.05, 0.01), "ShoulderR": (0.03, 0.05, 0.01),
                 "ElbowL": (-0.04, 0.03, 0.0), "ElbowR": (0.04, 0.03, 0.0), "HipL": (-0.03, -0.05, 0.01),
                 "HipR": (0.03, -0.05, 0.01), "KneeL": (-0.04, -0.07, 0.0), "KneeR": (0.04, -0.07, 0.0),
                 "HeadL": (-0.02, 0.11, 0.02), "ArmL": (-0.05, 0.04, 0.0), "ShinL": (-0.045, -0.09, 0.0),
                 "ArmR": (0.05, 0.04, 0.0), "ShinR": (0.045, -0.09, 0.0)}
RAT7M_UNMAPPED = ("HeadL", "ArmL", "ShinL", "ArmR", "ShinR")
RAT7M_SIZE = (64, 48)  # (width, height) of the fake camera videos


def _png(path: Path, size_hw: tuple[int, int], seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), (rng.random((*size_hw, 3)) * 255).astype(np.uint8))


def _lp_csv(path: Path, bodyparts: list[str], values: list[list[float]], index: list[str]) -> None:
    cols = pd.MultiIndex.from_product([["Skeleton5"], bodyparts, ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    frame = pd.DataFrame(np.asarray(values, float).reshape(len(index), -1), columns=cols, index=index)
    frame.index.name = "file_name"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _mp4(path: Path, n: int, size_wh: tuple[int, int], fps: float = 25.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size_wh
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        img[8:24, (4 * t) % (w - 12):(4 * t) % (w - 12) + 12] = 220
        vw.write(img)
    vw.release()


def _read(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text())


def _canonical_of(ann: dict) -> np.ndarray:
    return np.asarray(ann["keypoints"], float).reshape(CANONICAL.num_keypoints, 3)


def _check_canonical(doc: dict) -> None:
    """One class-agnostic category over the full vocabulary, and geometry that survives conversion."""
    assert [(c["id"], c["name"]) for c in doc["categories"]] == [(1, "animal")]
    assert doc["categories"][0]["keypoints"] == CANONICAL.names
    assert doc["categories"][0]["skeleton"] == []
    images = {im["id"]: im for im in doc["images"]}
    assert sorted(images) == list(range(len(images)))
    for a in doc["annotations"]:
        im = images[a["image_id"]]
        assert a["category_id"] == 1 and a["iscrowd"] == 0
        x, y, w, h = a["bbox"]
        assert 0 <= x and 0 <= y and 0 < w <= im["width"] and 0 < h <= im["height"]
        assert a["area"] == pytest.approx(w * h)
        kp = _canonical_of(a)
        seen = kp[:, 2] > 0
        assert a["num_keypoints"] == int(seen.sum()) > 0
        assert np.isfinite(kp[seen, :2]).all()
        assert (kp[seen, 0] <= im["width"]).all() and (kp[seen, 1] <= im["height"]).all()
        assert (kp[seen, 0] >= 0).all() and (kp[seen, 1] >= 0).all()
        assert np.isin(kp[:, 2], (0, 1, 2)).all()


def _mirror_rows(offset: int) -> list[list[float]]:
    row = []
    for bp in MIRROR_MOUSE:
        i = CANONICAL.index(MIRROR_MOUSE[bp])
        row += [10 + 3 * i + offset, 10 + i + offset, 10 + 3 * i + offset, MIRROR_SPLIT_Y + 5 + i + offset]
    return row


def test_mirror_mouse_cuts_side_and_bottom_views_and_keeps_the_split(tmp_path):
    root, out = tmp_path / "mirror", tmp_path / "out"
    parts = [bp + s for bp in MIRROR_MOUSE for s in ("_top", "_bot")]
    rows = [_mirror_rows(4 * f) for f in range(2)]
    sparse = [np.nan] * (4 * len(MIRROR_MOUSE))
    sparse[4 * list(MIRROR_MOUSE).index("nose")], sparse[4 * list(MIRROR_MOUSE).index("nose") + 1] = 10.0, 20.0
    rows += [sparse, [np.nan] * (4 * len(MIRROR_MOUSE))]
    ins = [f"sess1/frame{f:04d}.png" for f in range(2)] + ["sess1/frame0002.png", "sess1/gone.png"]
    for rel in ins[:3]:
        _png(root / rel, (200, 120), seed=abs(hash(rel)) % 99)
    _lp_csv(root / "labels_InD.csv", parts, rows, ins)
    _lp_csv(root / "labels_OOD.csv", parts, [_mirror_rows(0)], ["other_animal/frame0000.png"])
    _png(root / "other_animal/frame0000.png", (200, 120), seed=7)

    stats = convert_mirror_mouse(root, out)
    two = {"images": 2, "animals": 2}
    assert stats == {"side_train": two, "bottom_train": two, "side_val": {"images": 1, "animals": 1},
                     "bottom_val": {"images": 1, "animals": 1}}

    side = _read(out, "side_train.json")
    _check_canonical(side)
    assert side["images"][0]["file_name"] == "side/sess1_frame0000.jpg"
    assert (side["images"][0]["height"], side["images"][0]["width"]) == (MIRROR_SPLIT_Y, 120)
    assert (out / "images" / "side" / "sess1_frame0000.jpg").exists()
    kp = _canonical_of(side["annotations"][0])
    for canon in MIRROR_MOUSE.values():
        i = CANONICAL.index(canon)
        assert kp[i].tolist() == [10 + 3 * i, 10 + i, 2]
    assert side["annotations"][0]["num_keypoints"] == len(MIRROR_MOUSE)
    untouched = [i for i in range(CANONICAL.num_keypoints) if i not in CANONICAL.indices(MIRROR_MOUSE.values())]
    assert (kp[untouched] == 0).all()  # the other 18 canonical points stay unlabeled, never guessed
    assert (np.asarray(side["annotations"][0]["bbox"])[:2] >= 0).all()

    bottom = _read(out, "bottom_train.json")
    _check_canonical(bottom)
    assert bottom["images"][0]["height"] == 200 - MIRROR_SPLIT_Y
    kpb = _canonical_of(bottom["annotations"][0])
    for canon in MIRROR_MOUSE.values():  # the mirror half is re-anchored at its own top edge
        i = CANONICAL.index(canon)
        assert kpb[i, :2].tolist() == [10 + 3 * i, 5 + i]
    val = _read(out, "side_val.json")
    assert [im["file_name"] for im in val["images"]] == ["side/other_animal_frame0000.jpg"]
    assert side["images"][1]["file_name"] == "side/sess1_frame0001.jpg"
    kp1 = _canonical_of(side["annotations"][1])
    inose = CANONICAL.index("nose")
    assert kp1[inose].tolist() == [10 + 3 * inose + 4, 10 + inose + 4, 2]


CRIM13_PARTS = ["black_mouse_nose", "nose_white", "black_mouse_left_ear", "white_mouse_base_of_tail",
                "white_mouse_top_of_neck", "black_mouse_whisker_pad", "black_mouse_rear_end", "tunnel_center"]
CRIM13_XY = {"black_mouse_nose": (20, 30), "nose_white": (70, 30), "black_mouse_left_ear": (30, 25),
             "white_mouse_base_of_tail": (60, 60), "white_mouse_top_of_neck": (75, 40),
             "black_mouse_whisker_pad": (15, 35), "black_mouse_rear_end": (25, 55), "tunnel_center": (45, 45)}
CRIM13_UNMAPPED = ["black_mouse_whisker_pad", "black_mouse_rear_end", "tunnel_center"]


def test_lp_crim13_maps_both_spellings_and_drops_unmapped_bodyparts(tmp_path):
    root, out = tmp_path / "crim13", tmp_path / "out"
    values = [[v for p in CRIM13_PARTS for v in CRIM13_XY[p]] for _ in range(4)]
    _lp_csv(root / "labels_InD.csv", CRIM13_PARTS, values, ["s1/a.png", "s1/b.png", "s2/c.png", "s2/gone.png"])
    for rel in ("s1/a.png", "s1/b.png", "s2/c.png"):
        _png(root / rel, (100, 100), seed=len(rel))

    stats = convert_lp_crim13(root, out)
    assert stats["train"]["unmapped_bodyparts"] == CRIM13_UNMAPPED
    assert set(stats) == {"train"}  # labels_OOD.csv is absent, so there is no val split
    doc = _read(out, "train.json")
    _check_canonical(doc)
    assert stats["train"] == {"images": 3, "animals": 6, "unmapped_bodyparts": CRIM13_UNMAPPED}
    assert [im["file_name"] for im in doc["images"]] == ["s1_a.jpg", "s1_b.jpg", "s2_c.jpg"]
    assert (out / "images" / "s1_a.jpg").exists()
    assert {im["height"] for im in doc["images"]} == {100}

    per_image: dict[int, list[dict]] = {}
    for a in doc["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    base1, base2 = (zlib.crc32(b"s1") % 100000) * 16, (zlib.crc32(b"s2") % 100000) * 16
    ids = [sorted(a["track_id"] for a in per_image[i]) for i in (0, 1, 2)]
    assert ids == [[base1, base1 + 1], [base1, base1 + 1], [base2, base2 + 1]]
    assert base1 % 16 == 0 and base1 != base2  # one id per animal per session, stable across frames

    black, white = sorted(per_image[0], key=lambda a: a["track_id"])
    kb, kw = _canonical_of(black), _canonical_of(white)
    assert kb[CANONICAL.index("nose")].tolist() == [20, 30, 2] and kb[CANONICAL.index("left_ear")].tolist() == [30, 25, 2]
    assert kw[CANONICAL.index("nose")].tolist() == [70, 30, 2]  # the "nose_white" spelling still means nose
    assert kw[CANONICAL.index("tail_base")].tolist() == [60, 60, 2]
    assert kw[CANONICAL.index("neck")].tolist() == [75, 40, 2]  # top_of_neck -> neck
    assert black["num_keypoints"] == 2 and white["num_keypoints"] == 3
    for k in (kb, kw):
        for xy in ((15, 35), (25, 55), (45, 45)):  # none of the three unmapped bodyparts is guessed onto a point
            assert not ((k[:, :2] == np.array(xy)).all(-1) & (k[:, 2] > 0)).any()


def test_coco_keypoints_honours_an_explicit_mapping_and_its_order(tmp_path):
    names = ["Root of Tail", "Nose", "Left Back Paw", "Right Eye", "Not A Point", "Left Hip"]
    kps = [1, 2, 2, 10, 20, 2, 30, 40, 1, 50, 60, 2, 7, 8, 0, 90, 95, 0]
    src = {"images": [{"id": 5, "file_name": "vid/a.jpg", "height": 90, "width": 80},
                      {"id": 6, "file_name": "vid/b.jpg", "height": 90, "width": 80}],
           "categories": [{"id": 1, "name": "rat", "keypoints": names},
                          {"id": 2, "name": "mouse", "keypoints": names}],
           "annotations": [{"id": 1, "image_id": 5, "category_id": 1, "keypoints": kps, "num_keypoints": 4,
                            "bbox": [4, 5, 40, 30], "iscrowd": 0},
                           {"id": 2, "image_id": 5, "category_id": 2, "keypoints": kps, "num_keypoints": 0,
                            "bbox": [4, 5, 40, 30], "iscrowd": 0},
                           {"id": 3, "image_id": 6, "category_id": 2, "keypoints": kps, "num_keypoints": 4,
                            "bbox": [1, 1, 10, 10], "iscrowd": 1}]}
    (tmp_path / "in.json").write_text(json.dumps(src))
    stats = convert_coco_keypoints(tmp_path / "in.json", tmp_path / "out.json", mapping="ap10k")
    assert stats == {"images": 1, "animals": 1, "mapping": "ap10k", "species": {"rat": 1},
                     "mapped": ["Left Back Paw", "Left Hip", "Nose", "Right Eye", "Root of Tail"],
                     "unmapped": ["Not A Point"]}
    doc = _read(tmp_path, "out.json")
    _check_canonical(doc)
    kp = _canonical_of(doc["annotations"][0])
    assert kp[CANONICAL.index("tail_base")].tolist() == [1, 2, 2]  # first source name is the tail
    assert kp[CANONICAL.index("nose")].tolist() == [10, 20, 2]
    assert kp[CANONICAL.index("left_hind_paw")].tolist() == [30, 40, 1]  # v == 1 stays visible-but-occluded
    assert kp[CANONICAL.index("right_eye")].tolist() == [50, 60, 2]
    assert kp[[CANONICAL.index(n) for n in ("left_eye", "neck", "tail_end", "left_hip")]].tolist() == [[0, 0, 0]] * 4
    assert doc["annotations"][0]["bbox"] == [4, 5, 40, 30]  # a num_keypoints == 0 and an iscrowd annotation drop out


def test_coco_keypoints_reports_when_no_mapping_matches(tmp_path):
    names = ["implant_1", "implant_2", "ear_tag"]
    src = {"images": [{"id": 0, "file_name": "x.jpg", "height": 10, "width": 10}],
           "categories": [{"id": 1, "name": "gerbil", "keypoints": names}],
           "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "keypoints": [1, 2, 2, 3, 4, 1, 5, 6, 2],
                            "num_keypoints": 3, "bbox": [0, 0, 8, 8], "iscrowd": 0}]}
    (tmp_path / "in.json").write_text(json.dumps(src))
    assert detect_mapping(names) == ("", {})
    stats = convert_coco_keypoints(tmp_path / "in.json", tmp_path / "out.json")
    assert stats["mapping"] == "" and stats["unmapped"] == ["ear_tag", "implant_1", "implant_2"]
    doc = _read(tmp_path, "out.json")
    assert (_canonical_of(doc["annotations"][0]) == 0).all()


def test_detect_mapping_prefers_the_dataset_that_matches_most():
    assert detect_mapping(list(DATASET_MAPPINGS["ap10k"]))[0] == "ap10k"
    assert detect_mapping(list(DATASET_MAPPINGS["homecage_dlc"]))[0] == "homecage_dlc"
    hit = detect_mapping(["LEFT_EYE", "front-left-paw", "whatever"])[1]
    assert hit == {"LEFT_EYE": "left_eye", "front-left-paw": "left_front_paw"}


def _rat7m_mat(path: Path, cam: Camera) -> np.ndarray:
    import scipy.io as sio

    frame_of = np.r_[np.arange(8), 105000 + np.arange(8)]
    markers = {}
    for name, xyz in RAT7M_MARKERS.items():
        arr = np.tile(np.asarray(xyz, float), (len(frame_of), 1))
        arr[:, 1] += 0.004 * np.arange(len(frame_of))  # the animal walks along its own axis
        markers[name] = arr
    mapped = [n for n in RAT7M_MARKERS if n in DATASET_MAPPINGS["rat7m"]]
    for name in mapped:  # a frame where the mocap is lost
        markers[name][4] = np.nan
    for name in [n for n in mapped if n not in ("SpineF", "SpineM", "SpineL")]:
        markers[name][6] = np.nan  # a frame with three usable markers: too few to trust
    k = cam.K.copy()
    k[0, 2] += 1.0
    k[1, 2] += 1.0  # MATLAB intrinsics are 1-based
    cameras = {"Camera1": {"IntrinsicMatrix": k.T, "RadialDistortion": [0.01, 0.0, 0.0],
                           "TangentialDistortion": [0.001, 0.002], "rotationMatrix": cam.R.T,
                           "translationVector": cam.t, "frame": frame_of},
               "Camera2": {"IntrinsicMatrix": k.T, "RadialDistortion": [0.11], "TangentialDistortion": [0.01, 0.02],
                           "rotationMatrix": cam.R.T, "translationVector": cam.t, "frame": frame_of}}
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), {"cameras": cameras, "mocap": markers})
    return frame_of


def _rat7m_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Camera]:
    cam = Camera.look_at([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], focal=60.0, size=RAT7M_SIZE, name="Camera1")
    mat = tmp_path / "mocap-s1-d1.mat"
    _rat7m_mat(mat, cam)
    videos = tmp_path / "videos"
    _mp4(videos / "camera1-0.mp4", 8, RAT7M_SIZE)
    _mp4(videos / "camera1-105000.mp4", 8, RAT7M_SIZE)
    _mp4(videos / "camera1-500000.mp4", 4, RAT7M_SIZE)  # no mocap frames cover this chunk
    return mat, videos, tmp_path / "out", cam


def test_rat7m_cameras_apply_the_1_based_origin_and_distortion_defaults(tmp_path):
    import scipy.io as sio

    _, _, _, cam = _rat7m_fixture(tmp_path)
    loaded = rat7m_cameras(sio.loadmat(str(tmp_path / "mocap-s1-d1.mat"), squeeze_me=True, struct_as_record=False))
    assert sorted(loaded) == ["Camera1", "Camera2"]
    assert np.allclose(loaded["Camera1"].K, cam.K)  # 1-based MATLAB origin -> 0-based pixels
    assert loaded["Camera1"].size == (1328, 1048)
    assert np.allclose(loaded["Camera2"].dist, [0.11, 0.0, 0.01, 0.02, 0.0])
    assert np.allclose(loaded["Camera1"].center, [0.5, 0.0, 0.0], atol=1e-9)


def test_rat7m_projects_mocap_to_the_documented_splits(tmp_path):
    mat, videos, out, _ = _rat7m_fixture(tmp_path)
    stats = convert_rat7m(mat, videos, out, every=2)
    assert stats == {"train": {"images": 2, "animals": 2}, "val": {"images": 4, "animals": 4}}
    train, val = _read(out, "train.json"), _read(out, "val.json")
    _check_canonical(train)
    _check_canonical(val)
    assert all(im["file_name"].startswith("camera1-0_") for im in train["images"])
    assert all(im["file_name"].startswith("camera1-105000_") for im in val["images"])
    assert [im["file_name"] for im in train["images"]] == [f"camera1-0_{j:05d}.jpg" for j in (0, 2)]
    assert (out / "images" / "camera1-0_00000.jpg").exists()
    assert {(im["height"], im["width"]) for im in val["images"]} == {RAT7M_SIZE[::-1]}

    cam = Camera.from_dict(_read(out, "cameras.json")[0])
    assert cam.name == "Camera1"
    frame_of = np.r_[np.arange(8), 105000 + np.arange(8)]
    markers = {n: np.tile(np.asarray(xyz, float), (len(frame_of), 1)) for n, xyz in RAT7M_MARKERS.items()}
    for arr in markers.values():
        arr[:, 1] += 0.004 * np.arange(len(frame_of))
    left = ["left_shoulder", "left_front_knee", "left_hip", "left_hind_knee"]
    right = ["right_shoulder", "right_front_knee", "right_hip", "right_hind_knee"]
    mid = ["head_center", "spine_upper", "spine_mid", "spine_lower"]
    for doc, start in ((train, 0), (val, 105000)):
        for im, a in zip(doc["images"], doc["annotations"]):
            row = int(np.nonzero(frame_of == start + int(im["file_name"].split("_")[1][:5]))[0][0])
            kp = _canonical_of(a)
            names = [n for n in RAT7M_MARKERS if n in DATASET_MAPPINGS["rat7m"]]
            uv = cam.project(np.stack([markers[n][row] for n in names]))
            got = kp[CANONICAL.indices(DATASET_MAPPINGS["rat7m"][n] for n in names), :2]
            np.testing.assert_allclose(got, uv, atol=1e-6)
            vis = kp[:, 2]
            assert (vis[CANONICAL.indices(left)] == 1).all()  # far-side markers are occluded, not deleted
            assert (vis[CANONICAL.indices(right)] == 2).all()
            assert (vis[CANONICAL.indices(mid)] == 2).all()
            assert a["num_keypoints"] == len(DATASET_MAPPINGS["rat7m"])
            rest = [i for i in range(CANONICAL.num_keypoints)
                    if i not in CANONICAL.indices(DATASET_MAPPINGS["rat7m"].values())]
            assert (kp[rest] == 0).all()  # markers outside the vocabulary are ignored


def test_rat7m_val_starts_none_holds_out_every_chunk(tmp_path):
    mat, videos, out, _ = _rat7m_fixture(tmp_path)
    stats = convert_rat7m(mat, videos, out / "valonly", every=2, val_starts=None)
    assert stats == {"train": {"images": 0, "animals": 0}, "val": {"images": 6, "animals": 6}}
    assert _read(out / "valonly", "train.json")["images"] == []
