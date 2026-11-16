import cv2
import numpy as np

from animalpose.data.degrade import degrade, disk_kernel, motion_kernel, zoom_out
from animalpose.frames import estimate_fps, read_frames


def test_kernels_are_normalised():
    for k in (motion_kernel(9, 0.7), disk_kernel(3)):
        assert abs(k.sum() - 1) < 1e-5 and (k >= 0).all()


def test_degrade_keeps_shape_and_blurs():
    rng = np.random.default_rng(0)
    img = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
    sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    out = [degrade(img, np.random.default_rng(i), animal_px=80) for i in range(20)]
    assert all(o.shape == img.shape and o.dtype == np.uint8 for o in out)
    blur = np.median([cv2.Laplacian(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for o in out])
    assert blur < sharp
    assert np.array_equal(degrade(img, np.random.default_rng(1), 80, strength=0.0), img)


def test_zoom_out_moves_boxes_with_content():
    img = np.zeros((100, 200, 3), np.uint8)
    img[20:40, 50:90] = 255
    out, boxes = zoom_out(img, np.array([[50, 20, 90, 40]], np.float32), np.random.default_rng(3))
    x0, y0, x1, y1 = boxes[0].round().astype(int)
    assert out.shape == img.shape and out[y0 + 1:y1 - 1, x0 + 1:x1 - 1].mean() > 200
    assert (x1 - x0) <= 40 and (y1 - y0) <= 20


def test_read_frames_from_tif_folder(tmp_path):
    for i, ms in enumerate([112, 312, 513, 713]):
        img = np.full((24, 32), 1000 * (i + 1), np.uint16)  # 16-bit grayscale
        cv2.imwrite(str(tmp_path / f"img1030_13_47_27_{ms:03d}.tif"), img)
    frames, fps = read_frames(tmp_path)
    assert len(frames) == 4 and frames[0].shape == (24, 32, 3) and frames[0].dtype == np.uint8
    assert frames[0].mean() < frames[-1].mean()  # one intensity range for the sequence
    assert abs(fps - 5.0) < 0.1 and estimate_fps(sorted(tmp_path.iterdir())) == fps


def test_background_filter_separates_static_objects_from_the_animal():
    from animalpose.pipeline import foreground_fraction, median_background

    frames = []
    for t in range(20):
        f = np.full((60, 80, 3), 200, np.uint8)
        f[10:20, 10:20] = 20  # static cup
        f[40:48, 4 * t % 70:4 * t % 70 + 10] = 20  # moving mouse
        frames.append(f)
    bg = median_background(frames, 16)
    assert foreground_fraction(frames[5], bg, (10, 10, 20, 20)) < 0.05
    assert foreground_fraction(frames[5], bg, (20, 40, 30, 48)) > 0.5


def test_blob_candidates_find_the_animal_and_outrank_the_detector():
    from animalpose.pipeline import BlobProposer, median_background, merge_candidates

    frames = []
    for t in range(20):
        f = np.full((60, 80, 3), 200, np.uint8)
        f[10:20, 10:20] = 20  # static cup: never a candidate
        f[40:48, 4 * t % 70:4 * t % 70 + 10] = 20  # moving mouse
        frames.append(f)
    bg = median_background(frames, 16)
    prop = BlobProposer(bg, diff=25.0, calibrate=4)
    boxes = [prop(f)[0] for f in frames]
    assert all(len(b) == 1 for b in boxes)  # the animal is proposed in every frame, before and after calibration
    x0, y0, x1, y1 = boxes[5][0]
    assert 20 <= (x0 + x1) / 2 <= 30 and 40 <= (y0 + y1) / 2 <= 48  # the mouse, never the cup
    assert prop.area is not None and prop.area[0] < prop.area[1]  # plausible size learned from the recording
    # A blob box outranks a network box on the same animal; a distant network box survives next to it.
    net = np.vstack([boxes[5], np.array([10, 10, 20, 20], np.float32)])
    merged, scores = merge_candidates(net, np.array([0.6, 0.4], np.float32), boxes[5], np.array([0.9], np.float32), 0.5)
    assert len(merged) == 2 and abs(scores[0] - 0.9) < 1e-6
    assert np.allclose(merged[1], [10, 10, 20, 20])


def test_find_sources_split_and_sampled_frames(tmp_path):
    from animalpose.frames import find_sources, is_held_out

    seq = tmp_path / "a" / "cam1"
    seq.mkdir(parents=True)
    for i in range(5):
        cv2.imwrite(str(seq / f"f{i:03d}.png"), np.full((8, 8), 40 * i, np.uint8))
    vw = cv2.VideoWriter(str(tmp_path / "b.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 12))
    for i in range(6):
        vw.write(np.full((12, 16, 3), 30 * i, np.uint8))
    vw.release()
    srcs = {s.kind: s for s in find_sources(tmp_path)}
    assert set(srcs) == {"images", "video"} and srcs["images"].n_frames == 5
    got = srcs["images"].frames([4, 1])
    assert [t for t, _ in got] == [4, 1] and got[0][1].shape == (8, 8, 3) and got[0][1].mean() > got[1][1].mean()
    assert len(srcs["video"].frames([0, 5])) == 2
    assert [is_held_out(s, tmp_path) for s in find_sources(tmp_path)] == [is_held_out(s, tmp_path)
                                                                           for s in find_sources(tmp_path)]


def test_flip_consistency_maps_back_and_swaps_sides():
    from animalpose.selftrain import consistent_keypoints
    from animalpose.skeleton import CANONICAL

    class FakePipe:  # anatomy-consistent: left points on the box's left quarter, right on its right quarter
        def estimate(self, img, boxes):
            kp = np.zeros((len(boxes), CANONICAL.num_keypoints, 2), np.float32)
            for i, (x0, y0, x1, y1) in enumerate(boxes):
                for k, p in enumerate(CANONICAL.keypoints):
                    fx = {"left": 0.25, "right": 0.75}.get(p.side, 0.5)
                    kp[i, k] = (x0 + fx * (x1 - x0), (y0 + y1) / 2)
            return kp, np.ones(kp.shape[:2], np.float32), None

    img = np.zeros((50, 100, 3), np.uint8)
    boxes = np.array([[10, 10, 50, 30]], np.float32)
    kp, ok, _ = consistent_keypoints(FakePipe(), img, boxes, 0.5, 0.05)
    # Mirroring moves every side point to the other quarter; with the left/right swap the labels agree
    # only up to the 1 px pixel-centre offset of the flip.
    assert ok.all() and abs(kp[0, CANONICAL.index("nose"), 0] - 29.5) < 1e-4
