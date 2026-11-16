"""Multi-view geometry, the tracker and the inference pipeline: the paths that are
not the happy path.

Checks the degenerate ends of the camera maths (zero and 180-degree rotations, the
projection matrix), what calibration promises when it has too little data, that identity
fusion needs frames in common and that left/right resolution repairs a mirrored view
before triangulation, that fundamental matrices work in either direction and that a view
which does not see a pair cannot anchor it, the tracker's no-appearance costs and its
tentative/lost/max_animals rules, and the pipeline's CSV, background, candidate,
checkpoint, short-track and 3D-behaviour paths.
"""

import csv

import cv2
import numpy as np
import pytest

from animalpose.behavior.catalog import NAMES as CATALOG_NAMES
from animalpose.data.synthetic import make_cameras, render_video, simulate
from animalpose.geometry import box_iou
from animalpose.multiview.calibration import (
    calibrate_dlt,
    calibrate_intrinsics_checkerboard,
    solve_extrinsics,
)
from animalpose.multiview.camera import Camera, rodrigues, rotation_to_rvec
from animalpose.multiview.fusion import ViewTrack, epipolar_cost, triangulate_identities
from animalpose.multiview.symmetry import (
    F_SHARED_X,
    bilateral_pairs,
    fundamental_from_cameras,
    resolve_left_right,
)
from animalpose.pipeline import (
    BlobProposer,
    MultiViewPipeline,
    PipelineConfig,
    PosePipeline,
    TrackResult,
    VideoResult,
    foreground_fraction,
    largest_moving_blob,
    median_background,
    merge_candidates,
)
from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS
from animalpose.tracking.tracker import (
    ColorHistogramEmbedder,
    Detection,
    MultiAnimalTracker,
    TrackerConfig,
    _cos_dist,
)
from animalpose.train.checkpoint import save_checkpoint
from animalpose.train.loops import train_behavior, train_detector, train_pose, train_temporal

TINY = {"name": "tiny", "width": 8, "out_channels": 16}
K = CANONICAL.num_keypoints


@pytest.fixture(scope="module")
def nets(tmp_path_factory):
    """A few steps of tiny training: the pose head has to be confident enough that
    cross-view fusion and triangulation see anything at all."""
    out = tmp_path_factory.mktemp("mv-runs")
    train = {"steps": 12, "batch_size": 4, "log_every": 0, "workers": 0}
    data = {"n_sims": 3, "frames": 40, "image_size": [160, 128]}
    det, _ = train_detector({"device": "cpu", "output": str(out / "det.pt"), "model": {"backbone": TINY},
                             "data": {**data, "input_size": [128, 160]}, "train": train})
    pose, _ = train_pose({"device": "cpu", "output": str(out / "pose.pt"), "model": {"backbone": TINY, "embed_dim": 16},
                          "data": {**data, "crop_size": 64}, "train": {**train, "steps": 40, "batch_size": 8}})
    loop_data = {"window": 32, "pool_sims": 2, "pool_frames": 120}
    trained = {}
    for dims, kind in ((2, "t2"), (2, "b2"), (3, "t3"), (3, "b3")):
        model_cfg = {"dims": dims, "hidden": 16}
        cfg = {"device": "cpu", "model": model_cfg, "data": loop_data, "train": train}
        trainer = train_temporal if kind.startswith("t") else train_behavior
        trained[kind], _ = trainer(cfg)
    return {"det": det, "pose": pose, **trained}


def _pipe(nets, **cfg_kw):
    cfg = PipelineConfig(det_size=(128, 160), crop_size=64, temporal_window=32,
                         tracker=TrackerConfig(max_animals=2), **cfg_kw)
    return PosePipeline(nets["det"], nets["pose"], nets.get("t2"), nets.get("b2"), cfg, device="cpu")


def _scene(n_animals=2, frames=60, views=("top", "side", "angled"), size=(640, 480), seed=3):
    sim = simulate(n_animals, frames, "rat", seed=seed)
    return sim, make_cameras(sim.arena_cm, views, size=size)


# --------------------------------------------------------------------------------- camera
def test_zero_and_half_turn_rotations_still_round_trip():
    np.testing.assert_allclose(rodrigues(np.zeros(3)), np.eye(3))  # no angle: no rotation, no division by zero
    np.testing.assert_allclose(rotation_to_rvec(np.eye(3)), np.zeros(3))
    generic = np.array([0.3, -0.7, 1.1])
    np.testing.assert_allclose(rotation_to_rvec(rodrigues(generic)), generic, atol=1e-9)

    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8], [1.0, 1.0, 1.0]):
        rvec = np.array(axis) / np.linalg.norm(axis) * np.pi
        rot = rodrigues(rvec)
        back = rotation_to_rvec(rot)
        assert abs(np.linalg.norm(back) - np.pi) < 1e-6  # the angle survives
        np.testing.assert_allclose(rodrigues(back), rot, atol=1e-9)  # the axis sign may flip, the rotation cannot
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0)


def test_the_projection_matrix_is_the_camera_that_projects():
    cam = Camera.look_at([50.0, 40.0, 60.0], [0.0, 0.0, 0.0], focal=600.0, size=(320, 240))
    assert cam.P.shape == (3, 4)
    np.testing.assert_allclose(cam.P, cam.K @ np.hstack([cam.R, cam.t[:, None]]))
    X = np.array([[1.0, 2.0, 3.0], [10.0, -5.0, 4.0]])
    hom = cam.P @ np.c_[X, np.ones(len(X))].T
    np.testing.assert_allclose((hom[:2] / hom[2]).T, cam.project(X), atol=1e-9)


# --------------------------------------------------------------------------------- calibration
def test_dlt_refuses_fewer_than_six_correspondences():
    X = np.random.default_rng(0).uniform(-10, 10, (5, 3))
    with pytest.raises(ValueError, match="at least 6 correspondences"):
        calibrate_dlt(X, X[:, :2])


def _board_views(pattern=(7, 5), px=30, size=(520, 400), focal=600.0):
    """Six rendered views of a planar board seen by a known pinhole camera.

    ``pattern`` counts inner corners; the rendered board has one more square in each
    direction, ringed by a light margin, which is what the OpenCV finder insists on.
    """
    sx, sy = pattern[0] + 1, pattern[1] + 1
    grid = (np.indices((sy, sx)).sum(0) % 2).astype(np.uint8)
    board = np.repeat(np.repeat(grid, px, 0), px, 1) * 255
    board = np.pad(board, px, constant_values=255)
    corners = np.float32([[0, 0], [board.shape[1], 0], [board.shape[1], board.shape[0]], [0, board.shape[0]]])
    plane = lambda u, v: np.array([(u - px) / px, (v - px) / px, 0.0])  # noqa: E731
    center = np.array([sx / 2.0, sy / 2.0, 0.0])
    views = []
    for az, el in ((-25, 45), (20, 58), (-5, 35), (35, 68), (-40, 62), (8, 80)):
        az, el = np.radians(az), np.radians(el)
        for dist in np.arange(6.0, 60.0, 0.5):  # back off until the whole board is inside the frame
            eye = center + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            cam = Camera.look_at(eye, center, focal=focal, size=size)
            projected = np.float32([cam.project(plane(u, v)) for u, v in corners])
            if ((projected > 20) & (projected < np.array(size) - 20)).all():
                break
        gray = cv2.warpPerspective(board, cv2.getPerspectiveTransform(corners, projected), size, borderValue=255)
        views.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return views


def test_checkerboard_calibration_recovers_the_rendered_intrinsics():
    K, dist, rms = calibrate_intrinsics_checkerboard(_board_views(), pattern=(7, 5))
    assert K.shape == (3, 3) and dist.shape == (5,) and np.isfinite(dist).all()
    assert rms < 0.5 and K[2, 2] == pytest.approx(1.0)
    assert abs(K[0, 0] - 600) < 30 and abs(K[1, 1] - 600) < 30  # the focal length it was rendered with
    assert abs(K[0, 2] - 259.5) < 10 and abs(K[1, 2] - 199.5) < 10  # the principal point of that camera
    assert abs(dist[0]) < 0.05 and np.abs(dist[2:4]).max() < 0.01  # a pinhole lens: no real distortion


def test_checkerboard_frames_without_a_board_are_skipped_then_refused():
    views = _board_views()
    blank = cv2.cvtColor(np.full((400, 520), 128, np.uint8), cv2.COLOR_GRAY2BGR)
    mixed = [views[0], blank, blank, views[1], views[2], blank]  # three boards left: exactly enough
    _K, _dist, rms = calibrate_intrinsics_checkerboard(mixed, pattern=(7, 5))
    assert rms < 0.5
    with pytest.raises(ValueError, match="found in only 0 images; need >= 3"):
        calibrate_intrinsics_checkerboard([blank] * 4, pattern=(7, 5))


def test_solve_extrinsics_reports_a_pnp_failure(monkeypatch):
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **kw: (False, np.zeros(3), np.zeros(3)))
    with pytest.raises(RuntimeError, match="solvePnP failed"):
        solve_extrinsics(np.eye(3), np.zeros(5), np.zeros((6, 3)), np.zeros((6, 2)))


# --------------------------------------------------------------------------------- fusion / symmetry
def test_epipolar_cost_needs_the_two_tracks_to_share_frames():
    sim, cams = _scene(1, 30, ("top", "side"))
    pts = np.stack([cams[v].project(sim.kpts3d[0, :30]) for v in range(2)])
    a = ViewTrack(0, 1, np.arange(30), pts[0], np.ones((30, K)))
    b = ViewTrack(1, 2, np.arange(30) + 100, pts[1], np.ones((30, K)))
    assert np.isinf(epipolar_cost(a, b, cams[0], cams[1]))  # never co-visible: no evidence at all
    same = ViewTrack(1, 2, np.arange(30), pts[1], np.ones((30, K)))
    assert epipolar_cost(a, same, cams[0], cams[1]) < 1.0  # the same animal, consistent cameras


def test_left_right_resolution_repairs_a_mirrored_view_before_triangulation():
    sim, cams = _scene(1, 24, ("top", "side"), seed=7)
    frames = np.arange(24)
    uv = np.stack([c.project(sim.kpts3d[0]) for c in cams])
    uv[1] = uv[1][:, CANONICAL.flip_index]  # the side view cannot tell the near paw from the far one
    conf = np.ones((2, 24, K), np.float32)
    conf[1] *= 0.6  # the view that mirrors the labels is also the shakier one
    tracks = [ViewTrack(v, 0, frames, uv[v], conf[v]) for v in range(2)]
    ids = {(v, 0): 0 for v in range(2)}
    kw = dict(conf_thresh=0.3, reproj_thresh=1e9)
    naive = triangulate_identities(tracks, cams, ids, 24, **kw)[0]["keypoints3d"]
    fixed = triangulate_identities(tracks, cams, ids, 24, resolve_left_right=True, **kw)[0]["keypoints3d"]
    truth = sim.kpts3d[0]
    assert np.nanmax(np.linalg.norm(naive - truth, axis=-1)) > 1.0  # two views cannot outvote a mirrored one
    assert np.nanmax(np.linalg.norm(fixed - truth, axis=-1)) < 0.1  # relabelling first recovers the animal


def test_fundamental_matrices_work_whichever_way_they_are_given():
    sim, cams = _scene(1, 20, ("top", "side"), seed=2)
    uv = np.stack([c.project(sim.kpts3d[0, 10]) for c in cams])
    conf = np.full((2, K), 0.9)
    uv[1] = uv[1][CANONICAL.flip_index]
    forward = {(0, 1): fundamental_from_cameras(cams[0], cams[1])}
    backward = {(1, 0): fundamental_from_cameras(cams[1], cams[0])}
    _kp_a, _cf_a, swapped_fwd = resolve_left_right(uv, conf, forward)
    kp_b, _cf_b, swapped_bwd = resolve_left_right(uv, conf, backward)
    assert swapped_bwd[1].sum() == len(bilateral_pairs()) and (swapped_fwd == swapped_bwd).all()
    np.testing.assert_allclose(kp_b[1], uv[1][CANONICAL.flip_index], atol=1e-9)

    untouched, conf_out, none_swapped = resolve_left_right(uv, conf, {})
    assert not none_swapped.any() and np.array_equal(untouched, uv) and np.array_equal(conf_out, conf)


def test_a_view_that_does_not_see_the_pair_cannot_anchor_the_others():
    pair = [(0, 1)]
    seen = np.array([[[0.0, 5.0], [10.0, 5.0]], [[10.0, 5.0], [0.0, 5.0]], [[10.0, 5.0], [0.0, 5.0]]])
    f = {(0, 1): F_SHARED_X}  # only views 0 and 1 have geometry between them
    low = np.array([0.05, 0.9, 0.9])[:, None] * np.ones((3, 2))
    # View 0 never sees the pair, so it cannot be the anchor and nothing can be decided.
    kept, _cf, swapped = resolve_left_right(seen.copy(), low, f, pairs=pair)
    assert not swapped.any() and np.array_equal(kept, seen)
    # Trust the same view and it anchors: view 1, which disagrees with it, is corrected instead.
    trusted = np.full((3, 2), 0.9)
    fixed, _cf, swapped = resolve_left_right(seen.copy(), trusted, f, pairs=pair)
    assert swapped[1, 0] and not swapped[0, 0] and not swapped[2, 0]
    np.testing.assert_allclose(fixed[1], seen[1, ::-1])


# --------------------------------------------------------------------------------- tracker
BOX_A = np.array([10.0, 10.0, 30.0, 30.0])
BOX_B = np.array([120.0, 90.0, 150.0, 120.0])


def test_a_missing_embedding_drops_the_appearance_term_from_the_cost():
    assert np.isnan(_cos_dist(None, np.array([1.0, 0.0]))) and np.isnan(_cos_dist(np.array([1.0, 0.0]), None))
    tracker = MultiAnimalTracker(TrackerConfig())
    bare = tracker._new_track(Detection(BOX_A, 0.9))
    pred = np.array([10.0, 10.0, 30.0, 30.0])
    cost = tracker._pair_cost(bare, Detection(BOX_B, 0.9), pred, 0.4)
    assert cost == pytest.approx(0.6)  # only the IoU term is left, so it is the whole cost
    look = tracker._new_track(Detection(BOX_A, 0.9, embedding=np.array([1.0, 0.0])))
    other = Detection(BOX_B, 0.9, embedding=np.array([0.0, 1.0]))  # orthogonal: cosine distance 0.5
    cost = tracker._pair_cost(look, other, pred, 0.4)
    assert cost == pytest.approx((0.5 * 0.6 + 0.2 * 0.5) / 0.7)  # the appearance term re-enters


def test_tracker_config_overrides_are_applied():
    tracker = MultiAnimalTracker(TrackerConfig(), max_animals=3, n_init=1, high_thresh=0.1)
    assert (tracker.cfg.max_animals, tracker.cfg.n_init, tracker.cfg.high_thresh) == (3, 1, 0.1)
    assert tracker.cfg.w_iou == TrackerConfig().w_iou  # everything else keeps its default


def test_without_appearance_only_a_nearby_reappearance_counts():
    tracker = MultiAnimalTracker(TrackerConfig())
    lost = tracker._new_track(Detection(BOX_A, 0.9))  # no embedding: an empty gallery
    lost.state = "lost"
    same_place = Detection(np.array([2.0, 2.0, 22.0, 22.0]), 0.9)
    assert tracker._reid_cost(lost, same_place) == pytest.approx(0.1)  # zero spatial penalty
    lost.time_since_update = 0
    far = Detection(np.array([60.0, 60.0, 80.0, 80.0]), 0.9)
    immediately = tracker._reid_cost(lost, far)
    lost.time_since_update = 40
    after_a_while = tracker._reid_cost(lost, far)
    assert immediately > 0.5 > after_a_while  # the spatial prior widens with the time lost


def test_a_tentative_track_that_misses_once_is_dropped():
    tracker = MultiAnimalTracker(TrackerConfig(n_init=3))
    tracker.update([Detection(BOX_A, 0.9)])  # first frame: trusted immediately
    second = tracker.update([Detection(BOX_A, 0.9), Detection(BOX_B, 0.9)])
    assert second == {0: 0}  # a second-old animal is only tentative, so it is not reported yet
    assert [t.state for t in tracker.tracks] == ["confirmed", "tentative"]
    third = tracker.update([Detection(BOX_A, 0.9)])
    assert set(third) == {0} and [t.id for t in tracker.tracks] == [0]  # one miss and it is gone
    assert tracker._next_id == 2  # its id is not reused


def test_a_lost_track_is_forgotten_after_max_lost_frames_but_not_with_a_known_cohort():
    strict = MultiAnimalTracker(TrackerConfig(max_lost=1))
    strict.update([Detection(BOX_A, 0.9)])
    strict.update([])
    assert strict.tracks[0].state == "lost"
    strict.update([])
    assert strict.tracks == []  # 2 frames > max_lost: no longer re-identifiable

    cohort = MultiAnimalTracker(TrackerConfig(max_animals=1, max_lost=1))
    cohort.update([Detection(BOX_A, 0.9)])
    for _ in range(5):
        cohort.update([])
    assert [t.id for t in cohort.tracks] == [0] and cohort.tracks[0].state == "lost"


def test_a_full_cohort_never_creates_an_identity():
    tracker = MultiAnimalTracker(TrackerConfig(max_animals=1))
    tracker.update([Detection(BOX_A, 0.9)])
    assert tracker.update([Detection(BOX_B, 0.9)]) == {}  # a second animal is refused, not created
    assert [t.id for t in tracker.tracks] == [0] and tracker.tracks[0].state == "lost"
    assert set(tracker.update([Detection(BOX_B, 0.9)])) == {0}  # re-identified as the one animal
    for _ in range(3):
        assert set(tracker.update([Detection(BOX_B, 0.9)])) <= {0}
    assert tracker._next_id == 1 and [t.id for t in tracker.tracks] == [0]  # no identity was ever minted


def test_the_colour_embedder_returns_a_neutral_vector_for_a_box_outside_the_image():
    emb = ColorHistogramEmbedder()
    img = np.full((32, 40, 3), 120, np.uint8)
    outside = emb(img, np.array([1000.0, 1000.0, 1010.0, 1010.0]))
    assert outside.shape == (73,) and np.allclose(outside, 1.0 / np.sqrt(73))
    assert np.isclose(np.linalg.norm(outside), 1.0)  # still usable as an embedding, and never the best match
    inside = emb(img, np.array([0.0, 0.0, 30.0, 30.0]))
    assert np.isclose(np.linalg.norm(inside), 1.0) and not np.allclose(inside, outside)


# --------------------------------------------------------------------------------- pipeline output
def _moving_clip(n=10):
    frames = []
    for t in range(n):
        f = np.full((60, 80, 3), 200, np.uint8)
        f[10:20, 10:20] = 20  # a cup that never moves
        x = 4 * t % 60
        f[40:48, x:x + 10] = 20  # the mouse
        frames.append(f)
    return frames


def _track_result(tid, frames):
    frames = np.asarray(frames)
    n = len(frames)
    boxes = np.stack([2.0 * frames, np.full(n, 10.0), 2.0 * frames + 30.0, np.full(n, 40.0)], 1).astype(np.float32)
    kp = boxes[:, None, :2] + (np.arange(K)[:, None] % 5) * 0.5
    return TrackResult(tid, frames, boxes, kp.astype(np.float32), np.full((n, K), 0.9, np.float32))


def test_behavior_csv_reports_the_top_k_and_the_whole_catalogue(tmp_path):
    frames = np.array([0, 2, 3, 7])
    rng = np.random.default_rng(0)
    full = rng.random((len(frames), len(CATALOG_NAMES))).astype(np.float64)
    full /= full.sum(1, keepdims=True)
    times = np.arange(8) * 0.5
    result = VideoResult({0: _track_result(0, frames), 1: _track_result(1, np.arange(4))},
                         8, CATALOG_NAMES, times=times)
    result.tracks[0].behavior_full = full
    k = 3
    result.to_behavior_csv(tmp_path / "beh.csv", k=k)
    rows = list(csv.reader((tmp_path / "beh.csv").read_text().splitlines()))
    header, rows = rows[0], rows[1:]
    assert header[:3 + 2 * k] == ["frame", "time_s", "track"] + [f"top{i + 1}{s}"
                                                                 for i in range(k) for s in ("", "_p")]
    assert len(header) == 3 + 2 * k + len(CATALOG_NAMES) and header[-1] == f"p_{CATALOG_NAMES[-1]}"
    assert rows and all(r[2] == "0" for r in rows)  # the track without probabilities contributes nothing
    assert len(rows) == len(frames)
    order = np.argsort(-full, axis=1)
    for i, row in enumerate(rows):
        assert row[0] == str(frames[i]) and row[1] == f"{times[frames[i]]:.4f}"
        assert [row[2 * j + 3] for j in range(k)] == [CATALOG_NAMES[j_] for j_ in order[i, :k]]
        assert [float(row[2 * j + 4]) for j in range(k)] == pytest.approx(full[i][order[i, :k]], abs=5e-4)
        assert np.allclose([float(v) for v in row[3 + 2 * k:]], full[i], atol=5e-5)


def test_a_degenerate_or_outside_box_has_no_foreground():
    frames = _moving_clip(6)
    bg = median_background(frames, 6)
    assert foreground_fraction(frames[3], bg, (30, 40, 10, 48)) == 0.0  # x1 <= x0
    assert foreground_fraction(frames[3], bg, (40, 50, 50, 40)) == 0.0  # y1 <= y0
    assert foreground_fraction(frames[3], bg, (500.0, 500.0, 520.0, 520.0)) == 0.0  # off the frame
    x = 4 * 3 % 60
    assert foreground_fraction(frames[3], bg, (x, 40, x + 10, 48)) > 0.5  # the mouse is real foreground
    assert foreground_fraction(frames[3], bg, (10, 10, 20, 20)) < 0.05  # the cup is not


def test_the_blob_area_window_rejects_an_implausible_size():
    frames = _moving_clip(12)
    bg = median_background(frames, 12)
    box, area = largest_moving_blob(frames[3], bg, 25.0, area=(2000.0, 5000.0))
    assert box is None and 0.0 < area < 2000.0  # rejected, but its size is still reported
    loose, same = largest_moving_blob(frames[3], bg, 25.0, area=(10.0, 1000.0))
    assert loose is not None and same == area
    prop = BlobProposer(bg, diff=25.0, area=(2000.0, 5000.0))
    boxes, scores = prop(frames[3])
    assert boxes.shape == (0, 4) and scores.shape == (0,)  # the size window starves the proposal
    still, quiet = prop(bg)  # nothing moves at all
    assert still.shape == (0, 4) and quiet.shape == (0,)
    accepted, accepted_scores = BlobProposer(bg, diff=25.0, area=(10.0, 1000.0))(frames[3])
    assert accepted.shape == (1, 4) and np.allclose(accepted_scores, [0.9])  # the same blob, inside the window


def test_merge_candidates_caps_at_max_det_and_survives_no_boxes():
    boxes = np.stack([np.arange(0, 60, 10.0), np.zeros(6), np.arange(8, 68, 10.0), np.full(6, 8.0)], 1)
    scores = np.array([0.2, 0.9, 0.4, 0.7, 0.1, 0.5], np.float32)
    kept, kept_scores = merge_candidates(boxes, scores, np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                                         iou_thresh=0.5, max_det=3)
    assert np.allclose(kept_scores, [0.9, 0.7, 0.5])  # the three best, in order
    assert np.allclose(kept, boxes[[1, 3, 5]])
    empty, empty_scores = merge_candidates(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                                          np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    assert empty.shape == (0, 4) and empty_scores.shape == (0,) and empty.dtype == np.float32


def test_blob_candidates_need_a_median_background(nets):
    assert _pipe(nets).proposals(None) is None  # the network alone needs nothing
    pipe = _pipe(nets, detector_candidates=("blob",))
    with pytest.raises(ValueError, match="blob candidates need a median background"):
        pipe.proposals(None)
    frames = _moving_clip(12)
    bg = median_background(frames, 12)
    prop = pipe.proposals(bg)
    assert isinstance(prop, BlobProposer) and prop.area is None and prop.calibrate == 64

    boxes, scores = pipe.detect_candidates(frames[3], propose=lambda img: (np.array([[20.0, 40.0, 30.0, 48.0]],
                                                                                   np.float32),
                                                                          np.array([1.0], np.float32)))
    blob = np.array([20.0, 40.0, 30.0, 48.0])
    assert np.isclose(scores[0], 1.0) and np.allclose(scores, np.sort(scores)[::-1])
    assert int((box_iou(boxes, blob[None])[:, 0] > 0.9).sum()) == 1  # the proposal, exactly once
    for i in range(len(boxes)):  # and no two survivors overlap
        for j in range(i + 1, len(boxes)):
            assert box_iou(boxes[i:i + 1], boxes[j:j + 1])[0, 0] < pipe.cfg.merge_iou


def test_pipeline_is_rebuilt_from_checkpoints_alone(tmp_path, nets):
    save_checkpoint(tmp_path / "d.pt", "detector", nets["det"], {"backbone": TINY})
    save_checkpoint(tmp_path / "p.pt", "pose", nets["pose"], {"backbone": TINY, "embed_dim": 16})
    save_checkpoint(tmp_path / "t.pt", "temporal", nets["t2"], {"dims": 2, "hidden": 16})
    save_checkpoint(tmp_path / "b.pt", "behavior", nets["b2"], {"dims": 2, "hidden": 16})
    pipe = PosePipeline.from_checkpoints(str(tmp_path / "d.pt"), str(tmp_path / "p.pt"), device="cpu")
    assert pipe.temporal is None and pipe.behavior is None
    assert pipe.pose.model.num_keypoints == K and pipe.detector.model.stride == 4
    full = PosePipeline.from_checkpoints(str(tmp_path / "d.pt"), str(tmp_path / "p.pt"), str(tmp_path / "t.pt"),
                                         str(tmp_path / "b.pt"), PipelineConfig(crop_size=48), device="cpu")
    assert full.temporal is not None and full.behavior is not None and full.cfg.crop_size == 48
    with pytest.raises(ValueError, match="no checkpoint at"):
        PosePipeline.from_checkpoints(str(tmp_path / "missing.pt"), str(tmp_path / "p.pt"), device="cpu")


def test_the_background_filter_keeps_the_animal_and_drops_the_furniture(nets):
    frames = _moving_clip(10)

    def dets(t, _img):
        x = 4 * t % 60
        return np.array([[10.0, 10.0, 20.0, 20.0], [x, 40.0, x + 10.0, 48.0]]), np.array([0.9, 0.9])

    plain = _pipe(nets).track(frames, dets)
    assert len(plain.tracks) == 2  # cup and mouse both tracked when everything is trusted
    filtered = _pipe(nets, background_filter=True).track(frames, dets)
    assert len(filtered.tracks) == 1
    track = next(iter(filtered.tracks.values()))
    assert np.nanmedian(track.boxes[:, 1]) >= 39.0  # the mouse survived, the cup never became a track
    assert track.frames.min() == 0 and len(track.frames) >= 5


def test_refine_leaves_tracks_shorter_than_min_len_alone(nets):
    result = VideoResult({0: _track_result(0, np.arange(12)), 1: _track_result(1, np.arange(3))}, 12)
    out = _pipe(nets).refine(result, min_len=5, fps=30.0)
    short, long = out.tracks[1], out.tracks[0]
    assert short.smoothed is None and short.imputed is None and short.behavior is None
    assert short.behavior_probs is None and short.behavior_full is None
    assert long.smoothed.shape == long.keypoints.shape and np.isfinite(long.smoothed).all()
    assert long.behavior_probs.shape == (12, len(nets["b2"].classes))
    assert long.behavior_full.shape == (12, len(CATALOG_NAMES)) and long.behavior_full.sum(1).std() < 1e-6
    assert np.array_equal(long.behavior, long.behavior_full.argmax(-1))
    assert out.behaviors == CATALOG_NAMES and len(out.behavior_info) == len(CATALOG_NAMES)
    assert any(entry["modelled"] for entry in out.behavior_info)
    assert out.times is not None and len(out.times) == 12


def test_behavior_keypoints_restrict_what_the_classifier_may_read(nets):
    wild, clean = _track_result(0, np.arange(40)), _track_result(0, np.arange(40))
    keep = CANONICAL.mask(list(DATASET_MAPPINGS["homecage_dlc"].values()))
    assert 0 < keep.sum() < K
    wild.keypoints[:, ~keep] = 1.0e4  # points the mapping does not define, read from an unrelated source
    clean.keypoints[:, ~keep] = 0.0
    times = np.arange(40) / 30.0
    everywhere = _pipe(nets, behavior_conf=0.5)._behavior_probs(wild, times)
    restricted = _pipe(nets, behavior_conf=0.5, behavior_keypoints="homecage_dlc")._behavior_probs(wild, times)
    zeroed = _pipe(nets, behavior_conf=0.5, behavior_keypoints="homecage_dlc")._behavior_probs(clean, times)
    assert np.allclose(restricted, zeroed, atol=1e-6)  # points the mapping lacks are dropped, not read
    assert np.abs(everywhere - restricted).max() > 1e-4  # and without the restriction they poison the input
    assert np.allclose(restricted.sum(1), 1.0, atol=1e-4)


def test_the_multi_view_run_can_score_behaviour_in_three_dimensions(nets):
    sim = simulate(2, 12, "rat", seed=11)
    cams = make_cameras(sim.arena_cm, ("top", "side"), size=(160, 128))
    views = [render_video(sim, c, seed=i) for i, c in enumerate(cams)]
    fns = [lambda t, img, fr=fr: (fr[t].boxes[fr[t].present], np.full(int(fr[t].present.sum()), 0.9))
           for fr in views]
    # kp_conf_thresh=0: 40 steps of a tiny pose head is confident but far from accurate, so only
    # the geometry is trusted here.
    mv = MultiViewPipeline(_pipe(nets, embedder="color", kp_conf_thresh=0.0), cams, temporal3d=nets["t3"],
                           behavior3d=nets["b3"], fusion_max_cost=1e9, reproj_thresh=1e9, max_animals=2)
    res = mv.run([[f.image for f in frames] for frames in views], fns)
    scored = [a for a in res["animals"].values() if np.isfinite(a["keypoints3d"]).any()]
    assert scored  # at least one identity was seen by both cameras
    for animal in scored:
        n = len(animal["keypoints3d"])
        assert "smoothed3d" in animal  # with a 3D temporal model the classifier reads the smoothed tracks
        assert animal["behavior_probs"].shape == (n, len(nets["b3"].classes))
        assert np.allclose(animal["behavior_probs"].sum(1), 1.0, atol=1e-4)
        assert np.array_equal(animal["behavior"], animal["behavior_probs"].argmax(-1))
        assert animal["behavior"].shape == (n,)
