import numpy as np

from animalpose.data.synthetic import make_cameras, render_video, simulate
from animalpose.eval.metrics import tracking_metrics
from animalpose.tracking.kalman import BoxKalman
from animalpose.tracking.tracker import ColorHistogramEmbedder, Detection, MultiAnimalTracker, TrackerConfig


def test_kalman_tracks_constant_velocity():
    kf = BoxKalman(np.array([0, 0, 10, 10.0]))
    for t in range(1, 15):
        kf.predict()
        kf.update(np.array([2 * t, t, 2 * t + 10, t + 10.0]))
    pred = kf.predict()
    np.testing.assert_allclose(pred, [30, 15, 40, 25], atol=0.5)


def _run(frames, drop=None, cfg=None):
    emb = ColorHistogramEmbedder()
    tracker = MultiAnimalTracker(cfg or TrackerConfig())
    gt_ids, gt_boxes, pr_ids, pr_boxes = [], [], [], []
    for t, f in enumerate(frames):
        keep = [a for a in np.nonzero(f.present)[0] if not (drop and drop(t, a))]
        dets = [Detection(f.boxes[a].astype(float), 0.9, embedding=emb(f.image, f.boxes[a])) for a in keep]
        out = tracker.update(dets)
        gt_ids.append(np.array(keep, int)), gt_boxes.append(f.boxes[keep])
        pr_ids.append(np.array(list(out), int)), pr_boxes.append(np.array([dets[j].box for j in out.values()]))
    return tracking_metrics(gt_ids, gt_boxes, pr_ids, pr_boxes)


def test_tracker_keeps_identities_on_synthetic_video():
    sim = simulate(3, 150, "mouse", seed=5, coats=["black", "white", "brown"])
    frames = render_video(sim, make_cameras(sim.arena_cm, ("top",))[0], seed=0)
    m = _run(frames)
    assert m["idf1"] > 0.9, m
    assert m["id_switches"] <= 2, m


def test_reidentification_after_long_occlusion():
    sim = simulate(2, 160, "mouse", seed=6, coats=["black", "white"])
    frames = render_video(sim, make_cameras(sim.arena_cm, ("top",))[0], seed=0)
    hidden = lambda t, a: a == 1 and 40 <= t < 110  # noqa: E731 - animal 1 vanishes for 70 frames
    m = _run(frames, hidden, TrackerConfig(max_animals=2))
    assert m["n_pred_ids"] == 2, m
    assert m["id_switches"] == 0, m


def test_pair_identity_carries_confident_labels_through_ambiguous_frames():
    from animalpose.temporal.pairs import resolve_pair_identity

    t = np.arange(40, dtype=np.float32)
    left = np.stack([t, np.zeros_like(t)], 1)  # left paw sweeps forward
    right = np.stack([np.full_like(t, 20.0), np.full_like(t, 12.0)], 1)  # right paw planted
    kpts = np.stack([left, right], 1).copy()
    kpts[10:] = kpts[10:, ::-1]  # per-frame decoding swaps the labels from frame 10 on...
    margin = np.zeros((40, 1), np.float32)
    margin[:10] = 1.0  # ...after confident correct frames
    margin[10:] = 0.05  # with weak (wrong) evidence for the swapped labels
    kpts[25] = np.nan  # a dropped frame does not break the chain
    fixed, swapped = resolve_pair_identity(kpts, margin, [(0, 1)], scale=100.0)
    assert not swapped[:10].any() and swapped[10:25].all() and swapped[26:].all()
    np.testing.assert_allclose(fixed[30, 0], left[30])
    # Without temporal support the evidence alone decides.
    _, swapped = resolve_pair_identity(kpts[10:20], -np.ones((10, 1)), [(0, 1)], scale=100.0)
    assert swapped.all()
