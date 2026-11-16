"""Every metric in :mod:`animalpose.eval.metrics` against hand-computed arithmetic.

Each check pins a value that follows from the definition (AP by all-point interpolation, PCK at
exactly one sigma, CLEAR-MOT MOTA/IDF1, macro-F1 over present classes, Mann-Whitney AUROC with
ties) plus the monotonicity the definition implies. Not checked: what these metrics say about our
own networks, which is the synthetic benchmark's job, and the pixel-level provenance of the inputs.
"""

import numpy as np
import pytest

from animalpose.eval.metrics import auroc, average_precision, f1_scores, linear_interpolate, pck, rmse, tracking_metrics

BOX = np.array([0.0, 0, 10, 10])


def test_average_precision_is_one_when_the_box_matches_and_zero_below_the_threshold():
    gt = [np.array([[0.0, 0, 10, 10]])]
    assert average_precision([BOX[None]], [np.array([0.9])], gt) == pytest.approx(1.0)
    shifted = np.array([[4.0, 0, 14, 10]])  # IoU 60/140 = 0.43
    assert average_precision([shifted[None]], [np.array([0.9])], gt) == pytest.approx(0.0)
    assert average_precision([shifted[None]], [np.array([0.9])], gt, iou_thresh=0.4) == pytest.approx(1.0)


def test_average_precision_is_the_area_under_the_interpolated_precision_recall_curve():
    # score order -> TP, FP, TP with two GT boxes: precision 1 at recall 0.5, 2/3 at recall 1.
    gt = [np.array([[0.0, 0, 10, 10], [50.0, 50, 60, 60]])]
    preds = [np.array([[0.0, 0, 10, 10], [20.0, 20, 24, 24], [50.0, 50, 60, 60]])]
    assert average_precision(preds, [np.array([0.9, 0.7, 0.5])], gt) == pytest.approx(5 / 6)
    # A second box on an already claimed GT is a false positive, so recall never reaches 1.
    assert average_precision([np.stack([BOX, BOX])], [np.array([0.9, 0.8])], gt) == pytest.approx(0.5)


def test_extra_detections_never_add_recall_and_no_ground_truth_is_undefined():
    gt = [np.array([[0.0, 0, 10, 10]]), np.zeros((0, 4))]
    perfect, spurious = np.array([[0.0, 0, 10, 10]]), np.array([[80.0, 80, 90, 90]])
    # The image without GT can only add a false positive: it costs nothing once the GT box has the
    # top score, and caps precision at 1/2 when it outranks the true detection.
    assert average_precision([perfect, spurious], [[0.99], [0.5]], gt) == pytest.approx(1.0)
    assert average_precision([perfect, spurious], [[0.5], [0.99]], gt) == pytest.approx(0.5)
    assert average_precision([spurious, spurious], [[0.9], [0.8]], gt) == pytest.approx(0.0)
    assert np.isnan(average_precision([np.zeros((0, 4))], [np.zeros(0)], [np.zeros((0, 4))]))


def test_pck_counts_keypoints_within_alpha_times_the_normalising_distance():
    gt = np.zeros((1, 4, 2))
    pred = np.array([[[6.0, 8], [0.0, 11.0], [0.0, 1.0], [0.0, 500.0]]])  # 10, 11, 1, (masked)
    mask = np.array([[True, True, True, False]])
    assert pck(pred, gt, mask, np.array([100.0]), alpha=0.1) == pytest.approx(2 / 3)
    assert pck(pred, gt, mask, np.array([100.0]), alpha=0.2) == pytest.approx(1.0)
    assert pck(gt, gt, mask, np.array([100.0])) == pytest.approx(1.0)
    assert np.isnan(pck(pred, gt, np.zeros((1, 4), bool), np.array([100.0])))


def test_rmse_averages_squared_error_and_skips_unusable_keypoints():
    gt = np.zeros((1, 3, 2))
    pred = np.array([[[3.0, 0], [0.0, 4.0], [np.nan, 1.0]]])
    assert rmse(pred, gt, np.ones((1, 3), bool)) == pytest.approx(np.sqrt(12.5))
    assert rmse(pred, gt, np.array([[True, True, False]])) == pytest.approx(np.sqrt(12.5))
    assert rmse(pred, gt, np.array([[False, True, False]])) == pytest.approx(4.0)
    assert np.isnan(rmse(pred, gt, np.zeros((1, 3), bool)))


def test_tracking_metrics_penalise_an_identity_switch_and_score_idf1_globally():
    gt_ids, gt_boxes, pred_ids, pred_boxes = [[0], [0], [0]], [[BOX]] * 3, [[1], [1], [2]], [[BOX]] * 3
    m = tracking_metrics(gt_ids, gt_boxes, pred_ids, pred_boxes)
    assert m["id_switches"] == 1 and m["fn"] == 0 and m["fp"] == 0 and m["n_gt"] == 3
    assert m["mota"] == pytest.approx(1 - 1 / 3)  # one switch over three GT frames
    assert m["idf1"] == pytest.approx(2 * 2 / 6)  # best global mapping covers 2 of 3 frames
    assert m["n_gt_ids"] == 1 and m["n_pred_ids"] == 2
    stable = tracking_metrics(gt_ids, gt_boxes, [[1], [1], [1]], [[BOX]] * 3)
    assert stable["mota"] == pytest.approx(1.0) and stable["idf1"] == pytest.approx(1.0)
    assert stable["id_switches"] == 0


def test_tracking_metrics_count_misses_and_spurious_boxes():
    a, b, junk = np.array([0.0, 0, 10, 10]), np.array([50.0, 50, 60, 60]), np.array([90.0, 90, 95, 95])
    m = tracking_metrics([[0, 1]], [[a, b]], [[1, 2]], [[a, junk]])
    assert m["fn"] == 1 and m["fp"] == 1 and m["n_pred_ids"] == 2
    assert m["mota"] == pytest.approx(0.0)  # (1 fn + 1 fp) / 2 gt
    assert m["idf1"] == pytest.approx(0.5)  # 2 * 1 / (2 gt + 2 pred)
    empty = tracking_metrics([[0]], [[a]], [[]], [np.zeros((0, 4))])
    assert empty["mota"] == pytest.approx(0.0) and empty["idf1"] == pytest.approx(0.0)  # 1 fn of 1 gt
    assert empty["fn"] == 1 and empty["fp"] == 0 and empty["n_pred_ids"] == 0


def test_f1_scores_averages_only_over_classes_present_in_the_ground_truth():
    assert f1_scores([0, 0], [0, 0], 3)["macro_f1"] == pytest.approx(1.0)
    r = f1_scores([0, 1, 1, 1], [0, 0, 1, 1], 2)
    assert r["confusion"] == [[1, 1], [0, 2]] and r["accuracy"] == pytest.approx(0.75)
    assert r["per_class_f1"] == [pytest.approx(2 / 3), pytest.approx(0.8)]  # prec 1/rec .5, prec 2/3/rec 1
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert f1_scores([1], [0], 2)["macro_f1"] == pytest.approx(0.0)


def test_linear_interpolate_fills_gaps_straight_line_and_holds_the_ends():
    x = np.zeros((5, 2, 1))
    x[:, 0, 0] = [0, 5, 10, 30, 40]
    x[:, 1, 0] = [7, 1, 2, 30, 40]
    observed = np.array([[True, False], [False, True], [False, True], [True, True], [True, True]])
    out = linear_interpolate(x, observed)
    assert out.shape == x.shape
    assert np.allclose(out[:, 0, 0], [0, 10, 20, 30, 40])  # straight line between frames 0 and 3
    assert np.allclose(out[:, 1, 0], [1, 1, 2, 30, 40])  # before the first observation: nearest hold
    assert np.isnan(linear_interpolate(x, np.zeros((5, 2), bool))).all()  # never observed -> undefined
    assert np.isfinite(linear_interpolate(x, np.ones((5, 2), bool))).all()


def test_auroc_is_the_probability_a_positive_outranks_a_negative():
    assert auroc([1.0, 2, 3], [0.5, 0.9]) == pytest.approx(1.0)
    assert auroc([0.5, 0.9], [1.0, 2, 3]) == pytest.approx(0.0)
    assert auroc([1.0, 1], [1.0, 1]) == pytest.approx(0.5)
    assert auroc([1.0, 2], [2.0, 3]) == pytest.approx(0.125)  # one pair ties, three lose
    assert auroc([3.0, 2], [1.0, 4]) == pytest.approx(0.5)
    assert np.isnan(auroc([], [1.0])) and np.isnan(auroc([1.0], []))
