"""Evaluation metrics: detection AP, keypoint PCK/RMSE, tracking IDF1/ID
switches, gap-filling error, behavior F1, triangulation error."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from animalpose.geometry import box_iou


def average_precision(pred_boxes, pred_scores, gt_boxes, iou_thresh: float = 0.5) -> float:
    """VOC-style (all-point interpolated) AP over a list of images."""
    records, n_gt = [], 0
    for pb, ps, gb in zip(pred_boxes, pred_scores, gt_boxes):
        gb = np.asarray(gb).reshape(-1, 4)
        n_gt += len(gb)
        used = np.zeros(len(gb), bool)
        order = np.argsort(-np.asarray(ps))
        ious = box_iou(np.asarray(pb).reshape(-1, 4), gb) if len(gb) else np.zeros((len(pb), 0))
        for i in order:
            j = int(np.argmax(ious[i])) if ious.shape[1] else -1
            hit = j >= 0 and ious[i, j] >= iou_thresh and not used[j]
            if hit:
                used[j] = True
            records.append((ps[i], hit))
    if n_gt == 0:
        return float("nan")
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.r_[0, recall, 1]
    mpre = np.r_[0, precision, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pck(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, norm: np.ndarray, alpha: float = 0.1) -> float:
    """Fraction of masked keypoints within ``alpha * norm`` (norm per instance, e.g. box diagonal)."""
    d = np.linalg.norm(pred - gt, axis=-1)
    thr = alpha * np.asarray(norm)[..., None]
    m = mask.astype(bool)
    return float((d[m] <= np.broadcast_to(thr, d.shape)[m]).mean()) if m.any() else float("nan")


def rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool) & np.isfinite(pred).all(-1) & np.isfinite(gt).all(-1)
    return float(np.sqrt(np.mean(np.sum((pred[m] - gt[m]) ** 2, -1)))) if m.any() else float("nan")


def tracking_metrics(gt_ids: list[np.ndarray], gt_boxes: list[np.ndarray], pred_ids: list[np.ndarray],
                     pred_boxes: list[np.ndarray], iou_thresh: float = 0.5) -> dict:
    """CLEAR-MOT style MOTA / ID switches and IDF1 from per-frame boxes and ids."""
    matches_per_frame, n_gt, n_pred, fn, fp, idsw = [], 0, 0, 0, 0, 0
    last: dict[int, int] = {}
    for gi, gb, pi, pb in zip(gt_ids, gt_boxes, pred_ids, pred_boxes):
        gb, pb = np.asarray(gb).reshape(-1, 4), np.asarray(pb).reshape(-1, 4)
        n_gt += len(gb)
        n_pred += len(pb)
        pairs = []
        if len(gb) and len(pb):
            iou = box_iou(gb, pb)
            r, c = linear_sum_assignment(-iou)
            pairs = [(gi[a], pi[b]) for a, b in zip(r, c) if iou[a, b] >= iou_thresh]
        fn += len(gb) - len(pairs)
        fp += len(pb) - len(pairs)
        for g, p in pairs:
            if g in last and last[g] != p:
                idsw += 1
            last[g] = p
        matches_per_frame.append(pairs)
    # IDF1: best global one-to-one mapping between gt and predicted identities.
    g_list = sorted({g for ids in gt_ids for g in ids})
    p_list = sorted({p for ids in pred_ids for p in ids})
    co = np.zeros((len(g_list), len(p_list)))
    gi_ = {g: i for i, g in enumerate(g_list)}
    pi_ = {p: i for i, p in enumerate(p_list)}
    for pairs in matches_per_frame:
        for g, p in pairs:
            co[gi_[g], pi_[p]] += 1
    idtp = co[linear_sum_assignment(-co)].sum() if co.size else 0.0
    idf1 = 2 * idtp / max(n_gt + n_pred, 1)
    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)
    return {"mota": float(mota), "idf1": float(idf1), "id_switches": int(idsw), "fn": int(fn), "fp": int(fp),
            "n_gt": int(n_gt), "n_pred_ids": len(p_list), "n_gt_ids": len(g_list)}


def f1_scores(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> dict:
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (gt, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    present = cm.sum(1) > 0
    return {"macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
            "accuracy": float(tp.sum() / max(cm.sum(), 1)), "per_class_f1": f1.tolist(), "confusion": cm.tolist()}


def linear_interpolate(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Baseline gap filler: per-keypoint linear interpolation (nearest-hold at ends)."""
    out = x.copy().astype(float)
    t = np.arange(x.shape[0])
    for k in range(x.shape[1]):
        o = observed[:, k].astype(bool)
        if o.sum() == 0:
            out[:, k] = np.nan
            continue
        for d in range(x.shape[2]):
            out[:, k, d] = np.interp(t, t[o], x[o, k, d])
    return out


def auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (Mann-Whitney U)."""
    pos, neg = np.asarray(scores_pos, float).ravel(), np.asarray(scores_neg, float).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
