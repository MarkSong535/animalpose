"""Behavior metrics over sessions, with post-processing variants from one forward pass.

The behavior TCN is frame-wise, so bout smoothing and prior adaptation
(:mod:`animalpose.behavior.postprocess`) change the score without retraining.
:func:`behavior_report` runs the model once per session and scores every variant, so comparing
post-processing costs one forward pass instead of one training run per variant. Both steps work
per session, never across a session boundary: sessions are different animals, cameras and days.

Labels arrive as indices of one of the vocabularies in :mod:`animalpose.labels`; a model's classes are
usually a different vocabulary, so ``label_classes`` names the one the ground truth uses and the two
are reconciled by name (:func:`animalpose.data.behavior_sets.remap_labels`).
"""

from __future__ import annotations

import numpy as np

from animalpose.behavior.postprocess import adapt_prior, median_smooth
from animalpose.data.behavior_sets import remap_labels
from animalpose.data.tracks import IGNORE
from animalpose.eval.metrics import f1_scores
from animalpose.labels import BEHAVIORS_EXT, LABELLED, SHARED

#: The two protocols behavior models are reported on: the classes any label set and any classifier
#: share, and the classes real data actually labels (which adds "other").
PROTOCOLS: dict[str, tuple[str, ...]] = {"shared4": SHARED, "labelled5": LABELLED}


def session_logits(model, sessions, conf_thresh: float = 0.6) -> list[np.ndarray]:
    """Classifier logits ``(T, C)`` per session, from the whole track in one pass."""
    import torch

    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in sessions:
            x = torch.from_numpy(np.nan_to_num(s.coords).astype(np.float32))[None].to(dev)
            m = torch.from_numpy(((s.conf >= conf_thresh) & np.isfinite(s.coords).all(-1)).astype(np.float32))[None].to(dev)
            out.append(model.eval()(x, m)[0].cpu().numpy())
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def protocol_metrics(probs: np.ndarray, gt: np.ndarray, classes, candidates,
                     label_classes: tuple[str, ...] | list[str] | None = None) -> dict:
    """Frame-wise macro-F1 on labelled frames whose class is in ``candidates``, argmax over
    ``candidates`` only (so a model with extra classes is scored on the common subset).

    ``gt`` holds indices of ``label_classes`` when that differs from the model's ``classes``.
    """
    classes, candidates = list(classes), list(candidates)
    if label_classes is not None and list(label_classes) != classes:
        gt = remap_labels(gt, label_classes, classes)
    cols = [classes.index(c) for c in candidates]
    lab = gt != IGNORE
    sel = lab & np.isin(gt, [classes.index(c) for c in candidates])
    if not sel.any():
        return {"macro_f1": float("nan"), "accuracy": float("nan"), "n_frames": 0, "per_class_f1": {}}
    pred = np.argmax(probs[sel][:, cols], -1)
    gt_idx = np.array([candidates.index(classes[int(g)]) for g in gt[sel]])
    r = f1_scores(pred, gt_idx, len(candidates))
    return {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "per_class_f1": dict(zip(candidates, r["per_class_f1"])),
            "confusion": r["confusion"], "n_frames": int(sel.sum())}


def behavior_report(model, sessions, protocols: dict[str, list[str] | tuple[str, ...]],
                    smoothing: tuple[int, ...] = (1, 3, 5, 9, 15), prior_adapt: bool = True,
                    prior_max_ratio: float = 4.0, conf_thresh: float = 0.6,
                    label_classes: tuple[str, ...] | list[str] | None = None) -> dict:
    """``{variant: {protocol: metrics}}`` for each smoothing window and with/without prior
    adaptation; ``variant`` is ``raw``, ``smooth5`` or ``smooth5+prior``.

    Sessions are concatenated before scoring, so pass sessions that share a label vocabulary;
    ``label_classes`` is that vocabulary when it is not the model's own classes.
    """
    logits = session_logits(model, sessions, conf_thresh)
    labels = [np.asarray(s.labels) for s in sessions]
    variants: dict[str, list[np.ndarray]] = {"raw": []}
    for w in smoothing:
        if w > 1:
            variants[f"smooth{w}"] = []
    for z in logits:
        p = _softmax(z)
        variants["raw"].append(p)
        for w in smoothing:
            if w > 1:
                variants[f"smooth{w}"].append(median_smooth(p, w))
    out = {}
    for name, per_session in variants.items():
        for use_prior in ((False, True) if prior_adapt else (False,)):
            key = name if not use_prior else f"{name}+prior"
            rows = [adapt_prior(p, max_ratio=prior_max_ratio)[0] if use_prior else p for p in per_session]
            out[key] = {proto: protocol_metrics(np.concatenate(rows), np.concatenate(labels), model.classes, cands,
                                                label_classes)
                        for proto, cands in protocols.items()}
    return out


def protocol_scores(model, sessions, protocols: dict[str, tuple[str, ...]] | None = None,
                    label_classes: tuple[str, ...] | list[str] | None = BEHAVIORS_EXT,
                    conf_thresh: float = 0.6) -> dict:
    """The raw (unsmoothed, unadapted) ``{protocol: metrics}`` score of ``model``.

    The plain comparison behind :func:`behavior_report`, which is what a results file records;
    ``protocols`` defaults to :data:`PROTOCOLS`, and sessions must share one label vocabulary
    (``label_classes``, by default :data:`animalpose.labels.BEHAVIORS_EXT`).
    """
    wanted = {name: cands for name, cands in (protocols or PROTOCOLS).items()
              if all(c in model.classes for c in cands)}  # a model without "other" cannot be scored on it
    report = behavior_report(model, sessions, wanted, smoothing=(1,), prior_adapt=False,
                             conf_thresh=conf_thresh, label_classes=label_classes)
    return {proto: {k: v for k, v in m.items() if k != "confusion"} for proto, m in report["raw"].items()}
