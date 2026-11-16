"""Full behavior catalogue and per-frame probabilities for every behavior.

The catalogue (``behaviors.json``) holds 132 mouse and rat behaviors compiled from
published ethograms, datasets and tools (Jhuang et al. 2010, CRIM13, CalMS21/MARS,
MABe22/MABe2025, Live Mouse Tracker, DeepEthogram, SimBA, B-SOiD, Keypoint-MoSeq,
LISBET, ABODe, HomeCageScan, EthoVision, the Stanford Mouse Ethogram, rat ethograms, pain
and seizure scales). Each entry has a definition, group, synonyms, its sources, and
whether it is plausibly detectable from 2D keypoints in top or side view. An ``other``
class is added for labelled activity outside the catalogue.

The behavior TCN only knows the classes it was trained on. This module places its output
in the catalogue and fills in what can be derived from poses without training labels:

* ``classifier``: probability from the trained classifier, for its classes (names or
  synonyms, e.g. ``grooming`` -> ``self_groom``);
* ``rule``: ``run`` is split off ``walk`` above ``RUN_SPEED`` body lengths/s;
* ``pair rule``: social behaviors from the geometry between two tracked animals (nose-face,
  nose-anogenital and nose-body distance, approach, following, chasing, fleeing, huddling,
  attack). These are transparent soft rules, not trained classifiers.

Every frame gets a probability for every behavior (they sum to 1): the chance ``s`` that
any social rule fires is shared between the social behaviors, ``1 - s`` between the
solitary ones. Behaviors that neither the classifier nor a rule covers get 0 and are
listed by :func:`describe` as not modelled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from animalpose.skeleton import CANONICAL


@dataclass(frozen=True)
class Behavior:
    name: str
    group: str
    social: bool
    definition: str
    synonyms: tuple[str, ...] = field(default=())
    sources: tuple[str, ...] = field(default=())
    detectable_top: str = ""
    detectable_side: str = ""


def _load() -> tuple[Behavior, ...]:
    raw = json.loads((Path(__file__).with_name("behaviors.json")).read_text())
    items = [Behavior(r["name"], r["group"], bool(r["social"]), r["definition"], tuple(r.get("synonyms", ())),
                      tuple(r.get("sources", ())), r.get("detectable_top", ""), r.get("detectable_side", ""))
             for r in raw]
    items.append(Behavior("other", "other", False, "labelled activity outside the catalogue (object play, ...)"))
    return tuple(items)


CATALOG: tuple[Behavior, ...] = _load()
NAMES: tuple[str, ...] = tuple(b.name for b in CATALOG)
SOCIAL: tuple[str, ...] = tuple(b.name for b in CATALOG if b.social)
RUN_SPEED = 2.0  # body lengths per second


def _key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


# Classifier class names (ours and common dataset labels) -> catalogue names.
ALIASES = {
    "resting": "rest", "immobile": "rest", "sniffing": "sniff_environment", "exploring": "sniff_environment",
    "walking": "walk", "locomotion": "walk", "running": "run", "turning": "turn", "rearing": "rear",
    "grooming": "self_groom", "self_grooming": "self_groom", "eating": "eat", "drinking": "drink",
    "digging": "dig", "nesting": "nest_build", "climbing": "climb", "scratching": "scratch",
    "freezing": "freeze", "jumping": "jump", "stretch_attend": "stretched_attend_posture",
}
for _b in CATALOG:  # dataset synonyms, e.g. "Oral-genital Contact (LMT; ...)", "wash / washing (...)"
    for _s in _b.synonyms:
        for _alt in _s.split("(")[0].split("/"):
            if _alt.strip():
                ALIASES.setdefault(_key(_alt.strip()), _b.name)
PAIR_RULES = ("sniff_face", "sniff_anogenital", "sniff_body", "approach", "follow", "chase", "flee", "huddle", "attack")


def canonical(name: str) -> str | None:
    """Catalogue name for a class or label name (or None)."""
    return name if name in NAMES else ALIASES.get(_key(name))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def body_kinematics(kp: np.ndarray, conf: np.ndarray, fps: float, times: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-frame centre, unit body axis, body length (px) and speed (body lengths / s).

    ``times`` (s, per frame) handles variable frame rates and dropped frames; without it
    frames are assumed ``1 / fps`` apart."""
    idx = CANONICAL.index
    core = [idx(n) for n in ("head_center", "neck", "spine_upper", "spine_mid", "spine_lower", "tail_base")]
    w = (conf[:, core] >= 0.3)[..., None].astype(float)
    centre = (np.nan_to_num(kp[:, core]) * w).sum(1) / np.maximum(w.sum(1), 1e-6)
    nose, tail = kp[:, idx("nose")], kp[:, idx("tail_base")]
    length = np.linalg.norm(nose - tail, axis=-1)
    ref = float(np.nanmedian(length)) if np.isfinite(length).any() else 1.0
    axis = (nose - tail) / (np.nan_to_num(length)[:, None] + 1e-6)
    if len(centre) > 1:
        t = np.asarray(times, float) if times is not None else np.arange(len(centre)) / fps
        v = np.gradient(centre, t, axis=0) / max(ref, 1e-6)
        rate = 1.0 / max(float(np.median(np.diff(t))), 1e-6)
    else:
        v, rate = np.zeros_like(centre), fps
    k = max(1, int(round(0.3 * rate)))  # ~0.3 s smoothing of speed
    sp = np.convolve(np.linalg.norm(v, axis=-1), np.ones(k) / k, mode="same")
    return {"centre": centre, "axis": np.nan_to_num(axis), "length": ref, "speed": sp, "velocity": v,
            "nose": nose, "tail": tail, "body": kp[:, idx("spine_mid")]}


def pair_probabilities(a: dict, b: dict) -> dict[str, np.ndarray]:
    """Soft social-behavior probabilities of animal ``a`` towards ``b`` (aligned frames)."""
    bl = a["length"]
    d_nn = np.linalg.norm(a["nose"] - b["nose"], axis=-1) / bl
    d_ng = np.linalg.norm(a["nose"] - b["tail"], axis=-1) / bl
    d_nb = np.linalg.norm(a["nose"] - b["body"], axis=-1) / bl
    d_c = np.linalg.norm(a["centre"] - b["centre"], axis=-1) / bl
    sa, sb = a["speed"], b["speed"]
    align = (a["axis"] * b["axis"]).sum(-1)
    behind = _sig(-((a["centre"] - b["centre"]) * b["axis"]).sum(-1) / bl / 0.2)
    gap = a["centre"] - b["centre"]
    dist = np.linalg.norm(gap, axis=-1) + 1e-6
    away = (a["velocity"] * gap).sum(-1) / dist  # a moving away from b (body lengths / s)
    b_toward = (b["velocity"] * gap).sum(-1) / dist  # b moving towards a
    trailing = _sig((1.5 - d_c) / 0.3) * _sig((align - 0.5) / 0.15) * behind
    out = {
        "sniff_face": _sig((0.5 - d_nn) / 0.1),
        "sniff_anogenital": _sig((0.5 - d_ng) / 0.1) * _sig((d_nn - 0.6) / 0.1),
        "sniff_body": _sig((0.4 - d_nb) / 0.1) * _sig((d_nn - 0.6) / 0.1) * _sig((d_ng - 0.6) / 0.1),
        "approach": _sig((3.0 - d_c) / 0.5) * _sig((d_c - 1.0) / 0.2) * _sig((-away - 0.8) / 0.2),
        "follow": trailing * _sig((sa - 0.8) / 0.2) * _sig((sb - 0.8) / 0.2) * _sig((RUN_SPEED - sa) / 0.4),
        "chase": trailing * _sig((sa - RUN_SPEED) / 0.4) * _sig((sb - RUN_SPEED) / 0.4),
        "flee": _sig((2.0 - d_c) / 0.4) * _sig((away - RUN_SPEED) / 0.4) * _sig((b_toward - 0.5) / 0.3),
        "huddle": _sig((0.9 - d_c) / 0.15) * _sig((0.2 - sa) / 0.1) * _sig((0.2 - sb) / 0.1),
        "attack": _sig((1.0 - d_c) / 0.2) * _sig((sa - RUN_SPEED) / 0.4) * _sig((sb - RUN_SPEED) / 0.4)
        * _sig((0.3 - align) / 0.3),
    }
    return {k: np.nan_to_num(v) for k, v in out.items()}


def full_probabilities(model_classes, probs: np.ndarray, kin: dict, partners: list[dict] | None = None):
    """``(T, len(CATALOG))`` probabilities and the ``modelled`` mask over the catalogue.

    ``probs (T, C)`` are the classifier's probabilities over ``model_classes``;
    ``partners`` are :func:`body_kinematics` dicts of other animals aligned to these frames
    (NaN where absent).
    """
    t = len(probs)
    full = np.zeros((t, len(NAMES)))
    modelled = np.zeros(len(NAMES), bool)
    for j, c in enumerate(model_classes):
        name = canonical(c)
        if name is not None:
            full[:, NAMES.index(name)] += probs[:, j]
            modelled[NAMES.index(name)] = True
    if modelled[NAMES.index("walk")]:  # running: fast walking
        walk, run = NAMES.index("walk"), NAMES.index("run")
        moved = full[:, walk] * _sig((kin["speed"] - RUN_SPEED) / 0.3)
        full[:, run] += moved
        full[:, walk] -= moved
        modelled[run] = True
    solitary = ~np.array([b.social for b in CATALOG])
    full[:, solitary] /= np.maximum(full[:, solitary].sum(1, keepdims=True), 1e-9)
    social = np.zeros((t, len(PAIR_RULES)))
    for b in partners or []:
        p = pair_probabilities(kin, b)
        social = np.maximum(social, np.stack([p[k] for k in PAIR_RULES], 1))
    for k in PAIR_RULES:
        modelled[NAMES.index(k)] = True
    s = 1.0 - np.prod(1.0 - social, axis=1, keepdims=True)  # chance of any social behavior
    full[:, solitary] *= 1.0 - s
    share = social / np.maximum(social.sum(1, keepdims=True), 1e-9)
    for i, k in enumerate(PAIR_RULES):
        full[:, NAMES.index(k)] = (share[:, i:i + 1] * s)[:, 0]
    return full, modelled


def top_k(full: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Indices and probabilities of the ``k`` most likely behaviors per frame."""
    idx = np.argsort(-full, axis=1)[:, :k]
    return idx, np.take_along_axis(full, idx, 1)


def describe(modelled: np.ndarray) -> list[dict]:
    how = {**{k: "pair rule" for k in PAIR_RULES}, "run": "rule (fast walk)"}
    return [{"name": b.name, "group": b.group, "social": b.social, "definition": b.definition,
             "synonyms": list(b.synonyms), "sources": list(b.sources), "detectable_top": b.detectable_top,
             "detectable_side": b.detectable_side, "modelled": bool(m),
             "source": how.get(b.name, "classifier") if m else "not modelled (no training data)"}
            for b, m in zip(CATALOG, modelled)]
