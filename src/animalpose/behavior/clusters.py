"""Behavior labels for a new setup from its pose tracks alone (no prior labels).

1. :func:`cluster_tracks`: per-frame kinematic features (body speed, body length relative to its
   median, head motion relative to the body, posture change, bend) averaged over ~0.5 s are
   clustered with k-means fitted on the training range only; example bouts are saved as montages.
2. :func:`write_naming`: each cluster's median kinematics are matched against :data:`NAME_RULES`, a
   short list of published-signature thresholds (a curled, bent animal with a moving face grooms; a
   top-view body that foreshortens while the head rises rears; more than 8 body lengths per second
   is a tracking jump, not a behavior).
3. :func:`label_clusters`: a ``cluster -> behavior`` mapping turns clusters into per-frame labels
   (short bouts merged into neighbours) and writes session ``.npz`` files for train and held-out
   ranges, usable with ``session_dirs`` in a behavior config.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from animalpose.behavior.catalog import canonical
from animalpose.data.behavior_sets import save_session
from animalpose.data.tracks import IGNORE, Session
from animalpose.skeleton import CANONICAL


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = np.pad(x, [(w // 2, w - 1 - w // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def kinematics(kp: np.ndarray, conf: np.ndarray, fps: float, times: np.ndarray | None = None):
    """``(T, F)`` features from smoothed keypoints ``(T, K, 2)``; derivatives use ``times`` (s) when
    given, so variable frame rates and dropped frames do not inflate speeds."""
    t = np.asarray(times, float) if times is not None else np.arange(len(kp)) / fps
    idx = CANONICAL.index
    body = [idx(n) for n in ("neck", "spine_upper", "spine_mid", "spine_lower", "tail_base", "head_center")]
    ok = conf >= 0.3
    w = np.where(ok, 1.0, 0.0)[..., None]
    centre = (kp[:, body] * w[:, body]).sum(1) / np.maximum(w[:, body].sum(1), 1e-6)
    head = kp[:, idx("head_center")]
    nose = kp[:, idx("nose")]
    tail = kp[:, idx("tail_base")]
    length = np.linalg.norm(nose - tail, axis=-1)
    ref = np.nanmedian(length) + 1e-6
    v = np.gradient(centre, t, axis=0) / ref  # body lengths / s
    speed = np.linalg.norm(v, axis=-1)
    axis = (nose - tail) / (length[:, None] + 1e-6)
    ang = np.arctan2(axis[:, 1], axis[:, 0])
    turn = np.abs(np.gradient(np.unwrap(ang), t))
    rel = (kp - centre[:, None]) / ref
    posture = np.nanmean(np.linalg.norm(np.gradient(rel, t, axis=0), axis=-1), 1)
    head_rel = np.linalg.norm(np.gradient((nose - head) / ref, t, axis=0), axis=-1)
    hv, tv = nose - centre, centre - tail
    bend = np.abs(np.arctan2(hv[:, 0] * tv[:, 1] - hv[:, 1] * tv[:, 0], (hv * tv).sum(-1)))
    feats = np.stack([speed, length / ref, turn, posture, head_rel, bend], 1)
    names = ["speed", "length", "turn", "posture", "head_motion", "bend"]
    w = max(1, int(round(0.5 * fps)))
    sm = _smooth(np.nan_to_num(feats), w)
    sd = np.sqrt(np.maximum(_smooth(np.nan_to_num(feats) ** 2, 2 * w) - _smooth(np.nan_to_num(feats), 2 * w) ** 2, 0))
    return np.concatenate([sm, sd[:, [0, 1, 3]]], 1), names + ["speed_sd", "length_sd", "posture_sd"]


def bouts(lab: np.ndarray):
    """``(start, end, label)`` for each run of one label."""
    starts = np.r_[0, np.nonzero(np.diff(lab))[0] + 1]
    ends = np.r_[starts[1:], len(lab)]
    return list(zip(starts, ends, lab[starts]))


# Kinematic signatures, checked against the clusters a person named by eye on one recording (see ``agree_with``).
# Features are the ones :func:`kinematics` returns: ``speed`` and ``posture`` in body lengths/s, ``length`` the apparent
# nose-to-tail length over its median (a top camera foreshortens a rearing or curled animal), ``turn`` rad/s,
# ``head_motion`` nose speed relative to the head, ``bend`` the angle between the front and rear halves. The
# first matching rule wins, so specific postures come first; a cluster whose kinematics are physically
# impossible (a tracking jump) is named ``other``.
NAME_RULES: tuple[tuple[str, tuple[tuple[str, str, float], ...]], ...] = (
    ("other", (("speed", ">", 8.0),)),  # a tracking jump, not a behavior; any one of these is enough,
    ("other", (("speed_sd", ">", 20.0),)),  # so they are separate rules (a rule ANDs its criteria)
    ("other", (("posture_sd", ">", 20.0),)),
    ("running", (("speed", ">", 3.0), ("length", ">", 0.9), ("bend", "<", 1.0))),
    ("grooming", (("bend", ">", 1.0), ("head_motion", ">", 0.8), ("length", "<", 0.6))),
    ("turning", (("turn", ">", 4.0), ("posture", ">", 0.5))),
    ("rearing", (("length", "<", 0.85), ("bend", "<", 1.0), ("head_motion", ">", 0.4), ("posture", ">", 0.5))),
    ("walking", (("speed", ">", 0.4), ("length", ">", 0.9), ("bend", "<", 1.0))),
    ("sniffing", (("speed", "<", 0.35), ("length", ">", 0.95), ("bend", "<", 0.4))),
    ("resting", (("speed", "<", 0.35), ("length", "<", 0.95))),
)


def _matches(feat: dict[str, float], crit: tuple[tuple[str, str, float], ...]) -> bool:
    for name, op, thr in crit:
        v = feat.get(name, float("nan"))
        if not np.isfinite(v) or (v >= thr if op == "<" else v <= thr):
            return False
    return True


def name_clusters(summary: dict, rules=NAME_RULES) -> dict:
    """``{cluster: {name, features, share}}``: the first rule whose criteria all hold, else ``other``."""
    out = {}
    for c, stats in summary.items():
        name = next((n for n, crit in rules if _matches(stats, crit)), "other")
        out[c] = {"name": name, "share_of_train": stats.get("share_of_train"),
                  "features": {k: v for k, v in stats.items() if k != "share_of_train"}}
    return out


def naming_agreement(naming: dict, human: dict) -> dict:
    """Agreement with hand-named clusters (``{"0": "resting", ...}``), weighted by their share of frames."""
    rows = []
    for c, want in human.items():
        key = c if c in naming else str(c)
        if key in naming:
            rows.append((float(naming[key].get("share_of_train") or 0.0), str(c), naming[key]["name"], str(want)))
    if not rows:
        return {"n_named_clusters": 0, "weighted_agreement": float("nan"), "mismatches": []}
    total = sum(w for w, _, _, _ in rows)
    return {"n_named_clusters": len(rows),
            "weighted_agreement": sum(w * (got == want) for w, _, got, want in rows) / max(total, 1e-9),
            "mismatches": [f"cluster {c}: rule says {got}, human said {want}" for _, c, got, want in rows
                           if got != want]}


def montage(video: str | Path, centre: np.ndarray, frames: list[int], half: int = 45, scale: int = 2) -> np.ndarray:
    """A strip of crops around the animal at ``frames``, for naming a cluster by eye."""
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, f = cap.read()
        if not ok:
            continue
        cx, cy = np.nan_to_num(centre[t]).astype(int)
        f = cv2.copyMakeBorder(f, half, half, half, half, cv2.BORDER_REPLICATE)
        tiles.append(cv2.resize(f[cy:cy + 2 * half, cx:cx + 2 * half], None, fx=scale, fy=scale))
    cap.release()
    return np.concatenate(tiles, 1) if tiles else np.zeros((10, 10, 3), np.uint8)


def cluster_tracks(tracks: str | Path, video: str | Path, out: str | Path, *, fps: float = 30.0,
                   train_range: Sequence[float] = (0.0, 0.8), k: int = 8) -> dict:
    """Cluster the kinematics of ``tracks.npz``; writes ``clusters.npz``, ``summary.json`` and montages."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    z = np.load(tracks)
    kp, cf = z["track0_smoothed"], z["track0_smoothed_conf"]
    n = len(kp)
    times = z["times"] if "times" in z.files else None
    feats, names = kinematics(kp, cf, fps, times)
    a, b = int(train_range[0] * n), int(train_range[1] * n)
    scaler = StandardScaler().fit(feats[a:b])
    km = KMeans(k, n_init=10, random_state=0).fit(scaler.transform(feats[a:b]))
    lab = km.predict(scaler.transform(feats))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    centre = kp[:, CANONICAL.index("spine_mid")]
    summary = {}
    rng = np.random.default_rng(0)
    for c in range(k):
        sel = lab[a:b] == c
        all_bouts = [(s, e) for s, e, lb in bouts(lab[a:b]) if lb == c]
        long_bouts = [x for x in all_bouts if x[1] - x[0] >= fps] or [x for x in all_bouts if x[1] - x[0] >= 0.3 * fps]
        summary[c] = {"share_of_train": float(sel.mean()),
                      **{f: round(float(np.median(feats[a:b][sel, i])), 3) for i, f in enumerate(names)}}
        picks = [long_bouts[i] for i in rng.choice(len(long_bouts), min(3, len(long_bouts)), replace=False)]
        rows = [montage(Path(video), centre, list(np.linspace(s + a, min(e, s + 2 * fps) + a - 1, 6).astype(int)))
                for s, e in picks]
        if rows:
            wmax = max(r.shape[1] for r in rows)
            rows = [cv2.copyMakeBorder(r, 2, 2, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
            cv2.imwrite(str(out / f"cluster{c}.png"), np.concatenate(rows, 0))
    np.savez_compressed(out / "clusters.npz", labels=lab, features=feats, fps=fps, train_range=train_range,
                        tracks=str(tracks))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def write_naming(clusters: str | Path, agree_with: str | Path | None = None) -> dict:
    """Name clusters by rule, score the rules against hand-named ones, and write ``naming.json``."""
    d = Path(clusters)
    naming = name_clusters(json.loads((d / "summary.json").read_text()))
    report = {"naming": naming, "rules": {n: list(c) for n, c in NAME_RULES}}
    if agree_with:
        human = json.loads(Path(agree_with).read_text()).get("mapping", {})
        report["agreement"] = naming_agreement(naming, human)
    (d / "naming.json").write_text(json.dumps(report, indent=1))
    return report


def label_clusters(clusters: str | Path, mapping: dict[int, str], out: str | Path, *, name: str = "recording",
                   view: str = "top", min_bout: float = 0.5) -> dict:
    """Turn a ``cluster -> behavior`` mapping into train/held-out session files; returns the class shares."""
    d = Path(clusters)
    c = np.load(d / "clusters.npz")
    lab, fps = c["labels"], float(c["fps"])
    unknown = sorted(v for v in mapping.values() if canonical(v) is None)
    if unknown:
        raise ValueError(f"not in the behavior catalogue (names or synonyms): {unknown}")
    names = tuple(dict.fromkeys(mapping.values()))
    beh = np.array([names.index(mapping[x]) if x in mapping else IGNORE for x in lab])
    min_len = int(round(min_bout * fps))  # merge bouts shorter than min_bout into the preceding bout
    for s, e, _ in bouts(beh):
        if e - s < min_len and s > 0:
            beh[s:e] = beh[s - 1]
    z = np.load(str(c["tracks"]))
    kp, cf = z["track0_smoothed"], z["track0_smoothed_conf"]
    n = len(lab)
    a, b = (int(x * n) for x in c["train_range"])
    out = Path(out)
    for part, (lo, hi) in {"train": (a, b), "heldout": (b, n)}.items():
        save_session(out / part / f"{name}_{part}.npz",
                     Session(f"{name}_{part}", view, kp[lo:hi], cf[lo:hi], beh[lo:hi], fps), label_names=names)
    share = {names[k]: round(float((beh == k).mean()), 3) for k in np.unique(beh[beh != IGNORE])}
    (out / "labels.json").write_text(json.dumps({"mapping": mapping, "share": share, "min_bout_s": min_bout}, indent=1))
    return share


def parse_mapping(specs: Sequence[str]) -> dict[int, str]:
    """``["0=resting", "2=grooming"]`` -> ``{0: "resting", 2: "grooming"}``."""
    return {int(k): v for k, v in (s.split("=") for s in specs)}
