"""Behavior datasets built from *video*: our pose model makes the keypoint tracks.

Each reader turns a dataset's videos plus its labels into
:class:`animalpose.data.tracks.Session` objects (canonical 25 keypoints,
per-frame behavior class, ``IGNORE`` where unlabeled), saved as one ``.npz``
per animal and video. :func:`load_session_dir` reads them back for training
and evaluation, next to DeepLabCut-based home-cage sessions.

* **MIT home-cage** (Jhuang et al. 2010, cbmm.mit.edu/mouse-dataset): 12
  side-view videos of single-housed mice, 30 fps, frame-wise labels for eat,
  drink, groom, hang, micromovement, rear, rest and walk, from two annotator
  groups. The site states no license for the videos and asks for a citation.
* **ABODe** (Camilleri et al., Edinburgh DataShare 10.7488/ds/3833, CC BY 4.0):
  200 two-minute side-view infrared snippets of group-housed mice, 25 fps,
  with a box per mouse per frame and a behavior per mouse per second.
* **BORIS-labelled videos** (any list of home-cage videos with BORIS projects):
  BORIS state events. Only ``eating`` maps onto the core classes; the other
  coded activities become ``other``, and uncoded time is ignored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from animalpose.data.tracks import IGNORE, Session
from animalpose.labels import BEHAVIORS_EXT

MIT_LABELS = {"eat": "eating", "drink": "other", "groom": "grooming", "hang": "other",
              "micromovement": "other", "rear": "rearing", "rest": "resting", "walk": "walking"}
ABODE_LABELS = {"Imm": "resting", "Feed": "eating", "S-Grm": "grooming", "Loco": "walking",
                "Drink": "other", "A-Grm": "other", "Other": "other"}  # N/Obs, N/Adm -> ignored
BORIS_LABELS = {"eating": "eating"}  # every other coded state event -> "other"


# ----------------------------------------------------------------------------- session files

def save_session(path: Path, s: Session, boxes: np.ndarray | None = None,
                 label_names: tuple[str, ...] = BEHAVIORS_EXT) -> None:
    """``s.labels`` index ``label_names`` (stored with the file, so any label set can be remapped later)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coords=s.coords.astype(np.float32), conf=s.conf.astype(np.float32),
                        labels=s.labels.astype(np.int64), fps=s.fps, animal=s.animal, view=s.view,
                        label_names=np.array(label_names),
                        **({"boxes": boxes.astype(np.float32)} if boxes is not None else {}))


def remap_labels(labels: np.ndarray, src: tuple[str, ...] | list[str], dst: tuple[str, ...] | list[str]) -> np.ndarray:
    """Class indices in ``src`` -> indices in ``dst`` (``IGNORE`` for names ``dst`` lacks)."""
    table = np.array([list(dst).index(n) if n in dst else IGNORE for n in src] + [IGNORE])
    lab = np.asarray(labels)
    return np.where(lab == IGNORE, IGNORE, table[np.clip(lab, 0, len(src))])


def load_session_dir(root: str | Path, target_fps: float = 15.0, groups: list[str] | None = None,
                     classes: tuple[str, ...] | list[str] = BEHAVIORS_EXT) -> list[Session]:
    """All ``*.npz`` sessions under ``root``, resampled to about ``target_fps``, labels as indices of ``classes``."""
    out = []
    for p in sorted(Path(root).rglob("*.npz")):
        z = np.load(p, allow_pickle=False)
        animal = str(z["animal"])
        if groups is not None and animal not in groups:
            continue
        fps = float(z["fps"])
        step = max(1, int(round(fps / target_fps)))
        names = [str(n) for n in z["label_names"]] if "label_names" in z.files else list(BEHAVIORS_EXT)
        labels = remap_labels(z["labels"][::step], names, classes)
        out.append(Session(animal, str(z["view"]), z["coords"][::step].astype(np.float64),
                           z["conf"][::step].astype(np.float64), labels, fps / step))
    return out


# ----------------------------------------------------------------------------- tracking

def read_frames(video: Path, stride: int = 1):
    cap = cv2.VideoCapture(str(video))
    t = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if t % stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            yield t, img
        t += 1
    cap.release()


def track_main_animal(pipe, video: Path, stride: int = 1, batch: int = 64):
    """Single-animal tracks: per frame the most confident detection that is not much
    smaller than the largest one (prey or debris are smaller than the mouse).
    Returns ``coords (T,K,2)``, ``conf (T,K)``, ``boxes (T,4)`` at the strided rate (NaN without detection)."""
    from animalpose.skeleton import CANONICAL

    coords, confs, boxes = [], [], []
    for _, img in read_frames(video, stride):
        b, s = pipe.detect(img)
        if len(b) == 0:
            coords.append(np.full((CANONICAL.num_keypoints, 2), np.nan)), confs.append(np.zeros(CANONICAL.num_keypoints))
            boxes.append(np.full(4, np.nan))
            continue
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ok = area >= 0.5 * area.max()
        j = int(np.argmax(np.where(ok, s, -1)))
        kp, cf, _ = pipe.estimate(img, b[j:j + 1])
        coords.append(kp[0]), confs.append(cf[0]), boxes.append(b[j])
    return np.asarray(coords, np.float32), np.asarray(confs, np.float32), np.asarray(boxes, np.float32)


# ----------------------------------------------------------------------------- MIT home-cage

def mit_labels(txt: Path, n_frames: int) -> np.ndarray:
    """``frame: 00000001-00001717 rest`` lines (1-based, inclusive) -> per-frame class index."""
    out = np.full(n_frames, IGNORE, np.int64)
    for line in Path(txt).read_text().splitlines():
        m = re.match(r"\s*frame:\s*(\d+)-(\d+)\s+(\S+)", line)
        if m and m.group(3) in MIT_LABELS:
            a, b = int(m.group(1)) - 1, min(int(m.group(2)), n_frames)
            out[a:b] = BEHAVIORS_EXT.index(MIT_LABELS[m.group(3)])
    return out


def build_mit(pipe, root: Path, out: Path, annotators: str = "Annotator_group_1", stride: int = 2,
              only: list[str] | None = None) -> list[str]:
    """Track each MIT video and attach its labels -> ``out/mit/<video>.npz``. ``stride`` 2 = 15 fps."""
    done = []
    for video in sorted(Path(root).rglob("*.mpg")):
        if only and video.stem not in only:
            continue
        txt = next(Path(root).rglob(f"{annotators}/{video.stem}.txt"), None)
        if txt is None:
            continue
        n = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
        labels = mit_labels(txt, n)[::stride]
        coords, conf, boxes = track_main_animal(pipe, video, stride)
        t = min(len(coords), len(labels))
        fps = 30.0 / stride
        save_session(out / "mit" / f"{video.stem}.npz",
                     Session(video.stem, "side", coords[:t], conf[:t], labels[:t], fps), boxes[:t])
        done.append(video.stem)
    return done


# ----------------------------------------------------------------------------- ABODe

def _read_df(path: Path):
    """ABODe ``.df`` files are pandas pickles (bz2). Pickles can run code: only open the files
    downloaded from the DataShare record, verified by ``animalpose data download``."""
    import pandas as pd

    try:
        return pd.read_pickle(path, compression="bz2")
    except OSError:
        return pd.read_pickle(path, compression=None)


def abode_boxes(identities, n_frames: int) -> dict[str, np.ndarray]:
    """Identities.df -> ``{mouse: (n_frames, 4) xyxy}`` (NaN where absent)."""
    frame_level = [i for i, n in enumerate(identities.index.names or []) if n and "frame" in str(n).lower()]
    frames = (identities.index.get_level_values(frame_level[0]).to_numpy() if frame_level
              else np.arange(len(identities)))
    frames = frames - frames.min()
    out = {}
    for mouse in dict.fromkeys(identities.columns.get_level_values(0)):
        xywh = identities[mouse].to_numpy(float).reshape(len(identities), -1)[:, :4]
        box = np.full((n_frames, 4), np.nan)
        keep = frames < n_frames
        box[frames[keep]] = np.c_[xywh[:, :2], xywh[:, :2] + xywh[:, 2:4]][keep]
        out[str(mouse)] = box
    return out


def build_abode(pipe, root: Path, out: Path, stride: int = 2, only: list[str] | None = None) -> list[str]:
    """Pose for every mouse inside its annotated box; labels per second -> per frame.
    ``out/abode/<cage>_<segment>_<snippet>_<mouse>.npz``; the animal group is the cage."""
    ann = _read_df(next(Path(root).rglob("Annotations.df")))
    done = []
    for snip_dir in sorted(p.parent for p in Path(root).rglob("Snippets/*/Video.mp4")):
        if only and snip_dir.name not in only:
            continue
        cage, seg, snip = snip_dir.name.split("_")
        key = next((k for k in ann.index.droplevel(-1).unique() if [str(x) for x in k] == [cage, seg, snip]), None)
        if key is None:
            continue
        lab = ann.loc[key]  # rows: BTI (1 s), columns: (mouse, field)
        n = int(cv2.VideoCapture(str(snip_dir / "Video.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        boxes = abode_boxes(_read_df(snip_dir / "Identities.df"), n)
        frames = list(read_frames(snip_dir / "Video.mp4", stride))
        for mouse, box in boxes.items():
            coords = np.full((len(frames), 25, 2), np.nan, np.float32)
            conf = np.zeros((len(frames), 25), np.float32)
            for i, (t, img) in enumerate(frames):
                if np.isfinite(box[t]).all() and (box[t][2:] > box[t][:2]).all():
                    kp, cf, _ = pipe.estimate(img, box[t][None])
                    coords[i], conf[i] = kp[0], cf[0]
            beh = lab[(mouse, "GT.Behaviour")].astype(str).to_numpy() if (mouse, "GT.Behaviour") in lab.columns else []
            per_frame = np.full(n, IGNORE, np.int64)
            for sec, b in enumerate(beh):
                if b in ABODE_LABELS:
                    per_frame[sec * 25:(sec + 1) * 25] = BEHAVIORS_EXT.index(ABODE_LABELS[b])
            labels = per_frame[[t for t, _ in frames]]
            save_session(out / "abode" / f"{snip_dir.name}_{mouse}.npz",
                         Session(f"cage{cage}", "side", coords, conf, labels, 25.0 / stride), box[[t for t, _ in frames]])
        done.append(snip_dir.name)
    return done


# ----------------------------------------------------------------------------- BORIS-labelled videos

def boris_labels(boris: Path, video_name: str, n_frames: int, fps: float) -> np.ndarray | None:
    """Per-frame labels for one media file of a BORIS project (state events toggle on/off;
    times of multi-media observations are offset by the preceding media lengths)."""
    proj = json.loads(Path(boris).read_text())
    for ob in proj.get("observations", {}).values():
        media = [p for v in ob.get("file", {}).values() for p in v]
        names = [Path(p).name for p in media]
        if video_name not in names:
            continue
        lengths = ob.get("media_info", {}).get("length", {})
        offset = sum(float(lengths.get(p, 0.0)) for p in media[:names.index(video_name)])
        out = np.full(n_frames, IGNORE, np.int64)
        start: dict[str, float] = {}
        states = {v["code"] for v in proj.get("behaviors_conf", {}).values() if "State" in v.get("type", "")}
        for ev in sorted(ob.get("events", []), key=lambda e: e[0]):
            t, code = float(ev[0]) - offset, ev[2]
            if code not in states:
                continue
            if code in start:
                a, b = int(start.pop(code) * fps), int(t * fps)
                cls = BEHAVIORS_EXT.index(BORIS_LABELS.get(code, "other"))
                lo, hi = max(a, 0), min(b + 1, n_frames)
                if lo < hi:  # eating overrides "other" where both are coded
                    seg = out[lo:hi]
                    seg[(seg == IGNORE) | (cls == BEHAVIORS_EXT.index("eating"))] = cls
            else:
                start[code] = t
        return out
    return None


def build_boris_videos(pipe, videos: list[Path], boris_dir: Path | None, out: Path, stride: int = 1) -> list[str]:
    """Track the listed videos (read only); attach BORIS labels where a project covers them."""
    done = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
        cap.release()
        labels = None
        for b in sorted(Path(boris_dir).rglob("*.boris")) if boris_dir else []:
            labels = boris_labels(b, video.name, n, fps)
            if labels is not None:
                break
        labels = np.full(n, IGNORE, np.int64) if labels is None else labels
        coords, conf, boxes = track_main_animal(pipe, video, stride)
        t = min(len(coords), len(labels[::stride]))
        animal = video.name.split("-")[0]
        save_session(out / "boris" / f"{video.stem}.npz",
                     Session(animal, "cage", coords[:t], conf[:t], labels[::stride][:t], fps / stride), boxes[:t])
        done.append(video.stem)
    return done
