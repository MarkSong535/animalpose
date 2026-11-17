"""Click arenas on top-view frames, one video at a time, for the arena detector.

What is clicked, the order the points are saved in and the zones built from them come from the
arena definition, `configs/arenas/<type>.yaml` (see `animalpose.arena`). Click the points in any
order: they are put into the saved order before saving, and a set that cannot be ordered is
refused with the reason.

Frames: 30 per experiment folder (`--frames`), spread evenly over its videos and counting frames
already labelled, or the definition's `sources.per_video` from each video. They come from the
top-view videos on larcc named by the definition's `sources`, or with `--videos a.mp4 …` from
local files (a new setup needs no `sources`). Only one video's frames are on this computer at a
time; they are deleted before the next video's are fetched. Labels are saved after every frame
(`arena_labels/<type>.json`, copied to larcc), so you can quit and resume.

Keys:  Enter/Space accept and next (the points carry over) · d next without saving · a previous
       u/Backspace/right-click undo the last click · c clear · r back to the carried-over points
       x frame unusable (skipped, recorded) · v skip the rest of this video
       m swap the open and closed arms (arenas that have both) · q/Esc save and quit
Mouse: one click per point, in any order. With points carried over from the previous frame
       (hollow), a click moves the nearest one not yet clicked on this frame; Enter keeps the rest.
       The panel on the right magnifies the cursor 4x, its cross exactly where a click lands.

    python3 scripts/label_arena.py all                      # every defined type in turn
    python3 scripts/label_arena.py three_chamber
    python3 scripts/label_arena.py my_maze --videos a.mp4 b.mp4   # configs/arenas/my_maze.yaml
    python3 scripts/label_arena.py all --check              # reorder saved labels, list outliers
    python3 scripts/label_arena.py all --export --zones ~/AnimalPose/zones   # <rig>.json per setup
"""

import argparse
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from animalpose import arena as arenas  # noqa: E402

HOST = "z0song04@larcc.hpc.louisville.edu"
CONTROL = "/tmp/larcc-cm.sock"
REMOTE_PY = "~/animal/animalpose/.venv/bin/python"
REMOTE_TMP = "arena_label_tmp"
ORDER = ["plus_maze", "three_chamber", "open_field", "grooming"]  # `all`: these first

SCALE = 1.5  # window pixels per frame pixel
OUTLIER_PX = 10.0  # a frame this far from its video's median is listed by the check


def swap_arms(zones):
    """Zones with the open and closed arms exchanged."""
    flip = {"open_arm": "closed_arm", "closed_arm": "open_arm"}
    return [{**z, "kind": flip.get(z["kind"], z["kind"])} for z in zones]


# ---------------------------------------------------------------- frames


def ssh(args, command, **kw):
    return subprocess.run(
        ["ssh", "-o", f"ControlPath={args.control}", "-o", "BatchMode=yes", args.host, command], check=True, **kw
    )


def remote_videos(args, defn):
    """(experiment folder, video path) of every top-view video the definition's `sources` names."""
    src = defn.sources
    if src.get("folder"):
        folder = src["folder"]
        out = ssh(args, f"ls -1 {shlex.quote(folder)}", capture_output=True).stdout.decode()
        names = [n for n in out.split("\n") if n.endswith(".mp4") and not n.startswith("._")]
        names.sort(key=lambda n: (len(n), n))  # Test 2 before Test 10
        return [(Path(folder).name, f"{folder}/{n}") for n in names]
    if not src.get("experiments"):
        sys.exit(f"configs/arenas/{defn.name}.yaml names no sources: pass --videos <files>")
    script = (
        "import sys, json; sys.path.insert(0, '/home/z0song04/upload/pose_3d')\n"
        "from paths3d import videos, is_top\n"
        f"exps = {src['experiments']!r}\n"
        "print(json.dumps([[e, str(v)] for e in exps for v in videos(e) if is_top(e, v.stem)]))\n"
    )
    out = ssh(args, f"{REMOTE_PY} -", input=script.encode(), capture_output=True).stdout.decode()
    return [(exp, path) for exp, path in json.loads(out.strip().splitlines()[-1])]


def fetch_remote(args, path, count, dest):
    """`count` evenly spaced frames of one video on larcc into `dest` (JPEG), through a tar stream;
    the remote copies are removed."""
    script = (
        "import cv2, numpy as np, os\n"
        f"cap = cv2.VideoCapture({path!r}); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        f"os.makedirs(os.path.expanduser('~/{REMOTE_TMP}'), exist_ok=True)\n"
        f"for i in sorted(set(np.linspace(0.05 * n, 0.95 * n, {count}).astype(int).tolist())):\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()\n"
        f"    ok and cv2.imwrite(os.path.expanduser('~/{REMOTE_TMP}/%07d.jpg' % i), f, [cv2.IMWRITE_JPEG_QUALITY, 92])\n"
    )
    ssh(args, f"rm -rf ~/{REMOTE_TMP} && {REMOTE_PY} -", input=script.encode(), capture_output=True)
    tar = ssh(args, f"tar -C ~/{REMOTE_TMP} -cf - . && rm -rf ~/{REMOTE_TMP}", capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as archive:
        for member in archive.getmembers():  # only the JPEGs, by bare file name
            name = Path(member.name).name
            if member.isfile() and name.endswith(".jpg"):
                (dest / name).write_bytes(archive.extractfile(member).read())
    return sorted(dest.glob("*.jpg"))


def fetch_local(path, count, dest):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in sorted(set(np.linspace(0.05 * n, 0.95 * n, count).astype(int).tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(dest / f"{i:07d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return sorted(dest.glob("*.jpg"))


def allocate(defn, plan, labels, per_experiment):
    """Frames to fetch per video: `sources.per_video` from each, or each experiment folder's budget
    minus what is already labelled there, spread evenly over its unfinished videos."""
    per_video = defn.sources.get("per_video")
    counts = [0] * len(plan)
    for exp in dict.fromkeys(e for e, _ in plan):
        idx = [i for i, (e, path) in enumerate(plan) if e == exp and not labels.video_done(e, path)]
        if per_video:
            for i in idx:
                counts[i] = per_video
            continue
        left = per_experiment - labels.frames_in(exp)
        if left <= 0 or not idx:
            continue
        base, extra = divmod(left, len(idx))
        if base == 0:  # fewer frames than videos: one each from evenly spaced videos
            idx = [idx[int(k * len(idx) / left)] for k in range(left)]
            base, extra = 1, 0
        for k, i in enumerate(idx):
            counts[i] = base + (1 if k < extra else 0)
    return counts


# ---------------------------------------------------------------- labels


class Labels:
    def __init__(self, path, defn):
        self.path, self.defn = path, defn
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault("type", defn.name)
        self.data["points"] = defn.points
        self.data.setdefault("videos", {})

    def video(self, key):
        return self.data["videos"].setdefault(key, {"frames": {}, "unusable": [], "done": False})

    def video_done(self, exp, path):
        return self.data["videos"].get(f"{exp}/{Path(path).stem}", {}).get("done", False)

    def frames_in(self, exp):
        return sum(len(v["frames"]) for k, v in self.data["videos"].items() if k.split("/", 1)[0] == exp)

    def frames_total(self):
        return sum(len(v["frames"]) for v in self.data["videos"].values())

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)

    def check(self):
        """Puts every saved frame into the saved order; lists frames that cannot be ordered or sit
        more than OUTLIER_PX from their video's median, and videos shifted as a whole from the rest
        of their setup."""
        adjusted, problems = 0, []
        for key, video in self.data["videos"].items():
            for frame_id, pts in list(video["frames"].items()):
                try:
                    fixed = [list(q) for q in self.defn.ordered(pts)]
                except ValueError as err:
                    problems.append(f"{key} frame {frame_id}: {err}")
                    continue
                adjusted += not np.allclose(fixed, pts)
                video["frames"][frame_id] = fixed
            if len(video["frames"]) >= 3:
                arr = np.asarray(list(video["frames"].values()), float)
                med = np.median(arr, axis=0)
                for frame_id, pts in video["frames"].items():
                    off = float(np.linalg.norm(np.asarray(pts) - med, axis=1).max())
                    if off > OUTLIER_PX:
                        problems.append(f"{key} frame {frame_id}: a point is {off:.0f} px from the video's median")
        # A video whose points all sit apart from the other videos of its experiment folder (same
        # camera setup): a shifted labelling, or a camera that moved — either way, look again.
        by_exp = {}
        for key, video in self.data["videos"].items():
            if video["frames"]:
                med = np.median(np.asarray(list(video["frames"].values()), float), axis=0)
                by_exp.setdefault(key.split("/", 1)[0], {})[key] = med
        for videos in by_exp.values():
            for key, med in videos.items():
                rest = [m for k, m in videos.items() if k != key]
                if len(rest) >= 2:
                    off = np.linalg.norm(med - np.median(rest, axis=0), axis=1)
                    if off.min() > OUTLIER_PX:
                        problems.append(f"{key}: every point is {off.min():.0f}+ px from the other videos of this setup")
        self.save()
        return adjusted, problems


# ---------------------------------------------------------------- window


class Window:
    """One click per point, in any order. Points carried over from the previous frame are hollow;
    a click moves the nearest one not yet clicked on this frame. Once all points are set, they are
    shown numbered in the saved order with the floor and zones they make."""

    NAME = "Arena labels"
    BAR = 48  # text bar above the frame (window pixels)
    PANEL = 260  # magnifier panel right of the frame (window pixels)

    def __init__(self, defn):
        self.defn = defn
        self.points, self.carried, self.clicked, self.history = [], [], set(), []
        self.mouse = None  # frame pixels, None outside the frame
        self.size = (0, 0)
        cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.NAME, self.on_mouse)

    def start(self, points):
        self.carried = [tuple(p) for p in (points or [])][: len(self.defn.points)]
        self.points = list(self.carried)
        self.clicked, self.history = set(), []

    def clear(self):
        self.carried, self.points, self.clicked, self.history = [], [], set(), []

    def ordered(self):
        try:
            return self.defn.ordered(self.points), None
        except ValueError as err:
            return None, str(err)

    def click(self, q):
        if len(self.points) < len(self.defn.points):
            self.points.append(q)
            i, old = len(self.points) - 1, None
        else:
            free = [i for i in range(len(self.points)) if i not in self.clicked] or range(len(self.points))
            i = min(free, key=lambda j: math.dist(self.points[j], q))
            old, self.points[i] = self.points[i], q
        self.history.append((i, old))
        self.clicked.add(i)

    def undo(self):
        if self.history:
            i, old = self.history.pop()
            self.clicked.discard(i)
            if old is None:
                del self.points[i]
            else:
                self.points[i] = old

    def on_mouse(self, event, x, y, flags, _):
        fx, fy = x / SCALE, (y - self.BAR) / SCALE  # window pixels -> frame pixels
        inside = 0 <= fx < self.size[0] and 0 <= fy < self.size[1]
        self.mouse = (fx, fy) if inside else None
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self.click((fx, fy))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def draw(self, frame, status, swapped):
        self.size = (frame.shape[1], frame.shape[0])
        img = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        clean = img.copy()
        green, yellow, cyan, red, orange = (80, 220, 80), (0, 220, 255), (255, 220, 0), (60, 60, 255), (0, 140, 255)
        at = lambda q: (int(round(q[0] * SCALE)), int(round(q[1] * SCALE)))  # noqa: E731
        n = len(self.defn.points)
        ordered, problem = self.ordered() if len(self.points) == n else (None, None)
        if ordered:
            layout = self.defn.layout(ordered)
            zones = swap_arms(layout["zones"]) if swapped else layout["zones"]
            for z in zones:
                colour = orange if z["kind"] == "closed_arm" else cyan
                cv2.polylines(img, [np.int32([at(q) for q in z["polygon"]])], True, colour, 1, cv2.LINE_AA)
            if layout["floor"]:
                cv2.polylines(img, [np.int32([at(q) for q in layout["floor"]])], True, green, 1, cv2.LINE_AA)
            for k, q in enumerate(ordered):
                cv2.putText(img, str(k + 1), (at(q)[0] + 6, at(q)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
        for i, q in enumerate(self.points):
            cv2.circle(img, at(q), 4, yellow, -1 if i in self.clicked else 1, cv2.LINE_AA)
        if len(self.points) < n:
            prompt = f"Click {self.defn.ask or 'the points'}, any order: {len(self.points)}/{n}"
        elif problem:
            prompt = f"Cannot order these points: {problem}. Click to move one, or c to clear"
        elif self.carried and len(self.clicked) < n:
            prompt = f"Click to move points ({len(self.clicked)} moved); Enter keeps the rest as shown"
        else:
            prompt = "All points set: Enter to accept"
        if any(z["kind"] in ("open_arm", "closed_arm") for z in self.defn.zones):
            prompt += "   [orange = closed arms, m swaps]"
        h, w = img.shape[:2]
        canvas = np.zeros((self.BAR + h, w + self.PANEL, 3), np.uint8)
        canvas[self.BAR :, :w] = img
        cv2.putText(canvas, prompt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        side = self.PANEL - 20
        if self.mouse is not None:  # 4x magnifier, its cross exactly where a click lands
            cx, cy = self.mouse[0] * SCALE, self.mouse[1] * SCALE
            r = side / 8
            x0, y0 = int(math.floor(cx - r)), int(math.floor(cy - r))
            pad = cv2.copyMakeBorder(clean, side, side, side, side, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            crop = pad[y0 + side : y0 + side + int(2 * r), x0 + side : x0 + side + int(2 * r)]
            zoom = cv2.resize(crop, (side, side), interpolation=cv2.INTER_NEAREST)
            mx, my = int(round((cx - x0) * side / (2 * r))), int(round((cy - y0) * side / (2 * r)))
            cv2.drawMarker(zoom, (mx, my), red, cv2.MARKER_CROSS, 30, 1)
            canvas[self.BAR + 10 : self.BAR + 10 + side, w + 10 : w + 10 + side] = zoom
        cv2.rectangle(canvas, (w + 10, self.BAR + 10), (w + 10 + side, self.BAR + 10 + side), (120, 120, 120), 1)
        for k, name in enumerate(self.defn.points):
            colour = (230, 230, 230) if ordered else (130, 130, 130)
            y = self.BAR + side + 40 + 18 * k
            cv2.putText(canvas, f"{k + 1}. {name}", (w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        return canvas


# ---------------------------------------------------------------- export


def rig_name(folder, prefix=""):
    """The rig name of an experiment folder: the file the assay looks up is `<zones>/<rig>.json`,
    with the rig spelled as the assay spells file names (letters, digits, `-` and `_`)."""
    slug = re.sub(r"[^0-9A-Za-z_-]", "-", (prefix + folder).lower())[:64].strip("-")
    return slug or "arena"


def export(labels, out_dir, prefix=""):
    """One hand-made layout per experiment folder (`"source": "manual"`, which the assay always
    uses): the median of its labelled frames, with the definition's floor and zones. Set the
    setup's `assay.arena.rig` to the printed rig name and put the file in the zones folder the
    assay reads (`~/AnimalPose/zones` for the apps, `<analysis>/../zones` for `animalpose assay`)."""
    defn = labels.defn
    by_exp = {}
    for key, video in labels.data["videos"].items():
        for pts in video["frames"].values():
            by_exp.setdefault(key.split("/", 1)[0], []).append((pts, video))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for exp, frames in sorted(by_exp.items()):
        pts = defn.ordered(np.median(np.asarray([p for p, _ in frames], float), axis=0).tolist())
        swapped = sum(bool(v.get("swapped")) for _, v in frames) > len(frames) / 2
        layout = defn.layout(pts)
        rig = rig_name(exp, prefix)
        doc = {
            "rig": rig,
            "method": 0,
            "source": "manual",
            "status": "ok",
            "reason": f"median of {len(frames)} hand-labelled frames (scripts/label_arena.py, {defn.name})",
            "image_size": next((v["size"] for _, v in frames if v.get("size")), [640, 480]),
            "floor": layout["floor"],
            "zones": swap_arms(layout["zones"]) if swapped else layout["zones"],
            "wall_scores": {},
            "finder": "manual",
        }
        path = out_dir / f"{rig}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        written.append((path, len(frames)))
    return written


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("type", help=f"arena type ({', '.join(arenas.available())}), a definition .yaml, or all")
    parser.add_argument("--frames", type=int, default=30, help="frames per experiment folder (default 30)")
    parser.add_argument("--videos", nargs="+", type=Path, help="label these local videos instead of larcc's")
    parser.add_argument("--out", type=Path, default=ROOT / "arena_labels")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--control", default=CONTROL, help="ssh ControlPath of a logged-in connection")
    parser.add_argument("--check", action="store_true", help="reorder saved labels, list frames to recheck, exit")
    parser.add_argument("--export", action="store_true", help="write <zones>/<rig>.json from the labels and exit")
    parser.add_argument("--zones", type=Path, help="where --export writes [default: <out>/zones]")
    parser.add_argument("--rig-prefix", default="", help="prefix of the exported rig names (the samples use sample-)")
    args = parser.parse_args()
    names = sorted(arenas.available(), key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    defs = [arenas.load(n) for n in names] if args.type == "all" else [arenas.load(args.type)]
    if args.videos and len(defs) > 1:
        parser.error("--videos labels one type: name it instead of all")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.check or args.export:
        for defn in defs:
            path = args.out / f"{defn.name}.json"
            if not path.exists():
                continue
            labels = Labels(path, defn)
            adjusted, problems = labels.check()
            print(f"{defn.name}: {labels.frames_total()} frames, {adjusted} adjusted, {len(problems)} to recheck")
            for line in problems:
                print(f"  {line}")
            if args.export:
                for out, n in export(labels, args.zones or args.out / "zones", args.rig_prefix):
                    print(f"  {out}  ({n} frames; assay.arena.rig: {out.stem})")
        return
    if not args.videos and not Path(args.control).exists():
        sys.exit(f"log in to larcc first (ssh ControlMaster socket {args.control} not found)")
    try:
        for defn in defs:
            if label_type(args, defn):
                break
    finally:
        cv2.destroyAllWindows()


def label_type(args, defn):
    """Labels one arena type; True when the person quit."""
    labels = Labels(args.out / f"{defn.name}.json", defn)
    adjusted, problems = labels.check()
    if adjusted or problems:
        print(f"{defn.name}: {adjusted} saved frames adjusted; {len(problems)} to recheck (--check lists them)")
    remote = not args.videos
    plan = remote_videos(args, defn) if remote else [(p.parent.name, str(p)) for p in args.videos]
    counts = allocate(defn, plan, labels, args.frames)
    for exp in dict.fromkeys(e for e, _ in plan):
        todo = sum(c for (e, _), c in zip(plan, counts) if e == exp)
        print(f"{defn.name} / {exp}: {labels.frames_in(exp)} frames labelled, {todo} to fetch")
    print(f"labels in {labels.path}")

    window, carried, quit_all = None, None, False
    try:
        for v_index, (exp, path) in enumerate(plan):
            key = f"{exp}/{Path(path).stem}"
            if labels.video_done(exp, path):
                carried = next(iter(labels.video(key)["frames"].values()), carried)
                continue
            if counts[v_index] == 0:
                continue
            video = labels.video(key)
            work = Path(tempfile.mkdtemp(prefix="arena_frames_"))
            try:
                print(f"[{v_index + 1}/{len(plan)}] fetching {counts[v_index]} frames of {key} …", flush=True)
                if remote:
                    files = fetch_remote(args, path, counts[v_index], work)
                else:
                    files = fetch_local(path, counts[v_index], work)
                if not files:
                    print(f"  no frames read from {key}; skipped")
                    continue
                window = window or Window(defn)
                first = cv2.imread(str(files[0]))
                video["size"] = [first.shape[1], first.shape[0]]
                ids = [f.stem.lstrip("0") or "0" for f in files]
                i = next((j for j, f in enumerate(ids) if f not in video["frames"] and f not in video["unusable"]), 0)
                while 0 <= i < len(files):
                    frame_id, frame = ids[i], cv2.imread(str(files[i]))
                    window.start(video["frames"].get(frame_id) or carried)
                    where = f"{key}  video {v_index + 1}/{len(plan)}  frame {i + 1}/{len(files)}"
                    action = label_frame(window, labels, video, frame_id, frame, where)
                    if action == "quit":
                        raise KeyboardInterrupt
                    i = {"next": i + 1, "back": max(0, i - 1), "skip": len(files)}[action]
                    carried = video["frames"].get(frame_id, carried)
                video["done"] = True
                labels.save()
            finally:
                shutil.rmtree(work, ignore_errors=True)  # only one video's frames here at a time
            if remote:
                upload(args, labels.path)
    except KeyboardInterrupt:
        quit_all = True
    finally:
        labels.save()
        if remote:
            upload(args, labels.path)
    print(f"{defn.name}: {labels.frames_total()} frames labelled in {labels.path}")
    return quit_all


def label_frame(window, labels, video, frame_id, frame, where):
    """Shows one frame until a key moves on, recording accepted points or an unusable frame in
    `video`. Returns what to do next: "next", "back", "skip" (the rest of the video) or "quit"."""
    keys = "Enter accept · a/d prev/next · u undo · c clear · x unusable · v skip video · q quit"
    while True:
        state = "accepted" if frame_id in video["frames"] else ("unusable" if frame_id in video["unusable"] else "new")
        status = f"{where} (#{frame_id}, {state})  ·  {labels.frames_total()} labelled  ·  {keys}"
        cv2.imshow(Window.NAME, window.draw(frame, status, video.get("swapped", False)))
        code = cv2.waitKeyEx(20)
        if code == -1:
            continue
        k = code & 0xFF
        if k in (13, 10, 32):
            ordered, problem = window.ordered()
            if len(window.points) == len(window.defn.points) and not problem:
                video["frames"][frame_id] = [list(q) for q in ordered]
                if frame_id in video["unusable"]:
                    video["unusable"].remove(frame_id)
                labels.save()
                return "next"
        elif k == ord("d"):
            return "next"
        elif k == ord("a"):
            return "back"
        elif k in (ord("u"), 8, 127):
            window.undo()
        elif k == ord("c"):
            window.clear()
        elif k == ord("r"):
            window.start(window.carried)
        elif k == ord("x"):
            if frame_id not in video["unusable"]:
                video["unusable"].append(frame_id)
            video["frames"].pop(frame_id, None)
            labels.save()
            return "next"
        elif k == ord("m"):
            video["swapped"] = not video.get("swapped", False)
            labels.save()
        elif k == ord("v"):
            return "skip"
        elif k in (ord("q"), 27):
            return "quit"


def upload(args, path):
    """A copy of the labels on larcc (~/upload/arena_labels/), where the training runs."""
    try:
        ssh(args, "mkdir -p ~/upload/arena_labels", capture_output=True)
        target = f"{args.host}:upload/arena_labels/{path.name}"
        cmd = ["scp", "-q", "-o", f"ControlPath={args.control}", "-o", "BatchMode=yes", str(path), target]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        print(f"could not copy the labels to larcc ({err}); they are saved locally", file=sys.stderr)


if __name__ == "__main__":
    main()
