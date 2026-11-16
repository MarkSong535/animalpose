"""Convert downloaded public datasets into canonical COCO JSON under data/processed/.

Downloads (CC BY 4.0, figshare; see docs/DATASETS.md) are expected under data/raw/:

    data/raw/lp_mirror_mouse/extracted/mirror-mouse/   (mirror-mouse.zip, Lightning Pose)
    data/raw/lp_crim13/extracted/crim13/               (crim13.zip, Lightning Pose)
    data/raw/rat7m/mocap-s1-d1.mat + data/raw/rat7m/s1-d1/*.mp4   (Rat 7M, subject 1 day 1)
    data/raw/rat7m/mocap-s2-d1.mat + data/raw/rat7m/s2-d1/*.mp4   (Rat 7M, subject 2: test only)
    data/raw/irodent/extracted/                        (irodent.tar.gz, Zenodo 8250392: zero-shot test)

    animalpose data prepare --all
"""

from __future__ import annotations

import json
from pathlib import Path

from animalpose.data.real import convert_coco_keypoints, convert_lp_crim13, convert_mirror_mouse, convert_rat7m

NAME = "prepare"


def add_arguments(parser) -> None:
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--mirror-mouse", action="store_true")
    parser.add_argument("--crim13", action="store_true")
    parser.add_argument("--rat7m", action="store_true")
    parser.add_argument("--rat7m-s2", action="store_true", help="subject 2 day 1, all frames held out")
    parser.add_argument("--irodent", action="store_true")
    parser.add_argument("--rat7m-every", type=int, default=25, help="keep every n-th video frame")


def run(args) -> None:
    raw, out = Path(args.raw), Path(args.out)
    stats = {}
    if args.all or args.mirror_mouse:
        stats["mirror_mouse"] = convert_mirror_mouse(raw / "lp_mirror_mouse/extracted/mirror-mouse", out / "mirror_mouse")
    if args.all or args.crim13:
        stats["crim13"] = convert_lp_crim13(raw / "lp_crim13/extracted/crim13", out / "crim13")
    if args.all or args.rat7m:
        stats["rat7m"] = convert_rat7m(raw / "rat7m/mocap-s1-d1.mat", raw / "rat7m/s1-d1", out / "rat7m", args.rat7m_every)
    if args.all or args.rat7m_s2:
        stats["rat7m_s2"] = convert_rat7m(raw / "rat7m/mocap-s2-d1.mat", raw / "rat7m/s2-d1", out / "rat7m_s2",
                                          args.rat7m_every, val_starts=None)
    if args.all or args.irodent:
        found = sorted((raw / "irodent/extracted").rglob("*.json"))
        anns = [p for p in found if "annotation" in p.name.lower() or "coco" in p.name.lower()] or found
        (out / "irodent").mkdir(parents=True, exist_ok=True)
        stats["irodent"] = {"source": str(anns[0]), **convert_coco_keypoints(anns[0], out / "irodent/test.json")}
        (out / "irodent/IMAGE_ROOT").write_text(str(_image_root(anns[0])))
    print(json.dumps(stats, indent=1))


def _image_root(ann: Path) -> Path:
    """Directory that the annotation's ``file_name`` entries are relative to."""
    first = json.loads(ann.read_text())["images"][0]["file_name"]
    for d in [ann.parent, *ann.parents]:
        if (d / first).exists():
            return d.resolve()
        hits = list(d.rglob(Path(first).name))
        if hits:
            return hits[0].parent.resolve() if Path(first).name == first else d.resolve()
    raise FileNotFoundError(first)
