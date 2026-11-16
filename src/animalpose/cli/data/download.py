"""Download the public datasets into data/raw/, verifying every file's published MD5.

Sources are resolved through the figshare and Zenodo APIs at run time (see
docs/DATASETS.md for licenses: CC BY 4.0, except iRodent (Apache-2.0) and the MIT
home-cage videos, which state no license and ask for a citation of Jhuang et al. 2010).
Files that already exist with the right checksum are skipped, so this command
can be re-run after an interruption. Archives are extracted into
``<dataset>/extracted/``.

    animalpose data download --all          # ~13 GB
    animalpose data download rat7m_s2 irodent
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

NAME = "download"

RAT7M_MOCAP = 13739233  # figshare article: motion capture + camera parameters
RAT7M_VIDEOS = {"s1-d1": 13751023, "s2-d1": 13753417}
RAT7M_CHUNKS = (0, 17500, 35000, 52500, 70000, 87500, 105000, 122500)  # chunk start frames used here

DATASETS = {
    "crim13": ("figshare", 24993384, "lp_crim13", lambda n: n == "crim13.zip"),
    "mirror_mouse": ("figshare", 24993315, "lp_mirror_mouse", lambda n: n == "mirror-mouse.zip"),
    "rat7m_s1": ("rat7m", "s1-d1", "rat7m", None),
    "rat7m_s2": ("rat7m", "s2-d1", "rat7m", None),
    "irodent": ("zenodo", 8250392, "irodent", lambda n: n.endswith(".tar.gz")),
    "mit_homecage": ("direct", None, "mit_homecage", None),
    "abode": ("direct", None, "abode", None),
}

# Sources without published checksums: the expected size is checked and the SHA-256 printed (record it).
DIRECT = {
    # Jhuang et al. 2010, cbmm.mit.edu/mouse-dataset. No license stated for the videos; cite the paper.
    "mit_homecage": [{"name": "full_database.zip", "size": 886_000_000,
                      "url": "https://www.dropbox.com/s/1vo4jcdbslgsfzb/full_database.zip?dl=1"}],
    # ABODe, Edinburgh DataShare 10.7488/ds/3833, CC BY 4.0 (Camilleri, Williams, Bains 2023).
    "abode": [
        {"name": "Annotations.df", "size": 14_247,
         "url": "https://datashare.ed.ac.uk/bitstreams/ea0c5254-000d-42fe-81a3-692bd16bd64d/download"},
        {"name": "Datasplits.df", "size": None,
         "url": "https://datashare.ed.ac.uk/bitstreams/4ab36ed7-44d4-4cef-aa67-d2f2fc6f9eb0/download"},
        {"name": "README.md", "size": None,
         "url": "https://datashare.ed.ac.uk/bitstreams/c205c586-8a63-4a69-a22a-34b77dd25d44/download"},
        {"name": "Snippets.zip", "size": 5_049_000_000,
         "url": "https://datashare.ed.ac.uk/bitstreams/1d40c69a-427c-4153-9806-c5984fbfb8ec/download"},
    ],
}


def _json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def figshare_files(article: int) -> list[dict]:
    files = _json(f"https://api.figshare.com/v2/articles/{article}/files?page_size=1000")
    return [{"name": f["name"], "url": f["download_url"], "md5": f["computed_md5"], "size": f["size"]} for f in files]


def zenodo_files(record: int) -> list[dict]:
    rec = _json(f"https://zenodo.org/api/records/{record}")
    return [{"name": f["key"], "url": f["links"]["self"], "md5": f["checksum"].split(":", 1)[1], "size": f["size"]}
            for f in rec["files"]]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(f: dict, dest: Path) -> bool:
    """Download unless present with the right MD5; returns True when the file is good."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5(dest) == f["md5"]:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(f["url"], tmp)
    if md5(tmp) != f["md5"]:
        print(f"MD5 MISMATCH {dest.name}", flush=True)
        return False
    tmp.rename(dest)
    print(f"ok {dest} ({f['size'] / 1e6:.0f} MB)", flush=True)
    return True


def fetch_unverified(f: dict, dest: Path) -> bool:
    """For sources without MD5s: size within 1% of the expected one, SHA-256 printed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(f["url"], tmp)
        tmp.rename(dest)
    size = dest.stat().st_size
    if f["size"] and abs(size - f["size"]) > 0.01 * f["size"]:
        print(f"SIZE MISMATCH {dest.name}: {size} bytes, expected about {f['size']}", flush=True)
        return False
    h = hashlib.sha256()
    with open(dest, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    print(f"ok {dest} ({size / 1e6:.0f} MB) sha256 {h.hexdigest()}", flush=True)
    return True


def extract(archive: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        return
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(out, filter="data")
            except TypeError:  # Python without tarfile filters (< 3.10.12 / 3.11.4)
                t.extractall(out)


def add_arguments(parser) -> None:
    parser.add_argument("datasets", nargs="*", help=f"any of {', '.join(DATASETS)} (default: all)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--raw", default="data/raw")


def run(args) -> None:
    names = list(DATASETS) if args.all or not args.datasets else args.datasets
    unknown = set(names) - set(DATASETS)
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)}")
    raw, bad = Path(args.raw), []
    for name in names:
        kind, ident, folder, pick = DATASETS[name]
        out = raw / folder
        if kind == "rat7m":
            mocap = [f for f in figshare_files(RAT7M_MOCAP) if f["name"] == f"mocap-{ident}.mat"]
            videos = [f for f in figshare_files(RAT7M_VIDEOS[ident])
                      if any(f["name"].endswith(f"-{c}.mp4") for c in RAT7M_CHUNKS)]
            bad += [f["name"] for f in mocap if not fetch(f, out / f["name"])]
            bad += [f["name"] for f in videos if not fetch(f, out / ident / f["name"])]
            continue
        if kind == "direct":
            for f in DIRECT[name]:
                if not fetch_unverified(f, out / f["name"]):
                    bad.append(f["name"])
                elif f["name"].endswith(".zip"):
                    extract(out / f["name"], out / "extracted")
            continue
        files = figshare_files(ident) if kind == "figshare" else zenodo_files(ident)
        for f in files:
            if pick(f["name"]):
                if fetch(f, out / f["name"]):
                    extract(out / f["name"], out / "extracted")
                else:
                    bad.append(f["name"])
    print("FAILED: " + ", ".join(bad) if bad else "all downloads verified")
    raise SystemExit(1 if bad else 0)
