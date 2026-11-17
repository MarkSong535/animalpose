"""Build `samples/`: the sample profiles the desktop apps install on first launch.

Each profile is one experiment setup from the lab's recordings: a standalone pipeline config (the
repository's `configs/pipeline.yaml` plus the setup's `assay:` section), the fine-tuned networks,
the setup's arena detector (scripts/train_arena.py) and, where it has one, its calibrated camera
pair. `manifest.json` lists every file with its SHA-256; the apps and `animalpose samples check`
refuse a file whose hash differs (for example a git-lfs pointer).

    python scripts/build_samples.py --models <dir with detector_ft.pt, pose_*.pt, temporal_real.pt> \\
        --calibration <dir with <rig>_cams.json and <rig>_calib.json> --arena <dir with arena_<type>.pt>

Run it again when new networks, detectors or calibrations arrive: a setup missing its calibration
(or, without one, its arena detector) is skipped and listed. The output replaces `samples/`.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

# (profile name, rig, assay experiment = arena type, folder on larcc, has a calibrated camera pair).
# Setups of one experiment were filmed with the cameras placed differently (each has its own
# calibration), so each is its own profile, named after its recording batch.
SETUPS = [
    ("Three Chamber (Pre-Cd F2)", "3_Chamber", "three_chamber", "~/upload/Pre-Cd F2 13wks/3 Chamber", True),
    (
        "Three Chamber (Pre-Cd F2 Video)",
        "3_Chambers_Video",
        "three_chamber",
        "~/upload/Pre-Cd F2 13wks/3 Chambers Video",
        True,
    ),
    ("Open Field (Pre-Cd F2 Video)", "Open_Field_Video", "open_field", "~/upload/Pre-Cd F2 13wks/Open Field Video", True),
    ("Three Chamber (Batch 2)", "Three_Chamber", "three_chamber", "~/upload2/Three Chamber", True),
    ("Grooming", "grooming", "grooming", "~/upload2/grooming", True),
    ("Open Field (Batch 2)", "opf", "open_field", "~/upload2/opf", True),
    ("Plus Maze", "plus_maze", "plus_maze", "~/upload3/plus maze", False),
]

# The networks, newest first: the first file found for each kind is bundled, under the second name.
NETWORKS = {
    "detector": [("detector_ft.pt", "detector-ft")],
    "pose": [("pose_mv2.pt", "pose-mv2"), ("pose_mv.pt", "pose-mv"), ("pose_ft.pt", "pose-ft")],
    "temporal": [("temporal_real.pt", "temporal-real")],
    "behavior": [("behavior_cai3d.pt", "behavior-3d"), ("behavior_cai.pt", "behavior-2d")],
}

# Every bundled file name starts with this, so samples never collide with a user's own files.
PREFIX = "sample-"

# The default cup radius when a calibration has none (box units).
CUP_RADIUS = 0.08

# samples/ is stored in plain git, not git-lfs (.gitattributes); GitHub warns above 50 MB a file.
MAX_BYTES = 50_000_000


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path):
    with path.open("rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def bundle(source, dest):
    """Copies a checkpoint, saved again so its archive is named after the bundled file rather than
    the training run (torch names the archive's folder after the file it was saved as)."""
    import torch

    torch.save(torch.load(source, map_location="cpu", weights_only=False), dest)


def config_for(base, rig, experiment, calib):
    config = {key: value for key, value in base.items() if key not in ("checkpoints", "extends")}
    config["checkpoints"] = {kind: None for kind in ("detector", "pose", "temporal", "behavior", "arena")}
    tracker = dict(config.get("tracker", {}))
    # Three-chamber side views also see the caged mice; the assay picks the test mouse.
    tracker["max_animals"] = 3 if experiment == "three_chamber" else 1
    if tracker["max_animals"] == 1:
        # One free animal, fixed cameras: a detection on part of the scene (a table leg, a wall
        # bracket) differs nowhere from the median background and is dropped. Without this the
        # lost animal is re-identified with that detection and stays on it. Not with caged mice,
        # which sit in one place all session and so belong to the median background.
        config["pipeline"] = {**config.get("pipeline", {}), "background_filter": True}
    config["tracker"] = tracker
    assay = {
        "experiment": experiment,
        "arena": {
            "rig": rig,
            "width": round(float(calib["box_width"]) / float(calib.get("box_length", 1.0)), 4) if calib else 1.0,
            "top_view": "top",
            "side_view": "side",
        },
    }
    cups = (calib or {}).get("cups_xy") or {}
    if experiment == "three_chamber" and cups:
        assay["cups"] = [
            {
                "name": name,
                "center": [round(float(x), 4), round(float(y), 4)],
                "radius": CUP_RADIUS,
                "height": round(float(calib.get("cup_height", 0.15)), 4),
            }
            for name, (x, y) in sorted(cups.items())
        ]
    config["assay"] = assay
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--arena", type=Path, help="trained arena detectors, arena_<type>.pt (scripts/train_arena.py)")
    parser.add_argument("--out", type=Path, default=SAMPLES)
    args = parser.parse_args()

    networks, bundled = {}, {}
    for kind, names in NETWORKS.items():
        found = next(((args.models / n, b) for n, b in names if (args.models / n).is_file()), None)
        if found is None:
            if kind in ("detector", "pose"):
                looked = ", ".join(n for n, _ in names)
                parser.error(f"no {kind} network in {args.models} (looked for {looked})")
            continue
        found, bundled[kind] = found
        if is_lfs_pointer(found):
            parser.error(f"{found} is a git-lfs pointer, not the network")
        if found.stat().st_size > MAX_BYTES:
            size = found.stat().st_size / 1e6
            parser.error(f"{found} is {size:.0f} MB; samples are stored in plain git, keep each under 50 MB")
        networks[kind] = found

    base = yaml.safe_load((ROOT / "configs/pipeline.yaml").read_text())
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    files = []

    def add(rel, write):
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path)
        files.append({"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size})

    model_names = {}
    for kind, source in networks.items():
        name = f"{PREFIX}{bundled[kind]}.pt"
        model_names[kind] = name
        add(f"models/{kind}/{name}", lambda p, s=source: bundle(s, p))
    arena_names = {}  # arena type -> bundled detector
    for _, _, experiment, _, _ in SETUPS:
        source = args.arena / f"arena_{experiment}.pt" if args.arena else None
        if source is None or experiment in arena_names or not source.is_file():
            continue
        if source.stat().st_size > MAX_BYTES or is_lfs_pointer(source):
            parser.error(f"{source} is not a network under 50 MB")
        arena_names[experiment] = f"{PREFIX}arena-{experiment.replace('_', '-')}.pt"
        add(f"models/arena/{arena_names[experiment]}", lambda p, s=source: bundle(s, p))

    profiles, skipped = [], []
    for profile_name, rig, experiment, folder, calibrated in SETUPS:
        cams = args.calibration / f"{rig}_cams.json"
        calib = args.calibration / f"{rig}_calib.json"
        if calibrated and not (cams.is_file() and calib.is_file()):
            skipped.append(f"{profile_name}: no {cams.name} / {calib.name} yet")
            continue
        if not calibrated and experiment not in arena_names:
            skipped.append(f"{profile_name}: no calibration and no arena detector")
            continue
        slug = PREFIX + rig.lower().replace("_", "-")
        calib_data = json.loads(calib.read_text()) if calibrated else None
        if calibrated:
            names = [c.get("name") for c in json.loads(cams.read_text())]
            if names != ["top", "side"]:
                parser.error(f"{cams} cameras are {names}, expected [top, side]")
        config = config_for(base, slug, experiment, calib_data)
        header = (
            f"# {profile_name}: {folder} (larcc). Standalone: no `extends`, the networks come from the profile.\n"
            f"# Multiview: put each top/side pair in its own folder as top.mp4 and side.mp4.\n"
        )
        add(f"configs/{slug}.yaml", lambda p, c=config: p.write_text(header + yaml.safe_dump(c, sort_keys=False)))
        if calibrated:
            add(f"cameras/{slug}.json", lambda p, s=cams: shutil.copy2(s, p))
        models = dict(model_names)
        if experiment in arena_names:
            models["arena"] = arena_names[experiment]
        profile = {
            "name": profile_name,
            "config": f"{slug}.yaml",
            "models": models,
            "classify_behavior": "behavior" in model_names,
            "camera_model": f"{slug}.json" if calibrated else None,
            "views": {},
        }
        file_name = profile_name.replace(" ", "-")
        add(f"profiles/{file_name}.json", lambda p, d=profile: p.write_text(json.dumps(d, indent=2) + "\n"))
        profiles.append(profile_name)

    manifest = {
        "version": 1,
        "about": "Sample profiles from the lab's recordings (larcc). Built by scripts/build_samples.py.",
        "networks": model_names,
        "arena_detectors": arena_names,
        "profiles": profiles,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = "\n".join(f"| {name} | {folder} | {exp} |" for name, _, exp, folder, _ in SETUPS if name in profiles)
    networks_text = ", ".join(f"{kind}: {name}" for kind, name in model_names.items())
    networks_text += "; arena detectors: " + (", ".join(arena_names.values()) or "none")
    pending = "".join(f"\n- {line}" for line in skipped) or "\n- none"
    (out / "README.md").write_text(
        "# Sample profiles\n\n"
        "Installed into `~/AnimalPose` by the desktop apps on launch (see the root `README.md`).\n"
        "Generated by `scripts/build_samples.py`; do not edit by hand. `manifest.json` lists every file and its SHA-256.\n\n"
        "| Profile | Recordings | Experiment |\n| --- | --- | --- |\n"
        f"{rows}\n\nNetworks: {networks_text}.\n\nNot bundled yet:{pending}\n"
    )
    print(f"{len(profiles)} profiles, {len(files)} files in {out}")
    for line in skipped:
        print(f"skipped {line}")


if __name__ == "__main__":
    main()
