# Adding an experiment setup

A setup is an arena (what the camera sees from above) plus the behavior rules scored in it. The
arena is described once, in one file; the labeller, the trainer and both desktop apps read that
description, so a new setup needs no code.

| Step | What you do | Result |
| --- | --- | --- |
| 1 | Describe the arena: `configs/arenas/<name>.yaml` | points, their order, floor and zones |
| 2 | Click ~30 frames per camera setup: `scripts/label_arena.py` | `arena_labels/<name>.json` |
| 3 | Train the arena detector: `scripts/train_arena.py` | `arena_<name>.pt` and its accuracy report |
| 4 | Choose the behavior rules: an `assay:` section in a pipeline config | per-frame and summary CSVs |
| 5 | Use it: a profile with the config and the detector | the apps find the arena in every video |

## 1. Describe the arena

```yaml
# configs/arenas/light_dark.yaml — a light–dark box seen from above: 4 corners and the 2 ends of
# the partition (on the top and bottom walls). Clockwise from the top-left corner the points are
# corner, partition end, corner, corner, partition end, corner.
name: light_dark
experiment: light_dark        # the assay experiment this arena serves
ask: the 4 corners and both ends of the partition
order: polygon                # how clicks in any order are put in order (below)
points: [corner top-left, partition top end, corner top-right, corner bottom-right,
         partition bottom end, corner bottom-left]
floor: [1, 3, 4, 6]           # 4 references for a box floor; empty when there is none
zones:
  - {name: dark, kind: chamber, polygon: [1, 2, 5, 6]}
  - {name: light, kind: chamber, polygon: [2, 3, 4, 5]}
sources:                      # optional: where scripts/label_arena.py finds top-view videos on larcc
  folder: /home/z0song04/upload4/light dark
```

- **points**: names in the order they are saved. You click them in any order.
- **order**: `box` (4 corners, clockwise from the one nearest the image's top-left), `polygon` (any
  number of points clockwise around their centre, from the one nearest the top-left — right for
  arenas whose outline goes once around the centre, including points on the walls),
  `three_chamber` or `plus` (see `src/animalpose/arena.py`). Pick one where every clicked set comes
  out in the same order however the arena sits in the image; a new rule is one function in
  `ORDERS` there.
- **Point references** in `floor` and `zones`: a point number (1-based) or `"a-b x c-d"`, where the
  line through points a and b crosses the line through c and d (inner corners of mazes).
- **Zone kinds** are what the behavior modules read: `center`, `open_arm` / `closed_arm` (the plus
  maze; checked against the walls in each video), `chamber`, or any name of your own (the `zones`
  module reports time and entries for every zone by name).

The definition is checked when it loads (`python3 -c "from animalpose import arena; arena.load('light_dark')"`);
add a case to `tests/test_arena.py` with a few clicked sets in shuffled order.

## 2. Label

```bash
python3 scripts/label_arena.py light_dark --videos /path/to/top_view_1.mp4 /path/to/top_view_2.mp4
```

30 frames per experiment folder (camera setup) are enough to start; more setups beat more frames
of one setup. Click the points in any order; the window shows them numbered in the saved order
with the zones they make. `--check` lists frames to look at again. `--export --zones
~/AnimalPose/zones` writes a hand-made layout per camera setup, named after its folder
(`light_dark/` → `light_dark.json`, printed with the `assay.arena.rig` to set); the apps use it as
is, before any training.

## 3. Train

```bash
python scripts/train_arena.py --type light_dark --labels arena_labels --out runs/arena
```

On larcc, `~/upload/arena/train_arena.sbatch` runs it for every type (a GPU, ~4 min per run).
`report.txt` gives the error per point in frame pixels on videos left out of training (`heldout`)
and, with several setups, on a setup left out entirely (`loeo:<setup>`): the second is what a new
lab's camera will see. Under ~5 px is good; a setup with large `loeo` errors needs a few labelled
frames of its own.

## 4. Behavior rules

In the pipeline config (see `configs/assay_example.yaml`):

```yaml
assay:
  experiment: light_dark
  modules: [zones, grooming, rearing]   # presets exist for open_field, plus_maze, three_chamber, grooming
  arena: {rig: light-dark-room-2}
```

The `zones` module writes, per frame, the zone the body is in and — when the arena has a floor —
its position on the floor (`floor_u`, `floor_v`, 0–1) and distance to the wall in body lengths.
A new rule is one `Module` in `rust/src/assay/modules.rs`.

## 5. Use it in the apps

**Settings → Profiles**: pick the config, and under **Arena** the **Arena Detector** (import the
`.pt` with **Settings → General → Import → Model…**). Each processed video's arena is saved with a
picture as `zones/<rig>.<video>.json`; a layout the detector doubts (a point found with low
confidence or moving between frames, a floor that is not a convex quadrilateral) is marked
`needs_review` with the reason, and the run says so. A hand-made `zones/<rig>.json` always wins.

To ship the setup with the apps, add it to `SETUPS` in `scripts/build_samples.py`, rebuild
`samples/`, and run `animalpose samples check` (built with `--features ffi` or `gui`) or `animalpose-gui --check-samples`.
