"""Arena definitions (configs/arenas) and the click ordering of animalpose.arena."""

import math
import random

import pytest

from animalpose import arena

BOX = [(170, 114), (455, 113), (456, 349), (169, 349)]
THREE_CHAMBER = BOX + [(265, 113.7), (266, 349), (362, 113.3), (361, 349)]
PLUS = [(250, 70), (275, 70), (420, 235), (420, 262), (275, 420), (250, 420), (112, 262), (112, 235)]


def rotate(pts, deg, c=(320, 240)):
    cos, sin = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(c[0] + (x - c[0]) * cos - (y - c[1]) * sin, c[1] + (x - c[0]) * sin + (y - c[1]) * cos) for x, y in pts]


def same(a, b, tol=1e-6):
    return all(math.dist(p, q) <= tol for p, q in zip(a, b))


def test_every_definition_loads_and_checks():
    names = arena.available()
    assert {"three_chamber", "open_field", "grooming", "plus_maze"} <= set(names)
    for name in names:
        defn = arena.load(name)
        assert defn.points and defn.order in arena.ORDERS


@pytest.mark.parametrize(
    "name, pts",
    [
        ("open_field", BOX),
        ("three_chamber", THREE_CHAMBER),
        ("three_chamber", rotate(THREE_CHAMBER, 8)),
        ("plus_maze", PLUS),
        ("plus_maze", rotate(PLUS, 30)),
    ],
)
def test_any_click_order_gives_one_saved_order(name, pts):
    defn = arena.load(name)
    ref = defn.ordered(pts)
    rng = random.Random(0)
    for _ in range(300):
        assert same(defn.ordered(rng.sample(pts, len(pts))), ref)
    assert same(defn.ordered(ref), ref)  # idempotent


def test_three_chamber_walls_land_on_the_perimeter_and_chambers_follow_the_walls():
    defn = arena.load("three_chamber")
    clicked = THREE_CHAMBER[:4] + [(265, 118), (266, 345), (362, 110), (361, 352)]
    pts = defn.ordered(clicked)
    for end, (a, b) in zip(pts[4:], [(0, 1), (3, 2), (0, 1), (3, 2)]):
        assert arena.segment_distance(end, pts[a], pts[b]) < 1e-9
    # Walls across the box the other way: the same zone template still gives three chambers.
    sideways = [(200, 60), (440, 60), (440, 420), (200, 420), (200, 180), (440, 180), (200, 300), (440, 300)]
    zones = defn.layout(defn.ordered(sideways))["zones"]
    areas = [arena.shoelace([tuple(p) for p in z["polygon"]]) for z in zones]
    assert all(a == pytest.approx(240 * 120) for a in areas)


def test_plus_maze_centre_is_where_the_two_strips_overlap():
    zones = {z["name"]: z for z in arena.load("plus_maze").layout(PLUS)["zones"]}
    assert same(zones["center"]["polygon"], [(275, 235), (275, 262), (250, 262), (250, 235)])
    assert zones["arm_north"]["kind"] == "closed_arm" and zones["arm_east"]["kind"] == "open_arm"


def test_impossible_sets_are_refused():
    with pytest.raises(ValueError, match="wall ends"):
        arena.load("three_chamber").ordered(BOX + [(265, 114), (265, 200), (362, 114), (362, 200)])
    with pytest.raises(ValueError, match="points"):
        arena.load("open_field").ordered(BOX[:3])


def test_definitions_are_checked(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: bad\norder: box\npoints: [a, b, c, d]\nzones:\n  - {name: z, kind: k, polygon: [1, 2, 9]}\n")
    with pytest.raises(ValueError, match="point 9"):
        arena.load(bad)
    bad.write_text("name: bad\norder: plus\npoints: [a, b, c, d]\n")
    with pytest.raises(ValueError, match="needs 8 points"):
        arena.load(bad)
