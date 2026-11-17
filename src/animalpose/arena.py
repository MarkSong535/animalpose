"""Arena definitions: what is clicked on a top-view frame, and the geometry built from it.

One YAML file per experiment type in ``configs/arenas/`` is the single source of truth for the
labeller (``scripts/label_arena.py``), the trainer (``scripts/train_arena.py``) and the apps (the
definition travels inside the trained model, ``model_cfg["arena"]``, and the Rust engine builds
the layout from it). A definition holds:

``points``   the names of the points, in saved order.
``order``    how clicks in any order are put into that order: ``box`` (4 corners clockwise from
             the one nearest the image's top-left), ``polygon`` (any number of points clockwise
             around their centre, from the one nearest the top-left), ``three_chamber`` or
             ``plus`` (below).
``floor``    the floor as point references (4 for a box; empty when the arena has no floor quad).
``zones``    named polygons of point references, each with a ``kind`` the behavior modules read
             (``center``, ``chamber``, ``open_arm``, ``closed_arm``, ...).

A point reference is a point number (1-based, as in ``points``) or ``"a-b x c-d"``: where the line
through points a and b crosses the line through c and d.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = ROOT / "configs" / "arenas"
CROSS = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*x\s*(\d+)\s*-\s*(\d+)\s*$")


@dataclass(frozen=True)
class Arena:
    name: str
    points: list[str]
    order: str
    floor: list = field(default_factory=list)
    zones: list[dict] = field(default_factory=list)
    ask: str = ""
    experiment: str = ""
    sources: dict = field(default_factory=dict)

    def as_model_cfg(self, input_size: tuple[int, int]) -> dict:
        """What a trained model carries (``model_cfg["arena"]``) for the apps to build layouts."""
        return {
            "name": self.name,
            "experiment": self.experiment,
            "points": self.points,
            "floor": self.floor,
            "zones": self.zones,
            "input_size": list(input_size),
        }

    def ordered(self, pts) -> list[tuple[float, float]]:
        """The points in saved order, whatever order they were clicked in (``ValueError`` when
        they cannot be ordered)."""
        if len(pts) != len(self.points):
            raise ValueError(f"{len(pts)} points, {len(self.points)} needed")
        return ORDERS[self.order]([(float(x), float(y)) for x, y in pts])

    def resolve(self, ref, pts) -> tuple[float, float]:
        """A point reference against points in saved order."""
        return resolve(ref, pts)

    def layout(self, pts) -> dict:
        """Floor and zones (lists of [x, y]) from points in saved order."""
        return {
            "floor": [list(resolve(r, pts)) for r in self.floor],
            "zones": [
                {"name": z["name"], "kind": z["kind"], "polygon": [list(resolve(r, pts)) for r in z["polygon"]]}
                for z in self.zones
            ],
        }


def load(name: str | Path, root: Path = DEFINITIONS) -> Arena:
    """The definition called ``name`` (``configs/arenas/<name>.yaml``) or at a path; checked."""
    path = Path(name) if str(name).endswith((".yaml", ".yml")) else root / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())
    arena = Arena(
        name=raw["name"],
        points=list(raw["points"]),
        order=raw.get("order", "polygon"),
        floor=list(raw.get("floor") or []),
        zones=[dict(z) for z in raw.get("zones") or []],
        ask=raw.get("ask", ""),
        experiment=raw.get("experiment", raw["name"]),
        sources=dict(raw.get("sources") or {}),
    )
    check(arena, path)
    return arena


def available(root: Path = DEFINITIONS) -> list[str]:
    return sorted(p.stem for p in root.glob("*.yaml"))


def check(arena: Arena, where) -> None:
    n = len(arena.points)
    if arena.order not in ORDERS:
        raise ValueError(f"{where}: order {arena.order!r} is not one of {sorted(ORDERS)}")
    fixed = {"box": 4, "three_chamber": 8, "plus": 8}.get(arena.order)
    if fixed and n != fixed:
        raise ValueError(f"{where}: order {arena.order!r} needs {fixed} points, the definition has {n}")
    if arena.floor and len(arena.floor) != 4:
        raise ValueError(f"{where}: the floor needs 4 point references, it has {len(arena.floor)}")
    refs = list(arena.floor) + [r for z in arena.zones for r in z.get("polygon", [])]
    for z in arena.zones:
        if not {"name", "kind", "polygon"} <= set(z) or len(z["polygon"]) < 3:
            raise ValueError(f"{where}: zone {z} needs a name, a kind and at least 3 points")
    for r in refs:
        for i in indices(r):
            if not 1 <= i <= n:
                raise ValueError(f"{where}: point {i} in {r!r} is not one of the {n} points")


def indices(ref) -> list[int]:
    if isinstance(ref, int):
        return [ref]
    m = CROSS.match(str(ref))
    if not m:
        raise ValueError(f"{ref!r} is neither a point number nor 'a-b x c-d'")
    return [int(g) for g in m.groups()]


def resolve(ref, pts) -> tuple[float, float]:
    idx = indices(ref)
    p = [np.asarray(pts[i - 1], float) for i in idx]
    if len(p) == 1:
        return tuple(map(float, p[0]))
    a, b, c, d = p
    u, v = b - a, d - c
    det = u[0] * (-v[1]) - u[1] * (-v[0])
    if abs(det) < 1e-9:  # parallel: the midpoint of the four
        return tuple(map(float, np.mean(p, axis=0)))
    s = ((c - a)[0] * (-v[1]) - (c - a)[1] * (-v[0])) / det
    return tuple(map(float, a + s * u))


# ---------------------------------------------------------------- click order


def shoelace(poly) -> float:
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])))


def nearest_on_segment(q, a, b) -> tuple[float, float]:
    a, b, q = (np.asarray(v, float) for v in (a, b, q))
    d = b - a
    t = 0.0 if not d.any() else float(np.clip(np.dot(q - a, d) / np.dot(d, d), 0.0, 1.0))
    return tuple(map(float, a + t * d))


def segment_distance(q, a, b) -> float:
    return float(np.linalg.norm(np.subtract(q, nearest_on_segment(q, a, b))))


def clockwise(pts) -> list[int]:
    """Indices clockwise on screen (y down) around the centre, from the point nearest the
    image's top-left."""
    p = np.asarray(pts, float)
    c = p.mean(axis=0)
    order = list(np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])))
    start = min(range(len(order)), key=lambda k: p[order[k]].sum())
    return order[start:] + order[:start]


def order_polygon(pts):
    return [pts[i] for i in clockwise(pts)]


def order_three_chamber(pts):
    """Outer corners (the 4 points spanning the largest area), then the two walls. Each wall end
    belongs to the outer edge it is nearest to and is moved onto it. Corner 1 is chosen so the
    walls run from edge 1-2 to edge 4-3 (of the two such corners, the one nearer the image's
    top-left); wall 1 is the one nearer corner 1, each from its end on edge 1-2."""
    corners = max(itertools.combinations(range(8), 4), key=lambda idx: shoelace(order_polygon([pts[i] for i in idx])))
    ring = order_polygon([pts[i] for i in corners])
    ends = [pts[i] for i in range(8) if i not in corners]
    edges = [(ring[k], ring[(k + 1) % 4]) for k in range(4)]
    side = [min(range(4), key=lambda e: segment_distance(q, *edges[e])) for q in ends]
    used = sorted(set(side))
    if used not in ([0, 2], [1, 3]) or side.count(used[0]) != 2:
        raise ValueError("the wall ends must sit two on one outer edge and two on the opposite edge")
    # Corner 1 starts an edge holding wall ends; take the candidate nearer the image's top-left.
    start = min((used[0], used[1]), key=lambda k: sum(ring[k]))
    ring = ring[start:] + ring[:start]
    edges = [(ring[k], ring[(k + 1) % 4]) for k in range(4)]
    side = [min((0, 2), key=lambda e: segment_distance(q, *edges[e])) for q in ends]
    ends = [nearest_on_segment(q, *edges[e]) for q, e in zip(ends, side)]
    along = np.subtract(ring[1], ring[0])
    pos = lambda q: float(np.dot(np.subtract(q, ring[0]), along))  # noqa: E731
    a = sorted((q for q, e in zip(ends, side) if e == 0), key=pos)
    b = sorted((q for q, e in zip(ends, side) if e == 2), key=pos)
    return ring + [a[0], b[0], a[1], b[1]]


def order_plus(pts):
    """The 8 outer corners clockwise from the arm nearest the top: consecutive corners around the
    centre pair up into arm ends (the pairing with the shorter ends), and the arm whose end is most
    nearly straight up from the centre is north."""
    p = np.asarray(pts, float)
    c = p.mean(axis=0)
    order = list(np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])))
    width = lambda o: sum(np.linalg.norm(p[o[2 * k]] - p[o[2 * k + 1]]) for k in range(4))  # noqa: E731
    if width(order[1:] + order[:1]) < width(order):
        order = order[1:] + order[:1]
    pairs = [(order[2 * k], order[2 * k + 1]) for k in range(4)]
    widths = [np.linalg.norm(p[a] - p[b]) for a, b in pairs]
    gaps = [np.linalg.norm(p[pairs[k][1]] - p[pairs[(k + 1) % 4][0]]) for k in range(4)]
    if max(widths) >= min(gaps):
        raise ValueError("the corners do not pair up into 4 arm ends")
    mids = [(p[a] + p[b]) / 2 - c for a, b in pairs]
    north = int(np.argmax([-m[1] / (np.linalg.norm(m) + 1e-9) for m in mids]))
    pairs = pairs[north:] + pairs[:north]
    return [tuple(map(float, p[i])) for pair in pairs for i in pair]


ORDERS = {
    "box": order_polygon,
    "polygon": order_polygon,
    "three_chamber": order_three_chamber,
    "plus": order_plus,
}
