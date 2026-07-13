"""Geometry for the cage-placement engine.

The heart of Part 1 of the algorithm is finding, for one target, the point in
its *legal region* with the most clearance to every edge: the Chebyshev center,
the center of the largest circle that fits inside the region. Placing the cage
there makes it maximally robust to a small error in the detected outlines.

The legal region is described as an intersection of "must be inside" convex
shapes (the enclosure) minus a set of "must be outside" convex shapes (the
exclusions). Each shape exposes one thing: a signed distance to its boundary,
positive inside. That single abstraction lets the same region code, and the same
Chebyshev search, serve both circular cages (the shapes are disks) and hexagonal
cages (the shapes are convex polygons), with no branching in the search itself.

The region is generally non-convex, so its Chebyshev center has no closed form.
We locate it with a best-first spatial subdivision (the "polylabel" method):
recursively split the bounding box into cells, keep an upper bound on the best
clearance reachable inside each cell, and only refine the cells that can still
beat the incumbent. It returns a point within a set precision of the true
optimum and never gets trapped in a local maximum.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_SQRT2 = math.sqrt(2.0)
COS30 = math.sqrt(3.0) / 2.0  # inradius / circumradius for a regular hexagon


@runtime_checkable
class Convex(Protocol):
    """A convex shape that can report a signed distance and a bounding box."""

    def signed_distance(self, x: float, y: float) -> float:
        """Distance from ``(x, y)`` to the boundary, positive inside the shape."""

    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds ``(minx, miny, maxx, maxy)``."""


@dataclass(frozen=True)
class Disk:
    """A disk in image coordinates."""

    x: float
    y: float
    r: float

    def signed_distance(self, x: float, y: float) -> float:
        return self.r - math.hypot(x - self.x, y - self.y)

    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x - self.r, self.y - self.r, self.x + self.r, self.y + self.r)


@dataclass(frozen=True)
class ConvexPolygon:
    """A convex polygon given by vertices in counter-clockwise order."""

    vertices: tuple[tuple[float, float], ...]

    def signed_distance(self, x: float, y: float) -> float:
        verts = self.vertices
        n = len(verts)
        max_half_plane = -math.inf  # >0 means outside that edge
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            # Outward normal of a CCW edge is (ey, -ex).
            length = math.hypot(ex, ey)
            nx, ny = ey / length, -ex / length
            d = (x - ax) * nx + (y - ay) * ny
            if d > max_half_plane:
                max_half_plane = d
        if max_half_plane <= 0.0:
            # Inside: distance to the nearest edge.
            return -max_half_plane
        # Outside: exact distance to the polygon's nearest edge segment.
        return -_distance_to_boundary(x, y, verts)

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))


def _distance_to_boundary(x: float, y: float, verts: tuple[tuple[float, float], ...]) -> float:
    """Shortest distance from a point to a polygon's boundary."""
    n = len(verts)
    best = math.inf
    for i in range(n):
        d = _point_segment_distance(x, y, verts[i], verts[(i + 1) % n])
        if d < best:
            best = d
    return best


def _point_segment_distance(px, py, a, b) -> float:
    ax, ay = a
    bx, by = b
    ex, ey = bx - ax, by - ay
    denom = ex * ex + ey * ey
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * ex + (py - ay) * ey) / denom
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * ex), py - (ay + t * ey))


def regular_hexagon(cx: float, cy: float, circumradius: float, orientation: float = 0.0) -> ConvexPolygon:
    """A regular hexagon centered at ``(cx, cy)`` with the given circumradius."""
    verts = tuple(
        (
            cx + circumradius * math.cos(orientation + k * math.pi / 3.0),
            cy + circumradius * math.sin(orientation + k * math.pi / 3.0),
        )
        for k in range(6)
    )
    return ConvexPolygon(vertices=verts)


def convex_overlap(a: ConvexPolygon, b: ConvexPolygon) -> bool:
    """Whether two convex polygons overlap, by the separating-axis theorem.

    If any edge normal of either polygon separates their projections, they do
    not overlap; otherwise they do.
    """
    for poly in (a, b):
        verts = poly.vertices
        n = len(verts)
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            axis = (by - ay, -(bx - ax))  # edge normal
            if _separated_on_axis(a.vertices, b.vertices, axis):
                return False
    return True


def _separated_on_axis(va, vb, axis) -> bool:
    ax, ay = axis
    a_proj = [vx * ax + vy * ay for vx, vy in va]
    b_proj = [vx * ax + vy * ay for vx, vy in vb]
    # `<=` so that shapes sharing only a boundary count as separated (touching is
    # allowed, matching the strict-inequality convention used for circles).
    return max(a_proj) <= min(b_proj) or max(b_proj) <= min(a_proj)


@dataclass(frozen=True)
class FeasibleRegion:
    """A target's legal region: inside every enclosure shape and outside every
    exclusion shape.

    ``clearance(p)`` is the signed distance from ``p`` to the nearest region
    boundary: positive inside the region, negative outside. The Chebyshev center
    is the point that maximizes it.
    """

    enclosure: tuple[Convex, ...]
    exclusion: tuple[Convex, ...]

    def clearance(self, x: float, y: float) -> float:
        """Signed clearance at a point: the tightest of the per-shape margins.

        Staying inside an enclosure shape contributes its signed distance
        (positive inside). Staying outside an exclusion shape contributes the
        negated signed distance (positive outside). The largest circle that fits
        at ``p`` has exactly this radius.
        """
        best = math.inf
        for shape in self.enclosure:
            margin = shape.signed_distance(x, y)
            if margin < best:
                best = margin
        for shape in self.exclusion:
            margin = -shape.signed_distance(x, y)
            if margin < best:
                best = margin
        return best

    def bounding_box(self) -> tuple[float, float, float, float] | None:
        """Box that contains the region, or ``None`` if the enclosure shapes do
        not share a common box (region is empty).

        The region lies inside every enclosure shape, so it lies inside the
        intersection of their bounding boxes.
        """
        if not self.enclosure:
            return None
        boxes = [s.bbox() for s in self.enclosure]
        minx = max(b[0] for b in boxes)
        miny = max(b[1] for b in boxes)
        maxx = min(b[2] for b in boxes)
        maxy = min(b[3] for b in boxes)
        if minx > maxx or miny > maxy:
            return None
        return (minx, miny, maxx, maxy)


@dataclass(frozen=True)
class ChebyshevResult:
    """Result of a Chebyshev-center search."""

    x: float
    y: float
    clearance: float

    @property
    def feasible(self) -> bool:
        """Whether the best point actually lies in the region (clearance >= 0).

        A non-negative clearance means a cage placed here is valid; a negative
        one means the region is empty and the target cannot be caged.
        """
        return self.clearance >= 0.0


@dataclass(order=True)
class _Cell:
    """A square cell in the subdivision search.

    Ordered by ``-bound`` so the priority queue (a min-heap) always yields the
    cell with the largest reachable clearance first.
    """

    neg_bound: float
    cx: float = 0.0
    cy: float = 0.0
    half: float = 0.0
    value: float = 0.0


def chebyshev_center(
    region: FeasibleRegion,
    precision: float = 0.05,
    max_iterations: int = 100_000,
) -> ChebyshevResult:
    """Find the point of maximum clearance in ``region`` (its Chebyshev center).

    Uses best-first quadtree subdivision. Each square cell is scored by the
    clearance at its center, and bounded above by that clearance plus the
    distance to the cell's farthest corner (``half * sqrt(2)``), which is the
    most the clearance could improve anywhere inside the cell. Cells that cannot
    beat the incumbent by more than ``precision`` are discarded; the rest are
    split into four. This returns a point within ``precision`` of the true
    optimum without getting stuck in a local maximum.

    Returns the best point found; check ``ChebyshevResult.feasible`` to see
    whether it actually lies in the region.
    """
    box = region.bounding_box()
    if box is None:
        # Enclosure shapes do not overlap: no point can enclose the whole target.
        return ChebyshevResult(0.0, 0.0, -math.inf)

    minx, miny, maxx, maxy = box
    width, height = maxx - minx, maxy - miny
    cell_size = min(width, height)
    if cell_size <= 0.0:
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        return ChebyshevResult(cx, cy, region.clearance(cx, cy))

    def make_cell(cx: float, cy: float, half: float) -> _Cell:
        value = region.clearance(cx, cy)
        bound = value + half * _SQRT2
        return _Cell(neg_bound=-bound, cx=cx, cy=cy, half=half, value=value)

    # Seed with a regular grid covering the box, so the search starts near every
    # part of the region and cannot miss the basin holding the optimum.
    heap: list[_Cell] = []
    half = cell_size / 2.0
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            heapq.heappush(heap, make_cell(x + half, y + half, half))
            y += cell_size
        x += cell_size

    # Incumbent: start from the box center.
    cx0, cy0 = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    best = _Cell(neg_bound=0.0, cx=cx0, cy=cy0, half=0.0, value=region.clearance(cx0, cy0))

    iterations = 0
    while heap and iterations < max_iterations:
        iterations += 1
        cell = heapq.heappop(heap)
        if cell.value > best.value:
            best = cell
        # Discard cells that cannot improve on the incumbent by `precision`.
        if -cell.neg_bound - best.value <= precision:
            continue
        half = cell.half / 2.0
        heapq.heappush(heap, make_cell(cell.cx - half, cell.cy - half, half))
        heapq.heappush(heap, make_cell(cell.cx + half, cell.cy - half, half))
        heapq.heappush(heap, make_cell(cell.cx - half, cell.cy + half, half))
        heapq.heappush(heap, make_cell(cell.cx + half, cell.cy + half, half))

    return ChebyshevResult(best.cx, best.cy, best.value)
