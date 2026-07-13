"""Geometry for the cage-placement engine.

The heart of Part 1 of the algorithm is finding, for one target, the point in
its *legal region* with the most clearance to every edge: the Chebyshev center,
the center of the largest circle that fits inside the region. Placing the cage
there makes it maximally robust to a small error in the detected outlines.

The legal region for a circular cage is a disk (the enclosure) minus a set of
disks (the exclusions). That region is generally non-convex, so its Chebyshev
center has no closed form. We locate it with a best-first spatial subdivision
(the "polylabel" method): recursively split the bounding box into cells, keep an
upper bound on the best clearance reachable inside each cell, and only refine the
cells that can still beat the incumbent. It returns a point within a set
precision of the true optimum and never gets trapped in a local maximum.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Disk:
    """A disk in image coordinates."""

    x: float
    y: float
    r: float


@dataclass(frozen=True)
class FeasibleRegion:
    """A target's legal region: inside every enclosure disk and outside every
    exclusion disk.

    ``clearance(p)`` is the signed distance from ``p`` to the nearest region
    boundary: positive inside the region, negative outside. The Chebyshev center
    is the point that maximizes it.
    """

    enclosure: tuple[Disk, ...]
    exclusion: tuple[Disk, ...]

    def clearance(self, x: float, y: float) -> float:
        """Signed clearance at a point.

        - To stay inside an enclosure disk of radius ``R`` centered at ``c``:
          margin is ``R - dist(p, c)`` (shrinks to zero at the boundary).
        - To stay outside an exclusion disk of radius ``S`` centered at ``e``:
          margin is ``dist(p, e) - S``.

        The region clearance is the tightest of these, so the largest circle
        that fits at ``p`` has exactly this radius.
        """
        best = math.inf
        for d in self.enclosure:
            margin = d.r - math.hypot(x - d.x, y - d.y)
            if margin < best:
                best = margin
        for d in self.exclusion:
            margin = math.hypot(x - d.x, y - d.y) - d.r
            if margin < best:
                best = margin
        return best

    def bounding_box(self) -> tuple[float, float, float, float] | None:
        """Axis-aligned box that contains the region, or ``None`` if the
        enclosure disks do not even share a common box (region is empty).

        The region lies inside every enclosure disk, so it lies inside the
        intersection of their bounding boxes.
        """
        if not self.enclosure:
            return None
        minx = max(d.x - d.r for d in self.enclosure)
        maxx = min(d.x + d.r for d in self.enclosure)
        miny = max(d.y - d.r for d in self.enclosure)
        maxy = min(d.y + d.r for d in self.enclosure)
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
        # Enclosure disks do not overlap: no point can enclose the whole target.
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
