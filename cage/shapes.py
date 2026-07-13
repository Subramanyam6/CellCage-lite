"""Cage-shape geometry: how a `CageSpec` becomes concrete shapes for each rule.

The placement algorithm is written once, in terms of "enclosure" and "exclusion"
regions and a cage-overlap test. This module is the only place that knows whether
those regions are disks (circular cage) or hexagons (hexagonal cage). Switching
shape is therefore a configuration change, not a rewrite, exactly as the README
describes.

For a hexagon of circumradius ``R``, the perpendicular (inradius) distance from
center to an edge is ``R * cos(30 deg)``. Growing or shrinking a hexagon by a
perpendicular distance ``d`` changes its circumradius by ``d / cos(30 deg)``;
that factor is why the hexagon formulas below divide by ``COS30``.
"""

from __future__ import annotations

import math

from .geometry import COS30, Convex, Disk, convex_overlap, regular_hexagon
from .types import Cage, CageSpec, Cell


def _margin(spec: CageSpec, confidence: float) -> float:
    """Exclusion margin for a non-target, widened when the classifier is unsure."""
    return spec.exclusion_margin + spec.uncertainty_scale * (1.0 - confidence) * spec.radius


def enclosure_shape(spec: CageSpec, cell: Cell) -> Convex | None:
    """Region of cage centers that enclose ``cell`` with clearance, or ``None``
    if the cell is too large for this cage.

    For a circle this is a disk of radius ``r - w - rho - delta``. For a hexagon
    it is the interior hexagon eroded by the cell radius plus clearance.
    """
    if spec.shape == "circle":
        r = spec.enclosure_radius(cell.radius)
        return Disk(cell.x, cell.y, r) if r > 0.0 else None
    circumradius = spec.radius - (spec.wall + cell.radius + spec.clearance) / COS30
    return regular_hexagon(cell.x, cell.y, circumradius, spec.orientation) if circumradius > 0.0 else None


def exclusion_shape(spec: CageSpec, cell: Cell) -> Convex:
    """Region of cage centers that would trap non-target ``cell`` (to subtract).

    For a circle this is a disk of radius ``r + rho_n + margin``. For a hexagon it
    is the cage hexagon grown outward by ``rho_n + margin`` (a conservative offset
    polygon, so the engine never leaves a non-target trapped).
    """
    margin = _margin(spec, cell.confidence)
    if spec.shape == "circle":
        return Disk(cell.x, cell.y, spec.radius + cell.radius + margin)
    circumradius = spec.radius + (cell.radius + margin) / COS30
    return regular_hexagon(cell.x, cell.y, circumradius, spec.orientation)


def enclosure_reach(spec: CageSpec, cell_radius: float) -> float:
    """Bounding radius of the enclosure region, for sizing a neighbor query."""
    if spec.shape == "circle":
        return max(0.0, spec.enclosure_radius(cell_radius))
    return max(0.0, spec.radius - (spec.wall + cell_radius + spec.clearance) / COS30)


def exclusion_reach(spec: CageSpec, other_radius: float, confidence: float) -> float:
    """Bounding radius of an exclusion region, for sizing a neighbor query."""
    margin = _margin(spec, confidence)
    if spec.shape == "circle":
        return spec.radius + other_radius + margin
    return spec.radius + (other_radius + margin) / COS30


def outer_shape(spec: CageSpec, x: float, y: float) -> Convex:
    """The cage's outer boundary centered at ``(x, y)``."""
    if spec.shape == "circle":
        return Disk(x, y, spec.radius)
    return regular_hexagon(x, y, spec.radius, spec.orientation)


def interior_shape(spec: CageSpec, x: float, y: float) -> Convex:
    """The cage's inner boundary (inside the wall) centered at ``(x, y)``."""
    if spec.shape == "circle":
        return Disk(x, y, spec.radius - spec.wall)
    return regular_hexagon(x, y, spec.radius - spec.wall / COS30, spec.orientation)


def cages_overlap(spec: CageSpec, a: Cage, b: Cage, shrink: float = 0.0) -> bool:
    """Whether two placed cages overlap.

    Circles overlap when their centers are closer than ``2r``. Hexagons are
    tested exactly with the separating-axis theorem. ``shrink`` pulls both outer
    boundaries in slightly, so two cages that merely touch are not counted as
    overlapping (used by the validator to avoid flagging exact contact).
    """
    if spec.shape == "circle":
        min_sep = 2.0 * (spec.radius - shrink)
        return math.hypot(a.x - b.x, a.y - b.y) < min_sep
    r = spec.radius - shrink
    return convex_overlap(
        regular_hexagon(a.x, a.y, r, spec.orientation),
        regular_hexagon(b.x, b.y, r, spec.orientation),
    )
