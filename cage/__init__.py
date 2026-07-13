"""CellCage-lite constrained cage-placement engine.

The core of the project: given labeled cells and a cage specification, place the
maximum number of valid cages over the targets. See the project README, section
3, for the algorithm this implements.
"""

from __future__ import annotations

from .geometry import (
    ChebyshevResult,
    ConvexPolygon,
    Disk,
    FeasibleRegion,
    chebyshev_center,
    convex_overlap,
    regular_hexagon,
)
from .placement import place_cages, place_cages_greedy_baseline
from .types import Cage, CageSpec, Cell, Target, build_targets
from .validate import ValidationReport, Violation, validate_placement

__all__ = [
    "Cell",
    "CageSpec",
    "Cage",
    "Target",
    "build_targets",
    "place_cages",
    "place_cages_greedy_baseline",
    "validate_placement",
    "ValidationReport",
    "Violation",
    "Disk",
    "ConvexPolygon",
    "regular_hexagon",
    "convex_overlap",
    "FeasibleRegion",
    "ChebyshevResult",
    "chebyshev_center",
]
