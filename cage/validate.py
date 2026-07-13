"""Independent checker for a set of cage placements.

The placement engine is built to satisfy the three validity rules by
construction. This module verifies them after the fact against the raw geometry,
for either cage shape. It is what the tests assert on and what the benchmark uses
to report a constraint-violation rate; keeping it separate from the engine's
placement path means a bug there cannot hide itself here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .shapes import cages_overlap, interior_shape, outer_shape
from .types import Cage, CageSpec, Cell

# Tolerance for floating-point comparisons and the Chebyshev-center precision.
_TOL = 0.25


@dataclass
class Violation:
    """A single broken rule."""

    rule: str  # "enclosure" | "exclusion" | "collision"
    detail: str


@dataclass
class ValidationReport:
    """Outcome of checking a full placement."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.valid


def validate_placement(
    cells: Sequence[Cell],
    cages: Sequence[Cage],
    spec: CageSpec,
    tol: float = _TOL,
) -> ValidationReport:
    """Check every cage against the three validity rules.

    1. Enclosure: each of a cage's target cells fits inside the wall.
    2. Exclusion: no non-target cell is trapped inside a cage.
    3. Collision: no two cages overlap.
    """
    report = ValidationReport()
    by_id = {c.id: c for c in cells}
    non_targets = [c for c in cells if c.label != "target"]

    # Rule 1: enclosure. Each target cell (a disk of radius rho) must sit inside
    # the cage interior, i.e. the interior boundary is at least rho away.
    for cage in cages:
        interior = interior_shape(spec, cage.x, cage.y)
        for cid in cage.target_ids:
            cell = by_id.get(cid)
            if cell is None:
                continue
            margin = interior.signed_distance(cell.x, cell.y)
            if margin < cell.radius - tol:
                report.violations.append(
                    Violation(
                        "enclosure",
                        f"cage {cage.center} does not enclose target cell {cid} "
                        f"(interior margin {margin:.2f} < cell radius {cell.radius:.2f})",
                    )
                )

    # Rule 2: exclusion. A non-target that is a member of a cage's own target is
    # allowed inside that cage (co-caging); every other cage must keep it fully
    # outside the outer wall.
    for cage in cages:
        outer = outer_shape(spec, cage.x, cage.y)
        for nt in non_targets:
            if nt.id in cage.target_ids:
                continue
            outside_margin = -outer.signed_distance(nt.x, nt.y)
            if outside_margin < nt.radius - tol:
                report.violations.append(
                    Violation(
                        "exclusion",
                        f"cage {cage.center} traps non-target cell {nt.id} "
                        f"(clearance {outside_margin:.2f} < cell radius {nt.radius:.2f})",
                    )
                )

    # Rule 3: collision. No two cages may overlap (touching is allowed, so the
    # boundaries are pulled in by `tol` before the test).
    for i in range(len(cages)):
        for j in range(i + 1, len(cages)):
            if cages_overlap(spec, cages[i], cages[j], shrink=tol):
                report.violations.append(
                    Violation(
                        "collision",
                        f"cages {cages[i].center} and {cages[j].center} overlap",
                    )
                )

    return report
