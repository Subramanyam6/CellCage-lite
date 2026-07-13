"""Independent checker for a set of cage placements.

The placement engine is built to satisfy the three validity rules by
construction. This module verifies them after the fact, from scratch, against
the raw geometry. It is what the tests assert on and what the benchmark uses to
report a constraint-violation rate; keeping it independent of the engine means a
bug in the engine cannot hide itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

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

    # Rule 1: enclosure.
    for cage in cages:
        for cid in cage.target_ids:
            cell = by_id.get(cid)
            if cell is None:
                continue
            limit = spec.enclosure_radius(cell.radius)
            d = math.hypot(cage.x - cell.x, cage.y - cell.y)
            if d > limit + tol:
                report.violations.append(
                    Violation(
                        "enclosure",
                        f"cage {cage.center} does not enclose target cell {cid} "
                        f"(distance {d:.2f} > limit {limit:.2f})",
                    )
                )

    # Rule 2: exclusion. A non-target that is a member of a cage's own target is
    # allowed inside that cage (co-caging); every other cage must keep it clear.
    for cage in cages:
        for nt in non_targets:
            if nt.id in cage.target_ids:
                continue
            keep_out = spec.radius + nt.radius
            d = math.hypot(cage.x - nt.x, cage.y - nt.y)
            if d < keep_out - tol:
                report.violations.append(
                    Violation(
                        "exclusion",
                        f"cage {cage.center} traps non-target cell {nt.id} "
                        f"(distance {d:.2f} < keep-out {keep_out:.2f})",
                    )
                )

    # Rule 3: collision.
    min_sep = spec.collision_distance()
    for i in range(len(cages)):
        for j in range(i + 1, len(cages)):
            a, b = cages[i], cages[j]
            d = math.hypot(a.x - b.x, a.y - b.y)
            if d < min_sep - tol:
                report.violations.append(
                    Violation(
                        "collision",
                        f"cages {a.center} and {b.center} overlap "
                        f"(distance {d:.2f} < min separation {min_sep:.2f})",
                    )
                )

    return report
