"""Core data types for the cage-placement engine.

These are deliberately small, immutable value objects. Every downstream stage
(detection, classification, tracking) speaks in terms of `Cell`; the cage engine
adds `CageSpec` (the instrument's fixed cage geometry) and returns `Cage`
placements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

Label = Literal["target", "non-target"]
Shape = Literal["circle", "hexagon"]


@dataclass(frozen=True)
class Cell:
    """A single detected cell.

    Attributes
    ----------
    id:
        Stable identifier assigned by detection.
    x, y:
        Center of the cell in image coordinates (pixels).
    radius:
        Effective radius of the cell. A cell is modeled as a disk of this
        radius for the placement geometry; it is the maximum extent the cage
        must enclose (for a target) or must avoid trapping (for a non-target).
    label:
        Whether classification marked this cell as a target to cage.
    confidence:
        Classifier confidence in ``label`` in ``[0, 1]``. Used to widen the
        exclusion margin around non-targets the classifier is unsure about, so
        the engine hedges on doubtful cells.
    group:
        Optional co-caging group id. Cells sharing a non-``None`` group are
        caged together inside one cage (e.g. an interacting T cell and cancer
        cell). ``None`` means the cell is caged on its own.
    """

    id: int
    x: float
    y: float
    radius: float
    label: Label = "non-target"
    confidence: float = 1.0
    group: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class CageSpec:
    """The instrument's fixed cage geometry and safety margins.

    A cage is a fixed shape (circle or hexagon) of outer ``radius`` with a wall
    of thickness ``wall``. The remaining parameters are the safety clearances
    that make a placement robust to imperfect detection.

    Attributes
    ----------
    shape:
        ``"circle"`` or ``"hexagon"``.
    radius:
        Outer radius of the cage (for a hexagon, the circumradius).
    wall:
        Wall thickness. The usable interior radius is ``radius - wall``.
    clearance:
        Extra gap required between a target cell and the inner wall (``delta``).
    exclusion_margin:
        Extra gap required between the outer wall and any non-target cell it
        must not trap (``epsilon``).
    uncertainty_scale:
        How aggressively to widen the exclusion margin for a non-target the
        classifier is unsure about. The effective margin for a non-target with
        confidence ``c`` is ``exclusion_margin + uncertainty_scale * (1 - c) *
        radius``. Zero disables uncertainty hedging.
    orientation:
        Hexagon rotation in radians (ignored for circles).
    """

    shape: Shape = "circle"
    radius: float = 20.0
    wall: float = 2.0
    clearance: float = 1.0
    exclusion_margin: float = 1.0
    uncertainty_scale: float = 0.0
    orientation: float = 0.0

    def enclosure_radius(self, cell_radius: float) -> float:
        """Max distance a cage center may sit from a target cell of the given
        radius and still enclose it with clearance: ``r - w - rho - delta``.

        A negative value means the cell is too large for this cage.
        """
        return self.radius - self.wall - cell_radius - self.clearance

    def exclusion_radius(self, other_radius: float, confidence: float = 1.0) -> float:
        """Min distance a cage center must keep from a non-target cell so the
        cell stays outside the wall: ``r + rho_n + epsilon``.

        The margin is widened for low-confidence non-targets.
        """
        margin = self.exclusion_margin + self.uncertainty_scale * (1.0 - confidence) * self.radius
        return self.radius + other_radius + margin

    def collision_distance(self) -> float:
        """Min center-to-center distance between two cages that do not overlap.

        For circles this is ``2 * radius``.
        """
        return 2.0 * self.radius


@dataclass(frozen=True)
class Cage:
    """A placed cage: its center, the spec it was built from, and the target
    cells it holds.

    ``clearance`` is the robustness margin achieved at placement time: the
    radius of the largest circle that fits inside the target's legal region
    centered here. A larger value means a small error in the detected outlines
    is less likely to invalidate the cage.
    """

    x: float
    y: float
    spec: CageSpec
    target_ids: tuple[int, ...]
    clearance: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Target:
    """One caging job: the cell or group of cells a single cage must hold."""

    cells: tuple[Cell, ...]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(c.id for c in self.cells)


def build_targets(cells: Sequence[Cell]) -> list[Target]:
    """Split the target cells into caging jobs.

    Cells with the same non-``None`` ``group`` are co-caged into one target;
    every other target cell becomes its own single-cell target. Non-target
    cells are ignored here (they only act as exclusions later).
    """
    singles: list[Target] = []
    groups: dict[int, list[Cell]] = {}
    for cell in cells:
        if cell.label != "target":
            continue
        if cell.group is None:
            singles.append(Target(cells=(cell,)))
        else:
            groups.setdefault(cell.group, []).append(cell)
    grouped = [Target(cells=tuple(members)) for members in groups.values()]
    return singles + grouped
