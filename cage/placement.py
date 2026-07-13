"""The two-pass cage-placement algorithm.

Part 1 finds the single best legal cage for each target on its own. Part 2 keeps
the largest non-overlapping set across all of those cages. This module wires the
geometry (`geometry.py`) and the independent-set solver (`mis.py`) together
behind one entry point, `place_cages`.

The structure mirrors the pseudocode in the project README section 3.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Convex, FeasibleRegion, chebyshev_center
from .mis import SMALL_CLUSTER, connected_components, exact_mis, greedy_then_local_search
from .shapes import (
    cages_overlap,
    enclosure_reach,
    enclosure_shape,
    exclusion_reach,
    exclusion_shape,
)
from .types import Cage, CageSpec, Cell, Target, build_targets


# --------------------------------------------------------------------------- #
# Part 1: the best legal cage for one target
# --------------------------------------------------------------------------- #
def _best_cage_for_target(
    target: Target,
    non_targets: Sequence[Cell],
    nt_tree: cKDTree | None,
    spec: CageSpec,
    precision: float,
    max_nt_radius: float,
    min_nt_confidence: float,
) -> Cage | None:
    """Compute the most robust valid cage for a single target, or ``None`` if
    the target cannot be caged without breaking a rule.

    Mirrors lines 4-10 of the pseudocode: build the enclosure region, subtract
    the nearby non-targets, and place the cage at the region's Chebyshev center.

    ``max_nt_radius`` and ``min_nt_confidence`` size the neighbor query: they give
    the widest exclusion disk any non-target could contribute, so the query
    radius is guaranteed to capture every non-target that could constrain this
    target.
    """
    # Enclosure: the cage center must lie inside the enclosure region of every
    # cell in the target. A missing region means the cell is larger than the cage
    # can hold, so the target is infeasible outright.
    enclosure: list[Convex] = []
    for cell in target.cells:
        shape = enclosure_shape(spec, cell)
        if shape is None:
            return None
        enclosure.append(shape)

    # Exclusion: subtract a region around each nearby non-target the cage must not
    # trap. Only non-targets whose exclusion region can reach an enclosure matter,
    # so we query the spatial index within a bounding radius rather than scanning
    # the whole plate.
    exclusion: list[Convex] = []
    if nt_tree is not None and len(non_targets) > 0:
        seen: set[int] = set()
        for cell in target.cells:
            # Widest reach of any exclusion region from this enclosure.
            query_r = enclosure_reach(spec, cell.radius) + exclusion_reach(
                spec, max_nt_radius, min_nt_confidence
            )
            for idx in nt_tree.query_ball_point((cell.x, cell.y), query_r):
                if idx in seen:
                    continue
                nt = non_targets[idx]
                if nt.id in target.ids:
                    continue
                seen.add(idx)
                exclusion.append(exclusion_shape(spec, nt))

    region = FeasibleRegion(enclosure=tuple(enclosure), exclusion=tuple(exclusion))
    result = chebyshev_center(region, precision=precision)
    if not result.feasible:
        return None
    return Cage(
        x=result.x,
        y=result.y,
        spec=spec,
        target_ids=target.ids,
        clearance=result.clearance,
    )


# --------------------------------------------------------------------------- #
# Part 2: keep the largest non-overlapping set of cages
# --------------------------------------------------------------------------- #
def _overlap_edges(cages: Sequence[Cage], spec: CageSpec) -> list[tuple[int, int]]:
    """Edges of the overlap graph: pairs of cages whose walls collide.

    Two cages centered farther apart than ``2 * radius`` can never overlap, so a
    KD-tree first narrows the field to candidate pairs; each candidate is then
    confirmed with the exact shape overlap test (a distance check for circles,
    the separating-axis theorem for hexagons).
    """
    if len(cages) < 2:
        return []
    xy = np.array([(c.x, c.y) for c in cages], dtype=float)
    tree = cKDTree(xy)
    edges: list[tuple[int, int]] = []
    for a, b in tree.query_pairs(spec.collision_distance()):
        if cages_overlap(spec, cages[a], cages[b]):
            edges.append((a, b))
    return edges


def _select_non_overlapping(cages: Sequence[Cage], spec: CageSpec) -> list[Cage]:
    """Keep the maximum set of non-overlapping cages (Part 2, lines 11-16).

    Split the overlap graph into connected components, then solve each: exactly
    on small clusters, near-optimally on any large one. Isolated cages (no
    overlaps) are kept for free.
    """
    edges = _overlap_edges(cages, spec)
    if not edges:
        return list(cages)

    weights = [c.clearance for c in cages]
    chosen_indices: list[int] = []
    for component in connected_components(len(cages), edges):
        if len(component) == 1:
            chosen_indices.extend(component)
            continue
        comp_set = set(component)
        comp_edges = [(a, b) for (a, b) in edges if a in comp_set and b in comp_set]
        if len(component) <= SMALL_CLUSTER:
            chosen_indices.extend(exact_mis(component, comp_edges, weights))
        else:
            chosen_indices.extend(greedy_then_local_search(component, comp_edges, weights))

    return [cages[i] for i in sorted(chosen_indices)]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _candidate_cages(cells: Sequence[Cell], spec: CageSpec, precision: float) -> list[Cage]:
    """Part 1 for every target: the best legal cage for each, on its own."""
    targets = build_targets(cells)
    non_targets = [c for c in cells if c.label != "target"]

    # Sizing constants for the per-target neighbor query.
    max_nt_radius = max((c.radius for c in non_targets), default=0.0)
    min_nt_confidence = min((c.confidence for c in non_targets), default=1.0)

    nt_tree: cKDTree | None = None
    if non_targets:
        nt_xy = np.array([(c.x, c.y) for c in non_targets], dtype=float)
        nt_tree = cKDTree(nt_xy)

    candidates: list[Cage] = []
    for target in targets:
        cage = _best_cage_for_target(
            target, non_targets, nt_tree, spec, precision, max_nt_radius, min_nt_confidence
        )
        if cage is not None:
            candidates.append(cage)
    return candidates


def place_cages(
    cells: Sequence[Cell],
    spec: CageSpec | None = None,
    precision: float = 0.05,
) -> list[Cage]:
    """Place the largest valid set of cages over the target cells.

    Parameters
    ----------
    cells:
        Every detected cell, each labeled target or non-target.
    spec:
        The cage geometry and safety margins. Defaults to a circular cage.
    precision:
        Chebyshev-center tolerance in pixels; smaller is more accurate and
        slightly slower.

    Returns the chosen cages, each holding one target (a single cell or a
    co-caged group) and reporting the clearance achieved.
    """
    spec = spec or CageSpec()
    candidates = _candidate_cages(cells, spec, precision)  # Part 1
    return _select_non_overlapping(candidates, spec)  # Part 2


def place_cages_greedy_baseline(
    cells: Sequence[Cell],
    spec: CageSpec | None = None,
    precision: float = 0.05,
) -> list[Cage]:
    """Naive baseline: build the same per-target candidates, then keep them in
    arbitrary order, dropping any that collide with one already kept.

    This is the "cage every target on its own" strategy with a first-come overlap
    check and no global optimization. The benchmark compares its coverage against
    :func:`place_cages` to show the uplift from solving the independent set.
    """
    spec = spec or CageSpec()
    candidates = _candidate_cages(cells, spec, precision)
    kept: list[Cage] = []
    for cage in candidates:
        if not any(cages_overlap(spec, cage, k) for k in kept):
            kept.append(cage)
    return kept
