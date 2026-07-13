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

from .geometry import Disk, FeasibleRegion, chebyshev_center
from .mis import SMALL_CLUSTER, connected_components, exact_mis, greedy_then_local_search
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
    # Enclosure: the cage center must lie within R_enc of every cell in the
    # target. A non-positive R_enc means the cell is larger than the cage can
    # hold, so the target is infeasible outright.
    enclosure: list[Disk] = []
    for cell in target.cells:
        r_enc = spec.enclosure_radius(cell.radius)
        if r_enc <= 0.0:
            return None
        enclosure.append(Disk(cell.x, cell.y, r_enc))

    # Exclusion: subtract a disk around each nearby non-target the cage must not
    # trap. Only non-targets whose exclusion disk can reach the enclosure matter,
    # so we query the spatial index within a bounding radius rather than scanning
    # the whole plate.
    exclusion: list[Disk] = []
    if nt_tree is not None and len(non_targets) > 0:
        seen: set[int] = set()
        for cell, enc in zip(target.cells, enclosure):
            # Widest possible exclusion disk reachable from this enclosure disk.
            query_r = enc.r + spec.exclusion_radius(max_nt_radius, min_nt_confidence)
            for idx in nt_tree.query_ball_point((cell.x, cell.y), query_r):
                if idx in seen:
                    continue
                nt = non_targets[idx]
                if nt.id in target.ids:
                    continue
                seen.add(idx)
                s = spec.exclusion_radius(nt.radius, nt.confidence)
                exclusion.append(Disk(nt.x, nt.y, s))

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

    Two circular cages overlap when their centers are closer than ``2 * radius``.
    A KD-tree finds the colliding pairs without comparing every pair.
    """
    if len(cages) < 2:
        return []
    xy = np.array([(c.x, c.y) for c in cages], dtype=float)
    tree = cKDTree(xy)
    dist = spec.collision_distance()
    edges: list[tuple[int, int]] = []
    for a, b in tree.query_pairs(dist):
        # query_pairs includes exactly-touching pairs (== dist); those are valid.
        if np.hypot(xy[a, 0] - xy[b, 0], xy[a, 1] - xy[b, 1]) < dist - 1e-9:
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
    dist = spec.collision_distance()
    kept: list[Cage] = []
    for cage in candidates:
        if all(np.hypot(cage.x - k.x, cage.y - k.y) >= dist - 1e-9 for k in kept):
            kept.append(cage)
    return kept
