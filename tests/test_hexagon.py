"""Tests for hexagonal cages and the convex-polygon geometry."""

import math
import random

from cage import CageSpec, Cell, place_cages, place_cages_greedy_baseline, validate_placement
from cage.geometry import ConvexPolygon, convex_overlap, regular_hexagon

HEX = CageSpec(shape="hexagon", radius=20.0, wall=2.0, clearance=1.0, exclusion_margin=1.0)


def _target(i, x, y, r=5.0, group=None):
    return Cell(id=i, x=x, y=y, radius=r, label="target", group=group)


def _other(i, x, y, r=5.0):
    return Cell(id=i, x=x, y=y, radius=r, label="non-target")


# --- polygon primitives ---------------------------------------------------- #
def test_hexagon_signed_distance_center_is_inradius():
    hexagon = regular_hexagon(0.0, 0.0, 10.0)
    # Deepest interior point is the center; its distance to an edge is the
    # inradius, circumradius * cos(30 deg).
    assert math.isclose(hexagon.signed_distance(0.0, 0.0), 10.0 * math.sqrt(3) / 2, abs_tol=1e-6)
    # A point outside has negative signed distance.
    assert hexagon.signed_distance(20.0, 0.0) < 0.0


def test_convex_overlap_detects_separation():
    a = regular_hexagon(0.0, 0.0, 10.0)
    b = regular_hexagon(5.0, 0.0, 10.0)  # clearly overlapping
    far = regular_hexagon(100.0, 0.0, 10.0)
    assert convex_overlap(a, b)
    assert not convex_overlap(a, far)


def test_square_overlap_touching_is_not_overlap():
    unit = ConvexPolygon(((0, 0), (1, 0), (1, 1), (0, 1)))
    touching = ConvexPolygon(((1, 0), (2, 0), (2, 1), (1, 1)))
    assert not convex_overlap(unit, touching)


# --- placement with hexagonal cages ---------------------------------------- #
def test_hexagon_two_far_targets_both_caged():
    cells = [_target(0, 0, 0), _target(1, 200, 0)]
    cages = place_cages(cells, HEX)
    assert len(cages) == 2
    assert validate_placement(cells, cages, HEX).valid


def test_hexagon_avoids_trapping_neighbor():
    cells = [_target(0, 0, 0), _other(1, 18, 0)]
    cages = place_cages(cells, HEX)
    assert len(cages) == 1
    report = validate_placement(cells, cages, HEX)
    assert report.valid, report.violations


def test_hexagon_cocaging_group():
    cells = [_target(0, 0, 0, r=4, group=1), _target(1, 12, 0, r=4, group=1)]
    cages = place_cages(cells, HEX)
    assert len(cages) == 1
    assert set(cages[0].target_ids) == {0, 1}
    assert validate_placement(cells, cages, HEX).valid


def test_hexagon_random_plate_is_valid_and_beats_baseline():
    rng = random.Random(2)
    for _ in range(15):
        cells = []
        for cid in range(35):
            x, y = rng.uniform(0, 400), rng.uniform(0, 400)
            label = "target" if rng.random() < 0.6 else "non-target"
            cells.append(Cell(id=cid, x=x, y=y, radius=rng.uniform(4, 7), label=label))
        optimized = place_cages(cells, HEX)
        baseline = place_cages_greedy_baseline(cells, HEX)
        assert validate_placement(cells, optimized, HEX).valid
        assert len(optimized) >= len(baseline)
