"""End-to-end tests for the cage-placement engine."""

import math
import random

from cage import (
    CageSpec,
    Cell,
    place_cages,
    place_cages_greedy_baseline,
    validate_placement,
)

SPEC = CageSpec(shape="circle", radius=20.0, wall=2.0, clearance=1.0, exclusion_margin=1.0)


def _target(i, x, y, r=5.0, group=None):
    return Cell(id=i, x=x, y=y, radius=r, label="target", group=group)


def _other(i, x, y, r=5.0, confidence=1.0):
    return Cell(id=i, x=x, y=y, radius=r, label="non-target", confidence=confidence)


def test_two_far_targets_both_caged():
    cells = [_target(0, 0, 0), _target(1, 200, 0)]
    cages = place_cages(cells, SPEC)
    assert len(cages) == 2
    assert validate_placement(cells, cages, SPEC).valid


def test_two_close_targets_only_one_caged():
    """Centers 30 apart with a 40-unit collision distance: the two cages would
    overlap, so only one survives Part 2."""
    cells = [_target(0, 0, 0), _target(1, 30, 0)]
    cages = place_cages(cells, SPEC)
    assert len(cages) == 1
    assert validate_placement(cells, cages, SPEC).valid


def test_cage_avoids_trapping_neighbor():
    """A non-target next to a target must not be trapped; the cage shifts away
    and stays valid."""
    cells = [_target(0, 0, 0), _other(1, 18, 0)]
    cages = place_cages(cells, SPEC)
    assert len(cages) == 1
    report = validate_placement(cells, cages, SPEC)
    assert report.valid, report.violations


def test_impossible_target_is_skipped():
    """A target ringed by nearby non-targets has no legal region and is left
    uncaged, without trapping anyone."""
    ring = [
        _other(i + 1, 10 * math.cos(t), 10 * math.sin(t))
        for i, t in enumerate([k * math.pi / 4 for k in range(8)])
    ]
    cells = [_target(0, 0, 0), *ring]
    cages = place_cages(cells, SPEC)
    assert len(cages) == 0


def test_cocaging_group_single_cage_holds_both():
    """Two target cells sharing a group are held by one cage."""
    cells = [_target(0, 0, 0, r=4, group=1), _target(1, 12, 0, r=4, group=1)]
    cages = place_cages(cells, SPEC)
    assert len(cages) == 1
    assert set(cages[0].target_ids) == {0, 1}
    assert validate_placement(cells, cages, SPEC).valid


def test_low_confidence_neighbor_widens_exclusion():
    """A doubtful non-target should push the cage farther away than a certain
    one at the same spot."""
    spec = CageSpec(radius=20, wall=2, clearance=1, exclusion_margin=1, uncertainty_scale=0.1)
    sure = [_target(0, 0, 0), _other(1, 18, 0, confidence=1.0)]
    unsure = [_target(0, 0, 0), _other(1, 18, 0, confidence=0.0)]
    cage_sure = place_cages(sure, spec)[0]
    cage_unsure = place_cages(unsure, spec)[0]
    d_sure = math.hypot(cage_sure.x - 18, cage_sure.y)
    d_unsure = math.hypot(cage_unsure.x - 18, cage_unsure.y)
    assert d_unsure > d_sure


def test_beats_or_matches_greedy_baseline_on_random_plates():
    """Optimizing the independent set should never cover fewer targets than the
    naive greedy baseline, and every placement stays valid."""
    rng = random.Random(1)
    for trial in range(20):
        cells = []
        cid = 0
        for _ in range(40):
            x, y = rng.uniform(0, 400), rng.uniform(0, 400)
            label = "target" if rng.random() < 0.6 else "non-target"
            cells.append(Cell(id=cid, x=x, y=y, radius=rng.uniform(4, 7), label=label))
            cid += 1
        optimized = place_cages(cells, SPEC)
        baseline = place_cages_greedy_baseline(cells, SPEC)
        assert validate_placement(cells, optimized, SPEC).valid
        assert validate_placement(cells, baseline, SPEC).valid
        assert len(optimized) >= len(baseline), f"trial {trial}"


def test_validator_flags_overlapping_cages():
    """Sanity check the independent validator: two hand-placed overlapping cages
    must register a collision."""
    from cage.types import Cage

    cells = [_target(0, 0, 0), _target(1, 5, 0)]
    bad = [
        Cage(0, 0, SPEC, (0,), clearance=5.0),
        Cage(5, 0, SPEC, (1,), clearance=5.0),
    ]
    report = validate_placement(cells, bad, SPEC)
    assert not report.valid
    assert any(v.rule == "collision" for v in report.violations)
