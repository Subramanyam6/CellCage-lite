"""Tests for the feasible-region geometry and the Chebyshev-center search."""

import math

from cage.geometry import Disk, FeasibleRegion, chebyshev_center


def test_single_disk_center():
    """With one enclosure disk and no exclusions, the deepest point is the disk
    center and the clearance is the disk radius."""
    region = FeasibleRegion(enclosure=(Disk(5.0, 5.0, 10.0),), exclusion=())
    result = chebyshev_center(region, precision=0.01)
    assert result.feasible
    assert math.isclose(result.x, 5.0, abs_tol=0.2)
    assert math.isclose(result.y, 5.0, abs_tol=0.2)
    assert math.isclose(result.clearance, 10.0, abs_tol=0.1)


def test_annulus_deepest_point():
    """Enclosure disk radius 10 with a concentric exclusion disk radius 4 leaves
    an annulus. The largest inscribed circle sits mid-ring at radius 7, with
    clearance 3."""
    region = FeasibleRegion(
        enclosure=(Disk(0.0, 0.0, 10.0),),
        exclusion=(Disk(0.0, 0.0, 4.0),),
    )
    result = chebyshev_center(region, precision=0.01)
    assert result.feasible
    assert math.isclose(result.clearance, 3.0, abs_tol=0.15)
    assert math.isclose(math.hypot(result.x, result.y), 7.0, abs_tol=0.5)


def test_empty_region_is_infeasible():
    """An exclusion disk that swallows the whole enclosure leaves no legal
    point."""
    region = FeasibleRegion(
        enclosure=(Disk(0.0, 0.0, 10.0),),
        exclusion=(Disk(0.0, 0.0, 20.0),),
    )
    result = chebyshev_center(region, precision=0.01)
    assert not result.feasible


def test_disjoint_enclosure_is_infeasible():
    """Two enclosure disks that do not overlap (a group too spread to co-cage)
    have no common interior."""
    region = FeasibleRegion(
        enclosure=(Disk(0.0, 0.0, 5.0), Disk(100.0, 0.0, 5.0)),
        exclusion=(),
    )
    result = chebyshev_center(region, precision=0.01)
    assert not result.feasible


def test_clearance_is_signed_distance():
    """clearance() equals the distance to the nearest boundary, negative
    outside."""
    region = FeasibleRegion(enclosure=(Disk(0.0, 0.0, 10.0),), exclusion=())
    assert math.isclose(region.clearance(0.0, 0.0), 10.0)
    assert math.isclose(region.clearance(8.0, 0.0), 2.0)
    assert math.isclose(region.clearance(12.0, 0.0), -2.0)
