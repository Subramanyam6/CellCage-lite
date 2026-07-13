"""Tests for the independent-set solvers and their supporting structures."""

import random

from cage.mis import (
    UnionFind,
    connected_components,
    exact_mis,
    greedy_then_local_search,
)


def _is_independent(chosen, edges):
    s = set(chosen)
    return all(not (a in s and b in s) for a, b in edges)


def test_union_find_basic():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)


def test_connected_components():
    comps = connected_components(6, [(0, 1), (1, 2), (4, 5)])
    sizes = sorted(len(c) for c in comps)
    assert sizes == [1, 2, 3]  # {3}, {4,5}, {0,1,2}


def test_exact_mis_on_path():
    """A path 0-1-2-3-4 has maximum independent set {0, 2, 4}, size 3."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    chosen = exact_mis(list(range(5)), edges)
    assert len(chosen) == 3
    assert _is_independent(chosen, edges)


def test_exact_mis_on_triangle():
    """A triangle admits only one vertex."""
    edges = [(0, 1), (1, 2), (0, 2)]
    chosen = exact_mis([0, 1, 2], edges)
    assert len(chosen) == 1


def test_exact_mis_prefers_higher_weight_on_ties():
    """Two disconnected vertices with different weights: both are kept, but a
    single-choice tie should take the heavier one."""
    edges = [(0, 1)]  # one edge, pick exactly one of {0, 1}
    chosen = exact_mis([0, 1], edges, weights=[1.0, 5.0])
    assert chosen == [1]


def test_greedy_matches_exact_on_random_small_graphs():
    """On small graphs greedy + local search should return an independent set;
    compare its size against the exact optimum."""
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(4, 9)
        edges = []
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < 0.35:
                    edges.append((a, b))
        nodes = list(range(n))
        exact = exact_mis(nodes, edges)
        greedy = greedy_then_local_search(nodes, edges)
        assert _is_independent(greedy, edges)
        # Greedy is a heuristic; it must never beat exact and should stay close.
        assert len(greedy) <= len(exact)
        assert len(greedy) >= len(exact) - 1


def test_empty_inputs():
    assert exact_mis([], []) == []
    assert greedy_then_local_search([], []) == []
