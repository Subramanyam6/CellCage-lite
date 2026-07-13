"""Maximum independent set for Part 2 of the algorithm.

Part 1 produces one candidate cage per target, ignoring the other cages. Some of
those cages overlap and cannot all be kept. Part 2 keeps the largest set of
cages with no two overlapping. Modeling each cage as a vertex and each overlap as
an edge, that is a *maximum independent set* (MIS): the largest set of vertices
with no edge between any two. MIS is NP-hard in general, but the plate's geometry
keeps the overlap graph sparse and broken into small, independent clusters, so we
can solve it exactly on the small clusters and near-optimally on the rare large
one.

Selecting the MIS of each connected component independently is optimal: no edge
crosses between components, so choices in one never constrain another.
"""

from __future__ import annotations

from collections.abc import Sequence

# A cluster small enough to solve exactly with branch and bound. Above this,
# fall back to greedy + local search. Exact search is exponential in the worst
# case, so the threshold caps its cost; overlap clusters are almost always well
# under it.
SMALL_CLUSTER = 12


class UnionFind:
    """Disjoint-set forest with path compression and union by rank.

    Used to group overlapping cages into connected components in near-linear
    time.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, a: int) -> int:
        root = a
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression: point every node on the way to the root.
        while self._parent[a] != root:
            self._parent[a], a = root, self._parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def connected_components(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Group ``0..n-1`` into connected components given an edge list."""
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for v in range(n):
        groups.setdefault(uf.find(v), []).append(v)
    return list(groups.values())


def _adjacency_bitsets(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> list[int]:
    """Build a bitset adjacency for a subgraph on ``nodes``.

    Node ``nodes[i]`` maps to local index ``i``; bit ``j`` of ``adj[i]`` is set
    when ``i`` and ``j`` are adjacent. Bitsets make the exact search fast: the
    neighbors of a vertex, and set differences, are single integer operations.
    """
    index = {node: i for i, node in enumerate(nodes)}
    adj = [0] * len(nodes)
    node_set = set(nodes)
    for a, b in edges:
        if a in node_set and b in node_set:
            ia, ib = index[a], index[b]
            if ia != ib:
                adj[ia] |= 1 << ib
                adj[ib] |= 1 << ia
    return adj


def exact_mis(
    nodes: Sequence[int],
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float] | None = None,
) -> list[int]:
    """Maximum independent set of a small cluster by branch and bound.

    Maximizes the number of cages kept; ties are broken by total ``weights``
    (we pass each cage's clearance, so among equal-size solutions the most
    robust one wins). Complexity is exponential in the worst case, which is why
    the caller only routes small clusters here.

    Branch and bound: repeatedly pick the lowest-degree remaining vertex, and
    branch on taking it (drop it and its neighbors) versus dropping it. A simple
    upper bound (current size plus the number of vertices still available) prunes
    branches that cannot beat the best solution found so far.
    """
    local = list(nodes)
    n = len(local)
    if n == 0:
        return []
    adj = _adjacency_bitsets(local, edges)
    w = [1.0] * n if weights is None else [weights[i] for i in range(n)]

    full_mask = (1 << n) - 1
    best_count = -1
    best_weight = -1.0
    best_set = 0

    def popcount(x: int) -> int:
        return x.bit_count()

    def branch(available: int, chosen: int, count: int, weight: float) -> None:
        nonlocal best_count, best_weight, best_set
        if available == 0:
            if count > best_count or (count == best_count and weight > best_weight):
                best_count, best_weight, best_set = count, weight, chosen
            return
        # Upper bound: everything still available might join the current set.
        if count + popcount(available) < best_count:
            return

        # Pick the available vertex of minimum degree within the available set:
        # it branches into the fewest cases and tightens the bound fastest.
        pivot = -1
        pivot_deg = n + 1
        avail = available
        while avail:
            v = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            deg = popcount(adj[v] & available)
            if deg < pivot_deg:
                pivot_deg, pivot = deg, v

        v_bit = 1 << pivot
        # Branch 1: take the pivot; drop it and all its neighbors.
        branch(
            available & ~(v_bit | adj[pivot]),
            chosen | v_bit,
            count + 1,
            weight + w[pivot],
        )
        # Branch 2: drop the pivot; keep the rest available.
        branch(available & ~v_bit, chosen, count, weight)

    branch(full_mask, 0, 0, 0.0)
    return [local[i] for i in range(n) if best_set & (1 << i)]


def greedy_then_local_search(
    nodes: Sequence[int],
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float] | None = None,
) -> list[int]:
    """Near-optimal independent set for a large cluster.

    Greedy: repeatedly add the remaining vertex that overlaps the fewest others
    (minimum degree in the residual graph) and remove it and its neighbors.
    Minimum-degree greedy is a strong, cheap heuristic for independent set.

    Local search: a (1, 2)-swap improvement. For each kept vertex, if removing it
    frees two mutually non-adjacent vertices that are otherwise blocked only by
    it, swap the one out for the two in, raising the count. Repeat until no swap
    helps.
    """
    local = list(nodes)
    n = len(local)
    if n == 0:
        return []
    adj = _adjacency_bitsets(local, edges)
    w = [1.0] * n if weights is None else [weights[i] for i in range(n)]
    full_mask = (1 << n) - 1

    # --- Greedy construction ---
    chosen = 0
    available = full_mask
    while available:
        # Minimum residual degree, breaking ties toward higher weight.
        best_v, best_deg, best_w = -1, n + 1, -1.0
        avail = available
        while avail:
            v = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            deg = (adj[v] & available).bit_count()
            if deg < best_deg or (deg == best_deg and w[v] > best_w):
                best_v, best_deg, best_w = v, deg, w[v]
        v_bit = 1 << best_v
        chosen |= v_bit
        available &= ~(v_bit | adj[best_v])

    # --- (1, 2)-swap local search ---
    improved = True
    while improved:
        improved = False
        chosen_list = [i for i in range(n) if chosen & (1 << i)]
        for v in chosen_list:
            rest = chosen & ~(1 << v)
            # Vertices blocked only by v: not chosen, not adjacent to any kept
            # vertex other than v.
            blocked = 0
            for u in range(n):
                u_bit = 1 << u
                if chosen & u_bit:
                    continue
                if adj[u] & rest:
                    continue
                blocked |= u_bit
            # Need two of them that are not adjacent to each other.
            candidates = [u for u in range(n) if blocked & (1 << u)]
            pair = _find_nonadjacent_pair(candidates, adj)
            if pair is not None:
                a, b = pair
                chosen = rest | (1 << a) | (1 << b)
                improved = True
                break

    return [local[i] for i in range(n) if chosen & (1 << i)]


def _find_nonadjacent_pair(candidates: Sequence[int], adj: Sequence[int]) -> tuple[int, int] | None:
    """Return two candidates with no edge between them, or ``None``."""
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if not (adj[a] & (1 << b)):
                return (a, b)
    return None
