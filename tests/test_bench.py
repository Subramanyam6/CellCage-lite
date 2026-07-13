"""Smoke tests for the benchmark harness (small, fast configuration)."""

from bench.run import benchmark_cage, benchmark_tracking


def test_cage_benchmark_runs_clean():
    rows = benchmark_cage(densities=[60], shapes=["circle"], repeats=2, n_plates=2, seed=0)
    assert len(rows) == 1
    row = rows[0]
    assert row.violation_rate == 0.0  # engine never produces an invalid placement
    assert row.coverage_pct > 0.0
    assert row.uplift_pct >= 0.0  # optimizing never covers fewer than greedy
    assert row.latency_ms > 0.0


def test_tracking_benchmark_runs():
    rows = benchmark_tracking(seed=0)
    assert len(rows) == 3
    assert all(r.id_accuracy >= 0.0 for r in rows)
