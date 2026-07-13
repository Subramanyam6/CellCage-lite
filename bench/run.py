"""Benchmark harness for the cage engine and the tracker.

Reports the numbers that fill the README's Validation section: cage-placement
latency, throughput, and peak memory across plate densities and cage shapes; the
target coverage and constraint-violation rate; the coverage uplift over the naive
greedy baseline; and tracking identity switches.

    python -m bench.run                 # print the tables
    python -m bench.run --json out.json # also dump raw numbers
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Callable

from cage import CageSpec, place_cages, place_cages_greedy_baseline, validate_placement
from data.synthetic import generate_plate, generate_sequence
from track import count_id_switches, id_accuracy, run_tracker


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #
def _time_median_ms(fn: Callable[[], object], repeats: int) -> float:
    """Median wall-clock time of ``fn`` over ``repeats`` runs, in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _peak_memory_mb(fn: Callable[[], object]) -> float:
    """Peak Python heap allocated during a single run of ``fn``, in MB."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Cage-engine benchmark
# --------------------------------------------------------------------------- #
@dataclass
class CageRow:
    shape: str
    n_cells: int
    latency_ms: float
    throughput_fps: float
    peak_mb: float
    coverage_pct: float
    baseline_coverage_pct: float
    uplift_pct: float
    violation_rate: float


def benchmark_cage(
    densities: list[int],
    shapes: list[str],
    repeats: int = 5,
    n_plates: int = 5,
    seed: int = 0,
) -> list[CageRow]:
    """Benchmark the cage engine across plate densities and cage shapes."""
    rows: list[CageRow] = []
    for shape in shapes:
        spec = CageSpec(shape=shape, radius=18.0, wall=2.0, clearance=1.0, exclusion_margin=1.0)
        for n in densities:
            plates = [generate_plate(n_cells=n, seed=seed + i) for i in range(n_plates)]

            latencies, mems = [], []
            cover, base_cover, violations = [], [], 0
            for plate in plates:
                cells = plate.cells
                n_targets = max(1, len(plate.targets))

                latencies.append(_time_median_ms(lambda c=cells: place_cages(c, spec), repeats))
                mems.append(_peak_memory_mb(lambda c=cells: place_cages(c, spec)))

                chosen = place_cages(cells, spec)
                baseline = place_cages_greedy_baseline(cells, spec)
                cover.append(100.0 * len(chosen) / n_targets)
                base_cover.append(100.0 * len(baseline) / n_targets)
                if not validate_placement(cells, chosen, spec).valid:
                    violations += 1

            latency = statistics.median(latencies)
            coverage = statistics.mean(cover)
            baseline_coverage = statistics.mean(base_cover)
            rows.append(
                CageRow(
                    shape=shape,
                    n_cells=n,
                    latency_ms=latency,
                    throughput_fps=1000.0 / latency if latency > 0 else float("inf"),
                    peak_mb=statistics.median(mems),
                    coverage_pct=coverage,
                    baseline_coverage_pct=baseline_coverage,
                    uplift_pct=coverage - baseline_coverage,
                    violation_rate=violations / len(plates),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Tracking benchmark
# --------------------------------------------------------------------------- #
@dataclass
class TrackRow:
    n_cells: int
    miss_rate: float
    id_switches: int
    id_accuracy: float


def benchmark_tracking(seed: int = 0) -> list[TrackRow]:
    """Benchmark the tracker on clean and lossy drift sequences."""
    rows: list[TrackRow] = []
    for n_cells, miss in [(20, 0.0), (20, 0.15), (40, 0.15)]:
        seq = generate_sequence(n_cells=n_cells, n_frames=30, miss_rate=miss, seed=seed)
        labels = run_tracker(seq.frames, gating_distance=30.0, max_age=5)
        rows.append(
            TrackRow(
                n_cells=n_cells,
                miss_rate=miss,
                id_switches=count_id_switches(labels, seq.truth),
                id_accuracy=id_accuracy(labels, seq.truth),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_cage_table(rows: list[CageRow]) -> None:
    print("\n### Cage engine\n")
    print("| Shape | Cells/field | Latency (ms) | Throughput (fields/s) | Peak mem (MB) | "
          "Coverage | Baseline | Uplift | Violations |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r.shape} | {r.n_cells} | {r.latency_ms:.2f} | {r.throughput_fps:.0f} | "
            f"{r.peak_mb:.2f} | {r.coverage_pct:.1f}% | {r.baseline_coverage_pct:.1f}% | "
            f"+{r.uplift_pct:.1f} pts | {r.violation_rate:.0%} |"
        )


def _print_track_table(rows: list[TrackRow]) -> None:
    print("\n### Tracking\n")
    print("| Cells | Miss rate | ID switches | Identity accuracy |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r.n_cells} | {r.miss_rate:.0%} | {r.id_switches} | {r.id_accuracy:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CellCage-lite benchmarks.")
    parser.add_argument(
        "--densities",
        type=int,
        nargs="+",
        default=[100, 250, 500],
        help="cells per field to sweep",
    )
    parser.add_argument("--shapes", nargs="+", default=["circle"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=str, default=None, help="path to dump raw results")
    args = parser.parse_args()

    print("Running cage benchmark...", flush=True)
    cage_rows = benchmark_cage(args.densities, args.shapes, repeats=args.repeats)
    print("Running tracking benchmark...", flush=True)
    track_rows = benchmark_tracking()

    _print_cage_table(cage_rows)
    _print_track_table(track_rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "cage": [asdict(r) for r in cage_rows],
                    "tracking": [asdict(r) for r in track_rows],
                },
                f,
                indent=2,
            )
        print(f"\nWrote raw results to {args.json}")


if __name__ == "__main__":
    main()
