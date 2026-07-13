"""Synthetic plate and sequence generators.

Cage placement has no public ground truth, so we generate it. A plate is a set
of non-overlapping cells with known positions, radii, and target/non-target
labels, which makes the correct occupancy and the achievable coverage known
exactly. A sequence is a plate whose cells drift over time along known
trajectories, which gives ground-truth identities for scoring the tracker.

Everything is seeded, so a given seed always reproduces the same plate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cage.types import Cell


@dataclass(frozen=True)
class Plate:
    """A generated field of cells."""

    width: float
    height: float
    cells: list[Cell]

    @property
    def targets(self) -> list[Cell]:
        return [c for c in self.cells if c.label == "target"]


@dataclass(frozen=True)
class Sequence:
    """A time-lapse of one plate.

    ``frames[t]`` is an ``(m, 2)`` array of detected cell centers in frame ``t``;
    ``truth[t]`` is the matching length-``m`` array of ground-truth identities.
    Detection order within a frame is shuffled, so a tracker cannot cheat by
    relying on row order.
    """

    frames: list[np.ndarray]
    truth: list[np.ndarray]


def generate_plate(
    n_cells: int = 200,
    size: tuple[float, float] = (512.0, 512.0),
    target_fraction: float = 0.5,
    radius: tuple[float, float] = (5.0, 9.0),
    min_gap: float = 1.0,
    clustered: bool = False,
    n_clusters: int = 6,
    confidence: tuple[float, float] = (0.7, 1.0),
    seed: int = 0,
) -> Plate:
    """Generate a plate of non-overlapping cells.

    Parameters
    ----------
    n_cells:
        Target number of cells (fewer may be placed if the field is too dense to
        fit them all without overlap).
    size:
        Plate width and height in pixels.
    target_fraction:
        Fraction of cells labeled ``"target"``.
    radius:
        Uniform range for cell radii.
    min_gap:
        Minimum empty gap between two cell outlines, so cells never overlap.
    clustered:
        If true, draw cells around a few cluster centers to stress local
        crowding; otherwise spread them uniformly.
    confidence:
        Uniform range for the classifier confidence attached to each cell.
    seed:
        Random seed.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    margin = radius[1]
    centers = _cluster_centers(rng, n_clusters, width, height, margin) if clustered else None

    cells: list[Cell] = []
    positions: list[tuple[float, float]] = []
    radii: list[float] = []
    max_attempts = n_cells * 200

    for _ in range(max_attempts):
        if len(cells) >= n_cells:
            break
        r = float(rng.uniform(*radius))
        x, y = _sample_point(rng, width, height, margin, centers)
        # Reject if it would overlap an existing cell (distance must exceed the
        # sum of radii plus the gap).
        if any(
            np.hypot(x - px, y - py) < r + rj + min_gap
            for (px, py), rj in zip(positions, radii)
        ):
            continue
        label = "target" if rng.random() < target_fraction else "non-target"
        cells.append(
            Cell(
                id=len(cells),
                x=x,
                y=y,
                radius=r,
                label=label,
                confidence=float(rng.uniform(*confidence)),
            )
        )
        positions.append((x, y))
        radii.append(r)

    return Plate(width=width, height=height, cells=cells)


def _cluster_centers(rng, k, width, height, margin) -> np.ndarray:
    xs = rng.uniform(margin, width - margin, size=k)
    ys = rng.uniform(margin, height - margin, size=k)
    return np.column_stack([xs, ys])


def _sample_point(rng, width, height, margin, centers) -> tuple[float, float]:
    if centers is None:
        return (
            float(rng.uniform(margin, width - margin)),
            float(rng.uniform(margin, height - margin)),
        )
    cx, cy = centers[rng.integers(len(centers))]
    spread = min(width, height) / 12.0
    x = float(np.clip(rng.normal(cx, spread), margin, width - margin))
    y = float(np.clip(rng.normal(cy, spread), margin, height - margin))
    return (x, y)


def generate_sequence(
    n_cells: int = 30,
    n_frames: int = 20,
    size: tuple[float, float] = (512.0, 512.0),
    speed: float = 3.0,
    jitter: float = 0.6,
    miss_rate: float = 0.0,
    seed: int = 0,
) -> Sequence:
    """Generate a drift sequence with known identities.

    Each cell starts at a random position with a random constant velocity of
    magnitude ``speed``, then moves with a small random ``jitter`` each frame
    (near-linear motion, the tracker's design assumption). With ``miss_rate`` a
    cell can be missing from a frame, simulating a dropped detection.
    """
    rng = np.random.default_rng(seed)
    width, height = size

    pos = np.column_stack(
        [rng.uniform(0, width, n_cells), rng.uniform(0, height, n_cells)]
    )
    angle = rng.uniform(0, 2 * np.pi, n_cells)
    vel = np.column_stack([np.cos(angle), np.sin(angle)]) * speed

    frames: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for _ in range(n_frames):
        pos = pos + vel + rng.normal(0.0, jitter, pos.shape)
        # Reflect off the plate edges so cells stay in view.
        for dim, hi in enumerate((width, height)):
            below = pos[:, dim] < 0
            above = pos[:, dim] > hi
            pos[below, dim] = -pos[below, dim]
            pos[above, dim] = 2 * hi - pos[above, dim]
            vel[below | above, dim] *= -1

        keep = rng.random(n_cells) >= miss_rate
        ids = np.nonzero(keep)[0]
        points = pos[ids].copy()

        order = rng.permutation(len(ids))
        frames.append(points[order])
        truth.append(ids[order])

    return Sequence(frames=frames, truth=truth)
