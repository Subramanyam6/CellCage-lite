"""Matplotlib rendering for the demo.

Draws a plate with its cells, the cages the engine placed, and, for a sequence,
the trajectories the tracker recovered. Uses the non-interactive Agg backend so
it renders the same whether or not a display is attached.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from cage.geometry import regular_hexagon
from cage.types import Cage, CageSpec, Cell

_TARGET_COLOR = "#2e7d32"
_NON_TARGET_COLOR = "#9e9e9e"
_CAGE_COLOR = "#1565c0"


def figure_placement(
    cells: Sequence[Cell],
    cages: Sequence[Cage],
    spec: CageSpec,
    size: tuple[float, float] = (512.0, 512.0),
    title: str | None = None,
) -> "plt.Figure":
    """Render cells (targets vs non-targets) and the placed cages."""
    fig, ax = plt.subplots(figsize=(7, 7))
    width, height = size

    for cell in cells:
        color = _TARGET_COLOR if cell.label == "target" else _NON_TARGET_COLOR
        ax.add_patch(Circle((cell.x, cell.y), cell.radius, color=color, alpha=0.55, lw=0))

    for cage in cages:
        _draw_cage(ax, cage, spec)

    caged = sum(len(c.target_ids) for c in cages)
    n_targets = sum(1 for c in cells if c.label == "target")
    ax.set_title(
        title
        or f"{spec.shape} cages: {len(cages)} placed, {caged}/{n_targets} targets covered"
    )
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coordinates: y grows downward
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def _draw_cage(ax, cage: Cage, spec: CageSpec) -> None:
    if spec.shape == "circle":
        ax.add_patch(Circle((cage.x, cage.y), spec.radius, fill=False, edgecolor=_CAGE_COLOR, lw=1.6))
    else:
        verts = regular_hexagon(cage.x, cage.y, spec.radius, spec.orientation).vertices
        ax.add_patch(Polygon(verts, closed=True, fill=False, edgecolor=_CAGE_COLOR, lw=1.6))


def figure_tracking(
    frames: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    size: tuple[float, float] = (512.0, 512.0),
    title: str = "Recovered trajectories",
) -> "plt.Figure":
    """Render each tracked cell's trajectory, colored by its recovered identity.

    A stable trajectory is one continuous colored path; an identity switch shows
    up as a color change along a path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    width, height = size

    # Gather each track id's positions in time order.
    paths: dict[int, list[tuple[float, float]]] = {}
    for frame, ids in zip(frames, labels):
        for (x, y), track_id in zip(frame, ids.tolist()):
            paths.setdefault(track_id, []).append((x, y))

    cmap = plt.get_cmap("tab20")
    for track_id, pts in paths.items():
        arr = np.array(pts)
        color = cmap(track_id % 20)
        ax.plot(arr[:, 0], arr[:, 1], "-", color=color, lw=1.2, alpha=0.8)
        ax.plot(arr[0, 0], arr[0, 1], "o", color=color, ms=4)

    ax.set_title(f"{title}: {len(paths)} tracks")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig
