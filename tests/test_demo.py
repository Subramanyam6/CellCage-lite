"""Smoke tests for the demo rendering (no display required)."""

import matplotlib

from app.demo import run_placement
from app.render import figure_tracking
from data.synthetic import generate_sequence
from track import run_tracker


def test_run_placement_returns_figure():
    fig, caption = run_placement(
        n_cells=80, target_fraction=0.5, shape="circle", radius=18.0, seed=0
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "cages" in caption
    matplotlib.pyplot.close(fig)


def test_run_placement_hexagon():
    fig, caption = run_placement(
        n_cells=80, target_fraction=0.5, shape="hexagon", radius=18.0, seed=0
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


def test_tracking_figure():
    seq = generate_sequence(n_cells=12, n_frames=10, seed=0)
    labels = run_tracker(seq.frames, gating_distance=30.0)
    fig = figure_tracking(seq.frames, labels)
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)
