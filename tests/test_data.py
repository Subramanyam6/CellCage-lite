"""Tests for the synthetic plate and sequence generators."""

import math

import numpy as np

from data.synthetic import generate_plate, generate_sequence


def test_cells_do_not_overlap():
    plate = generate_plate(n_cells=150, size=(400, 400), min_gap=1.0, seed=3)
    cells = plate.cells
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            d = math.hypot(a.x - b.x, a.y - b.y)
            assert d >= a.radius + b.radius + 1.0 - 1e-6


def test_generation_is_deterministic():
    a = generate_plate(n_cells=80, seed=7)
    b = generate_plate(n_cells=80, seed=7)
    assert [(c.x, c.y, c.radius, c.label) for c in a.cells] == [
        (c.x, c.y, c.radius, c.label) for c in b.cells
    ]


def test_target_fraction_is_respected():
    plate = generate_plate(n_cells=400, target_fraction=0.3, seed=1)
    frac = len(plate.targets) / len(plate.cells)
    assert 0.2 < frac < 0.4


def test_sequence_shapes_and_identities():
    seq = generate_sequence(n_cells=25, n_frames=15, seed=2)
    assert len(seq.frames) == 15
    assert len(seq.truth) == 15
    for frame, ids in zip(seq.frames, seq.truth):
        assert frame.shape[0] == ids.shape[0]
        assert frame.shape[1] == 2
        assert set(ids.tolist()) <= set(range(25))


def test_sequence_missing_detections():
    seq = generate_sequence(n_cells=40, n_frames=10, miss_rate=0.3, seed=5)
    counts = [len(f) for f in seq.frames]
    # With a 30% miss rate, frames should usually be short of the full 40.
    assert min(counts) < 40
