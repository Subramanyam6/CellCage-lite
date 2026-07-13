"""Tests for the torch-free detection utilities: flow round-trip and cell
extraction."""

import numpy as np

from detect import flows_to_labels, labels_to_cells, masks_to_flows


def _two_cell_mask(h=100, w=100):
    labels = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    labels[(xx - 30) ** 2 + (yy - 30) ** 2 <= 10**2] = 1
    labels[(xx - 70) ** 2 + (yy - 70) ** 2 <= 10**2] = 2
    return labels


def test_flow_roundtrip_recovers_two_instances():
    labels = _two_cell_mask()
    flow, foreground = masks_to_flows(labels)
    recovered = flows_to_labels(flow, foreground)
    n_instances = len(np.unique(recovered)) - 1  # minus background
    assert n_instances == 2


def test_labels_to_cells_centroids_and_radius():
    labels = _two_cell_mask()
    cells = labels_to_cells(labels)
    assert len(cells) == 2
    centers = sorted((round(c.x), round(c.y)) for c in cells)
    assert centers[0] == (30, 30)
    assert centers[1] == (70, 70)
    for c in cells:
        assert abs(c.radius - 10.0) < 1.5  # sqrt(area/pi) of a radius-10 disk
        assert c.label == "non-target"


def test_empty_mask_gives_no_cells():
    labels = np.zeros((40, 40), dtype=np.int32)
    assert labels_to_cells(labels) == []
    flow, fg = masks_to_flows(labels)
    assert flows_to_labels(flow, fg).max() == 0
