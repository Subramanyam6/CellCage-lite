"""Tests for the classifier heads, cropping, and the classifier wiring.

These cover the torch-free parts of the classification stage; the DINOv2
embedder itself is exercised only when torch is installed.
"""

import numpy as np

from cage.types import Cell
from classify import CellClassifier, LinearProbe, NearestClassMean, crop_cells


def _blobs(seed=0, n_per_class=40, dim=16, n_classes=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(n_classes, dim))
    X = np.vstack([rng.normal(centers[c], 1.0, (n_per_class, dim)) for c in range(n_classes)])
    y = np.concatenate([np.full(n_per_class, c) for c in range(n_classes)])
    return X, y


def test_linear_probe_separates_blobs():
    X, y = _blobs(seed=1)
    probe = LinearProbe().fit(X, y)
    assert np.mean(probe.predict(X) == y) > 0.95
    probs = probe.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_nearest_class_mean_separates_blobs():
    X, y = _blobs(seed=2)
    ncm = NearestClassMean().fit(X, y)
    assert np.mean(ncm.predict(X) == y) > 0.95


def test_crop_cells_shape():
    image = np.zeros((100, 100), dtype=np.float32)
    cells = [Cell(0, 30, 30, 8), Cell(1, 70, 70, 5)]
    crops = crop_cells(image, cells, out_size=32)
    assert crops.shape == (2, 32, 32)


def test_crop_handles_edge_cells():
    image = np.ones((50, 50), dtype=np.float32)
    cells = [Cell(0, 2, 2, 6)]  # window runs off the top-left corner
    crops = crop_cells(image, cells, out_size=16)
    assert crops.shape == (1, 16, 16)


def test_classifier_relabels_cells():
    X, _ = _blobs(seed=3, n_per_class=2, n_classes=1)  # placeholder embeddings
    # Two target-ish and two non-target embeddings, well separated.
    emb = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 9.9]])
    labels = ["target", "target", "non-target", "non-target"]
    cells = [Cell(i, i * 10, 0, 5) for i in range(4)]

    clf = CellClassifier(head=LinearProbe()).fit_embeddings(emb, labels)
    labeled = clf.label_embeddings(cells, emb)
    assert [c.label for c in labeled] == labels
    assert all(0.0 <= c.confidence <= 1.0 for c in labeled)
