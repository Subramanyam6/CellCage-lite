"""End-to-end integration tests: image in, cages out, and tracking across frames.

These run the whole system on the torch-free path (mask-based detection, a
statistics embedder, the linear-probe head), so the stages are proven to compose
without any model downloads.
"""

import numpy as np

from cage import CageSpec, Cell, validate_placement
from classify import CellClassifier, LinearProbe, StatsEmbedder
from data.synthetic import generate_plate, generate_sequence, rasterize_plate
from detect import MaskDetector
from pipeline import Pipeline
from track import count_id_switches

SPEC = CageSpec(radius=18.0, wall=2.0, clearance=1.0, exclusion_margin=1.0)


def test_end_to_end_image_to_cages():
    # Train a torch-free classifier on a labeled, rasterized plate.
    train = generate_plate(n_cells=120, size=(320, 320), target_fraction=0.5, seed=0)
    train_img, _ = rasterize_plate(train, seed=0)
    clf = CellClassifier(embedder=StatsEmbedder(), head=LinearProbe())
    clf.fit(train_img, train.cells, [c.label for c in train.cells])

    # Run the full pipeline on a fresh plate.
    test = generate_plate(n_cells=120, size=(320, 320), target_fraction=0.5, seed=1)
    test_img, test_mask = rasterize_plate(test, seed=1)
    pipe = Pipeline(cage_spec=SPEC, detector=MaskDetector(test_mask), classifier=clf)
    result = pipe.run_image(test_img)

    # Detection recovered every cell.
    assert len(result.cells) == len(test.cells)
    # Classification is accurate (targets are painted brighter).
    correct = sum(rc.label == tc.label for rc, tc in zip(result.cells, test.cells))
    assert correct / len(test.cells) > 0.9
    # Placement is valid and covers targets.
    assert validate_placement(result.cells, result.cages, SPEC).valid
    assert len(result.cages) > 0
    # Every stage was timed.
    assert {"detect", "classify", "cage"} <= set(result.latency_ms)
    assert result.total_latency_ms > 0.0


def test_run_cells_without_detector():
    plate = generate_plate(n_cells=60, seed=5)
    pipe = Pipeline(cage_spec=SPEC)
    result = pipe.run_cells(plate.cells)
    assert validate_placement(plate.cells, result.cages, SPEC).valid
    assert "cage" in result.latency_ms
    assert 0.0 <= result.coverage <= 1.0


def test_end_to_end_tracking():
    seq = generate_sequence(n_cells=15, n_frames=20, speed=2.5, jitter=0.3, seed=3)
    frames_cells = [
        [Cell(id=i, x=float(p[0]), y=float(p[1]), radius=5.0) for i, p in enumerate(frame)]
        for frame in seq.frames
    ]
    pipe = Pipeline(tracker_kwargs={"gating_distance": 30.0})
    labels = pipe.track_sequence(frames_cells)
    assert count_id_switches(labels, seq.truth) == 0
