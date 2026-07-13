"""Tests for the Kalman filter and the tracker."""

import numpy as np

from data.synthetic import generate_sequence
from track import KalmanFilter2D, Tracker, count_id_switches, id_accuracy, run_tracker


def test_kalman_learns_constant_velocity():
    """Fed a cell moving at a steady +5 in x, the filter's prediction should lock
    onto the motion after a few frames."""
    kf = KalmanFilter2D((0.0, 0.0), process_noise=0.01, measurement_noise=1.0)
    true_x = 0.0
    pred = None
    for step in range(10):
        pred = kf.predict()
        true_x += 5.0
        kf.update(np.array([true_x, 0.0]))
    # After settling, the next prediction should be within a pixel of the truth.
    nxt = kf.predict()
    assert abs(nxt[0] - (true_x + 5.0)) < 1.5
    assert abs(nxt[1]) < 1.0


def test_clean_sequence_holds_all_identities():
    """Well-separated, slow cells with no missed detections should be tracked
    with no identity switches."""
    seq = generate_sequence(n_cells=12, n_frames=15, speed=2.0, jitter=0.3, seed=4)
    labels = run_tracker(seq.frames, gating_distance=30.0)
    assert count_id_switches(labels, seq.truth) == 0
    assert id_accuracy(labels, seq.truth) == 1.0


def test_survives_missed_detections():
    """A moderate miss rate should still be tracked with high identity
    consistency, because prediction carries a track through a dropped frame."""
    seq = generate_sequence(n_cells=15, n_frames=20, speed=2.5, jitter=0.4, miss_rate=0.15, seed=8)
    labels = run_tracker(seq.frames, gating_distance=30.0, max_age=5)
    assert id_accuracy(labels, seq.truth) > 0.9


def test_new_and_leaving_cells():
    """Cells present only in later frames should get fresh ids; the tracker
    should not crash on frames with no detections."""
    tracker = Tracker(gating_distance=30.0)
    labels0 = tracker.step(np.array([[10.0, 10.0], [100.0, 100.0]]))
    assert len(set(labels0.tolist())) == 2
    empty = tracker.step(np.empty((0, 2)))
    assert len(empty) == 0
    labels2 = tracker.step(np.array([[12.0, 11.0]]))  # first cell continues
    assert labels2[0] == labels0[0]
