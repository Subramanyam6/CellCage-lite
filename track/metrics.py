"""Tracking-quality metrics scored against ground-truth identities."""

from __future__ import annotations

import numpy as np


def count_id_switches(
    pred_labels: list[np.ndarray],
    true_labels: list[np.ndarray],
) -> int:
    """Count identity switches across a sequence.

    For each ground-truth cell we remember the track id it was last assigned. A
    switch is counted whenever that cell reappears under a different track id.
    Fewer switches means the tracker held identities more stably.

    ``pred_labels[t][i]`` and ``true_labels[t][i]`` describe the same detection:
    the assigned track id and the ground-truth cell id, respectively.
    """
    last_track: dict[int, int] = {}
    switches = 0
    for pred, true in zip(pred_labels, true_labels):
        for track_id, gt_id in zip(pred.tolist(), true.tolist()):
            if gt_id in last_track and last_track[gt_id] != track_id:
                switches += 1
            last_track[gt_id] = track_id
    return switches


def id_accuracy(
    pred_labels: list[np.ndarray],
    true_labels: list[np.ndarray],
) -> float:
    """Fraction of detections whose track id matches the most common track id
    assigned to their ground-truth cell (a simple identity-consistency score in
    ``[0, 1]``)."""
    # Majority track id per ground-truth cell.
    from collections import Counter, defaultdict

    votes: dict[int, Counter] = defaultdict(Counter)
    for pred, true in zip(pred_labels, true_labels):
        for track_id, gt_id in zip(pred.tolist(), true.tolist()):
            votes[gt_id][track_id] += 1
    majority = {gt: c.most_common(1)[0][0] for gt, c in votes.items()}

    correct = total = 0
    for pred, true in zip(pred_labels, true_labels):
        for track_id, gt_id in zip(pred.tolist(), true.tolist()):
            total += 1
            if majority[gt_id] == track_id:
                correct += 1
    return correct / total if total else 1.0
