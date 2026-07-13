"""Online tracking-by-detection: Kalman prediction + Hungarian assignment.

Each frame, every existing track predicts where its cell should appear; the
predictions are matched to the new detections by the assignment that minimizes
total distance, with a gating radius so distant pairs are never matched. Matched
tracks are corrected with their detection, unmatched detections open new tracks,
and tracks that go unmatched too long are retired. This mirrors the pseudocode in
README section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanFilter2D


@dataclass
class Track:
    """One ongoing cell identity and its motion model."""

    id: int
    kf: KalmanFilter2D
    hits: int = 1  # number of detections matched so far
    time_since_update: int = 0  # frames since last matched
    predicted: np.ndarray = field(default_factory=lambda: np.zeros(2))


class Tracker:
    """Multi-object tracker over a stream of per-frame detections.

    Parameters
    ----------
    gating_distance:
        Maximum distance between a prediction and a detection for them to be
        eligible to match. Prevents identity swaps across the field.
    max_age:
        Frames a track may go unmatched before it is retired.
    process_noise, measurement_noise:
        Passed through to each track's Kalman filter.
    """

    def __init__(
        self,
        gating_distance: float = 30.0,
        max_age: int = 5,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ) -> None:
        self.gating_distance = gating_distance
        self.max_age = max_age
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks: list[Track] = []
        self._next_id = 0

    def step(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame of detections and return their assigned track ids.

        ``detections`` is an ``(m, 2)`` array of cell centers. The return value
        is a length-``m`` integer array giving the track id each detection was
        assigned to (a fresh id for a detection that started a new track).
        """
        detections = np.asarray(detections, dtype=float).reshape(-1, 2)
        m = len(detections)

        # Line 3-4: predict every track's position for this frame.
        for tr in self.tracks:
            tr.predicted = tr.kf.predict()

        labels = np.full(m, -1, dtype=np.int64)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        # Lines 5-7: gate, then solve the optimal one-to-one assignment.
        if self.tracks and m > 0:
            for ti, di in self._assign(detections):
                tr = self.tracks[ti]
                tr.kf.update(detections[di])
                tr.hits += 1
                tr.time_since_update = 0
                labels[di] = tr.id
                matched_tracks.add(ti)
                matched_dets.add(di)

        # Line 10: unmatched detections open new tracks.
        for di in range(m):
            if di not in matched_dets:
                tr = Track(
                    id=self._next_id,
                    kf=KalmanFilter2D(
                        tuple(detections[di]),
                        process_noise=self.process_noise,
                        measurement_noise=self.measurement_noise,
                    ),
                )
                tr.predicted = detections[di].copy()
                labels[di] = tr.id
                self.tracks.append(tr)
                self._next_id += 1

        # Line 11: age unmatched tracks and retire the stale ones.
        for ti, tr in enumerate(self.tracks):
            if ti not in matched_tracks:
                tr.time_since_update += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return labels

    def _assign(self, detections: np.ndarray) -> list[tuple[int, int]]:
        """Gated Hungarian assignment between tracks and detections.

        Builds a track-by-detection distance matrix, solves the minimum-cost
        one-to-one matching, and drops any matched pair whose distance exceeds
        the gating radius.
        """
        preds = np.array([tr.predicted for tr in self.tracks])  # (n, 2)
        # Pairwise Euclidean distances, shape (n_tracks, m_detections).
        cost = np.linalg.norm(preds[:, None, :] - detections[None, :, :], axis=2)

        # Disallow matches beyond the gate by making them prohibitively costly,
        # so the solver only uses them when nothing else is available (and we
        # discard those afterwards).
        big = self.gating_distance * 1e3
        gated = np.where(cost <= self.gating_distance, cost, big)

        rows, cols = linear_sum_assignment(gated)
        return [
            (int(r), int(c))
            for r, c in zip(rows, cols)
            if cost[r, c] <= self.gating_distance
        ]


def run_tracker(frames: list[np.ndarray], **kwargs) -> list[np.ndarray]:
    """Run a fresh tracker over a whole sequence.

    Returns one label array per frame (see :meth:`Tracker.step`).
    """
    tracker = Tracker(**kwargs)
    return [tracker.step(frame) for frame in frames]
