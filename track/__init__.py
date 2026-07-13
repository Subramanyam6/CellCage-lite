"""Online cell tracking for CellCage-lite.

Kalman prediction paired with Hungarian assignment: the standard online
tracking-by-detection approach, which suits the near-linear drift of caged cells.
See README section 4.
"""

from __future__ import annotations

from .kalman import KalmanFilter2D
from .metrics import count_id_switches, id_accuracy
from .tracker import Track, Tracker, run_tracker

__all__ = [
    "KalmanFilter2D",
    "Track",
    "Tracker",
    "run_tracker",
    "count_id_switches",
    "id_accuracy",
]
