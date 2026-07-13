"""A constant-velocity Kalman filter for a single cell.

The state is position and velocity, ``[x, y, vx, vy]``. Each frame the filter
predicts the next position from the current state, then corrects itself with the
measured position. Velocity is never measured; the filter infers it from how the
position shifts across frames, which is what lets it predict through a brief
missed detection or a close pass.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Constant-velocity Kalman filter over a 2-D position.

    Parameters
    ----------
    initial_position:
        The first measured position ``(x, y)``; velocity starts at zero.
    dt:
        Time step between frames.
    process_noise:
        How much the constant-velocity assumption is trusted. Larger values let
        the cell accelerate more freely between frames.
    measurement_noise:
        How noisy detections are. Larger values make the filter lean on its
        prediction over the measurement.
    """

    def __init__(
        self,
        initial_position: tuple[float, float],
        dt: float = 1.0,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ) -> None:
        # State transition: position advances by velocity * dt; velocity holds.
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        # Measurement model: we observe position only.
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise

        self.x = np.array([initial_position[0], initial_position[1], 0.0, 0.0], dtype=float)
        # Start uncertain about velocity, fairly sure about the seen position.
        self.P = np.diag([measurement_noise, measurement_noise, 1e3, 1e3]).astype(float)

    def predict(self) -> np.ndarray:
        """Advance the state one step and return the predicted position ``(x, y)``."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: np.ndarray) -> None:
        """Correct the state with a measured position ``(x, y)``."""
        z = np.asarray(measurement, dtype=float)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()
