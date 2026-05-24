import cv2
import numpy as np
from typing import Optional, Tuple


class PenKalmanFilter:
    """
    2D Kalman filter for smoothing pen tip position on the plane.

    State vector : [x, y, vx, vy]  (position and velocity in plane mm coords)
    Measurement  : [x, y]           (fused plane coordinate from SensorFuser)

    Behaviour
    ---------
    - When a measurement arrives above min_confidence: filter corrects state
    - When measurement is missing (occluded): filter predicts using velocity
    - Velocity state allows the filter to coast through short occlusions instead
      of freezing the last known position

    Parameters
    ----------
    process_noise     : trust the motion model less (higher = more responsive,
                        lower = smoother trajectory during fast movement)
    measurement_noise : trust measurements less (higher = smoother under noise,
                        lower = snaps to measurements aggressively)
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # 4 state variables [x, y, vx, vy], 2 measurement variables [x, y]
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition matrix (constant-velocity model):
        # x_new  = x  + vx*dt   (dt = 1 frame, absorbed into the matrix)
        # y_new  = y  + vy*dt
        # vx_new = vx
        # vy_new = vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: we only observe x and y directly, not velocity
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance: uncertainty in the motion model
        # Scale the velocity entries slightly higher — velocity is harder to predict
        self.kf.processNoiseCov = np.diag([
            process_noise,
            process_noise,
            process_noise * 2,
            process_noise * 2,
        ]).astype(np.float32)

        # Measurement noise covariance: uncertainty in HSV centroid measurements
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        # Keep the base value so per-frame confidence scaling rebuilds the
        # covariance from a constant instead of compounding the previous frame.
        self._base_meas_noise = float(measurement_noise)

        # Initial error covariance: high uncertainty before first measurement
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self._initialized = False

    def update(
        self,
        measurement: Optional[Tuple[float, float]],
        confidence: float,
        min_confidence: float = 0.2,
    ) -> Tuple[float, float]:
        """
        Feed one fused measurement into the filter and return the smoothed estimate.

        Parameters
        ----------
        measurement    : (x, y) in plane mm coords, or None if pen not detected
        confidence     : fused confidence [0, 1]
        min_confidence : confidence threshold below which measurement is ignored

        Returns
        -------
        (x, y) smoothed plane coordinate — always a valid point once initialized
        """
        if measurement is not None and not self._initialized:
            # Seed the filter state with the first valid measurement
            self.kf.statePost = np.array(
                [[measurement[0]], [measurement[1]], [0.0], [0.0]],
                dtype=np.float32
            )
            self._initialized = True

        if not self._initialized:
            # No measurement ever — return origin as placeholder
            return (0.0, 0.0)

        # ── Predict step ──────────────────────────────────────────────────────
        # Advances the state by one time step using the transition matrix.
        # This MUST be called every frame, even when there is no measurement.
        predicted = self.kf.predict()

        # ── Correct step ──────────────────────────────────────────────────────
        if measurement is not None and confidence >= min_confidence:
            # Scale measurement noise inversely with confidence:
            # high confidence → trust measurement more (smaller noise)
            # low  confidence → trust prediction more (larger noise)
            noise_scale = 1.0 + (1.0 - confidence) * 5.0
            self.kf.measurementNoiseCov = (
                np.eye(2, dtype=np.float32)
                * self._base_meas_noise
                * noise_scale
            )

            meas_array = np.array(
                [[measurement[0]], [measurement[1]]], dtype=np.float32
            )
            corrected = self.kf.correct(meas_array)
            x, y = float(corrected[0]), float(corrected[1])
        else:
            # No valid measurement — use pure prediction (coasting)
            x, y = float(predicted[0]), float(predicted[1])

        return (x, y)

    def reset(self):
        """Reset the filter state — call this when tracking resumes after a long gap."""
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self._initialized    = False

    @classmethod
    def from_config(cls, cfg: dict, screen_mode: bool = False) -> "PenKalmanFilter":
        """Build from config. When `screen_mode` is True, prefer the
        `screen_*_noise` knobs because measurements live in screen pixels
        (variance is tens to hundreds of px²) rather than plane mm. Falls
        back to the mm-tuned values if the screen-specific keys are absent."""
        k = cfg["kalman"]
        if screen_mode:
            return cls(
                process_noise     = float(k.get("screen_process_noise",     k["process_noise"])),
                measurement_noise = float(k.get("screen_measurement_noise", k["measurement_noise"])),
            )
        return cls(
            process_noise     = k["process_noise"],
            measurement_noise = k["measurement_noise"],
        )
