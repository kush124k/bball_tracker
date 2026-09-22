"""
Kalman-filter-based ball tracker with a state machine.

The basketball is the hardest object to track: ~20 px, fast, frequently
occluded by hands/bodies, and YOLO misses it 30-50 % of frames.
Raw per-frame detection is useless for analytics.

This module provides a **continuous, smooth ball position every frame**
by combining YOLO detections (when available) with Kalman prediction
(when occluded).

State machine
─────────────
  MISSING ──detect──▶ HELD / LOOSE
  HELD    ──release──▶ IN_FLIGHT
  IN_FLIGHT ──catch──▶ HELD
  IN_FLIGHT ──timeout──▶ LOOSE
  DRIBBLE ──stop──▶ HELD
  * ──no detect for N frames──▶ MISSING
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple
import supervision as sv

from core.frame_state import BallState, FrameState, Processor
from utils.geometry import bbox_center, euclidean_distance
from utils.logger import get_logger

log = get_logger(__name__)


class BallTracker(Processor):
    """
    Kalman-filter ball tracker.

    Wraps ``cv2.KalmanFilter`` with:
    - Automatic predict/correct cycle
    - Occlusion bridging (up to ``max_missing`` frames)
    - Ball state machine (HELD, IN_FLIGHT, DRIBBLE, LOOSE, MISSING)
    - Trajectory buffer for action classification
    """

    def __init__(
        self,
        max_missing: int = 30,
        process_noise: float = 0.05,
        measurement_noise: float = 0.5,
        trajectory_length: int = 90,
    ):
        """
        Args:
            max_missing:        Max frames to predict without a detection
                                before declaring MISSING.
            process_noise:      Kalman process noise covariance scalar.
            measurement_noise:  Kalman measurement noise covariance scalar.
            trajectory_length:  How many past positions to keep.
        """
        self.max_missing = max_missing
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.trajectory_length = trajectory_length

        # ── Kalman filter: state = (x, y, vx, vy), measurement = (x, y)
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise

        # ── Internal state
        self._initialized = False
        self._frames_missing = 0
        self._trajectory: deque = deque(maxlen=trajectory_length)
        self._state = BallState.MISSING

    # ─── Processor interface ───────────────────────────────────────────

    def setup(self, fps: float, width: int, height: int) -> None:
        self._fps = fps

    def process(self, state: FrameState) -> FrameState:
        """Read ball_detections, write ball_position / ball_velocity / ball_state."""
        ball_dets = state.ball_detections

        measurement: Optional[np.ndarray] = None
        if ball_dets is not None and len(ball_dets) > 0:
            # Pick the highest-confidence ball detection
            best_idx = int(np.argmax(ball_dets.confidence))
            measurement = bbox_center(ball_dets.xyxy[best_idx])

        pos, vel = self._step(measurement)

        state.ball_position = pos
        state.ball_velocity = vel
        state.ball_visible = measurement is not None
        state.ball_state = self._state

        if pos is not None:
            state.ball_trajectory.append(pos.copy())

        return state

    # ─── Core Kalman logic ─────────────────────────────────────────────

    def _step(
        self, measurement: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run one predict/correct cycle.

        Args:
            measurement: (x, y) from YOLO, or None if ball not detected.

        Returns:
            (position, velocity) — both (2,) arrays, or (None, None) if
            the tracker hasn't been initialized yet.
        """
        if measurement is not None:
            if not self._initialized:
                # First-ever detection — seed the filter
                self._kf.statePre = np.array(
                    [[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32
                )
                self._kf.statePost = self._kf.statePre.copy()
                self._initialized = True
                self._frames_missing = 0
                self._state = BallState.LOOSE
                pos = measurement.astype(np.float32)
                self._trajectory.append(pos.copy())
                return pos, np.zeros(2, dtype=np.float32)

            # Predict then correct
            self._kf.predict()
            corrected = self._kf.correct(
                measurement.reshape(2, 1).astype(np.float32)
            )
            self._frames_missing = 0
            pos = corrected[:2].flatten()
            vel = corrected[2:].flatten()
            self._trajectory.append(pos.copy())
            self._update_state(detected=True, vel=vel)
            return pos, vel

        # ── No measurement — predict only ──────────────────────────────
        if not self._initialized:
            return None, None

        self._frames_missing += 1

        if self._frames_missing > self.max_missing:
            self._state = BallState.MISSING
            return None, None

        predicted = self._kf.predict()
        pos = predicted[:2].flatten()
        vel = predicted[2:].flatten()
        self._trajectory.append(pos.copy())
        self._update_state(detected=False, vel=vel)
        return pos, vel

    def _update_state(self, detected: bool, vel: np.ndarray) -> None:
        """
        Transition the ball state machine.

        This is a simplified version; the full transitions (HELD ↔ IN_FLIGHT)
        are driven by the possession + action engines in later pipeline stages.
        Here we only handle:
          - MISSING → (detected) → LOOSE
          - * → (missing too long) → MISSING
        """
        speed = float(np.linalg.norm(vel))

        if self._state == BallState.MISSING and detected:
            self._state = BallState.LOOSE

        # The possession processor will upgrade LOOSE → HELD
        # The action processor will set IN_FLIGHT / DRIBBLE

    # ─── Public helpers ────────────────────────────────────────────────

    @property
    def trajectory(self) -> deque:
        return self._trajectory

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, new_state: str) -> None:
        """Allow external processors (possession, actions) to update state."""
        self._state = new_state

    def get_recent_positions(self, n: int = 15) -> np.ndarray:
        """
        Last N ball positions as (N, 2) array.
        Useful for shot-arc fitting and pass-direction analysis.
        """
        pts = list(self._trajectory)[-n:]
        if not pts:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(pts, dtype=np.float32)
