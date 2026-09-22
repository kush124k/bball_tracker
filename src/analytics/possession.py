"""
Possession engine with temporal smoothing and multi-point proximity.

Critical improvements over original:
- Uses Kalman-filtered ball position (always available, vs. raw YOLO missing 30-50%)
- Temporal smoothing: requires N consecutive frames before switching possession
- Multi-point proximity: checks body center + foot + hands (if available)
- Returns confidence score alongside possessor ID
- Implements Processor interface
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import supervision as sv

from core.frame_state import BallState, FrameState, Processor
from utils.geometry import bbox_center, bbox_foot_center, euclidean_distance
from utils.logger import get_logger

log = get_logger(__name__)


class PossessionEngine(Processor):
    """
    Determines which player has the ball using proximity analysis
    with temporal smoothing to eliminate single-frame flicker.
    """

    def __init__(
        self,
        threshold: float = 160.0,
        smoothing_frames: int = 5,
    ):
        """
        Args:
            threshold:        Max pixel distance for "in possession".
            smoothing_frames: Consecutive frames a candidate must hold
                              before possession officially transfers.
        """
        self.threshold = threshold
        self.smoothing_frames = max(1, smoothing_frames)

        self._current_possessor: Optional[int] = None
        self._candidate: Optional[int] = None
        self._candidate_streak: int = 0
        self._hand_positions: dict = {}   # set externally if keypoint is enabled

    # ─── Processor interface ───────────────────────────────────────────

    def process(self, state: FrameState) -> FrameState:
        # Use Kalman-filtered ball position (always available after init)
        ball_pos = state.ball_position
        tracked = state.tracked_players

        if ball_pos is None or tracked is None or len(tracked) == 0:
            state.possessor_id = self._current_possessor
            state.possession_confidence = 0.0
            return state

        if tracked.tracker_id is None:
            state.possessor_id = self._current_possessor
            state.possession_confidence = 0.0
            return state

        raw_id, confidence = self._find_closest(ball_pos, tracked, state.team_assignments)

        # Temporal smoothing
        possessor = self._apply_smoothing(raw_id)

        state.possessor_id = possessor
        state.possession_confidence = confidence

        # Update ball state
        if possessor is not None:
            state.ball_state = BallState.HELD
        elif state.ball_state == BallState.HELD:
            state.ball_state = BallState.IN_FLIGHT

        return state

    # ─── Core logic ────────────────────────────────────────────────────

    def _find_closest(
        self,
        ball_pos: np.ndarray,
        tracked: sv.Detections,
        team_assignments: dict,
    ) -> Tuple[Optional[int], float]:
        """
        Find the closest player to the ball using multi-point proximity.
        Returns (tracker_id, confidence).
        """
        best_dist = float("inf")
        best_id = None

        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None:
                continue

            box = tracked.xyxy[i]

            # Multi-point: check foot, body center, and hands (if available)
            foot = bbox_foot_center(box)
            center = bbox_center(box)

            d_foot = euclidean_distance(ball_pos, foot)
            d_center = euclidean_distance(ball_pos, center)
            d_min = min(d_foot, d_center)

            # If hand positions are available (from keypoint detector)
            if tid in self._hand_positions:
                d_hand = euclidean_distance(ball_pos, self._hand_positions[tid])
                d_min = min(d_min, d_hand)

            if d_min < best_dist:
                best_dist = d_min
                best_id = tid

        if best_dist <= self.threshold:
            confidence = max(0.0, 1.0 - (best_dist / self.threshold))
            return best_id, confidence

        return None, 0.0

    def _apply_smoothing(self, raw_id: Optional[int]) -> Optional[int]:
        """
        Require N consecutive frames of the same candidate before
        officially switching possession.
        """
        if raw_id == self._current_possessor:
            # Same as current — reset candidate tracking
            self._candidate = None
            self._candidate_streak = 0
            return self._current_possessor

        if raw_id is None:
            # Ball is loose — keep current possessor for a short grace period
            self._candidate = None
            self._candidate_streak = 0
            return self._current_possessor

        # New candidate detected
        if raw_id == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = raw_id
            self._candidate_streak = 1

        if self._candidate_streak >= self.smoothing_frames:
            old = self._current_possessor
            self._current_possessor = raw_id
            self._candidate = None
            self._candidate_streak = 0
            if old != raw_id:
                log.debug("Possession: #%s → #%s", old, raw_id)
            return raw_id

        return self._current_possessor

    def set_hand_positions(self, hand_positions: dict) -> None:
        """Inject hand positions from keypoint detector for this frame."""
        self._hand_positions = hand_positions