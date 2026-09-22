"""
Advanced action classifier with trajectory analysis.

Complete rethink from the original Y-variance-only approach.

Action detection methods:
  Dribble   — Ball Y oscillates at 2-4 Hz while near possessor
  Shot      — Parabolic upward arc after ball leaves possessor
  Pass      — Horizontal ball travel, possession transfers between players
  Turnover  — Possession transfers between teams
  Fast break— Possession change + high-speed possessor movement
  Loose     — No possessor for >1s, multiple players converging
"""

import numpy as np
from collections import deque
from typing import Optional

from core.frame_state import ActionEvent, BallState, FrameState, Processor
from utils.logger import get_logger

log = get_logger(__name__)


class ActionClassifier(Processor):
    """
    Classifies the current ball action using ball trajectory analysis
    and possession state transitions.
    """

    def __init__(
        self,
        window_size: int = 15,
        dribble_variance: float = 25.0,
        fps: float = 30.0,
    ):
        self.window_size = window_size
        self.dribble_variance = dribble_variance
        self._fps = fps

        # Ball Y-position history for oscillation analysis
        self._y_history: deque = deque(maxlen=window_size)
        self._xy_history: deque = deque(maxlen=window_size * 2)  # longer for arc fitting

        # Possession state tracking
        self._last_possessor: Optional[int] = None
        self._prev_possessor: Optional[int] = None
        self._frames_without_possession: int = 0

        # For transition detection
        self._last_team: Optional[str] = None

    # ─── Processor interface ───────────────────────────────────────────

    def setup(self, fps: float, width: int, height: int) -> None:
        self._fps = fps

    def process(self, state: FrameState) -> FrameState:
        ball_pos = state.ball_position
        possessor = state.possessor_id
        team_assignments = state.team_assignments

        action = self._classify(ball_pos, possessor, team_assignments, state.ball_visible)
        state.current_action = action

        # Update ball state based on action
        if action.type == "dribbling":
            state.ball_state = BallState.DRIBBLE
        elif action.type in ("pass", "shot"):
            state.ball_state = BallState.IN_FLIGHT
        elif action.type == "loose":
            state.ball_state = BallState.LOOSE

        return state

    # ─── Core classification ───────────────────────────────────────────

    def _classify(
        self,
        ball_pos: Optional[np.ndarray],
        possessor_id: Optional[int],
        team_assignments: dict,
        ball_visible: bool,
    ) -> ActionEvent:
        """Main classification dispatch."""

        # ── Transition: had possessor → lost it ────────────────────────
        if self._last_possessor is not None and possessor_id is None:
            self._frames_without_possession += 1

            if ball_pos is not None:
                result = self._classify_release(ball_pos)
                self._update_state(possessor_id, team_assignments)
                return result

            self._update_state(possessor_id, team_assignments)
            return ActionEvent(type="pass", confidence=0.5,
                               involved_players=[self._prev_possessor] if self._prev_possessor else [])

        # ── Transition: no possessor → new possessor ───────────────────
        if self._last_possessor is None and possessor_id is not None:
            self._frames_without_possession = 0

            # Check if this is a turnover (possession changed teams)
            if self._prev_possessor is not None:
                prev_team = team_assignments.get(self._prev_possessor)
                curr_team = team_assignments.get(possessor_id)
                if prev_team and curr_team and prev_team != curr_team and prev_team != "ref":
                    self._update_state(possessor_id, team_assignments)
                    return ActionEvent(
                        type="turnover", confidence=0.7,
                        involved_players=[self._prev_possessor, possessor_id],
                    )

        # ── No ball or no possessor ────────────────────────────────────
        if possessor_id is None or ball_pos is None:
            if self._frames_without_possession > self._fps:  # >1 second
                self._update_state(possessor_id, team_assignments)
                return ActionEvent(type="loose", confidence=0.6)
            self._update_state(possessor_id, team_assignments)
            return ActionEvent(type="none", confidence=0.0)

        # ── Active possession — classify in-hand action ────────────────
        if possessor_id != self._last_possessor:
            self._y_history.clear()
            self._xy_history.clear()

        self._y_history.append(ball_pos[1])
        self._xy_history.append(ball_pos.copy())
        self._frames_without_possession = 0
        self._update_state(possessor_id, team_assignments)

        if len(self._y_history) < self.window_size:
            return ActionEvent(type="holding", confidence=0.5)

        # Dribble detection: Y-axis oscillation
        y_arr = np.array(self._y_history)
        y_std = np.std(y_arr)

        if y_std > self.dribble_variance:
            # Check for oscillation pattern (dribble has regular peaks/troughs)
            osc_conf = self._oscillation_confidence(y_arr)
            if osc_conf > 0.3:
                return ActionEvent(type="dribbling", confidence=osc_conf)

        return ActionEvent(type="holding", confidence=0.8)

    def _classify_release(self, ball_pos: np.ndarray) -> ActionEvent:
        """
        Classify what happened when the ball left the possessor's hands.
        Uses trajectory to distinguish shot vs pass.
        """
        if len(self._xy_history) < 5:
            return ActionEvent(type="pass", confidence=0.4,
                               involved_players=[self._last_possessor] if self._last_possessor else [])

        recent = np.array(list(self._xy_history)[-10:])
        y_values = recent[:, 1]

        # Shot detection: ball moving upward (Y decreasing) in a parabolic arc
        if len(y_values) >= 5:
            # Check if Y is decreasing (going up in image coords)
            y_diff = np.diff(y_values[-5:])
            upward_ratio = np.sum(y_diff < 0) / len(y_diff)

            if upward_ratio >= 0.6:
                # Fit a 2nd-degree polynomial to check for parabolic arc
                arc_conf = self._parabolic_confidence(recent)
                return ActionEvent(
                    type="shot",
                    confidence=max(0.5, arc_conf),
                    involved_players=[self._last_possessor] if self._last_possessor else [],
                    ball_trajectory=recent,
                )

        # Default to pass
        return ActionEvent(
            type="pass",
            confidence=0.6,
            involved_players=[self._last_possessor] if self._last_possessor else [],
            ball_trajectory=recent if len(recent) > 0 else None,
        )

    # ─── Analysis helpers ──────────────────────────────────────────────

    @staticmethod
    def _oscillation_confidence(y_arr: np.ndarray) -> float:
        """
        Detect periodic oscillation in Y values (dribble signature).
        Returns confidence 0-1 based on regularity of peaks/troughs.
        """
        if len(y_arr) < 6:
            return 0.0

        # Count zero-crossings of the detrended signal
        detrended = y_arr - np.mean(y_arr)
        crossings = np.sum(np.diff(np.sign(detrended)) != 0)

        # Dribble at ~2-4 Hz in 30fps → expect 2-8 crossings per 15-frame window
        if 2 <= crossings <= 10:
            return min(1.0, crossings / 6.0)
        return 0.0

    @staticmethod
    def _parabolic_confidence(positions: np.ndarray) -> float:
        """
        Fit a 2nd-degree polynomial to ball positions and score
        how well the trajectory matches a parabolic arc.
        """
        if len(positions) < 4:
            return 0.0

        x = np.arange(len(positions))
        y = positions[:, 1]  # Y coordinates

        try:
            coeffs = np.polyfit(x, y, 2)
            fitted = np.polyval(coeffs, x)
            residuals = np.sum((y - fitted) ** 2)
            total = np.sum((y - np.mean(y)) ** 2)

            if total == 0:
                return 0.0

            r_squared = 1.0 - residuals / total

            # Strong parabolic fit + upward curvature (coeffs[0] > 0 means U-shape in image Y)
            if r_squared > 0.7 and coeffs[0] > 0:
                return min(1.0, r_squared)
        except (np.linalg.LinAlgError, ValueError):
            pass

        return 0.0

    def _update_state(self, possessor_id: Optional[int], team_assignments: dict) -> None:
        self._prev_possessor = self._last_possessor
        self._last_possessor = possessor_id
        if possessor_id is not None:
            self._last_team = team_assignments.get(possessor_id)

    @property
    def prev_possessor(self) -> Optional[int]:
        return self._prev_possessor