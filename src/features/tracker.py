"""
Player tracker with trajectory history and velocity estimation.

Wraps ``sv.ByteTrack`` and adds:
- Per-ID trajectory deque  (last N positions)
- Per-ID instantaneous velocity / speed
- ``sv.DetectionsSmoother`` integration to reduce ID flicker
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Deque
import supervision as sv

from core.frame_state import FrameState, Processor
from utils.geometry import bbox_foot_center, speed_from_positions
from utils.logger import get_logger

log = get_logger(__name__)


class PlayerTracker(Processor):
    """
    Multi-object tracker for players.

    Builds on ByteTrack and adds trajectory + velocity bookkeeping
    that downstream analytics (movement, HUD trails) depend on.
    """

    def __init__(
        self,
        profile: dict | None = None,
        trail_length: int = 60,
        smoother_length: int = 5,
    ):
        """
        Args:
            profile:         Angle-specific tracker profile dict (from config).
            trail_length:    How many past foot-positions to keep per track.
            smoother_length: Window size for DetectionsSmoother.
        """
        p = profile or {}
        self._tracker = sv.ByteTrack(
            track_activation_threshold=p.get("track_activation_threshold", 0.25),
            lost_track_buffer=p.get("lost_track_buffer", 60),
            minimum_matching_threshold=p.get("minimum_matching_threshold", 0.80),
            minimum_consecutive_frames=p.get("minimum_consecutive_frames", 2),
            frame_rate=p.get("frame_rate", 30),
        )
        self._smoother = sv.DetectionsSmoother(length=smoother_length)

        self._trail_length = trail_length
        self._trajectories: Dict[int, Deque] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )
        self._velocities: Dict[int, float] = {}
        self._fps = 30.0

    # ─── Processor interface ───────────────────────────────────────────

    def setup(self, fps: float, width: int, height: int) -> None:
        self._fps = fps

    def process(self, state: FrameState) -> FrameState:
        """
        Read player_detections, write tracked_players / trajectories / velocities.
        """
        dets = state.player_detections
        if dets is None or len(dets) == 0:
            state.tracked_players = sv.Detections.empty()
            return state

        # Track → smooth
        tracked = self._tracker.update_with_detections(dets)
        tracked = self._smoother.update_with_detections(tracked)
        state.tracked_players = tracked

        # Update per-ID trajectories & velocities
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                foot = bbox_foot_center(tracked.xyxy[i])
                self._trajectories[tid].append(foot)
                self._velocities[tid] = speed_from_positions(
                    list(self._trajectories[tid]),
                    dt=1.0 / self._fps if self._fps > 0 else 1.0,
                )

        state.trajectories = dict(self._trajectories)
        state.player_velocities = dict(self._velocities)
        return state

    # ─── Public helpers ────────────────────────────────────────────────

    @property
    def trajectories(self) -> Dict[int, Deque]:
        return dict(self._trajectories)

    @property
    def velocities(self) -> Dict[int, float]:
        return dict(self._velocities)

    def get_trail(self, tracker_id: int) -> np.ndarray:
        """Return (N, 2) array of past foot positions for one player."""
        pts = list(self._trajectories.get(tracker_id, []))
        if not pts:
            return np.empty((0, 2))
        return np.array(pts)