"""
Player movement analytics: speed, distance, heat maps, sprint detection.

Uses trajectory data from the PlayerTracker to compute per-player
movement metrics.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from core.frame_state import FrameState, Processor
from utils.geometry import euclidean_distance
from utils.logger import get_logger

log = get_logger(__name__)


class MovementAnalyzer(Processor):
    """
    Computes per-player movement analytics from trajectory data.

    Writes to FrameState.player_velocities (already set by tracker)
    and maintains internal accumulators for:
    - Cumulative distance per player
    - Top / average speed
    - Sprint detection (speed bursts above threshold)
    - Heat map accumulation
    """

    DEFAULT_SPRINT_THRESHOLD = 8.0   # px/frame — tune based on resolution

    def __init__(
        self,
        sprint_threshold: float = DEFAULT_SPRINT_THRESHOLD,
        heatmap_resolution: Tuple[int, int] = (94, 50),
    ):
        """
        Args:
            sprint_threshold: Speed in px/frame above which = sprint.
            heatmap_resolution: (cols, rows) for the heat map grid.
        """
        self.sprint_threshold = sprint_threshold
        self._heatmap_res = heatmap_resolution

        # Per-player accumulators
        self._distance: Dict[int, float] = defaultdict(float)
        self._speeds: Dict[int, list] = defaultdict(list)
        self._top_speed: Dict[int, float] = defaultdict(float)
        self._sprint_frames: Dict[int, int] = defaultdict(int)
        self._last_pos: Dict[int, Optional[np.ndarray]] = {}

        # Heat maps: (rows, cols) per player
        self._heatmaps: Dict[int, np.ndarray] = {}
        self._frame_width = 1280
        self._frame_height = 720

    # ─── Processor interface ───────────────────────────────────────────

    def setup(self, fps: float, width: int, height: int) -> None:
        self._fps = fps
        self._frame_width = width
        self._frame_height = height

    def process(self, state: FrameState) -> FrameState:
        trajectories = state.trajectories

        for tid, trail in trajectories.items():
            if len(trail) < 2:
                continue

            curr = np.array(trail[-1])
            prev = np.array(trail[-2])

            # Distance
            d = euclidean_distance(curr, prev)
            self._distance[tid] += d

            # Speed (px/frame)
            speed = d  # per-frame distance = instantaneous speed in px/frame
            self._speeds[tid].append(speed)
            if speed > self._top_speed[tid]:
                self._top_speed[tid] = speed

            # Sprint detection
            if speed > self.sprint_threshold:
                self._sprint_frames[tid] += 1

            # Heat map accumulation
            self._accumulate_heatmap(tid, curr)

        return state

    # ─── Heat map ──────────────────────────────────────────────────────

    def _accumulate_heatmap(self, tid: int, pos: np.ndarray) -> None:
        rows, cols = self._heatmap_res[1], self._heatmap_res[0]
        if tid not in self._heatmaps:
            self._heatmaps[tid] = np.zeros((rows, cols), dtype=np.float32)

        # Map pixel position to grid cell
        gx = int(pos[0] / self._frame_width * (cols - 1))
        gy = int(pos[1] / self._frame_height * (rows - 1))
        gx = max(0, min(gx, cols - 1))
        gy = max(0, min(gy, rows - 1))
        self._heatmaps[tid][gy, gx] += 1.0

    # ─── Public accessors ──────────────────────────────────────────────

    def get_distance(self, tid: int) -> float:
        """Cumulative distance in pixels."""
        return self._distance.get(tid, 0.0)

    def get_avg_speed(self, tid: int) -> float:
        """Average speed in px/frame."""
        speeds = self._speeds.get(tid, [])
        return float(np.mean(speeds)) if speeds else 0.0

    def get_top_speed(self, tid: int) -> float:
        """Top instantaneous speed in px/frame."""
        return self._top_speed.get(tid, 0.0)

    def get_sprint_time(self, tid: int) -> float:
        """Total sprint time in seconds (at current fps)."""
        return self._sprint_frames.get(tid, 0) / self._fps if self._fps > 0 else 0.0

    def get_heatmap(self, tid: int) -> Optional[np.ndarray]:
        """Return the (rows, cols) heat map for a player, or None."""
        return self._heatmaps.get(tid)

    def get_all_stats(self) -> Dict[int, dict]:
        """Return a summary dict for all tracked players."""
        all_tids = set(self._distance.keys())
        stats = {}
        for tid in all_tids:
            stats[tid] = {
                "distance_px": round(self._distance[tid], 1),
                "avg_speed": round(self.get_avg_speed(tid), 2),
                "top_speed": round(self._top_speed[tid], 2),
                "sprint_time_s": round(self.get_sprint_time(tid), 1),
            }
        return stats

    def get_team_spacing(
        self, trajectories: Dict[int, deque], team_assignments: Dict[int, str], team: str
    ) -> float:
        """
        Average pairwise distance between teammates' current positions.
        Higher = more spread out (better offensive spacing).
        """
        positions = []
        for tid, trail in trajectories.items():
            if team_assignments.get(tid) == team and len(trail) > 0:
                positions.append(np.array(trail[-1]))

        if len(positions) < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total += euclidean_distance(positions[i], positions[j])
                count += 1

        return total / count if count > 0 else 0.0
