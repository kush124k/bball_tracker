"""
Game state manager with video-time tracking and event logging.

Critical fix: Uses frame_index / fps for all timing instead of time.time().
The old version measured how long your CPU took to process, not game time.

New capabilities:
- Video-time durations
- Play-by-play event log
- Frame-by-frame possession timeline
- Team-level stat aggregation
- Per-player detailed stats
"""

from collections import defaultdict
from typing import Dict, List, Optional

from core.frame_state import GameEvent, FrameState, Processor
from utils.logger import get_logger

log = get_logger(__name__)


class StateManager(Processor):
    """
    Central game state accumulator.

    Tracks possession durations, action counts, and builds a
    play-by-play event log using video timestamps.
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps

        # Possession tracking
        self._current_possessor: Optional[int] = None
        self._possession_start_frame: int = 0

        # Per-player stats
        self._stats: Dict[int, dict] = {}

        # Event log — the play-by-play
        self._events: List[GameEvent] = []

        # Frame-by-frame possession timeline
        self._timeline: List[Optional[int]] = []

        # Team possession frames
        self._team_possession_frames: Dict[str, int] = defaultdict(int)

    # ─── Processor interface ───────────────────────────────────────────

    def setup(self, fps: float, width: int, height: int) -> None:
        self._fps = fps

    def process(self, state: FrameState) -> FrameState:
        possessor = state.possessor_id
        action = state.current_action
        team_assignments = state.team_assignments
        frame_idx = state.frame_index

        # Timeline
        self._timeline.append(possessor)

        # Track team possession
        if possessor is not None:
            team = team_assignments.get(possessor, "unknown")
            if team in ("team_a", "team_b"):
                self._team_possession_frames[team] += 1

        # Possession change
        if possessor is not None and possessor != self._current_possessor:
            # Log duration for previous possessor
            self._log_hold_time(self._current_possessor, frame_idx)

            # Log event
            self._events.append(GameEvent(
                frame_index=frame_idx,
                timestamp=frame_idx / self._fps,
                event_type="possession_change",
                player_id=possessor,
                target_player_id=self._current_possessor,
                team=team_assignments.get(possessor),
            ))

            self._current_possessor = possessor
            self._possession_start_frame = frame_idx

        # Action events
        if action.type in ("pass", "shot", "turnover"):
            player = action.involved_players[0] if action.involved_players else possessor
            self._ensure_player(player)

            self._events.append(GameEvent(
                frame_index=frame_idx,
                timestamp=frame_idx / self._fps,
                event_type=action.type,
                player_id=player,
                team=team_assignments.get(player) if player else None,
                details=f"confidence={action.confidence:.2f}",
            ))

            if player is not None:
                if action.type == "pass":
                    self._stats[player]["passes"] += 1
                elif action.type == "shot":
                    self._stats[player]["shots"] += 1
                elif action.type == "turnover":
                    self._stats[player]["turnovers"] += 1

        elif action.type == "dribbling" and possessor is not None:
            self._ensure_player(possessor)
            self._stats[possessor]["dribble_frames"] += 1

        return state

    def teardown(self) -> None:
        """Flush the last possessor's hold time."""
        total_frames = len(self._timeline)
        self._log_hold_time(self._current_possessor, total_frames)
        log.info("StateManager finalized: %d events, %d players tracked",
                 len(self._events), len(self._stats))

    # ─── Internal ──────────────────────────────────────────────────────

    def _log_hold_time(self, player_id: Optional[int], current_frame: int) -> None:
        if player_id is None:
            return
        self._ensure_player(player_id)
        duration_frames = current_frame - self._possession_start_frame
        duration_seconds = duration_frames / self._fps if self._fps > 0 else 0
        self._stats[player_id]["hold_time"] += duration_seconds

    def _ensure_player(self, player_id: Optional[int]) -> None:
        if player_id is None:
            return
        if player_id not in self._stats:
            self._stats[player_id] = {
                "hold_time": 0.0,
                "dribble_frames": 0,
                "passes": 0,
                "shots": 0,
                "turnovers": 0,
            }

    # ─── Public accessors ──────────────────────────────────────────────

    def get_summary(self) -> Dict[int, dict]:
        """Per-player stats with dribble_frames converted to seconds."""
        out = {}
        for pid, s in self._stats.items():
            out[pid] = {
                "hold_time": round(s["hold_time"], 1),
                "dribble_time": round(s["dribble_frames"] / self._fps, 1) if self._fps > 0 else 0,
                "passes": s["passes"],
                "shots": s["shots"],
                "turnovers": s["turnovers"],
            }
        return out

    def get_events(self) -> List[GameEvent]:
        return self._events

    def get_timeline(self) -> List[Optional[int]]:
        return self._timeline

    def get_possession_pct(self) -> Dict[str, float]:
        """Return possession percentage per team."""
        total = sum(self._team_possession_frames.values())
        if total == 0:
            return {"team_a": 0.0, "team_b": 0.0}
        return {
            team: round(frames / total * 100, 1)
            for team, frames in self._team_possession_frames.items()
        }