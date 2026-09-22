"""
Central data model for the processing pipeline.

Every pipeline stage reads and writes to a single FrameState instance.
This replaces the loose-variable passing in the old main.py.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
import supervision as sv


# ── Action & Game events ───────────────────────────────────────────────────

@dataclass
class ActionEvent:
    """Result of the action classifier for one frame."""
    type: str = "none"                     # dribbling | holding | pass | shot | turnover | fast_break | loose | none
    confidence: float = 0.0                # 0.0 – 1.0
    involved_players: List[int] = field(default_factory=list)   # tracker IDs
    ball_trajectory: Optional[np.ndarray] = None                # recent (N, 2) positions

    def __repr__(self) -> str:
        return f"Action({self.type}, conf={self.confidence:.2f})"


@dataclass
class GameEvent:
    """A logged event in the play-by-play timeline."""
    frame_index: int
    timestamp: float                       # video-time seconds
    event_type: str                        # possession_change | pass | shot | turnover | fast_break
    player_id: Optional[int] = None
    target_player_id: Optional[int] = None # e.g. pass recipient
    team: Optional[str] = None             # team_a | team_b
    details: str = ""

    def __repr__(self) -> str:
        return f"[{self.timestamp:.1f}s] {self.event_type} #{self.player_id}"


# ── Court boundary (kept from original, promoted here for shared access) ──

@dataclass
class CourtBoundary:
    polygon: np.ndarray           # convex hull of the court floor
    mask: np.ndarray              # binary mask  (H, W) uint8
    confidence: float             # fraction of frame covered


# ── Ball state machine ─────────────────────────────────────────────────────

class BallState:
    """Enum-like constants for the ball state machine."""
    HELD      = "held"
    IN_FLIGHT = "in_flight"
    DRIBBLE   = "dribble"
    LOOSE     = "loose"
    MISSING   = "missing"        # YOLO can't see it, Kalman is predicting


# ── The main FrameState ───────────────────────────────────────────────────

@dataclass
class FrameState:
    """
    Single source of truth for one frame of the pipeline.

    Created fresh for each frame by the pipeline engine.
    Each processor reads what it needs and writes its results.
    """

    # ─── Identity ──────────────────────────────────────────────────────
    frame_index: int = 0
    timestamp: float = 0.0                 # video-time seconds  (frame_index / fps)
    fps: float = 30.0

    # ─── Raw input ─────────────────────────────────────────────────────
    raw_frame: Optional[np.ndarray] = None

    # ─── Detection results ─────────────────────────────────────────────
    all_detections: Optional[sv.Detections] = None      # raw YOLO output (filtered by class)
    player_detections: Optional[sv.Detections] = None    # class 0, post-court/jersey filter
    ball_detections: Optional[sv.Detections] = None      # class 32, raw

    # ─── Tracking results ──────────────────────────────────────────────
    tracked_players: Optional[sv.Detections] = None      # with tracker_id assigned
    trajectories: Dict[int, Deque] = field(
        default_factory=dict
    )  # tracker_id → deque of (x, y)
    player_velocities: Dict[int, float] = field(
        default_factory=dict
    )  # tracker_id → speed (px/frame)

    # ─── Ball tracking (Kalman) ────────────────────────────────────────
    ball_position: Optional[np.ndarray] = None   # Kalman-filtered (x, y) — always available
    ball_velocity: Optional[np.ndarray] = None   # (vx, vy) from Kalman state
    ball_visible: bool = False                   # True if YOLO saw the ball this frame
    ball_state: str = BallState.MISSING
    ball_trajectory: Deque = field(
        default_factory=lambda: deque(maxlen=90)
    )  # last ~3s at 30fps

    # ─── Court ─────────────────────────────────────────────────────────
    court_boundary: Optional[CourtBoundary] = None
    homography: Optional[np.ndarray] = None      # 3×3 pixel → court coords

    # ─── Team info ─────────────────────────────────────────────────────
    team_assignments: Dict[int, str] = field(
        default_factory=dict
    )  # tracker_id → "team_a" | "team_b" | "ref"

    # ─── Analytics ─────────────────────────────────────────────────────
    possessor_id: Optional[int] = None
    possession_confidence: float = 0.0
    current_action: ActionEvent = field(default_factory=ActionEvent)

    # ─── Rendering flags (set by HUD) ─────────────────────────────────
    annotated_frame: Optional[np.ndarray] = None


# ── Pipeline processor interface ───────────────────────────────────────────

class Processor:
    """
    Base class for pipeline stages.

    Subclass and implement ``process(state)`` to read/write
    fields on FrameState.
    """

    def setup(self, fps: float, width: int, height: int) -> None:
        """Called once before the first frame. Override if needed."""
        pass

    def process(self, state: FrameState) -> FrameState:
        """Process a single frame. Must return the (mutated) state."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Called after the last frame. Override for cleanup."""
        pass
