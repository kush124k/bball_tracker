"""Tests for analytics: possession, actions, state manager, movement."""

import numpy as np
import pytest
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import supervision as sv
from analytics.possession import PossessionEngine
from analytics.actions import ActionClassifier
from core.state_manager import StateManager
from analytics.movement import MovementAnalyzer
from core.frame_state import FrameState, ActionEvent


# ── Possession Engine ──────────────────────────────────────────────────────

class TestPossessionEngine:
    def setup_method(self):
        self.engine = PossessionEngine(threshold=100, smoothing_frames=3)

    def _make_state(self, ball_pos, tracked_xyxy, tracker_ids):
        state = FrameState(frame_index=1, fps=30.0)
        state.ball_position = np.array(ball_pos, dtype=np.float32) if ball_pos else None
        if tracked_xyxy is not None:
            state.tracked_players = sv.Detections(
                xyxy=np.array(tracked_xyxy, dtype=np.float32),
                class_id=np.zeros(len(tracked_xyxy), dtype=int),
                confidence=np.ones(len(tracked_xyxy), dtype=np.float32),
                tracker_id=np.array(tracker_ids),
            )
        else:
            state.tracked_players = sv.Detections.empty()
        return state

    def test_no_ball_returns_none(self):
        state = self._make_state(None, [[100, 200, 150, 400]], [1])
        state = self.engine.process(state)
        assert state.possessor_id is None

    def test_no_players_returns_none(self):
        state = self._make_state([130, 380], None, None)
        state = self.engine.process(state)
        assert state.possessor_id is None

    def test_ball_near_player(self):
        """Ball at (130, 380), player foot at (125, 400). Distance ~22px → within threshold."""
        state = self._make_state(
            [130, 380],
            [[100, 200, 150, 400]],
            [1]
        )
        # Run multiple times for smoothing
        for _ in range(5):
            state.frame_index += 1
            state = self.engine.process(state)
        assert state.possessor_id == 1

    def test_ball_far_from_all(self):
        """Ball at (700, 100), player foot at (125, 400). Too far."""
        state = self._make_state(
            [700, 100],
            [[100, 200, 150, 400]],
            [1]
        )
        state = self.engine.process(state)
        assert state.possessor_id is None

    def test_smoothing_prevents_flicker(self):
        """Possession shouldn't change on a single frame."""
        # First: establish possession with player 1
        for _ in range(5):
            state = self._make_state([130, 380], [[100, 200, 150, 400], [300, 150, 370, 380]], [1, 2])
            self.engine.process(state)

        # Single frame near player 2 → should NOT switch yet
        state = self._make_state([340, 360], [[100, 200, 150, 400], [300, 150, 370, 380]], [1, 2])
        state = self.engine.process(state)
        assert state.possessor_id == 1  # still player 1 due to smoothing


# ── Action Classifier ──────────────────────────────────────────────────────

class TestActionClassifier:
    def setup_method(self):
        self.clf = ActionClassifier(window_size=10, dribble_variance=25.0, fps=30.0)
        self.clf.setup(30.0, 1280, 720)

    def _make_state(self, ball_pos, possessor_id, team_assignments=None):
        state = FrameState(frame_index=1, fps=30.0)
        state.ball_position = np.array(ball_pos, dtype=np.float32) if ball_pos else None
        state.possessor_id = possessor_id
        state.ball_visible = ball_pos is not None
        state.team_assignments = team_assignments or {}
        return state

    def test_no_ball_no_possessor(self):
        state = self._make_state(None, None)
        state = self.clf.process(state)
        assert state.current_action.type == "none"

    def test_holding_with_small_variance(self):
        """Steady ball Y → holding."""
        for i in range(15):
            state = self._make_state([300, 350 + np.random.uniform(-2, 2)], 1)
            state.frame_index = i
            state = self.clf.process(state)
        assert state.current_action.type == "holding"

    def test_dribbling_with_oscillation(self):
        """Oscillating ball Y → dribbling."""
        for i in range(20):
            y = 350 + 40 * np.sin(i * 0.8)  # strong oscillation
            state = self._make_state([300, y], 1)
            state.frame_index = i
            state = self.clf.process(state)
        assert state.current_action.type in ("dribbling", "holding")

    def test_pass_on_possession_loss(self):
        """Possession → None → pass event."""
        # Establish possession
        for i in range(5):
            state = self._make_state([300, 350], 1)
            state.frame_index = i
            self.clf.process(state)

        # Lose possession
        state = self._make_state([300, 350], None)
        state.frame_index = 6
        state = self.clf.process(state)
        assert state.current_action.type in ("pass", "shot")


# ── State Manager ──────────────────────────────────────────────────────────

class TestStateManager:
    def test_video_time_tracking(self):
        """Hold time should be in video seconds, not wall-clock."""
        mgr = StateManager(fps=30.0)
        mgr.setup(30.0, 1280, 720)

        # Simulate 90 frames (3 seconds) of possession by player 1
        for i in range(1, 91):
            state = FrameState(frame_index=i, fps=30.0, timestamp=i / 30.0)
            state.possessor_id = 1
            state.current_action = ActionEvent(type="holding", confidence=0.8)
            state.team_assignments = {}
            mgr.process(state)

        mgr.teardown()
        summary = mgr.get_summary()
        assert 1 in summary
        assert 2.5 <= summary[1]["hold_time"] <= 3.5  # ~3 seconds

    def test_event_logging(self):
        mgr = StateManager(fps=30.0)
        mgr.setup(30.0, 1280, 720)

        # Player 1 has ball for 30 frames, then player 2
        for i in range(1, 31):
            state = FrameState(frame_index=i, fps=30.0, timestamp=i / 30.0)
            state.possessor_id = 1
            state.current_action = ActionEvent(type="holding")
            state.team_assignments = {}
            mgr.process(state)

        for i in range(31, 61):
            state = FrameState(frame_index=i, fps=30.0, timestamp=i / 30.0)
            state.possessor_id = 2
            state.current_action = ActionEvent(type="holding")
            state.team_assignments = {}
            mgr.process(state)

        events = mgr.get_events()
        possession_changes = [e for e in events if e.event_type == "possession_change"]
        assert len(possession_changes) >= 2  # at least p1 and p2

    def test_action_counting(self):
        mgr = StateManager(fps=30.0)
        mgr.setup(30.0, 1280, 720)

        state = FrameState(frame_index=1, fps=30.0, timestamp=1 / 30.0)
        state.possessor_id = 1
        state.current_action = ActionEvent(type="pass", confidence=0.8, involved_players=[1])
        state.team_assignments = {}
        mgr.process(state)

        summary = mgr.get_summary()
        assert summary[1]["passes"] == 1


# ── Movement Analyzer ─────────────────────────────────────────────────────

class TestMovementAnalyzer:
    def test_distance_accumulation(self):
        analyzer = MovementAnalyzer()
        analyzer.setup(30.0, 1280, 720)

        trajectories = {1: deque(maxlen=60)}
        for i in range(10):
            trajectories[1].append(np.array([100.0 + i * 10, 300.0]))

        state = FrameState(frame_index=10, fps=30.0)
        state.trajectories = trajectories
        analyzer.process(state)

        dist = analyzer.get_distance(1)
        assert dist > 0

    def test_speed_calculation(self):
        analyzer = MovementAnalyzer()
        analyzer.setup(30.0, 1280, 720)

        trajectories = {1: deque(maxlen=60)}
        trajectories[1].append(np.array([100.0, 300.0]))
        trajectories[1].append(np.array([110.0, 300.0]))

        state = FrameState(frame_index=2, fps=30.0)
        state.trajectories = trajectories
        analyzer.process(state)

        speed = analyzer.get_avg_speed(1)
        assert speed > 0
