"""Tests for detection layer: detector filtering, court filtering, ball tracker."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import supervision as sv
from features.ball_tracker import BallTracker
from features.court_detector import CourtDetector
from core.frame_state import FrameState, CourtBoundary, BallState
from utils.geometry import bbox_foot_center, point_in_mask


# ── Ball Tracker (Kalman filter) ───────────────────────────────────────────

class TestBallTracker:
    def setup_method(self):
        self.tracker = BallTracker(max_missing=10)
        self.tracker.setup(30.0, 1280, 720)

    def _make_state(self, ball_xyxy=None):
        state = FrameState(frame_index=1, fps=30.0)
        if ball_xyxy is not None:
            state.ball_detections = sv.Detections(
                xyxy=np.array([ball_xyxy], dtype=np.float32),
                class_id=np.array([32]),
                confidence=np.array([0.8]),
            )
        else:
            state.ball_detections = sv.Detections.empty()
        return state

    def test_first_detection_initializes(self):
        state = self._make_state([100, 200, 120, 220])
        state = self.tracker.process(state)
        assert state.ball_position is not None
        assert state.ball_visible is True

    def test_no_detection_returns_none_initially(self):
        state = self._make_state(None)
        state = self.tracker.process(state)
        assert state.ball_position is None
        assert state.ball_state == BallState.MISSING

    def test_prediction_during_occlusion(self):
        """After initialization, missing detections should still predict."""
        # Initialize with a detection
        state = self._make_state([100, 200, 120, 220])
        state = self.tracker.process(state)
        assert state.ball_position is not None

        # Next frame: no detection → should still have position from prediction
        state2 = self._make_state(None)
        state2.frame_index = 2
        state2 = self.tracker.process(state2)
        assert state2.ball_position is not None
        assert state2.ball_visible is False

    def test_missing_too_long_becomes_missing(self):
        """After max_missing frames without detection → MISSING."""
        # Initialize
        state = self._make_state([100, 200, 120, 220])
        self.tracker.process(state)

        # Miss for 15 frames (max_missing=10)
        for i in range(15):
            state = self._make_state(None)
            state.frame_index = i + 2
            state = self.tracker.process(state)

        assert state.ball_state == BallState.MISSING

    def test_trajectory_accumulates(self):
        """Ball trajectory should grow as detections come in."""
        for i in range(5):
            x = 100 + i * 10
            state = self._make_state([x, 200, x + 20, 220])
            state.frame_index = i + 1
            self.tracker.process(state)

        assert len(self.tracker.trajectory) == 5

    def test_get_recent_positions(self):
        for i in range(10):
            x = 100 + i * 10
            state = self._make_state([x, 200, x + 20, 220])
            state.frame_index = i + 1
            self.tracker.process(state)

        recent = self.tracker.get_recent_positions(5)
        assert recent.shape == (5, 2)


# ── Court Filtering ────────────────────────────────────────────────────────

class TestCourtFiltering:
    def test_filter_to_court(self, sample_court_boundary):
        detector = CourtDetector()

        # Player inside court (foot at 125, 400 — within the mask)
        inside = sv.Detections(
            xyxy=np.array([[100, 200, 150, 400]], dtype=np.float32),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )

        # Player outside court (foot at 30, 700 — bottom-left, outside mask)
        outside = sv.Detections(
            xyxy=np.array([[10, 600, 50, 750]], dtype=np.float32),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
        )

        # Check that foot position of 'inside' player is actually in the mask
        foot = bbox_foot_center(inside.xyxy[0])
        assert point_in_mask(foot, sample_court_boundary.mask)

        # Filter
        filtered_in = detector.filter_to_court(inside, sample_court_boundary)
        assert len(filtered_in) == 1

    def test_empty_detections(self):
        detector = CourtDetector()
        boundary = CourtBoundary(
            polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
            mask=np.ones((100, 100), dtype=np.uint8) * 255,
            confidence=0.5,
        )
        result = detector.filter_to_court(sv.Detections.empty(), boundary)
        assert len(result) == 0
