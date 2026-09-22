"""
Shared pytest fixtures for bball_tracker tests.
"""

import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import supervision as sv
from collections import deque
from core.frame_state import FrameState, ActionEvent, CourtBoundary


@pytest.fixture
def sample_player_detections():
    """3 players at known positions."""
    return sv.Detections(
        xyxy=np.array([
            [100, 200, 150, 400],   # Player at x=125, foot_y=400
            [300, 150, 370, 380],   # Player at x=335, foot_y=380
            [500, 180, 560, 390],   # Player at x=530, foot_y=390
        ], dtype=np.float32),
        class_id=np.array([0, 0, 0]),
        confidence=np.array([0.9, 0.85, 0.88]),
        tracker_id=np.array([1, 2, 3]),
    )


@pytest.fixture
def sample_ball_detection():
    """Ball near player #1."""
    return sv.Detections(
        xyxy=np.array([[120, 350, 140, 370]], dtype=np.float32),
        class_id=np.array([32]),
        confidence=np.array([0.75]),
    )


@pytest.fixture
def sample_ball_detection_far():
    """Ball far from all players."""
    return sv.Detections(
        xyxy=np.array([[700, 100, 720, 120]], dtype=np.float32),
        class_id=np.array([32]),
        confidence=np.array([0.6]),
    )


@pytest.fixture
def empty_detections():
    """Empty detection set."""
    return sv.Detections.empty()


@pytest.fixture
def sample_frame():
    """A blank 720p frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_state(sample_frame, sample_player_detections, sample_ball_detection):
    """A populated FrameState for pipeline testing."""
    state = FrameState(
        frame_index=100,
        timestamp=100 / 30.0,
        fps=30.0,
        raw_frame=sample_frame,
        player_detections=sample_player_detections,
        ball_detections=sample_ball_detection,
        tracked_players=sample_player_detections,  # pretend already tracked
    )
    # Add some trajectory data
    for tid in [1, 2, 3]:
        state.trajectories[tid] = deque(maxlen=60)
        for j in range(10):
            state.trajectories[tid].append(np.array([100.0 + tid * 50 + j, 300.0 + j]))
    return state


@pytest.fixture
def sample_court_boundary(sample_frame):
    """A simple rectangular court boundary."""
    h, w = sample_frame.shape[:2]
    polygon = np.array([
        [50, 150],
        [w - 50, 150],
        [w - 50, h - 50],
        [50, h - 50],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    import cv2
    cv2.fillConvexPoly(mask, polygon, 255)
    return CourtBoundary(polygon=polygon, mask=mask, confidence=0.6)
