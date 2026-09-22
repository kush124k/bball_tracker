"""Tests for utils/geometry.py — all pure math, no YOLO needed."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.geometry import (
    bbox_foot_center,
    bbox_center,
    bbox_top_center,
    bbox_dimensions,
    bbox_area,
    bbox_aspect_ratio,
    bbox_iou,
    euclidean_distance,
    point_in_mask,
    clamp_point,
    crop_region,
    velocity_from_positions,
    speed_from_positions,
)


class TestBboxPrimitives:
    def test_foot_center(self):
        box = np.array([100, 200, 200, 400])
        result = bbox_foot_center(box)
        assert result[0] == 150.0  # x center
        assert result[1] == 400.0  # bottom y

    def test_center(self):
        box = np.array([100, 200, 200, 400])
        result = bbox_center(box)
        assert result[0] == 150.0
        assert result[1] == 300.0

    def test_top_center(self):
        box = np.array([100, 200, 200, 400])
        result = bbox_top_center(box)
        assert result[0] == 150.0
        assert result[1] == 200.0

    def test_dimensions(self):
        box = np.array([100, 200, 250, 500])
        w, h = bbox_dimensions(box)
        assert w == 150.0
        assert h == 300.0

    def test_area(self):
        box = np.array([0, 0, 10, 20])
        assert bbox_area(box) == 200.0

    def test_area_zero(self):
        box = np.array([10, 10, 10, 10])
        assert bbox_area(box) == 0.0

    def test_aspect_ratio(self):
        box = np.array([0, 0, 100, 200])
        assert bbox_aspect_ratio(box) == 2.0

    def test_aspect_ratio_zero_width(self):
        box = np.array([10, 10, 10, 20])
        assert bbox_aspect_ratio(box) == 0.0


class TestIoU:
    def test_perfect_overlap(self):
        box = np.array([0, 0, 10, 10])
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        # Intersection = 5x5 = 25
        # Union = 100 + 100 - 25 = 175
        assert bbox_iou(a, b) == pytest.approx(25.0 / 175.0, abs=0.01)


class TestPointUtils:
    def test_euclidean(self):
        a = np.array([0, 0])
        b = np.array([3, 4])
        assert euclidean_distance(a, b) == pytest.approx(5.0)

    def test_point_in_mask_inside(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        assert point_in_mask(np.array([50, 50]), mask) is True

    def test_point_in_mask_outside(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        assert point_in_mask(np.array([10, 10]), mask) is False

    def test_clamp_point(self):
        assert clamp_point(np.array([-5, 500]), 100, 100) == (0, 99)
        assert clamp_point(np.array([50, 50]), 100, 100) == (50, 50)


class TestCropRegion:
    def test_basic_crop(self):
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        box = np.array([10, 20, 50, 120])
        crop = crop_region(frame, box, 0.0, 0.5)
        assert crop.shape[0] == 50  # half of 100px height
        assert crop.shape[1] == 40  # 50 - 10

    def test_invalid_crop(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        box = np.array([10, 10, 10, 10])  # zero-size box
        crop = crop_region(frame, box)
        assert crop.shape == (1, 1, 3)


class TestVelocity:
    def test_velocity_from_two_positions(self):
        pos = [np.array([0, 0]), np.array([3, 4])]
        vel = velocity_from_positions(pos, dt=1.0)
        assert vel[0] == pytest.approx(3.0)
        assert vel[1] == pytest.approx(4.0)

    def test_speed_from_positions(self):
        pos = [np.array([0, 0]), np.array([3, 4])]
        assert speed_from_positions(pos, dt=1.0) == pytest.approx(5.0)

    def test_insufficient_positions(self):
        pos = [np.array([0, 0])]
        vel = velocity_from_positions(pos)
        assert vel[0] == 0.0
        assert vel[1] == 0.0
