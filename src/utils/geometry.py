"""
Shared geometry utilities for bounding-box operations.

Centralizes math that was previously duplicated across
possession.py, court_detector.py, and jersey_classifier.py.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Bounding-box primitives  (all expect xyxy = [x1, y1, x2, y2])
# ---------------------------------------------------------------------------

def bbox_foot_center(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-center — the player's "foot position"."""
    return np.array([(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]])


def bbox_center(xyxy: np.ndarray) -> np.ndarray:
    """Geometric center of a bounding box."""
    return np.array([(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0])


def bbox_top_center(xyxy: np.ndarray) -> np.ndarray:
    """Top-center — useful for label placement."""
    return np.array([(xyxy[0] + xyxy[2]) / 2.0, xyxy[1]])


def bbox_dimensions(xyxy: np.ndarray) -> Tuple[float, float]:
    """Returns (width, height)."""
    return float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])


def bbox_area(xyxy: np.ndarray) -> float:
    """Area (clamped to >= 0)."""
    w, h = bbox_dimensions(xyxy)
    return max(0.0, w) * max(0.0, h)


def bbox_aspect_ratio(xyxy: np.ndarray) -> float:
    """Height / width. Returns 0 if width is zero."""
    w, h = bbox_dimensions(xyxy)
    return h / w if w > 0 else 0.0


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union of two xyxy boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Point utilities
# ---------------------------------------------------------------------------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


def point_in_mask(point: np.ndarray, mask: np.ndarray) -> bool:
    """
    Check if a 2-D point lies inside a binary mask.

    Args:
        point: (x, y)
        mask:  (H, W) uint8 array where non-zero = inside
    """
    h, w = mask.shape[:2]
    x = int(np.clip(point[0], 0, w - 1))
    y = int(np.clip(point[1], 0, h - 1))
    return bool(mask[y, x] > 0)


def clamp_point(
    point: np.ndarray, width: int, height: int
) -> Tuple[int, int]:
    """Clamp a point to valid pixel coordinates."""
    x = int(max(0, min(point[0], width - 1)))
    y = int(max(0, min(point[1], height - 1)))
    return x, y


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------

def crop_region(
    frame: np.ndarray,
    xyxy: np.ndarray,
    top_frac: float = 0.0,
    bottom_frac: float = 1.0,
) -> np.ndarray:
    """
    Crop a vertical sub-region of a bounding box from a frame.

    Args:
        frame:       (H, W, C) image
        xyxy:        [x1, y1, x2, y2]
        top_frac:    Start of crop as fraction of bbox height (0 = top)
        bottom_frac: End   of crop as fraction of bbox height (1 = bottom)

    Returns:
        Cropped region, or a 1×1 zero array if the crop is invalid.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

    bh = y2 - y1
    cy1 = y1 + int(bh * top_frac)
    cy2 = y1 + int(bh * bottom_frac)

    cy1 = max(0, min(cy1, fh - 1))
    cy2 = max(0, min(cy2, fh - 1))
    x1  = max(0, min(x1, fw - 1))
    x2  = max(0, min(x2, fw - 1))

    if cy2 <= cy1 or x2 <= x1:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    return frame[cy1:cy2, x1:x2]


# ---------------------------------------------------------------------------
# Trajectory / physics helpers
# ---------------------------------------------------------------------------

def velocity_from_positions(
    positions: list, dt: float = 1.0
) -> np.ndarray:
    """
    Instantaneous velocity from the last two positions.

    Args:
        positions: list of (x, y) arrays/tuples, at least 2 elements
        dt:        time delta between frames (1/fps)

    Returns:
        (vx, vy) velocity vector.  Zero vector if < 2 positions.
    """
    if len(positions) < 2:
        return np.zeros(2)
    p1 = np.asarray(positions[-2])
    p2 = np.asarray(positions[-1])
    return (p2 - p1) / dt if dt > 0 else np.zeros(2)


def speed_from_positions(positions: list, dt: float = 1.0) -> float:
    """Scalar speed (magnitude of velocity)."""
    return float(np.linalg.norm(velocity_from_positions(positions, dt)))
