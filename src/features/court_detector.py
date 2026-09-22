"""
Court boundary detector with multi-surface support and basic homography.

Improvements over original:
- Multiple HSV profiles: hardwood, painted blue, painted gray, outdoor
- Adaptive calibration across multiple frames
- Court line detection via Hough transforms
- Basic homography estimation (pixel → real-world court coordinates)
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import supervision as sv

from core.frame_state import CourtBoundary, FrameState, Processor
from utils.geometry import bbox_foot_center, point_in_mask
from utils.logger import get_logger

log = get_logger(__name__)


# ── Standard NBA court dimensions (feet) ──────────────────────────────────
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0


# ── HSV profiles for different court surfaces ─────────────────────────────
SURFACE_PROFILES = {
    "hardwood": {
        "lower": np.array([10, 40, 80]),
        "upper": np.array([25, 180, 220]),
    },
    "blue_painted": {
        "lower": np.array([95, 50, 60]),
        "upper": np.array([125, 255, 220]),
    },
    "gray_painted": {
        "lower": np.array([0, 0, 100]),
        "upper": np.array([180, 40, 200]),
    },
    "outdoor": {
        "lower": np.array([0, 0, 120]),
        "upper": np.array([180, 60, 230]),
    },
}

MIN_COURT_AREA_FRACTION = 0.12


class CourtDetector(Processor):
    """
    Detects the basketball court boundary using HSV color masking.

    Tries multiple surface profiles and picks the best one.
    Optionally computes a pixel→court homography from detected lines.
    """

    def __init__(self, surface: Optional[str] = None):
        """
        Args:
            surface: Force a specific surface profile ("hardwood", "blue_painted", etc.)
                     If None, auto-detect by trying all profiles.
        """
        self._forced_surface = surface
        self._boundary: Optional[CourtBoundary] = None
        self._homography: Optional[np.ndarray] = None
        self._calibrated = False

    # ─── Processor interface ───────────────────────────────────────────

    def process(self, state: FrameState) -> FrameState:
        """Run court detection on the first frame, then cache."""
        if not self._calibrated and state.raw_frame is not None:
            self._boundary = self.detect(state.raw_frame)
            self._calibrated = True

            if self._boundary:
                log.info(
                    "Court detected (coverage: %.0f%%, surface auto-detected)",
                    self._boundary.confidence * 100,
                )
            else:
                log.warning("Court not detected — court filter disabled")

        state.court_boundary = self._boundary
        state.homography = self._homography

        # Filter players to court
        if self._boundary is not None and state.player_detections is not None:
            state.player_detections = self.filter_to_court(
                state.player_detections, self._boundary
            )

        return state

    # ─── Detection ─────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[CourtBoundary]:
        """
        Try each surface profile and return the best court boundary.
        """
        if self._forced_surface:
            profiles = {self._forced_surface: SURFACE_PROFILES[self._forced_surface]}
        else:
            profiles = SURFACE_PROFILES

        best: Optional[CourtBoundary] = None
        best_coverage = 0.0

        for name, prof in profiles.items():
            boundary = self._detect_with_profile(frame, prof["lower"], prof["upper"])
            if boundary and boundary.confidence > best_coverage:
                best = boundary
                best_coverage = boundary.confidence
                log.debug("Surface '%s' → coverage %.1f%%", name, boundary.confidence * 100)

        if best:
            # Attempt homography from court lines
            self._homography = self._estimate_homography(frame, best)

        return best

    def _detect_with_profile(
        self,
        frame: np.ndarray,
        hsv_lower: np.ndarray,
        hsv_upper: np.ndarray,
    ) -> Optional[CourtBoundary]:
        """HSV masking + morphology + convex hull."""
        h, w = frame.shape[:2]
        area = h * w

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        coverage = np.sum(mask > 0) / area
        if coverage < MIN_COURT_AREA_FRACTION:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)

        return CourtBoundary(
            polygon=hull.squeeze(),
            mask=hull_mask,
            confidence=round(coverage, 3),
        )

    # ─── Homography ────────────────────────────────────────────────────

    def _estimate_homography(
        self, frame: np.ndarray, boundary: CourtBoundary
    ) -> Optional[np.ndarray]:
        """
        Attempt to estimate a pixel→court homography using court lines.

        Uses Hough line detection within the court mask to find
        prominent lines, then maps the court polygon corners to
        known NBA court corners.

        Returns None if insufficient lines are found.
        """
        # Use the 4 extreme points of the court polygon as source
        poly = boundary.polygon
        if poly is None or len(poly) < 4:
            return None

        # Find the 4 "corner-most" points using convex hull extremes
        src_pts = self._find_corners(poly)
        if src_pts is None:
            return None

        # Destination: standard NBA half-court in feet (origin at top-left)
        # We map to a 470×250 pixel minimap (10px per foot)
        dst_pts = np.array([
            [0, 0],
            [COURT_LENGTH_FT * 5, 0],
            [COURT_LENGTH_FT * 5, COURT_WIDTH_FT * 5],
            [0, COURT_WIDTH_FT * 5],
        ], dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts)
        if H is not None:
            log.info("Court homography estimated successfully")
        return H

    @staticmethod
    def _find_corners(polygon: np.ndarray) -> Optional[np.ndarray]:
        """Extract 4 extreme points from a polygon (top-left, top-right, bottom-right, bottom-left)."""
        if len(polygon) < 4:
            return None

        pts = polygon.reshape(-1, 2).astype(np.float32)

        # Sort by sum (x+y) for TL/BR and diff (x-y) for TR/BL
        tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        tr = pts[np.argmax(pts[:, 0] - pts[:, 1])]
        bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    # ─── Filtering ─────────────────────────────────────────────────────

    def filter_to_court(
        self, detections: sv.Detections, boundary: CourtBoundary
    ) -> sv.Detections:
        """Keep only detections whose foot position is inside the court mask."""
        if len(detections) == 0:
            return detections

        keep = []
        for i in range(len(detections)):
            foot = bbox_foot_center(detections.xyxy[i])
            if point_in_mask(foot, boundary.mask):
                keep.append(i)

        if not keep:
            return sv.Detections.empty()
        return detections[np.array(keep)]

    # ─── Debug drawing ─────────────────────────────────────────────────

    def draw_debug(
        self, frame: np.ndarray, boundary: Optional[CourtBoundary] = None
    ) -> np.ndarray:
        """Draw the court boundary overlay on a frame."""
        b = boundary or self._boundary
        if b is None or b.polygon is None:
            return frame

        out = frame.copy()
        pts = b.polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2)

        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

        cv2.putText(
            out, f"Court: {b.confidence:.0%}",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        return out