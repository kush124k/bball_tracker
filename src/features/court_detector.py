import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import supervision as sv


@dataclass
class CourtBoundary:
    polygon: np.ndarray          # Convex hull of the court floor region
    mask: np.ndarray             # Binary mask of the court area
    confidence: float            # How much of the frame the court occupies


class CourtDetector:
    """
    Detects the basketball court boundary in sideline footage by isolating
    the hardwood floor using HSV color masking.

    Standard hardwood is a warm tan/orange-brown — reliably distinct from
    crowd seating, scoreboards, and walls in most arenas.

    Exposes:
        - detect(frame) -> CourtBoundary
        - filter_to_court(detections, boundary) -> sv.Detections
    """

    # HSV range for hardwood tan — tunable if court color varies
    # Hue: 10-25 covers orange-tan-brown
    # Sat: 40-180 avoids washed-out whites and dark shadows
    # Val: 80-220 avoids pure black (shadows) and pure white (lines)
    HARDWOOD_HSV_LOWER = np.array([10, 40, 80])
    HARDWOOD_HSV_UPPER = np.array([25, 180, 220])

    # Minimum fraction of frame that must be court for a valid detection
    MIN_COURT_AREA_FRACTION = 0.15

    def detect(self, frame: np.ndarray) -> Optional[CourtBoundary]:
        """
        Runs HSV masking on a single frame and returns the court boundary.
        Returns None if no court region is confidently detected.
        Call this on the first frame (or periodically) and cache the result —
        no need to run every frame for a fixed camera.
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(hsv, self.HARDWOOD_HSV_LOWER, self.HARDWOOD_HSV_UPPER)

        # Morphological cleanup — fill small holes and remove specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        court_area = np.sum(mask > 0)
        confidence = court_area / frame_area

        if confidence < self.MIN_COURT_AREA_FRACTION:
            return None

        # Find the convex hull of the court region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Take the largest contour — that's the court floor
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        # Rebuild a clean mask from just the hull polygon
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)

        return CourtBoundary(
            polygon=hull.squeeze(),
            mask=hull_mask,
            confidence=round(confidence, 3)
        )

    def filter_to_court(
        self,
        detections: sv.Detections,
        boundary: CourtBoundary
    ) -> sv.Detections:
        """
        Filters detections to only those whose foot position (bottom-center
        of bbox) falls inside the court boundary mask.

        This removes crowd members, courtside staff, and anyone in the stands
        without needing any ML — pure geometry.
        """
        if len(detections) == 0:
            return detections

        keep = []
        h, w = boundary.mask.shape[:2]

        for i in range(len(detections)):
            box = detections.xyxy[i]

            # Foot position — bottom center of bbox
            foot_x = int((box[0] + box[2]) / 2)
            foot_y = int(box[3])

            # Clamp to frame bounds
            foot_x = max(0, min(foot_x, w - 1))
            foot_y = max(0, min(foot_y, h - 1))

            if boundary.mask[foot_y, foot_x] > 0:
                keep.append(i)

        if not keep:
            return sv.Detections.empty()

        return detections[np.array(keep)]

    def draw_debug(self, frame: np.ndarray, boundary: CourtBoundary) -> np.ndarray:
        """
        Draws the detected court boundary on a frame for visual debugging.
        """
        annotated = frame.copy()

        if boundary is not None and boundary.polygon is not None:
            pts = boundary.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

            # Semi-transparent court overlay
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 255))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            cv2.putText(
                annotated,
                f"Court: {boundary.confidence:.0%}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        return annotated