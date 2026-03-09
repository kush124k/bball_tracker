import cv2
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from typing import Optional
import supervision as sv


@dataclass
class JerseyProfile:
    team_a_color: np.ndarray     # Mean HSV color of team A
    team_b_color: np.ndarray     # Mean HSV color of team B
    team_a_ids: list             # Tracker IDs assigned to team A
    team_b_ids: list             # Tracker IDs assigned to team B
    outlier_ids: list            # IDs that don't match either team (refs, coaches)


class JerseyClassifier:
    """
    Separates players into two teams and identifies non-players (refs,
    coaches) using KMeans clustering on jersey colors.

    Runs on the upper-body crop of each detection bbox to avoid picking
    up court color from legs/shoes.

    Usage:
        - Call build_profile() on the first N frames to establish team colors
        - Call classify(detections) per-frame to assign team labels
        - Call filter_non_players(detections) to remove refs/coaches
    """

    # How far a detection's color can be from a team cluster before
    # being flagged as an outlier (ref, coach, etc.)
    # In HSV Euclidean distance — tune if refs share team colors
    OUTLIER_DISTANCE_THRESHOLD = 40.0

    # Fraction of bbox height to use for jersey crop (top portion)
    JERSEY_CROP_TOP = 0.15
    JERSEY_CROP_BOTTOM = 0.55

    def __init__(self):
        self.team_a_color: Optional[np.ndarray] = None
        self.team_b_color: Optional[np.ndarray] = None
        self._is_calibrated = False
        self._id_to_team: dict = {}

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def build_profile(self, frame: np.ndarray, detections: sv.Detections) -> bool:
        """
        Extracts jersey colors from all current detections and runs KMeans
        to find two team clusters. Call this on an early frame where most
        players are visible.

        Returns True if calibration succeeded (found 2 clear clusters).
        """
        if len(detections) < 4:
            return False  # Need enough players to find two clusters reliably

        crops = self._extract_jersey_crops(frame, detections)
        if len(crops) < 4:
            return False

        dominant_colors = np.array([self._dominant_color(crop) for crop in crops])

        # KMeans with k=2 — find two team jersey colors
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(dominant_colors)

        self.team_a_color = kmeans.cluster_centers_[0]
        self.team_b_color = kmeans.cluster_centers_[1]
        self._is_calibrated = True

        return True

    def classify(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> dict:
        """
        Assigns each detection to team_a, team_b, or "outlier".
        Returns a dict mapping detection index -> "team_a" | "team_b" | "outlier"
        """
        if not self._is_calibrated or len(detections) == 0:
            return {}

        crops = self._extract_jersey_crops(frame, detections)
        assignments = {}

        for i, crop in enumerate(crops):
            color = self._dominant_color(crop)
            dist_a = np.linalg.norm(color - self.team_a_color)
            dist_b = np.linalg.norm(color - self.team_b_color)

            min_dist = min(dist_a, dist_b)
            if min_dist > self.OUTLIER_DISTANCE_THRESHOLD:
                assignments[i] = "outlier"
            elif dist_a < dist_b:
                assignments[i] = "team_a"
            else:
                assignments[i] = "team_b"

        return assignments

    def filter_non_players(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> sv.Detections:
        """
        Removes detections classified as outliers (refs, coaches).
        Returns filtered detections containing only team_a and team_b players.
        """
        if not self._is_calibrated or len(detections) == 0:
            return detections

        assignments = self.classify(frame, detections)
        keep = [i for i, label in assignments.items() if label != "outlier"]

        if not keep:
            return detections  # Fallback — don't wipe everything if calibration is off

        return detections[np.array(keep)]

    def get_team_label(self, frame: np.ndarray, detections: sv.Detections, index: int) -> str:
        """
        Returns the team label for a single detection by index.
        """
        assignments = self.classify(frame, detections)
        return assignments.get(index, "unknown")

    def _extract_jersey_crops(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> list:
        """
        Crops the upper-body region of each detection for jersey color sampling.
        Skips crops that are too small to be reliable.
        """
        crops = []
        h_frame, w_frame = frame.shape[:2]

        for i in range(len(detections)):
            box = detections.xyxy[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            box_h = y2 - y1
            crop_y1 = y1 + int(box_h * self.JERSEY_CROP_TOP)
            crop_y2 = y1 + int(box_h * self.JERSEY_CROP_BOTTOM)

            # Clamp to frame
            crop_y1 = max(0, min(crop_y1, h_frame - 1))
            crop_y2 = max(0, min(crop_y2, h_frame - 1))
            x1 = max(0, min(x1, w_frame - 1))
            x2 = max(0, min(x2, w_frame - 1))

            if crop_y2 <= crop_y1 or x2 <= x1:
                crops.append(np.zeros((1, 1, 3), dtype=np.uint8))
                continue

            crops.append(frame[crop_y1:crop_y2, x1:x2])

        return crops

    def _dominant_color(self, crop: np.ndarray) -> np.ndarray:
        """
        Returns the dominant HSV color of a crop using KMeans with k=1.
        More robust than a simple mean — ignores background pixels at edges.
        """
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(3)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Filter near-white (court lines) and near-black (shadows) pixels
        sat_mask = (pixels[:, 1] > 30) & (pixels[:, 2] > 40)
        filtered = pixels[sat_mask]

        if len(filtered) < 10:
            return np.mean(pixels, axis=0)

        kmeans = KMeans(n_clusters=1, n_init=3, random_state=0)
        kmeans.fit(filtered)
        return kmeans.cluster_centers_[0]