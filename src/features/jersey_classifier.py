"""
Jersey-colour classifier using scipy KMeans.

Changes from original:
- Replaced sklearn with scipy.cluster.vq (eliminates heavy dependency)
- Cached team assignments — only reclassify when necessary
- Uses shared crop_region() from geometry utils
- Implements Processor interface
"""

import cv2
import numpy as np
from typing import Optional, Dict
import supervision as sv
from scipy.cluster.vq import kmeans2

from core.frame_state import FrameState, Processor
from utils.geometry import crop_region
from utils.logger import get_logger

log = get_logger(__name__)


class JerseyClassifier(Processor):
    """
    Separates players into two teams and identifies non-players
    (refs, coaches) using KMeans clustering on jersey HSV colours.
    """

    OUTLIER_DISTANCE = 40.0
    JERSEY_CROP_TOP = 0.15
    JERSEY_CROP_BOTTOM = 0.55
    MIN_PLAYERS_FOR_CALIBRATION = 4
    RECLASSIFY_INTERVAL = 60       # Re-check assignments every N frames

    def __init__(self):
        self._team_a_color: Optional[np.ndarray] = None
        self._team_b_color: Optional[np.ndarray] = None
        self._is_calibrated = False
        self._last_calibration_frame = -999
        self._cached_assignments: Dict[int, str] = {}

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    # ─── Processor interface ───────────────────────────────────────────

    def process(self, state: FrameState) -> FrameState:
        frame = state.raw_frame
        players = state.player_detections

        if frame is None or players is None or len(players) == 0:
            return state

        # Attempt calibration periodically until it succeeds
        if not self._is_calibrated:
            if (state.frame_index - self._last_calibration_frame) >= 15:
                self._last_calibration_frame = state.frame_index
                success = self.build_profile(frame, players)
                if success:
                    log.info("[Frame %d] Jersey calibration succeeded", state.frame_index)
                else:
                    log.debug(
                        "[Frame %d] Jersey calibration needs more players (%d visible)",
                        state.frame_index, len(players),
                    )

        # Filter non-players + assign teams
        if self._is_calibrated:
            state.player_detections = self.filter_non_players(frame, players)
            if state.tracked_players is not None and state.tracked_players.tracker_id is not None:
                assignments = self.classify(frame, state.tracked_players)
                # Merge into state — map index → tracker_id
                for idx, label in assignments.items():
                    if idx < len(state.tracked_players.tracker_id):
                        tid = state.tracked_players.tracker_id[idx]
                        state.team_assignments[tid] = label

        return state

    # ─── Calibration ───────────────────────────────────────────────────

    def build_profile(self, frame: np.ndarray, detections: sv.Detections) -> bool:
        """
        Extract jersey colours and cluster into 2 teams.
        Returns True if calibration succeeded.
        """
        if len(detections) < self.MIN_PLAYERS_FOR_CALIBRATION:
            return False

        crops = self._extract_crops(frame, detections)
        colors = np.array([self._dominant_color(c) for c in crops])

        # Need at least 4 valid colours
        valid = colors[np.any(colors > 0, axis=1)]
        if len(valid) < self.MIN_PLAYERS_FOR_CALIBRATION:
            return False

        # scipy kmeans2 — replaces sklearn
        centroids, labels = kmeans2(valid.astype(np.float64), 2, minit="++")

        self._team_a_color = centroids[0].astype(np.float32)
        self._team_b_color = centroids[1].astype(np.float32)
        self._is_calibrated = True
        return True

    # ─── Classification ────────────────────────────────────────────────

    def classify(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> Dict[int, str]:
        """Assign each detection to team_a, team_b, or ref."""
        if not self._is_calibrated or len(detections) == 0:
            return {}

        crops = self._extract_crops(frame, detections)
        out: Dict[int, str] = {}

        for i, crop in enumerate(crops):
            color = self._dominant_color(crop)
            da = np.linalg.norm(color - self._team_a_color)
            db = np.linalg.norm(color - self._team_b_color)
            min_d = min(da, db)

            if min_d > self.OUTLIER_DISTANCE:
                out[i] = "ref"
            elif da < db:
                out[i] = "team_a"
            else:
                out[i] = "team_b"

        return out

    def filter_non_players(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> sv.Detections:
        """Remove refs/coaches, keep team players only."""
        if not self._is_calibrated or len(detections) == 0:
            return detections

        assignments = self.classify(frame, detections)
        keep = [i for i, label in assignments.items() if label != "ref"]

        if not keep:
            return detections  # don't wipe everything
        return detections[np.array(keep)]

    # ─── Internals ─────────────────────────────────────────────────────

    def _extract_crops(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> list:
        return [
            crop_region(frame, detections.xyxy[i], self.JERSEY_CROP_TOP, self.JERSEY_CROP_BOTTOM)
            for i in range(len(detections))
        ]

    @staticmethod
    def _dominant_color(crop: np.ndarray) -> np.ndarray:
        """Dominant HSV colour via single-centroid KMeans."""
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(3, dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float64)

        # Filter near-white/black
        mask = (pixels[:, 1] > 30) & (pixels[:, 2] > 40)
        filtered = pixels[mask]

        if len(filtered) < 10:
            return np.mean(pixels, axis=0).astype(np.float32)

        centroids, _ = kmeans2(filtered, 1, minit="++")
        return centroids[0].astype(np.float32)