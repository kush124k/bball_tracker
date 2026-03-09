import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal

AngleType = Literal["sideline", "broadcast", "overhead", "unknown"]


@dataclass
class ViewProfile:
    angle: AngleType
    confidence: float
    notes: str


class ViewClassifier:
    """
    Classifies camera angle from a video by sampling frames and
    analyzing geometric properties of player bounding boxes.

    Heuristics used:
    - Sideline:   Players are tall/narrow, distributed horizontally,
                  relatively large in frame
    - Broadcast:  Players are smaller, wider court visible, distributed
                  across most of frame width
    - Overhead:   Player bboxes are roughly square, distributed across
                  full 2D space, very small relative to frame

    No ML model required — runs purely on YOLO detections.
    """

    # Aspect ratio = height / width of person bbox
    SIDELINE_ASPECT_MIN = 1.8   # Players appear tall and narrow
    BROADCAST_ASPECT_MIN = 1.2  # Players smaller, more compressed
    OVERHEAD_ASPECT_MAX = 1.3   # Players appear nearly square from above

    # Player bbox height as % of frame height
    SIDELINE_HEIGHT_MIN = 0.25  # Players take up significant frame height
    BROADCAST_HEIGHT_MIN = 0.10
    OVERHEAD_HEIGHT_MAX = 0.15  # Players tiny from above

    def classify_video(self, video_path: str, sample_count: int = 12) -> ViewProfile:
        """
        Samples frames evenly across the video and returns the most
        likely camera angle with a confidence score.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        all_aspect_ratios = []
        all_relative_heights = []
        all_x_positions = []  # Normalized x-center of each detection

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            boxes = self._detect_players(frame)
            for box in boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                aspect = h / w
                rel_height = h / frame_height
                x_center = ((x1 + x2) / 2) / frame_width

                all_aspect_ratios.append(aspect)
                all_relative_heights.append(rel_height)
                all_x_positions.append(x_center)

        cap.release()

        if len(all_aspect_ratios) < 3:
            return ViewProfile(
                angle="unknown",
                confidence=0.0,
                notes="Not enough player detections to classify angle"
            )

        return self._classify_from_stats(
            np.median(all_aspect_ratios),
            np.median(all_relative_heights),
            np.std(all_x_positions)
        )

    def _classify_from_stats(
        self,
        median_aspect: float,
        median_rel_height: float,
        x_spread: float
    ) -> ViewProfile:
        """
        Applies heuristic rules to stats derived from sampled frames.
        x_spread: std dev of player x-positions (high = players spread wide = court visible)
        """
        scores = {"sideline": 0, "broadcast": 0, "overhead": 0}

        # --- Aspect ratio scoring ---
        if median_aspect >= self.SIDELINE_ASPECT_MIN:
            scores["sideline"] += 2
        elif median_aspect >= self.BROADCAST_ASPECT_MIN:
            scores["broadcast"] += 2
        elif median_aspect <= self.OVERHEAD_ASPECT_MAX:
            scores["overhead"] += 2

        # --- Relative height scoring ---
        if median_rel_height >= self.SIDELINE_HEIGHT_MIN:
            scores["sideline"] += 2
        elif median_rel_height >= self.BROADCAST_HEIGHT_MIN:
            scores["broadcast"] += 2
        else:
            scores["overhead"] += 2

        # --- Horizontal spread scoring ---
        # Broadcast/overhead show more of the court so players spread wider
        if x_spread > 0.30:
            scores["broadcast"] += 1
            scores["overhead"] += 1
        else:
            scores["sideline"] += 1

        best_angle = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_angle] / total if total > 0 else 0.0

        notes = (
            f"median_aspect={median_aspect:.2f}, "
            f"median_rel_height={median_rel_height:.2f}, "
            f"x_spread={x_spread:.2f}"
        )

        return ViewProfile(angle=best_angle, confidence=round(confidence, 2), notes=notes)

    def _detect_players(self, frame: np.ndarray) -> list:
        """
        Lightweight person detection using a background subtractor
        and contour analysis — avoids loading YOLO twice.
        Falls back to HOG for reliability.
        """
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Resize for speed — HOG doesn't need full res for classification
        scale = 640 / max(frame.shape[:2])
        if scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            small = frame
            scale = 1.0

        boxes, _ = hog.detectMultiScale(
            small,
            winStride=(16, 16),
            padding=(8, 8),
            scale=1.05
        )

        if len(boxes) == 0:
            return []

        # Scale boxes back to original resolution
        result = []
        for (x, y, w, h) in boxes:
            result.append([
                x / scale, y / scale,
                (x + w) / scale, (y + h) / scale
            ])

        return result