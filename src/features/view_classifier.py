"""
Camera-angle classifier using YOLO detections.

Replaces the old HOG-based approach with the already-loaded YOLO model,
eliminating duplicate model loading and improving accuracy.

Supports all three angles: sideline, broadcast, overhead.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)

AngleType = Literal["sideline", "broadcast", "overhead", "unknown"]


@dataclass
class ViewProfile:
    angle: AngleType
    confidence: float
    notes: str


class ViewClassifier:
    """
    Classifies camera angle by sampling frames and analysing
    geometric properties of player bounding boxes.

    Heuristics:
    - Sideline:   tall/narrow players, large in frame, narrow horizontal spread
    - Broadcast:  smaller players, wider court, moderate spread
    - Overhead:   square-ish players, tiny, full 2D spread
    """

    # Aspect ratio thresholds (height / width)
    SIDELINE_ASPECT_MIN = 1.8
    BROADCAST_ASPECT_MIN = 1.2
    OVERHEAD_ASPECT_MAX = 1.3

    # Player bbox height as fraction of frame height
    SIDELINE_HEIGHT_MIN = 0.25
    BROADCAST_HEIGHT_MIN = 0.10
    OVERHEAD_HEIGHT_MAX = 0.15

    def __init__(self, yolo_model=None):
        """
        Args:
            yolo_model: An already-loaded YOLO model instance.
                        If None, a lightweight one is loaded on demand.
        """
        self._model = yolo_model

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            project_root = Path(__file__).resolve().parent.parent.parent
            local = project_root / "models" / "yolov8n.pt"
            self._model = YOLO(str(local) if local.exists() else "yolov8n.pt")

    def classify_video(
        self, video_path: str, sample_count: int = 12
    ) -> ViewProfile:
        """
        Sample frames across the video and classify the camera angle.
        """
        self._ensure_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        indices = np.linspace(0, max(total - 1, 0), sample_count, dtype=int)

        aspects, heights, x_positions = [], [], []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
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
                aspects.append(h / w)
                heights.append(h / fh)
                x_positions.append(((x1 + x2) / 2) / fw)

        cap.release()

        if len(aspects) < 3:
            log.warning("Only %d player detections — insufficient for angle classification", len(aspects))
            return ViewProfile("unknown", 0.0, "Not enough detections")

        return self._classify_from_stats(
            np.median(aspects), np.median(heights), np.std(x_positions)
        )

    def _detect_players(self, frame: np.ndarray) -> list:
        """Use YOLO to detect persons (class 0) in a frame."""
        results = self._model(frame, verbose=False, imgsz=640, conf=0.30)[0]
        boxes = []
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) == 0:  # person
                boxes.append(box.tolist())
        return boxes

    def _classify_from_stats(
        self,
        median_aspect: float,
        median_height: float,
        x_spread: float,
    ) -> ViewProfile:
        scores = {"sideline": 0, "broadcast": 0, "overhead": 0}

        # Aspect ratio
        if median_aspect >= self.SIDELINE_ASPECT_MIN:
            scores["sideline"] += 2
        elif median_aspect >= self.BROADCAST_ASPECT_MIN:
            scores["broadcast"] += 2
        elif median_aspect <= self.OVERHEAD_ASPECT_MAX:
            scores["overhead"] += 2

        # Relative height
        if median_height >= self.SIDELINE_HEIGHT_MIN:
            scores["sideline"] += 2
        elif median_height >= self.BROADCAST_HEIGHT_MIN:
            scores["broadcast"] += 2
        else:
            scores["overhead"] += 2

        # Horizontal spread
        if x_spread > 0.30:
            scores["broadcast"] += 1
            scores["overhead"] += 1
        else:
            scores["sideline"] += 1

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        conf = scores[best] / total if total > 0 else 0.0

        notes = (
            f"median_aspect={median_aspect:.2f}, "
            f"median_rel_height={median_height:.2f}, "
            f"x_spread={x_spread:.2f}"
        )

        log.info("View classification: %s (%.0f%%) — %s", best, conf * 100, notes)
        return ViewProfile(angle=best, confidence=round(conf, 2), notes=notes)