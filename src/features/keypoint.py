"""
Optional pose estimation via YOLOv8-pose.

Provides hand/wrist positions for more accurate possession detection.
Disabled by default — enabled when ``model.pose_model`` is set in config.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import supervision as sv

from utils.logger import get_logger

log = get_logger(__name__)

# COCO keypoint indices for wrists
LEFT_WRIST = 9
RIGHT_WRIST = 10


class KeypointDetector:
    """
    Extracts hand/wrist positions from player detections using
    a YOLOv8-pose model.

    Usage:
        kd = KeypointDetector("yolov8n-pose.pt")
        hands = kd.get_hand_positions(frame, player_detections)
        # hands = {tracker_id: (x, y), ...}
    """

    def __init__(self, model_name: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO

        project_root = Path(__file__).resolve().parent.parent.parent
        local = project_root / "models" / model_name
        if local.exists():
            self._model = YOLO(str(local))
        else:
            self._model = YOLO(model_name)

        log.info("Pose model loaded: %s", model_name)

    def get_hand_positions(
        self,
        frame: np.ndarray,
        detections: Optional[sv.Detections] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Run pose estimation and return the closest hand position (left or right wrist)
        for each tracked player.

        Args:
            frame:      BGR image
            detections: Player detections with tracker_id assigned.

        Returns:
            Dict mapping tracker_id → (x, y) of the hand nearest to ball-height.
        """
        results = self._model(frame, verbose=False, imgsz=640)[0]

        if results.keypoints is None or results.keypoints.xy is None:
            return {}

        kps = results.keypoints.xy.cpu().numpy()  # (N, 17, 2)

        # If detections with tracker IDs are provided, match by IoU
        if detections is not None and detections.tracker_id is not None:
            return self._match_to_tracks(kps, results.boxes.xyxy.cpu().numpy(), detections)

        # Otherwise return by detection index
        hands = {}
        for i in range(len(kps)):
            left = kps[i, LEFT_WRIST]
            right = kps[i, RIGHT_WRIST]
            # Pick the lower wrist (more likely to be near ball during dribble)
            if left[1] > right[1] and left[0] > 0:
                hands[i] = left
            elif right[0] > 0:
                hands[i] = right
            elif left[0] > 0:
                hands[i] = left

        return hands

    def _match_to_tracks(
        self,
        keypoints: np.ndarray,
        pose_boxes: np.ndarray,
        detections: sv.Detections,
    ) -> Dict[int, np.ndarray]:
        """Match pose detections to tracked players by IoU."""
        from utils.geometry import bbox_iou

        hands = {}
        for pi in range(len(keypoints)):
            # Find best IoU match among tracked detections
            best_iou = 0.3  # minimum threshold
            best_tid = None
            for di in range(len(detections)):
                iou = bbox_iou(pose_boxes[pi], detections.xyxy[di])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = detections.tracker_id[di]

            if best_tid is not None:
                left = keypoints[pi, LEFT_WRIST]
                right = keypoints[pi, RIGHT_WRIST]
                if left[1] > right[1] and left[0] > 0:
                    hands[best_tid] = left
                elif right[0] > 0:
                    hands[best_tid] = right
                elif left[0] > 0:
                    hands[best_tid] = left

        return hands
