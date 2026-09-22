"""
YOLO-based object detector with frame-stride caching and auto-resize.

Changes from original:
- Accepts typed profile from config (not raw dict)
- Frame stride: skip N frames, return cached detections
- Auto-resize: downscale 4K frames before inference
- Model auto-download: if file not in models/, ultralytics fetches it
- Implements Processor interface for pipeline integration
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

from core.frame_state import FrameState, Processor
from utils.logger import get_logger

log = get_logger(__name__)


class VisionDetector(Processor):
    """
    Runs YOLO inference and splits results into player / ball detections.

    Writes to ``state.all_detections``, ``state.player_detections``,
    ``state.ball_detections``.
    """

    PERSON_CLASS = 0
    BALL_CLASS = 32

    def __init__(
        self,
        profile: dict | None = None,
        frame_stride: int = 1,
        resize_width: int | None = None,
    ):
        p = profile or {}
        model_name = p.get("model_name", "yolov8n.pt")
        self._inference_size = p.get("inference_size", 640)
        self._person_conf = p.get("person_confidence", 0.40)
        self._ball_conf = p.get("ball_confidence", 0.20)
        self._frame_stride = max(1, frame_stride)
        self._resize_width = resize_width

        # Resolve model path — try local models/ dir first, fall back to
        # ultralytics auto-download
        project_root = Path(__file__).resolve().parent.parent.parent
        local_path = project_root / "models" / model_name
        if local_path.exists():
            self._model = YOLO(str(local_path))
            log.info("Loaded model from %s", local_path)
        else:
            log.info("Model not found locally, downloading %s via ultralytics", model_name)
            self._model = YOLO(model_name)

        # Cache for frame-stride skipping
        self._cache: sv.Detections | None = None
        self._cache_players: sv.Detections | None = None
        self._cache_ball: sv.Detections | None = None

    # ─── Processor interface ───────────────────────────────────────────

    def process(self, state: FrameState) -> FrameState:
        frame = state.raw_frame
        if frame is None:
            return state

        # Frame stride — return cached results for skipped frames
        if self._frame_stride > 1 and state.frame_index % self._frame_stride != 0:
            if self._cache is not None:
                state.all_detections = self._cache
                state.player_detections = self._cache_players
                state.ball_detections = self._cache_ball
                return state

        # Optionally resize
        inference_frame = frame
        if self._resize_width and frame.shape[1] > self._resize_width:
            scale = self._resize_width / frame.shape[1]
            import cv2
            inference_frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
            )

        detections = self._detect(inference_frame)

        # If we resized, scale boxes back to original resolution
        if inference_frame is not frame:
            inv_scale = frame.shape[1] / inference_frame.shape[1]
            detections.xyxy = (detections.xyxy * inv_scale).astype(np.float32)

        # Split by class
        players = self._filter_class(detections, self.PERSON_CLASS, self._person_conf)
        ball = self._filter_class(detections, self.BALL_CLASS, self._ball_conf)

        # Cache
        self._cache = detections
        self._cache_players = players
        self._cache_ball = ball

        state.all_detections = detections
        state.player_detections = players
        state.ball_detections = ball
        return state

    # ─── Internals ─────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> sv.Detections:
        """Run inference and return detections for person + ball classes."""
        results = self._model(
            frame,
            verbose=False,
            imgsz=self._inference_size,
            conf=min(self._person_conf, self._ball_conf),
        )[0]

        dets = sv.Detections.from_ultralytics(results)

        # Keep only person and ball
        mask = np.isin(dets.class_id, [self.PERSON_CLASS, self.BALL_CLASS])
        return dets[mask]

    @staticmethod
    def _filter_class(
        dets: sv.Detections, class_id: int, min_conf: float
    ) -> sv.Detections:
        """Filter detections to a single class with a confidence threshold."""
        if len(dets) == 0:
            return sv.Detections.empty()

        mask = (dets.class_id == class_id) & (dets.confidence >= min_conf)
        filtered = dets[mask]
        return filtered if len(filtered) > 0 else sv.Detections.empty()