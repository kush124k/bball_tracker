import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv


class VisionDetector:
    def __init__(self, profile: dict):
        """
        Accepts an angle-specific profile dict loaded from angle_profiles.yaml.
        Example profile:
            model_name: "yolov8n.pt"
            inference_size: 640
            person_confidence: 0.40
            ball_confidence: 0.15
        """
        model_name = profile['model_name']
        self.inference_size = profile.get('inference_size', 640)
        self.person_confidence = profile.get('person_confidence', 0.40)
        self.ball_confidence = profile.get('ball_confidence', 0.20)

        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / "models" / model_name
        self.model = YOLO(str(model_path))

        # 0 = person, 32 = sports ball
        self.person_class = 0
        self.ball_class = 32

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Runs inference and returns detections with per-class confidence filtering.
        Ball uses a lower threshold than persons since it's harder to detect.
        """
        results = self.model(
            frame,
            verbose=False,
            imgsz=self.inference_size,
            # Use the lower of the two thresholds so both classes pass through
            # We then filter per-class below
            conf=min(self.person_confidence, self.ball_confidence)
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Only keep the classes we care about
        class_mask = np.isin(detections.class_id, [self.person_class, self.ball_class])
        detections = detections[class_mask]

        if len(detections) == 0:
            return detections

        # Apply per-class confidence thresholds
        person_mask = (detections.class_id == self.person_class) & \
                      (detections.confidence >= self.person_confidence)
        ball_mask = (detections.class_id == self.ball_class) & \
                    (detections.confidence >= self.ball_confidence)

        return detections[person_mask | ball_mask]

    def draw_debug(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        labels = [
            f"{self.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        annotated = frame.copy()
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        return annotated