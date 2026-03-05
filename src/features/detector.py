import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from pathlib import Path  # Added missing import

class VisionDetector:
    def __init__(self, model_name: str = "yolov8m.pt"):
        """
        Initializes the YOLO model and annotators using a model name
        from the config.yaml.
        """
        # Dynamically locate the project root
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Priority 1: Check the /models/ folder
        model_path = project_root / "models" / model_name
        
        # Priority 2: Fallback to root (where they were originally)
        if not model_path.exists():
            model_path = project_root / model_name
            
        # Load the YOLO model
        self.model = YOLO(str(model_path))
        
        # 0 = 'person', 32 = 'sports ball'
        self.target_classes = [0, 32]
        
        # Supervision tools initialized once to prevent memory leaks
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Runs inference on a single frame and returns filtered detections.
        """
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for only players and the ball
        mask = np.isin(detections.class_id, self.target_classes)
        return detections[mask]

    def draw_debug(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Standard drawing method for basic visual confirmation.
        """
        labels = [
            f"{self.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame