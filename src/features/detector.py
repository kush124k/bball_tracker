import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

class VisionDetector:
    def __init__(self, model_path: str = "yolov8m.pt"):
        """
        Initializes the YOLO model and the annotators.
        Using yolov8n.pt (nano) by default for fast testing.
        """
        
        project_root = Path(__file__).resolve().parent.parent.parent
        model_path = project_root / "models" / model_name
        # Load the YOLO model (it will auto-download the weights the first time)
        self.model = YOLO(str(model_path))
        
        # In the standard COCO dataset:
        # 0 = 'person'
        # 32 = 'sports ball'
        self.target_classes = [0, 32]
        
        # Supervision tools for easy debugging overlays
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Runs inference on a single frame and returns filtered detections.
        """
        # 1. Run YOLO inference (verbose=False keeps your terminal clean)
        results = self.model(frame, verbose=False)[0]
        
        # 2. Convert raw YOLO output into a Supervision Detections object
        detections = sv.Detections.from_ultralytics(results)
        
        # 3. Filter out the noise (we only want players and the ball)
        mask = np.isin(detections.class_id, self.target_classes)
        filtered_detections = detections[mask]
        
        return filtered_detections

    def draw_debug(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Draws bounding boxes and labels on the frame for visual confirmation.
        """
        # Extract the class names ('person', 'sports ball') for the labels
        labels = [
            f"{self.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        # Draw the boxes and labels onto a copy of the frame
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame