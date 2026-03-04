import cv2
import numpy as np
import supervision as sv
import os
from pathlib import Path

from features.detector import VisionDetector
from features.tracker import ObjectTracker
from analytics.possession import PossessionEngine
from core.state_manager import StateManager  # <-- Added the import

def run_test_pipeline(input_video_path: str, output_video_path: str):
    print(f"Loading models and processing {input_video_path}...")
    
    # 1. Initialize our modules
    detector = VisionDetector(model_path="yolov8m.pt") 
    tracker = ObjectTracker()
    possession_engine = PossessionEngine(possession_threshold=150) 
    
    # Initialize the memory module outside the loop so it persists!
    state_manager = StateManager() 
    
    # Annotators for drawing
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 2. Define what happens to each individual frame
    def process_frame(frame: np.ndarray, _: int) -> np.ndarray:
        # Detect everything
        detections = detector.detect(frame)
        
        # Separate the ball (Class 32)
        ball_mask = detections.class_id == 32
        ball_detections = detections[ball_mask]
        
        # Separate the players (Class 0) and track them
        player_mask = detections.class_id == 0
        player_detections = detections[player_mask]
        tracked_players = tracker.update(player_detections)
        
        # --- THE MATH & MEMORY LAYER (Must happen before drawing) ---
        # 1. Get raw visual distance
        raw_possessor_id = possession_engine.get_possessor(players=tracked_players, ball=ball_detections)
        
        # 2. Filter it through the Rule Book memory
        true_possessor_id = state_manager.update_possession(raw_possessor_id)
        # -------------------------------------------------------------
        
        # --- THE DRAWING LAYER ---
        # Create labels for players using the TRUE possessor
        labels = []
        for class_id, tracker_id, confidence in zip(
            tracked_players.class_id, 
            tracked_players.tracker_id, 
            tracked_players.confidence
        ):
            if tracker_id == true_possessor_id:
                labels.append(f"⭐ #{tracker_id} HAS BALL")
            else:
                labels.append(f"#{tracker_id} {detector.model.names[class_id]}")
        
        # Draw player boxes and labels
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_players)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_players, labels=labels)
        
        # Draw a simple box around the ball
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=ball_detections)
        
        # Draw the big HUD overlay text using the TRUE possessor
        hud_text = f"Possession: Player {true_possessor_id}" if true_possessor_id else "Possession: Loose Ball"
        text_color = (0, 255, 0) if true_possessor_id else (0, 0, 255) 
        
        cv2.putText(
            annotated_frame, 
            hud_text, 
            (40, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5,      
            text_color, 
            4         
        )
        
        return annotated_frame

    # 3. Run the video processor
    sv.process_video(
        source_path=input_video_path,
        target_path=output_video_path,
        callback=process_frame
    )
    print(f" Done! Annotated video saved to {output_video_path}")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_video = str(PROJECT_ROOT / "data" / "raw" / "sample.mp4")
    output_video = str(PROJECT_ROOT / "data" / "processed" / "output.mp4")
    
    if not os.path.exists(input_video):
        print(f" Error: Could not find {input_video}")
    else:
        run_test_pipeline(input_video, output_video)