import cv2
import yaml
import os
from pathlib import Path
import supervision as sv

from features.detector import VisionDetector
from features.tracker import ObjectTracker
from analytics.possession import PossessionEngine
from analytics.actions import ActionClassifier
from core.state_manager import StateManager

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def select_video(raw_dir):
    videos = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        print(f"❌ No videos found in {raw_dir}")
        return None
    
    print("\n--- Available Videos ---")
    for i, v in enumerate(videos):
        print(f"[{i}] {v}")
    
    choice = int(input("\nSelect video index to process: "))
    return videos[choice]

def run_bball_pipeline():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    config = load_config(PROJECT_ROOT / "config.yaml")
    
    # 1. Dynamic Selection
    video_name = select_video(PROJECT_ROOT / "data" / "raw")
    if not video_name: return

    input_path = str(PROJECT_ROOT / "data" / "raw" / video_name)
    output_path = str(PROJECT_ROOT / "data" / "processed" / f"processed_{video_name}")

    # 2. Initialize with YAML settings
    detector = VisionDetector(model_name=config['model']['name']) 
    tracker = ObjectTracker()
    state_manager = StateManager()
    possession_engine = PossessionEngine(possession_threshold=config['possession']['threshold'])
    action_classifier = ActionClassifier(window_size=config['actions']['window_size'])
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def process_frame(frame, _):
        detections = detector.detect(frame)
        ball_detections = detections[detections.class_id == 32]
        player_detections = detections[detections.class_id == 0]
        tracked_players = tracker.update(player_detections)
        
        raw_possessor_id = possession_engine.get_possessor(tracked_players, ball_detections)
        true_possessor_id = state_manager.update_possession(raw_possessor_id)
        current_action = action_classifier.classify(ball_detections, true_possessor_id)

        # Adaptive UI
        h, w, _ = frame.shape
        font_scale = w / config['ui']['font_scale_base']
        thickness = max(1, int(w / 400))

        labels = [f"#{tid} {'BALL' if tid == true_possessor_id else 'Player'}" 
                  for tid in tracked_players.tracker_id]

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_players)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_players, labels=labels)
        
        cv2.putText(annotated_frame, f"Possession: {true_possessor_id} | {current_action}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, config['ui']['hud_color'], thickness)
        
        return annotated_frame

    print(f"🚀 Processing: {video_name} -> processed_{video_name}")
    sv.process_video(source_path=input_path, target_path=output_path, callback=process_frame)

if __name__ == "__main__":
    run_bball_pipeline()