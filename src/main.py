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
        print(f" No videos found in {raw_dir}")
        return None

    print("\n--- Available Videos ---")
    for i, v in enumerate(videos):
        print(f"[{i}] {v}")

    choice = int(input("\nSelect video index to process: "))
    return videos[choice]


def run_bball_pipeline():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    config = load_config(PROJECT_ROOT / "config.yaml")

    video_name = select_video(PROJECT_ROOT / "data" / "raw")
    if not video_name:
        return

    input_path = str(PROJECT_ROOT / "data" / "raw" / video_name)
    output_path = str(PROJECT_ROOT / "data" / "processed" / f"processed_{video_name}")

    # Fix: constructor uses model_name, not model_path
    detector = VisionDetector(model_name=config['model']['name'])
    tracker = ObjectTracker()
    state_manager = StateManager()
    possession_engine = PossessionEngine(possession_threshold=config['possession']['threshold'])
    action_classifier = ActionClassifier(window_size=config['actions']['window_size'])

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    hud_color = tuple(config['ui']['hud_color'])

    def process_frame(frame, _):
        detections = detector.detect(frame)
        ball_detections = detections[detections.class_id == 32]
        player_detections = detections[detections.class_id == 0]
        tracked_players = tracker.update(player_detections)

        raw_possessor_id = possession_engine.get_possessor(tracked_players, ball_detections)
        true_possessor_id = state_manager.update_possession(raw_possessor_id)
        current_action = action_classifier.classify(ball_detections, true_possessor_id)

        # Log discrete events and per-frame dribble time into state
        if current_action in ("pass", "shot"):
            # The action happened — attribute it to the previous possessor
            state_manager.log_action(action_classifier.prev_possessor, current_action)
        elif current_action == "dribbling":
            state_manager.log_action(true_possessor_id, "dribbling")

        # Adaptive UI scaling
        h, w, _ = frame.shape
        font_scale = w / config['ui']['font_scale_base']
        thickness = max(1, int(w / 400))

        # Guard: tracker_id can be None in early frames
        if tracked_players.tracker_id is not None:
            labels = [
                f"#{tid} {'BALL' if tid == true_possessor_id else 'Player'}"
                for tid in tracked_players.tracker_id
            ]
        else:
            labels = ["Player"] * len(tracked_players)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_players)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=tracked_players, labels=labels
        )

        # Default action display so "none" doesn't show raw on screen
        action_display = current_action if current_action != "none" else "—"
        hud_text = f"Possession: {true_possessor_id if true_possessor_id else 'Loose'} | {action_display}"

        cv2.putText(
            annotated_frame, hud_text,
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, hud_color, thickness
        )

        return annotated_frame

    print(f" Processing: {video_name} -> processed_{video_name}")
    sv.process_video(source_path=input_path, target_path=output_path, callback=process_frame)

    # Flush the last possessor's time — without this their final stint is never logged
    state_manager.finalize()

    print("\n--- Session Summary ---")
    for player_id, stats in state_manager.get_summary().items():
        print(
            f"Player #{player_id}: "
            f"hold={stats['hold_time']:.1f}s  "
            f"dribble={stats['dribble_time']:.1f}s  "
            f"passes={stats['passes']}  "
            f"shots={stats['shots']}"
        )


if __name__ == "__main__":
    run_bball_pipeline()