import cv2
import yaml
import os
import numpy as np
from pathlib import Path
import supervision as sv

from features.detector import VisionDetector
from features.tracker import ObjectTracker
from features.view_classifier import ViewClassifier
from features.court_detector import CourtDetector
from features.jersey_classifier import JerseyClassifier
from analytics.possession import PossessionEngine
from analytics.actions import ActionClassifier
from core.state_manager import StateManager


def load_config(path):
    with open(path, 'r') as f:
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


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def run_bball_pipeline():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    base_config = load_config(PROJECT_ROOT / "config.yaml")
    angle_profiles = load_config(PROJECT_ROOT / "angle_profiles.yaml")

    SUPPORTED_ANGLES = {"sideline"}
    CALIBRATION_ATTEMPT_INTERVAL = 15   # Retry jersey calibration every N frames
    PROGRESS_INTERVAL = 30              # Print progress every N frames

    video_name = select_video(PROJECT_ROOT / "data" / "raw")
    if not video_name:
        return

    input_path = str(PROJECT_ROOT / "data" / "raw" / video_name)
    output_path = str(PROJECT_ROOT / "data" / "processed" / f"processed_{video_name}")

    # --- Step 1: Classify camera angle ---
    print(f"\n Analysing camera angle for {video_name}...")
    view_classifier = ViewClassifier()
    view = view_classifier.classify_video(input_path)
    print(f" Detected angle : {view.angle} (confidence: {view.confidence:.0%})")
    print(f" Details        : {view.notes}")

    if view.angle not in SUPPORTED_ANGLES:
        print(f"\n Angle '{view.angle}' not supported. Defaulting to sideline.")
        view.angle = "sideline"

    profile = angle_profiles[view.angle]

    # --- Step 2: Initialize modules ---
    detector = VisionDetector(profile=profile['detector'])
    tracker = ObjectTracker(profile=profile['tracker'])
    state_manager = StateManager()
    possession_engine = PossessionEngine(
        possession_threshold=profile['possession']['threshold']
    )
    action_classifier = ActionClassifier(
        window_size=profile['actions']['window_size'],
        dribble_variance=profile['actions']['dribble_variance']
    )
    court_detector = CourtDetector()
    jersey_classifier = JerseyClassifier()

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    hud_color = tuple(base_config['ui']['hud_color'])
    font_scale_base = base_config['ui']['font_scale_base']

    # --- Step 3: Open video and set up writer ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f" Could not open video: {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n Video info: {width}x{height} @ {fps:.1f}fps — "
          f"{total_frames} frames ({format_time(total_duration)})")
    print(f" Processing with '{view.angle}' profile...\n")

    court_boundary = None
    court_calibrated = False
    jersey_calibrated = False
    frame_index = 0

    # --- Step 4: Manual frame loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # --- Court calibration: once, on first frame ---
        if not court_calibrated:
            court_boundary = court_detector.detect(frame)
            court_calibrated = True
            if court_boundary:
                print(f" [Frame {frame_index:>4}] Court detected "
                      f"(coverage: {court_boundary.confidence:.0%})")
            else:
                print(f" [Frame {frame_index:>4}] Court not detected — "
                      f"court filter disabled")

        # --- Jersey calibration: retry every N frames until it succeeds ---
        if not jersey_calibrated and frame_index % CALIBRATION_ATTEMPT_INTERVAL == 0:
            raw = detector.detect(frame)
            people = raw[raw.class_id == 0]
            if court_boundary:
                people = court_detector.filter_to_court(people, court_boundary)

            success = jersey_classifier.build_profile(frame, people)
            if success:
                jersey_calibrated = True
                print(f" [Frame {frame_index:>4}] Jersey calibration succeeded")
            else:
                print(f" [Frame {frame_index:>4}] Jersey calibration retrying "
                      f"({len(people)} players visible, need 4+)...")

        # --- Detection ---
        detections = detector.detect(frame)
        ball_detections = detections[detections.class_id == 32]
        player_detections = detections[detections.class_id == 0]

        # --- Filter 1: Court boundary ---
        if court_boundary is not None:
            player_detections = court_detector.filter_to_court(player_detections, court_boundary)

        # --- Filter 2: Jersey classifier ---
        if jersey_classifier.is_calibrated:
            player_detections = jersey_classifier.filter_non_players(frame, player_detections)

        # --- Tracking and analytics ---
        tracked_players = tracker.update(player_detections)
        raw_possessor_id = possession_engine.get_possessor(tracked_players, ball_detections)
        true_possessor_id = state_manager.update_possession(raw_possessor_id)
        current_action = action_classifier.classify(ball_detections, true_possessor_id)

        if current_action in ("pass", "shot"):
            state_manager.log_action(action_classifier.prev_possessor, current_action)
        elif current_action == "dribbling":
            state_manager.log_action(true_possessor_id, "dribbling")

        # --- Annotation ---
        h, w, _ = frame.shape
        font_scale = w / font_scale_base
        thickness = max(1, int(w / 400))

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

        if court_boundary is not None:
            annotated_frame = court_detector.draw_debug(annotated_frame, court_boundary)

        action_display = current_action if current_action != "none" else "—"
        hud_text = (
            f"Possession: {true_possessor_id if true_possessor_id else 'Loose'} "
            f"| {action_display}"
        )
        cv2.putText(
            annotated_frame, hud_text,
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, hud_color, thickness
        )

        writer.write(annotated_frame)

        # --- Progress reporting ---
        if frame_index % PROGRESS_INTERVAL == 0 or frame_index == total_frames:
            pct = frame_index / total_frames * 100
            elapsed_video_time = frame_index / fps
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(
                f" [{bar}] {pct:5.1f}%  "
                f"frame {frame_index:>5}/{total_frames}  "
                f"video time {format_time(elapsed_video_time)}/{format_time(total_duration)}  "
                f"possessor={true_possessor_id or 'Loose':<6}  "
                f"action={action_display}",
                end="\r"
            )

    print()  # newline after progress bar
    cap.release()
    writer.release()
    state_manager.finalize()

    print(f"\n Done! Saved to: {output_path}")
    print("\n--- Session Summary ---")
    for player_id, stats in state_manager.get_summary().items():
        print(
            f"  Player #{player_id}: "
            f"hold={stats['hold_time']:.1f}s  "
            f"dribble={stats['dribble_time']:.1f}s  "
            f"passes={stats['passes']}  "
            f"shots={stats['shots']}"
        )


if __name__ == "__main__":
    run_bball_pipeline()