"""
bball_tracker — Pipeline orchestrator and CLI.

This is the entry point. It:
1. Parses CLI arguments (or runs interactive mode)
2. Classifies the camera angle
3. Builds the processor pipeline
4. Loops over frames, passing FrameState through each processor
5. Runs post-processing (export, summary)

Usage:
    python -m src.main --input path/to/video.mp4
    python -m src.main --interactive
    python -m src.main --input video.mp4 --export-csv --export-json --no-display
"""

import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path

from core.config import load_app_config, AppConfig
from core.frame_state import FrameState, Processor
from core.state_manager import StateManager
from features.detector import VisionDetector
from features.tracker import PlayerTracker
from features.ball_tracker import BallTracker
from features.court_detector import CourtDetector
from features.jersey_classifier import JerseyClassifier
from features.view_classifier import ViewClassifier
from analytics.possession import PossessionEngine
from analytics.actions import ActionClassifier
from analytics.movement import MovementAnalyzer
from analytics.exporter import Exporter
from rendering.hud import HUDRenderer
from utils.logger import setup_logging, get_logger, log_progress

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="bball_tracker",
        description="Basketball video analysis with tracking, possession, and action detection.",
    )
    p.add_argument("--input", "-i", type=str, help="Path to input video file")
    p.add_argument("--output", "-o", type=str, help="Path to output video file (default: auto)")
    p.add_argument("--interactive", action="store_true", help="Interactive video picker mode")
    p.add_argument("--profile", type=str, help="Force camera angle profile (sideline/broadcast/overhead)")
    p.add_argument("--export-csv", action="store_true", help="Export per-player stats CSV")
    p.add_argument("--export-json", action="store_true", help="Export full session JSON")
    p.add_argument("--no-display", action="store_true", help="Disable live preview window")
    p.add_argument("--no-render", action="store_true", help="Skip rendering (faster, no annotated output)")
    p.add_argument("--log-level", type=str, default="INFO", help="Log level (DEBUG/INFO/WARNING/ERROR)")
    p.add_argument("--config", type=str, help="Path to config.yaml override")
    p.add_argument("--frame-stride", type=int, help="Process every Nth frame")
    return p.parse_args()


def select_video_interactive(raw_dir: Path) -> str | None:
    """Interactive video picker from data/raw/."""
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)

    videos = [f for f in os.listdir(raw_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not videos:
        log.error("No videos found in %s", raw_dir)
        return None

    print("\n--- Available Videos ---")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v}")

    try:
        choice = int(input("\nSelect video index: "))
        return str(raw_dir / videos[choice])
    except (ValueError, IndexError):
        log.error("Invalid selection")
        return None


# ── Pipeline builder ───────────────────────────────────────────────────────

def build_pipeline(config: AppConfig, profile: dict) -> list[Processor]:
    """
    Build the ordered list of processors for the pipeline.
    Each processor reads/writes to a shared FrameState.
    """
    detector_profile = profile.get("detector", {})
    tracker_profile = profile.get("tracker", {})
    possession_cfg = profile.get("possession", {})
    action_cfg = profile.get("actions", {})

    processors: list[Processor] = [
        # 1. Detection — YOLO inference, splits into players + ball
        VisionDetector(
            profile=detector_profile,
            frame_stride=config.processing.frame_stride,
            resize_width=config.processing.resize_width,
        ),

        # 2. Court detection — first frame only, filters players
        CourtDetector(),

        # 3. Jersey classification — calibrates then filters non-players
        JerseyClassifier(),

        # 4. Player tracking — ByteTrack + trajectory + velocity
        PlayerTracker(
            profile=tracker_profile,
            trail_length=config.ui.trail_length,
        ),

        # 5. Ball tracking — Kalman filter with occlusion bridging
        BallTracker(max_missing=30),

        # 6. Possession — multi-point proximity + temporal smoothing
        PossessionEngine(
            threshold=possession_cfg.get("threshold", config.possession.threshold),
            smoothing_frames=possession_cfg.get("smoothing_frames", config.possession.smoothing_frames),
        ),

        # 7. Actions — trajectory analysis, shot arcs, pass detection
        ActionClassifier(
            window_size=action_cfg.get("window_size", config.actions.window_size),
            dribble_variance=action_cfg.get("dribble_variance", config.actions.dribble_variance),
        ),

        # 8. Movement analytics — speed, distance, heat maps
        MovementAnalyzer(),

        # 9. State manager — event log, timeline, stats
        StateManager(),

        # 10. HUD renderer — all visual overlays
        HUDRenderer(ui_config=config.ui),
    ]

    return processors


# ── Main pipeline loop ─────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str | None,
    config: AppConfig,
    profile: dict,
    show_display: bool = True,
    render: bool = True,
    export_csv: bool = False,
    export_json: bool = False,
    video_name: str = "unknown",
):
    """
    Core pipeline: read frames → process → write output.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", input_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps

    log.info("Video: %dx%d @ %.1f fps — %d frames (%.1fs)", width, height, fps, total_frames, total_duration)

    # Build pipeline
    processors = build_pipeline(config, profile)

    # Setup all processors
    for p in processors:
        p.setup(fps, width, height)

    # Video writer
    writer = None
    if output_path and render:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        log.info("Output: %s", output_path)

    log.info("Pipeline: %d processors", len(processors))
    log.info("Processing...")

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # Create fresh FrameState
            state = FrameState(
                frame_index=frame_index,
                timestamp=frame_index / fps,
                fps=fps,
                raw_frame=frame,
            )

            # Run through pipeline
            for processor in processors:
                state = processor.process(state)

            # Write output frame
            if writer and state.annotated_frame is not None:
                writer.write(state.annotated_frame)

            # Live preview
            if show_display and state.annotated_frame is not None:
                preview = cv2.resize(state.annotated_frame, (960, 540))
                cv2.imshow("bball_tracker", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log.info("User quit at frame %d", frame_index)
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)  # pause

            # Progress bar
            if frame_index % 30 == 0 or frame_index == total_frames:
                action_str = state.current_action.type if state.current_action.type != "none" else "—"
                log_progress(
                    frame_index, total_frames, fps, total_duration,
                    state.possessor_id, action_str,
                )

    except KeyboardInterrupt:
        log.warning("Interrupted at frame %d", frame_index)

    # Cleanup
    print()  # newline after progress bar
    cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()

    # Teardown processors
    for p in processors:
        p.teardown()

    # ── Post-processing ────────────────────────────────────────────────
    state_mgr = next((p for p in processors if isinstance(p, StateManager)), None)
    movement = next((p for p in processors if isinstance(p, MovementAnalyzer)), None)
    jersey = next((p for p in processors if isinstance(p, JerseyClassifier)), None)

    if state_mgr:
        player_stats = state_mgr.get_summary()
        events = state_mgr.get_events()
        timeline = state_mgr.get_timeline()
        poss_pct = state_mgr.get_possession_pct()

        # Console summary
        print(f"\n{'═' * 60}")
        print(f"  SESSION SUMMARY — {video_name}")
        print(f"{'═' * 60}")

        if poss_pct:
            print(f"\n  Possession:  Team A {poss_pct.get('team_a', 0):.0f}%  |  "
                  f"Team B {poss_pct.get('team_b', 0):.0f}%")

        print(f"\n  {'Player':<10} {'Hold':>8} {'Dribble':>8} {'Pass':>6} {'Shot':>6} {'TO':>6}")
        print(f"  {'─' * 50}")
        for pid, s in sorted(player_stats.items()):
            print(f"  #{pid:<8} {s['hold_time']:>7.1f}s {s['dribble_time']:>7.1f}s "
                  f"{s['passes']:>5} {s['shots']:>5} {s['turnovers']:>5}")

        print(f"\n  Events logged: {len(events)}")
        if output_path:
            print(f"  Video saved:   {output_path}")

        # Exports
        if export_csv or export_json:
            exporter = Exporter(str(config.project_root / config.export.output_dir))

            team_assignments = {}
            # Collect from last state's assignments would be ideal,
            # but we'll build from what jersey classifier has
            move_stats = movement.get_all_stats() if movement else None

            if export_csv:
                exporter.export_csv(player_stats, move_stats, team_assignments)

            if export_json:
                exporter.export_json(
                    player_stats, events, timeline, poss_pct,
                    move_stats, team_assignments, fps, video_name,
                )

                exporter.export_timeline_chart(timeline, team_assignments, fps)

        print(f"{'═' * 60}\n")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(level=args.log_level)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    config = load_app_config(
        project_root=PROJECT_ROOT,
        config_path=args.config,
    )

    if args.frame_stride:
        config.processing.frame_stride = args.frame_stride

    # Resolve input video
    input_path = args.input
    if args.interactive or not input_path:
        input_path = select_video_interactive(PROJECT_ROOT / "data" / "raw")
        if not input_path:
            return

    video_name = Path(input_path).stem

    # Classify camera angle
    log.info("Classifying camera angle for %s...", video_name)
    view_classifier = ViewClassifier()
    view = view_classifier.classify_video(input_path)
    log.info("Detected angle: %s (confidence: %.0f%%)", view.angle, view.confidence * 100)

    # Force profile if specified
    if args.profile:
        view.angle = args.profile
        log.info("Forced profile: %s", args.profile)

    # Load angle profile
    angle_profiles = config.angle_profiles
    if view.angle in angle_profiles:
        ap = angle_profiles[view.angle]
        profile = {
            "detector": {
                "model_name": ap.detector.model_name,
                "inference_size": ap.detector.inference_size,
                "person_confidence": ap.detector.person_confidence,
                "ball_confidence": ap.detector.ball_confidence,
            },
            "tracker": {
                "track_activation_threshold": ap.tracker.track_activation_threshold,
                "lost_track_buffer": ap.tracker.lost_track_buffer,
                "minimum_matching_threshold": ap.tracker.minimum_matching_threshold,
                "minimum_consecutive_frames": ap.tracker.minimum_consecutive_frames,
                "frame_rate": ap.tracker.frame_rate,
            },
            "possession": {
                "threshold": ap.possession.threshold,
                "smoothing_frames": ap.possession.smoothing_frames,
            },
            "actions": {
                "window_size": ap.actions.window_size,
                "dribble_variance": ap.actions.dribble_variance,
            },
        }
    else:
        log.warning("No profile for '%s', using defaults", view.angle)
        profile = {}

    # Output path
    output_path = args.output or str(
        PROJECT_ROOT / "data" / "processed" / f"processed_{video_name}.mp4"
    )

    # Run
    run_pipeline(
        input_path=input_path,
        output_path=output_path,
        config=config,
        profile=profile,
        show_display=not args.no_display,
        render=not args.no_render,
        export_csv=args.export_csv,
        export_json=args.export_json,
        video_name=video_name,
    )


if __name__ == "__main__":
    main()