# 🏀 BBall Tracker

Real-time basketball video analysis with player tracking, ball tracking, possession detection, action classification, and team analytics.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     CLI / main.py                            │
│  argparse, video I/O, pipeline orchestration                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Pipeline Engine                            │
│  FrameState flows through ordered processors                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌──────────────┐  ┌───────────────┐
  │ Detection│  │   Tracking   │  │   Analytics   │
  ├──────────┤  ├──────────────┤  ├───────────────┤
  │ YOLO Det │  │ PlayerTracker│  │ Possession    │
  │ Court    │  │ BallTracker  │  │ Actions       │
  │ Jersey   │  │ (Kalman)     │  │ Movement      │
  │ View     │  │              │  │ StateManager  │
  └──────────┘  └──────────────┘  └───────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  ┌──────────────┐            ┌──────────────┐
  │ HUD Renderer │            │   Exporter   │
  │ (video out)  │            │ CSV/JSON/PNG │
  └──────────────┘            └──────────────┘
```

### Key Design: `FrameState`

Every processor reads/writes to a single `FrameState` dataclass — the single source of truth for each frame. This replaces loose variable passing and makes every module independently testable.

### Key Feature: Kalman Ball Tracker

The basketball is the hardest object to track (~20px, fast, occluded 30–50% of frames). Instead of using raw YOLO detections per-frame, a Kalman filter provides **continuous, smooth ball position every frame** — even during occlusion. This unblocks accurate possession, shot arcs, and trajectory analysis.

## Installation

```bash
# Clone
git clone https://github.com/kush124k/bball_tracker.git
cd bball_tracker

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (auto-downloads on first run, or manually)
mkdir models
# Model will be downloaded automatically by ultralytics
```

## Quick Start

### CLI Mode
```bash
# Basic usage
python -m src.main --input path/to/basketball_video.mp4

# With exports
python -m src.main -i video.mp4 --export-csv --export-json

# Headless (no preview window)
python -m src.main -i video.mp4 --no-display

# Force camera angle
python -m src.main -i video.mp4 --profile sideline

# Debug logging
python -m src.main -i video.mp4 --log-level DEBUG

# Skip every other frame for speed
python -m src.main -i video.mp4 --frame-stride 2
```

### Interactive Mode
```bash
python -m src.main --interactive
# Lists videos in data/raw/ and lets you pick one
```

### Live Preview Controls
- **Space** — Pause/resume
- **Q** — Quit

## Project Structure

```
bball_tracker/
├── config.yaml              # Global settings
├── angle_profiles.yaml      # Per-camera-angle tuning
├── requirements.txt
├── src/
│   ├── main.py              # Pipeline orchestrator + CLI
│   ├── core/
│   │   ├── config.py        # Typed config loader
│   │   ├── frame_state.py   # FrameState + ActionEvent + GameEvent
│   │   └── state_manager.py # Game state + event log + stats
│   ├── features/
│   │   ├── detector.py      # YOLO inference + frame stride
│   │   ├── tracker.py       # ByteTrack + trajectories + velocity
│   │   ├── ball_tracker.py  # Kalman filter ball tracker
│   │   ├── court_detector.py # Multi-surface court detection + homography
│   │   ├── jersey_classifier.py # Team separation via colour clustering
│   │   ├── view_classifier.py   # Camera angle classification
│   │   └── keypoint.py      # Optional YOLOv8-pose for hand positions
│   ├── analytics/
│   │   ├── possession.py    # Multi-point proximity + temporal smoothing
│   │   ├── actions.py       # Shot arc, pass, dribble, turnover detection
│   │   ├── movement.py      # Speed, distance, heat maps, sprints
│   │   └── exporter.py      # CSV, JSON, charts, DataFrames
│   ├── rendering/
│   │   └── hud.py           # All visual annotations
│   └── utils/
│       ├── geometry.py      # Shared bbox/point math
│       └── logger.py        # Structured colored logging
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_geometry.py
│   ├── test_analytics.py
│   └── test_detector.py
├── data/
│   ├── raw/                 # Input videos (git-ignored)
│   ├── processed/           # Output videos
│   └── exports/             # CSV, JSON, charts
└── models/                  # YOLO weights (git-ignored)
```

## Configuration

### `config.yaml`

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model.name` | `yolov8n.pt` | YOLO model to use |
| `model.pose_model` | `null` | Optional pose model for hand tracking |
| `possession.threshold` | `160` | Max px distance for possession |
| `possession.smoothing_frames` | `5` | Frames before possession switches |
| `processing.frame_stride` | `1` | Process every Nth frame |
| `processing.resize_width` | `1280` | Downscale width before inference |
| `ui.show_trails` | `true` | Draw player trajectory trails |
| `ui.show_minimap` | `true` | Show court minimap overlay |
| `ui.show_team_colors` | `true` | Colour boxes by team |
| `export.csv` | `true` | Export player stats CSV |
| `export.json` | `true` | Export full session JSON |

### `angle_profiles.yaml`

Per-angle tuning for detector confidence, tracker buffer, possession threshold, and dribble variance. Three profiles: `sideline`, `broadcast`, `overhead`.

## Testing

```bash
pytest tests/ -v
```

## Output

### Annotated Video
- Team-coloured bounding boxes (blue / red)
- Kalman-filtered ball position with glow effect
- Player trajectory trails
- HUD panel with live stats
- Action toasts (PASS, SHOT, TURNOVER)
- Court minimap with player dots

### Exports
- `player_stats.csv` — Per-player: hold time, dribble time, passes, shots, turnovers, distance, speed
- `session.json` — Full session data with play-by-play event log
- `possession_timeline.png` — Horizontal bar chart of possession over time
- `heatmap_player_N.png` — Per-player spatial heat maps

## License

MIT
