"""
Typed configuration for bball_tracker.

Loads config.yaml + angle_profiles.yaml, validates, and provides
structured access.  Supports environment-variable overrides.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Nested config sections ─────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str = "yolov8n.pt"
    target_classes: List[int] = field(default_factory=lambda: [0, 32])
    inference_size: int = 640
    confidence: float = 0.35
    pose_model: Optional[str] = None          # e.g. "yolov8n-pose.pt"


@dataclass
class PossessionConfig:
    threshold: int = 160                      # px distance for "in possession"
    smoothing_frames: int = 5                 # consecutive frames before flip


@dataclass
class ActionConfig:
    window_size: int = 10
    dribble_variance: float = 25.0


@dataclass
class ProcessingConfig:
    frame_stride: int = 1                     # 1 = every frame
    resize_width: int = 1280


@dataclass
class UIConfig:
    hud_color: Tuple[int, int, int] = (0, 255, 0)
    font_scale_base: int = 1200
    show_trails: bool = True
    trail_length: int = 30
    show_minimap: bool = True
    show_team_colors: bool = True
    team_a_color: Tuple[int, int, int] = (255, 140, 50)   # BGR warm blue
    team_b_color: Tuple[int, int, int] = (50, 50, 255)    # BGR red


@dataclass
class ExportConfig:
    csv: bool = True
    json: bool = True
    timeline_chart: bool = True
    output_dir: str = "data/exports"


@dataclass
class LogConfig:
    level: str = "INFO"
    file: Optional[str] = None


# ── Per-angle profile pieces ───────────────────────────────────────────────

@dataclass
class DetectorProfile:
    model_name: str = "yolov8n.pt"
    inference_size: int = 640
    person_confidence: float = 0.40
    ball_confidence: float = 0.20


@dataclass
class TrackerProfile:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 60
    minimum_matching_threshold: float = 0.80
    minimum_consecutive_frames: int = 2
    frame_rate: int = 30


@dataclass
class AngleProfile:
    detector: DetectorProfile = field(default_factory=DetectorProfile)
    tracker: TrackerProfile = field(default_factory=TrackerProfile)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)


# ── Top-level config ───────────────────────────────────────────────────────

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    log: LogConfig = field(default_factory=LogConfig)
    angle_profiles: Dict[str, AngleProfile] = field(default_factory=dict)
    project_root: Path = field(default_factory=lambda: Path.cwd())


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _env_overrides(raw: dict) -> dict:
    """Apply BBALL_* environment variable overrides."""
    mapping = {
        "BBALL_MODEL_NAME":              ("model", "name"),
        "BBALL_MODEL_INFERENCE_SIZE":    ("model", "inference_size"),
        "BBALL_MODEL_CONFIDENCE":        ("model", "confidence"),
        "BBALL_PROCESSING_FRAME_STRIDE": ("processing", "frame_stride"),
        "BBALL_PROCESSING_RESIZE_WIDTH": ("processing", "resize_width"),
        "BBALL_LOG_LEVEL":               ("log", "level"),
    }
    for env_key, (section, key) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None:
            raw.setdefault(section, {})
            # coerce to number if possible
            for cast in (int, float):
                try:
                    val = cast(val)
                    break
                except (ValueError, TypeError):
                    pass
            raw[section][key] = val
    return raw


def _to_detector(d: dict) -> DetectorProfile:
    return DetectorProfile(
        model_name=d.get("model_name", "yolov8n.pt"),
        inference_size=d.get("inference_size", 640),
        person_confidence=d.get("person_confidence", 0.40),
        ball_confidence=d.get("ball_confidence", 0.20),
    )


def _to_tracker(d: dict) -> TrackerProfile:
    return TrackerProfile(
        track_activation_threshold=d.get("track_activation_threshold", 0.25),
        lost_track_buffer=d.get("lost_track_buffer", 60),
        minimum_matching_threshold=d.get("minimum_matching_threshold", 0.80),
        minimum_consecutive_frames=d.get("minimum_consecutive_frames", 2),
        frame_rate=d.get("frame_rate", 30),
    )


def _to_angle(d: dict) -> AngleProfile:
    return AngleProfile(
        detector=_to_detector(d.get("detector", {})),
        tracker=_to_tracker(d.get("tracker", {})),
        possession=PossessionConfig(
            threshold=d.get("possession", {}).get("threshold", 160),
            smoothing_frames=d.get("possession", {}).get("smoothing_frames", 5),
        ),
        actions=ActionConfig(
            window_size=d.get("actions", {}).get("window_size", 10),
            dribble_variance=d.get("actions", {}).get("dribble_variance", 25.0),
        ),
    )


def _tuple3(lst, default=(0, 0, 0)):
    if lst is None:
        return default
    return tuple(lst[:3])


# ── Public loader ──────────────────────────────────────────────────────────

def load_app_config(
    project_root: Optional[Path] = None,
    config_path: Optional[str] = None,
    profiles_path: Optional[str] = None,
) -> AppConfig:
    """
    Load the full application configuration.

    Resolves project_root automatically if not provided (parent of src/).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    cfg = _load_yaml(Path(config_path) if config_path else project_root / "config.yaml")
    cfg = _env_overrides(cfg)
    prof_raw = _load_yaml(
        Path(profiles_path) if profiles_path else project_root / "angle_profiles.yaml"
    )

    angle_profiles = {name: _to_angle(data) for name, data in prof_raw.items()}

    ui = cfg.get("ui", {})
    exp = cfg.get("export", {})
    log = cfg.get("log", {})
    mdl = cfg.get("model", {})
    pos = cfg.get("possession", {})
    act = cfg.get("actions", {})
    prc = cfg.get("processing", {})

    return AppConfig(
        model=ModelConfig(
            name=mdl.get("name", "yolov8n.pt"),
            target_classes=mdl.get("target_classes", [0, 32]),
            inference_size=mdl.get("inference_size", 640),
            confidence=mdl.get("confidence", 0.35),
            pose_model=mdl.get("pose_model"),
        ),
        possession=PossessionConfig(
            threshold=pos.get("threshold", 160),
            smoothing_frames=pos.get("smoothing_frames", 5),
        ),
        actions=ActionConfig(
            window_size=act.get("window_size", 10),
            dribble_variance=act.get("dribble_variance", 25.0),
        ),
        processing=ProcessingConfig(
            frame_stride=prc.get("frame_stride", 1),
            resize_width=prc.get("resize_width", 1280),
        ),
        ui=UIConfig(
            hud_color=_tuple3(ui.get("hud_color"), (0, 255, 0)),
            font_scale_base=ui.get("font_scale_base", 1200),
            show_trails=ui.get("show_trails", True),
            trail_length=ui.get("trail_length", 30),
            show_minimap=ui.get("show_minimap", True),
            show_team_colors=ui.get("show_team_colors", True),
            team_a_color=_tuple3(ui.get("team_a_color"), (255, 140, 50)),
            team_b_color=_tuple3(ui.get("team_b_color"), (50, 50, 255)),
        ),
        export=ExportConfig(
            csv=exp.get("csv", True),
            json=exp.get("json", True),
            timeline_chart=exp.get("timeline_chart", True),
            output_dir=exp.get("output_dir", "data/exports"),
        ),
        log=LogConfig(
            level=log.get("level", "INFO"),
            file=log.get("file"),
        ),
        angle_profiles=angle_profiles,
        project_root=project_root,
    )
