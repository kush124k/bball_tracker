"""
HUD renderer — draws all visual annotations on the output frame.

Extracted from main.py so rendering is:
- Testable (can render to a frame without running the pipeline)
- Configurable (toggle trails, minimap, team colors individually)
- Clean (main.py only orchestrates, HUD only draws)

Visual elements:
- Team-coloured bounding boxes (blue / red / yellow for refs)
- Ball annotation (bright circle + glow)
- Player trajectory trails (fade-out tail)
- Possession indicator (highlighted border on possessor)
- HUD panel (semi-transparent stats overlay)
- Action toast (pop-up text on pass/shot/turnover)
- Court minimap (2D top-down player positions)
"""

import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple

from core.frame_state import FrameState, Processor
from core.config import UIConfig
from utils.geometry import bbox_center, bbox_foot_center, bbox_top_center
from utils.logger import get_logger

log = get_logger(__name__)

# ── Default colours (BGR) ──────────────────────────────────────────────────
COLOR_TEAM_A  = (255, 140, 50)    # warm blue
COLOR_TEAM_B  = (50, 50, 255)     # red
COLOR_REF     = (0, 220, 220)     # yellow
COLOR_DEFAULT = (180, 180, 180)   # gray
COLOR_BALL    = (0, 200, 255)     # bright orange
COLOR_HUD_BG  = (20, 20, 20)     # dark panel


class HUDRenderer(Processor):
    """Draws all visual overlays onto FrameState.annotated_frame."""

    def __init__(self, ui_config: Optional[UIConfig] = None):
        cfg = ui_config or UIConfig()
        self._show_trails = cfg.show_trails
        self._trail_length = cfg.trail_length
        self._show_minimap = cfg.show_minimap
        self._show_team_colors = cfg.show_team_colors
        self._team_a_color = cfg.team_a_color
        self._team_b_color = cfg.team_b_color
        self._font_scale_base = cfg.font_scale_base

        # Action toast state
        self._toast_text: str = ""
        self._toast_frames_remaining: int = 0
        self._toast_duration = 30  # ~1 second at 30fps

        # Possession percentage (updated externally)
        self._poss_pct: Dict[str, float] = {}

        self._frame_w = 1280
        self._frame_h = 720

    # ─── Processor interface ───────────────────────────────────────────

    def setup(self, fps: float, width: int, height: int) -> None:
        self._frame_w = width
        self._frame_h = height
        self._toast_duration = int(fps)

    def process(self, state: FrameState) -> FrameState:
        if state.raw_frame is None:
            return state

        canvas = state.raw_frame.copy()
        w = canvas.shape[1]
        font_scale = w / self._font_scale_base
        thickness = max(1, int(w / 500))

        # 1. Player bounding boxes (team-coloured)
        canvas = self._draw_players(canvas, state, font_scale, thickness)

        # 2. Ball annotation
        canvas = self._draw_ball(canvas, state)

        # 3. Player trails
        if self._show_trails:
            canvas = self._draw_trails(canvas, state)

        # 4. Court boundary debug (subtle)
        if state.court_boundary is not None:
            pts = state.court_boundary.polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 255), 1)

        # 5. HUD panel
        canvas = self._draw_hud_panel(canvas, state, font_scale)

        # 6. Action toast
        canvas = self._draw_toast(canvas, state, font_scale)

        # 7. Minimap
        if self._show_minimap:
            canvas = self._draw_minimap(canvas, state)

        state.annotated_frame = canvas
        return state

    # ─── Drawing methods ───────────────────────────────────────────────

    def _draw_players(
        self, canvas: np.ndarray, state: FrameState,
        font_scale: float, thickness: int
    ) -> np.ndarray:
        tracked = state.tracked_players
        if tracked is None or len(tracked) == 0:
            return canvas

        for i in range(len(tracked)):
            box = tracked.xyxy[i].astype(int)
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None

            # Pick colour
            color = COLOR_DEFAULT
            if self._show_team_colors and tid is not None:
                team = state.team_assignments.get(tid, "")
                if team == "team_a":
                    color = self._team_a_color
                elif team == "team_b":
                    color = self._team_b_color
                elif team == "ref":
                    color = COLOR_REF

            # Thicker border for possessor
            border = thickness + 2 if tid == state.possessor_id else thickness

            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, border)

            # Label
            if tid is not None:
                label = f"#{tid}"
                if tid == state.possessor_id:
                    label += " ●"
                lx, ly = box[0], max(box[1] - 8, 15)
                cv2.putText(canvas, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, thickness)

        return canvas

    def _draw_ball(self, canvas: np.ndarray, state: FrameState) -> np.ndarray:
        pos = state.ball_position
        if pos is None:
            return canvas

        x, y = int(pos[0]), int(pos[1])

        # Glow effect
        overlay = canvas.copy()
        cv2.circle(overlay, (x, y), 18, COLOR_BALL, -1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Core circle
        line_w = 2 if state.ball_visible else 1
        cv2.circle(canvas, (x, y), 10, COLOR_BALL, line_w)

        # If predicted (not visible), draw dashed indicator
        if not state.ball_visible:
            cv2.putText(canvas, "?", (x - 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return canvas

    def _draw_trails(self, canvas: np.ndarray, state: FrameState) -> np.ndarray:
        for tid, trail in state.trajectories.items():
            if len(trail) < 2:
                continue

            team = state.team_assignments.get(tid, "")
            color = self._team_a_color if team == "team_a" else (
                self._team_b_color if team == "team_b" else COLOR_DEFAULT
            )

            points = list(trail)
            for j in range(1, len(points)):
                alpha = j / len(points)  # fade in
                pt1 = tuple(points[j - 1].astype(int))
                pt2 = tuple(points[j].astype(int))
                t = max(1, int(alpha * 2))
                # Fade colour
                faded = tuple(int(c * alpha * 0.6) for c in color)
                cv2.line(canvas, pt1, pt2, faded, t)

        return canvas

    def _draw_hud_panel(
        self, canvas: np.ndarray, state: FrameState, font_scale: float
    ) -> np.ndarray:
        """Semi-transparent stats panel in the top-left corner."""
        panel_w, panel_h = 320, 110
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        fs = font_scale * 0.7
        white = (240, 240, 240)
        y = 35

        # Possessor
        poss_str = f"#{state.possessor_id}" if state.possessor_id else "Loose"
        cv2.putText(canvas, f"Possession: {poss_str}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, white, 1)
        y += 25

        # Action
        action_str = state.current_action.type if state.current_action.type != "none" else "—"
        cv2.putText(canvas, f"Action: {action_str}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, white, 1)
        y += 25

        # Ball state
        cv2.putText(canvas, f"Ball: {state.ball_state}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (180, 180, 180), 1)
        y += 25

        # Timestamp
        ts = state.timestamp
        cv2.putText(canvas, f"Time: {int(ts // 60):02d}:{int(ts % 60):02d}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (180, 180, 180), 1)

        return canvas

    def _draw_toast(
        self, canvas: np.ndarray, state: FrameState, font_scale: float
    ) -> np.ndarray:
        """Brief pop-up text for discrete events (pass, shot, turnover)."""
        action = state.current_action

        if action.type in ("pass", "shot", "turnover"):
            self._toast_text = action.type.upper()
            if action.type == "shot":
                self._toast_text = "🏀 SHOT!"
            elif action.type == "pass":
                self._toast_text = "⚡ PASS"
            elif action.type == "turnover":
                self._toast_text = "❌ TURNOVER"
            self._toast_frames_remaining = self._toast_duration

        if self._toast_frames_remaining > 0:
            alpha = self._toast_frames_remaining / self._toast_duration
            color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))

            cx = self._frame_w // 2
            cy = self._frame_h // 2 - 50
            fs = font_scale * 2.0

            cv2.putText(canvas, self._toast_text, (cx - 100, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 3)
            self._toast_frames_remaining -= 1

        return canvas

    def _draw_minimap(self, canvas: np.ndarray, state: FrameState) -> np.ndarray:
        """Small 2D court schematic in bottom-right with player dots."""
        mm_w, mm_h = 200, 110
        margin = 15
        x0 = self._frame_w - mm_w - margin
        y0 = self._frame_h - mm_h - margin

        # Draw minimap background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + mm_w, y0 + mm_h), (30, 60, 30), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + mm_w, y0 + mm_h), (100, 200, 100), 1)

        # Court lines (simplified)
        cx = x0 + mm_w // 2
        cv2.line(canvas, (cx, y0), (cx, y0 + mm_h), (100, 200, 100), 1)  # half court
        cv2.circle(canvas, (cx, y0 + mm_h // 2), 12, (100, 200, 100), 1)  # center circle

        # Player dots — map frame positions to minimap
        for tid, trail in state.trajectories.items():
            if len(trail) == 0:
                continue

            pos = trail[-1]
            mx = int(x0 + (pos[0] / self._frame_w) * mm_w)
            my = int(y0 + (pos[1] / self._frame_h) * mm_h)
            mx = max(x0 + 2, min(mx, x0 + mm_w - 2))
            my = max(y0 + 2, min(my, y0 + mm_h - 2))

            team = state.team_assignments.get(tid, "")
            color = self._team_a_color if team == "team_a" else (
                self._team_b_color if team == "team_b" else COLOR_DEFAULT
            )

            radius = 5 if tid == state.possessor_id else 3
            cv2.circle(canvas, (mx, my), radius, color, -1)

        # Ball dot
        if state.ball_position is not None:
            bx = int(x0 + (state.ball_position[0] / self._frame_w) * mm_w)
            by = int(y0 + (state.ball_position[1] / self._frame_h) * mm_h)
            bx = max(x0 + 2, min(bx, x0 + mm_w - 2))
            by = max(y0 + 2, min(by, y0 + mm_h - 2))
            cv2.circle(canvas, (bx, by), 3, COLOR_BALL, -1)

        return canvas

    def set_possession_pct(self, pct: Dict[str, float]) -> None:
        self._poss_pct = pct
