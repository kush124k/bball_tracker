"""
Data exporter — CSV, JSON, timeline chart, heat map images.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.frame_state import GameEvent
from utils.logger import get_logger

log = get_logger(__name__)


class Exporter:
    """
    Exports session data to multiple formats.

    Call after the pipeline finishes with data from StateManager
    and MovementAnalyzer.
    """

    def __init__(self, output_dir: str = "data/exports"):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ─── CSV ───────────────────────────────────────────────────────────

    def export_csv(
        self,
        player_stats: Dict[int, dict],
        movement_stats: Optional[Dict[int, dict]] = None,
        team_assignments: Optional[Dict[int, str]] = None,
        filename: str = "player_stats.csv",
    ) -> Path:
        """Export per-player stats as CSV."""
        path = self._dir / filename
        rows = []

        for pid, stats in sorted(player_stats.items()):
            row = {"player_id": pid, **stats}

            # Merge movement stats
            if movement_stats and pid in movement_stats:
                row.update(movement_stats[pid])

            # Team
            if team_assignments and pid in team_assignments:
                row["team"] = team_assignments[pid]

            rows.append(row)

        if not rows:
            log.warning("No player stats to export")
            return path

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        log.info("CSV exported: %s (%d players)", path, len(rows))
        return path

    # ─── JSON ──────────────────────────────────────────────────────────

    def export_json(
        self,
        player_stats: Dict[int, dict],
        events: List[GameEvent],
        timeline: List[Optional[int]],
        possession_pct: Dict[str, float],
        movement_stats: Optional[Dict[int, dict]] = None,
        team_assignments: Optional[Dict[int, str]] = None,
        fps: float = 30.0,
        video_name: str = "unknown",
        filename: str = "session.json",
    ) -> Path:
        """Export full session data as JSON."""
        path = self._dir / filename

        data = {
            "metadata": {
                "video": video_name,
                "fps": fps,
                "total_frames": len(timeline),
                "duration_seconds": round(len(timeline) / fps, 1) if fps > 0 else 0,
            },
            "possession_pct": possession_pct,
            "players": {},
            "events": [],
        }

        for pid, stats in sorted(player_stats.items()):
            p = {"stats": stats}
            if movement_stats and pid in movement_stats:
                p["movement"] = movement_stats[pid]
            if team_assignments and pid in team_assignments:
                p["team"] = team_assignments[pid]
            data["players"][str(pid)] = p

        for ev in events:
            data["events"].append({
                "frame": ev.frame_index,
                "time": round(ev.timestamp, 2),
                "type": ev.event_type,
                "player": ev.player_id,
                "target": ev.target_player_id,
                "team": ev.team,
                "details": ev.details,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        log.info("JSON exported: %s", path)
        return path

    # ─── Timeline chart ────────────────────────────────────────────────

    def export_timeline_chart(
        self,
        timeline: List[Optional[int]],
        team_assignments: Dict[int, str],
        fps: float = 30.0,
        filename: str = "possession_timeline.png",
    ) -> Path:
        """Generate a horizontal bar chart of possession over time."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed — skipping timeline chart")
            return self._dir / filename

        path = self._dir / filename

        fig, ax = plt.subplots(figsize=(16, 3))

        colors = {"team_a": "#3B82F6", "team_b": "#EF4444", "ref": "#FBBF24", None: "#6B7280"}
        times = np.arange(len(timeline)) / fps

        for i, pid in enumerate(timeline):
            team = team_assignments.get(pid, None) if pid is not None else None
            color = colors.get(team, colors[None])
            ax.axvspan(times[i], times[min(i + 1, len(times) - 1)],
                       color=color, alpha=0.8, linewidth=0)

        ax.set_xlim(0, times[-1] if len(times) > 0 else 1)
        ax.set_xlabel("Video Time (seconds)")
        ax.set_yticks([])
        ax.set_title("Possession Timeline")

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color="#3B82F6", label="Team A"),
            Patch(color="#EF4444", label="Team B"),
            Patch(color="#6B7280", label="Loose"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(str(path), dpi=150)
        plt.close()
        log.info("Timeline chart exported: %s", path)
        return path

    # ─── Heat map images ───────────────────────────────────────────────

    def export_heatmap(
        self,
        heatmap: np.ndarray,
        player_id: int,
        filename: Optional[str] = None,
    ) -> Path:
        """Export a player heat map as a PNG image."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed — skipping heat map")
            fname = filename or f"heatmap_player_{player_id}.png"
            return self._dir / fname

        fname = filename or f"heatmap_player_{player_id}.png"
        path = self._dir / fname

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(heatmap, cmap="hot", interpolation="bilinear", aspect="auto")
        ax.set_title(f"Player #{player_id} — Heat Map")
        ax.set_xlabel("Court X")
        ax.set_ylabel("Court Y")
        plt.tight_layout()
        plt.savefig(str(path), dpi=150)
        plt.close()
        log.info("Heat map exported: %s", path)
        return path

    # ─── DataFrame access ──────────────────────────────────────────────

    @staticmethod
    def stats_to_dataframe(
        player_stats: Dict[int, dict],
        movement_stats: Optional[Dict[int, dict]] = None,
    ) -> pd.DataFrame:
        """Return a pandas DataFrame of all player stats."""
        rows = []
        for pid, stats in sorted(player_stats.items()):
            row = {"player_id": pid, **stats}
            if movement_stats and pid in movement_stats:
                row.update(movement_stats[pid])
            rows.append(row)
        return pd.DataFrame(rows)
