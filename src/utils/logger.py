"""
Structured logging for bball_tracker.

Provides colored console output and optional file logging.
All modules should use `get_logger(__name__)` instead of print().
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class _Colors:
    """ANSI color codes for terminal output."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on log level."""

    LEVEL_COLORS = {
        logging.DEBUG:    _Colors.DIM,
        logging.INFO:     _Colors.CYAN,
        logging.WARNING:  _Colors.YELLOW,
        logging.ERROR:    _Colors.RED,
        logging.CRITICAL: _Colors.RED + _Colors.BOLD,
    }

    LEVEL_ICONS = {
        "DEBUG":    "DBG",
        "INFO":     " ⓘ ",
        "WARNING":  " ⚠ ",
        "ERROR":    "ERR",
        "CRITICAL": "!!!",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, _Colors.RESET)
        icon = self.LEVEL_ICONS.get(record.levelname, record.levelname)

        # Strip common prefixes for cleaner module names
        module = record.name.replace("bball.", "")

        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()

        return (
            f"{_Colors.DIM}{ts}{_Colors.RESET} "
            f"{color}{icon}{_Colors.RESET} "
            f"{_Colors.DIM}[{module}]{_Colors.RESET} "
            f"{msg}"
        )


class PlainFormatter(logging.Formatter):
    """Formatter for file output — no ANSI codes."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname:>8}] [{record.name}] {record.getMessage()}"


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_initialized = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Initialize the logging system. Call once at startup.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR"
        log_file: Optional path to a persistent log file.
    """
    global _initialized

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("bball")
    root.setLevel(numeric_level)
    root.handlers.clear()

    # Console — colored
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter())
    console.setLevel(numeric_level)
    root.addHandler(console)

    # File — plain, always DEBUG
    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(p), mode="a", encoding="utf-8")
        fh.setFormatter(PlainFormatter())
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Usage::

        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Frame %d processed", idx)
    """
    if not _initialized:
        setup_logging()

    # Namespace everything under 'bball'
    short = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"bball.{short}")


# ---------------------------------------------------------------------------
# Progress bar — separate from logging because it uses \r
# ---------------------------------------------------------------------------

def log_progress(
    frame_index: int,
    total_frames: int,
    fps: float,
    total_duration: float,
    possessor_id: Optional[int] = None,
    action: str = "—",
    bar_width: int = 20,
) -> None:
    """Print an overwriting progress bar to stdout."""
    pct = frame_index / total_frames * 100 if total_frames > 0 else 0
    vid_t = frame_index / fps if fps > 0 else 0

    filled = int(pct / (100 / bar_width))
    bar = "█" * filled + "░" * (bar_width - filled)

    poss = str(possessor_id) if possessor_id is not None else "Loose"

    def _fmt(s: float) -> str:
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    line = (
        f" [{bar}] {pct:5.1f}%  "
        f"frame {frame_index:>5}/{total_frames}  "
        f"{_fmt(vid_t)}/{_fmt(total_duration)}  "
        f"possessor={poss:<6}  "
        f"action={action}"
    )
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()
