# utils/logger.py
"""
Unified logging system with rotating file handler + colored console output.
"""

import logging
import os
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

# ── Color map per log level ──────────────────────────────────────────────────
_LEVEL_COLORS = {
    logging.DEBUG:    Fore.CYAN,
    logging.INFO:     Fore.GREEN,
    logging.WARNING:  Fore.YELLOW,
    logging.ERROR:    Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to console output."""

    _FMT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    _DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        formatter = logging.Formatter(
            f"{color}{self._FMT}{Style.RESET_ALL}", datefmt=self._DATE
        )
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    _FMT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        return logging.Formatter(self._FMT, datefmt=self._DATE).format(record)


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "security_system", log_dir: str = "logs") -> logging.Logger:
    """
    Return a named logger. Idempotent — repeated calls with same name return
    the same logger.

    Args:
        name:    Logger name (e.g., module name).
        log_dir: Directory where rotating log file is written.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    if name in _loggers:
        return _loggers[name]

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "detections.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Avoid duplicate logs from root logger

    # Console handler (colors)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColorFormatter())

    # Rotating file handler (10 MB × 5 backups)
    fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())

    logger.addHandler(ch)
    logger.addHandler(fh)

    _loggers[name] = logger
    return logger


def log_alert(event: dict, log_dir: str = "logs") -> None:
    """
    Append a structured alert event as a JSON line to alerts.jsonl.

    Args:
        event:   Dict with at minimum keys: timestamp, threat_type, confidence, track_id.
        log_dir: Directory for the JSONL file.
    """
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "alerts.jsonl")
    event.setdefault("timestamp", datetime.now().isoformat())
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
