# utils/alert.py
"""
Alert system: debounce, terminal print, screenshot saving, structured logging.
"""

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from colorama import Fore, Style

from utils.logger import get_logger, log_alert

logger = get_logger("alert")

# ANSI alert banner
_ALERT_HEADER = f"{Fore.RED}{Style.BRIGHT}"
_RESET = Style.RESET_ALL


class ThreatAlert:
    """
    Manages threat alerts with:
    - Per-track debouncing (avoid spamming same alert)
    - Screenshot saving annotated frame
    - Terminal print with color
    - Structured JSONL logging
    - Consecutive-frame confirmation (false positive reduction)
    """

    def __init__(
        self,
        screenshot_dir: str = "alerts",
        log_dir: str = "logs",
        debounce_seconds: float = 3.0,
        consecutive_required: int = 3,
    ):
        self.screenshot_dir = screenshot_dir
        self.log_dir = log_dir
        self.debounce_seconds = debounce_seconds
        self.consecutive_required = consecutive_required

        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # track_id → {"last_alert_time": float, "consecutive": int}
        self._state: dict[str, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        threat_type: str,
        confidence: float,
        frame: np.ndarray,
        track_id: Optional[int] = None,
        bbox: Optional[tuple] = None,
        extra: Optional[dict] = None,
    ) -> bool:
        """
        Call every frame when a threat is detected.  Returns True if an alert
        was fired (after confirming consecutive frames and debounce).

        Args:
            threat_type:  e.g. "FIRE", "KNIFE", "GUN", "SUSPICIOUS_ACTIVITY"
            confidence:   Detection confidence 0–1
            frame:        Current annotated BGR frame
            track_id:     Person/object track ID (None = global key)
            bbox:         (x1, y1, x2, y2) of detected threat
            extra:        Additional metadata dict for JSONL log

        Returns:
            True if alert was triggered this call.
        """
        key = f"{threat_type}_{track_id if track_id is not None else 'global'}"
        now = time.time()
        state = self._state.setdefault(key, {"last_alert_time": 0, "consecutive": 0})

        # Increment consecutive counter
        state["consecutive"] += 1

        # Only fire alert after N consecutive frames AND debounce window passed
        if state["consecutive"] >= self.consecutive_required:
            if (now - state["last_alert_time"]) >= self.debounce_seconds:
                state["last_alert_time"] = now
                state["consecutive"] = 0  # Reset after firing
                self._fire_alert(threat_type, confidence, frame, track_id, bbox, extra)
                return True

        return False

    def reset(self, threat_type: str, track_id: Optional[int] = None) -> None:
        """Reset consecutive counter when threat disappears (reduces false positives)."""
        key = f"{threat_type}_{track_id if track_id is not None else 'global'}"
        if key in self._state:
            self._state[key]["consecutive"] = 0

    def reset_all(self) -> None:
        """Clear all state (call at start of new session)."""
        self._state.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _fire_alert(
        self,
        threat_type: str,
        confidence: float,
        frame: np.ndarray,
        track_id: Optional[int],
        bbox: Optional[tuple],
        extra: Optional[dict],
    ) -> None:
        timestamp = datetime.now()
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        # 1. Terminal print
        print(
            f"\n{_ALERT_HEADER}"
            f"⚠  ALERT! {threat_type} DETECTED  "
            f"| Conf: {confidence:.2%}"
            f"| TrackID: {track_id}"
            f"| Time: {timestamp.strftime('%H:%M:%S')}"
            f"{_RESET}\n"
        )

        # 2. Save screenshot
        filename = f"{ts_str}_{threat_type}_id{track_id}.jpg"
        save_path = os.path.join(self.screenshot_dir, filename)
        annotated = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Burn-in timestamp + label
        cv2.putText(
            annotated,
            f"ALERT: {threat_type} ({confidence:.0%})",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(save_path, annotated)

        # 3. Structured log
        event = {
            "timestamp": timestamp.isoformat(),
            "threat_type": threat_type,
            "confidence": round(float(confidence), 4),
            "track_id": track_id,
            "bbox": list(bbox) if bbox else None,
            "screenshot": save_path,
            **(extra or {}),
        }
        log_alert(event, log_dir=self.log_dir)
        logger.warning(
            "ALERT fired: %s | conf=%.2f | track=%s | saved=%s",
            threat_type,
            confidence,
            track_id,
            save_path,
        )
