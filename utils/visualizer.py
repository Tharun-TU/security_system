# utils/visualizer.py
"""
Frame annotation utilities: bounding boxes, labels, FPS counter,
alert banners, and confidence overlays.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Color palette (BGR) ──────────────────────────────────────────────────────
COLORS = {
    "person":              (0, 200, 255),    # amber
    "fire":                (0, 80, 255),     # fiery red-orange
    "smoke":               (180, 180, 180),  # grey
    "gun":                 (0, 0, 255),      # red
    "knife":               (0, 0, 220),      # dark red
    "running_panic":       (0, 165, 255),    # orange
    "violence":            (0, 0, 255),      # red
    "suspicious":          (0, 128, 255),    # orange-red
    "normal":              (0, 220, 0),      # green
    "default":             (200, 200, 200),  # grey
}

_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 0.65
_THICKNESS   = 2
_LINE_TYPE   = cv2.LINE_AA


class Visualizer:
    """Stateless collection of drawing helpers."""

    @staticmethod
    def draw_box(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        label: str,
        confidence: float,
        color: tuple[int, int, int] | None = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw one bounding box with label + confidence on `frame` (in-place).

        Args:
            frame:      BGR image.
            bbox:       (x1, y1, x2, y2) absolute pixel coords.
            label:      Class name string.
            confidence: Float 0-1.
            color:      BGR tuple override; auto-picked from COLORS if None.
            thickness:  Box line thickness.

        Returns:
            Same frame (modified in-place, returned for chaining).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bgr = color or COLORS.get(label.lower(), COLORS["default"])

        # Rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness, _LINE_TYPE)

        # Label background pill
        text = f"{label.upper()}  {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, label_y2), bgr, -1, _LINE_TYPE)

        # Text
        cv2.putText(
            frame, text, (x1 + 4, label_y2 - baseline),
            _FONT, _FONT_SCALE, (255, 255, 255), _THICKNESS, _LINE_TYPE,
        )
        return frame

    @staticmethod
    def draw_boxes(
        frame: np.ndarray,
        detections: list[dict],
    ) -> np.ndarray:
        """
        Draw multiple detections at once.

        Each detection dict must have keys: ``bbox``, ``label``, ``confidence``.
        Optional ``color`` key overrides palette.
        """
        for det in detections:
            Visualizer.draw_box(
                frame,
                det["bbox"],
                det["label"],
                det["confidence"],
                det.get("color"),
            )
        return frame

    @staticmethod
    def draw_track_id(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        track_id: int,
    ) -> np.ndarray:
        """Small track-ID badge in the top-left corner of a bounding box."""
        x1, y1 = int(bbox[0]), int(bbox[1])
        text = f"#{track_id}"
        cv2.putText(frame, text, (x1 + 2, y1 + 18), _FONT, 0.5, (255, 255, 0), 1, _LINE_TYPE)
        return frame

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        """FPS counter in bottom-left corner."""
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (10, h - 10), _FONT, 0.6, (0, 255, 128), 2, _LINE_TYPE)
        return frame

    @staticmethod
    def draw_alert_banner(
        frame: np.ndarray,
        message: str,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Semi-transparent red banner at the top of the frame with alert text.

        Args:
            frame:   BGR image (modified in place).
            message: Text to display.
            alpha:   Transparency of the banner background (0=invisible,1=solid).

        Returns:
            Frame with banner drawn (in-place).
        """
        h, w = frame.shape[:2]
        banner_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(
            frame,
            f"⚠  THREAT DETECTED: {message}",
            (12, 36),
            _FONT,
            0.9,
            (255, 255, 255),
            2,
            _LINE_TYPE,
        )
        return frame

    @staticmethod
    def draw_info_panel(
        frame: np.ndarray,
        info: dict[str, str],
        start_y: int = 10,
        x: int = 10,
    ) -> np.ndarray:
        """Small info panel (key: value lines) for debug overlay."""
        for i, (k, v) in enumerate(info.items()):
            y = start_y + i * 22
            cv2.putText(frame, f"{k}: {v}", (x, y), _FONT, 0.5, (200, 200, 200), 1, _LINE_TYPE)
        return frame
