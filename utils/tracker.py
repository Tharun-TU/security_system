# utils/tracker.py
"""
Multi-object tracker wrapper around Ultralytics ByteTrack.

Provides a stable `TrackManager` that accepts raw YOLO Results objects
and returns a list of tracked detections with persistent IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackedObject:
    """Single tracked object with a persistent ID."""

    track_id: int
    cls_id: int
    class_name: str
    bbox: tuple[float, float, float, float]   # xyxy absolute pixels
    confidence: float
    is_lost: bool = False
    # Per-object state for activity classifier's sliding window
    frame_buffer: list = field(default_factory=list)


class TrackManager:
    """
    Thin stateful wrapper around Ultralytics built-in ByteTrack.

    Usage::

        tracker = TrackManager(model=yolo_person_model, config=cfg["tracking"])
        # Each frame:
        tracked = tracker.update(frame)
        for obj in tracked:
            print(obj.track_id, obj.bbox, obj.confidence)
    """

    def __init__(self, model, config: dict, class_names: list[str]):
        """
        Args:
            model:        Ultralytics YOLO model with tracking capability.
            config:       Tracking section from detection_config.yaml.
            class_names:  List of class names indexed by class id.
        """
        self.model = model
        self.config = config
        self.class_names = class_names
        # Track history: track_id → TrackedObject (for frame buffers)
        self._objects: dict[int, TrackedObject] = {}

    def update(
        self,
        frame: np.ndarray,
        conf: float = 0.40,
        iou: float = 0.45,
        classes: Optional[list[int]] = None,
        img_size: int = 640,
        device: str = "cpu",
    ) -> list[TrackedObject]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame:    BGR frame (H×W×3).
            conf:     Detection confidence threshold.
            iou:      NMS IoU threshold.
            classes:  Filter to specific class ids (e.g. [0] for person).
            img_size: Inference image size.
            device:   Torch device string.

        Returns:
            List of :class:`TrackedObject`.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            classes=classes,
            imgsz=img_size,
            device=device,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        active_ids: set[int] = set()
        tracked_objs: list[TrackedObject] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                if box.id is None:
                    continue  # Track not yet assigned

                tid = int(box.id.item())
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                xyxy = tuple(box.xyxy[0].cpu().numpy().tolist())  # (x1,y1,x2,y2)
                cname = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else str(cls_id)
                )

                active_ids.add(tid)

                if tid not in self._objects:
                    self._objects[tid] = TrackedObject(
                        track_id=tid,
                        cls_id=cls_id,
                        class_name=cname,
                        bbox=xyxy,
                        confidence=conf_val,
                    )
                else:
                    obj = self._objects[tid]
                    obj.bbox = xyxy
                    obj.confidence = conf_val
                    obj.is_lost = False

                tracked_objs.append(self._objects[tid])

        # Mark lost tracks (not seen this frame)
        for tid, obj in self._objects.items():
            if tid not in active_ids:
                obj.is_lost = True

        # Prune very old lost tracks (simple memory management)
        self._objects = {
            tid: obj for tid, obj in self._objects.items() if not obj.is_lost
        }

        return tracked_objs

    def get_frame_buffer(self, track_id: int) -> list:
        """Return the stored frame buffer for a given track (for activity model)."""
        if track_id in self._objects:
            return self._objects[track_id].frame_buffer
        return []

    def push_frame_to_buffer(
        self, track_id: int, crop: np.ndarray, max_len: int = 16
    ) -> None:
        """Append a cropped frame to the track's sliding window buffer."""
        if track_id in self._objects:
            buf = self._objects[track_id].frame_buffer
            buf.append(crop)
            if len(buf) > max_len:
                buf.pop(0)
