# utils/iou.py
"""
Bounding-box geometry helpers (xyxy format).
"""

from typing import Sequence


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Compute Intersection-over-Union for two boxes in xyxy format.

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        IoU value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """
    Fraction of `inner` box that lies inside `outer` box.
    Useful for checking if a weapon/fire is *inside* a person box.

    Returns value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    inner_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if inner_area <= 0:
        return 0.0
    return inter_area / inner_area


def expand_box(
    box: Sequence[float],
    factor: float = 1.1,
    frame_w: int = 9999,
    frame_h: int = 9999,
) -> tuple[float, float, float, float]:
    """
    Expand a bounding box by `factor` around its center, clamped to frame.

    Args:
        box:     (x1, y1, x2, y2)
        factor:  Expansion factor (1.1 = 10% larger on each side).
        frame_w: Frame width for clamping.
        frame_h: Frame height for clamping.

    Returns:
        Expanded (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * factor, (y2 - y1) * factor
    nx1 = max(0.0, cx - w / 2)
    ny1 = max(0.0, cy - h / 2)
    nx2 = min(float(frame_w), cx + w / 2)
    ny2 = min(float(frame_h), cy + h / 2)
    return nx1, ny1, nx2, ny2


def is_overlapping(
    box_a: Sequence[float],
    box_b: Sequence[float],
    iou_threshold: float = 0.05,
) -> bool:
    """
    Return True if two boxes overlap above a given IoU threshold.
    Default threshold is intentionally low (0.05) because a small weapon
    against a large person box still has low IoU despite clear proximity.

    Use :func:`compute_containment` for more intuitive "inside-person" logic.
    """
    return compute_iou(box_a, box_b) >= iou_threshold
