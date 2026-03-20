"""
src/inference/overlay.py
─────────────────────────
Visual overlay drawn on frames for the display output.

Draws:
  - Bounding boxes with class label and confidence
  - Alert border (red for critical, orange for high, yellow for warning)
  - Alert status text
  - FPS / inference time HUD
  - Temporal validator streak progress bar
"""

from typing import List, Optional

import cv2
import numpy as np

from src.inference.engine    import Detection
from src.inference.alerts    import Alert
from src.inference.validator import TemporalValidator


# Class-specific colours (BGR)
CLASS_COLOURS = {
    "smoke":    (100, 180, 80),    # Muted green
    "flame":    (0,   80,  255),   # Red-orange
    "overheat": (0,   140, 255),   # Orange
}
DEFAULT_COLOUR = (180, 180, 180)

ALERT_BORDER = {1: (0, 0, 255), 2: (0, 140, 255), 3: (0, 200, 255)}


def draw_frame(
    frame:       np.ndarray,
    detections:  List[Detection],
    alert:       Optional[Alert],
    validator:   Optional[TemporalValidator],
    inference_ms: float,
    fps:          float,
    cfg,
) -> np.ndarray:
    """
    Draw all overlays onto a copy of the frame and return it.
    No in-place mutation of the original frame.
    """
    out = frame.copy()

    if cfg.show_bbox:
        _draw_boxes(out, detections)

    if alert:
        _draw_alert_border(out, alert)

    if validator:
        _draw_streak_bar(out, validator, cfg)

    if cfg.show_fps_overlay:
        _draw_hud(out, inference_ms, fps, alert)

    return out


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_boxes(frame: np.ndarray, detections: List[Detection]):
    for det in detections:
        x1, y1, x2, y2 = det.bbox_ints
        colour  = CLASS_COLOURS.get(det.class_name, DEFAULT_COLOUR)
        label   = f"{det.class_name}  {det.confidence:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )


def _draw_alert_border(frame: np.ndarray, alert: Alert):
    """Full-frame coloured border indicating active alert level."""
    colour = ALERT_BORDER.get(alert.level, (0, 0, 255))
    h, w   = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, 5)

    text   = f"L{alert.level} {alert.label}: {alert.detection_type.upper()}"
    cv2.putText(
        frame, text, (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, colour, 2, cv2.LINE_AA,
    )


def _draw_streak_bar(
    frame:     np.ndarray,
    validator: TemporalValidator,
    cfg,
):
    """
    Small progress bar showing temporal validator streak progress.
    Fills left-to-right as consecutive detections accumulate.
    """
    h, w    = frame.shape[:2]
    streak  = validator.streak
    target  = cfg.alert_persistence_frames

    bar_w   = 200
    bar_h   = 8
    x0      = w - bar_w - 10
    y0      = h - 24

    # Background
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
    # Fill
    if streak > 0:
        fill_w  = int(bar_w * min(streak / target, 1.0))
        colour  = (0, 200, 0) if streak < target else (0, 80, 255)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), colour, -1)

    cv2.putText(
        frame, f"streak {streak}/{target}",
        (x0, y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA,
    )


def _draw_hud(
    frame:        np.ndarray,
    inference_ms: float,
    fps:          float,
    alert:        Optional[Alert],
):
    """Bottom-left HUD: FPS, inference time, cooldown state."""
    h, w   = frame.shape[:2]
    lines  = [
        f"FPS {fps:4.1f}  |  Inf {inference_ms:5.0f}ms",
    ]
    if alert is None:
        lines.append("No active alert")

    for i, line in enumerate(lines):
        y = h - 12 - i * 18
        cv2.putText(
            frame, line, (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )