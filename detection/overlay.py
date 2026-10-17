"""Draw hazard boxes and a status bar on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from detection.hazards import HAZARDS
from detection.isaac_client import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of ``frame`` with each detection boxed and labelled."""
    out = frame.copy()
    height, width = out.shape[:2]
    for det in detections:
        hazard = HAZARDS[det.hazard]
        x1, y1, x2, y2 = det.pixel_box(width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), hazard.color, 3)
        label = hazard.label if det.confidence is None else f"{hazard.label} {det.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
        ty = max(y1, th + baseline + 4)
        cv2.rectangle(out, (x1, ty - th - baseline - 4), (x1 + tw + 8, ty), hazard.color, -1)
        cv2.putText(out, label, (x1 + 4, ty - baseline - 2), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_status(frame: np.ndarray, message: str, alert: bool) -> np.ndarray:
    """Draw a banner across the top: green when clear, red when a hazard is in view."""
    out = frame.copy()
    width = out.shape[1]
    color = (0, 0, 200) if alert else (0, 140, 0)
    cv2.rectangle(out, (0, 0), (width, 32), color, -1)
    cv2.putText(out, message, (10, 22), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
