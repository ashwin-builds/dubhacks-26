"""Shrink a frame to a small JPEG before it is written to the database."""

from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 640
JPEG_QUALITY = 70


def compress_frame(frame: np.ndarray | bytes, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY) -> bytes:
    """Return ``frame`` as JPEG bytes, longest side at most ``max_side`` pixels.

    Accepts a BGR frame as OpenCV produces it, or already-encoded image bytes.
    """
    if isinstance(frame, (bytes, bytearray)):
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode image bytes")
    height, width = frame.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:  # never upscale
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not JPEG-encode frame")
    return buf.tobytes()
