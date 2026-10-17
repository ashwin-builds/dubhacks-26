"""Shrink a frame to a small JPEG before it goes to the API."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

MAX_SIDE = 640
JPEG_QUALITY = 70


def compress_frame(frame: Any, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY, bgr: bool = False) -> bytes:
    """Return ``frame`` as JPEG bytes, longest side at most ``max_side`` pixels.

    Accepts encoded image bytes, a PIL image, or an HxWx3 uint8 array. Arrays are
    assumed RGB; pass ``bgr=True`` for frames straight from OpenCV.
    """
    if isinstance(frame, (bytes, bytearray)):
        image = Image.open(io.BytesIO(frame))
    elif isinstance(frame, Image.Image):
        image = frame
    else:
        array = frame[..., ::-1] if bgr else frame
        image = Image.fromarray(array)

    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))  # keeps aspect ratio, never upscales
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()
