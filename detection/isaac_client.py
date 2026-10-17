"""Send a frame to Perceptron and get back hazard boxes.

Uses the ``perceptron`` SDK. Set ``PERCEPTRON_API_KEY`` in the environment (or ``.env``).
The model is ``isaac-0.3-fast`` unless ``PERCEPTRON_MODEL`` or the ``model`` argument says otherwise.

Confidence is self-reported: the model is asked to append a 0-1 score to each label. Perceptron's API
exposes no logprobs or detection scores, so treat it as a rough signal, not a calibrated probability.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from detection.hazards import HAZARDS, match_hazard

DEFAULT_MODEL = "isaac-0.3-fast"

# Perceptron returns coordinates on a 0-1000 grid regardless of image size.
GRID = 1000
# Frames are downscaled to this width before upload; plenty for the model and keeps requests small.
MAX_UPLOAD_WIDTH = 1024


@dataclass(frozen=True)
class Detection:
    hazard: str  # key in detection.hazards.HAZARDS
    mention: str  # the label exactly as the model wrote it
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 on the 0-1000 grid
    confidence: float | None = None  # model's own 0-1 estimate; None if it gave none

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.box
        return (
            round(x1 * width / GRID),
            round(y1 * height / GRID),
            round(x2 * width / GRID),
            round(y2 * height / GRID),
        )


@dataclass(frozen=True)
class FrameResult:
    detections: list[Detection]
    unmatched: list[str]  # labels the model gave that map to no known hazard
    latency_s: float
    raw_text: str | None


def build_system_prompt(hazards: list[str]) -> str:
    lines = "\n".join(f"- {key}: {HAZARDS[key].description}" for key in hazards)
    return (
        "You are a safety monitor watching a restaurant kitchen camera. "
        "Find every instance of these hazards in the image:\n"
        f"{lines}\n"
        "Draw one box around each hazard. Label it with exactly one of these keys, a space, and your "
        "confidence from 0 to 1 that it really is that hazard, for example \"spill 0.8\". "
        f"Keys: {', '.join(hazards)}. "
        "Only box real hazards. If there are no hazards, box nothing."
    )


# A trailing score on a label: "spill 0.8", "spill: 0.85", "spill (80%)".
_SCORE = re.compile(r"[\s:=(\[]*(\d*\.?\d+)\s*(%?)[)\]]?\s*$")


def split_confidence(mention: str) -> tuple[str, float | None]:
    """Split a label like ``"spill 0.8"`` into ``("spill", 0.8)``; the score is None if absent or out of range."""
    match = _SCORE.search(mention)
    if not match or match.start() == 0:
        return mention, None
    value = float(match.group(1))
    if match.group(2) or value > 1:
        value /= 100
    if not 0 <= value <= 1:
        return mention, None
    return mention[: match.start()].strip(), value


def encode_frame(frame_bgr: np.ndarray, max_width: int = MAX_UPLOAD_WIDTH) -> bytes:
    height, width = frame_bgr.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame_bgr = cv2.resize(frame_bgr, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not JPEG-encode frame")
    return buf.tobytes()


class IsaacClient:
    """Thin wrapper around ``perceptron.perceive`` for hazard detection.

    ``perceive_fn`` is injectable so tests can run without the network.
    """

    def __init__(
        self,
        model: str | None = None,
        hazards: list[str] | None = None,
        perceive_fn: Callable[..., Any] | None = None,
    ):
        self.model = model or os.environ.get("PERCEPTRON_MODEL") or DEFAULT_MODEL
        self.hazards = hazards or list(HAZARDS)
        unknown = set(self.hazards) - set(HAZARDS)
        if unknown:
            raise ValueError(f"unknown hazards: {sorted(unknown)}")
        if perceive_fn is None:
            from perceptron import perceive as perceive_fn
        self._perceive = perceive_fn
        self.system_prompt = build_system_prompt(self.hazards)

    def detect(self, frame_bgr: np.ndarray) -> FrameResult:
        from perceptron import image, system, text

        nodes = [
            system(self.system_prompt),
            image(encode_frame(frame_bgr)),
            text("List the hazards in this kitchen frame."),
        ]
        start = time.monotonic()
        result = self._perceive(
            *nodes,
            expects="box",
            model=self.model,
            temperature=0.0,
        )
        latency = time.monotonic() - start
        return self._parse(result, latency)

    def _parse(self, result: Any, latency: float) -> FrameResult:
        detections: list[Detection] = []
        unmatched: list[str] = []
        for box in result.boxes or []:
            mention = (box.mention or "").strip()
            label, confidence = split_confidence(mention)
            key = match_hazard(label)
            if key is None or key not in self.hazards:
                unmatched.append(mention)
                continue
            x1, y1 = box.top_left.x, box.top_left.y
            x2, y2 = box.bottom_right.x, box.bottom_right.y
            detections.append(
                Detection(
                    hazard=key,
                    mention=mention,
                    box=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                    confidence=confidence,
                )
            )
        return FrameResult(detections=detections, unmatched=unmatched, latency_s=latency, raw_text=result.text)
