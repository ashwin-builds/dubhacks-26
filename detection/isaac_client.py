"""Send a frame to Perceptron and get back hazard boxes.

Uses the ``perceptron`` SDK. Set ``PERCEPTRON_API_KEY`` in the environment (or ``.env``).
Model choice: the ``model`` argument, else ``PERCEPTRON_MODEL``, else the fastest model in
the preference lists below that the API key can actually use (asked once via the Models API).

This is per-frame classification only: each box gets a hazard label. Deciding when a hazard has lasted
long enough to be an incident happens elsewhere, not here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from detection.hazards import HAZARDS, match_hazard

# Fastest first. Which of these exist depends on the account; the gateway is the source of truth.
ISAAC_PREFERENCE = ("isaac-0.3-fast", "isaac-0.2-2b-preview", "isaac-0.2-1b", "isaac-0.1")
MK_PREFERENCE = ("perceptron-mk1", "perceptron-mk1.5")
FALLBACK_MODEL = "perceptron-mk1.5"  # the SDK default, used if the model list can't be fetched

# Perceptron returns coordinates on a 0-1000 grid regardless of image size.
GRID = 1000
# Frames are downscaled to this width before upload; plenty for the model and keeps requests small.
MAX_UPLOAD_WIDTH = 1024


@dataclass(frozen=True)
class Detection:
    hazard: str  # key in detection.hazards.HAZARDS
    mention: str  # the label exactly as the model wrote it
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 on the 0-1000 grid

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
        "Cooks handling knives and cooking at the stove is normal kitchen work, not a hazard. A knife or burner is "
        "only a hazard when no one is attending to it. When you cannot tell whether someone is attending to it, "
        "do not box it.\n"
        "Draw one box around each hazard and label it with exactly one of these keys: "
        f"{', '.join(hazards)}. "
        "Only box real hazards. If there are no hazards, box nothing."
    )



def encode_frame(frame_bgr: np.ndarray, max_width: int = MAX_UPLOAD_WIDTH) -> bytes:
    height, width = frame_bgr.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame_bgr = cv2.resize(frame_bgr, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not JPEG-encode frame")
    return buf.tobytes()


def list_models() -> list[str]:
    """Model ids this API key can use."""
    from perceptron import Client

    with Client() as client:
        return [m.id for m in client.models.list()]


def pick_model(available: list[str]) -> str:
    """Known Isaac models, then any other Isaac model, then the Mk models, then whatever is available."""
    for model in ISAAC_PREFERENCE:
        if model in available:
            return model
    isaac = [m for m in available if m.startswith("isaac")]
    if isaac:
        return isaac[0]
    for model in MK_PREFERENCE:
        if model in available:
            return model
    return available[0] if available else FALLBACK_MODEL


class IsaacClient:
    """Thin wrapper around ``perceptron.perceive`` for hazard detection.

    ``perceive_fn`` and ``list_models_fn`` are injectable so tests can run without the network.
    """

    def __init__(
        self,
        model: str | None = None,
        hazards: list[str] | None = None,
        perceive_fn: Callable[..., Any] | None = None,
        list_models_fn: Callable[[], list[str]] | None = None,
    ):
        self.model = model or os.environ.get("PERCEPTRON_MODEL") or self._auto_model(list_models_fn or list_models)
        self.hazards = hazards or list(HAZARDS)
        unknown = set(self.hazards) - set(HAZARDS)
        if unknown:
            raise ValueError(f"unknown hazards: {sorted(unknown)}")
        if perceive_fn is None:
            from perceptron import perceive as perceive_fn
        self._perceive = perceive_fn
        self.system_prompt = build_system_prompt(self.hazards)

    @staticmethod
    def _auto_model(list_models_fn: Callable[[], list[str]]) -> str:
        try:
            available = list_models_fn()
        except Exception:  # model listing is a convenience; detection should still start
            available = []
        return pick_model(available)

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
            key = match_hazard(mention)
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
                )
            )
        return FrameResult(detections=detections, unmatched=unmatched, latency_s=latency, raw_text=result.text)
