"""Incident detection: turn per-frame danger classifications into incidents.

The classifier labels each frame on its own. A single positive frame is not an
incident -- a knife on the board for half a second is normal kitchen work. This
module watches a sliding window of recent frames per danger type and opens an
incident only when the danger has been present for a sustained stretch.

Rule (per danger type, per camera):

* Keep the last ``window_frames`` classifications (60 frames = 30 s at 2 fps).
* Open an incident once the window is full and at least ``min_ratio`` of it
  (80% -> 48 of 60 frames) shows the danger.
* The incident's start is the earliest danger frame still in the window, and
  that frame's image is attached to the report.
* Close the incident once the danger has been absent for ``clear_after_frames``
  consecutive frames. Duration runs from the first danger frame to the end of
  the last one.

Durations are computed from the frames' own timestamps, never wall-clock time,
so a prerecorded video gives the same answer however fast it is played.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterable, Literal


class Danger(str, Enum):
    SHARPS = "sharps"
    BURNERS = "burners"
    SPILLS = "spills"
    CONTAMINATION = "contamination"

    @property
    def hazard_type(self) -> str:
        """The shared-contract ``hazard_type`` name the API expects."""
        return _HAZARD_TYPES[self]


_HAZARD_TYPES = {
    Danger.SHARPS: "knife_edge",
    Danger.BURNERS: "unattended_burner",
    Danger.SPILLS: "spill",
    Danger.CONTAMINATION: "raw_meat_contact",
}

FPS = 2.0

# TODO(zach): placeholder -- every danger uses the same 30 s window for now.
# Per-danger durations will come from the hazard_policy table later.
DANGER_SECONDS: dict[Danger, float] = {danger: 30.0 for danger in Danger}

MIN_RATIO = 0.8

# How long a danger must be gone before an open incident closes.
CLEAR_AFTER_SECONDS = 10.0

# Single camera for now.
DEFAULT_CAMERA_ID = "camera_1"


@dataclass(frozen=True)
class FrameResult:
    """One classified frame, as produced by the danger classifier."""

    timestamp: datetime
    dangers: frozenset[Danger]
    frame: Any = None  # the image itself (JPEG bytes, ndarray, ...); passed through untouched


@dataclass(frozen=True)
class IncidentReport:
    client_incident_id: str
    danger: Danger
    camera_id: str
    started_at: datetime  # timestamp of the first danger frame
    duration_seconds: float  # so far while open; final once closed
    first_frame: Any
    ended_at: datetime | None = None  # set on close

    @property
    def hazard_type(self) -> str:
        return self.danger.hazard_type


@dataclass(frozen=True)
class IncidentEvent:
    kind: Literal["opened", "closed"]
    report: IncidentReport


@dataclass
class _Sample:
    timestamp: datetime
    present: bool
    frame: Any = None


@dataclass
class _DangerTracker:
    danger: Danger
    window_frames: int
    min_positive: int
    clear_after_frames: int
    frame_seconds: float
    window: deque[_Sample] = field(init=False)
    positives: int = 0
    open_report: IncidentReport | None = None
    last_seen: datetime | None = None
    misses: int = 0

    def __post_init__(self) -> None:
        self.window = deque(maxlen=self.window_frames)

    def update(self, result: FrameResult, camera_id: str) -> IncidentEvent | None:
        present = self.danger in result.dangers
        if len(self.window) == self.window_frames and self.window[0].present:
            self.positives -= 1
        # Only danger frames keep their image; the rest would just hold memory.
        self.window.append(_Sample(result.timestamp, present, result.frame if present else None))
        self.positives += present

        if self.open_report is not None:
            return self._update_open(result, present)

        if len(self.window) == self.window_frames and self.positives >= self.min_positive:
            first = next(s for s in self.window if s.present)
            self.last_seen = result.timestamp
            self.misses = 0
            self.open_report = IncidentReport(
                client_incident_id=str(uuid.uuid4()),
                danger=self.danger,
                camera_id=camera_id,
                started_at=first.timestamp,
                duration_seconds=self._duration(first.timestamp, result.timestamp),
                first_frame=first.frame,
            )
            return IncidentEvent("opened", self.open_report)
        return None

    def _update_open(self, result: FrameResult, present: bool) -> IncidentEvent | None:
        if present:
            self.last_seen = result.timestamp
            self.misses = 0
            return None
        self.misses += 1
        if self.misses >= self.clear_after_frames:
            return self.close()
        return None

    def close(self) -> IncidentEvent | None:
        if self.open_report is None:
            return None
        assert self.last_seen is not None
        report = replace(
            self.open_report,
            ended_at=self.last_seen + timedelta(seconds=self.frame_seconds),
            duration_seconds=self._duration(self.open_report.started_at, self.last_seen),
        )
        self.open_report = None
        self.last_seen = None
        self.misses = 0
        # Forget the evidence that opened this incident, so a new one needs a
        # fresh 80% of the window rather than reopening off stale frames.
        for sample in self.window:
            sample.present = False
            sample.frame = None
        self.positives = 0
        return IncidentEvent("closed", report)

    def _duration(self, first: datetime, last: datetime) -> float:
        # Each frame stands for 1/fps seconds, so a single frame lasts 0.5 s, not 0.
        return (last - first).total_seconds() + self.frame_seconds


class IncidentDetector:
    """Tracks every danger type for one camera.

    Feed it frames in order with :meth:`update`; it returns the incidents that
    opened or closed on that frame. Call :meth:`flush` when the stream ends so
    any still-open incident is closed.
    """

    def __init__(
        self,
        camera_id: str = DEFAULT_CAMERA_ID,
        fps: float = FPS,
        danger_seconds: dict[Danger, float] | None = None,
        min_ratio: float = MIN_RATIO,
        clear_after_seconds: float = CLEAR_AFTER_SECONDS,
    ) -> None:
        self.camera_id = camera_id
        seconds = danger_seconds or DANGER_SECONDS
        self._trackers: dict[Danger, _DangerTracker] = {}
        for danger in Danger:
            window_frames = round(seconds[danger] * fps)
            self._trackers[danger] = _DangerTracker(
                danger=danger,
                window_frames=window_frames,
                min_positive=round(window_frames * min_ratio),
                clear_after_frames=max(1, round(clear_after_seconds * fps)),
                frame_seconds=1.0 / fps,
            )

    def update(self, result: FrameResult) -> list[IncidentEvent]:
        events = []
        for tracker in self._trackers.values():
            event = tracker.update(result, self.camera_id)
            if event is not None:
                events.append(event)
        return events

    def run(self, results: Iterable[FrameResult]) -> list[IncidentEvent]:
        """Process a whole sequence of frames, flushing at the end."""
        events = []
        for result in results:
            events.extend(self.update(result))
        events.extend(self.flush())
        return events

    def flush(self) -> list[IncidentEvent]:
        events = []
        for tracker in self._trackers.values():
            event = tracker.close()
            if event is not None:
                events.append(event)
        return events

    def open_incidents(self) -> list[IncidentReport]:
        return [t.open_report for t in self._trackers.values() if t.open_report is not None]
