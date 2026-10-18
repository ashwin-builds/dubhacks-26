"""Incident detection: turn per-frame danger classifications into incidents.

The classifier labels each frame on its own. A single positive frame is not an
incident -- a knife on the board for half a second is normal kitchen work. This
module watches a sliding window of recent frames per danger type and opens an
incident only when the danger has been present for a sustained stretch.

Rule (per danger type, per camera):

* Keep the last ``window_frames`` classifications (40 frames = 20 s at 2 fps).
* Open an incident once the window is full and at least ``min_ratio`` of it
  (80% -> 32 of 40 frames) shows the danger.
* The incident's start is the earliest danger frame still in the window, and
  that frame's image is attached to the report.
* Close the incident once the danger has been absent for ``clear_after_frames``
  consecutive frames. Duration runs from the first danger frame to the end of
  the last one.
* If the manager rejects the incident (false positive), stop tracking it and
  send nothing more. The same danger cannot reopen until it has cleared, so a
  persistent false positive does not text the manager again 16 s later.

Missing frames are found from gaps in the frame timestamps. They count as "no
danger seen" toward opening (an incident needs real evidence), but they neither
advance nor reset the clear countdown of an open incident (a dropped frame is
not proof the hazard is gone). Each report carries how many frames were missing.

Durations are computed from the frames' own timestamps, never wall-clock time,
so a prerecorded video gives the same answer however fast it is played.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterable, Literal

log = logging.getLogger(__name__)


class Danger(str, Enum):
    """The four dangers; values are the hazard keys in ``detection.hazards`` and the shared contract."""

    SHARPS = "knife_edge"
    BURNERS = "unattended_burner"
    SPILLS = "spill"
    CONTAMINATION = "raw_meat_contact"

    @property
    def hazard_type(self) -> str:
        return self.value

FPS = 2.0

# TODO(zach): placeholder -- every danger uses the same 20 s window for now.
# Per-danger durations will come from the hazard_policy table later.
DANGER_SECONDS: dict[Danger, float] = {danger: 20.0 for danger in Danger}

MIN_RATIO = 0.8

# How long a danger must be gone before an open incident closes.
CLEAR_AFTER_SECONDS = 10.0

# Single camera for now.
DEFAULT_CAMERA_ID = "camera_1"

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 on the model's 0-1000 grid


@dataclass(frozen=True)
class ClassifiedFrame:
    """One analysed frame: which dangers the model saw in it."""

    timestamp: datetime
    dangers: frozenset[Danger]
    frame: Any = None  # the image itself (JPEG bytes, ndarray, ...); passed through untouched
    boxes: tuple[tuple[Danger, Box], ...] = ()  # where each danger was seen, if known


@dataclass(frozen=True)
class IncidentReport:
    client_incident_id: str  # generated here; maps to the database row while the incident is open
    danger: Danger
    camera_id: str
    started_at: datetime  # timestamp of the first danger frame
    duration_seconds: float  # so far while open; final once closed
    first_frame: Any
    missing_frames: int = 0  # frames dropped between the first danger frame and now
    ended_at: datetime | None = None  # set on close
    boxes: tuple[Box, ...] = ()  # where the danger was last seen; set on reject

    @property
    def hazard_type(self) -> str:
        return self.danger.hazard_type


@dataclass(frozen=True)
class IncidentEvent:
    kind: Literal["opened", "closed", "rejected"]
    report: IncidentReport


@dataclass
class _Sample:
    present: bool
    missing: bool = False
    timestamp: datetime | None = None
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
    state: Literal["idle", "open", "rejected"] = "idle"
    open_report: IncidentReport | None = None
    last_seen: datetime | None = None
    last_boxes: tuple[Box, ...] = ()
    misses: int = 0

    def __post_init__(self) -> None:
        self.window = deque(maxlen=self.window_frames)

    def _push(self, sample: _Sample) -> None:
        if len(self.window) == self.window_frames and self.window[0].present:
            self.positives -= 1
        self.window.append(sample)
        self.positives += sample.present

    def add_missing(self, count: int) -> None:
        # Past one full window, more missing samples change nothing.
        for _ in range(min(count, self.window_frames)):
            self._push(_Sample(present=False, missing=True))
        if self.open_report is not None:
            self.open_report = replace(
                self.open_report, missing_frames=self.open_report.missing_frames + count
            )

    def update(self, result: ClassifiedFrame, camera_id: str) -> IncidentEvent | None:
        present = self.danger in result.dangers
        # Only danger frames keep their image; the rest would just hold memory.
        self._push(_Sample(present, timestamp=result.timestamp, frame=result.frame if present else None))

        if present:
            boxes = tuple(box for danger, box in result.boxes if danger == self.danger)
            self.last_boxes = boxes or self.last_boxes

        if self.state == "idle":
            return self._maybe_open(result, camera_id)

        if present:
            self.last_seen = result.timestamp
            self.misses = 0
            return None
        self.misses += 1
        if self.misses < self.clear_after_frames:
            return None
        if self.state == "rejected":
            self._reset()
            return None
        return self.close()

    def _maybe_open(self, result: ClassifiedFrame, camera_id: str) -> IncidentEvent | None:
        if len(self.window) < self.window_frames or self.positives < self.min_positive:
            return None
        samples = list(self.window)
        start = next(i for i, s in enumerate(samples) if s.present)
        first = samples[start]
        self.state = "open"
        self.last_seen = result.timestamp
        self.misses = 0
        self.open_report = IncidentReport(
            client_incident_id=str(uuid.uuid4()),
            danger=self.danger,
            camera_id=camera_id,
            started_at=first.timestamp,
            duration_seconds=self._duration(first.timestamp, result.timestamp),
            first_frame=first.frame,
            missing_frames=sum(s.missing for s in samples[start:]),
        )
        return IncidentEvent("opened", self.open_report)

    def reject(self) -> None:
        """Manager said it was a false positive: drop it, wait for it to clear."""
        self.state = "rejected"
        self.open_report = None
        self.misses = 0

    def close(self) -> IncidentEvent | None:
        report = self.open_report
        self._reset()
        if report is None:
            return None
        assert self.last_seen is not None
        event = IncidentEvent(
            "closed",
            replace(
                report,
                ended_at=self.last_seen + timedelta(seconds=self.frame_seconds),
                duration_seconds=self._duration(report.started_at, self.last_seen),
            ),
        )
        self.last_seen = None
        return event

    def _reset(self) -> None:
        self.state = "idle"
        self.open_report = None
        self.misses = 0
        self.last_boxes = ()
        # Forget the evidence behind the last incident, so a new one needs a
        # fresh 80% of the window rather than reopening off stale frames.
        for sample in self.window:
            sample.present = False
            sample.frame = None
        self.positives = 0

    def _duration(self, first: datetime, last: datetime) -> float:
        # Each frame stands for 1/fps seconds, so a single frame lasts 0.5 s, not 0.
        return (last - first).total_seconds() + self.frame_seconds


class IncidentDetector:
    """Tracks every danger type for one camera.

    Feed it frames in order with :meth:`update`; it returns the incidents that
    opened or closed on that frame. Call :meth:`reject` when the manager says an
    incident was a false positive, and :meth:`flush` when the stream ends so any
    still-open incident is closed.
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
        self.frame_seconds = 1.0 / fps
        self.missing_frames = 0  # total over the whole stream
        self._last_timestamp: datetime | None = None
        seconds = danger_seconds or DANGER_SECONDS
        self._trackers: dict[Danger, _DangerTracker] = {}
        for danger in Danger:
            window_frames = round(seconds[danger] * fps)
            self._trackers[danger] = _DangerTracker(
                danger=danger,
                window_frames=window_frames,
                min_positive=round(window_frames * min_ratio),
                clear_after_frames=max(1, round(clear_after_seconds * fps)),
                frame_seconds=self.frame_seconds,
            )

    def update(self, result: ClassifiedFrame) -> list[IncidentEvent]:
        missing = self._count_missing(result.timestamp)
        if missing:
            self.missing_frames += missing
            log.warning("%s: %d frame(s) missing before %s", self.camera_id, missing, result.timestamp)
            for tracker in self._trackers.values():
                tracker.add_missing(missing)

        events = []
        for tracker in self._trackers.values():
            event = tracker.update(result, self.camera_id)
            if event is not None:
                events.append(event)
        return events

    def _count_missing(self, timestamp: datetime) -> int:
        last, self._last_timestamp = self._last_timestamp, timestamp
        if last is None:
            return 0
        # Rounding absorbs jitter: a frame up to 0.75 intervals late is not a drop.
        return max(0, round((timestamp - last).total_seconds() / self.frame_seconds) - 1)

    def reject(self, client_incident_id: str) -> IncidentEvent | None:
        """Stop recording an open incident the manager marked as a false positive.

        Nothing further is reported for it. Returns None if no open incident
        has that ID (for example it already closed before the reply arrived).
        """
        for tracker in self._trackers.values():
            report = tracker.open_report
            if report is not None and report.client_incident_id == client_incident_id:
                tracker.reject()
                return IncidentEvent("rejected", replace(report, boxes=tracker.last_boxes))
        return None

    def run(self, results: Iterable[ClassifiedFrame]) -> list[IncidentEvent]:
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
