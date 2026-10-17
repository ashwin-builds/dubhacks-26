"""Connect Perceptron results to the incident detector.

The demo's ``Analyzer`` runs several model calls at once, so answers come back
out of order. The incident detector needs frames in order, so each sample gets
a sequence number when it is sent, and answers are held here until every
earlier sample has been answered (or has failed).

Samples the analyzer skipped (all request slots busy) are never sent and get no
sequence number; failed calls are dropped here. Either way the detector sees a
gap in the timestamps and counts those frames as missing.

When the manager marks an incident as a false positive, the spot it was seen in
becomes an ignore zone (see ``detection.suppression``): detections of that
hazard there are dropped here, before the detector sees them.

Frame timestamps are the stream's start time plus video time, so incident
durations come from video time and are unaffected by model latency. When a
looping video restarts, video time keeps counting up instead of jumping back.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

from detection.api_client import IncidentReporter
from detection.incidents import ClassifiedFrame, Danger, IncidentEvent
from detection.isaac_client import FrameResult
from detection.overlay import draw_detections
from detection.suppression import Suppressor
from detection.video_source import format_position


class IncidentPipeline:
    def __init__(
        self,
        reporter: IncidentReporter,
        fps: float,
        started_at: datetime | None = None,
        on_event: Callable[[IncidentEvent, "IncidentPipeline"], None] | None = None,
        suppressor: Suppressor | None = None,
    ):
        self.reporter = reporter
        self.suppressor = suppressor or Suppressor()
        self.interval = 1.0 / fps
        self.started_at = started_at or datetime.now(timezone.utc)
        self.on_event = on_event or print_event
        self._lock = threading.Lock()
        self._next_seq = 0
        self._waiting: dict[int, tuple[float, np.ndarray, FrameResult] | None] = {}
        self._offset = 0.0  # video time added by earlier loops of the video
        self._last_position: float | None = None
        # API calls can block for seconds; keep them off the model threads.
        self._worker = ThreadPoolExecutor(max_workers=1)

    def deliver(self, seq: int, position_s: float, frame: np.ndarray, result: FrameResult | None) -> None:
        """Called by the analyzer when sample ``seq`` is answered; ``result`` is None if the call failed."""
        with self._lock:
            self._waiting[seq] = (position_s, frame, result) if result is not None else None
            while self._next_seq in self._waiting:
                item = self._waiting.pop(self._next_seq)
                self._next_seq += 1
                if item is not None:
                    self._worker.submit(self._process, self._classify(*item))

    def close(self) -> None:
        """End of stream: process anything still held (skipping samples that never came back), then flush."""
        with self._lock:
            for seq in sorted(self._waiting):
                item = self._waiting.pop(seq)
                if item is not None:
                    self._worker.submit(self._process, self._classify(*item))
        self._worker.submit(lambda: self._emit(self.reporter.finish()))
        self._worker.shutdown(wait=True)

    def video_time(self, timestamp: datetime) -> float:
        return (timestamp - self.started_at).total_seconds()

    def _classify(self, position_s: float, frame: np.ndarray, result: FrameResult) -> ClassifiedFrame:
        if self._last_position is not None and position_s < self._last_position:
            self._offset += self._last_position + self.interval  # the video looped
        self._last_position = position_s
        timestamp = self.started_at + timedelta(seconds=self._offset + position_s)
        detections = [
            d for d in result.detections if not self.suppressor.is_suppressed(Danger(d.hazard), d.box, timestamp)
        ]
        dangers = frozenset(Danger(d.hazard) for d in detections)
        return ClassifiedFrame(
            timestamp=timestamp,
            dangers=dangers,
            # Only frames with a danger can become an incident's first frame; send it with the boxes drawn on.
            frame=draw_detections(frame, detections) if dangers else None,
            boxes=tuple((Danger(d.hazard), d.box) for d in detections),
        )

    def _process(self, frame: ClassifiedFrame) -> None:
        events = self.reporter.process(frame)
        for event in events:
            if event.kind == "rejected":
                report = event.report
                with self._lock:  # the suppressor is also read in _classify, under this lock
                    self.suppressor.add(report.danger, report.boxes, frame.timestamp, report.client_incident_id)
        self._emit(events)

    def _emit(self, events: list[IncidentEvent]) -> None:
        for event in events:
            self.on_event(event, self)

    def open_dangers(self) -> list[Danger]:
        return [report.danger for report in self.reporter.detector.open_incidents()]


def print_event(event: IncidentEvent, pipeline: IncidentPipeline) -> None:
    report = event.report
    start = format_position(pipeline.video_time(report.started_at))
    if event.kind == "opened":
        print(
            f"*** INCIDENT OPENED: {report.hazard_type} on {report.camera_id}, first seen {start} "
            f"(id {report.client_incident_id})",
            flush=True,
        )
    elif event.kind == "rejected":
        print(
            f"*** INCIDENT REJECTED by manager (false positive): {report.hazard_type}, first seen {start}; "
            f"no longer recording, ignoring {report.hazard_type} in that spot (id {report.client_incident_id})",
            flush=True,
        )
    else:
        print(
            f"*** INCIDENT CLOSED: {report.hazard_type}, first seen {start}, lasted {report.duration_seconds:.1f}s, "
            f"{report.missing_frames} frame(s) missing (id {report.client_incident_id})",
            flush=True,
        )
