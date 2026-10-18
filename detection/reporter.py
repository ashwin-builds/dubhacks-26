"""Glue the incident detector to the database.

* Incident opened -> insert a row (hazard type, start time, compressed first
  frame, zone); the database's ``incident_id`` is remembered for later writes.
* Incident resolved -> write the duration and mark it completed.
* Every 10 s while anything is open -> ask the database which of our open
  incidents the manager marked as false positives; those are marked completed
  there and force-stopped here.

Failed inserts and completions are queued in memory and retried in order on the
next frame, so a completion never runs before its insert. A failed
false-positive check is simply tried again 10 s later.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Callable

from detection.compression import compress_frame
from detection.incidents import ClassifiedFrame, IncidentDetector, IncidentEvent, IncidentReport
from detection.store import IncidentStore, StoreError

log = logging.getLogger(__name__)

# TODO(zach): placeholder -- the zone camera_1 is pointed at. Must match a `location` in the zones table.
CAMERA_LOCATIONS = {"camera_1": "prep_line"}

# How often open incidents are checked for the manager's false-positive marks.
FALSE_POSITIVE_CHECK_SECONDS = 10.0


class IncidentReporter:
    def __init__(
        self,
        detector: IncidentDetector,
        store: IncidentStore,
        location: str | None = None,
        compress: Callable[[object], bytes] = compress_frame,
        check_every_seconds: float = FALSE_POSITIVE_CHECK_SECONDS,
    ) -> None:
        self.detector = detector
        self.store = store
        self.location = location or CAMERA_LOCATIONS[detector.camera_id]
        self.compress = compress
        self.check_every_seconds = check_every_seconds
        self._pending: deque[Callable[[], None]] = deque()
        self._db_ids: dict[str, int] = {}  # client_incident_id -> incidents.incident_id, while open
        self._last_check: datetime | None = None

    def process(self, result: ClassifiedFrame) -> list[IncidentEvent]:
        """Feed one classified frame through the detector and record what changed."""
        events = self.handle(self.detector.update(result))
        return events + self._maybe_check(result.timestamp)

    def finish(self) -> list[IncidentEvent]:
        """End of stream: complete anything still open."""
        return self.handle(self.detector.flush())

    def handle(self, events: list[IncidentEvent]) -> list[IncidentEvent]:
        for event in events:
            report = event.report
            if event.kind == "opened":
                image = self.compress(report.first_frame) if report.first_frame is not None else None
                self._pending.append(lambda r=report, i=image: self._create(r, i))
            elif event.kind == "closed":
                self._pending.append(lambda r=report: self._complete(r))
        self._drain()
        return events

    def db_id(self, client_incident_id: str) -> int | None:
        return self._db_ids.get(client_incident_id)

    def _create(self, report: IncidentReport, image: bytes | None) -> None:
        db_id = self.store.create_incident(report.hazard_type, report.started_at, image, self.location)
        self._db_ids[report.client_incident_id] = db_id

    def _complete(self, report: IncidentReport) -> None:
        db_id = self._db_ids.get(report.client_incident_id)
        if db_id is None:  # stopped as a false positive while this write waited in the queue
            return
        self.store.complete_incident(db_id, report.duration_seconds)
        del self._db_ids[report.client_incident_id]

    def _maybe_check(self, now: datetime) -> list[IncidentEvent]:
        """Every ``check_every_seconds`` of stream time, stop open incidents marked as false positives."""
        if self._last_check is None:
            self._last_check = now
        if (now - self._last_check).total_seconds() < self.check_every_seconds:
            return []
        self._last_check = now
        # Only incidents that have a row, and that the detector still has open.
        still_open = {r.client_incident_id for r in self.detector.open_incidents()}
        by_db_id = {db_id: cid for cid, db_id in self._db_ids.items() if cid in still_open}
        if not by_db_id:
            return []
        try:
            stopped = self.store.stop_false_positives(sorted(by_db_id))
        except StoreError as exc:
            log.warning("false-positive check failed, retrying in %.0fs: %s", self.check_every_seconds, exc)
            return []
        events = []
        for db_id in stopped:
            client_id = by_db_id.get(db_id)
            if client_id is None:
                continue
            del self._db_ids[client_id]
            event = self.detector.reject(client_id)
            if event is not None:
                events.append(event)
        return events

    def _drain(self) -> None:
        while self._pending:
            try:
                self._pending[0]()
            except StoreError as exc:
                log.warning("database write failed, %d queued for retry: %s", len(self._pending), exc)
                return
            self._pending.popleft()
