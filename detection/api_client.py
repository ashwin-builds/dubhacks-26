"""Send incidents from the detection worker to the API.

Every call is keyed by the worker-generated ``client_incident_id``:

1. ``POST /incidents`` when an incident opens: the ID, start time, location,
   camera, hazard type, and the compressed first frame.
2. ``POST /incidents/false-positives`` every 10 s while any incident is open:
   the IDs of the open incidents the API knows about. The API answers with
   ``{id: true}`` for each one the manager marked as a false positive; those
   are dropped and never closed. Any other answer keeps the incident going.
3. ``POST /incidents/{client_incident_id}/close`` when it clears: the duration.

Failed open/close calls are queued in memory and retried in order on the next
frame, so a close is never sent ahead of its open. A failed false-positive
check is simply tried again 10 s later.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Callable, Protocol

import requests

from detection.compression import compress_frame
from detection.incidents import ClassifiedFrame, IncidentDetector, IncidentEvent, IncidentReport

log = logging.getLogger(__name__)

# TODO(zach): placeholder -- the zone camera_1 is pointed at. Must be a key in shared/zones.yaml.
CAMERA_LOCATIONS = {"camera_1": "prep_line"}

# How often open incidents are checked against the manager's replies.
FALSE_POSITIVE_CHECK_SECONDS = 10.0


class IncidentApi(Protocol):
    def open_incident(self, report: IncidentReport, location: str, image: bytes | None) -> None: ...
    def close_incident(self, report: IncidentReport) -> None: ...
    def false_positives(self, incident_ids: list[str]) -> dict[str, bool]: ...


class HttpIncidentApi:
    def __init__(self, base_url: str, token: str, timeout: float = 5.0, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def open_incident(self, report: IncidentReport, location: str, image: bytes | None) -> None:
        data = {
            "client_incident_id": report.client_incident_id,
            "started_at": report.started_at.isoformat(),
            "zone": location,
            "camera_id": report.camera_id,
            "hazard_type": report.hazard_type,
        }
        files = {"frame": ("frame.jpg", image, "image/jpeg")} if image else None
        resp = self.session.post(f"{self.base_url}/incidents", data=data, files=files, timeout=self.timeout)
        resp.raise_for_status()

    def close_incident(self, report: IncidentReport) -> None:
        assert report.ended_at is not None
        data = {
            "ended_at": report.ended_at.isoformat(),
            "exposure_seconds": report.duration_seconds,
            "missing_frames": report.missing_frames,
        }
        resp = self.session.post(
            f"{self.base_url}/incidents/{report.client_incident_id}/close", data=data, timeout=self.timeout
        )
        resp.raise_for_status()

    def false_positives(self, incident_ids: list[str]) -> dict[str, bool]:
        resp = self.session.post(
            f"{self.base_url}/incidents/false-positives", json={"incident_ids": incident_ids}, timeout=self.timeout
        )
        resp.raise_for_status()
        answers = resp.json()
        if not isinstance(answers, dict):
            raise ValueError(f"expected a JSON object of id -> bool, got {type(answers).__name__}")
        return answers


class StubIncidentApi:
    """Records calls instead of sending them, for tests and offline demos.

    Add IDs to ``false_positive_ids`` to play a manager answering "no".
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, IncidentReport | None, dict]] = []
        self.false_positive_ids: set[str] = set()

    def open_incident(self, report: IncidentReport, location: str, image: bytes | None) -> None:
        log.info("stub open %s %s at %s", report.client_incident_id, report.hazard_type, location)
        self.calls.append(("open", report, {"location": location, "image": image}))

    def close_incident(self, report: IncidentReport) -> None:
        log.info("stub close %s after %.1fs", report.client_incident_id, report.duration_seconds)
        self.calls.append(("close", report, {}))

    def false_positives(self, incident_ids: list[str]) -> dict[str, bool]:
        self.calls.append(("check", None, {"incident_ids": list(incident_ids)}))
        return {i: i in self.false_positive_ids for i in incident_ids}


class IncidentReporter:
    """Glues an :class:`IncidentDetector` to the API."""

    def __init__(
        self,
        detector: IncidentDetector,
        api: IncidentApi,
        location: str | None = None,
        compress: Callable[[object], bytes] = compress_frame,
        check_every_seconds: float = FALSE_POSITIVE_CHECK_SECONDS,
    ) -> None:
        self.detector = detector
        self.api = api
        self.location = location or CAMERA_LOCATIONS[detector.camera_id]
        self.compress = compress
        self.check_every_seconds = check_every_seconds
        self._pending: deque[Callable[[], None]] = deque()
        self._registered: set[str] = set()  # open incidents whose open call reached the API
        self._last_check: datetime | None = None

    def process(self, result: ClassifiedFrame) -> list[IncidentEvent]:
        """Feed one classified frame through the detector and report what changed."""
        events = self.handle(self.detector.update(result))
        return events + self._maybe_check(result.timestamp)

    def finish(self) -> list[IncidentEvent]:
        """End of stream: close anything still open."""
        return self.handle(self.detector.flush())

    def handle(self, events: list[IncidentEvent]) -> list[IncidentEvent]:
        for event in events:
            report = event.report
            if event.kind == "opened":
                image = self.compress(report.first_frame) if report.first_frame is not None else None
                self._pending.append(lambda r=report, i=image: self._send_open(r, i))
            elif event.kind == "closed":
                self._pending.append(lambda r=report: self._send_close(r))
        self._drain()
        return events

    def _send_open(self, report: IncidentReport, image: bytes | None) -> None:
        self.api.open_incident(report, self.location, image)
        self._registered.add(report.client_incident_id)

    def _send_close(self, report: IncidentReport) -> None:
        self.api.close_incident(report)
        self._registered.discard(report.client_incident_id)

    def _maybe_check(self, now: datetime) -> list[IncidentEvent]:
        """Every ``check_every_seconds`` of stream time, ask which open incidents were false positives."""
        if self._last_check is None:
            self._last_check = now
        if (now - self._last_check).total_seconds() < self.check_every_seconds:
            return []
        self._last_check = now
        # Only incidents the API has heard of, and that haven't closed since.
        still_open = {r.client_incident_id for r in self.detector.open_incidents()}
        ids = sorted(self._registered & still_open)
        if not ids:
            return []
        try:
            answers = self.api.false_positives(ids)
        except (requests.RequestException, ValueError) as exc:  # ValueError: the reply wasn't JSON
            log.warning("false-positive check failed, retrying in %.0fs: %s", self.check_every_seconds, exc)
            return []
        events = []
        for incident_id in ids:
            if answers.get(incident_id) is True:
                event = self.detector.reject(incident_id)
                self._registered.discard(incident_id)
                if event is not None:
                    events.append(event)
        return events

    def _drain(self) -> None:
        while self._pending:
            try:
                self._pending[0]()
            except requests.RequestException as exc:
                log.warning("API call failed, %d queued for retry: %s", len(self._pending), exc)
                return
            self._pending.popleft()
