"""Send incidents from the detection worker to the API.

Each incident makes at most two calls, both keyed by the worker-generated
``client_incident_id``:

1. ``POST /incidents`` when an incident opens: the ID, start time, location,
   camera, hazard type, and the compressed first frame.
2. ``POST /incidents/{client_incident_id}/close`` when it clears: the duration.

If the manager rejects an incident first, :meth:`IncidentReporter.reject` drops
it and the close call is never made.

Failed calls are queued in memory and retried in order on the next frame, so a
close is never sent ahead of its open.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Protocol

import requests

from detection.compression import compress_frame
from detection.incidents import FrameResult, IncidentDetector, IncidentEvent, IncidentReport

log = logging.getLogger(__name__)

# TODO(zach): placeholder -- the zone camera_1 is pointed at. Must be a key in shared/zones.yaml.
CAMERA_LOCATIONS = {"camera_1": "prep_line"}


class IncidentApi(Protocol):
    def open_incident(self, report: IncidentReport, location: str, image: bytes | None) -> None: ...
    def close_incident(self, report: IncidentReport) -> None: ...


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


class StubIncidentApi:
    """Records calls instead of sending them, for tests and offline demos."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, IncidentReport, dict]] = []

    def open_incident(self, report: IncidentReport, location: str, image: bytes | None) -> None:
        log.info("stub open %s %s at %s", report.client_incident_id, report.hazard_type, location)
        self.calls.append(("open", report, {"location": location, "image": image}))

    def close_incident(self, report: IncidentReport) -> None:
        log.info("stub close %s after %.1fs", report.client_incident_id, report.duration_seconds)
        self.calls.append(("close", report, {}))


class IncidentReporter:
    """Glues an :class:`IncidentDetector` to the API."""

    def __init__(
        self,
        detector: IncidentDetector,
        api: IncidentApi,
        location: str | None = None,
        compress: Callable[[object], bytes] = compress_frame,
    ) -> None:
        self.detector = detector
        self.api = api
        self.location = location or CAMERA_LOCATIONS[detector.camera_id]
        self.compress = compress
        self._pending: deque[Callable[[], None]] = deque()

    def process(self, result: FrameResult) -> None:
        """Feed one classified frame through the detector and report what changed."""
        self.handle(self.detector.update(result))

    def finish(self) -> None:
        """End of stream: close anything still open."""
        self.handle(self.detector.flush())

    def handle(self, events: list[IncidentEvent]) -> None:
        for event in events:
            report = event.report
            if event.kind == "opened":
                image = self.compress(report.first_frame) if report.first_frame is not None else None
                self._pending.append(lambda r=report, i=image: self.api.open_incident(r, self.location, i))
            else:
                self._pending.append(lambda r=report: self.api.close_incident(r))
        self._drain()

    def reject(self, client_incident_id: str) -> bool:
        """The API relayed a manager's "no": stop recording this incident."""
        return self.detector.reject(client_incident_id)

    def _drain(self) -> None:
        while self._pending:
            try:
                self._pending[0]()
            except requests.RequestException as exc:
                log.warning("API call failed, %d queued for retry: %s", len(self._pending), exc)
                return
            self._pending.popleft()
