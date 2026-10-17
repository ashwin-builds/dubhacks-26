import cv2
import numpy as np
import requests

from detection.api_client import IncidentReporter, StubIncidentApi
from detection.compression import compress_frame
from detection.incidents import IncidentDetector
from detection.tests.test_incidents import frames


def test_compress_frame_shrinks_to_jpeg():
    big = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    jpeg = compress_frame(big)
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
    assert image.shape[:2] == (360, 640)
    assert len(jpeg) < big.nbytes / 20


def test_compress_frame_accepts_encoded_bytes():
    ok, png = cv2.imencode(".png", np.zeros((50, 100, 3), np.uint8))
    image = cv2.imdecode(np.frombuffer(compress_frame(png.tobytes()), np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (50, 100)  # never upscaled


def make_reporter(api):
    return IncidentReporter(IncidentDetector(), api, location="prep_line", compress=lambda f: f"jpeg:{f}".encode())


def kinds(api, include_checks=False):
    return [c[0] for c in api.calls if include_checks or c[0] != "check"]


def run(reporter, results):
    events = []
    for result in results:
        events += reporter.process(result)
    return events


def test_reporter_sends_open_then_close():
    api = StubIncidentApi()
    reporter = make_reporter(api)
    run(reporter, frames("1" * 70 + "0" * 20))
    assert kinds(api) == ["open", "close"]
    (_, open_report, extra), (_, close_report, _) = [c for c in api.calls if c[0] != "check"]
    assert extra == {"location": "prep_line", "image": b"jpeg:img0"}
    assert open_report.client_incident_id == close_report.client_incident_id
    assert close_report.duration_seconds == 35.0


def test_open_incidents_are_checked_every_10_seconds():
    api = StubIncidentApi()
    reporter = make_reporter(api)
    # Opens at frame 39 (19.5 s); danger stays until frame 99, then clears.
    run(reporter, frames("1" * 100 + "0" * 20))
    incident_id = api.calls[0][1].client_incident_id
    checks = [c[2]["incident_ids"] for c in api.calls if c[0] == "check"]
    # Checks at 20, 30, 40, 50 s while open (none at 10 s: nothing was open yet).
    assert checks == [[incident_id]] * 4
    assert kinds(api) == ["open", "close"]


def test_false_positive_stops_recording_and_suppresses_close():
    api = StubIncidentApi()
    reporter = make_reporter(api)
    results = frames("1" * 200 + "0" * 20)
    events = run(reporter, results[:40])  # opens at frame 39
    incident_id = events[0].report.client_incident_id
    api.false_positive_ids.add(incident_id)  # the manager replies "no"
    events = run(reporter, results[40:])
    reporter.finish()
    assert [(e.kind, e.report.client_incident_id) for e in events] == [("rejected", incident_id)]
    assert kinds(api) == ["open"]
    # One check found it; with nothing open afterwards, no more checks are sent.
    assert kinds(api, include_checks=True) == ["open", "check"]


def test_only_ids_marked_true_are_dropped():
    api = StubIncidentApi()
    detector = IncidentDetector()
    reporter = IncidentReporter(detector, api, location="prep_line", compress=lambda f: b"")
    # A spill and a burner open together.
    from datetime import timedelta

    from detection.incidents import ClassifiedFrame, Danger
    from detection.tests.test_incidents import T0

    both = frozenset({Danger.SPILLS, Danger.BURNERS})
    first = [ClassifiedFrame(T0 + timedelta(seconds=i * 0.5), both, f"img{i}") for i in range(40)]
    run(reporter, first)
    spill, burner = sorted(detector.open_incidents(), key=lambda r: r.danger != Danger.SPILLS)
    api.false_positive_ids.add(spill.client_incident_id)
    later = [ClassifiedFrame(T0 + timedelta(seconds=i * 0.5), both, f"img{i}") for i in range(40, 42)]
    events = run(reporter, later)
    assert [(e.kind, e.report.danger) for e in events] == [("rejected", Danger.SPILLS)]
    assert [r.danger for r in detector.open_incidents()] == [Danger.BURNERS]


class FlakyApi(StubIncidentApi):
    def __init__(self, failures=0, check_failures=0):
        super().__init__()
        self.failures = failures
        self.check_failures = check_failures

    def open_incident(self, *args):
        if self.failures:
            self.failures -= 1
            raise requests.ConnectionError("wifi")
        super().open_incident(*args)

    def false_positives(self, incident_ids):
        if self.check_failures:
            self.check_failures -= 1
            raise requests.ConnectionError("wifi")
        return super().false_positives(incident_ids)


def test_failed_calls_retry_in_order():
    api = FlakyApi(failures=25)
    reporter = make_reporter(api)
    run(reporter, frames("1" * 70 + "0" * 20))
    # Open failed on frames 39..63; the close at frame 89 queued behind it.
    assert kinds(api) == ["open", "close"]


def test_incidents_the_api_has_not_heard_of_are_not_checked():
    api = FlakyApi(failures=1000)  # the open call never gets through
    reporter = make_reporter(api)
    run(reporter, frames("1" * 100))
    assert kinds(api, include_checks=True) == []


def test_failed_check_is_retried_next_time():
    api = FlakyApi(check_failures=1)
    reporter = make_reporter(api)
    results = frames("1" * 100)
    events = run(reporter, results[:40])
    api.false_positive_ids.add(events[0].report.client_incident_id)
    events = run(reporter, results[40:])
    # The 20 s check failed; the 30 s one found the false positive.
    assert [e.kind for e in events] == ["rejected"]
    assert reporter.detector.open_incidents() == []


def test_http_false_positive_check_request_shape():
    from types import SimpleNamespace

    from detection.api_client import HttpIncidentApi

    sent = {}

    class Session:
        headers = {}

        def post(self, url, json=None, timeout=None):
            sent.update(url=url, json=json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"a": True, "b": False})

    api = HttpIncidentApi("http://api:8000/", "tok", session=Session())
    assert api.false_positives(["a", "b"]) == {"a": True, "b": False}
    assert sent == {"url": "http://api:8000/incidents/false-positives", "json": {"incident_ids": ["a", "b"]}}
    assert Session.headers["Authorization"] == "Bearer tok"
