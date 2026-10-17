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


def test_reporter_sends_open_then_close():
    api = StubIncidentApi()
    reporter = make_reporter(api)
    for result in frames("1" * 70 + "0" * 20):
        reporter.process(result)
    (k1, open_report, extra), (k2, close_report, _) = api.calls
    assert (k1, k2) == ("open", "close")
    assert extra == {"location": "prep_line", "image": b"jpeg:img0"}
    assert open_report.client_incident_id == close_report.client_incident_id
    assert close_report.duration_seconds == 35.0


def test_reporter_reject_suppresses_close():
    api = StubIncidentApi()
    reporter = make_reporter(api)
    results = frames("1" * 100 + "0" * 20)
    for result in results[:60]:
        reporter.process(result)
    assert reporter.reject(api.calls[0][1].client_incident_id)
    for result in results[60:]:
        reporter.process(result)
    reporter.finish()
    assert [c[0] for c in api.calls] == ["open"]


class FlakyApi(StubIncidentApi):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def open_incident(self, *args):
        if self.failures:
            self.failures -= 1
            raise requests.ConnectionError("wifi")
        super().open_incident(*args)


def test_failed_calls_retry_in_order():
    api = FlakyApi(failures=25)
    reporter = make_reporter(api)
    for result in frames("1" * 70 + "0" * 20):
        reporter.process(result)
    # Open failed on frames 60..84; the close at frame 89 queued behind it.
    assert [c[0] for c in api.calls] == ["open", "close"]
