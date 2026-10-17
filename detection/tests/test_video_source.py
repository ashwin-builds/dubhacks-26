from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from detection import demo
from detection.isaac_client import Detection, FrameResult
from detection.video_source import VideoFileStream, format_position, is_url, resolve_video

FPS = 10
FRAMES = 20  # a 2-second clip


@pytest.fixture
def clip(tmp_path):
    """A 2 s, 10 fps clip whose frame i has every pixel set to i, so frames can be identified."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    assert writer.isOpened()
    for i in range(FRAMES):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return str(path)


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _frame_index(frame):
    return round(float(frame.mean()) / 10)


def test_plays_every_frame_in_real_time_when_keeping_up(clip):
    clock = FakeClock()
    stream = VideoFileStream(clip, clock=clock, sleep=clock.sleep)
    indices = []
    while True:
        ok, frame = stream.read()
        if not ok:
            break
        indices.append(_frame_index(frame))
    assert indices == list(range(FRAMES))
    assert stream.dropped == 0
    assert clock.now - 100.0 == pytest.approx(FRAMES / FPS)  # paced: ends when the 2 s clip would


def test_drops_frames_when_reader_falls_behind(clip):
    clock = FakeClock()
    stream = VideoFileStream(clip, clock=clock, sleep=clock.sleep)
    stream.read()  # frame 0 at t=0
    clock.now += 0.75  # the caller stalls for 0.75 s
    ok, frame = stream.read()
    assert ok
    assert _frame_index(frame) == 7  # frames 1-6 were dropped, as a live camera would
    assert stream.dropped == 6
    assert stream.position_s == pytest.approx(0.7)


def test_loop_restarts(clip):
    clock = FakeClock()
    stream = VideoFileStream(clip, loop=True, clock=clock, sleep=clock.sleep)
    indices = [_frame_index(stream.read()[1]) for _ in range(FRAMES + 3)]
    assert indices[FRAMES:] == [0, 1, 2]


def test_bad_inputs(tmp_path, clip):
    with pytest.raises(OSError):
        VideoFileStream(str(tmp_path / "missing.mp4"))
    with pytest.raises(FileNotFoundError):
        resolve_video(str(tmp_path / "missing.mp4"))


def test_resolve_video(clip, monkeypatch):
    assert resolve_video(clip) == clip
    assert resolve_video("https://example.com/kitchen.mp4") == "https://example.com/kitchen.mp4"
    monkeypatch.setattr("detection.video_source.download_video", lambda url: f"downloaded:{url}")
    assert resolve_video("https://www.youtube.com/watch?v=abc") == "downloaded:https://www.youtube.com/watch?v=abc"
    assert is_url("http://x.y") and not is_url("clip.mp4")


def test_format_position():
    assert format_position(0) == "00:00.0"
    assert format_position(75.25) == "01:15.2"


def _result(*hazards):
    return FrameResult(
        detections=[Detection(h, h, (0, 0, 10, 10)) for h in hazards], unmatched=[], latency_s=0.1, raw_text=""
    )


def test_summarize():
    history = [(1.5, _result("spill", "knife_edge")), (0.5, _result()), (1.0, _result("spill")), (2.0, _result())]
    text = demo.summarize(history)
    assert "Analysed 4 frames." in text
    assert "Spill" in text and "2 / 4" in text and "first 00:01.0" in text and "last 00:01.5" in text
    assert "Sharp tool" in text and "Unattended burner" not in text
    assert "No hazards detected." in demo.summarize([(0.0, _result())])


class FakeClient:
    """Sees a spill once the clip is past 1 s; records which frames it was sent."""

    def __init__(self, delay_s=0.0):
        self.delay_s = delay_s
        self.frames = []

    def detect(self, frame):
        import time

        time.sleep(self.delay_s)
        index = _frame_index(frame)
        self.frames.append(index)
        return _result("spill") if index >= 10 else _result()


def _stream(clip, clock=None, **kwargs):
    clock = clock or FakeClock()
    return VideoFileStream(clip, clock=clock, sleep=clock.sleep, **kwargs)


def test_samples_at_2fps_of_video_time(clip, capsys):
    """A 2 s clip at 10 fps, streamed live, sends exactly frames 0, 5, 10, 15 (every 0.5 s)."""
    client = FakeClient()
    code = demo.run_video(client, _stream(clip), fps=2, show=False)
    out = capsys.readouterr().out
    assert code == 0
    assert sorted(client.frames) == [0, 5, 10, 15]
    assert "Video finished." in out
    assert "Sent 4 frames to the model, skipped 0" in out
    assert "Spill" in out and "2 / 4 frames" in out and "first 00:01.0" in out


def test_slow_model_keeps_schedule_by_overlapping_requests(clip, capsys):
    """A model slower than the sample interval still gets every sample; requests run in parallel."""
    client = FakeClient(delay_s=0.2)
    clock = FakeClock()
    demo.run_video(client, _stream(clip, clock), fps=2, show=False, analyzer=demo.Analyzer(client, max_in_flight=4))
    assert sorted(client.frames) == [0, 5, 10, 15]


def test_busy_slots_skip_samples():
    import threading

    gate = threading.Event()

    class Blocking:
        def detect(self, frame):
            gate.wait(5)
            return _result()

    analyzer = demo.Analyzer(Blocking(), max_in_flight=1)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert analyzer.submit(frame, 0.0) is True
    assert analyzer.submit(frame, 0.5) is False  # the one slot is busy, as in a live system
    gate.set()
    analyzer.close()
    assert (analyzer.sent, analyzer.skipped, len(analyzer.history)) == (1, 1, 1)


def test_out_of_order_answers_keep_newest_on_screen():
    """If the answer for an older frame arrives after a newer one, the screen keeps the newer result."""
    analyzer = demo.Analyzer(FakeClient(), max_in_flight=2)
    newer, older = np.full((4, 4, 3), 150, np.uint8), np.full((4, 4, 3), 0, np.uint8)
    for frame, position in ((newer, 1.5), (older, 0.5)):  # answers land newest-first
        analyzer._slots.acquire()
        analyzer._analyze(frame, position)
    analyzer.close()
    assert analyzer.result_position == 1.5
    assert [d.hazard for d in analyzer.result.detections] == ["spill"]
    assert len(analyzer.history) == 2


def test_cli_requires_api_key(monkeypatch, clip):
    monkeypatch.delenv("PERCEPTRON_API_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(demo, "load_env", lambda: None)
    assert demo.main([clip, "--no-window"]) == 2


def test_cli_streams_a_file(monkeypatch, clip, capsys):
    monkeypatch.setenv("PERCEPTRON_API_KEY", "test")
    monkeypatch.setattr(demo, "load_env", lambda: None)
    monkeypatch.setattr(demo, "IsaacClient", lambda **kwargs: SimpleNamespace(model="fake", detect=FakeClient().detect))
    assert demo.main([clip, "--no-window"]) == 0
    out = capsys.readouterr().out
    assert "Streaming" in out and "sending 2 frames/sec" in out and "Video finished." in out
