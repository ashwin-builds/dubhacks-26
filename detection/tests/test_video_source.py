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


def test_speed_scales_playback(clip):
    clock = FakeClock()
    stream = VideoFileStream(clip, speed=2.0, clock=clock, sleep=clock.sleep)
    while stream.read()[0]:
        pass
    assert clock.now - 100.0 == pytest.approx(FRAMES / FPS / 2)


def test_loop_restarts(clip):
    clock = FakeClock()
    stream = VideoFileStream(clip, loop=True, clock=clock, sleep=clock.sleep)
    indices = [_frame_index(stream.read()[1]) for _ in range(FRAMES + 3)]
    assert indices[FRAMES:] == [0, 1, 2]


def test_bad_inputs(tmp_path, clip):
    with pytest.raises(OSError):
        VideoFileStream(str(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError):
        VideoFileStream(clip, speed=0)
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
    history = [(0.5, _result()), (1.0, _result("spill")), (1.5, _result("spill", "knife_edge")), (2.0, _result())]
    text = demo.summarize(history)
    assert "Analysed 4 frames." in text
    assert "Spill" in text and "2 / 4" in text and "first 00:01.0" in text and "last 00:01.5" in text
    assert "Sharp tool" in text and "Unattended burner" not in text
    assert "No hazards detected." in demo.summarize([(0.0, _result())])


def test_run_video_end_to_end(clip, capsys, monkeypatch):
    """Play the clip headless through the real loop with a fake model that sees a spill after 1 s."""
    seen_positions = []

    class FakeClient:
        def detect(self, frame):
            hazard = ("spill",) if _frame_index(frame) >= 10 else ()
            seen_positions.append(_frame_index(frame))
            return _result(*hazard)

    code = demo.run_video(FakeClient(), clip, fps=20, show=False, speed=4.0, loop=False)
    out = capsys.readouterr().out
    assert code == 0
    assert "Video finished." in out
    assert "Analysed" in out and "Spill" in out
    assert seen_positions  # the analyzer received frames
