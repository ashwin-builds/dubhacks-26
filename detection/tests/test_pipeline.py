from datetime import datetime, timezone

import numpy as np

from detection import demo
from detection.reporter import IncidentReporter
from detection.store import StubIncidentStore
from detection.incidents import Danger, IncidentDetector
from detection.isaac_client import Detection, FrameResult
from detection.pipeline import IncidentPipeline
from detection.tests.test_video_source import FakeClient, _stream, clip  # noqa: F401 - clip is a fixture

T0 = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)
FRAME = np.zeros((48, 64, 3), np.uint8)


def result(*hazards):
    return FrameResult([Detection(h, h, (0, 0, 10, 10)) for h in hazards], [], 0.1, None)


def make_pipeline(seconds=2.0, fps=2.0):
    """Small windows so tests stay short: 2 s at 2 fps = 4 frames, 80% = 3."""
    store = StubIncidentStore()
    detector = IncidentDetector(fps=fps, danger_seconds={d: seconds for d in Danger}, clear_after_seconds=1.0)
    events = []
    pipeline = IncidentPipeline(
        IncidentReporter(detector, store, location="prep_line", compress=lambda f: b"jpeg"),
        fps=fps,
        started_at=T0,
        on_event=lambda e, p: events.append(e),
    )
    return pipeline, store, events


def test_out_of_order_answers_are_fed_in_video_order():
    pipeline, store, events = make_pipeline()
    # Answers land 3, 1, 0, 2: nothing can be processed until 0 arrives.
    for seq in (3, 1, 0, 2):
        pipeline.deliver(seq, seq * 0.5, FRAME, result("spill"))
    pipeline.close()
    assert [e.kind for e in events] == ["opened", "closed"]
    opened = events[0].report
    assert opened.started_at == T0
    assert opened.danger is Danger.SPILLS
    assert events[1].report.duration_seconds == 2.0
    assert events[1].report.missing_frames == 0
    assert store.rows[1]["location"] == "prep_line"
    assert store.rows[1]["hazard_type"] == "spill" and store.rows[1]["completed"] is True


def test_failed_calls_become_missing_frames():
    pipeline, _, events = make_pipeline()
    for seq in range(6):
        pipeline.deliver(seq, seq * 0.5, FRAME, None if seq == 2 else result("spill"))
    pipeline.close()
    assert [e.kind for e in events] == ["opened", "closed"]
    assert events[1].report.missing_frames == 1


def test_samples_that_never_return_do_not_stall_the_pipeline():
    pipeline, _, events = make_pipeline()
    for seq in (0, 1, 3, 4, 5):  # seq 2 was cancelled when the user quit
        pipeline.deliver(seq, seq * 0.5, FRAME, result("spill"))
    assert events == []  # 3, 4, 5 are held waiting for 2
    pipeline.close()
    assert [e.kind for e in events] == ["opened", "closed"]


def test_looping_video_keeps_time_moving_forward():
    pipeline, _, events = make_pipeline()
    positions = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]  # a 1.5 s video played twice
    for seq, pos in enumerate(positions):
        pipeline.deliver(seq, pos, FRAME, result("spill"))
    pipeline.close()
    closed = events[-1].report
    assert closed.duration_seconds == 3.0
    assert closed.missing_frames == 0


def test_first_frame_has_boxes_drawn():
    pipeline, store, _ = make_pipeline()
    pipeline.reporter.compress = lambda f: f  # keep the raw frame to inspect
    for seq in range(4):
        pipeline.deliver(seq, seq * 0.5, FRAME, result("spill"))
    pipeline.close()
    first = store.rows[1]["image_as_binary"]
    assert first.shape == FRAME.shape and first.any()  # the box was drawn on a black frame


def test_run_video_end_to_end(clip):  # noqa: F811
    """2 s clip at 10 fps; FakeClient sees a spill from 1 s. With a 1 s window, that is an incident."""
    store = StubIncidentStore()
    events = []
    detector = IncidentDetector(fps=2, danger_seconds={d: 1.0 for d in Danger})
    pipeline = IncidentPipeline(
        IncidentReporter(detector, store, location="prep_line", compress=lambda f: b"jpeg"),
        fps=2,
        started_at=T0,
        on_event=lambda e, p: events.append(e),
    )
    assert demo.run_video(FakeClient(), _stream(clip), fps=2, show=False, pipeline=pipeline) == 0
    assert [(e.kind, e.report.danger) for e in events] == [("opened", Danger.SPILLS), ("closed", Danger.SPILLS)]
    assert pipeline.video_time(events[0].report.started_at) == 1.0
    assert events[1].report.duration_seconds == 1.0
    assert [c[0] for c in store.calls] == ["create", "complete"]
    assert store.rows[1]["duration"] == 1 and store.rows[1]["false_positive"] is None


def test_annotate_shows_open_incidents():
    out = demo.annotate(FRAME.copy(), result(), incidents=[Danger.BURNERS])
    assert out[5, 5].tolist() == [0, 0, 200]  # red banner even though this frame is clear


def boxed(*items):
    """A model result with (hazard, box) detections."""
    return FrameResult([Detection(h, h, box) for h, box in items], [], 0.1, None)


KNIFE = (100, 100, 200, 200)
ELSEWHERE = (600, 600, 700, 700)


def test_false_positive_spot_is_ignored_afterwards():
    """A knife opens an incident; the manager says no; the same knife never opens another one."""
    pipeline, store, events = make_pipeline()  # 2 s window, checks every 10 s
    seq = 0

    def feed(n, *items):
        nonlocal seq
        for _ in range(n):
            pipeline.deliver(seq, seq * 0.5, FRAME, boxed(*items))
            seq += 1
        pipeline._worker.submit(lambda: None).result()  # wait for the worker to catch up

    feed(4, ("knife_edge", KNIFE))  # opens at 1.5 s
    incident_id = events[0].report.client_incident_id
    store.rows[pipeline.reporter.db_id(incident_id)]["false_positive"] = True  # the manager says no
    feed(20, ("knife_edge", KNIFE))  # the 10 s check rejects it
    assert [e.kind for e in events] == ["opened", "rejected"]
    assert events[1].report.boxes == (KNIFE,)
    assert pipeline.suppressor.zones and pipeline.suppressor.zones[0].box == KNIFE

    # The knife stays for another 60 s, with the box jittering a little: nothing new opens.
    feed(120, ("knife_edge", (105, 98, 210, 204)))
    assert [e.kind for e in events] == ["opened", "rejected"]
    assert pipeline.suppressor.suppressed > 100

    # A different knife elsewhere in the frame is still a real incident.
    feed(4, ("knife_edge", KNIFE), ("knife_edge", ELSEWHERE))
    pipeline.close()
    assert [e.kind for e in events] == ["opened", "rejected", "opened", "closed"]
    assert events[2].report.client_incident_id != incident_id


def test_suppression_does_not_hide_other_hazards_in_the_same_spot():
    pipeline, store, events = make_pipeline()
    pipeline.suppressor.add(Danger.SHARPS, (KNIFE,), T0, "earlier")
    for seq in range(4):
        pipeline.deliver(seq, seq * 0.5, FRAME, boxed(("knife_edge", KNIFE), ("spill", KNIFE)))
    pipeline.close()
    assert [(e.kind, e.report.danger) for e in events] == [("opened", Danger.SPILLS), ("closed", Danger.SPILLS)]
