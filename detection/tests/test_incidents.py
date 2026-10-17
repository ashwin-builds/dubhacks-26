from datetime import datetime, timedelta, timezone

from detection.incidents import Danger, FrameResult, IncidentDetector

T0 = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)


def frames(pattern, danger=Danger.SPILLS, start=0):
    """Build 2 fps frames from a string: '1' = danger present, '0' = absent."""
    out = []
    for i, ch in enumerate(pattern, start=start):
        present = ch == "1"
        out.append(
            FrameResult(
                timestamp=T0 + timedelta(seconds=i * 0.5),
                dangers=frozenset({danger}) if present else frozenset(),
                frame=f"img{i}" if present else None,
            )
        )
    return out


def feed(detector, results):
    """Return (frame index, event) for every event, without flushing."""
    hits = []
    for i, result in enumerate(results):
        for event in detector.update(result):
            hits.append((i, event))
    return hits


def test_sustained_danger_opens_once_window_is_full():
    hits = feed(IncidentDetector(), frames("1" * 70))
    assert len(hits) == 1
    index, event = hits[0]
    assert index == 59  # 60th frame = 30 s
    assert event.kind == "opened"
    report = event.report
    assert report.danger is Danger.SPILLS
    assert report.hazard_type == "spill"
    assert report.camera_id == "camera_1"
    assert report.started_at == T0
    assert report.first_frame == "img0"
    assert report.duration_seconds == 30.0


def test_exactly_80_percent_opens_but_79_does_not():
    # 48 of 60 frames: opens.
    assert feed(IncidentDetector(), frames("0" * 12 + "1" * 48))
    # 47 of 60: never opens, even with gaps spread out.
    assert not feed(IncidentDetector(), frames("0" * 13 + "1" * 47))


def test_intermittent_detection_still_counts():
    # Flickering detection: 4 on, 1 off -> exactly 80%.
    hits = feed(IncidentDetector(), frames("11110" * 12))
    assert [e.kind for _, e in hits] == ["opened"]
    assert hits[0][1].report.started_at == T0


def test_short_exposure_never_opens():
    detector = IncidentDetector()
    assert detector.run(frames("0" * 10 + "1" * 40 + "0" * 60)) == []


def test_start_is_first_danger_frame_in_window():
    # Danger shows up at frame 20; earlier frames are clean.
    results = frames("0" * 20 + "1" * 60)
    hits = feed(IncidentDetector(), results)
    report = hits[0][1].report
    assert hits[0][0] == 20 + 47  # 48th danger frame
    assert report.started_at == results[20].timestamp
    assert report.first_frame == "img20"


def test_close_reports_full_duration():
    # 90 danger frames (45 s), then gone.
    detector = IncidentDetector()
    events = feed(detector, frames("1" * 90 + "0" * 30))
    assert [e.kind for _, e in events] == ["opened", "closed"]
    closed = events[1][1].report
    assert closed.duration_seconds == 45.0
    assert closed.ended_at == T0 + timedelta(seconds=45)
    assert closed.client_incident_id == events[0][1].report.client_incident_id
    assert detector.open_incidents() == []


def test_brief_gaps_do_not_close_an_open_incident():
    detector = IncidentDetector()
    events = feed(detector, frames("1" * 60 + "0" * 19 + "1" * 20))
    assert [e.kind for _, e in events] == ["opened"]


def test_flush_closes_open_incident_at_end_of_stream():
    events = IncidentDetector().run(frames("1" * 70))
    assert [e.kind for e in events] == ["opened", "closed"]
    assert events[1].report.duration_seconds == 35.0


def test_does_not_reopen_off_stale_frames():
    # Close, then a short re-appearance must not reopen immediately.
    detector = IncidentDetector()
    events = feed(detector, frames("1" * 60 + "0" * 20 + "1" * 20))
    assert [e.kind for _, e in events] == ["opened", "closed"]


def test_dangers_are_tracked_independently():
    detector = IncidentDetector()
    events = []
    for i in range(60):
        dangers = {Danger.SHARPS}
        if i >= 30:
            dangers.add(Danger.BURNERS)
        events += detector.update(
            FrameResult(T0 + timedelta(seconds=i * 0.5), frozenset(dangers), f"img{i}")
        )
    assert [(e.kind, e.report.danger) for e in events] == [("opened", Danger.SHARPS)]


def test_hazard_type_mapping():
    assert {d: d.hazard_type for d in Danger} == {
        Danger.SHARPS: "knife_edge",
        Danger.BURNERS: "unattended_burner",
        Danger.SPILLS: "spill",
        Danger.CONTAMINATION: "raw_meat_contact",
    }
