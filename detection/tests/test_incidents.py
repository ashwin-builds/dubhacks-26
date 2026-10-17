from datetime import datetime, timedelta, timezone

from detection.incidents import Danger, ClassifiedFrame, IncidentDetector

T0 = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)


def frames(pattern, danger=Danger.SPILLS, start=0):
    """Build 2 fps frames from a string: '1' = danger present, '0' = absent."""
    out = []
    for i, ch in enumerate(pattern, start=start):
        present = ch == "1"
        out.append(
            ClassifiedFrame(
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
    assert index == 39  # 40th frame = 20 s
    assert event.kind == "opened"
    report = event.report
    assert report.danger is Danger.SPILLS
    assert report.hazard_type == "spill"
    assert report.camera_id == "camera_1"
    assert report.started_at == T0
    assert report.first_frame == "img0"
    assert report.duration_seconds == 20.0


def test_exactly_80_percent_opens_but_79_does_not():
    # 32 of 40 frames: opens.
    assert feed(IncidentDetector(), frames("0" * 8 + "1" * 32))
    # 31 of 40: never opens.
    assert not feed(IncidentDetector(), frames("0" * 9 + "1" * 31))


def test_intermittent_detection_still_counts():
    # Flickering detection: 4 on, 1 off -> exactly 80%.
    hits = feed(IncidentDetector(), frames("11110" * 12))
    assert [e.kind for _, e in hits] == ["opened"]
    assert hits[0][1].report.started_at == T0


def test_short_exposure_never_opens():
    detector = IncidentDetector()
    assert detector.run(frames("0" * 10 + "1" * 31 + "0" * 60)) == []


def test_start_is_first_danger_frame_in_window():
    # Danger shows up at frame 20; earlier frames are clean.
    results = frames("0" * 20 + "1" * 60)
    hits = feed(IncidentDetector(), results)
    report = hits[0][1].report
    assert hits[0][0] == 20 + 31  # 32nd danger frame
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
            ClassifiedFrame(T0 + timedelta(seconds=i * 0.5), frozenset(dangers), f"img{i}")
        )
    assert [(e.kind, e.report.danger) for e in events] == [("opened", Danger.SHARPS)]


def test_hazard_type_mapping():
    assert {d: d.hazard_type for d in Danger} == {
        Danger.SHARPS: "knife_edge",
        Danger.BURNERS: "unattended_burner",
        Danger.SPILLS: "spill",
        Danger.CONTAMINATION: "raw_meat_contact",
    }


def drop(results, indices):
    return [r for i, r in enumerate(results) if i not in set(indices)]


def test_missing_frames_are_counted_and_reported():
    detector = IncidentDetector()
    # 70 danger frames with 5 dropped in the middle: 65 seen, 5 missing.
    results = drop(frames("1" * 70), range(30, 35))
    events = detector.run(results)
    assert detector.missing_frames == 5
    assert [e.kind for e in events] == ["opened", "closed"]
    assert events[0].report.missing_frames == 5
    assert events[1].report.missing_frames == 5
    # Duration still spans the real time, gaps included.
    assert events[1].report.duration_seconds == 35.0


def test_missing_frames_count_against_opening():
    # 9 of the 40-frame window dropped -> only 31 positive.
    results = drop(frames("1" * 40), range(10, 19))
    assert IncidentDetector().run(results) == []


def test_jitter_is_not_counted_as_missing():
    detector = IncidentDetector()
    t = T0
    for i in range(20):
        t += timedelta(seconds=0.4 if i % 2 else 0.6)
        detector.update(ClassifiedFrame(t, frozenset()))
    assert detector.missing_frames == 0


def test_missing_frames_do_not_close_an_open_incident():
    # Opens, then the feed drops for 15 s (30 frames), then danger is still there.
    results = drop(frames("1" * 120), range(60, 90))
    events = feed(IncidentDetector(), results)
    assert [e.kind for _, e in events] == ["opened"]


def test_reject_stops_recording_and_sends_no_close():
    detector = IncidentDetector()
    results = frames("1" * 200 + "0" * 30)
    events = feed(detector, results[:60])
    incident_id = events[0][1].report.client_incident_id
    assert detector.reject(incident_id)
    assert detector.open_incidents() == []
    # Danger still visible for 70 s more: no reopen, no close, nothing.
    assert feed(detector, results[60:]) == []
    assert detector.flush() == []


def test_rejected_danger_can_open_again_after_it_clears():
    detector = IncidentDetector()
    events = feed(detector, frames("1" * 60))
    detector.reject(events[0][1].report.client_incident_id)
    # Clears (20 absent frames), then a genuinely new 30 s exposure.
    later = feed(detector, frames("0" * 20 + "1" * 60, start=60))
    assert [e.kind for _, e in later] == ["opened"]
    assert later[0][1].report.client_incident_id != events[0][1].report.client_incident_id


def test_reject_unknown_or_closed_incident_returns_false():
    detector = IncidentDetector()
    events = detector.run(frames("1" * 70))
    assert not detector.reject(events[0].report.client_incident_id)
    assert not detector.reject("nope")
