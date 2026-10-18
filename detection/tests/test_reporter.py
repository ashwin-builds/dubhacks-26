from datetime import timedelta

from detection.incidents import ClassifiedFrame, Danger, IncidentDetector
from detection.reporter import IncidentReporter
from detection.store import StoreError, StubIncidentStore
from detection.tests.test_incidents import T0, frames


def make_reporter(store):
    return IncidentReporter(IncidentDetector(), store, location="prep_line", compress=lambda f: f"jpeg:{f}".encode())


def run(reporter, results):
    events = []
    for result in results:
        events += reporter.process(result)
    return events


def test_first_write_on_open_second_write_on_resolve():
    store = StubIncidentStore()
    reporter = make_reporter(store)
    run(reporter, frames("1" * 70 + "0" * 20))
    assert [c[0] for c in store.calls] == ["create", "complete"]
    row = store.rows[1]
    assert row["hazard_type"] == "spill"
    assert row["time"] == T0
    assert row["image_as_binary"] == b"jpeg:img0"
    assert row["location"] == "prep_line"
    assert row["duration"] == 35
    assert row["completed"] is True
    assert row["false_positive"] is None  # left for the manager


def test_row_is_not_completed_while_the_incident_is_open():
    store = StubIncidentStore()
    reporter = make_reporter(store)
    run(reporter, frames("1" * 50))
    assert store.rows[1]["completed"] is False and store.rows[1]["duration"] is None


def test_false_positive_is_completed_and_force_stopped():
    store = StubIncidentStore()
    reporter = make_reporter(store)
    results = frames("1" * 200 + "0" * 20)
    events = run(reporter, results[:40])  # opens at frame 39
    client_id = events[0].report.client_incident_id
    db_id = reporter.db_id(client_id)
    store.rows[db_id]["false_positive"] = True  # the manager says no
    events = run(reporter, results[40:])
    reporter.finish()
    assert [(e.kind, e.report.client_incident_id) for e in events] == [("rejected", client_id)]
    assert store.rows[db_id]["completed"] is True
    assert store.rows[db_id]["duration"] is None  # force-stopped, not resolved
    assert [c[0] for c in store.calls] == ["create", "stop"]
    assert reporter.db_id(client_id) is None


def test_unmarked_incidents_keep_going():
    store = StubIncidentStore()
    reporter = make_reporter(store)
    events = run(reporter, frames("1" * 40))
    store.rows[reporter.db_id(events[0].report.client_incident_id)]["false_positive"] = False  # confirmed real
    assert run(reporter, frames("1" * 60, start=40)) == []
    assert reporter.detector.open_incidents()


def test_only_marked_incidents_are_stopped():
    store = StubIncidentStore()
    detector = IncidentDetector()
    reporter = IncidentReporter(detector, store, location="prep_line", compress=lambda f: b"")
    both = frozenset({Danger.SPILLS, Danger.BURNERS})
    run(reporter, [ClassifiedFrame(T0 + timedelta(seconds=i * 0.5), both, "img") for i in range(40)])
    spill = next(r for r in detector.open_incidents() if r.danger is Danger.SPILLS)
    store.rows[reporter.db_id(spill.client_incident_id)]["false_positive"] = True
    events = run(reporter, [ClassifiedFrame(T0 + timedelta(seconds=i * 0.5), both, "img") for i in range(40, 42)])
    assert [(e.kind, e.report.danger) for e in events] == [("rejected", Danger.SPILLS)]
    assert [r.danger for r in detector.open_incidents()] == [Danger.BURNERS]


class FlakyStore(StubIncidentStore):
    def __init__(self, create_failures=0, check_failures=0):
        super().__init__()
        self.create_failures = create_failures
        self.check_failures = check_failures
        self.checks = 0

    def create_incident(self, *args):
        if self.create_failures:
            self.create_failures -= 1
            raise StoreError("connection lost")
        return super().create_incident(*args)

    def stop_false_positives(self, incident_ids):
        self.checks += 1
        if self.check_failures:
            self.check_failures -= 1
            raise StoreError("connection lost")
        return super().stop_false_positives(incident_ids)


def test_failed_writes_retry_in_order():
    store = FlakyStore(create_failures=25)
    reporter = make_reporter(store)
    run(reporter, frames("1" * 70 + "0" * 20))
    # The insert failed on frames 39..63; the completion at frame 89 queued behind it.
    assert [c[0] for c in store.calls] == ["create", "complete"]
    assert store.rows[1]["completed"] is True


def test_incidents_without_a_row_are_not_checked():
    store = FlakyStore(create_failures=1000)  # the insert never gets through
    reporter = make_reporter(store)
    run(reporter, frames("1" * 100))
    assert store.checks == 0


def test_failed_check_is_retried_next_time():
    store = FlakyStore(check_failures=1)
    reporter = make_reporter(store)
    results = frames("1" * 100)
    events = run(reporter, results[:40])
    store.rows[reporter.db_id(events[0].report.client_incident_id)]["false_positive"] = True
    events = run(reporter, results[40:])
    # The 20 s check failed; the 30 s one found the false positive.
    assert [e.kind for e in events] == ["rejected"]
    assert store.checks == 2
