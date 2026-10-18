from datetime import datetime, timezone

import pytest

from detection.store import (
    COMPLETE_SQL,
    INSERT_SQL,
    STOP_FALSE_POSITIVES_SQL,
    ZONE_SQL,
    PostgresIncidentStore,
    StoreError,
)

T0 = datetime(2026, 10, 18, 18, 0, tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, answers, fail=None):
        self.answers = answers  # sql -> rows
        self.executed = []
        self.closed = False
        self.broken = False
        self.fail = fail

    def execute(self, sql, params):
        if self.fail:
            self.broken = True
            raise self.fail
        self.executed.append((sql, params))
        return FakeCursor(self.answers.get(sql, []))


def make_store(conns):
    opened = []

    def connect(url, autocommit):
        assert autocommit and url == "postgres://test"
        conn = conns[len(opened)]
        opened.append(conn)
        return conn

    return PostgresIncidentStore("postgres://test", connect=connect), opened


def test_insert_writes_the_first_set_of_fields():
    conn = FakeConn({ZONE_SQL: [(3,)], INSERT_SQL: [(42,)]})
    store, _ = make_store([conn])
    assert store.create_incident("knife_edge", T0, b"jpeg", "prep_line") == 42
    assert conn.executed == [(ZONE_SQL, ("prep_line",)), (INSERT_SQL, (T0, "knife_edge", b"jpeg", 3))]


def test_zone_lookup_is_cached_and_unknown_zone_is_null():
    conn = FakeConn({INSERT_SQL: [(1,)]})
    store, _ = make_store([conn])
    store.create_incident("spill", T0, None, "nowhere")
    store.create_incident("spill", T0, None, "nowhere")
    assert [sql for sql, _ in conn.executed].count(ZONE_SQL) == 1
    assert conn.executed[-1][1] == (T0, "spill", None, None)


def test_complete_rounds_duration_to_whole_seconds():
    conn = FakeConn({})
    store, _ = make_store([conn])
    store.complete_incident(42, 35.4)
    assert conn.executed == [(COMPLETE_SQL, (35, 42))]


def test_stop_false_positives_marks_and_returns_in_one_statement():
    conn = FakeConn({STOP_FALSE_POSITIVES_SQL: [(7,), (9,)]})
    store, _ = make_store([conn])
    assert store.stop_false_positives([7, 8, 9]) == [7, 9]
    assert conn.executed == [(STOP_FALSE_POSITIVES_SQL, ([7, 8, 9],))]
    assert "false_positive IS TRUE" in STOP_FALSE_POSITIVES_SQL and "SET completed = TRUE" in STOP_FALSE_POSITIVES_SQL


def test_errors_are_wrapped_and_a_broken_connection_is_replaced():
    bad = FakeConn({}, fail=OSError("network down"))
    good = FakeConn({})
    store, opened = make_store([bad, good])
    with pytest.raises(StoreError, match="network down"):
        store.complete_incident(1, 5.0)
    store.complete_incident(1, 5.0)
    assert opened == [bad, good]
    assert good.executed == [(COMPLETE_SQL, (5, 1))]
