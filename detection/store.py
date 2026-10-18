"""Write incidents straight to the ``incidents`` table (see ``database/schema.sql``).

Each incident row is written in two steps:

1. When the incident opens: hazard type, start time, compressed first frame,
   and zone. ``completed`` is false; the database assigns ``incident_id``.
2. Either when it resolves: the duration, and ``completed = true``
   (``false_positive`` is left alone), or when the manager has marked it as a
   false positive: ``completed = true``, and the worker stops recording it.

Step 2's false-positive path is found by asking the database, every 10 s,
which of our open incidents are ``false_positive = true`` and not completed.
One ``UPDATE ... RETURNING`` both finds them and marks them completed, so an
incident can never be stopped without being marked, or marked twice.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Columns this worker writes. ``hazard_type`` and ``completed`` are not in database/schema.sql yet.
INSERT_SQL = """
    INSERT INTO incidents (time, hazard_type, image_as_binary, zone_id, completed)
    VALUES (%s, %s, %s, %s, FALSE)
    RETURNING incident_id
"""
COMPLETE_SQL = """
    UPDATE incidents SET duration = %s, completed = TRUE
    WHERE incident_id = %s
"""
STOP_FALSE_POSITIVES_SQL = """
    UPDATE incidents SET completed = TRUE
    WHERE incident_id = ANY(%s) AND false_positive IS TRUE AND completed IS NOT TRUE
    RETURNING incident_id
"""
ZONE_SQL = "SELECT zone_id FROM zones WHERE location = %s"


class StoreError(Exception):
    """A database write or read failed; the caller retries later."""


class IncidentStore(Protocol):
    def create_incident(self, hazard_type: str, started_at: datetime, image: bytes | None, location: str) -> int: ...
    def complete_incident(self, incident_id: int, duration_seconds: float) -> None: ...
    def stop_false_positives(self, incident_ids: list[int]) -> list[int]: ...


class PostgresIncidentStore:
    """Tiger Data (Postgres) via psycopg. Reconnects on the next call if the connection drops."""

    def __init__(self, url: str | None = None, connect: Any = None):
        self.url = url or os.environ["TIGER_DATABASE_URL"]
        if connect is None:
            import psycopg

            connect = psycopg.connect
        self._connect = connect
        self._conn = None
        self._zone_ids: dict[str, int | None] = {}

    def create_incident(self, hazard_type: str, started_at: datetime, image: bytes | None, location: str) -> int:
        zone_id = self._zone_id(location)
        return self._execute(INSERT_SQL, (started_at, hazard_type, image, zone_id)).fetchone()[0]

    def complete_incident(self, incident_id: int, duration_seconds: float) -> None:
        self._execute(COMPLETE_SQL, (round(duration_seconds), incident_id))  # duration is an INTEGER column

    def stop_false_positives(self, incident_ids: list[int]) -> list[int]:
        return [row[0] for row in self._execute(STOP_FALSE_POSITIVES_SQL, (incident_ids,)).fetchall()]

    def _zone_id(self, location: str) -> int | None:
        if location not in self._zone_ids:
            row = self._execute(ZONE_SQL, (location,)).fetchone()
            self._zone_ids[location] = row[0] if row else None
            if row is None:
                log.warning("zone %r is not in the zones table; incidents get zone_id NULL", location)
        return self._zone_ids[location]

    def _execute(self, sql: str, params: tuple) -> Any:
        try:
            if self._conn is None or self._conn.closed:
                self._conn = self._connect(self.url, autocommit=True)
            return self._conn.execute(sql, params)
        except Exception as exc:  # psycopg.Error, or the network under it
            if self._conn is not None and getattr(self._conn, "broken", False):
                self._conn = None  # reconnect next time
            raise StoreError(f"{type(exc).__name__}: {exc}") from exc


class StubIncidentStore:
    """In-memory stand-in, for tests and demos without a database.

    Set ``rows[id]["false_positive"] = True`` to play a manager answering "no".
    """

    def __init__(self) -> None:
        self.rows: dict[int, dict[str, Any]] = {}
        self.calls: list[tuple[str, int]] = []

    def create_incident(self, hazard_type: str, started_at: datetime, image: bytes | None, location: str) -> int:
        incident_id = len(self.rows) + 1
        self.rows[incident_id] = {
            "time": started_at,
            "hazard_type": hazard_type,
            "image_as_binary": image,
            "location": location,
            "duration": None,
            "false_positive": None,
            "completed": False,
        }
        self.calls.append(("create", incident_id))
        log.info("stub create incident %d: %s at %s", incident_id, hazard_type, location)
        return incident_id

    def complete_incident(self, incident_id: int, duration_seconds: float) -> None:
        self.rows[incident_id].update(duration=round(duration_seconds), completed=True)
        self.calls.append(("complete", incident_id))
        log.info("stub complete incident %d after %.1fs", incident_id, duration_seconds)

    def stop_false_positives(self, incident_ids: list[int]) -> list[int]:
        stopped = [
            i for i in incident_ids if self.rows[i]["false_positive"] is True and not self.rows[i]["completed"]
        ]
        for i in stopped:
            self.rows[i]["completed"] = True
            self.calls.append(("stop", i))
        return stopped
