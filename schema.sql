CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============ STATIC: kitchen layout (set once) ============
CREATE TABLE IF NOT EXISTS zones (
    zone_id         SERIAL PRIMARY KEY,
    location        TEXT NOT NULL,
    category        TEXT NOT NULL,
    description     TEXT,
    camera_id       TEXT NOT NULL DEFAULT 'cam1',
    polygon         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    manager_id_list TEXT
);

CREATE TABLE IF NOT EXISTS people (
    location     TEXT,
    manager_id   TEXT PRIMARY KEY,
    manager_info TEXT,
    zone_id      INT REFERENCES zones(zone_id)
);

-- ============ LIVE: one row per hazard incident ============
CREATE TABLE IF NOT EXISTS incidents (
    incident_id     BIGINT GENERATED ALWAYS AS IDENTITY,
    wrote_at        TIMESTAMPTZ NOT NULL,
    start_time      TIMESTAMPTZ NOT NULL,
    stop_time       TIMESTAMPTZ,                       -- NULL while hazard is still open
    classified_at   TIMESTAMPTZ,
    zone_id         INT NOT NULL REFERENCES zones(zone_id),
    peak_confidence REAL,
    snapshot_url    TEXT,
    status          TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open','closed')),
    verdict         TEXT NOT NULL DEFAULT 'pending'
                    CHECK (verdict IN ('pending','confirmed','rejected')),
    alert_sent_at   TIMESTAMPTZ,
    verified_at     TIMESTAMPTZ,
    staff_notes     TEXT,                              -- typed by manager, never camera
    PRIMARY KEY (incident_id, start_time)
);

SELECT create_hypertable('incidents', by_range('start_time', INTERVAL '7 days'),
                         if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS incidents_id_idx      ON incidents (incident_id);
CREATE INDEX IF NOT EXISTS incidents_open_idx    ON incidents (zone_id, classified_at DESC)
                                                  WHERE status = 'open';
CREATE INDEX IF NOT EXISTS incidents_verdict_idx ON incidents (verdict, alert_sent_at DESC);