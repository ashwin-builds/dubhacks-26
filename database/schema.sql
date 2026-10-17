CREATE EXTENSION IF NOT EXISTS timescaledb;
DROP TABLE IF EXISTS incidents CASCADE;

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
    time            TIMESTAMPTZ NOT NULL,
    duration        INTEGER,
    image_as_binary BYTEA,
    perceptron_desc TEXT,
    false_postitive BOOLEAN DEFAULT NULL,
    zone_id         INT DEFAULT NULL,
    expo_notes     TEXT DEFAULT NULL,       -- typed by expo, never camera
    PRIMARY KEY (incident_id, time)     -- hypertables need the time column in the key
);

SELECT create_hypertable('incidents', by_range('time', INTERVAL '7 days'),
                         if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS incidents_id_idx   ON incidents (incident_id);
CREATE INDEX IF NOT EXISTS incidents_zone_idx ON incidents (zone_id, time DESC);