# write_to_db.py
import os
from datetime import timedelta
import psycopg
from dotenv import load_dotenv

# Connect once and reuse the connection for every write
load_dotenv()
conn = psycopg.connect(os.environ["TIGER_DATABASE_URL"], autocommit=True)


def save_incident(hazard_type, time, duration, image, location, is_open):
    """Save one hazard the camera reported. Returns its id so we can update it later."""
    end_time = time + timedelta(seconds=duration)

    incident_id = conn.execute(
        """
        INSERT INTO incidents (type, location, started_at, ended_at, duration_s, snapshot, is_open)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING incident_id
        """,
        (hazard_type, location, time, end_time, duration, image, is_open),
    ).fetchone()[0]

    return incident_id


def save_manager_reply(incident_id, said_yes):
    """Record whether the manager confirmed (Y) or rejected (N) the alert."""
    status = "confirmed" if said_yes else "rejected"
    conn.execute(
        "UPDATE incidents SET status = %s WHERE incident_id = %s",
        (status, incident_id),
    )