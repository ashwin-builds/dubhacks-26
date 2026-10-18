# write_to_db.py
import os

import psycopg
from dotenv import load_dotenv

# Connect once and reuse the connection for every write
load_dotenv()
conn = psycopg.connect(os.environ["TIGER_DATABASE_URL"], autocommit=True)


def get_zone_id(location):
    """Turn a zone name like 'Fryer' into its zone_id. Returns None if unknown."""
    row = conn.execute(
        "SELECT zone_id FROM zones WHERE location = %s",
        (location,),
    ).fetchone()
    return row[0] if row else None


def save_incident(hazard_type, time, duration, image, location, description=None):
    """Save one hazard the camera reported. Returns its id so we can update it later."""
    incident_id = conn.execute(
        """
        INSERT INTO incidents (time, duration, image_as_binary, perceptron_desc, zone_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING incident_id
        """,
        (time, duration, image, description, get_zone_id(location)),
    ).fetchone()[0]

    return incident_id


def save_expo_reply(incident_id, said_yes, notes=None):
    """Record the expo's verdict: Y = real hazard, N = false alarm."""
    conn.execute(
        """
        UPDATE incidents
        SET false_positive = %s,
            expo_notes     = coalesce(%s, staff_notes)
        WHERE incident_id = %s
        """,
        (not said_yes, notes, incident_id),
    )