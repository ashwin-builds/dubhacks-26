# push_schema.py
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   # walks up from the script's folder until it finds .env

import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")   # .env lives in the repo root

with psycopg.connect(os.environ["TIGER_DATABASE_URL"], autocommit=True) as db:
    db.execute((Path(__file__).parent / "schema.sql").read_text())
    print("Schema pushed.")

# Run from the repo root:
#   python database/push_schema.py