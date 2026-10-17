import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

with psycopg.connect(os.environ["TIGER_DATABASE_URL"], autocommit=True) as db:
    db.execute(open("schema.sql").read())
    print("Schema pushed.")

#use this bash commmand:
# pip install "psycopg[binary]" python-dotenv && python push_schema.py