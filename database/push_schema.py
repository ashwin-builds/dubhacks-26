# push_schema.py
import os
from pathlib import Path
import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

with psycopg.connect(os.environ["TIGER_DATABASE_URL"], autocommit=True) as db:
    db.execute(open("schema.sql").read())
    print("Schema pushed.")
#use following bash command in termal
# echo 'TIGER_DATABASE_URL=postgres://tsdbadmin:dokxk5kumjn4ygw0@x55gag554r.jxejnexvsk.tsdb.cloud.timescale.com:32348/tsdb?sslmode=require' > .env && python push_schema.py