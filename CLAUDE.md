# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ownership and boundaries

Each agent or session works within its assigned area:

| Owner | Folder(s) |
|-------|-----------|
| Zach | `detection/` |
| Alex | `database/` |
| Ashwin | `website/` |

**Never modify files outside your assigned folder without explicit instruction from the file's owner.** Changes to
shared files (this file, the root `README.md`, `.gitignore`) need agreement from all three owners.

## Commands

```bash
# Create and activate the virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (each area has its own requirements file)
pip install -r detection/requirements.txt
pip install -r website/requirements.txt

# Environment variables live in .env at the repo root (git-ignored):
#   PERCEPTRON_API_KEY, TIGER_DATABASE_URL, optional PERCEPTRON_MODEL and ISAAC_QUERIES_PER_SEC

# Push the database schema (run from the repo root)
python database/push_schema.py

# Run detection on a video, played as a live 2 fps feed
python -m detection.demo clip.mp4                 # print incidents only
python -m detection.demo clip.mp4 --write-db      # also write them to TIGER_DATABASE_URL
python -m detection.demo clip.mp4 --no-window --loop
python -m detection.demo --list-models

# Tests (fake model and in-memory database: no API key, database, or network needed)
pytest detection
pytest detection/tests/test_incidents.py
pytest detection/tests/test_incidents.py::test_sustained_danger_opens_once_window_is_full
```

## Architecture

There is no API service. The detection worker writes incidents straight into the `incidents` table, and the
website reads that table. The database is the only thing the two sides share.

```
video file / URL (played in real time)
        │  frame every 0.5 s
        ▼
detection/  Perceptron (Isaac) labels each frame ──> incident detector (sliding window)
        │
        │  insert on open, update on resolve
        ▼
database/  incidents table (Tiger Data / Postgres + TimescaleDB)
        ▲                                   │
        │ false_positive = true             │ reads
        │                                   ▼
   manager / expo review  <──────────  website/  (FastAPI + Jinja + Altair)
```

### Detection pipeline (`detection/`)

- `demo.py` plays a video in real time and sends the current frame to Perceptron every 0.5 s. Up to 4 calls run
  at once; if all 4 are busy, that sample is skipped, as a live system would do.
- `isaac_client.py` and `hazards.py` hold the prompt. Each frame is judged on its own and returns boxes labelled
  with hazard keys. The model holds no memory between calls, so anything it should know goes in the prompt.
- `pipeline.py` puts out-of-order model answers back in video order. Frame timestamps are stream start time plus
  video time, so durations come from video time, not model latency.
- `incidents.py` decides when a hazard becomes an incident. The model never makes that decision.
- `reporter.py` turns incident events into database writes; `store.py` holds the SQL.
- `suppression.py` ignores spots the manager marked as false positives.

### Hazards

| Key (`hazard_type`) | Flagged when |
|---------------------|--------------|
| `knife_edge` | a sharp tool is set down with no one holding it or working at it (a knife in use is fine) |
| `unattended_burner` | a burner is on with no one at the stove tending it (cooking is fine) |
| `spill` | liquid, grease or food is on the floor or a work surface |
| `raw_meat_contact` | raw meat, poultry or fish touches cooked or ready-to-eat food |

To change what the model looks for, edit the hazard's `description` in `detection/hazards.py`. No retraining.

### Incident rules

- **Open**: at least 80% of the last 40 analysed frames (20 s at 2 fps) show the hazard. The incident's start time
  and image come from the first hazard frame in that window. The 20 s is hardcoded for every hazard
  (`DANGER_SECONDS` in `detection/incidents.py`).
- **Resolve**: the hazard has been gone for 10 s in a row.
- **Missing frames** (model call failed, or the sample was skipped) are found from timestamp gaps. They count as "no
  hazard" toward opening, but don't count toward resolving.
- **False positive**: the incident is force-stopped. The boxes it was last seen in become ignore zones for that
  hazard: detections that mostly overlap them are dropped. A zone ends after 30 minutes, or once nothing has
  matched it for 60 s. The same hazard elsewhere in the frame, and other hazards in the same spot, still count.

### Incident writes

Each incident is one row in `incidents`, written in two steps by `detection/store.py`:

1. **On open**: insert `time`, `hazard_type`, `image_as_binary` (first frame with boxes drawn, JPEG at most 640 px),
   `zone_id` (looked up from the camera's location in `zones`), and `completed = false`. The database assigns
   `incident_id`.
2. One of:
   - **Resolved**: set `duration` (whole seconds) and `completed = true`. `false_positive` is not touched.
   - **False positive**: every 10 s, detection runs one `UPDATE ... RETURNING` that sets `completed = true` on
     whichever of its open incidents have `false_positive = true`, then force-stops them. `duration` stays null.

Columns the website and review flow own: `false_positive` (set by the manager or expo), `expo_notes`,
`perceptron_desc`. Detection never writes them.

A row with `completed = false` is an incident still in progress. `false_positive` is `NULL` until someone reviews
the incident.

Store all timestamps as UTC `timestamptz`; render in kitchen-local time on the website.

## Open items

- **Schema**: `database/schema.sql` lacks two columns detection writes:
  ```sql
  ALTER TABLE incidents ADD COLUMN IF NOT EXISTS hazard_type TEXT;
  ALTER TABLE incidents ADD COLUMN IF NOT EXISTS completed BOOLEAN NOT NULL DEFAULT FALSE;
  ```
- **`database/write_to_db.py`**: its insert has 6 placeholders for 5 values, and `save_expo_reply` references a
  `staff_notes` column that doesn't exist. Detection doesn't use this file.
- **Camera zone**: `camera_1` maps to the placeholder location `prep_line` (`CAMERA_LOCATIONS` in
  `detection/reporter.py`). It must match a `zones.location` value, or `zone_id` is written as `NULL`.
- **Placeholders**: the 20 s danger window, the 30 min and 60 s ignore-zone lifetimes (`detection/suppression.py`).
- **Model speed**: if Perceptron takes longer than about 2 s per frame, samples get skipped. Over 20% skipped and an
  incident can never reach 80%.

## Git and GitHub workflow

**Never push to the remote repository.** The humans review and push all changes themselves.

When your work is ready:

1. Stage only the files you changed (avoid `git add -A`):
   ```bash
   git add detection/incidents.py detection/tests/test_incidents.py
   ```

2. Commit with a clear message. Use this format so authorship is attributed to the human, not Claude:
   ```bash
   git commit -m "feat(detection): add incident detector"
   ```
   Do not append `Co-Authored-By: Claude` lines — the owner wants commits to appear as theirs on GitHub.

3. Report what you committed and what still needs review. Do not run `git push`.

4. If a pull request is needed, prepare the PR description and post it in the chat so the owner can paste it into GitHub:
   ```
   Title: <short imperative title>

   ## What changed
   - <bullet>

   ## How to test
   - <bullet>

   Base branch: main
   ```
   The human will open the PR, request reviews, and merge — not Claude.
