# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ownership and boundaries

This repo has three owners. Each agent or session works within its assigned area:

| Owner | Folder(s) |
|-------|-----------|
| Zach | `detection/` |
| Ashwin | `alerts/`, `app/` |
| Alex | `db/` |
| All three | `shared/` |

**Never modify files outside your assigned folder without explicit instruction from the file's owner.** Changes to `shared/` require agreement from all three owners before committing.

## Commands

```bash
# Create and activate the virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env

# Run all tests
pytest

# Run a single test file
pytest tests/test_<module>.py

# Run a single test by name
pytest tests/test_<module>.py::test_<name>

# Run with DB stub (no real database needed)
DB_BACKEND=stub python -m <module>
```

Set `DB_BACKEND=stub` in `.env` to develop and test without a live Tiger Data connection.

## Architecture

```
webcam → detection/ → incident → db/ (pending)
                          │
                          └→ alerts/ → manager phone →Y/N→ db/ (confirmed | rejected)
                                                               │
                                           app/ dashboard ←───┘
```

**`detection/`** runs a webcam loop at ~2 fps, calls the Perceptron Isaac vision model, maps bounding boxes to zones via `shared/zones.yaml`, debounces repeated detections into a single `Incident`, saves an annotated snapshot, and calls `db.create_incident`.

**`alerts/`** receives a new incident, sends one SMS via Photon 2.0 (SMTP fallback), waits for a Y/N reply, then calls `db.set_status` to confirm or reject.

**`db/`** wraps Tiger Data. `client.py` is the real implementation; `stub.py` is an in-memory drop-in with the same interface — swap with `DB_BACKEND=stub`. Both expose the same six helpers (see below).

**`app/`** is a Streamlit/Flask dashboard that reads confirmed incidents from `db/queries.py` and exposes plain-English search via `db/nl_search.py`.

**`shared/`** is the contract all three areas depend on. Do not change it unilaterally.

## Shared contract

### Incident fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | str (UUID) | set by `create_incident` |
| `hazard_type` | enum | `spill`, `unattended_burner`, `raw_meat_contact`, `knife_edge` |
| `zone` | str | must match a key in `shared/zones.yaml` |
| `camera_id` | str | |
| `started_at` | timestamptz | |
| `ended_at` | timestamptz, nullable | set when hazard clears |
| `confidence` | float 0–1 | |
| `snapshot_path` | str | annotated image path |
| `status` | enum | `pending`, `confirmed`, `rejected` |
| `validated_by` | str, nullable | |
| `validated_at` | timestamptz, nullable | |
| `staff_tags` | list[str] | from manager/schedule only — never face recognition |

### DB interface (both `client.py` and `stub.py` must implement)

```python
create_incident(incident: Incident) -> str
close_incident(incident_id: str, ended_at: datetime) -> None
set_status(incident_id: str, status: Status, validated_by: str, staff_tags: list[str] | None = None) -> None
get_incident(incident_id: str) -> Incident
query_patterns(group_by: list[str], start: datetime, end: datetime, status: Status = "confirmed") -> list[dict]
nl_search(question: str) -> list[Incident]
```

### Zones

Valid zone names: `fryer`, `prep_line`, `stove`, `dish_station`, `walk_in`, `floor`. Always match `shared/zones.yaml`.

## Git and GitHub workflow

**Never push to the remote repository.** The humans review and push all changes themselves.

When your work is ready:

1. Stage only the files you changed (avoid `git add -A`):
   ```bash
   git add detection/capture.py detection/isaac_client.py
   ```

2. Commit with a clear message. Use this format so authorship is attributed to the human, not Claude:
   ```bash
   git commit -m "feat(detection): add zone debouncing logic"
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
