# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Ownership and boundaries

This repo has four areas. Each agent or session works within its assigned area:

| Owner | Folder(s) |
|-------|-----------|
| Zach | `detection/` |
| Ashwin | `api/`, `alerts/`, `app/` |
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

# Run the API (incident intake + notifications)
uvicorn api.main:app --reload --port 8000

# Run the detection worker against a webcam or a recorded file
python -m detection.worker --source 0
python -m detection.worker --source demo/kitchen.mp4

# Run the dashboard
streamlit run app/dashboard.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_<module>.py

# Run a single test by name
pytest tests/test_<module>.py::test_<name>

# Run with stubs (no real database, no real SMS)
DB_BACKEND=stub PHOTON_BACKEND=stub python -m <module>
```

Set `DB_BACKEND=stub` and `PHOTON_BACKEND=stub` in `.env` to develop and demo without a live Tiger Data connection or sending real texts.

## Architecture

The detection worker pushes to `api/`. `api/` is the only writer to `db/` and the only sender of
messages. Detection never talks to the database directly.

```
camera (live) or recorded video (demo)
            │
            ▼
   detection worker ──── no hazard ────> do nothing
            │
            │  hazard opens  ──> POST /incidents             (frame, time, description)
            │  hazard clears ──> POST /incidents/{id}/close  (clip, time, description,
            │                                                 confidence, exposure)
            ▼
   ┌────────────────────────────────────────────────────┐
   │                        api/                        │
   │  1. records EVERY incident in db/                  │
   │  2. decides separately whether to NOTIFY, using    │
   │     the hazard_policy thresholds in db/            │
   └────────────────────────────────────────────────────┘
            │                        │
            │ warning SMS            │ incident report SMS ("did this happen?")
            ▼                        ▼
      manager phones (db/ manager_contacts, not a .env variable)
            │
            │ Y / N reply ──> POST /webhooks/photon/inbound ──> db.set_status
            ▼
          db/ ──┬──> weekly report (charts)
                └──> app/ dashboard (live camera/recording preview + labels)
```

### Record vs. notify — the central rule

Recording and notifying are **separate decisions**. The worker always calls the API the moment a
hazard opens; the API always writes the incident to the database; the API then decides whether a
human hears about it.

An incident that falls below its policy threshold is **recorded but unalerted** — no warning text
and no incident report text. It is written with `alerted = false` and keeps `status = pending`
forever, because nobody is ever texted and so nobody ever confirms or rejects it.

That has a consequence the reporting path must honour: **`query_patterns` defaults to
`status="confirmed"`, so the default query cannot see near-misses at all.** The weekly report must
query across all statuses and separate the two populations — confirmed incidents (a manager said it
happened) from unalerted near-misses (`alerted = false`) — rather than relying on that default.
Counting near-misses is the whole reason they are recorded; reading them through the confirmed-only
default silently drops them again.

This split exists because the two requirements conflict at `t=0`: we cannot know at the moment a
knife appears whether it will stay out for more than a minute. So:

- On open, the API records the incident. If `warn_after_seconds` is `0`, it texts immediately. If it
  is greater than `0`, the API starts a timer (an `asyncio` task sleeping to `t+N`, no job queue
  needed) and texts only if the incident is still open when the timer fires.
- On close, the API compares the reported `exposure_seconds` against `min_exposure_seconds` and
  sends the incident report only if the hazard lasted long enough to be worth a manager's attention.

**All thresholds live in the database, not in code.** The worker holds no policy and reads no
thresholds — that is why adding a hazard type or changing a duration is a database row edit and
needs no redeploy.

### Exposure and time

`exposure_seconds` is **supplied by the worker and trusted as authoritative**. Do not recompute it
from `started_at` and `ended_at`: on the prerecorded-video path, video-timeline time is not
wall-clock time, and seeking or fast playback breaks any server-side subtraction. The API records
`received_at` on both calls separately, for audit and clock-drift debugging.

Store all timestamps as UTC `timestamptz`; render in kitchen-local time in `app/` and in reports.

## API contract (`api/`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/incidents` | Hazard opened. Multipart: a single annotated frame + JSON body. Returns `incident_id`. |
| `POST` | `/incidents/{id}/close` | Hazard cleared. Clip, final confidence, description, `exposure_seconds`. |
| `POST` | `/webhooks/photon/inbound` | Manager's Y/N reply. Maps the reply to an incident and calls `db.set_status`. |
| `GET`  | `/incidents/{id}/snapshot?token=…` | Token-gated public image URL for the text message. |
| `GET`  | `/incidents/{id}/clip?token=…` | Token-gated clip for the incident report. |
| `GET`  | `/cameras/{camera_id}/latest.jpg` | Most recent annotated frame, for the dashboard preview. |
| `GET`  | `/healthz` | Liveness. |

**Open payload** (`POST /incidents`): `client_incident_id`, `camera_id`, `hazard_type`, `zone`,
`started_at`, `description`, `confidence`, plus the frame as a file part.

**Close payload** (`POST /incidents/{id}/close`): `ended_at`, `exposure_seconds`, `confidence`,
`description`, optional clip file part.

### Idempotency

The worker generates a `client_incident_id` and sends it on open; the API stores it under a unique
index and returns the existing incident on a repeat. Venue Wi-Fi will cause retries, and a retry
must never produce a second text. The worker spools failed calls to disk and replays them.

### Auth

The worker authenticates with a bearer token (`API_WORKER_TOKEN`). The Photon inbound webhook must
be verified before it is trusted — anyone who can reach the endpoint can otherwise forge a `Y`.

### Message delivery

`snapshot_path` is a local filesystem path and means nothing to a phone. Texts always carry a
token-gated URL served by the API, never a path. Whether an image rides along as MMS or as a link
depends on Photon 2.0's capabilities — confirm against its docs before relying on MMS.

## Shared contract

### Incident fields

| Field | Type | Notes |
|-------|------|-------|
| `id` | str (UUID) | set by `create_incident` |
| `client_incident_id` | str | worker-generated idempotency key, unique |
| `hazard_type` | enum | `spill`, `unattended_burner`, `raw_meat_contact`, `knife_edge` |
| `zone` | str | must match a key in `shared/zones.yaml` |
| `camera_id` | str | |
| `started_at` | timestamptz | |
| `ended_at` | timestamptz, nullable | set when hazard clears |
| `exposure_seconds` | float, nullable | reported by the worker; set on close |
| `confidence` | float 0–1 | |
| `description` | str | the model's plain-text description of what it saw |
| `snapshot_path` | str | annotated image path |
| `clip_path` | str, nullable | footage of the event; set on close |
| `alerted` | bool | `false` when policy suppressed both texts — this is a near-miss |
| `status` | enum | `pending`, `confirmed`, `rejected` |
| `validated_by` | str, nullable | |
| `validated_at` | timestamptz, nullable | |
| `staff_tags` | list[str] | from manager/schedule only — never face recognition |

**Needs all three owners' agreement before anyone writes against it:**

- New `shared/models.py` fields: `client_incident_id`, `exposure_seconds`, `description`,
  `clip_path`, `alerted`.
- **Two changed signatures** on the existing DB interface, which Alex implements in both
  `client.py` and `stub.py`: `close_incident` gains a required `exposure_seconds` parameter, and
  `set_status` now returns the updated `Incident` instead of `None` (the API needs the row back to
  build the reply confirmation).
- The six new `api/`-only helpers below are additive and break nothing.

### DB interface (both `client.py` and `stub.py` must implement)

```python
create_incident(incident: Incident) -> str
close_incident(incident_id: str, ended_at: datetime, exposure_seconds: float) -> None
set_status(incident_id: str, status: Status, validated_by: str, staff_tags: list[str] | None = None) -> Incident
get_incident(incident_id: str) -> Incident
query_patterns(group_by: list[str], start: datetime, end: datetime, status: Status = "confirmed") -> list[dict]
nl_search(question: str) -> list[Incident]

# added for api/
get_incident_by_client_id(client_incident_id: str) -> Incident | None
get_hazard_policy(hazard_type: HazardType) -> HazardPolicy
list_manager_contacts(active_only: bool = True) -> list[ManagerContact]
record_notification(incident_id: str, kind: str, to: str, provider_msg_id: str) -> str
find_notification_for_reply(from_number: str, code: str | None) -> Notification | None
```

### New tables

**`hazard_policy`** — one row per hazard type, editable live, no redeploy:

| Column | Notes |
|--------|-------|
| `hazard_type` | PK |
| `warn_after_seconds` | `0` = text immediately on open; `60` = only if still open after a minute |
| `min_exposure_seconds` | below this, no incident report is sent |
| `min_confidence` | below this, record only, never text |
| `cooldown_seconds` | suppress repeats for the same (zone, hazard) within this window |
| `enabled` | kill switch per hazard |

**`manager_contacts`** — `name`, `phone`, `active`. Replaces the single `MANAGER_PHONE` env var.

**`notifications`** — `incident_id`, `kind` (`warning` | `report`), `to`, `provider_msg_id`,
`sent_at`, `delivery_status`, `reply_code`. This is the audit log for why a text did or did not
arrive, and it is how an inbound reply is mapped back to an incident. It also makes "awaiting a
reply" derivable per-message without adding states to the `status` enum. Note the division of
labour: `notifications` is the per-message audit trail, while `incidents.alerted` is the flag the
weekly report can actually filter on — reporting queries should not have to join `notifications`.

### Reply disambiguation

A bare `Y` is ambiguous when several managers are on the list or several incidents are open. Each
report text carries a short code (`Reply Y 4F2`); `notifications.reply_code` resolves it. A bare
`Y` with no code falls back to the most recent unanswered report sent to that number. First reply
wins; `validated_by` is the contact name resolved from the sending number.

### Zones

Valid zone names: `fryer`, `prep_line`, `stove`, `dish_station`, `walk_in`, `floor`. Always match
`shared/zones.yaml`.

## Open decisions (do not write code against these until settled)

These three are collisions where a teammate could be writing the wrong thing right now:

1. **`alerts/` vs. `api/`** — `alerts/` previously owned the SMS and reply loop that `api/` now
   owns. Decide: `alerts/` becomes a thin library (Photon client, SMTP fallback, message templates)
   that `api/` calls, or it is absorbed into `api/` entirely. Same owner either way, but it must be
   written once.
2. **`detection/` no longer calls `db.create_incident`** — it calls the API instead. This is a
   breaking change to the previous contract and needs Zach's sign-off.
3. **Weekly report ownership** — the queries belong in `db/queries.py` (Alex), the chart rendering
   in `app/` (Ashwin). Agree the split before either starts.

Also unsettled: whether the `status` enum needs more states than `pending | confirmed | rejected`
(currently answered by the `notifications` table instead), and whether Photon 2.0 supports MMS.

## Git and GitHub workflow

**Never push to the remote repository.** The humans review and push all changes themselves.

When your work is ready:

1. Stage only the files you changed (avoid `git add -A`):
   ```bash
   git add api/main.py api/notify.py
   ```

2. Commit with a clear message. Use this format so authorship is attributed to the human, not Claude:
   ```bash
   git commit -m "feat(api): add incident intake endpoint"
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
