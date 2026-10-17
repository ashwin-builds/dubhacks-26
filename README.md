# Kitchen Safety Monitor

A camera system that spots hazards in a restaurant kitchen, texts a manager within seconds, and keeps a record so the kitchen can see where and when problems happen. Built for a hackathon.

## How it works

1. **Watch** – A webcam sends frames (~2/sec) to Perceptron's Isaac vision model, which finds and locates hazards.
2. **Debounce** – Repeated detections of the same hazard in the same zone become one *incident*.
3. **Alert** – The manager gets one text (Photon 2.0) with an annotated snapshot and replies Y or N.
4. **Record** – Every incident is stored in Tiger Data as `pending`, then set to `confirmed` or `rejected`.
5. **Learn** – A dashboard shows patterns by zone, time and shift, plus plain-English search.

```
webcam ──> detection/ ──> incident ──> db/ (pending)
                              │
                              └──> alerts/ ──> manager phone ──Y/N──> db/ (confirmed | rejected)
                                                                         │
                                                         app/ dashboard <┘
```

## Team and ownership

| Area | Owner | Folder |
| --- | --- | --- |
| Detection (Step 1): capture, Isaac calls, zones, debouncing, overlay | Zach | `detection/` |
| Alerts and app (Step 2 + UI): Photon, SMTP fallback, replies, dashboard, demo script | Ashwin | `alerts/`, `app/` |
| Database (Step 3): schema, helpers, seed data, pattern queries, NL search | Alex | `db/` |
| Shared contract: incident model, zones, config | All three | `shared/` |

Changes to `shared/` need agreement from all three.

## Repo layout

```
kitchen-monitor/
├── shared/
│   ├── models.py        # Incident dataclass, HazardType, Status
│   ├── zones.yaml       # zone names + image regions per camera
│   └── config.py        # loads .env
├── detection/           # Zach
│   ├── capture.py       # webcam loop, frame sampling
│   ├── isaac_client.py  # async Perceptron SDK calls
│   ├── zones.py         # map detections to zones
│   ├── incidents.py     # debouncing: open / merge / close
│   └── overlay.py       # draw boxes on the live feed
├── alerts/              # Ashwin
│   ├── photon_client.py
│   ├── smtp_fallback.py
│   └── replies.py       # parse Y/N, call db.set_status
├── db/                  # Alex
│   ├── schema.sql
│   ├── client.py        # real Tiger Data implementation
│   ├── stub.py          # in-memory stub with the same interface
│   ├── seed.py          # fake historical incidents
│   ├── queries.py       # pattern queries
│   └── nl_search.py     # plain English -> SQL
├── app/
│   └── dashboard.py     # Streamlit or Flask
├── tests/
├── .env.example
└── requirements.txt
```

## The shared contract

### Incident

| Field | Type | Notes |
| --- | --- | --- |
| `id` | str (UUID) | set by `create_incident` |
| `hazard_type` | enum | `spill`, `unattended_burner`, `raw_meat_contact`, `knife_edge` |
| `zone` | str | must match `shared/zones.yaml` |
| `camera_id` | str | |
| `started_at` | timestamptz | |
| `ended_at` | timestamptz, nullable | set when the hazard clears |
| `confidence` | float 0–1 | |
| `snapshot_path` | str | annotated image |
| `status` | enum | `pending`, `confirmed`, `rejected` |
| `validated_by` | str, nullable | |
| `validated_at` | timestamptz, nullable | |
| `staff_tags` | list[str] | from manager or shift schedule, never face recognition |

### Database helpers (`db/client.py` and `db/stub.py`)

```python
create_incident(incident: Incident) -> str
close_incident(incident_id: str, ended_at: datetime) -> None
set_status(incident_id: str, status: Status, validated_by: str, staff_tags: list[str] | None = None) -> None
get_incident(incident_id: str) -> Incident
query_patterns(group_by: list[str], start: datetime, end: datetime, status: Status = "confirmed") -> list[dict]
nl_search(question: str) -> list[Incident]
```

Set `DB_BACKEND=stub` to develop without a database.

### Zones

Placeholder names, agree before building: `fryer`, `prep_line`, `stove`, `dish_station`, `walk_in`, `floor`.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
```

`.env` keys:

```
PERCEPTRON_API_KEY=
PHOTON_API_KEY=
MANAGER_PHONE=
SMTP_HOST=
SMTP_USER=
SMTP_PASSWORD=
TIGER_DATABASE_URL=
DB_BACKEND=stub        # stub | tiger
ISAAC_QUERIES_PER_SEC=2
```

## Milestones

| Checkpoint | Done when |
| --- | --- |
| Hour 1 | Contract in `shared/` agreed; stubs in place |
| Midpoint | Real incidents from `detection/` land in the database and trigger a text |
| Late | Full loop works; dashboard shows patterns from `db/queries.py` |
| Final | Demo tuned, backup video recorded, rehearsed |

## Demo script

1. Clean scene shows all clear.
2. Spill water, leave a pan on the burner; red markers appear.
3. Manager phone gets a text; reply Y; incident turns confirmed.
4. Ask "Where do most spills happen?" on the dashboard.
5. Add a new hazard prompt live to show no retraining is needed.

Backup: a pre-recorded video of the full loop, in case venue Wi‑Fi fails.

## Phase 2 (stretch)

Expo-style order tracking: group orders by table, track plates with a fast detector (YOLO + ByteTrack, not Isaac), match plates to orders with probabilities, and show readiness. Demo only a narrow slice if time allows.
