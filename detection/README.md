# detection/

Kitchen video (played as a live stream) → Perceptron at 2 fps → hazard boxes → incidents → `incidents` table. Owner: Zach.

## Demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r detection/requirements.txt
export PERCEPTRON_API_KEY=sk_live_...      # or put it in .env

python -m detection.demo clip.mp4                       # a video file
python -m detection.demo                                # pick the file in a dialog
python -m detection.demo "https://www.youtube.com/watch?v=..."   # downloaded once to detection/videos/
python -m detection.demo clip.mp4 --loop                # keep streaming until you press q
python -m detection.demo clip.mp4 --hazards spill knife_edge
python -m detection.demo --list-models                  # models your API key can use
python -m detection.demo clip.mp4 --show-prompt         # print the prompt sent to the model
python -m detection.demo clip.mp4 --write-db              # also write incidents to TIGER_DATABASE_URL
```

Nothing is analysed ahead of time. The video plays in real time, and every 0.5 s of playback the frame on
screen is sent to Perceptron, the way a live camera feed sampled at 2 fps would be (`--fps` or
`ISAAC_QUERIES_PER_SEC` changes the rate). Requests run in parallel, so a model that takes longer than 0.5 s
to answer doesn't slow the schedule; if 4 requests are already waiting, that sample is skipped, as a live
system would do. Each result is printed with its video time (`[01:12.5] HAZARD: Spill`), and at the end a
summary lists how many analysed frames showed each hazard and when it was first and last seen.

Direct links to `.mp4`/`.mov`/`.webm` files stream without downloading. Page URLs (YouTube, Vimeo, ...) are
downloaded once with `yt-dlp` before playback starts; only the file is fetched, the model sees nothing early.

In the window: `q` quits, `s` saves an annotated snapshot to `detection/snapshots/`. Every result is also
printed to the terminal, including labels the model returned that matched no hazard.

Model: the fastest one your API key can use, picked at startup (Isaac models first, then `perceptron-mk1`, then
`perceptron-mk1.5`). Run `--list-models` to see what your key has; override with `--model` or `PERCEPTRON_MODEL`.

Hazards (`detection/hazards.py`), judged one frame at a time:

| Key | Flagged when |
| --- | --- |
| `knife_edge` | a sharp tool is set down with no one holding it or working at it (a knife in use is fine) |
| `spill` | liquid, grease or food is on the floor or a work surface |
| `unattended_burner` | a burner is on with no one at the stove tending it (cooking is fine) |
| `raw_meat_contact` | raw meat, poultry or fish touches cooked or ready-to-eat food |

To tune what the model looks for, edit each hazard's `description`. How long a hazard must persist before it
becomes an incident is decided by the incident layer below, not by the model.

## Incidents

One frame showing a hazard is not an incident. `incidents.py` keeps the last 40 analysed frames (20 s at 2 fps)
per hazard and opens an incident once 80% of them show it. The incident closes after the hazard has been gone
for 10 s. Timing comes from video time, so model latency doesn't affect durations.

Incidents are written straight to the `incidents` table (`store.py`), one row per incident, in two writes:

- **Opened**: insert `time` (first hazard frame), `hazard_type`, `image_as_binary` (that frame with boxes drawn,
  compressed to a ≤640 px JPEG), `zone_id` (looked up from the camera's location), and `completed = false`.
- **Resolved**: set `duration` (whole seconds) and `completed = true`; `false_positive` is left alone.
- **False positive**: every 10 s, while anything is open, one `UPDATE ... RETURNING` marks `completed = true` on any
  of our open incidents with `false_positive = true`, and returns them. Those are force-stopped: no longer recorded,
  no duration written.
- **Ignored spot**: a false positive's boxes become ignore zones (`suppression.py`). Later detections of the same
  hazard that mostly overlap one are dropped before they reach the detector, so the same misread object can't open
  a new incident. Other hazards there, and the same hazard elsewhere in the frame, still count. A zone ends after
  30 minutes, or once nothing has matched it for 60 s. Every dropped detection is logged.

Failed writes are queued in memory and retried in order on the next frame; a failed false-positive check is retried
10 s later.

Frames the model never answered (call failed, or all request slots busy) are counted as missing from gaps in the
timestamps. They count as "no hazard" toward opening, don't count toward closing, and each report says how many
there were. The terminal prints `*** INCIDENT OPENED` / `CLOSED` lines and the banner shows open incidents.

Without `--write-db`, incidents are printed but not written. With it, the connection string is read from
`TIGER_DATABASE_URL` (environment or `.env`).

The `incidents` table needs two columns that `database/schema.sql` doesn't have yet:

```sql
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS hazard_type TEXT;
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS completed BOOLEAN NOT NULL DEFAULT FALSE;
```

TODO: the 20 s window is hardcoded for every hazard (`DANGER_SECONDS` in `incidents.py`), and `camera_1`'s zone is
a placeholder (`CAMERA_LOCATIONS` in `reporter.py`). The ignore-zone lifetimes (`MAX_AGE_SECONDS`,
`IDLE_SECONDS` in `suppression.py`) are placeholders too.

## Files

| File | What it does |
| --- | --- |
| `hazards.py` | the four hazards: prompt text, overlay colour, label matching |
| `isaac_client.py` | builds the prompt, sends a frame, turns boxes into `Detection`s |
| `overlay.py` | draws boxes and the status banner |
| `video_source.py` | plays a video file or URL in real time, dropping frames like a live camera |
| `demo.py` | the CLI: streams a video, samples 2 fps, runs the model, shows boxes, prints a summary |
| `incidents.py` | sliding-window incident detector: open, close, reject, missing frames |
| `suppression.py` | ignore zones for spots the manager marked as false positives |
| `pipeline.py` | puts out-of-order model answers back in video order and feeds them to the detector |
| `reporter.py` | turns incident events into database writes; checks for false positives every 10 s |
| `store.py` | the SQL for the `incidents` table (and an in-memory stub for tests and demos) |
| `compression.py` | shrinks a frame to a small JPEG |

## Tests

```bash
pytest detection
```

Tests use a fake model and an in-memory database, so they need no API key, database, or network.
