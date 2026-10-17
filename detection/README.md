# detection/

Kitchen video (played as a live stream) → Perceptron at 2 fps → hazard boxes → incidents → API. Owner: Zach.

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
python -m detection.demo clip.mp4 --api-url http://localhost:8000   # also send incidents to the API
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
| `knife_edge` | a sharp tool is not on a cutting board and no one is holding it |
| `spill` | liquid, grease or food is on the floor or a work surface |
| `unattended_burner` | a burner is on with no pan or pot on it |
| `raw_meat_contact` | raw meat, poultry or fish touches cooked or ready-to-eat food |

To tune what the model looks for, edit each hazard's `description`. How long a hazard must persist before it
becomes an incident is decided by the incident layer below, not by the model.

## Incidents

One frame showing a hazard is not an incident. `incidents.py` keeps the last 40 analysed frames (20 s at 2 fps)
per hazard and opens an incident once 80% of them show it. The incident closes after the hazard has been gone
for 10 s. Timing comes from video time, so model latency doesn't affect durations.

- **Opened**: the worker sends the incident ID it generated, the start time (first hazard frame), the location,
  the camera, the hazard type, and that first frame with boxes drawn, compressed to a ≤640 px JPEG.
- **Checked**: every 10 s, while anything is open, the worker sends `POST /incidents/false-positives` with
  `{"incident_ids": [...]}` (open incidents the API has acknowledged). The API answers `{"<id>": true, ...}` for the
  ones the manager marked as false positives; those stop being recorded and are never closed. Anything other than
  `true` keeps the incident going. The same hazard can't reopen until it has cleared.
- **Closed**: the worker sends the same ID with the duration.

Frames the model never answered (call failed, or all request slots busy) are counted as missing from gaps in the
timestamps. They count as "no hazard" toward opening, don't count toward closing, and each report says how many
there were. The terminal prints `*** INCIDENT OPENED` / `CLOSED` lines and the banner shows open incidents.

Without `--api-url` (or `API_URL`), incidents are printed but not sent. The worker token is read from
`API_WORKER_TOKEN`.

TODO: the 20 s window is hardcoded for every hazard (`DANGER_SECONDS` in `incidents.py`), and `camera_1`'s zone is
a placeholder (`CAMERA_LOCATIONS` in `api_client.py`).

## Files

| File | What it does |
| --- | --- |
| `hazards.py` | the four hazards: prompt text, overlay colour, label matching |
| `isaac_client.py` | builds the prompt, sends a frame, turns boxes into `Detection`s |
| `overlay.py` | draws boxes and the status banner |
| `video_source.py` | plays a video file or URL in real time, dropping frames like a live camera |
| `demo.py` | the CLI: streams a video, samples 2 fps, runs the model, shows boxes, prints a summary |
| `incidents.py` | sliding-window incident detector: open, close, reject, missing frames |
| `pipeline.py` | puts out-of-order model answers back in video order and feeds them to the detector |
| `api_client.py` | sends opened/closed incidents to the API, retrying failed calls in order |
| `compression.py` | shrinks a frame to a small JPEG |

## Tests

```bash
pytest detection
```

Tests use a fake model, so they need no API key or network.
