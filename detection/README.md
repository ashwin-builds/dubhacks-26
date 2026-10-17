# detection/

Webcam → Perceptron → hazard boxes. Owner: Zach.

## Live demo (no database, no alerts)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r detection/requirements.txt
export PERCEPTRON_API_KEY=sk_live_...      # or put it in .env

python -m detection.demo                   # default webcam, live window
python -m detection.demo --camera 1        # another camera
python -m detection.demo --image photo.jpg # one still image; annotated copy goes to detection/snapshots/
python -m detection.demo --list-models     # models your API key can use
python -m detection.demo --model perceptron-mk1.5 --hazards spill knife_edge
python -m detection.demo --show-prompt     # print the prompt sent to the model
```

## Testing on video (played back as a live feed)

```bash
python -m detection.demo --video clip.mp4             # a file on disk
python -m detection.demo --video                      # pick the file in a dialog
python -m detection.demo --video "https://www.youtube.com/watch?v=..."   # downloaded once to detection/videos/
python -m detection.demo --video clip.mp4 --speed 2   # twice real time
python -m detection.demo --video clip.mp4 --loop      # repeat until you press q
```

The video plays at real-time speed through the exact same loop as the webcam: frames the program can't keep up
with are dropped rather than played late, and the model is called at most `--fps` times a second. So the model
sees what it would see if the scene were happening in front of the camera. Each result is stamped with its video
time (`[01:12.5] HAZARD: Spill`), and when the video ends a summary lists how many analysed frames showed each
hazard and when it was first and last seen. Direct links to `.mp4`/`.mov`/`.webm` files stream without
downloading; page URLs (YouTube, Vimeo, ...) need `yt-dlp`, which is in `requirements.txt`.

In the window: `q` quits, `s` saves an annotated snapshot. Every result is also printed to the terminal,
including labels the model returned that matched no hazard.

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
becomes an incident is decided by the separate incident layer (not built yet), not by the model.

Requests go out at most `--fps` times a second (default `ISAAC_QUERIES_PER_SEC`, else 2). The video stays
smooth while a request is in flight; boxes update when the next answer arrives.

## Files

| File | What it does |
| --- | --- |
| `hazards.py` | the four hazards: prompt text, overlay colour, label matching |
| `isaac_client.py` | builds the prompt, sends a frame, turns boxes into `Detection`s |
| `overlay.py` | draws boxes and the status banner |
| `video_source.py` | plays a video file or URL as a real-time "camera" |
| `demo.py` | webcam / video / image CLI |

## Tests

```bash
pytest detection
```

Tests use a fake `perceive` function, so they need no API key or network.
