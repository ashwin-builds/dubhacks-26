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
python -m detection.demo --model isaac-0.1 --hazards spill knife_edge
python -m detection.demo --show-prompt     # print the prompt sent to the model
```

In the window: `q` quits, `s` saves an annotated snapshot. Every result is also printed to the terminal,
including labels the model returned that matched no hazard.

Model: `isaac-0.3-fast` by default; override with `--model` or `PERCEPTRON_MODEL`.

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
| `demo.py` | webcam / image CLI |

## Tests

```bash
pytest detection
```

Tests use a fake `perceive` function, so they need no API key or network.
