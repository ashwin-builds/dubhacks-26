"""Live hazard demo: webcam -> Perceptron -> boxes on screen.

No database or alerts; this is for trying the model out on your own camera.

    python -m detection.demo                    # default webcam, live window
    python -m detection.demo --camera 1         # another camera
    python -m detection.demo --image kitchen.jpg
    python -m detection.demo --hazards spill knife_edge --model isaac-0.1

Keys in the live window: q / Esc to quit, s to save an annotated snapshot.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detection.hazards import HAZARDS
from detection.isaac_client import FrameResult, IsaacClient
from detection.overlay import draw_detections, draw_status

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def load_env() -> None:
    """Load .env if python-dotenv is installed; the SDK reads PERCEPTRON_API_KEY from the environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def describe(result: FrameResult) -> str:
    if not result.detections:
        return f"All clear  ({result.latency_s:.1f}s)"
    names = sorted({HAZARDS[d.hazard].label for d in result.detections})
    return f"HAZARD: {', '.join(names)}  ({result.latency_s:.1f}s)"


def annotate(frame: np.ndarray, result: FrameResult | None, error: str | None = None) -> np.ndarray:
    if error:
        return draw_status(frame, f"Error: {error}", alert=True)
    if result is None:
        return draw_status(frame, "Waiting for first result...", alert=False)
    out = draw_status(frame, describe(result), alert=bool(result.detections))
    return draw_detections(out, result.detections)


def log_result(result: FrameResult) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {describe(result)}", flush=True)
    for det in result.detections:
        print(f"    {det.hazard:<18} {det.mention!r:<30} box={det.box}", flush=True)
    if result.unmatched:
        print(f"    (ignored labels: {result.unmatched})", flush=True)


def save_snapshot(frame: np.ndarray) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    return path


class Analyzer(threading.Thread):
    """Sends the newest frame to Perceptron, at most ``fps`` times a second.

    Requests take longer than a camera frame, so the display loop never waits on the model:
    it hands over the latest frame and draws whatever result came back most recently.
    """

    def __init__(self, client: IsaacClient, fps: float):
        super().__init__(daemon=True)
        self.client = client
        self.interval = 1.0 / fps
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stopped = threading.Event()
        self.result: FrameResult | None = None
        self.result_frame: np.ndarray | None = None  # the frame `result` was computed on
        self.error: str | None = None

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.is_set():
            started = time.monotonic()
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is not None:
                try:
                    result = self.client.detect(frame)
                except Exception as exc:  # keep the demo running through network hiccups
                    self.error = f"{type(exc).__name__}: {exc}"
                    print(f"Perceptron call failed: {self.error}", file=sys.stderr, flush=True)
                else:
                    self.result, self.result_frame, self.error = result, frame, None
                    log_result(result)
            self._stopped.wait(max(0.0, self.interval - (time.monotonic() - started)))


def run_image(client: IsaacClient, path: Path, show: bool) -> int:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return 1
    result = client.detect(frame)
    log_result(result)
    annotated = annotate(frame, result)
    out = save_snapshot(annotated)
    print(f"Annotated image saved to {out}")
    if show:
        cv2.imshow("Kitchen hazards", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_camera(client: IsaacClient, camera: int, fps: float, show: bool) -> int:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(
            f"Could not open camera {camera}. On macOS, allow camera access for your terminal "
            "in System Settings > Privacy & Security > Camera.",
            file=sys.stderr,
        )
        return 1

    analyzer = Analyzer(client, fps)
    analyzer.start()
    print(f"Watching camera {camera} at up to {fps:g} requests/sec. Press q to quit, s to save a snapshot.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.", file=sys.stderr)
                return 1
            analyzer.submit(frame)
            if not show:
                time.sleep(0.01)
                continue
            # Boxes are drawn on the live frame; with a moving camera they lag by one request.
            view = annotate(frame, analyzer.result, analyzer.error)
            cv2.imshow("Kitchen hazards", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s") and analyzer.result_frame is not None:
                path = save_snapshot(annotate(analyzer.result_frame, analyzer.result))
                print(f"Snapshot saved to {path}")
    except KeyboardInterrupt:
        return 0
    finally:
        analyzer.stop()
        cap.release()
        if show:
            cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    load_env()  # before argparse, so .env can set ISAAC_QUERIES_PER_SEC
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--image", type=Path, help="run once on an image file instead of the webcam")
    parser.add_argument(
        "--fps",
        type=float,
        default=float(os.environ.get("ISAAC_QUERIES_PER_SEC", 2)),
        help="max Perceptron requests per second (default ISAAC_QUERIES_PER_SEC or 2)",
    )
    parser.add_argument("--model", help="Perceptron model id (default: PERCEPTRON_MODEL or the SDK default)")
    parser.add_argument(
        "--hazards", nargs="+", choices=list(HAZARDS), default=list(HAZARDS), help="hazards to look for (default all)"
    )
    parser.add_argument("--no-window", action="store_true", help="print results only; no video window")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the model")
    args = parser.parse_args(argv)

    if not (os.environ.get("PERCEPTRON_API_KEY") or os.environ.get("FAL_KEY")):
        print("Set PERCEPTRON_API_KEY in your environment or .env first.", file=sys.stderr)
        return 2
    client = IsaacClient(model=args.model, hazards=args.hazards)
    if args.show_prompt:
        print(client.system_prompt, end="\n\n")

    if args.image:
        return run_image(client, args.image, show=not args.no_window)
    return run_camera(client, args.camera, args.fps, show=not args.no_window)


if __name__ == "__main__":
    sys.exit(main())
