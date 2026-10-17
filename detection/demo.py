"""Live hazard demo: webcam (or a video played back as if live) -> Perceptron -> boxes on screen.

No database or alerts; this is for trying the model out on your own camera or on test footage.

    python -m detection.demo                    # default webcam, live window
    python -m detection.demo --camera 1         # another camera
    python -m detection.demo --video clip.mp4   # play a video file as a live feed
    python -m detection.demo --video            # same, choosing the file in a dialog
    python -m detection.demo --video "https://www.youtube.com/watch?v=..."   # needs yt-dlp
    python -m detection.demo --image kitchen.jpg
    python -m detection.demo --hazards spill knife_edge --model perceptron-mk1.5
    python -m detection.demo --list-models      # which models your API key can use

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
from detection.isaac_client import FrameResult, IsaacClient, list_models
from detection.overlay import draw_detections, draw_status
from detection.video_source import VideoFileStream, format_position, pick_video_file, resolve_video

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


def log_result(result: FrameResult, position_s: float | None = None) -> None:
    """Print one result, stamped with its video time (or the wall clock for a live camera)."""
    stamp = format_position(position_s) if position_s is not None else datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {describe(result)}", flush=True)
    for det in result.detections:
        print(f"    {det.hazard:<18} {det.mention!r:<30} box={det.box}", flush=True)
    if result.unmatched:
        print(f"    (ignored labels: {result.unmatched})", flush=True)


def summarize(history: list[tuple[float | None, FrameResult]]) -> str:
    """Per-hazard tally of a run: how many analysed frames showed it, and when it was first and last seen."""
    if not history:
        return "No frames were analysed."
    lines = [f"Analysed {len(history)} frames."]
    for key, hazard in HAZARDS.items():
        seen = [pos for pos, result in history if any(d.hazard == key for d in result.detections)]
        if not seen:
            continue
        line = f"  {hazard.label:<24} {len(seen):>4} / {len(history)} frames"
        if seen[0] is not None:
            line += f"   first {format_position(seen[0])}   last {format_position(seen[-1])}"
        lines.append(line)
    if len(lines) == 1:
        lines.append("  No hazards detected.")
    return "\n".join(lines)


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
        self._pending: tuple[np.ndarray, float | None] | None = None
        self._stopped = threading.Event()
        self.result: FrameResult | None = None
        self.result_frame: np.ndarray | None = None  # the frame `result` was computed on
        self.error: str | None = None
        self.history: list[tuple[float | None, FrameResult]] = []  # (video time, result) per analysed frame

    def submit(self, frame: np.ndarray, position_s: float | None = None) -> None:
        with self._lock:
            self._pending = (frame, position_s)

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.is_set():
            started = time.monotonic()
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None:
                frame, position_s = pending
                try:
                    result = self.client.detect(frame)
                except Exception as exc:  # keep the demo running through network hiccups
                    self.error = f"{type(exc).__name__}: {exc}"
                    print(f"Perceptron call failed: {self.error}", file=sys.stderr, flush=True)
                else:
                    self.result, self.result_frame, self.error = result, frame, None
                    self.history.append((position_s, result))
                    log_result(result, position_s)
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
    print(f"Watching camera {camera} at up to {fps:g} requests/sec. Press q to quit, s to save a snapshot.")
    return run_stream(client, cap, fps, show)


def run_video(client: IsaacClient, video: str, fps: float, show: bool, speed: float, loop: bool) -> int:
    try:
        stream = VideoFileStream(resolve_video(video), speed=speed, loop=loop)
    except (OSError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    length = f"{format_position(stream.duration_s)} long, " if stream.duration_s else ""
    print(
        f"Playing {video} ({length}{stream.fps:g} fps) as a live feed at {speed:g}x, "
        f"up to {fps:g} requests/sec. Press q to stop, s to save a snapshot."
    )
    code = run_stream(client, stream, fps, show, video=stream)
    print(f"Dropped {stream.dropped} frames to keep pace with real time.")
    return code


def run_stream(client: IsaacClient, cap, fps: float, show: bool, video: VideoFileStream | None = None) -> int:
    """Shared loop for a camera or a video stream: show frames, hand the newest to the analyzer."""
    analyzer = Analyzer(client, fps)
    analyzer.start()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if video is not None:
                    print("Video finished.")
                    return 0
                print("Camera stopped returning frames.", file=sys.stderr)
                return 1
            analyzer.submit(frame, video.position_s if video is not None else None)
            if not show:
                if video is None:
                    time.sleep(0.01)  # a video stream paces itself
                continue
            # Boxes are drawn on the live frame; with a moving scene they lag by one request.
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
        if video is not None:
            analyzer.join(timeout=30)  # let the last in-flight request land in the summary
            print(summarize(analyzer.history))


def main(argv: list[str] | None = None) -> int:
    load_env()  # before argparse, so .env can set ISAAC_QUERIES_PER_SEC
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--image", type=Path, help="run once on an image file instead of the webcam")
    parser.add_argument(
        "--video",
        nargs="?",
        const="",
        help="play a video file or URL as if it were the live camera (no value: pick a file in a dialog)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="video playback speed (default 1.0 = real time)")
    parser.add_argument("--loop", action="store_true", help="restart the video when it ends")
    parser.add_argument(
        "--fps",
        type=float,
        default=float(os.environ.get("ISAAC_QUERIES_PER_SEC", 2)),
        help="max Perceptron requests per second (default ISAAC_QUERIES_PER_SEC or 2)",
    )
    parser.add_argument(
        "--model", help="Perceptron model id (default: PERCEPTRON_MODEL, else the fastest model your key can use)"
    )
    parser.add_argument("--list-models", action="store_true", help="print the models your API key can use and exit")
    parser.add_argument(
        "--hazards", nargs="+", choices=list(HAZARDS), default=list(HAZARDS), help="hazards to look for (default all)"
    )
    parser.add_argument("--no-window", action="store_true", help="print results only; no video window")
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the model")
    args = parser.parse_args(argv)

    if not (os.environ.get("PERCEPTRON_API_KEY") or os.environ.get("FAL_KEY")):
        print("Set PERCEPTRON_API_KEY in your environment or .env first.", file=sys.stderr)
        return 2
    if args.list_models:
        for model_id in list_models():
            print(model_id)
        return 0
    client = IsaacClient(model=args.model, hazards=args.hazards)
    print(f"Model: {client.model}")
    if args.show_prompt:
        print(client.system_prompt, end="\n\n")

    if args.image:
        return run_image(client, args.image, show=not args.no_window)
    if args.video is not None:
        video = args.video or pick_video_file()
        if not video:
            print("No video chosen. Pass a path: --video clip.mp4", file=sys.stderr)
            return 2
        return run_video(client, video, args.fps, not args.no_window, args.speed, args.loop)
    return run_camera(client, args.camera, args.fps, show=not args.no_window)


if __name__ == "__main__":
    sys.exit(main())
