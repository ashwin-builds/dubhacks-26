"""Hazard demo: play a kitchen video as if it were a live camera and run Perceptron on it at 2 fps.

Nothing is analysed ahead of time. The video plays in real time; every 0.5 s of playback the frame on
screen is sent to the model, exactly as a live 2 fps camera feed would be. Results feed the incident
detector; incidents are printed, and written to the incidents database only with --write-db.

    python -m detection.demo clip.mp4            # a video file
    python -m detection.demo                     # choose the file in a dialog
    python -m detection.demo "https://www.youtube.com/watch?v=..."   # downloaded once with yt-dlp
    python -m detection.demo clip.mp4 --loop     # keep streaming until you press q
    python -m detection.demo clip.mp4 --hazards spill knife_edge --model perceptron-mk1.5
    python -m detection.demo --list-models       # which models your API key can use
    python -m detection.demo clip.mp4 --write-db  # also write incidents to TIGER_DATABASE_URL

Keys in the window: q / Esc to quit, s to save an annotated snapshot.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from detection.hazards import HAZARDS
from detection.incidents import Danger, IncidentDetector
from detection.isaac_client import FrameResult, IsaacClient, list_models
from detection.overlay import draw_detections, draw_status
from detection.pipeline import IncidentPipeline
from detection.reporter import IncidentReporter
from detection.store import PostgresIncidentStore, StubIncidentStore
from detection.video_source import VideoFileStream, format_position, pick_video_file, resolve_video

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
# Requests allowed in flight at once. At 2 fps this covers up to ~2 s of model latency before frames are skipped.
MAX_IN_FLIGHT = 4


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


def annotate(
    frame: np.ndarray, result: FrameResult | None, error: str | None = None, incidents: list[Danger] = ()
) -> np.ndarray:
    if error:
        return draw_status(frame, f"Error: {error}", alert=True)
    if result is None:
        return draw_status(frame, "Waiting for first result...", alert=False)
    message = describe(result)
    if incidents:
        message += "  |  INCIDENT: " + ", ".join(HAZARDS[d.value].label for d in incidents)
    out = draw_status(frame, message, alert=bool(result.detections or incidents))
    return draw_detections(out, result.detections)


def log_result(result: FrameResult, position_s: float) -> None:
    """Print one result, stamped with the video time of the frame it was computed on."""
    print(f"[{format_position(position_s)}] {describe(result)}", flush=True)
    for det in result.detections:
        print(f"    {det.hazard:<18} {det.mention!r:<30} box={det.box}", flush=True)
    if result.unmatched:
        print(f"    (ignored labels: {result.unmatched})", flush=True)


def summarize(history: list[tuple[float, FrameResult]]) -> str:
    """Per-hazard tally of a run: how many analysed frames showed it, and when it was first and last seen."""
    if not history:
        return "No frames were analysed."
    history = sorted(history, key=lambda item: item[0])
    lines = [f"Analysed {len(history)} frames."]
    for key, hazard in HAZARDS.items():
        seen = [pos for pos, result in history if any(d.hazard == key for d in result.detections)]
        if seen:
            lines.append(
                f"  {hazard.label:<24} {len(seen):>4} / {len(history)} frames"
                f"   first {format_position(seen[0])}   last {format_position(seen[-1])}"
            )
    if len(lines) == 1:
        lines.append("  No hazards detected.")
    return "\n".join(lines)


def save_snapshot(frame: np.ndarray) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    return path


class Analyzer:
    """Runs Perceptron on sampled frames in the background, several requests at a time.

    A model call can take longer than the 0.5 s between samples, so calls overlap rather than queue: the
    stream keeps its 2 fps schedule however slow the model is. If ``max_in_flight`` calls are already
    running when the next sample is due, that sample is skipped, as a live system would do.
    """

    def __init__(
        self,
        client: IsaacClient,
        max_in_flight: int = MAX_IN_FLIGHT,
        on_result: Callable[[int, float, np.ndarray, FrameResult | None], None] | None = None,
    ):
        self.client = client
        # Called with (sequence number, video time, frame, result or None on failure) for every sent frame.
        self.on_result = on_result
        self._pool = ThreadPoolExecutor(max_workers=max_in_flight)
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._lock = threading.Lock()
        self.result: FrameResult | None = None  # newest result by video time
        self.result_frame: np.ndarray | None = None  # the frame `result` was computed on
        self.result_position: float = -1.0
        self.error: str | None = None
        self.history: list[tuple[float, FrameResult]] = []  # (video time, result) per analysed frame
        self.sent = 0
        self.skipped = 0

    def submit(self, frame: np.ndarray, position_s: float) -> bool:
        """Send a frame for analysis; False if every request slot is busy and the frame was skipped."""
        if not self._slots.acquire(blocking=False):
            self.skipped += 1
            return False
        seq = self.sent
        self.sent += 1
        self._pool.submit(self._analyze, frame, position_s, seq)
        return True

    def _analyze(self, frame: np.ndarray, position_s: float, seq: int | None = None) -> None:
        result = None
        try:
            result = self.client.detect(frame)
        except Exception as exc:  # keep the stream running through network hiccups
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[{format_position(position_s)}] Perceptron call failed: {self.error}", file=sys.stderr, flush=True)
        else:
            with self._lock:
                self.history.append((position_s, result))
                # Answers can come back out of order; only a newer frame replaces what's on screen.
                if position_s > self.result_position:
                    self.result, self.result_frame, self.result_position = result, frame, position_s
                    self.error = None
                log_result(result, position_s)
        finally:
            self._slots.release()
            if self.on_result is not None and seq is not None:
                self.on_result(seq, position_s, frame, result)

    def close(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)


def run_video(
    client: IsaacClient,
    stream: VideoFileStream,
    fps: float,
    show: bool,
    analyzer: Analyzer | None = None,
    pipeline: IncidentPipeline | None = None,
) -> int:
    """Play ``stream`` in real time, sending the current frame to the model every ``1 / fps`` seconds of video.

    If ``pipeline`` is given, every model result is fed to it so sustained hazards become incidents.
    """
    analyzer = analyzer or Analyzer(client, on_result=pipeline.deliver if pipeline else None)
    interval = 1.0 / fps
    next_sample = 0.0
    finished = True
    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                print("Video finished.")
                return 0
            position = stream.position_s
            if position < next_sample - interval:  # the video looped back to the start
                next_sample = 0.0
            if position >= next_sample:
                analyzer.submit(frame, position)
                # Next sample time, on the fps grid (skipping slots if playback jumped ahead).
                next_sample += interval * (int((position - next_sample) / interval) + 1)
            if not show:
                continue
            view = annotate(frame, analyzer.result, analyzer.error, pipeline.open_dangers() if pipeline else [])
            cv2.imshow("Kitchen hazards", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                finished = False
                return 0
            if key == ord("s") and analyzer.result_frame is not None:
                path = save_snapshot(annotate(analyzer.result_frame, analyzer.result))
                print(f"Snapshot saved to {path}")
    except KeyboardInterrupt:
        finished = False
        return 0
    finally:
        stream.release()
        if show:
            cv2.destroyAllWindows()
        # At the natural end, let requests still in flight finish so they count in the summary.
        analyzer.close(wait=finished)
        if pipeline is not None:
            pipeline.close()  # closes any incident still open when the video ended
        print(f"Sent {analyzer.sent} frames to the model, skipped {analyzer.skipped} (all request slots busy).")
        print(summarize(analyzer.history))


def main(argv: list[str] | None = None) -> int:
    load_env()  # before argparse, so .env can set ISAAC_QUERIES_PER_SEC
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", help="video file or URL to stream (omit to pick a file in a dialog)")
    parser.add_argument(
        "--fps",
        type=float,
        default=float(os.environ.get("ISAAC_QUERIES_PER_SEC", 2)),
        help="frames per second sent to the model (default ISAAC_QUERIES_PER_SEC or 2)",
    )
    parser.add_argument("--loop", action="store_true", help="restart the video when it ends, like a camera left on")
    parser.add_argument(
        "--model", help="Perceptron model id (default: PERCEPTRON_MODEL, else the fastest model your key can use)"
    )
    parser.add_argument("--list-models", action="store_true", help="print the models your API key can use and exit")
    parser.add_argument(
        "--hazards", nargs="+", choices=list(HAZARDS), default=list(HAZARDS), help="hazards to look for (default all)"
    )
    parser.add_argument("--no-window", action="store_true", help="print results only; no video window")
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="write incidents to the database at TIGER_DATABASE_URL (default: only print them)",
    )
    parser.add_argument("--show-prompt", action="store_true", help="print the prompt sent to the model")
    args = parser.parse_args(argv)

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not (os.environ.get("PERCEPTRON_API_KEY") or os.environ.get("FAL_KEY")):
        print("Set PERCEPTRON_API_KEY in your environment or .env first.", file=sys.stderr)
        return 2
    if args.list_models:
        for model_id in list_models():
            print(model_id)
        return 0

    video = args.video or pick_video_file()
    if not video:
        print("No video chosen. Pass one: python -m detection.demo clip.mp4", file=sys.stderr)
        return 2
    try:
        stream = VideoFileStream(resolve_video(video), loop=args.loop)
    except (OSError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    client = IsaacClient(model=args.model, hazards=args.hazards)
    print(f"Model: {client.model}")
    if args.show_prompt:
        print(client.system_prompt, end="\n\n")
    length = f"{format_position(stream.duration_s)}, " if stream.duration_s else ""
    print(
        f"Streaming {video} ({length}{stream.fps:g} fps) in real time, "
        f"sending {args.fps:g} frames/sec to the model. Press q to stop, s to save a snapshot."
    )
    if args.write_db:
        if not os.environ.get("TIGER_DATABASE_URL"):
            print("--write-db needs TIGER_DATABASE_URL in your environment or .env.", file=sys.stderr)
            return 2
        store = PostgresIncidentStore()
        print("Writing incidents to the incidents database.")
    else:
        store = StubIncidentStore()
        print("Incidents are printed, not written to the database (pass --write-db to write them).")
    pipeline = IncidentPipeline(IncidentReporter(IncidentDetector(fps=args.fps), store), fps=args.fps)
    return run_video(client, stream, args.fps, show=not args.no_window, pipeline=pipeline)


if __name__ == "__main__":
    sys.exit(main())
