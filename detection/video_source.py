"""Play a video file as if it were a live camera.

``VideoFileStream`` has the same ``read()`` / ``release()`` shape as ``cv2.VideoCapture``, but paces
frames to wall-clock time: if the caller falls behind (a slow window, a busy CPU), frames are dropped
rather than played late, exactly as a webcam would. The model therefore sees the same thing it would
see if the video were happening in front of the camera.

``resolve_video`` turns what the user passed (a path, a direct video URL, or a page URL such as
YouTube) into something OpenCV can open, downloading page URLs with yt-dlp.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import cv2
import numpy as np

VIDEO_DIR = Path(__file__).parent / "videos"  # downloads land here (git-ignored)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


class VideoFileStream:
    def __init__(
        self,
        source: str,
        speed: float = 1.0,
        loop: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if speed <= 0:
            raise ValueError("speed must be positive")
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise OSError(f"could not open video: {source}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_s = frames / self.fps if frames > 0 else None
        self.speed = speed
        self.loop = loop
        self._clock = clock
        self._sleep = sleep
        self._start: float | None = None
        self._next_index = 0
        self.position_s = 0.0  # video time of the frame last returned
        self.dropped = 0  # frames skipped to keep up with real time

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._start is None:
            self._start = self._clock()

        # Skip frames whose time has already passed, like a camera that kept rolling.
        due_index = int((self._clock() - self._start) * self.speed * self.fps)
        while self._next_index < due_index:
            if not self._cap.grab():
                return self._at_end()
            self._next_index += 1
            self.dropped += 1

        # Ahead of schedule: wait until this frame is due.
        wait = self._start + self._next_index / (self.fps * self.speed) - self._clock()
        if wait > 0:
            self._sleep(wait)

        ok, frame = self._cap.read()
        if not ok:
            return self._at_end()
        self.position_s = self._next_index / self.fps
        self._next_index += 1
        return True, frame

    def _at_end(self) -> tuple[bool, np.ndarray | None]:
        if not self.loop or self._next_index == 0:
            return False, None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._start = None
        self._next_index = 0
        return self.read()

    def release(self) -> None:
        self._cap.release()


def is_url(value: str) -> bool:
    return urlparse(value).scheme in ("http", "https")


def resolve_video(value: str) -> str:
    """Return a local path or direct URL OpenCV can open for ``value``."""
    if not is_url(value):
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"no such video: {path}")
        return str(path)
    if urlparse(value).path.lower().endswith(VIDEO_EXTENSIONS):
        return value  # a direct link to a video file streams fine through OpenCV
    return download_video(value)


def download_video(url: str, out_dir: Path = VIDEO_DIR) -> str:
    """Download a page URL (YouTube, Vimeo, ...) with yt-dlp; reuses an earlier download."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("downloading from a page URL needs yt-dlp: pip install yt-dlp") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        # A single file with video in it, at most 720p: plenty for the model and quick to fetch.
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    from yt_dlp.utils import DownloadError

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            path = Path(ydl.prepare_filename(info))
            if path.exists():
                print(f"Using cached download {path}")
            else:
                print(f"Downloading {info.get('title') or url} ...")
                ydl.download([url])
    except DownloadError as exc:
        raise RuntimeError(f"could not download {url}: {exc}") from exc
    return str(path)


def pick_video_file() -> str | None:
    """Open a file dialog to choose a video; None if cancelled or no GUI toolkit is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    patterns = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
    chosen = filedialog.askopenfilename(title="Choose a video", filetypes=[("Videos", patterns), ("All files", "*")])
    root.destroy()
    return chosen or None


def format_position(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"
