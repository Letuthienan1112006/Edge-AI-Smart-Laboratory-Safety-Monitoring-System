"""Non-blocking annotated-video recording for live evidence runs."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class AsyncVideoRecorder:
    """Encode frames on a worker thread so disk I/O is outside inference.

    The queue is intentionally bounded.  If encoding cannot keep up, the
    newest frame is dropped and counted instead of stalling the AI loop and
    making its latency numbers look worse than they really are.
    """

    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int],
        fps: float = 20.0,
        queue_size: int = 32,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(size[0]), int(size[1]))
        self.fps = max(1.0, float(fps))
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, queue_size))
        self._error: str | None = None
        self._thread = threading.Thread(target=self._encode, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        self.submitted += 1
        if not self._thread.is_alive():
            self.dropped += 1
            return
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            self.dropped += 1

    def _encode(self) -> None:
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )
        if not writer.isOpened():
            self._error = f"could not open VideoWriter for {self.path}"
            return
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    break
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size)
                writer.write(frame)
                self.written += 1
        except Exception as exc:  # worker errors must reach the manifest
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            writer.release()

    def close(self) -> dict:
        # Make room for the sentinel without blocking the inference thread's
        # shutdown forever if a codec has stopped consuming the queue.
        while self._thread.is_alive():
            try:
                self._q.put(None, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=15.0)
        if self._thread.is_alive() and self._error is None:
            self._error = "encoder thread did not stop within 15 seconds"
        return self.stats()

    def stats(self) -> dict:
        return {
            "path": str(self.path),
            "fps": self.fps,
            "size": list(self.size),
            "frames_submitted": self.submitted,
            "frames_written": self.written,
            "frames_dropped": self.dropped,
            "error": self._error,
        }
