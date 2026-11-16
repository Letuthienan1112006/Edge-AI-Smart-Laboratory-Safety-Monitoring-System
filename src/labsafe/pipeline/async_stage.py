"""Run stage B / stage C on their own threads so they stop adding to the frame time.

Why: stage A, B and C each have their own RKNN context and can be pinned to a different
NPU core (`core_mask: 0 | 1 | 2`), but the router used to call them one after another in the
same thread, so the cores never worked at the same time. Measured on the board (INT8):
A 30.5 ms, B 12.5 ms *per person*, C 17.8 ms per call, i.e. 43 ms for one person and 68 ms
for three when serial. Stage B/C results only need to be fresh on the scale of their own
cadence (PPE re-check 2 s, fire/smoke 1 s), so they can lag by one job without changing any
alert; stage A must not wait for them.

The RKNN runtime releases the GIL while the NPU runs (ctypes call), so a worker thread really
overlaps with the main thread. Nothing here is lock-free cleverness: one bounded queue in,
one deque out. Jobs that arrive while the queue is full are DROPPED and counted -- an
unbounded queue would turn a slow stage into unbounded latency and RAM growth, which is the
failure mode this project already has once (KNOWN_ISSUES.md #27).
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Any, Callable

import numpy as np

from ..models.backends import Backend, Detection
from .ppe import PPEChecker, PPEConfig
from .tracker import Track


class AsyncWorker:
    """One daemon thread running `fn(job) -> result`; bounded queue; drop-newest when full."""

    def __init__(self, fn: Callable[[Any], Any], queue_size: int = 1, name: str = "async-stage"):
        self._fn = fn
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._results: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.submitted = 0
        self.dropped = 0
        self.completed = 0
        self.errors = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, job: Any) -> bool:
        try:
            self._q.put_nowait(job)
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def drain(self) -> list:
        """All results finished since the last call, oldest first."""
        with self._lock:
            out = list(self._results)
            self._results.clear()
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                res = self._fn(job)
            except Exception as e:  # a dead worker must be visible, not silent
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                continue
            with self._lock:
                self._results.append(res)
            self.completed += 1

    def stats(self) -> dict:
        return {"submitted": self.submitted, "dropped": self.dropped, "completed": self.completed,
                "errors": self.errors, "last_error": self.last_error}

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class AsyncPPEChecker(PPEChecker):
    """PPEChecker whose model calls run on a worker thread.

    `check()` never blocks on the model: it applies whatever finished since the last frame to
    the tracks, queues crops for the tracks that are due, and returns who is violating NOW
    (based on the most recent finished result per track). The worker never touches a Track.
    """

    def __init__(self, backend: Backend, cfg: PPEConfig, queue_size: int = 8):
        super().__init__(backend, cfg)
        self.worker = AsyncWorker(self._job, queue_size=queue_size, name="stage-b")

    def _job(self, job: tuple[int, np.ndarray]) -> tuple[int, list[Detection], float]:
        track_id, crop = job
        dets = self.backend(crop)
        return track_id, dets, self.backend.last_latency_ms

    def check(self, frame, tracks, now=None, force=False):  # type: ignore[override]
        import time

        now = time.time() if now is None else now
        by_id = {t.track_id: t for t in tracks}
        self.last_detection_count = 0
        # 1) apply finished jobs (a track may have disappeared meanwhile: ignore it)
        for track_id, dets, _lat in self.worker.drain():
            self.calls += 1
            self.last_detection_count += len(dets)
            tr = by_id.get(track_id)
            if tr is None:
                continue
            items = {d.cls_name for d in dets if d.score >= self.backend.conf}
            tr.ppe_status = {req: (req in items) for req in self.cfg.required}
        # 2) queue the tracks that are due
        for tr in tracks:
            if not force and not self.due(tr, now):
                self.skipped += 1
                continue
            crop, ok = self._crop(frame, tr.det)
            tr.last_ppe_check = now
            if ok:
                self.worker.submit((tr.track_id, crop.copy()))  # copy: the caller draws on `frame`
        return [tr for tr in tracks if self._violates(tr)]

    def close(self) -> None:
        self.worker.close()


class AsyncStageC:
    """Stage C (whole-frame fire/smoke) on its own thread; latest finished result wins."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.worker = AsyncWorker(self._job, queue_size=1, name="stage-c")

    def _job(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        dets = self.backend(frame)
        return dets, self.backend.last_latency_ms

    def submit(self, frame: np.ndarray) -> bool:
        return self.worker.submit(frame.copy())  # copy: the caller draws on `frame`

    def poll(self) -> tuple[list[Detection], float] | None:
        res = self.worker.drain()
        return res[-1] if res else None

    def close(self) -> None:
        self.worker.close()
