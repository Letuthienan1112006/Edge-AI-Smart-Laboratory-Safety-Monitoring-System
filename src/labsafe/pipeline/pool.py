"""A worker pool for parallel inference -- the substrate for N4 (worker
sweep) and, eventually, per-stage engine placement (Phase 5 stops short of
wiring that in; see `optimize/engine_map.py`).

Two silent failure modes this exists to make impossible:

  * A pool that buffers instead of dropping under load reports a great
    FPS while end-to-end latency grows without bound -- the classic async
    pipeline bug (Kukushkin & Ukhov: past the optimal worker count you
    saturate memory bandwidth and heat, not throughput). So the queue
    policy here is drop-oldest, and drops are counted and surfaced, never
    silently absorbed.
  * Sharing one backend instance across workers races on `last_latency_ms`
    and, for `rknnlite`, cannot work at all -- an RKNN context is not
    fork-safe and not thread-safe across concurrent `inference()` calls.
    So every worker builds its OWN backend, in both modes:
      mode="thread"  : for backends that release the GIL during inference
                       (rknnlite, onnxruntime) -- concurrency without a
                       second process per worker.
      mode="process" : for backends that hold the GIL (plain numpy/cv2
                       heavy paths) or that cannot share any state across
                       threads at all.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Any

import numpy as np

from ..bench.runner import percentiles
from ..models.backends import Detection, build_backend

_SENTINEL_POLL_S = 0.1  # how often an idle worker checks the stop flag


def _run_worker(
    spec: dict[str, Any],
    names: list[str],
    worker_id: int,
    input_q: Any,
    output_q: Any,
    stop_flag: Any,
) -> None:
    """Body shared by both the thread and the process worker.

    Module-level (not a closure, not a bound method) on purpose: with the
    `spawn`/`forkserver` multiprocessing start methods the target must be
    picklable, and a closure over a live queue/backend is not.
    """
    backend = build_backend(spec, names)
    try:
        while True:
            if stop_flag.is_set():
                return
            try:
                item = input_q.get(timeout=_SENTINEL_POLL_S)
            except queue.Empty:
                continue
            idx, frame = item
            t0 = time.perf_counter()
            try:
                dets = backend(frame)
            except Exception as e:  # noqa: BLE001 - a dead worker must not hang the pool
                print(f"[pool worker {worker_id}] inference failed on frame {idx}: "
                      f"{type(e).__name__}: {e}")
                continue
            latency_ms = (time.perf_counter() - t0) * 1000.0
            output_q.put((idx, dets, latency_ms, worker_id))
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


class InferencePool:
    """N workers running the same backend spec in parallel.

    `submit()` is non-blocking and never raises: when the input queue is
    full the OLDEST queued frame is dropped to make room, and the drop is
    counted in `stats()["dropped"]`. `get()` returns `(frame_index,
    detections)` so a caller can tell which submitted frame a result
    belongs to once results start arriving out of order.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        names: list[str],
        workers: int,
        mode: str = "thread",
        queue_size: int = 4,
    ):
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if mode not in {"thread", "process"}:
            raise ValueError(f"mode must be 'thread' or 'process', got {mode!r}")

        self.spec = spec
        self.names = list(names)
        self.workers = workers
        self.mode = mode
        self._next_idx = 0
        self._submitted = 0
        self._dropped = 0
        self._completed = 0
        self._worker_latency_ms: dict[int, list[float]] = {i: [] for i in range(workers)}
        self._buffer: list[tuple[int, list[Detection]]] = []

        if mode == "thread":
            self._input_q: Any = queue.Queue(maxsize=max(1, queue_size))
            self._output_q: Any = queue.Queue()
            self._stop = threading.Event()
            worker_cls, self._Empty, self._Full = threading.Thread, queue.Empty, queue.Full
        else:
            # "fork" (the Linux default): cheap to start, and safe here
            # because the backend itself is only constructed AFTER the fork,
            # inside `_run_worker` -- no NPU/GPU handle exists in the parent
            # to be duplicated. Pinned explicitly rather than left as
            # whatever the platform default is, so behaviour does not shift
            # under us if that default ever changes.
            ctx = mp.get_context("fork")
            self._input_q = ctx.Queue(maxsize=max(1, queue_size))
            self._output_q = ctx.Queue()
            self._stop = ctx.Event()
            worker_cls, self._Empty, self._Full = ctx.Process, queue.Empty, queue.Full

        self._procs = []
        for i in range(workers):
            p = worker_cls(
                target=_run_worker,
                args=(spec, self.names, i, self._input_q, self._output_q, self._stop),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    # -- producer side ------------------------------------------------------

    def submit(self, frame: np.ndarray) -> None:
        idx = self._next_idx
        self._next_idx += 1
        self._submitted += 1
        try:
            self._input_q.put_nowait((idx, frame))
            return
        except self._Full:
            pass
        # Full: drop the oldest queued frame to make room, then enqueue the
        # new one. A pool that blocks here instead would turn "submit" into
        # a latency spike that the caller never asked for.
        try:
            self._input_q.get_nowait()
            self._dropped += 1
        except self._Empty:
            pass
        try:
            self._input_q.put_nowait((idx, frame))
        except self._Full:  # lost a race with a worker refilling it; drop the new one
            self._dropped += 1

    # -- consumer side --------------------------------------------------------

    def _drain(self) -> None:
        while True:
            try:
                idx, dets, latency_ms, worker_id = self._output_q.get_nowait()
            except self._Empty:
                return
            self._completed += 1
            self._worker_latency_ms.setdefault(worker_id, []).append(latency_ms)
            self._buffer.append((idx, dets))

    def get(self, timeout: float | None = None) -> tuple[int, list[Detection]] | None:
        self._drain()
        if self._buffer:
            return self._buffer.pop(0)
        if timeout is None:
            return None
        try:
            idx, dets, latency_ms, worker_id = self._output_q.get(timeout=timeout)
        except self._Empty:
            return None
        self._completed += 1
        self._worker_latency_ms.setdefault(worker_id, []).append(latency_ms)
        return idx, dets

    def stats(self) -> dict[str, Any]:
        self._drain()
        return {
            "submitted": self._submitted,
            "completed": self._completed,
            "dropped": self._dropped,
            "pending": self._submitted - self._completed - self._dropped,
            "per_worker_latency_ms": {
                w: percentiles(lat) for w, lat in self._worker_latency_ms.items()
            },
        }

    def close(self) -> None:
        """Stop every worker and join it. Must leave no live thread/process
        behind -- a daemon left running silently keeps burning a CPU core
        for the rest of the process's life."""
        self._stop.set()
        for p in self._procs:
            p.join(timeout=_SENTINEL_POLL_S * 5)
        self._procs = []
