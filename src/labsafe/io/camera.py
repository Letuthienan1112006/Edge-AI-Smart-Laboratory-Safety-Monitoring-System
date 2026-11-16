"""Frame source: threaded for the camera, strictly sequential for files.

The threaded grabber exists so `cap.read()` latency (USB transfer, which
on a Pi 4 is not negligible) does not land inside the measured inference
loop. For a video file it is switched off deliberately: dropping frames
would silently change the input of a benchmark and make two runs of the
"same" experiment incomparable.

Trade-off of the threaded grabber: it is a "latest wins" mailbox, not a
queue. If the pipeline loop processes faster than the camera delivers
frames (any USB webcam, worse in low light), `read()` can and will hand
back the same frame more than once. `CameraSource.duplicate_reads` and
`CameraSource.t_open` exist so callers can tell "we measured N frames per
second" apart from "we called read() N times per second and some of those
were the same picture" -- see `app.camera_stats()`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("labsafe.camera")


class CameraSource:
    def __init__(
        self,
        device: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        threaded: bool | None = None,
        realtime: bool = False,
        loop: bool = False,
        stall_s: float = 3.0,
        open_retry_s: float = 20.0,
    ):
        self.is_file = isinstance(device, str) and Path(device).exists()
        # Video files only, for demos (the dashboard): play at the file's own FPS instead of as fast as the model
        # allows, and start over at the end. Benchmarks keep the defaults (False): there the file must be consumed
        # at full speed and end, see KNOWN_ISSUES #6.
        self.realtime = realtime and self.is_file
        self.loop = loop and self.is_file
        self._next_due = 0.0
        self.device = device
        self._req = (width, height, fps)
        # A live camera that stops delivering (unplugged, USB reset) must not leave the system analysing its last frame
        # forever: after `stall_s` without a frame the grabber thread reopens the device and read() reports no frame.
        self.stall_s = float(stall_s)
        self.reopens = 0
        self.last_frame_t = time.time()
        self._last_reopen_t = 0.0
        self.cap = self._open()
        # A camera that is still being released by the previous process (service restart) or still enumerating (boot) fails the first open
        # for a few seconds; retrying here keeps that out of systemd's restart budget (5 starts in 5 minutes, then it gives up).
        t_open = time.time()
        while not self.cap.isOpened() and not self.is_file and time.time() - t_open < open_retry_s:
            time.sleep(1.0)
            self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source: {device!r}")
        # A resolution request can be silently ignored (GStreamer backends do
        # this routinely). Running a benchmark at 1280x720 while the config
        # and the paper both say 640x480 invalidates every latency number in
        # it, and nothing else in the system would ever notice.
        if not self.is_file:
            got_w, got_h = self.resolution
            if (got_w, got_h) != (width, height):
                log.warning(
                    "camera ignored the resolution request: asked %dx%d, got %dx%d. "
                    "Benchmark numbers must be reported at the ACTUAL resolution.",
                    width, height, got_w, got_h,
                )
        self.threaded = (not self.is_file) if threaded is None else threaded
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)   # notified on every new frame
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_read = 0
        # Wall-clock time the device was opened -- used to compute a real
        # capture FPS (frames_read / elapsed) independent of how fast the
        # pipeline loop calls read(). Set before the grabber thread starts
        # so it always predates the first frame it could possibly produce.
        self.t_open = time.time()
        # See read()'s docstring below: counts frames handed back more than
        # once because the pipeline asked faster than the camera produced.
        self.duplicate_reads = 0
        self._last_returned_seq = -1
        if self.threaded:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._await_first()

    def _open(self) -> "cv2.VideoCapture":
        cap = cv2.VideoCapture(self.device)
        if cap.isOpened() and not self.is_file:
            width, height, fps = self._req
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            # MJPG: the C270 delivers it natively; YUYV at 640x480 costs
            # USB bandwidth and shows up as capture latency on a Pi 4.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        return cap

    def _reopen(self) -> None:
        log.warning("camera delivered no frame for %.1f s: reopening %r", time.time() - self.last_frame_t, self.device)
        try:
            self.cap.release()
        except Exception:   # a dead device can raise on release; we are replacing it anyway
            pass
        self.cap = self._open()
        self.reopens += 1
        self._last_reopen_t = time.time()   # if it is still dead, the next attempt comes after another stall_s

    def _await_first(self, timeout_s: float = 5.0, min_std: float = 2.0) -> None:
        """Wait for the first USABLE frame, not just the first frame.

        Verified for real on a Logitech C270 (Orange Pi 5, 2026-09-28): a
        UVC webcam's very first frames after open are a degenerate,
        perfectly uniform "blank" image while auto-exposure has not
        converged yet -- byte-identical across many frames, not just dim.
        `frame is not None` is true for these immediately (sub-second), so
        the old check returned long before the sensor was actually
        delivering real data. Downstream code (and the person watching a
        black preview) had no way to tell "no camera" from "camera not
        warmed up yet" -- they looked identical. A degenerate frame has
        std ~ 0 (every pixel the same value); a real one, even in a dim
        room, does not. `min_std` catches this without a magic sleep
        duration that would depend on this specific camera/lighting.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is not None and float(frame.std()) >= min_std:
                return
            time.sleep(0.01)
        raise RuntimeError(
            f"no usable frame (std >= {min_std}) from camera within {timeout_s}s -- "
            "either no camera is attached, or it is still producing a blank "
            "auto-exposure warm-up frame after the full timeout"
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if not self.is_file and time.time() - max(self.last_frame_t, self._last_reopen_t) > self.stall_s:
                    self._reopen()
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self.last_frame_t = time.time()
                self.frames_read += 1
                self._cond.notify_all()

    def read(self, wait_new: bool = False, timeout_s: float = 0.25) -> tuple[bool, np.ndarray | None]:
        """Return the latest frame.

        In threaded mode this is a snapshot of whatever the grabber thread
        last wrote, not a queue -- if the pipeline loop is faster than the
        camera (a C270 tops out at ~30 fps, less in a dim room where
        auto-exposure stretches the shutter), the same frame is handed back
        more than once. `duplicate_reads` counts that so `fps_sustained`
        can be told apart from "the loop is just spinning on stale data"
        (a relative of KNOWN_ISSUES.md #6, which caught the same shape of
        bug for an exhausted video file). This does NOT change what is
        returned -- only adding an observation, not a different capture
        policy, which is a product decision out of scope here.

        `wait_new=True` is the opt-in fix for that: block (up to `timeout_s`)
        until the grabber has produced a frame this caller has not seen, so a
        loop faster than the camera processes every frame ONCE instead of
        re-running the model on a stale one. Measured on the board (C270,
        INT8 full system): 20.5% of loop iterations were repeats, i.e. 20% of
        NPU work and power spent re-detecting an unchanged picture, and the
        reported FPS was 37 while the camera delivered 30. A timeout returns
        the stale frame (and counts a duplicate) rather than hanging when the
        camera stalls. Note the wait is part of the caller's iteration time.
        """
        if self.threaded:
            with self._cond:
                if wait_new and self.frames_read == self._last_returned_seq and not self._stop.is_set():
                    self._cond.wait_for(
                        lambda: self.frames_read != self._last_returned_seq or self._stop.is_set(),
                        timeout=timeout_s,
                    )
                if self._frame is None:
                    return False, None
                if time.time() - self.last_frame_t > self.stall_s:
                    return False, None   # stalled camera: no frame is better than a frozen one that looks alive
                if self.frames_read == self._last_returned_seq:
                    self.duplicate_reads += 1
                self._last_returned_seq = self.frames_read
                return True, self._frame.copy()
        if self.realtime:
            period = 1.0 / max(1.0, float(self.cap.get(cv2.CAP_PROP_FPS) or 0) or 10.0)
            now = time.perf_counter()
            if self._next_due > now:
                time.sleep(self._next_due - now)
            self._next_due = max(now, self._next_due) + period
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok:
            self.frames_read += 1
        return ok, frame

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> "CameraSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
