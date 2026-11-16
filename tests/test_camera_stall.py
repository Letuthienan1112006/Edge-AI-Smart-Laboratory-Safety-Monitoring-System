"""A camera that stops delivering must not leave the system analysing a frozen frame (stall watchdog)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.io import camera as cam_mod  # noqa: E402


class FakeCap:
    """Delivers noisy frames until `dead` is set on its instance, then fails every read."""
    instances: list["FakeCap"] = []
    replacement_alive = True      # does a freshly opened device deliver?

    def __init__(self, device):
        self.dead = bool(FakeCap.instances) and not FakeCap.replacement_alive
        self.released = False
        self.opened = True
        FakeCap.instances.append(self)

    def isOpened(self):
        return self.opened

    def set(self, *a):
        return True

    def get(self, *a):
        return 640.0

    def read(self):
        if self.dead:
            time.sleep(0.005)
            return False, None
        time.sleep(0.005)
        return True, (np.random.rand(48, 64, 3) * 255).astype(np.uint8)

    def release(self):
        self.released = True


def test_stalled_camera_is_reopened_and_reports_no_frame(monkeypatch):
    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", FakeCap)
    FakeCap.instances.clear()
    FakeCap.replacement_alive = False   # the cable is still out when the watchdog reopens
    cam = cam_mod.CameraSource(0, 64, 48, 30, stall_s=0.3)
    try:
        ok, _ = cam.read(wait_new=True)
        assert ok
        FakeCap.instances[0].dead = True          # unplugged
        time.sleep(0.5)                           # longer than stall_s
        ok, frame = cam.read(wait_new=True, timeout_s=0.05)
        assert not ok and frame is None           # no frozen frame served as if alive
        time.sleep(0.5)
        assert cam.reopens >= 1                   # the grabber tried to get the device back
        assert FakeCap.instances[0].released
        # cable back in: the next reopen delivers, and reading resumes without restarting the app
        FakeCap.replacement_alive = True
        FakeCap.instances[-1].dead = False
        deadline = time.time() + 2.0
        while time.time() < deadline:
            ok, frame = cam.read(wait_new=True, timeout_s=0.05)
            if ok:
                break
        assert ok and frame is not None
    finally:
        cam.release()


def test_open_is_retried_until_the_device_appears(monkeypatch):
    """A camera still held by the previous process (restart) or still enumerating (boot) must not fail the start-up on the first try."""
    calls = {"n": 0}

    class LateCap(FakeCap):
        def __init__(self, device):
            super().__init__(device)
            calls["n"] += 1
            self.opened = calls["n"] >= 3          # the first two opens fail

    monkeypatch.setattr(cam_mod.cv2, "VideoCapture", LateCap)
    real_sleep = time.sleep
    monkeypatch.setattr(cam_mod.time, "sleep", lambda s: real_sleep(0.001 if s >= 1.0 else s))   # do not wait the real 1 s between tries
    FakeCap.instances.clear()
    FakeCap.replacement_alive = True
    cam = cam_mod.CameraSource(0, 64, 48, 30, open_retry_s=5.0)
    try:
        assert calls["n"] >= 3 and cam.cap.isOpened()
    finally:
        cam.release()
