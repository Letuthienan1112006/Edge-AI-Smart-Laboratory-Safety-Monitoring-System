"""Async stage B / C: the frame loop must not wait for them, and results must still arrive.

Stub backends with a fake model latency stand in for the NPU; the point under test is the
threading contract (non-blocking submit, bounded queue, results applied on a later frame),
not any model output.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.backends import Detection, StubBackend  # noqa: E402
from labsafe.pipeline.async_stage import AsyncPPEChecker, AsyncStageC, AsyncWorker  # noqa: E402
from labsafe.pipeline.fall import FallConfig  # noqa: E402
from labsafe.pipeline.ppe import PPEChecker, PPEConfig  # noqa: E402
from labsafe.pipeline.router import AdaptiveRouter  # noqa: E402
from labsafe.pipeline.tracker import Track  # noqa: E402


class _Fixed(StubBackend):
    """Returns a fixed detection list after `sim_latency_ms`; records its thread."""

    def __init__(self, dets, latency_ms=0.0, names=("a",)):
        super().__init__(list(names), 64, 0.1, sim_latency_ms=0.0)
        self._dets, self._lat = dets, latency_ms
        self.threads: set[str] = set()

    def _infer(self, frame):
        import threading
        self.threads.add(threading.current_thread().name)
        time.sleep(self._lat / 1000.0)
        return list(self._dets)


def _det(name, score=0.9, box=(10, 10, 110, 210)):
    return Detection(*box, score=score, cls_id=0, cls_name=name)


def _wait(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_worker_drops_when_the_queue_is_full_and_counts_it():
    w = AsyncWorker(lambda x: (time.sleep(0.2), x)[1], queue_size=1)
    results = [w.submit(i) for i in range(6)]
    assert results.count(True) >= 1 and results.count(False) >= 3
    assert w.dropped == results.count(False)
    assert _wait(lambda: w.completed >= 1)
    w.close()


def test_worker_exception_is_counted_not_swallowed_silently():
    def boom(_):
        raise RuntimeError("npu died")
    w = AsyncWorker(boom, queue_size=2)
    w.submit(1)
    assert _wait(lambda: w.errors == 1)
    assert "npu died" in w.last_error
    w.close()


def test_stage_c_submit_returns_immediately_and_result_arrives_later():
    be = _Fixed([_det("fire", 0.8)], latency_ms=300, names=("fire", "smoke"))
    c = AsyncStageC(be)
    frame = np.zeros((48, 64, 3), np.uint8)
    t0 = time.perf_counter()
    assert c.submit(frame)
    assert (time.perf_counter() - t0) * 1000 < 50          # did not wait for the 300 ms model
    assert c.poll() is None
    assert _wait(lambda: c.worker.completed == 1)
    dets, lat = c.poll()
    assert [d.cls_name for d in dets] == ["fire"] and lat >= 250
    c.close()


def test_stage_c_runs_on_a_worker_thread_not_the_caller():
    be = _Fixed([], names=("fire", "smoke"))
    c = AsyncStageC(be)
    c.submit(np.zeros((48, 64, 3), np.uint8))
    assert _wait(lambda: c.worker.completed == 1)
    assert be.threads == {"stage-c"}
    c.close()


def _track(tid=1, box=(10, 10, 110, 210)):
    return Track(tid, _det("person", box=box), 0.0, 0.0)


def test_async_ppe_check_does_not_block_and_applies_result_on_a_later_call():
    be = _Fixed([_det("goggles", 0.9, (0, 0, 20, 20))], latency_ms=200, names=("goggles", "gloves"))
    chk = AsyncPPEChecker(be, PPEConfig(required=["goggles"], recheck_interval_s=0.0))
    frame = np.zeros((300, 200, 3), np.uint8)
    tr = _track()
    t0 = time.perf_counter()
    first = chk.check(frame, [tr], now=1.0)
    assert (time.perf_counter() - t0) * 1000 < 50
    assert first == [] and tr.ppe_status == {}               # no result yet: not a violation either
    assert _wait(lambda: chk.worker.completed == 1)
    chk.check(frame, [tr], now=1.0 + 99)                      # applies the finished job
    assert tr.ppe_status == {"goggles": True}
    chk.close()


def test_async_ppe_marks_a_violation_when_the_item_is_missing():
    be = _Fixed([], latency_ms=0, names=("goggles", "gloves"))
    chk = AsyncPPEChecker(be, PPEConfig(required=["goggles"], recheck_interval_s=100.0))
    frame = np.zeros((300, 200, 3), np.uint8)
    tr = _track()
    chk.check(frame, [tr], now=1000.0)                        # first check is due (last check was t=0)
    assert _wait(lambda: chk.worker.completed == 1)
    assert chk.check(frame, [tr], now=1001.0) == [tr]
    assert tr.ppe_status == {"goggles": False}
    chk.close()


def test_async_ppe_ignores_a_result_for_a_track_that_vanished():
    be = _Fixed([], latency_ms=50, names=("goggles", "gloves"))
    chk = AsyncPPEChecker(be, PPEConfig(required=["goggles"], recheck_interval_s=0.0))
    frame = np.zeros((300, 200, 3), np.uint8)
    chk.check(frame, [_track(7)], now=1.0)
    assert _wait(lambda: chk.worker.completed == 1)
    assert chk.check(frame, [], now=2.0) == []               # track 7 is gone: no crash, no violation
    chk.close()


def test_async_ppe_respects_the_recheck_interval_like_the_sync_checker():
    be = _Fixed([], names=("goggles", "gloves"))
    chk = AsyncPPEChecker(be, PPEConfig(required=["goggles"], recheck_interval_s=100.0))
    frame = np.zeros((300, 200, 3), np.uint8)
    tr = _track()
    for i in range(5):
        chk.check(frame, [tr], now=1000.0 + i)                # only the first is due
    assert _wait(lambda: chk.worker.completed == 1)
    assert chk.worker.submitted == 1 and chk.skipped == 4
    chk.close()


def test_router_frame_time_does_not_include_async_stage_latency():
    a = _Fixed([_det("person")], latency_ms=0, names=("person",))
    b = _Fixed([], latency_ms=150, names=("goggles", "gloves"))
    c = _Fixed([], latency_ms=300, names=("fire", "smoke"))
    ppe = PPEChecker(b, PPEConfig(required=["goggles"], recheck_interval_s=0.0, min_person_px=8))
    r = AdaptiveRouter(a, ppe, c, [], FallConfig(), mode="adaptive", stage_c_interval_s=0.0, async_stages=True)
    frame = np.zeros((300, 200, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(10):
        r.process(frame, now=1.0 + i * 0.05)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.6, f"10 frames took {elapsed:.2f}s -- stage B/C latency leaked into the frame loop"
    assert _wait(lambda: r.async_stats()["c"]["completed"] >= 1 and r.async_stats()["b"]["completed"] >= 1)
    r.process(frame, now=5.0)                                 # collects results
    assert r.stats.stage_calls["c"] >= 1 and r.stats.stage_calls["b"] >= 1
    assert r.async_stats()["c"]["errors"] == 0
    r.close()


def test_router_without_async_keeps_the_old_synchronous_behaviour():
    a = _Fixed([_det("person")], names=("person",))
    c = _Fixed([], latency_ms=100, names=("fire", "smoke"))
    r = AdaptiveRouter(a, None, c, [], FallConfig(), mode="adaptive", stage_c_interval_s=0.0)
    frame = np.zeros((300, 200, 3), np.uint8)
    t0 = time.perf_counter()
    r.process(frame, now=1.0)
    assert (time.perf_counter() - t0) >= 0.09                 # the 100 ms stage C ran inline
    assert r.stats.stage_calls["c"] == 1 and r.async_stats() == {}
