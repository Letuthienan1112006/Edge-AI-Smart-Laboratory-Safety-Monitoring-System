"""InferencePool tests -- everything here uses `stub` + `sim_latency_ms`,
never a real model, per plan §2.6. Kept fast: no test should need more
than a couple hundred milliseconds of simulated latency.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.pipeline.pool import InferencePool  # noqa: E402

FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


def _spec(sim_latency_ms: float = 0.0) -> dict:
    return {"backend": "stub", "imgsz": 64, "conf": 0.3, "sim_latency_ms": sim_latency_ms}


def _drain_all(pool: InferencePool, expect: int, timeout_s: float = 5.0) -> list:
    out = []
    deadline = time.time() + timeout_s
    while len(out) < expect and time.time() < deadline:
        r = pool.get(timeout=0.2)
        if r is not None:
            out.append(r)
    return out


def test_pool_returns_every_submitted_frame_when_not_saturated():
    pool = InferencePool(_spec(), ["person"], workers=2, mode="thread", queue_size=32)
    try:
        for _ in range(10):
            pool.submit(FRAME)
        results = _drain_all(pool, expect=10)
        assert len(results) == 10
        assert {idx for idx, _ in results} == set(range(10))
    finally:
        pool.close()


def test_pool_counts_drops_when_saturated():
    pool = InferencePool(_spec(sim_latency_ms=50), ["person"], workers=1,
                          mode="thread", queue_size=1)
    try:
        for _ in range(20):
            pool.submit(FRAME)
        time.sleep(0.3)  # let the lone slow worker fall behind
        stats = pool.stats()
        assert stats["dropped"] > 0
        assert stats["submitted"] == stats["completed"] + stats["dropped"] + stats["pending"]
    finally:
        pool.close()


def test_pool_with_two_workers_is_faster_than_one():
    n = 12
    latency_ms = 40

    def run(workers: int) -> float:
        pool = InferencePool(_spec(sim_latency_ms=latency_ms), ["person"],
                              workers=workers, mode="thread", queue_size=n)
        try:
            t0 = time.perf_counter()
            for _ in range(n):
                pool.submit(FRAME)
            _drain_all(pool, expect=n, timeout_s=10.0)
            return time.perf_counter() - t0
        finally:
            pool.close()

    one = run(1)
    two = run(2)
    assert one / two >= 1.3, f"expected >=1.3x speedup, got {one/two:.2f}x ({one=:.3f}s {two=:.3f}s)"


def test_close_leaves_no_live_threads():
    before = threading.active_count()
    pool = InferencePool(_spec(), ["person"], workers=3, mode="thread", queue_size=8)
    for _ in range(5):
        pool.submit(FRAME)
    pool.close()
    time.sleep(0.05)
    assert threading.active_count() == before


def test_process_mode_builds_one_backend_per_worker():
    pool = InferencePool(_spec(), ["person"], workers=2, mode="process", queue_size=8)
    try:
        for _ in range(6):
            pool.submit(FRAME)
        results = _drain_all(pool, expect=6, timeout_s=10.0)
        assert len(results) == 6
        for _, dets in results:
            assert isinstance(dets, list)
    finally:
        pool.close()


def test_results_carry_frame_index():
    pool = InferencePool(_spec(), ["person"], workers=1, mode="thread", queue_size=8)
    try:
        for _ in range(5):
            pool.submit(FRAME)
        results = _drain_all(pool, expect=5)
        assert sorted(idx for idx, _ in results) == [0, 1, 2, 3, 4]
    finally:
        pool.close()


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        InferencePool(_spec(), ["person"], workers=1, mode="gevent")
