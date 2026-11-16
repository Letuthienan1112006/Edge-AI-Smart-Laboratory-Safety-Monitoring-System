#!/usr/bin/env python3
"""Sweep worker count and find the saturation point (N4 in docs/experiment-ids.md).

Kukushkin & Ukhov's warning motivates this: past the optimal worker count,
more workers saturate memory bandwidth and heat rather than raising
throughput, and end-to-end latency (p99, not just mean) is where that
shows up first. So this script reports `dropped` frames and p99 alongside
throughput, not throughput alone -- a pool that silently buffers instead
of dropping would otherwise look strictly better as workers increase,
which is the exact lie this repo's harnesses are built to avoid.

    python3 scripts/sweep_workers.py --backend stub --sim-latency-ms 30 \\
            --workers 1,2,3,4,6,8 --mode thread --frames 200 \\
            --out runs/sweep_workers.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from labsafe.bench.runner import percentiles  # noqa: E402
from labsafe.bench.sysmon import read_temp_c, read_throttled  # noqa: E402
from labsafe.pipeline.pool import InferencePool  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--backend", default="stub", help="stub|ncnn|tflite|onnx|rknn")
ap.add_argument("--weights", default=None)
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--sim-latency-ms", type=float, default=0.0, help="stub only, to rehearse a sweep")
ap.add_argument("--workers", default="1,2,3,4", help="comma separated worker counts")
ap.add_argument("--mode", choices=["thread", "process"], default="thread")
ap.add_argument("--queue-size", type=int, default=16,
                 help="input queue depth per config; small on purpose -- a queue sized to "
                      "absorb every burst would hide the saturation point this script exists "
                      "to find")
ap.add_argument("--frames", type=int, default=200, help="frames submitted per worker count")
ap.add_argument("--source", default=None, help="video/image dir; omitted = synthetic frames")
ap.add_argument("--out", default="runs/sweep_workers.json")
a = ap.parse_args()


def load_frames(n: int) -> list[np.ndarray]:
    if a.source:
        import cv2

        p = Path(a.source)
        if p.is_dir():
            files = [f for f in sorted(p.rglob("*")) if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
            frames = [cv2.imread(str(f)) for f in files[:n]]
            frames = [f for f in frames if f is not None]
        else:
            cap = cv2.VideoCapture(str(p))
            frames = []
            while len(frames) < n:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            cap.release()
        if not frames:
            sys.exit(f"no frames loaded from {a.source}")
        return frames
    rng = np.random.default_rng(0)
    return [(rng.random((480, 640, 3)) * 255).astype(np.uint8) for _ in range(min(n, 16))]


def spec() -> dict:
    s = {"backend": a.backend, "imgsz": a.imgsz, "conf": a.conf}
    if a.weights:
        s["weights"] = a.weights
    if a.backend == "stub" and a.sim_latency_ms:
        s["sim_latency_ms"] = a.sim_latency_ms
    return s


frames = load_frames(max(a.frames, 16))
worker_counts = [int(w) for w in a.workers.split(",")]

before = read_throttled()
if before and not before["clean"]:
    print(f"WARNING: board already flagged {before['raw']} before the sweep started.\n"
          f"         Let it cool; these numbers will be thermally limited.\n")

print(f"backend={a.backend} mode={a.mode} frames/config={a.frames} configs={len(worker_counts)}")
print(f"{'workers':>8} {'mode':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'fps':>9} {'dropped':>8} {'temp':>6}")
print("-" * 72)

results = []
for workers in worker_counts:
    try:
        pool = InferencePool(spec(), ["person"], workers=workers, mode=a.mode,
                              queue_size=a.queue_size)
    except Exception as e:  # a missing runtime should skip, not abort the sweep
        print(f"{workers:>8} {a.mode:>8}   skipped: {type(e).__name__}: {e}")
        continue

    # Submit as fast as the producer can (this is the point of having
    # workers at all) and track per-frame submit time separately from
    # retrieval, so latency reflects real end-to-end queueing delay rather
    # than an artificial one-in-one-out round trip that would hide any
    # benefit from running workers in parallel.
    submit_t: dict[int, float] = {}
    t_wall0 = time.perf_counter()
    for i in range(a.frames):
        submit_t[i] = time.perf_counter()
        pool.submit(frames[i % len(frames)])

    end_to_end_ms: list[float] = []
    collected = 0
    deadline = time.perf_counter() + max(5.0, a.frames * 0.05)
    while collected < a.frames and time.perf_counter() < deadline:
        r = pool.get(timeout=0.2)
        if r is None:
            if pool.stats()["pending"] <= 0:
                break  # nothing left in flight (rest were dropped)
            continue
        idx, _dets = r
        collected += 1
        if idx in submit_t:
            end_to_end_ms.append((time.perf_counter() - submit_t[idx]) * 1000.0)
    wall = time.perf_counter() - t_wall0
    stats = pool.stats()
    pool.close()

    pc = percentiles(end_to_end_ms)
    temp = read_temp_c()
    fps = round(stats["completed"] / wall, 2) if wall > 0 else 0.0
    row = {
        "workers": workers, "mode": a.mode, "latency_ms": pc, "fps": fps,
        "dropped": stats["dropped"], "submitted": stats["submitted"],
        "completed": stats["completed"], "temp_c": temp,
        "per_worker_latency_ms": stats["per_worker_latency_ms"],
    }
    results.append(row)
    print(f"{workers:>8} {a.mode:>8} {pc.get('p50', 0):>8.2f} {pc.get('p90', 0):>8.2f} "
          f"{pc.get('p99', 0):>8.2f} {fps:>9.2f} {stats['dropped']:>8} {temp if temp else '-':>6}")

after = read_throttled()
if not results:
    sys.exit("\nno configuration ran. Check --backend and --weights.")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"results": results, "throttled_before": before, "throttled_after": after}, indent=2))

# Saturation point: first worker count where throughput rises <5% over the
# previous one. Everything after it is buying heat and p99, not speed.
by_workers = sorted(results, key=lambda r: r["workers"])
saturation = None
for prev, cur in zip(by_workers, by_workers[1:]):
    if prev["fps"] <= 0:
        continue
    gain = (cur["fps"] - prev["fps"]) / prev["fps"]
    if gain < 0.05:
        saturation = cur["workers"]
        break

print("-" * 72)
if saturation is not None:
    print(f"saturation point: {saturation} workers (throughput gain <5% vs the previous count)")
    print("  adding workers beyond this point mainly raises temperature and p99, "
          "not throughput -- see docs/optimization.md O8/Kukushkin & Ukhov.")
else:
    print("no saturation point found in this sweep -- throughput was still rising "
          "at the highest worker count tried; consider sweeping higher.")

if after and not after["clean"]:
    print(f"\n!! board reported {after['raw']} during the sweep -- rerun after cooling")
print(f"\nwritten: {a.out}")
