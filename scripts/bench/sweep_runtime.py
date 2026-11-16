#!/usr/bin/env python3
"""Sweep runtime knobs on the target device and print the Pareto frontier.

The knobs that change latency on a Pi 4 without touching accuracy at all
are the ones most often left at their defaults:

  threads   4 cores, but the capture thread, the OS and the dashboard also
            want one. 4 inference threads frequently loses to 3, because
            the runtime's threads start fighting the pipeline's own. This
            is free performance and nobody measures it.
  backend   NCNN vs TFLite vs ONNX on the same weights differ by more than
            most model changes do.
  imgsz     the only knob here that costs accuracy, which is why it is
            reported per accuracy slice elsewhere, not just as latency.

Everything here is measured on ONE fixed set of frames so the comparison
is clean, and each configuration is measured with its own warmup, because
the first inferences after a backend switch are never representative.

    python3 scripts/bench/sweep_runtime.py --backend stub --threads 1,2,3,4 --imgsz 320,416
    python3 scripts/bench/sweep_runtime.py --backend ncnn \
        --weights runs/train/stage_a/weights/best_ncnn_model \
        --threads 1,2,3,4 --imgsz 256,320,416 --frames 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--backend", default="stub", help="stub|ncnn|tflite|onnx|ultralytics")
ap.add_argument("--weights", default=None)
ap.add_argument("--threads", default="1,2,3,4")
ap.add_argument("--imgsz", default="320")
ap.add_argument("--frames", type=int, default=50, help="measured inferences per config")
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--source", default=None, help="video/image dir; omitted = synthetic frames")
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--sim-latency-ms", type=float, default=0.0, help="stub only, to rehearse a sweep")
ap.add_argument("--out", default="runs/sweep.json")
a = ap.parse_args()

from labsafe.bench.runner import percentiles  # noqa: E402
from labsafe.bench.sysmon import read_temp_c, read_throttled  # noqa: E402
from labsafe.models.backends import build_backend  # noqa: E402


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
    # Synthetic frames are fine for a LATENCY sweep: the runtime does the
    # same work regardless of content. They are not fine for accuracy.
    rng = np.random.default_rng(0)
    return [(rng.random((480, 640, 3)) * 255).astype(np.uint8) for _ in range(min(n, 16))]


frames = load_frames(max(a.frames, 16))
thread_list = [int(t) for t in a.threads.split(",")]
imgsz_list = [int(s) for s in a.imgsz.split(",")]

before = read_throttled()
if before and not before["clean"]:
    print(f"WARNING: board already flagged {before['raw']} before the sweep started.\n"
          f"         Let it cool; these numbers will be thermally limited.\n")

print(f"backend={a.backend}  frames={a.frames}  configs={len(thread_list)*len(imgsz_list)}")
print(f"{'threads':>8} {'imgsz':>6} {'p50 ms':>9} {'p90 ms':>9} {'p99 ms':>9} {'fps':>8} {'temp':>6}")
print("-" * 60)

results = []
for imgsz in imgsz_list:
    for threads in thread_list:
        # These must be set before the runtime builds its thread pool, and
        # they are read at import/session time by several backends -- which
        # is why each config constructs a fresh backend rather than reusing one.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        try:
            spec = {"backend": a.backend, "imgsz": imgsz, "conf": a.conf, "threads": threads}
            if a.weights:
                spec["weights"] = a.weights
            if a.backend == "stub" and a.sim_latency_ms:
                spec["sim_latency_ms"] = a.sim_latency_ms
            be = build_backend(spec, ["person"])
        except Exception as e:  # a missing runtime should skip, not abort the sweep
            print(f"{threads:>8} {imgsz:>6}   skipped: {type(e).__name__}: {e}")
            continue

        for i in range(a.warmup):
            be(frames[i % len(frames)])

        lat = []
        t0 = time.perf_counter()
        for i in range(a.frames):
            f = frames[i % len(frames)]
            t = time.perf_counter()
            be(f)
            lat.append((time.perf_counter() - t) * 1000.0)
        wall = time.perf_counter() - t0

        pc = percentiles(lat)
        temp = read_temp_c()
        row = {"backend": a.backend, "threads": threads, "imgsz": imgsz,
               "latency_ms": pc, "fps": round(a.frames / wall, 2), "temp_c": temp}
        results.append(row)
        print(f"{threads:>8} {imgsz:>6} {pc['p50']:>9.2f} {pc['p90']:>9.2f} "
              f"{pc['p99']:>9.2f} {row['fps']:>8.2f} {temp if temp else '-':>6}")

after = read_throttled()
if not results:
    sys.exit("\nno configuration ran. Check --backend and --weights.")

best = min(results, key=lambda r: r["latency_ms"]["p50"])
worst = max(results, key=lambda r: r["latency_ms"]["p50"])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"results": results, "throttled_before": before, "throttled_after": after}, indent=2))

print("-" * 60)
print(f"fastest : threads={best['threads']} imgsz={best['imgsz']}  "
      f"p50={best['latency_ms']['p50']:.2f} ms")
print(f"slowest : threads={worst['threads']} imgsz={worst['imgsz']}  "
      f"p50={worst['latency_ms']['p50']:.2f} ms")
print(f"spread  : {worst['latency_ms']['p50']/max(1e-9, best['latency_ms']['p50']):.2f}x "
      f"from runtime knobs alone -- no accuracy spent")

same_size = [r for r in results if r["imgsz"] == best["imgsz"]]
if len(same_size) > 1:
    by_threads = sorted(same_size, key=lambda r: r["threads"])
    print("\nthread scaling at imgsz=%d (does more threads actually help?):" % best["imgsz"])
    base = by_threads[0]["latency_ms"]["p50"]
    for r in by_threads:
        print(f"  {r['threads']} thread(s): {r['latency_ms']['p50']:>7.2f} ms  "
              f"({base / max(1e-9, r['latency_ms']['p50']):.2f}x vs 1 thread)")
    print("  If 4 threads is not the fastest, say so in the report: on a 4-core Pi 4\n"
          "  the capture thread and the OS need a core too.")

if after and not after["clean"]:
    print(f"\n!! board reported {after['raw']} during the sweep -- rerun after cooling")
print(f"\nwritten: {a.out}")
