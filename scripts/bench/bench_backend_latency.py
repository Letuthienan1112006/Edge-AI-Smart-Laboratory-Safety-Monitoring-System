#!/usr/bin/env python3
"""Latency of the full backend call (preprocess + NPU + decode) per RKNN model, with the NPU/DDR/CPU clocks sampled DURING
the run, so a latency number always carries the frequency it was measured at (KNOWN_ISSUES #39, #42: the governors move
the same model by 5-7 ms).

    PYTHONPATH=src taskset -c 4-7 python3 scripts/bench/bench_backend_latency.py --models A.rknn:640 B.rknn:512 --out runs/lat.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from labsafe.models.backends import build_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="path.rknn:imgsz")
ap.add_argument("--images", default="data/labeled/boot018test/images")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--warmup", type=int, default=150)
ap.add_argument("--rounds", type=int, default=3, help="interleaved passes over the models (order effects)")
ap.add_argument("--label", default="")
ap.add_argument("--out", default=None)
a = ap.parse_args()


def rd(p: str) -> float | None:
    try:
        return float(Path(p).read_text().strip())
    except Exception:
        return None


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.rows: list[tuple] = []

    def run(self):
        while not self.stop.is_set():
            self.rows.append((rd("/sys/class/devfreq/fdab0000.npu/cur_freq"), rd("/sys/class/devfreq/dmc/cur_freq"),
                              rd("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq"),
                              rd("/sys/class/thermal/thermal_zone0/temp")))
            time.sleep(0.2)


def med(col, rows):
    v = [r[col] for r in rows if r[col] is not None]
    return round(float(np.median(v)) / (1e6 if col < 2 else 1e3), 0) if v else None


imgs = [cv2.imread(f) for f in sorted(glob.glob(a.images + "/*.jpg"))[:40]]
res = {}
for rnd in range(a.rounds):
    for spec in a.models:
        path, size = spec.rsplit(":", 1)
        be = build_backend({"backend": "rknn", "weights": path, "imgsz": int(size), "conf": 0.4, "iou": 0.6, "core_mask": "0"},
                           ["person", "lab_coat"])
        for k in range(a.warmup):
            be(imgs[k % len(imgs)])
        smp = Sampler(); smp.start()
        t = []
        for k in range(a.iters):
            s = time.perf_counter(); be(imgs[k % len(imgs)]); t.append((time.perf_counter() - s) * 1000)
        smp.stop.set(); smp.join()
        t = np.array(t)
        r = {"p50": round(float(np.median(t)), 2), "p95": round(float(np.percentile(t, 95)), 2), "p99": round(float(np.percentile(t, 99)), 2),
             "mean": round(float(t.mean()), 2), "npu_mhz": med(0, smp.rows), "ddr_mhz": med(1, smp.rows), "cpu4_mhz": med(2, smp.rows),
             "temp_c_start": round((smp.rows[0][3] or 0) / 1000, 1), "temp_c_end": round((smp.rows[-1][3] or 0) / 1000, 1)}
        res.setdefault(spec, []).append(r)
        print(f"[{a.label}] round {rnd} {Path(path).name} @{size}: " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"label": a.label, "results": res}, indent=1))
