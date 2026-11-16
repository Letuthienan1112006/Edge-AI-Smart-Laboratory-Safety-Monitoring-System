#!/usr/bin/env python3
"""Aggregate scripts/bench/ddr_ab_board.sh output: mean over rounds of FPS, frame-latency p50/p99/max, CPU %, end temperature."""
import json, statistics as st, sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for m in sorted(out.glob("*_r*/metrics.json")):
    mode = m.parent.name.split("_r")[0]
    d = json.loads(m.read_text())
    rows.setdefault(mode, []).append((d["fps_sustained"], d["latency_ms"]["p50"], d["latency_ms"]["p99"], d["latency_ms"]["max"],
                                     d["system"]["cpu_percent"]["mean"], d["system"]["temp_c"]["end"], d["wall_s"]))
print(f"{'mode':9s} {'n':>2s} {'FPS':>7s} {'p50 ms':>8s} {'p99 ms':>8s} {'max ms':>8s} {'CPU %':>7s} {'temp end':>9s}")
for mode, v in rows.items():
    cols = list(zip(*v))
    f = lambda i: st.mean(cols[i])
    print(f"{mode:9s} {len(v):2d} {f(0):7.2f} {f(1):8.2f} {f(2):8.2f} {f(3):8.2f} {f(4):7.1f} {f(5):9.1f}   per-run p99: {[round(x,1) for x in cols[2]]}")
