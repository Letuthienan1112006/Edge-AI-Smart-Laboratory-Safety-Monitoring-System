#!/usr/bin/env python3
"""Summarise a scripts/ram_logger.py CSV per process segment (one segment per PID): milestones, slope after the first hour, saturation hint."""
import csv
import sys
from datetime import datetime

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
segs = {}
for r in rows:
    if r["pid"] and r["pid"] != "0" and r["vmrss_kb"]:
        segs.setdefault(r["pid"], []).append(r)
print(f"{len(rows)} samples, {len(segs)} process segment(s); service restarts counted by systemd at the end: {rows[-1]['nrestarts']}")
for pid, rs in segs.items():
    up = np.array([float(r["uptime_s"]) for r in rs]) / 60.0
    rss = np.array([float(r["vmrss_kb"]) for r in rs]) / 1024.0
    hwm = max(float(r["vmhwm_kb"]) for r in rs) / 1024.0
    print(f"\nPID {pid}: {up[0]:.0f}-{up[-1]:.0f} min of uptime, RSS start {rss[0]:.0f} MB, end {rss[-1]:.0f} MB, max {rss.max():.0f} MB, VmHWM {hwm:.0f} MB")
    print("  milestone (min): " + "  ".join(f"{m}:{np.interp(m, up, rss):.0f}MB" for m in (0, 30, 60, 120, 240, 360) if m <= up[-1]))
    for lo, hi, name in ((0, 60, "first hour"), (60, 1e9, "after the first hour")):
        sel = (up >= lo) & (up < hi)
        if sel.sum() >= 5 and np.ptp(up[sel]) > 10:
            slope = np.polyfit(up[sel], rss[sel], 1)[0]
            print(f"  slope {name}: {slope:+.3f} MB/min  (n={sel.sum()})")
    pss = [(float(r["uptime_s"]) / 60, float(r["pss_kb"]) / 1024, float(r["pss_anon_kb"]) / 1024, float(r["private_dirty_kb"]) / 1024) for r in rs if r["pss_kb"]]
    if pss:
        print("  PSS / anon PSS / private dirty (MB) at start -> end: "
              f"{pss[0][1]:.0f}/{pss[0][2]:.0f}/{pss[0][3]:.0f} -> {pss[-1][1]:.0f}/{pss[-1][2]:.0f}/{pss[-1][3]:.0f}")
    temps = [float(r["temp_c"]) for r in rs if r["temp_c"]]
    fps = [float(r["fps"]) for r in rs if r["fps"]]
    if temps: print(f"  temp {temps[0]:.1f} -> {temps[-1]:.1f} C (max {max(temps):.1f})" + (f", FPS mean {np.mean(fps):.1f} min {min(fps):.1f}" if fps else ""))
