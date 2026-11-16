#!/usr/bin/env python3
"""Independent RAM logger for the live service (runs OUTSIDE the inference process, survives its restarts).

    python3 scripts/ram_logger.py [--unit labsafe-live] [--interval 60] [--smaps-every 900] [--out runs/ramlog/ram.csv]

Every `interval` s: VmRSS/VmHWM/VmSwap/Threads from /proc/<PID>/status, CPU % from /proc/<PID>/stat, SoC temperature, and the app's own FPS
from its dashboard /status. Every `smaps-every` s additionally /proc/<PID>/smaps_rollup (PSS, anon/file PSS, private dirty): the rollup is cheap
but walks the page tables, so it is not read every minute. The PID is asked from systemd each time, so an app restart simply starts a new segment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--unit", default="labsafe-live")
ap.add_argument("--pid", type=int, default=None, help="fixed PID instead of asking systemd (tests)")
ap.add_argument("--interval", type=float, default=60)
ap.add_argument("--smaps-every", type=float, default=900)
ap.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: run until killed)")
ap.add_argument("--out", default=None)
ap.add_argument("--status-url", default="http://127.0.0.1:8000/status")
a = ap.parse_args()
out = Path(a.out or f"runs/ramlog/ram-{time.strftime('%Y%m%d-%H%M%S')}.csv")
out.parent.mkdir(parents=True, exist_ok=True)
CLK = os.sysconf("SC_CLK_TCK")
COLS = ["t_iso", "pid", "uptime_s", "nrestarts", "vmrss_kb", "vmhwm_kb", "vmswap_kb", "threads", "cpu_pct", "temp_c", "fps",
        "pss_kb", "pss_anon_kb", "pss_file_kb", "private_dirty_kb", "swap_kb"]


def systemd_prop(name: str) -> str:
    if a.pid is not None and name == "MainPID":
        return str(a.pid)
    try:
        return subprocess.run(["systemctl", "show", "-p", name, "--value", a.unit], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def kv(path: str, keys: set[str]) -> dict[str, int]:
    d = {}
    for line in Path(path).read_text().splitlines():
        k, _, v = line.partition(":")
        if k in keys:
            d[k] = int(v.split()[0])
    return d


def cpu_ticks(pid: int) -> tuple[int, float]:
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12]), int(f[19]) / CLK     # utime+stime, start time since boot (s)


def uptime_s(start_since_boot: float) -> float:
    return float(Path("/proc/uptime").read_text().split()[0]) - start_since_boot


def temp_c() -> str:
    try:
        return f"{int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000:.1f}"
    except Exception:
        return ""


def fps() -> str:
    try:
        return str(json.load(urllib.request.urlopen(a.status_url, timeout=2)).get("fps", ""))
    except Exception:
        return ""


new = not out.exists()
f = out.open("a", newline="")
w = csv.writer(f)
if new:
    w.writerow(COLS)
t_start = time.time()
last_smaps = 0.0
prev = {}   # pid -> (ticks, wall)
while a.duration is None or time.time() - t_start < a.duration:
    now = time.time()
    pid = int(systemd_prop("MainPID") or 0)
    row = {"t_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": pid, "nrestarts": systemd_prop("NRestarts"), "temp_c": temp_c(), "fps": fps()}
    if pid and Path(f"/proc/{pid}").exists():
        try:
            st = kv(f"/proc/{pid}/status", {"VmRSS", "VmHWM", "VmSwap", "Threads"})
            ticks, start = cpu_ticks(pid)
            row.update(vmrss_kb=st.get("VmRSS"), vmhwm_kb=st.get("VmHWM"), vmswap_kb=st.get("VmSwap"), threads=st.get("Threads"),
                       uptime_s=round(uptime_s(start)))
            if pid in prev:
                row["cpu_pct"] = round((ticks - prev[pid][0]) / CLK / (now - prev[pid][1]) * 100, 1)
            prev = {pid: (ticks, now)}
            if now - last_smaps >= a.smaps_every:
                last_smaps = now
                sm = kv(f"/proc/{pid}/smaps_rollup", {"Pss", "Pss_Anon", "Pss_File", "Private_Dirty", "Swap"})
                row.update(pss_kb=sm.get("Pss"), pss_anon_kb=sm.get("Pss_Anon"), pss_file_kb=sm.get("Pss_File"),
                           private_dirty_kb=sm.get("Private_Dirty"), swap_kb=sm.get("Swap"))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass   # the process went away between the checks: the next tick sees the new PID
    w.writerow([row.get(c, "") for c in COLS])
    f.flush()
    time.sleep(max(0.0, a.interval - (time.time() - now)))
