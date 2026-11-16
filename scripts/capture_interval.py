#!/usr/bin/env python3
"""Unattended interval capture: leave the board running, collect RAW frames.

    nohup python3 scripts/capture_interval.py --session hoa01_morning --every 4 --hours 3 \
        --note "chemistry lab, camera on shelf" > /tmp/capture.log 2>&1 &

Saves one raw JPEG (no overlay) every --every seconds, skipping frames that
are near-identical to the last saved one (an empty room would otherwise fill
the SD card with the same picture) and stopping before the disk is full.
Needs no network while running: pull the files afterwards with rsync.
The board clock may be wrong, so the session id -- not the timestamp -- is
what identifies a run; train/val/test are split by session, never by frame.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--note", default="")
ap.add_argument("--every", type=float, default=4.0, help="seconds between shots")
ap.add_argument("--hours", type=float, default=2.0)
ap.add_argument("--min-diff", type=float, default=3.0,
                help="skip frames whose mean gray diff to the last saved one is below this "
                     "(0 disables; an empty static room is mostly skipped)")
ap.add_argument("--min-free-gb", type=float, default=2.0, help="stop when free disk drops below this")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--out", default="data/raw/lab_sessions")
a = ap.parse_args()

out = Path(a.out) / a.session
if out.exists():
    raise SystemExit(f"{out} exists -- sessions are never overwritten, pick a new id")
(out / "frames").mkdir(parents=True)

cap = cv2.VideoCapture(a.device)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("cannot open camera")

stop = False
def _stop(*_):
    global stop
    stop = True
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

t0 = time.time()
kept = skipped_static = skipped_dark = 0
last = None
reason = "duration_reached"
deadline = t0 + a.hours * 3600
next_shot = t0
print(f"capturing into {out} every {a.every}s for {a.hours}h", flush=True)

def write_meta():
    (out / "meta.json").write_text(json.dumps({
        "session": a.session, "note": a.note, "kept": kept, "skipped_static": skipped_static,
        "skipped_dark": skipped_dark, "every_s": a.every, "min_diff": a.min_diff,
        "elapsed_s": round(time.time() - t0), "stop_reason": reason,
        "started_at": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "clock_note": "board clock may be wrong; use the session id",
    }, indent=2))

while not stop and time.time() < deadline:
    ok, f = cap.read()   # keep draining so the driver buffer never serves an old frame
    now = time.time()
    if not ok or now < next_shot:
        continue
    next_shot = now + a.every
    if f.std() < 2.0:
        skipped_dark += 1
        continue
    g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (80, 60)).astype(np.float32)
    if a.min_diff > 0 and last is not None and float(np.abs(g - last).mean()) < a.min_diff:
        skipped_static += 1
        continue
    cv2.imwrite(str(out / "frames" / f"{a.session}__{kept:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    last = g; kept += 1
    if kept % 25 == 0:
        write_meta()
        print(f"{kept} saved, {skipped_static} static skipped, {(now - t0)/60:.0f} min", flush=True)
    if shutil.disk_usage(out).free / 1e9 < a.min_free_gb:
        reason = "disk_low"; break
if stop:
    reason = "interrupted"
cap.release()
write_meta()
print(f"done: {kept} saved ({reason}) -> {out}", flush=True)
