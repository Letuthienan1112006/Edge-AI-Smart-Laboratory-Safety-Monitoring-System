#!/usr/bin/env python3
"""Record a RAW (no overlay) camera session for building the lab dataset.

    python3 scripts/data/record_raw_session.py --session s01_blouse_glasses --note "blouse+goggles, lights on" --seconds 120

`make board-live` writes an ANNOTATED video (boxes, zones, text drawn on the
frames); training on that teaches the model the overlays. This writes the
untouched sensor frames. Waits out the auto-exposure warm-up first
(docs/KNOWN_ISSUES.md #19) so the first frames are not black.

One directory per session under data/raw/lab_sessions/. The session name is
what train/val/test are split on later (adjacent frames are near-duplicates;
never split them randomly).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "src"))
from labsafe.io.cues import due_cues  # noqa: E402
import signal
import time
from datetime import datetime
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True, help="unique id, e.g. s03_two_people_dim")
ap.add_argument("--note", default="", help="what is in the scene: PPE worn/absent, lighting, poses")
ap.add_argument("--seconds", type=float, default=120)
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--out", default="data/raw/lab_sessions")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--cues", default="[]", help="JSON list of {t, say, event?}: announced at second t of the recording, logged with the frame index")
a = ap.parse_args()

out = Path(a.out) / a.session
if out.exists():
    raise SystemExit(f"{out} already exists -- pick a new session id, sessions are never overwritten")
# The directory is created only AFTER the camera produced a usable frame: a failed attempt must leave
# nothing behind, or a resumable runner would mistake the empty folder for a finished recording.

cap = cv2.VideoCapture(a.device)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, a.fps)
if not cap.isOpened():
    raise SystemExit("cannot open camera")

t0 = time.time()
while True:  # auto-exposure warm-up: wait for a non-uniform frame
    ok, f = cap.read()
    if ok and f.std() >= 2.0 and time.time() - t0 > 2.0:
        break
    if time.time() - t0 > 10:
        raise SystemExit("camera produced no usable frame in 10s")

out.mkdir(parents=True)
h, w = f.shape[:2]
vw = cv2.VideoWriter(str(out / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (w, h))
stop = False
def _stop(*_):
    global stop
    stop = True
signal.signal(signal.SIGINT, _stop)

cues = json.loads(a.cues)
fired: set[int] = set()
cue_log = []
n, start = 0, time.time()
print(f"recording {out} for {a.seconds:.0f}s (Ctrl+C to stop early)")
while not stop and time.time() - start < a.seconds:
    ok, f = cap.read()
    if ok:
        vw.write(f)
        n += 1
    for _, c in due_cues(cues, time.time() - start, fired):
        print(f"  >> [{c['t']:>4}s] {c['say']}\a", flush=True)
        cue_log.append({**c, "frame": n, "elapsed_s": round(time.time() - start, 2)})
vw.release(); cap.release()
elapsed = time.time() - start
(out / "meta.json").write_text(json.dumps({
    "session": a.session, "note": a.note, "frames": n, "seconds": round(elapsed, 1),
    "measured_fps": round(n / elapsed, 2), "resolution": [w, h],
    "cues": cue_log,
    "recorded_at": datetime.now().isoformat(timespec="seconds"),
    "clock_note": "board clock may be wrong (KNOWN: AGENTS.md); use session id, not this timestamp",
}, indent=2))
print(f"done: {n} frames, {elapsed:.0f}s, {n/elapsed:.1f} fps -> {out}")
