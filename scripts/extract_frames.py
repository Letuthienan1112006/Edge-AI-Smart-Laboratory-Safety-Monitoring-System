#!/usr/bin/env python3
"""Cut raw session videos into training images, dropping near-duplicates.

    python3 scripts/extract_frames.py data/raw/lab_sessions/s01_x --fps 2 --min-diff 6

Images are named <session>__<index>.jpg so the session is recoverable from the
filename -- the train/val/test split must be by session, never by frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("session_dir")
ap.add_argument("--fps", type=float, default=2.0, help="candidate frames per second")
ap.add_argument("--min-diff", type=float, default=6.0,
                help="min mean abs gray diff (0-255) from the last kept frame")
ap.add_argument("--out", default="data/interim/lab_frames")
a = ap.parse_args()

sd = Path(a.session_dir)
video = next(iter(sorted(sd.glob("video.*"))), None)
if video is None:
    raise SystemExit(f"no video.* in {sd}")
out = Path(a.out) / sd.name
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
step = max(1, round(src_fps / a.fps))
kept, seen, last = 0, 0, None
i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % step == 0:
        seen += 1
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (80, 60)).astype(np.float32)
        if f.std() >= 2.0 and (last is None or float(np.abs(g - last).mean()) >= a.min_diff):
            cv2.imwrite(str(out / f"{sd.name}__{kept:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            last = g; kept += 1
    i += 1
cap.release()
(out / "extract.json").write_text(json.dumps({"video": str(video), "candidates": seen, "kept": kept,
                                              "fps": a.fps, "min_diff": a.min_diff}, indent=2))
print(f"{sd.name}: {i} frames read, {seen} candidates, {kept} kept -> {out}")
