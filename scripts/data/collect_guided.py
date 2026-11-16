#!/usr/bin/env python3
"""Guided lab-data collection: walks through configs/collection_plan.yaml, one RAW video per scene.

    python3 scripts/data/collect_guided.py                  # all scenes not yet recorded (resumes after a stop)
    python3 scripts/data/collect_guided.py --only 02,03     # just these scene ids (prefix match)
    python3 scripts/data/collect_guided.py --list           # show the plan and what is already recorded
    python3 scripts/data/collect_guided.py --auto           # no "press Enter" (unattended / testing)

For each scene it prints what to do, waits for Enter, counts down 5 s (so you can step into the
frame), records with scripts/data/record_raw_session.py (raw frames, no overlay, auto-exposure warm-up
handled) and writes scene.json (the plan text and split hint) next to the video. A scene whose
session directory already exists is skipped, never overwritten. Run it ON THE BOARD (camera there):
    ssh -t orangepi-lan 'cd ~/AI_Challenge && python3 scripts/data/collect_guided.py'
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="configs/collection_plan.yaml")
ap.add_argument("--out", default="data/raw/lab_sessions")
ap.add_argument("--only", default="", help="comma-separated scene id prefixes, e.g. 02,05")
ap.add_argument("--list", action="store_true")
ap.add_argument("--auto", action="store_true", help="do not wait for Enter")
ap.add_argument("--countdown", type=int, default=5)
ap.add_argument("--seconds-scale", type=float, default=1.0, help="multiply every scene length (tests)")
ap.add_argument("--device", type=int, default=0)
a = ap.parse_args()

plan = yaml.safe_load(Path(a.plan).read_text())
batch, scenes = plan["batch"], plan["scenes"]
only = [s for s in a.only.split(",") if s]
out = Path(a.out)
session = lambda sc: f"{batch}_{sc['id']}"
# "recorded" = the finished marker exists. A directory alone is not proof (a crash mid-recording).
done = lambda sc: (out / session(sc) / "scene.json").exists()

if a.list:
    tot = sum(sc["seconds"] for sc in scenes)
    print(f"batch {batch}: {len(scenes)} scenes, ~{tot/60:.0f} min of recording")
    for sc in scenes:
        print(f"  [{'x' if done(sc) else ' '}] {session(sc):28s} {sc['seconds']:3d}s  {sc['split']:5s}  {sc['text'][:70]}")
    raise SystemExit(0)

todo = [sc for sc in scenes if not done(sc) and (not only or any(sc["id"].startswith(o) for o in only))]
print(f"{len(todo)} scene(s) to record, {len(scenes) - len(todo)} skipped (already recorded or filtered).")
for i, sc in enumerate(todo, 1):
    secs = max(2.0, sc["seconds"] * a.seconds_scale)
    print("\n" + "=" * 78)
    print(f"Scene {i}/{len(todo)}  {session(sc)}   ({secs:.0f} s, split gợi ý: {sc['split']})")
    print(f"  >> {sc['text']}")
    if not a.auto:
        try:
            input("  Chuẩn bị xong thì nhấn Enter (Ctrl+C để dừng, lần sau chạy lại sẽ tiếp tục): ")
        except (KeyboardInterrupt, EOFError):
            print("\nstopped; recorded scenes are kept.")
            raise SystemExit(0)
    for n in range(a.countdown, 0, -1):
        print(f"  bắt đầu sau {n} ...", end="\r", flush=True)
        time.sleep(1)
    print("  ĐANG QUAY ...                       \a")
    stale = out / session(sc)
    if stale.exists() and not (stale / "scene.json").exists():
        import shutil
        print(f"  (removing an unfinished earlier attempt: {stale})")
        shutil.rmtree(stale)
    r = subprocess.run([sys.executable, "scripts/data/record_raw_session.py", "--session", session(sc),
                        "--note", sc["text"], "--seconds", str(secs), "--out", str(out),
                        "--device", str(a.device), "--cues", json.dumps(sc.get("cues", []), ensure_ascii=False)])
    if r.returncode != 0:
        print(f"  !! recording failed for {session(sc)} (exit {r.returncode}); run again to retry")
        raise SystemExit(r.returncode)
    (out / session(sc) / "scene.json").write_text(json.dumps(
        {"scene": sc["id"], "split_hint": sc["split"], "text": sc["text"], "seconds_planned": sc["seconds"], "cues": sc.get("cues", [])},
        ensure_ascii=False, indent=2))
    print("  xong. \a")
print("\nAll requested scenes recorded. Next, on the laptop: make fetch-sessions")
