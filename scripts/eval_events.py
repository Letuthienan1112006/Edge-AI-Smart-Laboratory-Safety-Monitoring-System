#!/usr/bin/env python3
"""Event-level evaluation of the alarm logic on recorded scripted sessions (scripts/collect_guided.py + configs/collection_plan_events.yaml).

The detector runs ONCE per frame (cached); the whole decision chain (tracker, attire attach, zone rule, fall, debounce/grace/cooldown) is then replayed
for every setting on simulated time (frame index / fps), so settings are compared on identical detections and no board is needed.

    PYTHONPATH=src python3 scripts/eval_events.py --sessions 'data/raw/lab_sessions/EV_*' --grace 0 0.5 1 1.5 --out runs/events/grace.json

Reports, per setting: expected violations, detected, missed, recall, false alarms and per hour, median delay (from the cue, so it includes the
actor's own reaction), duplicate alarms per detected event, falls detected and false fall alarms. See src/labsafe/bench/event_eval.py for the definitions.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labsafe.app import build_system  # noqa: E402
from labsafe.bench.event_eval import fall_times, merge, score, zone_intervals  # noqa: E402
from labsafe.config import Config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sessions", nargs="+", required=True, help="session dirs or globs (video.avi + meta.json with cues)")
ap.add_argument("--config", default="configs/live/dashboard_lab_board.yaml")
ap.add_argument("--classes", default="configs/classes_lab_v3.yaml")
ap.add_argument("--weights", default="runs/export/split/split640_s0.onnx", help="detector for the cache pass (ONNX runs on the laptop CPU)")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--grace", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
ap.add_argument("--debounce", type=float, default=3.0)
ap.add_argument("--cooldown", type=float, default=60.0)
ap.add_argument("--attire-min-conf", type=float, nargs="+", default=[0.6], help="one or several values: swept together with --grace and --memory")
ap.add_argument("--partial-min-h", type=float, nargs="+", default=[0.0], help="zone_check ignores a person whose box touches the frame bottom and is shorter than this fraction of the frame height (0 = off)")
ap.add_argument("--memory", type=float, nargs="+", default=[2.0], help="attire_memory_s values (how long a seen coat keeps the person 'wearing')")
ap.add_argument("--zones", default=None, help="zone file for THIS recording (same format as configs/zones_lab.yaml, or the board's data/zones_live.yaml drawn in the dashboard); default: the one in --config")
ap.add_argument("--out", default=None)
a = ap.parse_args()


def make_cfg(grace: float, attire: float = None, memory: float = None, partial: float = None) -> Config:
    cfg = Config.load(a.config, a.classes)
    p = cfg.pipeline
    p["stage_a"].update({"backend": "onnx", "weights": a.weights, "imgsz": a.imgsz, "engine": "cpu_any"})
    p["stage_a"].pop("core_mask", None)
    p["stage_c"] = {"enabled": False}
    p["stage_b"] = {"enabled": False}
    p["async_stages"] = False
    p["dashboard"] = {"enabled": False}
    p["evidence"] = {"enabled": False}
    p["zones"]["attire_min_conf"] = attire if attire is not None else a.attire_min_conf[0]
    p["zones"]["attire_memory_s"] = memory if memory is not None else a.memory[0]
    p["zones"]["partial_min_h"] = partial if partial is not None else a.partial_min_h[0]
    p["zones"]["live_file"] = "/nonexistent/zones_live.yaml"   # never let a zone file left in data/ silently replace the one under test
    if a.zones:
        p["zones"]["file"] = a.zones
    p["alerts"] = {"debounce_s": a.debounce, "cooldown_s": a.cooldown, "grace_s": grace, "gpio": "off"}
    return cfg


class Cached:
    """Stands in for stage A: returns the detections computed in the cache pass, frame by frame."""
    def __init__(self, real, dets):
        self._real, self._dets, self.i = real, dets, 0

    def __call__(self, frame):
        d = self._dets[self.i]
        self.i += 1
        return d

    def __getattr__(self, k):
        return getattr(self._real, k)


sessions = []
for pat in a.sessions:
    sessions += sorted(glob.glob(pat)) or [pat]
sessions = [Path(s) for s in sessions if (Path(s) / "video.avi").exists() and (Path(s) / "meta.json").exists()]
if not sessions:
    raise SystemExit("no session with video.avi + meta.json found")

# --- pass 1: detections per frame, once ---
base_router, _, _ = build_system(make_cfg(0.0), "adaptive", None)
cache = {}
for s in sessions:
    meta = json.loads((s / "meta.json").read_text())
    cap = cv2.VideoCapture(str(s / "video.avi"))
    dets, h, w = [], 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        dets.append(base_router.stage_a(f))
    fps = float(meta.get("measured_fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cache[s] = (meta, dets, fps, (h, w))
    print(f"{s.name}: {len(dets)} frames, {fps:.1f} fps, {len(meta.get('cues', []))} cues", flush=True)

# --- pass 2: replay the decision chain per setting (grace_s x attire_min_conf x attire_memory_s) ---
import itertools

results = {}
per_scene = {}
for g, ac, mem, ph in itertools.product(a.grace, a.attire_min_conf, a.memory, a.partial_min_h):
    key = f"grace={g:g} attire>={ac:g} mem={mem:g} partial<{ph:g}"
    scores = []
    for s, (meta, dets, fps, (h, w)) in cache.items():
        router, _, _ = build_system(make_cfg(g, ac, mem, ph), "adaptive", None)
        router.stage_a = Cached(router.stage_a, dets)
        blank = np.zeros((h, w, 3), np.uint8)
        alarms = []
        for i in range(len(dets)):
            _, alarming = router.process(blank, now=i / fps)
            alarms += [{"t": i / fps, "kind": e.kind, "track": e.track_id} for e in alarming]
        cues = [{**c, "t": c["frame"] / fps} for c in meta.get("cues", [])]
        total = len(dets) / fps
        sc = score(alarms, zone_intervals(cues, total), fall_times(cues), total, a.debounce)
        scores.append(sc)
        per_scene.setdefault(s.name, {})[key] = {**sc.row(), "alarms": [(round(x["t"], 1), x["kind"], x["track"]) for x in alarms]}
    results[key] = merge(scores).row()

cols = ["expected", "detected", "missed", "recall", "false_alarms", "fa_per_hour", "median_delay_s", "dup_per_event", "falls", "fall_false_alarms"]
print("\n" + f"{'setting':>32s} " + " ".join(f"{c:>13s}" for c in cols))
for g, r in results.items():
    print(f"{g:>32s} " + " ".join(f"{str(r[c]):>13s}" for c in cols))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"settings": {"debounce": a.debounce, "cooldown": a.cooldown, "attire_min_conf": a.attire_min_conf, "memory": a.memory, "weights": a.weights},
                                       "overall": results,
                                       "per_scene": per_scene}, indent=1, ensure_ascii=False))
