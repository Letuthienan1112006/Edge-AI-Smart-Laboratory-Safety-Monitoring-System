#!/usr/bin/env python3
"""Where does the per-frame time go OUTSIDE the model? cProfile the real router loop.

    python3 scripts/profile_pipeline.py --video some.mp4 --frames 600 --backend onnx --weights runs/export/yolo26n_640.onnx

Prints the top functions by own-time and a per-stage split (letterbox / model / decode+NMS /
tracker / rest). Use it on the board with --backend rknn to see how much of the budget the
NPU really owns; anything above ~10% that is not the model is a bottleneck to remove.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labsafe.models.backends import build_backend  # noqa: E402
from labsafe.pipeline.fall import FallConfig  # noqa: E402
from labsafe.pipeline.router import AdaptiveRouter  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--frames", type=int, default=600)
ap.add_argument("--backend", default="onnx")
ap.add_argument("--weights", default="runs/export/yolo26n_640.onnx")
ap.add_argument("--classes", default="configs/classes_coco80.yaml")
ap.add_argument("--top", type=int, default=18)
a = ap.parse_args()

names = yaml.safe_load(open(a.classes))["stage_a"]["names"]
spec = {"backend": a.backend, "weights": a.weights, "imgsz": 640, "conf": 0.4, "iou": 0.5, "keep_classes": ["person"]}
be = build_backend(spec, names)
router = AdaptiveRouter(be, None, None, [], FallConfig(), mode="adaptive")

frames = []
cap = cv2.VideoCapture(a.video)
while len(frames) < a.frames:
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not frames:
            raise SystemExit("cannot read video")
        continue
    frames.append(f)
cap.release()
for f in frames[:20]:
    router.process(f)          # warm-up, not profiled

pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
for f in frames:
    router.process(f)
pr.disable()
wall = time.perf_counter() - t0
print(f"{len(frames)} frames, {wall/len(frames)*1000:.2f} ms/frame wall (profiler overhead included)\n")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(a.top)
print("\n".join(l for l in s.getvalue().splitlines() if l.strip())[:5000])
