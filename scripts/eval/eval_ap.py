#!/usr/bin/env python3
"""AP of any repo backend (onnx / rknn / ultralytics) against hand labels, one evaluator for every precision.

Why one evaluator: comparing "mAP from ultralytics on FP32" with "mAP from something else on INT8" mixes two metric
implementations. Here every backend runs through the SAME backend interface (the one the live system uses, so the
decode / letterbox / class handling being measured is the deployed one) and the SAME AP code.

    # on the laptop (ONNX, CPU)                      # on the board (NPU)
    python3 scripts/eval/eval_ap.py --backend onnx --weights runs/export/lab_v6_best_640.onnx --test data/labeled/boot018test
    PYTHONPATH=src python3 scripts/eval/eval_ap.py --backend rknn --weights runs/export/lab_v6_best_int8_640.rknn --test ...

AP = area under the precision-recall curve (101-point interpolation, COCO style) per class, at IoU 0.5 and the mean over
IoU 0.50:0.05:0.95 (mAP50-95). Detections are taken at a very low confidence (0.001) so the whole curve exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from labsafe.models.backends import build_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True, help="onnx | rknn | ultralytics")
ap.add_argument("--weights", required=True)
ap.add_argument("--test", nargs="+", required=True, help="labelled session dirs (images/ + labels/)")
ap.add_argument("--names", default="person,lab_coat")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--core-mask", default="auto")
ap.add_argument("--out", default=None, help="write the numbers as JSON")
a = ap.parse_args()
names = a.names.split(",")

spec = {"backend": a.backend, "weights": a.weights, "imgsz": a.imgsz, "conf": 0.001, "iou": 0.6}
if a.backend == "rknn":
    spec["core_mask"] = a.core_mask
be = build_backend(spec, names)

imgs, gts, dets = [], [], []
lat = []
for d in a.test:
    for f in sorted((Path(d) / "images").glob("*.jpg")):
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        g = []
        for line in (Path(d) / "labels" / (f.stem + ".txt")).read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, w, h = line.split()
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            g.append((int(c), x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        t = time.perf_counter()
        out = be(img)
        lat.append((time.perf_counter() - t) * 1000)
        gts.append(g)
        dets.append([(names.index(o.cls_name), o.score, o.x1, o.y1, o.x2, o.y2) for o in out if o.cls_name in names])


def iou_matrix(b, g):
    b, g = np.asarray(b, np.float64), np.asarray(g, np.float64)
    x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
    x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ab[:, None] + ag[None, :] - inter + 1e-9)


def ap_class(cls: int, thr: float) -> float:
    scores, tp, n_gt = [], [], 0
    for g, d in zip(gts, dets):
        gc = [x[1:] for x in g if x[0] == cls]
        dc = sorted([x for x in d if x[0] == cls], key=lambda x: -x[1])
        n_gt += len(gc)
        used = set()
        iou = iou_matrix([x[2:] for x in dc], gc) if dc and gc else None
        for i, x in enumerate(dc):
            hit = False
            if iou is not None:
                j = int(np.argmax([(iou[i, k] if k not in used else -1) for k in range(len(gc))]))
                if iou[i, j] >= thr and j not in used:
                    used.add(j); hit = True
            scores.append(x[1]); tp.append(hit)
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp, np.float64)[order]
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]           # monotone envelope
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))


res = {"backend": a.backend, "weights": a.weights, "n_images": len(gts), "latency_ms_mean": round(float(np.mean(lat)), 2)}
for ci, n in enumerate(names):
    res[f"AP50_{n}"] = round(ap_class(ci, 0.5), 4)
    res[f"AP50-95_{n}"] = round(float(np.mean([ap_class(ci, t) for t in np.arange(0.5, 0.96, 0.05)])), 4)
res["mAP50"] = round(float(np.mean([res[f"AP50_{n}"] for n in names])), 4)
res["mAP50-95"] = round(float(np.mean([res[f"AP50-95_{n}"] for n in names])), 4)
print(json.dumps(res, ensure_ascii=False))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
