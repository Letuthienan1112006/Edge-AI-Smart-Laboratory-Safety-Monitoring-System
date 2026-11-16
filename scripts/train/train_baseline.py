#!/usr/bin/env python3
"""Train a stage on the laptop GPU. Verification is not optional here.

Deliberately refuses to start if the dataset verifier found a structural
problem: a training run started on a mislabelled dataset costs hours and
produces a model whose numbers nobody can trust afterwards.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from labsafe.datasets.verify import format_report, verify_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the YOLO data.yaml")
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=16, help="RTX 3050 6GB: 16 @640 is safe")
ap.add_argument("--name", default="stage_a")
ap.add_argument("--device", default="0")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--force", action="store_true", help="train despite dataset problems")
ap.add_argument("--seed", type=int, default=0,
                help="keep 0 for ablations; change it ONLY to measure run-to-run spread (same data, same settings)")
a = ap.parse_args()

rep = verify_dataset(a.data)
print(format_report(rep))
if rep["problems"] and not a.force:
    print("\nrefusing to train on a dataset with unresolved problems (--force to override)")
    sys.exit(1)

from ultralytics import YOLO  # noqa: E402  (slow import, only needed past this point)

model = YOLO(a.model)
model.train(
    data=a.data,
    imgsz=a.imgsz,
    epochs=a.epochs,
    batch=a.batch,
    device=a.device,
    workers=a.workers,
    patience=a.patience,
    project="runs/train",
    name=a.name,
    seed=a.seed,       # fixed (0) for ablations; other values only to measure run-to-run spread
    deterministic=True,
    plots=True,
)
metrics = model.val(data=a.data, imgsz=a.imgsz, split="val")
print("\nper-class AP50 -- read this, not just the mean:")
names = model.names
ap50s = getattr(metrics.box, "ap50", None)  # numpy array: never use `or []` on it
for i, ap50 in enumerate([] if ap50s is None else ap50s):
    print(f"  {names.get(i, i):<12} {float(ap50):.3f}")
