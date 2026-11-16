#!/usr/bin/env python3
"""Export one trained checkpoint to every runtime the matrix compares.

INT8 needs real calibration images. Exporting INT8 without them quantises
against random noise and produces a model whose accuracy drop is an
artefact of the export, not of quantisation -- which then gets written up
as "INT8 costs 9% mAP". The script refuses that case.
"""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--formats", default="onnx,ncnn,tflite", help="comma separated")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--data", default=None, help="data.yaml for INT8 calibration (required with --int8)")
a = ap.parse_args()

if a.int8 and not a.data:
    print("--int8 requires --data (calibration set). Refusing to export.", file=sys.stderr)
    sys.exit(1)

from ultralytics import YOLO  # noqa: E402

model = YOLO(a.weights)
failed = []
for fmt in [f.strip() for f in a.formats.split(",") if f.strip()]:
    kw = dict(format=fmt, imgsz=a.imgsz)
    if fmt == "tflite" and a.int8:
        kw.update(int8=True, data=a.data)
    print(f"\n=== export {fmt} imgsz={a.imgsz} int8={a.int8 and fmt == 'tflite'} ===")
    try:
        out = model.export(**kw)
    except Exception as e:
        # One format's export dependencies (e.g. TFLite's litert-torch) can
        # crash independently of the others, and can even mutate shared
        # packages like torch via ultralytics' own AutoUpdate as a side
        # effect (KNOWN_ISSUES.md #14) -- a crash here must not silently
        # cancel the remaining formats or hide which ones actually worked.
        print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        failed.append(fmt)
        continue
    p = Path(out)
    size = sum(f.stat().st_size for f in p.rglob("*")) if p.is_dir() else p.stat().st_size
    print(f"    -> {out}  ({size/1e6:.1f} MB)")

if failed:
    print(f"\n{len(failed)} format(s) failed: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1)
