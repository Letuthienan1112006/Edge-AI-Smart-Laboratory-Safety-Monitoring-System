#!/usr/bin/env python3
"""Run before every training run. See src/labsafe/datasets/verify.py."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.datasets.verify import format_report, verify_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data_yaml")
ap.add_argument("--crops", type=int, default=0, help="dump N crops per class and LOOK at them")
ap.add_argument("--out", default="data/interim/crops")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()

rep = verify_dataset(a.data_yaml, sample_crops=a.crops, out_dir=a.out if a.crops else None)
print(json.dumps(rep, indent=2) if a.json else format_report(rep))
sys.exit(1 if rep["problems"] else 0)
