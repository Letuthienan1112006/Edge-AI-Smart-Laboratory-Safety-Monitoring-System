#!/usr/bin/env python3
"""Re-split a Roboflow-style YOLO dataset by SOURCE image, not by augmented copy.

    python3 scripts/data/regroup_split.py data/raw/jimei_fire_smoke --out data/processed/fire_smoke --names fire smoke

Roboflow exports several augmented copies of one photo as `<stem>_jpg.rf.<hash>.jpg`. When those
copies land in different splits the val/test scores measure memorisation, not generalisation
(docs/KNOWN_ISSUES.md #29). This pools train/valid/test, groups by <stem>, and splits whole groups
80/10/10 with a fixed seed. Images are symlinked; labels are copied.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out", required=True)
ap.add_argument("--names", nargs="+", required=True, help="class names in id order")
ap.add_argument("--val", type=float, default=0.1)
ap.add_argument("--test", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
stem = lambda n: re.sub(r"_(jpg|jpeg|png|bmp)\.rf\..*$", "", n)
groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
for split in ("train", "valid", "test"):
    for img in sorted((src / split / "images").glob("*")):
        lbl = src / split / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            groups[stem(img.name)].append((img, lbl))
keys = sorted(groups)
random.Random(a.seed).shuffle(keys)
n = len(keys)
cut = {"val": int(n * a.val), "test": int(n * a.test)}
assign = {}
for i, k in enumerate(keys):
    assign[k] = "val" if i < cut["val"] else "test" if i < cut["val"] + cut["test"] else "train"
stats = defaultdict(lambda: {"images": 0, "groups": 0})
for k, items in groups.items():
    s = assign[k]
    stats[s]["groups"] += 1
    (out / "images" / s).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    for img, lbl in items:
        dst = out / "images" / s / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        (out / "labels" / s / lbl.name).write_text(lbl.read_text())
        stats[s]["images"] += 1
(out / "data.yaml").write_text(yaml.safe_dump({
    "path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
    "names": dict(enumerate(a.names))}, sort_keys=False))
(out / "regroup.json").write_text(json.dumps({"source": str(src), "seed": a.seed, "groups": n, "stats": stats}, indent=2))
print(json.dumps({"groups": n, **stats}, indent=2))
