#!/usr/bin/env python3
"""Hard negatives for `lab_coat` from COCO: images of people wearing WHITE / light tops that are NOT lab coats.

Why: every lab model so far calls a plain white T-shirt a lab coat (20/20 on the negative test, every seed,
KNOWN_ISSUES #34/#35); the lab sessions have too few such people. COCO gives thousands of people with exact boxes.
Each selected image is written with ALL its people as `person` and NO `lab_coat` box, as session `cocow`
(data/labeled/cocow/{images,labels}), so build_lab_dataset can add it to train. Never used for testing.

Selection (all of it is a heuristic -- check the montage it writes):
  * person (iscrowd=0), box >= 120 px tall; the upper body (15-50% of height, central 60% of width) INSIDE the person
    mask is >= 45% "light" pixels (HSV V >= 170, S <= 45);
  * images with crowd regions are skipped (people we could not label would teach "person = background");
  * images whose captions mention a coat-wearing job (doctor, nurse, lab, scientist, chef, hospital, medical, coat,
    uniform, surgeon) are skipped: COCO has no lab-coat label, those are the likely real lab coats.

    python3 scripts/data/coco_negatives.py --coco data/external/coco --max-images 400
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--coco", default="data/external/coco", help="dir with val2017/ and annotations/")
ap.add_argument("--split", default="val2017")
ap.add_argument("--out", default="data/labeled/cocow")
ap.add_argument("--max-images", type=int, default=400)
ap.add_argument("--min-height", type=int, default=120)
ap.add_argument("--light-frac", type=float, default=0.45)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

root = Path(a.coco)
inst = json.loads((root / "annotations" / f"instances_{a.split}.json").read_text())
caps_path = root / "annotations" / f"captions_{a.split}.json"
caps = defaultdict(str)
if caps_path.exists():
    for c in json.loads(caps_path.read_text())["annotations"]:
        caps[c["image_id"]] += " " + c["caption"].lower()
JOB = re.compile(r"\b(doctor|nurse|lab|laboratory|scientist|chef|cook|hospital|medical|coat|uniform|surgeon|dentist)s?\b")

imgs = {i["id"]: i for i in inst["images"]}
people = defaultdict(list)
crowd = set()
for an in inst["annotations"]:
    if an["category_id"] != 1:
        continue
    if an.get("iscrowd"):
        crowd.add(an["image_id"])
    else:
        people[an["image_id"]].append(an)


def poly_mask(an, h, w):
    m = np.zeros((h, w), np.uint8)
    seg = an.get("segmentation")
    if isinstance(seg, list):
        for p in seg:
            cv2.fillPoly(m, [np.array(p, np.float32).reshape(-1, 2).astype(np.int32)], 1)
    return m


def light_share(img, an):
    x, y, w, h = an["bbox"]
    if h < a.min_height:
        return 0.0
    H, W = img.shape[:2]
    m = poly_mask(an, H, W)
    y1, y2 = int(y + 0.15 * h), int(y + 0.50 * h)
    x1, x2 = int(x + 0.20 * w), int(x + 0.80 * w)
    roi, mroi = img[y1:y2, x1:x2], m[y1:y2, x1:x2]
    if mroi.sum() < 200:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    light = (hsv[..., 2] >= 170) & (hsv[..., 1] <= 45) & (mroi > 0)
    return float(light.sum() / mroi.sum())


picked, crops = [], []
stats = defaultdict(int)
for iid, anns in people.items():
    stats["images_with_people"] += 1
    if iid in crowd:
        stats["skip_crowd"] += 1
        continue
    if JOB.search(caps.get(iid, "")):
        stats["skip_job_caption"] += 1
        continue
    info = imgs[iid]
    img = cv2.imread(str(root / a.split / info["file_name"]))
    if img is None:
        continue
    best = max(anns, key=lambda an: light_share(img, an))
    if light_share(img, best) < a.light_frac:
        continue
    picked.append(iid)
    x, y, w, h = (int(v) for v in best["bbox"])
    crops.append(cv2.resize(img[y:y + h, x:x + w], (90, 180)))
stats["candidates"] = len(picked)
random.Random(a.seed).shuffle(picked)
picked = picked[: a.max_images]

out = Path(a.out)
shutil.rmtree(out, ignore_errors=True)
(out / "images").mkdir(parents=True)
(out / "labels").mkdir()
n_boxes = 0
for iid in picked:
    info = imgs[iid]
    W, H = info["width"], info["height"]
    stem = f"cocow__{iid:012d}"
    shutil.copy(root / a.split / info["file_name"], out / "images" / f"{stem}.jpg")
    lines = []
    for an in people[iid]:
        x, y, w, h = an["bbox"]
        lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    n_boxes += len(lines)
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")

# montage of the light-top people that made an image eligible, for a human look at contamination (real lab coats)
random.Random(a.seed).shuffle(crops)
grid = crops[:60]
if grid:
    rows = [np.hstack(grid[i:i + 12] + [np.zeros_like(grid[0])] * (12 - len(grid[i:i + 12]))) for i in range(0, len(grid), 12)]
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("data/interim/cocow_montage.jpg", np.vstack(rows))
print(dict(stats), f"-> {len(picked)} images, {n_boxes} person boxes, 0 lab_coat boxes in {out}")
