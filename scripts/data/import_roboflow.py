#!/usr/bin/env python3
"""Turn a Roboflow YOLO export into data/labeled/<session>/{images,labels} for build_lab_dataset.py.

    python3 scripts/data/import_roboflow.py ~/Downloads/lab.v1-xxx.yolov8.zip           # or an unzipped folder
    python3 scripts/data/import_roboflow.py <export> --merge-coats                       # one box per worn coat

What it fixes, because each of these corrupts a dataset without any error:
  * CLASS ORDER: Roboflow writes its own class list in data.yaml (often alphabetical, so goggles=0,
    lab_coat=1 ...). Class ids are remapped BY NAME to this project's fixed order
    (configs/lab_classes.yaml). An unknown class name stops the import.
  * FILE NAMES: Roboflow renames files to <stem>_jpg.rf.<hash>.jpg. The original name (boot006__00012)
    is recovered, and the session = the part before "__", so the split can be done BY SESSION.
  * SPLITS: Roboflow's own train/valid/test folders are IGNORED and flattened (its split is random per
    image; our frames are 4 s apart and near-identical, so a random split leaks -- KNOWN_ISSUES #29).
  * POLYGONS: a polygon line is converted to its bounding box (the detector predicts boxes).
  * AUGMENTATION: copies made by Roboflow augmentation (several hashes of one source image) are dropped:
    only one copy per source image is kept (the first, preferring the un-augmented one is not knowable
    from the name, so a warning is printed when this happens -- export WITHOUT augmentation).
--merge-coats: all lab_coat boxes inside the same person box become ONE box covering them (open coat drawn as
two panels -> one garment). Needs each coat box to lie mostly inside its own person box.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASSES = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "lab_classes.yaml").read_text())["names"]
IMG = {".jpg", ".jpeg", ".png", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("export", help="Roboflow export: .zip or an unzipped folder (YOLOv8 format)")
ap.add_argument("--out", default="data/labeled")
ap.add_argument("--merge-coats", action="store_true")
ap.add_argument("--drop-classes", default="", help="comma-separated class names present in the export that are NOT lab classes and must be removed (e.g. gloves)")
ap.add_argument("--coat-in-person", type=float, default=0.6, help="min share of a coat box inside a person box")
a = ap.parse_args()

tmp = None
root = Path(a.export)
if root.suffix.lower() == ".zip":
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(root).extractall(tmp)
    root = Path(tmp)
ymls = sorted(root.rglob("data.yaml"))
if not ymls:
    sys.exit("no data.yaml in the export: choose the YOLOv8 format when exporting")
cfg = yaml.safe_load(ymls[0].read_text())
names = cfg["names"]
names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
drop = {x for x in a.drop_classes.split(",") if x}
unknown = [n for n in names if n not in CLASSES and n not in drop]
if unknown:
    sys.exit(f"export has class names this project does not know: {unknown}. Expected {CLASSES}. "
             f"Rename them in Roboflow (or tell me to add the class) -- not guessing.")
remap = {i: CLASSES.index(n) for i, n in enumerate(names) if n in CLASSES}   # dropped classes have no entry
print("dropping classes (their boxes are removed):", sorted(drop & set(names)) or "none")
dropped_boxes = 0
print("class order in the export:", names, "-> remapped to", CLASSES)



def orig_stem(name: str) -> str:
    base = Path(name).name
    m = re.match(r"^(.*?)_(?:jpg|jpeg|png|bmp)\.rf\.[0-9a-f]+\.[A-Za-z]+$", base)
    return m.group(1) if m else Path(base).stem


def to_boxes(parts: list[str]) -> tuple[int, float, float, float, float] | None:
    c = int(parts[0]); v = [float(x) for x in parts[1:]]
    if c not in remap:
        return "dropped"
    if len(v) == 4:
        x, y, w, h = v
    elif len(v) >= 6 and len(v) % 2 == 0:                       # polygon -> bounding box
        xs, ys = v[0::2], v[1::2]
        x, y, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
    else:
        return None
    return remap[c], x, y, w, h


seen: dict[str, Path] = {}
dups = 0
items: dict[str, tuple[Path, list]] = {}
poly = 0
for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG):
    lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
    if not lbl.exists():
        lbl = img.with_suffix(".txt")
    stem = orig_stem(img.name)
    if stem in items:
        dups += 1
        continue
    rows = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) > 5:
                poly += 1
            b = to_boxes(parts)
            if b == "dropped":
                dropped_boxes += 1
                continue
            if b is None:
                sys.exit(f"{lbl.name}: cannot parse line {line[:60]!r}")
            rows.append(b)
    items[stem] = (img, rows)
if dups:
    print(f"WARNING: {dups} extra copies of the same source image were dropped (augmentation?). Export without augmentation.")
if poly:
    print(f"{poly} polygon line(s) converted to bounding boxes")
if drop:
    print(f"{dropped_boxes} box(es) of the dropped classes {sorted(drop)} removed")


def merge_coats(rows: list) -> tuple[list, int]:
    P, C = CLASSES.index("person"), CLASSES.index("lab_coat")
    persons = [r for r in rows if r[0] == P]
    coats = [r for r in rows if r[0] == C]
    others = [r for r in rows if r[0] not in (P, C)]
    owner: dict[int, list] = defaultdict(list)
    loose = []
    for ci, c in enumerate(coats):
        cx1, cy1, cx2, cy2 = c[1] - c[3] / 2, c[2] - c[4] / 2, c[1] + c[3] / 2, c[2] + c[4] / 2
        best, bi = 0.0, -1
        for pi, p in enumerate(persons):
            ix = max(0, min(cx2, p[1] + p[3] / 2) - max(cx1, p[1] - p[3] / 2)); iy = max(0, min(cy2, p[2] + p[4] / 2) - max(cy1, p[2] - p[4] / 2))
            share = ix * iy / max((cx2 - cx1) * (cy2 - cy1), 1e-9)
            if share > best:
                best, bi = share, pi
        (owner[bi].append(c) if bi >= 0 and best >= a.coat_in_person else loose.append(c))
    merged, n = [], 0
    for cs in owner.values():
        if len(cs) > 1:
            n += len(cs) - 1
        x1 = min(c[1] - c[3] / 2 for c in cs); y1 = min(c[2] - c[4] / 2 for c in cs)
        x2 = max(c[1] + c[3] / 2 for c in cs); y2 = max(c[2] + c[4] / 2 for c in cs)
        merged.append((C, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return persons + others + merged + loose, n


out = Path(a.out)
counts: Counter = Counter(); merged_total = 0; per_session: Counter = Counter(); lost_names = []
for stem, (img, rows) in sorted(items.items()):
    if "__" not in stem:
        lost_names.append(stem)
        continue
    session = stem.split("__")[0]
    if a.merge_coats:
        rows, n = merge_coats(rows); merged_total += n
    d = out / session
    (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(exist_ok=True)
    shutil.copy2(img, d / "images" / f"{stem}{img.suffix.lower()}")
    (d / "labels" / f"{stem}.txt").write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))
    per_session[session] += 1
    counts.update(CLASSES[r[0]] for r in rows)
if lost_names:
    print(f"WARNING: {len(lost_names)} file(s) skipped, original name lost (no '<session>__' prefix): {lost_names[:3]}")
if a.merge_coats:
    print(f"merged lab_coat boxes: {merged_total} extra box(es) folded into their person's coat")
print("images per session:", dict(per_session))
print("boxes per class   :", {c: counts.get(c, 0) for c in CLASSES})
print(f"-> {out}/<session>/{{images,labels}}   next: make build-lab-dataset")
if tmp:
    shutil.rmtree(tmp, ignore_errors=True)
