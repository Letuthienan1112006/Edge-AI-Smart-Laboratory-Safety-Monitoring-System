#!/usr/bin/env python3
"""Turn a full-frame YOLO dataset into a person-CROP dataset for Stage B.

    python3 scripts/make_person_crops.py data/processed/ppe_public --person 0 --out data/processed/ppe_crops

Stage B only ever sees the crop of a tracked person (README / research-plan 2.2), so it must
be trained and evaluated on crops: a goggle that is 12 px in a 640 frame is 60 px in a crop.
Splits are copied as-is (train->train, val->val, test->test) so the split-by-source rule of the
original dataset is preserved -- crops of one photo never straddle two splits.

Each crop keeps the non-person boxes whose CENTRE lies inside it, clipped to the crop.
The person class is dropped from the output; remaining class ids are re-numbered densely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("src", help="dataset dir with images/{train,val,test}, labels/..., data.yaml")
ap.add_argument("--person", type=int, default=0, help="class id of the person box")
ap.add_argument("--out", required=True)
ap.add_argument("--pad", type=float, default=0.10, help="relative padding around the person box")
ap.add_argument("--min-side", type=int, default=48, help="drop crops whose short side (px) is below this")
a = ap.parse_args()

src, out = Path(a.src), Path(a.out)
cfg = yaml.safe_load((src / "data.yaml").read_text())
names = cfg["names"]
names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
keep = [i for i in range(len(names)) if i != a.person]
remap = {old: new for new, old in enumerate(keep)}
stats: dict = {}
for split in ("train", "val", "test"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_crops = n_small = n_box = 0
    for img_path in sorted((src / "images" / split).glob("*")):
        lbl = src / "labels" / split / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            c, cx, cy, bw, bh = int(p[0]), *(float(v) for v in p[1:5])
            boxes.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        for k, (c, x1, y1, x2, y2) in enumerate(b for b in boxes if b[0] == a.person):
            pw, ph = (x2 - x1) * a.pad, (y2 - y1) * a.pad
            X1, Y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            X2, Y2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
            cw, ch = X2 - X1, Y2 - Y1
            if min(cw, ch) < a.min_side:
                n_small += 1
                continue
            rows = []
            for c2, bx1, by1, bx2, by2 in boxes:
                if c2 == a.person:
                    continue
                mx, my = (bx1 + bx2) / 2, (by1 + by2) / 2
                if not (X1 <= mx <= X2 and Y1 <= my <= Y2):
                    continue
                ux1, uy1, ux2, uy2 = max(bx1, X1) - X1, max(by1, Y1) - Y1, min(bx2, X2) - X1, min(by2, Y2) - Y1
                if ux2 - ux1 < 2 or uy2 - uy1 < 2:
                    continue
                rows.append(f"{remap[c2]} {(ux1+ux2)/2/cw:.6f} {(uy1+uy2)/2/ch:.6f} {(ux2-ux1)/cw:.6f} {(uy2-uy1)/ch:.6f}")
            name = f"{img_path.stem}__p{k}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), im[Y1:Y2, X1:X2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            n_crops += 1; n_box += len(rows)
    stats[split] = {"crops": n_crops, "too_small_dropped": n_small, "boxes": n_box}
(out / "data.yaml").write_text(yaml.safe_dump({
    "path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
    "names": {new: names[old] for old, new in remap.items()}}, sort_keys=False))
(out / "crops.json").write_text(json.dumps({"source": str(src), "pad": a.pad, "min_side": a.min_side, "stats": stats}, indent=2))
print(json.dumps(stats, indent=2))
