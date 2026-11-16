#!/usr/bin/env python3
"""Draft YOLO labels for a folder of lab images so a human only has to CORRECT them.

    python3 scripts/prelabel.py data/interim/lab_frames/s01 --out data/interim/prelabel/s01 \
        --lab-coat --copy-images
    python3 scripts/prelabel.py <images> --out <dir> --gt-dir <labels>   # also score the drafts

Classes in the output: configs/lab_classes.yaml (currently 0 person, 1 gloves, 2 lab_coat).
  * person    : stage A, the COCO YOLO26n (class 0).
  * gloves    : stage B run on PERSON CROPS (its goggles output is ignored: goggles is not a lab class) exactly as the system runs it (crop with 10%
                padding, 320 px), boxes mapped back to the full frame.
  * lab_coat  : optional (--lab-coat), zero-shot YOLOE-11l with the text prompt "lab coat".
                Untested on real lab footage -- measure it on the first reviewed batch before
                trusting it (KNOWN_ISSUES.md #26: zero-shot is weak on small items).
The drafts are deliberately LOW-threshold (recall over precision): deleting a wrong box is a click,
finding a missed one costs a careful look. review.csv lists images least-confident-first so the
reviewer can start where the model is most likely wrong. labels/*.txt are standard YOLO;
data.yaml + obj.names import into CVAT / Label Studio. scores.json keeps each box's confidence.

Split train/val/test BY SESSION afterwards, never by frame (adjacent frames are near-duplicates).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import yaml
CLASSES = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "lab_classes.yaml").read_text())["names"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("images", help="folder of images (searched recursively)")
ap.add_argument("--out", required=True)
ap.add_argument("--stage-a", default="/mnt/big/AGV_ANLEE/weights/yolo26n.pt")
ap.add_argument("--stage-b", default="/mnt/big/AGV_ANLEE/weights/stageB_crops_y26n_best.pt")
ap.add_argument("--person-conf", type=float, default=0.35)
ap.add_argument("--ppe-conf", type=float, default=0.20)
ap.add_argument("--crop-pad", type=float, default=0.10)
ap.add_argument("--min-person-px", type=int, default=48)
ap.add_argument("--lab-coat", action="store_true", help="add zero-shot lab_coat boxes with YOLOE")
ap.add_argument("--coat-conf", type=float, default=0.25)
ap.add_argument("--yoloe", default="/mnt/big/AGV_ANLEE/weights/ovd/yoloe-11l-seg.pt")
ap.add_argument("--review-below", type=float, default=0.5, help="flag an image if any box is under this conf")
ap.add_argument("--copy-images", action="store_true", help="copy images into out/images (else symlink)")
ap.add_argument("--gt-dir", default=None, help="YOLO labels with the SAME 4-class ids, to score the drafts")
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402  (slow import)

imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXT)
if not imgs:
    raise SystemExit(f"no images under {a.images}")
out = Path(a.out)
(out / "labels").mkdir(parents=True, exist_ok=True)
(out / "images").mkdir(parents=True, exist_ok=True)

det_a, det_b = YOLO(a.stage_a), YOLO(a.stage_b)
b_names = {v: k for k, v in det_b.names.items()}         # e.g. goggles->0, gloves->1 in the B model
coat = None
if a.lab_coat:
    from ultralytics import YOLOE
    old = os.getcwd()
    os.chdir(Path(a.yoloe).parent)    # set_classes() downloads/loads mobileclip into the CWD (KNOWN_ISSUES #26)
    try:
        coat = YOLOE(a.yoloe)
        coat.set_classes(["lab coat"], coat.get_text_pe(["lab coat"]))
    finally:
        os.chdir(old)


def nms(boxes: list[tuple], thr: float = 0.5) -> list[tuple]:
    """boxes: (cls, conf, x1, y1, x2, y2); greedy per-class NMS."""
    keep: list[tuple] = []
    for b in sorted(boxes, key=lambda t: -t[1]):
        ok = True
        for k in keep:
            if k[0] != b[0]:
                continue
            ix = max(0, min(k[4], b[4]) - max(k[2], b[2])); iy = max(0, min(k[5], b[5]) - max(k[3], b[3]))
            inter = ix * iy
            ua = (k[4]-k[2])*(k[5]-k[3]) + (b[4]-b[2])*(b[5]-b[3]) - inter
            if ua > 0 and inter / ua > thr:
                ok = False; break
        if ok:
            keep.append(b)
    return keep


rows, all_scores, counts = [], {}, {c: 0 for c in CLASSES}
for p in imgs:
    im = cv2.imread(str(p))
    if im is None:
        continue
    H, W = im.shape[:2]
    boxes: list[tuple] = []
    ra = det_a.predict(im, conf=a.person_conf, classes=[0], verbose=False)[0]
    persons = [(float(c), *map(float, xy)) for c, xy in zip(ra.boxes.conf, ra.boxes.xyxy)]
    for conf, x1, y1, x2, y2 in persons:
        boxes.append((0, conf, x1, y1, x2, y2))
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < a.min_person_px:
            continue
        X1, Y1 = int(max(0, x1 - bw * a.crop_pad)), int(max(0, y1 - bh * a.crop_pad))
        X2, Y2 = int(min(W, x2 + bw * a.crop_pad)), int(min(H, y2 + bh * a.crop_pad))
        if X2 - X1 < 2 or Y2 - Y1 < 2:
            continue
        rb = det_b.predict(im[Y1:Y2, X1:X2], conf=a.ppe_conf, imgsz=320, verbose=False)[0]
        for c, s, xy in zip(rb.boxes.cls, rb.boxes.conf, rb.boxes.xyxy):
            name = det_b.names[int(c)]
            if name in CLASSES:
                bx1, by1, bx2, by2 = (float(v) for v in xy)
                boxes.append((CLASSES.index(name), float(s), bx1 + X1, by1 + Y1, bx2 + X1, by2 + Y1))
    if coat is not None:
        rc = coat.predict(im, conf=a.coat_conf, verbose=False)[0]
        for s, xy in zip(rc.boxes.conf, rc.boxes.xyxy):
            boxes.append((CLASSES.index("lab_coat"), float(s), *map(float, xy)))
    boxes = nms(boxes)
    lines, scores = [], []
    for c, s, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        scores.append({"cls": CLASSES[c], "conf": round(s, 3)})
        counts[CLASSES[c]] += 1
    (out / "labels" / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    dst = out / "images" / p.name
    if not dst.exists():
        shutil.copy2(p, dst) if a.copy_images else os.symlink(p.resolve(), dst)
    all_scores[p.name] = scores
    mn = min((x["conf"] for x in scores), default=None)
    rows.append((p.name, len(lines), mn, "REVIEW_FIRST" if (mn is None or mn < a.review_below) else ""))

rows.sort(key=lambda r: (r[2] is not None, r[2] if r[2] is not None else 0))   # no boxes / least confident first
with open(out / "review.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image", "n_boxes", "min_conf", "flag"]); w.writerows(rows)
(out / "obj.names").write_text("\n".join(CLASSES) + "\n")
(out / "data.yaml").write_text("path: " + str(out.resolve()) + "\ntrain: images\nval: images\nnames:\n" +
                               "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
(out / "scores.json").write_text(json.dumps(all_scores, indent=1))
(out / "prelabel.json").write_text(json.dumps({"images": len(imgs), "boxes": counts, "stage_a": a.stage_a, "stage_b": a.stage_b,
                                               "lab_coat": a.lab_coat, "thresholds": {"person": a.person_conf, "ppe": a.ppe_conf, "coat": a.coat_conf}}, indent=2))
print(f"{len(imgs)} images -> {out}  boxes: {counts}  flagged for review first: {sum(1 for r in rows if r[3])}")

if a.gt_dir:
    def iou(p, q):
        ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
        return i / ((p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - i + 1e-9)
    def read(path, W, H):
        res = []
        for l in Path(path).read_text().split("\n") if Path(path).exists() else []:
            if l.strip():
                c, x, y, w, h = l.split(); x, y, w, h = float(x)*W, float(y)*H, float(w)*W, float(h)*H
                res.append((int(c), x-w/2, y-h/2, x+w/2, y+h/2))
        return res
    tp = {c: 0 for c in range(len(CLASSES))}; npred = dict(tp); ngt = dict(tp)
    for p in imgs:
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        pred, gt = read(out / "labels" / f"{p.stem}.txt", W, H), read(Path(a.gt_dir) / f"{p.stem}.txt", W, H)
        for c in range(len(CLASSES)):
            P = [b[1:] for b in pred if b[0] == c]; G = [b[1:] for b in gt if b[0] == c]; used = [False]*len(G)
            npred[c] += len(P); ngt[c] += len(G)
            for pb in P:
                best, bj = 0, -1
                for j, g in enumerate(G):
                    v = iou(pb, g)
                    if v > best and not used[j]: best, bj = v, j
                if best >= 0.5: used[bj] = True; tp[c] += 1
    print("draft quality vs ground truth (IoU>=0.5): to fix = wrong boxes to delete + missed boxes to draw")
    for c in range(len(CLASSES)):
        if ngt[c] or npred[c]:
            print(f"  {CLASSES[c]:9s} GT {ngt[c]:4d}  drafted {npred[c]:4d}  correct {tp[c]:4d}  precision {tp[c]/max(npred[c],1):.2f}  recall {tp[c]/max(ngt[c],1):.2f}  wrong {npred[c]-tp[c]}  missed {ngt[c]-tp[c]}")
