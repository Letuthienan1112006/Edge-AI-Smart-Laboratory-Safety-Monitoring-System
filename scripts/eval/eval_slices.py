#!/usr/bin/env python3
"""Where does the detector fail? Recall / precision by slice, with bootstrap 95% confidence intervals over IMAGES.

Slices (all computed from the hand labels / the image, never from the model):
  object level  : person height in pixels (distance), occlusion (box overlaps another labelled person), person count
                  of the image (crowding)
  image level   : brightness tercile (mean grey), crowd level (people per image)
Operating points are the deployed ones: person conf >= 0.40, lab_coat conf >= 0.60, IoU >= 0.5.

    PYTHONPATH=src python3 scripts/eval/eval_slices.py --backend ultralytics --weights W.pt --test data/labeled/boot014test data/labeled/boot018test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from labsafe.models.backends import build_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="ultralytics")
ap.add_argument("--weights", required=True)
ap.add_argument("--test", nargs="+", required=True)
ap.add_argument("--names", default="person,lab_coat")
ap.add_argument("--person-conf", type=float, default=0.40)
ap.add_argument("--coat-conf", type=float, default=0.60)
ap.add_argument("--boots", type=int, default=1000)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", default=None)
a = ap.parse_args()
names = a.names.split(",")
be = build_backend({"backend": a.backend, "weights": a.weights, "imgsz": a.imgsz, "conf": 0.3, "iou": 0.6}, names)


def iou(a_, b_):
    x1, y1, x2, y2 = max(a_[0], b_[0]), max(a_[1], b_[1]), min(a_[2], b_[2]), min(a_[3], b_[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - i + 1e-9)


imgs = []
for d in a.test:
    for f in sorted((Path(d) / "images").glob("*.jpg")):
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        gt = {0: [], 1: []}
        for line in (Path(d) / "labels" / (f.stem + ".txt")).read_text().splitlines():
            if line.strip():
                c, x, y, w, h = line.split()
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                gt[int(c)].append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        out = be(img)
        thr = {0: a.person_conf, 1: a.coat_conf}
        dt = {c: [(o.x1, o.y1, o.x2, o.y2) for o in out if names.index(o.cls_name) == c and o.score >= thr[c]] for c in (0, 1)}
        imgs.append({"set": Path(d).name, "gt": gt, "dt": dt, "bright": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
                     "n_people": len(gt[0])})


def match(gts, dts):
    """greedy one-to-one at IoU 0.5 -> (list of matched flags for gts, number of unmatched dets)."""
    used, hit = set(), []
    for g in gts:
        best, bj = 0.0, -1
        for j, d in enumerate(dts):
            if j in used:
                continue
            v = iou(g, d)
            if v > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj); hit.append(True)
        else:
            hit.append(False)
    return hit, len(dts) - len(used)


# per-object records (person and coat), with the slice attributes of the object / its image
objs = []
for k, im in enumerate(imgs):
    for c in (0, 1):
        hit, _ = match(im["gt"][c], im["dt"][c])
        for g, h in zip(im["gt"][c], hit):
            occl = any(iou(g, o) > 0.2 for o in im["gt"][0] if o is not g)
            objs.append({"img": k, "cls": c, "hit": h, "height": g[3] - g[1], "occl": occl, "n_people": im["n_people"]})
heights = np.array([o["height"] for o in objs if o["cls"] == 0])
h_lo, h_hi = np.percentile(heights, [33, 67])
br = np.array([im["bright"] for im in imgs])
b_lo, b_hi = np.percentile(br, [33, 67])


def boot_ci(idx_img, fn):
    """95% CI of fn(set of image indices) resampling images with replacement."""
    rng = np.random.default_rng(0)
    idx = np.array(sorted(idx_img))
    if len(idx) < 3:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(a.boots):
        s = rng.choice(idx, len(idx), replace=True)
        v = fn(s)
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def recall_of(sel_objs, cls):
    by_img = {}
    for o in sel_objs:
        if o["cls"] == cls:
            t = by_img.setdefault(o["img"], [0, 0]); t[0] += o["hit"]; t[1] += 1
    if not by_img:
        return None
    pt = sum(v[0] for v in by_img.values()) / max(1, sum(v[1] for v in by_img.values()))
    def fn(s):
        n = sum(by_img.get(i, [0, 0])[1] for i in s)
        return sum(by_img.get(i, [0, 0])[0] for i in s) / n if n else float("nan")
    lo, hi = boot_ci(by_img.keys(), fn)
    return pt, lo, hi, sum(v[1] for v in by_img.values())


def precision_of(img_idx, cls):
    tp = fp = 0
    per = {}
    for i in img_idx:
        im = imgs[i]
        hit, fpn = match(im["gt"][cls], im["dt"][cls])
        # detections matched = number of hits; FP = detections not matched
        per[i] = (sum(hit), len(im["dt"][cls]))
    if not per or sum(v[1] for v in per.values()) == 0:
        return None
    pt = sum(v[0] for v in per.values()) / sum(v[1] for v in per.values())
    def fn(s):
        d = sum(per[i][1] for i in s)
        return sum(per[i][0] for i in s) / d if d else float("nan")
    lo, hi = boot_ci(per.keys(), fn)
    return pt, lo, hi, sum(v[1] for v in per.values())


rows = []
def add(name, kind, cls, res):
    if res is None:
        return
    rows.append({"slice": name, "metric": kind, "cls": names[cls], "value": round(res[0], 3), "ci95": [round(res[1], 3), round(res[2], 3)], "n": res[3]})


allimg = list(range(len(imgs)))
for c in (0, 1):
    add("tất cả", "recall", c, recall_of(objs, c))
    add("tất cả", "precision", c, precision_of(allimg, c))
for c in (0,):   # height / occlusion are properties of the PERSON box (a coat box would inherit its own person's overlap)
    for nm, sel in [(f"cao < {h_lo:.0f}px (xa)", lambda o: o["height"] < h_lo), (f"cao {h_lo:.0f}-{h_hi:.0f}px", lambda o: h_lo <= o["height"] < h_hi),
                    (f"cao >= {h_hi:.0f}px (gần)", lambda o: o["height"] >= h_hi),
                    ("không bị che (khung người không chồng)", lambda o: not o["occl"]), ("bị che / chồng người khác", lambda o: o["occl"])]:
        add(nm, "recall", c, recall_of([o for o in objs if sel(o)], c))
crowd = {"thưa (<=3 người)": [i for i, im in enumerate(imgs) if im["n_people"] <= 3],
         "vừa (4-6 người)": [i for i, im in enumerate(imgs) if 4 <= im["n_people"] <= 6],
         "đông (>=7 người)": [i for i, im in enumerate(imgs) if im["n_people"] >= 7]}
bright = {f"tối nhất (độ sáng < {b_lo:.0f})": [i for i, im in enumerate(imgs) if im["bright"] < b_lo],
          "vừa": [i for i, im in enumerate(imgs) if b_lo <= im["bright"] < b_hi],
          f"sáng nhất (>= {b_hi:.0f})": [i for i, im in enumerate(imgs) if im["bright"] >= b_hi]}
for grp in (crowd, bright):
    for nm, idx in grp.items():
        sel = [o for o in objs if o["img"] in set(idx)]
        for c in (0, 1):
            add(nm, "recall", c, recall_of(sel, c))
            add(nm, "precision", c, precision_of(idx, c))
print(f"{len(imgs)} ảnh, {len(objs)} đối tượng; độ sáng trung bình ảnh {br.min():.0f}-{br.max():.0f}")
print(f"{'lát cắt':42s} {'chỉ số':9s} {'lớp':9s} {'giá trị':>7s} {'KTC 95%':>14s} {'n':>5s}")
for r in rows:
    print(f"{r['slice']:42s} {r['metric']:9s} {r['cls']:9s} {r['value']:7.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] {r['n']:5d}")
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"weights": a.weights, "rows": rows}, ensure_ascii=False, indent=1))
