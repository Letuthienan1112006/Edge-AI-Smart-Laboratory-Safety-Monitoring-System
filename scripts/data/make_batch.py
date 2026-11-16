#!/usr/bin/env python3
"""Next labelling batch: pick the most DIFFERENT unlabelled frames, draft their labels with a trained model,
package for Roboflow.

    python3 scripts/data/make_batch.py --session boot006 --n 250 --weights /mnt/big/AGV_ANLEE/weights/lab_v1_pilot_best.pt --out data/interim/roboflow_upload/boot006_b2

Selection = farthest-point sampling on small grey thumbnails, starting from everything already labelled
(data/labeled/<session>), so each pick is as unlike the labelled set and the earlier picks as possible. It does NOT
use any model output to choose (the first batch preferred 'frames with gloves', which turned out to be one pair
lying on a table, KNOWN_ISSUES #33).
Classes come from configs/lab_classes.yaml and the weights must have been trained on exactly those classes.
Drafts are LOW threshold (default 0.30): a wrong box is a click to delete. review.csv lists least-confident first.
A model trained on one session over-reports white clothing as lab_coat (KNOWN_ISSUES #34): expect false lab_coat boxes.

--hard K (active learning, hard-negative mining): the first K picks are the frames where the model is most UNSURE
about lab_coat (boxes with conf in [0.25, 0.75]), at least --hard-gap frames apart so they are not near-copies; the
other n-K are farthest-point as above. Model output chooses the hard part on purpose here: the test session is never
in the pool, so this cannot leak into the score.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--weights", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--frames", default=None, help="default data/raw/lab_sessions/<session>/frames")
ap.add_argument("--labeled", default=None, help="default data/labeled/<session>")
ap.add_argument("--only", default=None, metavar="LO:HI", help="only frames whose number is in [LO, HI]")
ap.add_argument("--skip", action="append", default=[], metavar="LO:HI", help="drop frames in [LO, HI] (repeatable)")
ap.add_argument("--hard", type=int, default=0, help="first K picks = frames where the model is least sure about lab_coat")
ap.add_argument("--hard-gap", type=int, default=8, help="min frame-number gap between hard picks")
ap.add_argument("--min-background", type=float, default=0.35,
                help="drop frames where fewer than this share of pixels match the session's usual background "
                     "(camera carried around / pointed elsewhere). 0 disables. Farthest-point sampling LOVES such "
                     "frames: they are the most different ones, which is exactly why they must be filtered first")
ap.add_argument("--tag", default=None, help="rename the session prefix of the output files (e.g. boot014test), so a test block is its own session")
a = ap.parse_args()

root = Path(__file__).resolve().parents[2]
CLASSES = yaml.safe_load((root / "configs" / "lab_classes.yaml").read_text())["names"]
frames = Path(a.frames or root / "data/raw/lab_sessions" / a.session / "frames")
labeled = Path(a.labeled or root / "data/labeled" / a.session)
allf = sorted(p for p in frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
done = {p.name.replace(f"{a.session}__{a.session}__", f"{a.session}__") for p in (labeled / "images").iterdir()} if (labeled / "images").is_dir() else set()
def _num(p): return int(p.stem.rsplit("__", 1)[1])
def _rng(t): lo, hi = t.split(":"); return int(lo), int(hi)
pool = [p for p in allf if p.name not in done]
if a.only:
    lo, hi = _rng(a.only); pool = [p for p in pool if lo <= _num(p) <= hi]
for t in a.skip:
    lo, hi = _rng(t); pool = [p for p in pool if not lo <= _num(p) <= hi]
print(f"{len(allf)} frames, {len(done)} already labelled, {len(pool)} candidates")

thumb = lambda p: cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY), (48, 36)).astype(np.float32).ravel()
if a.min_background > 0 and len(pool) > 20:
    # background = per-pixel median over the WHOLE session (people cover only part of a frame, the room does not move)
    allT = np.stack([thumb(p) for p in allf])
    med = np.median(allT, axis=0)
    share = {p.name: float((np.abs(t - med) < 25).mean()) for p, t in zip(allf, allT)}
    off = [p for p in pool if share[p.name] < a.min_background]
    if off:
        nums = sorted(_num(p) for p in off)
        print(f"dropped {len(off)} frames that are not the usual camera view (frames {nums[0]}..{nums[-1]})")
    pool = [p for p in pool if share[p.name] >= a.min_background]
T = np.stack([thumb(p) for p in pool])
L = np.stack([thumb(p) for p in allf if p.name in done]) if done else np.empty((0, T.shape[1]), np.float32)
mind = np.full(len(pool), np.inf, np.float32)
if len(L):
    for l in L:
        mind = np.minimum(mind, np.abs(T - l).mean(axis=1))
picked: list[int] = []
if a.hard:
    from ultralytics import YOLO
    m_hard = YOLO(a.weights)
    coat = [k for k, v in m_hard.names.items() if v == "lab_coat"][0]
    score = np.zeros(len(pool), np.float32)
    for j in range(0, len(pool), 32):
        for k, r in enumerate(m_hard.predict([str(p) for p in pool[j:j + 32]], conf=0.25, imgsz=640, verbose=False)):
            cf = r.boxes.conf[r.boxes.cls == coat].cpu().numpy()
            score[j + k] = float(((cf >= 0.25) & (cf <= 0.75)).sum())
    taken: list[int] = []
    for i in np.argsort(-score, kind="stable"):
        if len(picked) >= min(a.hard, a.n) or score[i] <= 0:
            break
        if all(abs(_num(pool[i]) - t) >= a.hard_gap for t in taken):
            picked.append(int(i)); taken.append(_num(pool[i]))
    for i in picked:
        mind = np.minimum(mind, np.abs(T - T[i]).mean(axis=1)); mind[i] = -1
    print(f"hard picks: {len(picked)} (unsure lab_coat boxes per frame: median {np.median(score[picked]) if picked else 0:.0f})")
for _ in range(min(a.n, len(pool)) - len(picked)):
    i = int(np.argmax(mind)) if np.isfinite(mind).any() else 0
    picked.append(i)
    mind = np.minimum(mind, np.abs(T - T[i]).mean(axis=1))
    mind[i] = -1
chosen = sorted((pool[i] for i in picked), key=lambda p: p.name)
print(f"picked {len(chosen)} (spread: farthest-point)")

from ultralytics import YOLO
model = YOLO(a.weights)
names = [model.names[i] for i in sorted(model.names)]
if names != CLASSES:
    raise SystemExit(f"weights were trained on {names} but configs/lab_classes.yaml says {CLASSES}: not guessing a mapping")
out = Path(a.out)
shutil.rmtree(out, ignore_errors=True)
(out / "images").mkdir(parents=True); (out / "labels").mkdir()
rows, counts = [], {c: 0 for c in CLASSES}
for p in chosen:
    r = model.predict(str(p), conf=a.conf, imgsz=640, verbose=False)[0]
    H, W = r.orig_shape
    lines, confs = [], []
    for c, s, xyxy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy):
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        lines.append(f"{int(c)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        confs.append(float(s)); counts[CLASSES[int(c)]] += 1
    oname = f"{a.tag}__{_num(p):05d}.jpg" if a.tag else p.name
    shutil.copy2(p, out / "images" / oname)
    (out / "labels" / f"{Path(oname).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    rows.append((oname, len(lines), round(min(confs), 3) if confs else "", "REVIEW_FIRST" if (not confs or min(confs) < 0.5) else ""))
rows.sort(key=lambda r: (r[2] != "", r[2] if r[2] != "" else 0))
with open(out / "review.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image", "n_boxes", "min_conf", "flag"]); w.writerows(rows)
(out / "data.yaml").write_text("train: images\nval: images\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
zp = out.parent / f"{out.name}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.rglob("*")):
        if f.is_file():
            z.write(f, Path(out.name) / f.relative_to(out))
print("draft boxes:", counts, "| images with no box:", sum(1 for r in rows if r[1] == 0), "| flagged REVIEW_FIRST:", sum(1 for r in rows if r[3]))
print("zip:", zp, f"({zp.stat().st_size/1e6:.1f} MB)")
