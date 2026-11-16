#!/usr/bin/env python3
"""Score lab models on a human-labelled test session: detector AP, the white-T-shirt negative test, and the
"zone requires lab_coat" rule judged per person (KNOWN_ISSUES #35).

    PYTHONPATH=src python3 scripts/eval/eval_zone_rule.py --weights W1.pt [W2.pt ...] --attire-min-conf 0.6

Ground truth for "wears a coat" = the SAME attach rule (router._attach_attire) applied to the human boxes, so the
rule is scored, not the geometry convention. Every person counts as inside the zone (zone = whole frame), each image
is judged on its own (no tracking memory: the test frames are 4 s apart).
"""
from __future__ import annotations

import argparse
import glob
import shutil
import tempfile
from pathlib import Path

import cv2

from labsafe.models.backends import Detection, build_backend
from labsafe.pipeline.fall import FallConfig
from labsafe.pipeline.router import AdaptiveRouter
from labsafe.pipeline.zones import Zone

ap = argparse.ArgumentParser()
ap.add_argument("--weights", nargs="+", required=True)
ap.add_argument("--test", default="data/labeled/boot014test", help="labelled session dir with images/ labels/")
ap.add_argument("--negatives", default="runs/board/N1L_withperson", help="frames of people in white T-shirts")
ap.add_argument("--attire-min-conf", type=float, nargs="+", default=[0.0])
ap.add_argument("--person-conf", type=float, default=0.4)
ap.add_argument("--backend", default="ultralytics", help="ultralytics (.pt, FP32) | rknn (.rknn, run on the board)")
ap.add_argument("--core-mask", default="auto")
ap.add_argument("--vis", default="", help="attire visibility rule, e.g. 'min_h=0.22,tall_aspect=2,min_w_px=24,max_occlusion=0.4'. Persons whose LABELLED box fails it are not scored (their clothes cannot be judged from the image either).")
a = ap.parse_args()

ZONE = [Zone("all", "high", [(0, 0), (1, 0), (1, 1), (0, 1)], ("lab_coat",))]
VIS = {k: float(v) for k, v in (kv.split("=") for kv in a.vis.split(",") if kv)}


class Fixed:
    name, imgsz, last_latency_ms = "fixed", 640, 0.0

    def __init__(self, dets):
        self.dets = dets

    def __call__(self, frame):
        return self.dets


def iou(p, q):
    x1, y1, x2, y2 = max(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), min(p[3], q[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def box(t):
    return (t.det.x1, t.det.y1, t.det.x2, t.det.y2)


test_imgs = sorted(glob.glob(f"{a.test}/images/*.jpg"))
gts = []
for f in test_imgs:
    img = cv2.imread(f)
    H, W = img.shape[:2]
    dets = []
    for line in open(Path(a.test) / "labels" / (Path(f).stem + ".txt")):
        c, x, y, w, h = map(float, line.split())
        dets.append(Detection((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H, 1.0, int(c),
                              "person" if int(c) == 0 else "lab_coat"))
    g = AdaptiveRouter(Fixed(dets), None, None, ZONE, FallConfig(), visibility=VIS)
    g.process(img, now=0.0)
    gts.append((img, {box(t): ("lab_coat" in t.attire_seen) for t in g.tracks if not t.attire_unknown}))
print(f"scored persons (visible by the rule {VIS or 'off'}): {sum(len(g) for _, g in gts)}", flush=True)

# ultralytics val wants a data.yaml; dataloader workers hang when run from a heredoc -> workers=0
tmp = Path(tempfile.mkdtemp())
for sub in ("images", "labels"):
    shutil.copytree(Path(a.test) / sub, tmp / sub / "test")
(tmp / "data.yaml").write_text(f"path: {tmp}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: person\n  1: lab_coat\n")
negs = sorted(glob.glob(f"{a.negatives}/**/*.jpg", recursive=True))

for wpath in a.weights:
    if a.backend == "ultralytics":
        from ultralytics import YOLO   # not installed on the board: only needed for FP32 .pt scoring

        m = YOLO(wpath)
        r = m.val(data=str(tmp / "data.yaml"), split="test", imgsz=640, conf=0.25, verbose=False, plots=False,
                  workers=0, project=str(tmp / "val"), name="v", exist_ok=True)
        shirt = sum(int((x.boxes.cls == 1).sum()) > 0 for x in m.predict(negs, conf=0.25, verbose=False)) if negs else 0
        print(f"{Path(wpath).name}: person AP50 {r.box.ap50[0]:.2f} | lab_coat AP50 {r.box.ap50[1]:.2f} "
              f"P {r.box.p[1]:.2f} R {r.box.r[1]:.2f} | white T-shirt flagged as coat {shirt}/{len(negs)}", flush=True)
    else:
        probe = build_backend({"backend": a.backend, "weights": wpath, "imgsz": 640, "conf": 0.25, "iou": 0.5,
                               "keep_classes": ["person", "lab_coat"], "core_mask": a.core_mask}, ["person", "lab_coat"])
        shirt = sum(any(d.cls_name == "lab_coat" for d in probe(cv2.imread(f))) for f in negs) if negs else 0
        print(f"{Path(wpath).name}: white T-shirt flagged as coat {shirt}/{len(negs)}", flush=True)
    stage_a = build_backend({"backend": a.backend, "weights": wpath, "imgsz": 640, "conf": min(a.person_conf, 0.25),
                             "iou": 0.5, "keep_classes": ["person", "lab_coat"],
                             **({"core_mask": a.core_mask} if a.backend == "rknn" else {})}, ["person", "lab_coat"])
    raw = [stage_a(img) for img, _ in gts]
    for th in a.attire_min_conf:
        tp = fp = fn = tn = 0
        for (img, gt), dets in zip(gts, raw):
            dets = [d for d in dets if (d.cls_name == "person" and d.score >= a.person_conf) or d.cls_name != "person"]
            rr = AdaptiveRouter(Fixed(dets), None, None, ZONE, FallConfig(), attire_min_conf=th, visibility=VIS)
            ev, _ = rr.process(img, now=0.0)
            flagged = {e.track_id for e in ev}
            used = set()
            for t in rr.tracks:
                best = max(gt, key=lambda q: iou(box(t), q), default=None)
                if best is None or iou(box(t), best) < 0.5 or best in used:
                    continue
                used.add(best)
                wear, fl = gt[best], t.track_id in flagged
                tp += fl and not wear
                fp += fl and wear
                fn += (not fl) and not wear
                tn += (not fl) and wear
        print(f"   attire_min_conf {th:.2f}: no-coat caught {tp}/{tp + fn} ({tp / max(1, tp + fn):.2f}) | "
              f"coat wearers falsely flagged {fp}/{fp + tn} ({fp / max(1, fp + tn):.2f})", flush=True)
shutil.rmtree(tmp, ignore_errors=True)
