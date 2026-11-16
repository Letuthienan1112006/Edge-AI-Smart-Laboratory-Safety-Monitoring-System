#!/usr/bin/env python3
"""How reliable is "is this person in the zone"? Compares the foot point of the PREDICTED person box with the foot point of the hand-labelled box on boot014test (a different session from training),
using the lab polygon configs/zones_lab.yaml (drawn on the boot014 camera angle).  PYTHONPATH=src python3 scripts/eval_zone_foot.py   (docs/results.md 6j)
Caveat: the reference foot is the bottom of the LABELLED box, not the true floor contact, so occluded feet are not measured."""
import cv2, numpy as np, yaml
sys.path.insert(0, "src")
from labsafe.app import build_system
from labsafe.config import Config
from labsafe.pipeline.zones import load_zones, point_in_polygon

cfg = Config.load("configs/live/dashboard_lab_board.yaml", "configs/classes_lab_v3.yaml"); p = cfg.pipeline
p["stage_a"].update({"backend": "onnx", "weights": "runs/export/split/split640_s0.onnx", "imgsz": 640, "engine": "cpu_any", "conf": 0.4}); p["stage_a"].pop("core_mask", None)
for k in ("stage_b", "stage_c"): p[k] = {"enabled": False}
p["async_stages"] = False; p["dashboard"] = {"enabled": False}; p["evidence"] = {"enabled": False}
st = build_system(cfg, "adaptive", None)[0].stage_a
z = load_zones(yaml.safe_load(open("configs/zones_lab.yaml")))[0]

def inzone(fx, fy, w, h):                       # same clamp as Zone.contains_foot
    eps = 1e-6
    return point_in_polygon(min(max(fx / w, eps), 1 - eps), min(max(fy / h, eps), 1 - eps), z.polygon)

rows = []; n_gt = n_match = 0
for img_path in sorted(glob.glob("data/labeled/boot014test/images/*.jpg")):
    lab = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    img = cv2.imread(img_path); h, w = img.shape[:2]
    gt = []
    for ln in open(lab):
        c, cx, cy, bw, bh = ln.split()[:5]
        if int(c) == 0:
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            gt.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    P = [d for d in st(img) if d.cls_name == "person"]
    polyp = np.array(z.pixel_polygon(w, h), np.int32)
    for g in gt:
        n_gt += 1
        best, bi = 0.0, None
        for d in P:
            iw = max(0, min(g[2], d.x2) - max(g[0], d.x1)); ih = max(0, min(g[3], d.y2) - max(g[1], d.y1)); inter = iw * ih
            u = (g[2]-g[0])*(g[3]-g[1]) + (d.x2-d.x1)*(d.y2-d.y1) - inter
            if u > 0 and inter / u > best: best, bi = inter / u, d
        if bi is None or best < 0.5: continue
        n_match += 1
        gfx, gfy = (g[0] + g[2]) / 2, g[3]; pfx, pfy = bi.foot
        cut = g[3] >= h - 4 or bi.y2 >= h - 4
        rows.append(dict(err=float(np.hypot(pfx - gfx, pfy - gfy)), ey=float(pfy - gfy), gt_in=inzone(gfx, gfy, w, h), pr_in=inzone(pfx, pfy, w, h),
                         dist=float(cv2.pointPolygonTest(polyp, (float(gfx), float(gfy)), True)), cut=bool(cut)))
r = rows
print(f"GT persons {n_gt}, matched (IoU>=0.5) {n_match} ({100*n_match/n_gt:.0f}%), unmatched = no decision possible: {n_gt-n_match}")
e = np.array([x["err"] for x in r]); print(f"foot-point error px (640x480): median {np.median(e):.1f}  p90 {np.percentile(e,90):.1f}  p95 {np.percentile(e,95):.1f}")
def report(name, sub):
    if not sub: print(f"{name}: none"); return
    agree = sum(x["gt_in"] == x["pr_in"] for x in sub); print(f"{name}: {agree}/{len(sub)} same in/out decision ({100*agree/len(sub):.0f}%)")
report("ALL matched", r)
report("feet visible (not cut by frame bottom)", [x for x in r if not x["cut"]])
report("feet cut off at frame bottom", [x for x in r if x["cut"]])
for lo, hi in ((0, 10), (10, 30), (30, 80), (80, 1e9)):
    report(f"  foot {lo}-{hi if hi<1e8 else 'inf'} px from the zone edge", [x for x in r if lo <= abs(x["dist"]) < hi])
inside = [x for x in r if x["gt_in"]]; outside = [x for x in r if not x["gt_in"]]
print(f"persons really IN the zone: {len(inside)}  -> system says in: {sum(x['pr_in'] for x in inside)} ({100*sum(x['pr_in'] for x in inside)/max(1,len(inside)):.0f}%)")
print(f"persons really OUT of the zone: {len(outside)}  -> system says in (false 'entered'): {sum(x['pr_in'] for x in outside)} ({100*sum(x['pr_in'] for x in outside)/max(1,len(outside)):.0f}%)")
print("median signed y error (pred foot minus labelled foot, + = lower in the image):", round(float(np.median([x['ey'] for x in r])), 1))
