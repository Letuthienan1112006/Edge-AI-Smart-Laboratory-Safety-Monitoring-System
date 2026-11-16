#!/usr/bin/env python3
"""How far do INT8 boxes sit from the FP32 boxes? dump detections per backend, then compare matched pairs (pixels).

    python3 scripts/box_error.py dump --backend onnx --weights W.onnx --test data/labeled/boot018test --out /tmp/fp32.json
    PYTHONPATH=src python3 scripts/box_error.py dump --backend rknn --weights W.rknn --test ... --out /tmp/int8.json   # on the board
    python3 scripts/box_error.py compare /tmp/fp32.json /tmp/int8.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
d = sp.add_parser("dump"); d.add_argument("--backend", required=True); d.add_argument("--weights", required=True)
d.add_argument("--test", nargs="+", required=True); d.add_argument("--out", required=True); d.add_argument("--conf", type=float, default=0.4)
c = sp.add_parser("compare"); c.add_argument("ref"); c.add_argument("cand")
a = ap.parse_args()

if a.cmd == "dump":
    import cv2
    from labsafe.models.backends import build_backend
    be = build_backend({"backend": a.backend, "weights": a.weights, "imgsz": 640, "conf": a.conf, "iou": 0.5,
                        **({"core_mask": "0"} if a.backend == "rknn" else {})}, ["person", "lab_coat"])
    res = {}
    for t in a.test:
        for f in sorted((Path(t) / "images").glob("*.jpg")):
            res[f.name] = [[o.cls_name, o.score, o.x1, o.y1, o.x2, o.y2] for o in be(cv2.imread(str(f)))]
    Path(a.out).write_text(json.dumps(res))
    print(len(res), "images")
else:
    ref, cand = json.loads(Path(a.ref).read_text()), json.loads(Path(a.cand).read_text())
    def iou(p, q):
        x1, y1, x2, y2 = max(p[0], q[0]), max(p[1], q[1]), min(p[2], q[2]), min(p[3], q[3])
        i = max(0, x2 - x1) * max(0, y2 - y1)
        return i / ((p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - i + 1e-9)
    errs, ious, sizes = [], [], []
    n_ref = n_cand = n_match = 0
    for k in ref:
        R = ref[k]; C = cand.get(k, []); used = set(); n_ref += len(R); n_cand += len(C)
        for r in R:
            best, bj = 0, -1
            for j, cc in enumerate(C):
                if j in used or cc[0] != r[0]:
                    continue
                v = iou(r[2:], cc[2:])
                if v > best: best, bj = v, j
            if best >= 0.5:
                used.add(bj); n_match += 1
                e = np.abs(np.array(r[2:]) - np.array(C[bj][2:])); errs.append(e); ious.append(best); sizes.append(r[5] - r[3])
    errs = np.array(errs); sizes = np.array(sizes); ious = np.array(ious)
    print(f"ref {n_ref} cand {n_cand} matched {n_match}  mean IoU {ious.mean():.3f}  median IoU {np.median(ious):.3f}")
    print("mean |edge error| px  x1 %.2f y1 %.2f x2 %.2f y2 %.2f  (all edges %.2f)" % (*errs.mean(0), errs.mean()))
    print("median all edges %.2f px, p90 %.2f px" % (np.median(errs), np.percentile(errs, 90)))
    for nm, sel in [("small (h<150px)", sizes < 150), ("medium", (sizes >= 150) & (sizes < 280)), ("large (h>=280)", sizes >= 280)]:
        if sel.sum(): print(f"  {nm:18s} n={int(sel.sum()):4d} mean edge error {errs[sel].mean():.2f} px  mean IoU {ious[sel].mean():.3f}")
