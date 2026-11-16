# Zero-shot open-vocab teacher (YOLOE) vs human labels on data/processed/ppe_public test. Run with cwd containing mobileclip_blt.ts (KNOWN_ISSUES #26).
import glob, os, numpy as np, sys
from ultralytics import YOLOE
ROOT = "/home/an-lee/AI_Challenge/data/processed/ppe_public"
imgs = sorted(glob.glob(f"{ROOT}/images/test/*"))
def gt(img):
    from PIL import Image
    w, h = Image.open(img).size
    out = []
    for l in open(f"{ROOT}/labels/test/{os.path.splitext(os.path.basename(img))[0]}.txt"):
        c, x, y, bw, bh = l.split(); c = int(c); x, y, bw, bh = (float(x)*w, float(y)*h, float(bw)*w, float(bh)*h)
        out.append((c, x-bw/2, y-bh/2, x+bw/2, y+bh/2))
    return out
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)
GT = {p: gt(p) for p in imgs}
MODEL, IMGSZ = sys.argv[1], int(sys.argv[2])
m = YOLOE(MODEL)
variants = {"C": ["person", "protective glasses", "hand wearing glove"]}
print(f"### model={MODEL} imgsz={IMGSZ}")
for tag, names in variants.items():
    m.set_classes(names, m.get_text_pe(names))
    preds = {p: [] for p in imgs}
    for i in range(0, len(imgs), 16):
        for p, r in zip(imgs[i:i+16], m.predict(imgs[i:i+16], conf=0.05, imgsz=IMGSZ, verbose=False)):
            preds[p] = [(int(c), float(s), *map(float, b)) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    res = []
    for c in range(3):
        recs = []; ngt = sum(1 for p in imgs for g in GT[p] if g[0] == c)
        for p in imgs:
            gts = [g[1:] for g in GT[p] if g[0] == c]; used = [False]*len(gts)
            for _, s, *b in sorted([q for q in preds[p] if q[0] == c], key=lambda q: -q[1]):
                best, bj = 0, -1
                for j, g in enumerate(gts):
                    v = iou(b, g)
                    if v > best and not used[j]: best, bj = v, j
                tp = best >= 0.5
                if tp: used[bj] = True
                recs.append((s, tp))
        recs.sort(key=lambda r: -r[0]); tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
        rec = tp / max(ngt, 1); prec = tp / np.maximum(tp + fp, 1)
        ap = float(np.trapz(np.maximum.accumulate(prec[::-1])[::-1], rec)) if len(recs) else 0.0
        k = [i for i, r in enumerate(recs) if r[0] >= 0.25]; kk = k[-1] if k else -1
        p25 = prec[kk] if kk >= 0 else 0; r25 = rec[kk] if kk >= 0 else 0
        res.append((ap, p25, r25, ngt))
    print(f"[{tag}] {names}")
    for n, (ap, p, r, ngt) in zip(["person", "goggles", "gloves"], res):
        print(f"      {n:8s} AP50 {ap:.3f} | conf>=0.25: P {p:.3f} R {r:.3f} (GT {ngt})")
