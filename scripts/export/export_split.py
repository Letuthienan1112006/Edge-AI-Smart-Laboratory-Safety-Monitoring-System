#!/usr/bin/env python3
"""Export a YOLO26 (no-DFL) checkpoint with SPLIT head outputs, then build INT8 RKNN variants from it.

Output tensors (ONNX order): box_P3, cls_P3, box_P4, cls_P4, box_P5, cls_P5
  box_i: (1, 4, H, W) distances ltrb from the cell centre in stride units; cls_i: (1, nc, H, W) probabilities.
Decoded on the CPU by `labsafe.models.postprocess.assemble_split_outputs`. Why: KNOWN_ISSUES #42 (INT8 box error).

    python3 scripts/export/export_split.py --pt W.pt --imgsz 512 --tag split512 --calib-list <dataset.txt> --n-calib 300 --out runs/export/split
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pt", required=True)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--tag", default="split")
ap.add_argument("--card", default="runs/export/lab_v6_best_int8_640.rknn.card.json")
ap.add_argument("--calib-list", default=None, help="if given, also build the INT8 .rknn")
ap.add_argument("--n-calib", type=int, default=300)
ap.add_argument("--out", default="runs/export/split")
ap.add_argument("--variants", default="base", help="comma list: base,mmse,kl,hybrid1")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO  # noqa: E402

net = copy.deepcopy(YOLO(a.pt).model).float().eval().fuse()
head = net.model[-1]
if getattr(head, "reg_max", 1) != 1:
    raise SystemExit("this split export needs a head without DFL (YOLO26); YOLOv8/11 have reg_max=16")
head.export = True
head.format = "onnx"
if getattr(head, "end2end", False):
    head.end2end = False


def forward_head(x, box_head=None, cls_head=None, **kw):
    # keep the per-scale tensors; the originals are concatenated, which would force one shared INT8 scale
    head._raw = [(box_head[i](x[i]), cls_head[i](x[i])) for i in range(head.nl)]
    return {"boxes": torch.cat([b.flatten(2) for b, _ in head._raw], dim=-1),
            "scores": torch.cat([c.flatten(2) for _, c in head._raw], dim=-1), "feats": x}


def inference(preds):
    return tuple(t for b, c in head._raw for t in (b, c.sigmoid()))


head.forward_head = forward_head
head._inference = inference
im = torch.zeros(1, 3, a.imgsz, a.imgsz)
with torch.no_grad():
    y = net(im)
names = [f"{k}{i}" for i in range(head.nl) for k in ("box_P", "cls_P")]
names = [f"{'box' if j % 2 == 0 else 'cls'}_P{3 + j // 2}" for j in range(2 * head.nl)]
onnx_path = out / f"{a.tag}.onnx"
torch.onnx.export(net, im, str(onnx_path), opset_version=19, input_names=["images"], output_names=names,
                  dynamic_axes=None, dynamo=False)
print("onnx:", onnx_path, [tuple(t.shape) for t in y], "box range per scale:", [round(float(b.abs().max()), 1) for b in y[0::2]])

if a.calib_list:
    lines = [l.strip() for l in Path(a.calib_list).read_text().splitlines() if l.strip()]
    if a.n_calib and a.n_calib < len(lines):
        idx = np.linspace(0, len(lines) - 1, a.n_calib).round().astype(int)
        lines = [lines[i] for i in sorted(set(idx.tolist()))]
    ds = out / f"dataset_{a.tag}_{len(lines)}.txt"
    ds.write_text("\n".join(lines) + "\n")
    from rknn.api import RKNN

    VARS = {"base": {}, "mmse": {"quantized_algorithm": "mmse"}, "kl": {"quantized_algorithm": "kl_divergence"},
            "hybrid1": {"quantized_hybrid_level": 1}}
    card = json.loads(Path(a.card).read_text())
    for v in a.variants.split(","):
        t0 = time.time()
        r = RKNN(verbose=False)
        r.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588", **VARS[v])
        assert r.load_onnx(model=str(onnx_path)) == 0
        assert r.build(do_quantization=True, dataset=str(ds)) == 0
        dest = out / f"{a.tag}_{v}_{len(lines)}_int8_{a.imgsz}.rknn"
        assert r.export_rknn(str(dest)) == 0
        r.release()
        c = dict(card)
        c.update({"imgsz": a.imgsz, "box_coords": "pixels", "layout": card.get("layout", "nhwc"), "parity": None,
                  "split_head": True, "source_weights": a.pt})
        Path(str(dest) + ".card.json").write_text(json.dumps(c, indent=2))
        print(f"[{v}] -> {dest.name} in {time.time() - t0:.0f}s", flush=True)
