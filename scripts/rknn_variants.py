#!/usr/bin/env python3
"""Recover INT8 accuracy: build several RKNN variants of the SAME ONNX straight from RKNN-Toolkit2 (not through the
Ultralytics exporter, which exposes none of these knobs), all with the same calibration list, so that the only thing
that differs between two files is the quantisation setting. Each .rknn gets a model card copied from the baseline.

Finding that motivated this (KNOWN_ISSUES #42): INT8 loses ~3 points of mAP50 but ~19 points of mAP50-95 (box tightness),
and the parity IoU of INT8 vs ONNX is only 0.84 (FP16: 0.997): the single [1,6,8400] output holds normalised box
coordinates and class scores in one 8-bit tensor, so a box edge is quantised to ~1/255 of the image (~2.5 px at 640).

    python3 scripts/rknn_variants.py --onnx runs/export/lab_v6_best_640.onnx --card runs/export/lab_v6_best_int8_640.rknn.card.json \
        --calib-list /mnt/big/AGV_ANLEE/weights/lab_v6_best_rknn_model/dataset.txt --n-calib 300 --variants base,mmse,kl,hybrid1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=None, help="pixel-box ONNX (or give --pt and it is exported)")
ap.add_argument("--pt", default=None, help="Ultralytics checkpoint: exports the pixel-box ONNX at --imgsz first")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--card", required=True, help="model card of an existing export of this model (copied and edited)")
ap.add_argument("--calib-list", required=True, help="text file, one image path per line")
ap.add_argument("--n-calib", type=int, default=300, help="evenly spaced subset (all sessions represented); 0 = all")
ap.add_argument("--variants", default="base,mmse,kl,hybrid1")
ap.add_argument("--out", default="runs/export/var")
ap.add_argument("--tag", default="lab_v6")
a = ap.parse_args()

# name -> (config kwargs, build kwargs)
VARIANTS = {
    "base":    ({}, {}),                                                        # w8a8, normal, per-channel (= the shipped one)
    "mmse":    ({"quantized_algorithm": "mmse"}, {}),
    "kl":      ({"quantized_algorithm": "kl_divergence"}, {}),
    "layer":   ({"quantized_method": "layer"}, {}),                             # per-tensor weights (expected worse)
    "hybrid1": ({"quantized_hybrid_level": 1}, {}),
    "hybrid2": ({"quantized_hybrid_level": 2}, {}),
    "hybrid3": ({"quantized_hybrid_level": 3}, {}),
    "auto":    ({}, {"auto_hybrid": True}),                                     # layers whose cosine similarity < 0.98 stay float
}

if a.pt:
    from ultralytics import YOLO

    exported = YOLO(a.pt).export(format="onnx", imgsz=a.imgsz, opset=19, simplify=True, dynamic=False)
    a.onnx = str(exported)
    print("exported", a.onnx)


def normalise_onnx(src: str, dst: str, size: int = 640) -> None:
    """Pixel-box ONNX (our parity reference) -> the graph Ultralytics' RKNN exporter feeds the toolkit: the 4 box channels
    divided by the input size, so boxes and class scores share a 0..1 range under RKNN's per-tensor INT8 scale
    (ultralytics/engine/exporter.py `_NormalizeCoords`). Feeding the pixel graph instead gives 0 mAP (scores crushed)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(src)
    g = m.graph
    out = g.output[0].name
    new_out = out + "_norm"
    c = lambda n: n + "_" + "norm"
    inits = [numpy_helper.from_array(np.array([0], np.int64), c("s0")), numpy_helper.from_array(np.array([4], np.int64), c("e4")),
             numpy_helper.from_array(np.array([6], np.int64), c("e6")), numpy_helper.from_array(np.array([1], np.int64), c("ax")),
             numpy_helper.from_array(np.full((1, 4, 1), float(size), np.float32), c("div"))]
    nodes = [helper.make_node("Slice", [out, c("s0"), c("e4"), c("ax")], [c("box")]),
             helper.make_node("Slice", [out, c("e4"), c("e6"), c("ax")], [c("cls")]),
             helper.make_node("Div", [c("box"), c("div")], [c("boxn")]),
             helper.make_node("Concat", [c("boxn"), c("cls")], [new_out], axis=1)]
    g.initializer.extend(inits)
    g.node.extend(nodes)
    dims = [d.dim_value for d in g.output[0].type.tensor_type.shape.dim]
    del g.output[:]
    g.output.extend([helper.make_tensor_value_info(new_out, TensorProto.FLOAT, dims)])
    onnx.save(m, dst)


norm_onnx = str(Path(a.out) / f"{a.tag}_norm.onnx")
Path(a.out).mkdir(parents=True, exist_ok=True)
normalise_onnx(a.onnx, norm_onnx, a.imgsz)
a.onnx_for_build = norm_onnx

lines = [l.strip() for l in Path(a.calib_list).read_text().splitlines() if l.strip()]
if a.n_calib and a.n_calib < len(lines):
    idx = np.linspace(0, len(lines) - 1, a.n_calib).round().astype(int)
    lines = [lines[i] for i in sorted(set(idx.tolist()))]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ds = out / f"dataset_{a.tag}_{len(lines)}.txt"
ds.write_text("\n".join(lines) + "\n")
print(f"calibration list: {len(lines)} images -> {ds}")

from rknn.api import RKNN  # noqa: E402

card = json.loads(Path(a.card).read_text())
for name in a.variants.split(","):
    cfg, bld = VARIANTS[name]
    dest = out / f"{a.tag}_{name}_{len(lines)}_int8_{a.imgsz}.rknn"
    t0 = time.time()
    r = RKNN(verbose=False)
    r.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588", **cfg)
    assert r.load_onnx(model=a.onnx_for_build) == 0, "load_onnx failed"
    rc = r.build(do_quantization=True, dataset=str(ds), **bld)
    if rc != 0:
        print(f"[{name}] build FAILED ({rc})")
        r.release()
        continue
    assert r.export_rknn(str(dest)) == 0
    r.release()
    c = dict(card)
    c["source_weights"] = a.onnx
    c["quant_variant"] = {"name": name, "n_calib": len(lines), **{k: str(v) for k, v in {**cfg, **bld}.items()}}
    c["parity"] = None
    c["imgsz"] = a.imgsz
    Path(str(dest) + ".card.json").write_text(json.dumps(c, indent=2))
    print(f"[{name}] -> {dest.name} in {time.time() - t0:.0f}s", flush=True)
