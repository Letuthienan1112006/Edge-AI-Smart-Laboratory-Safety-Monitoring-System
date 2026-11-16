#!/usr/bin/env python3
"""Structured channel pruning for a YOLO detector -- the kind that is actually faster.

The distinction this script exists to enforce:

  UNSTRUCTURED pruning sets individual weights to zero. The tensor keeps its
  shape, the GEMM keeps its dimensions, and every dense kernel on a CPU runs
  it at exactly the same speed. A 90%-sparse model and a dense one have the
  same latency on a Raspberry Pi 4. What changes is the compressed FILE SIZE,
  which is why reports built on it claim a speedup that the device never saw.

  STRUCTURED pruning removes whole channels/filters. The tensors physically
  shrink, so the convolution really does less work, and ordinary dense ARM
  kernels pick the gain up with no sparse-kernel support needed.

So: only structured pruning here, and the script prints the before/after
parameter and FLOP counts so the reduction is a measurement, not a claim.
A latency number still has to come from the target device -- a FLOP
reduction is a prediction of speedup, never evidence of it.

Method: iterative magnitude-based channel pruning with fine-tuning between
steps (Li et al., "Pruning Filters for Efficient ConvNets", ICLR 2017).
One-shot pruning to a high ratio collapses accuracy; several small steps,
each followed by a short fine-tune, recovers most of it.

    python3 scripts/prune_structured.py --weights runs/train/stage_a/weights/best.pt \
        --data data/processed/person/data.yaml --ratio 0.4 --steps 4 --epochs 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--weights", required=True)
ap.add_argument("--data", required=True, help="data.yaml, needed for fine-tuning between steps")
ap.add_argument("--ratio", type=float, default=0.4, help="total fraction of channels to remove")
ap.add_argument("--steps", type=int, default=4, help="iterative steps; 1 = one-shot (not advised)")
ap.add_argument("--epochs", type=int, default=15, help="fine-tune epochs after each step")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="0")
ap.add_argument("--out", default="runs/prune")
ap.add_argument("--unstructured", action="store_true",
                help="refused on purpose; the flag exists to explain why")
a = ap.parse_args()

if a.unstructured:
    sys.exit(
        "Refused.\n\n"
        "Unstructured pruning does not make inference faster on a Raspberry Pi 4.\n"
        "Zeroed weights leave the tensor shapes unchanged, so NCNN/TFLite run the\n"
        "same dense GEMM at the same speed. You would get a smaller .tflite file\n"
        "and identical latency.\n\n"
        "If the goal is model SIZE, say so explicitly and report size, not speed.\n"
        "If the goal is SPEED, drop --unstructured and prune channels."
    )

if not 0.0 < a.ratio < 0.95:
    sys.exit(f"--ratio {a.ratio} is outside a sane range (0, 0.95)")
if a.steps == 1:
    print("WARNING: --steps 1 is one-shot pruning. Expect a large accuracy drop that\n"
          "         fine-tuning cannot fully recover. Use 3-5 steps.\n")

try:
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(
        f"missing dependency: {e.name}\n"
        "  pip install torch torchvision ultralytics torch-pruning"
    )


def summarise(model, imgsz: int) -> tuple[int, float]:
    example = torch.randn(1, 3, imgsz, imgsz)
    macs, params = tp.utils.count_ops_and_params(model, example)
    return params, macs * 2 / 1e9          # MACs -> GFLOPs


out_dir = Path(a.out)
out_dir.mkdir(parents=True, exist_ok=True)

yolo = YOLO(a.weights)
model = yolo.model
model.eval()

p0, g0 = summarise(model, a.imgsz)
print(f"before pruning: {p0/1e6:.2f} M params, {g0:.2f} GFLOPs")

# Never prune the detection head's outputs: its channel count encodes the
# number of classes and the DFL bins. Removing channels there changes what
# the model predicts, not how fast it predicts it.
ignored = []
for m in model.modules():
    if m.__class__.__name__ in {"Detect", "Segment", "Pose", "OBB"}:
        ignored.extend(mm for mm in m.modules() if isinstance(mm, torch.nn.Conv2d))
print(f"protected {len(ignored)} conv layers in the detection head")

per_step = 1.0 - (1.0 - a.ratio) ** (1.0 / a.steps)
print(f"pruning {a.ratio:.0%} of channels over {a.steps} steps "
      f"({per_step:.1%} per step), fine-tuning {a.epochs} epochs between steps\n")

# The dependency graph tracer walks the model via forward hooks and can
# silently trace almost nothing if it does not understand the forward
# pass -- verified for real, 2026-09-26 (KNOWN_ISSUES.md #12): on this
# environment's torch 2.14 + torch_pruning 1.6.1, YOLO's eval-mode forward
# returning (Tensor, dict) instead of a plain Tensor left the graph with
# a SINGLE traced node out of 88 Conv2d layers, `pruner.step()` raised no
# error, and the script printed a plausible "0.0% removed" result -- the
# exact silent-success shape this project exists to catch. Setting the
# detection head's `export` flag (the same flag ultralytics itself sets
# before ONNX/NCNN/RKNN export) fixes the (Tensor, dict) case, but is not
# proof the tracer worked in general, so it does not replace the guard
# below -- it only removes one known cause of tripping it.
head = model.model[-1] if hasattr(model, "model") else None
if head is not None and hasattr(head, "export"):
    head.export = True

total_prunable = sum(
    1 for m in model.modules() if isinstance(m, torch.nn.Conv2d) and m not in ignored
)

example = torch.randn(1, 3, a.imgsz, a.imgsz)
for step in range(1, a.steps + 1):
    model.eval()
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),   # L2 norm per channel
        pruning_ratio=per_step,
        ignored_layers=ignored,
        global_pruning=False,   # per-layer: global pruning can delete a whole
                                # narrow layer and break the graph
    )
    traced = len(pruner.DG.module2node)
    if traced < total_prunable:
        sys.exit(
            f"[step {step}/{a.steps}] dependency graph only traced {traced} "
            f"module(s) against {total_prunable} prunable Conv2d layer(s) "
            f"(plus {len(ignored)} protected). This model's forward pass is "
            f"not fully visible to torch_pruning's tracer -- pruning would "
            f"silently do little or nothing rather than fail, which is worse "
            f"than crashing here. Known cause on this repo's dev environment: "
            f"torch_pruning 1.6.1 vs torch {torch.__version__} (KNOWN_ISSUES.md "
            f"#12). Try pinning torch to a version torch_pruning was tested "
            f"against (ultralytics' own isolated RKNN build env uses "
            f"torch==2.4) before assuming the model architecture is the "
            f"problem."
        )
    p_before = sum(p.numel() for p in model.parameters())
    pruner.step()
    p_after = sum(p.numel() for p in model.parameters())
    if p_after >= p_before:
        sys.exit(
            f"[step {step}/{a.steps}] pruner.step() ran without error but "
            f"removed 0 parameters ({p_before} -> {p_after}). Refusing to "
            f"report this as a completed pruning step -- see KNOWN_ISSUES.md #12."
        )
    p, g = summarise(model, a.imgsz)
    print(f"[step {step}/{a.steps}] {p/1e6:.2f} M params ({100*p/p0:.0f}% of original), "
          f"{g:.2f} GFLOPs ({100*g/g0:.0f}%)")

    ckpt = out_dir / f"pruned_step{step}.pt"
    yolo.model = model
    torch.save({"model": model}, ckpt)

    if a.epochs > 0:
        print(f"[step {step}/{a.steps}] fine-tuning...")
        yolo.model = model
        yolo.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                   device=a.device, project=str(out_dir), name=f"finetune_step{step}",
                   seed=0, deterministic=True, pretrained=False)
        model = yolo.model

p1, g1 = summarise(model, a.imgsz)
final = out_dir / "pruned_final.pt"
torch.save({"model": model}, final)

print(f"""
========================================================================
  params : {p0/1e6:.2f} M  ->  {p1/1e6:.2f} M   ({100*(1-p1/p0):.1f}% removed)
  GFLOPs : {g0:.2f}     ->  {g1:.2f}      ({100*(1-g1/g0):.1f}% removed)
  saved  : {final}
========================================================================

  This is a FLOP reduction, not a measured speedup. Next:
    1. export it     : scripts/export_models.py {final} --formats ncnn
    2. measure on Pi : python3 -m labsafe.app --bench --exp E_prune ...
    3. re-check mAP per SLICE, not just overall -- pruning spends the
       capacity the hard cases were using (docs/measurement-protocol.md).
""")
