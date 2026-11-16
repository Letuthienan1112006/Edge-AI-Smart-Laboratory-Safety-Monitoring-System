#!/usr/bin/env python3
"""Classic teacher-student knowledge distillation (Hinton et al. 2015,
arXiv:1503.02531), adapted to a YOLO detection head's classification
branch. See `src/labsafe/models/kd_loss.py` for the full scope/adaptation
writeup -- read it before citing this script as "we implemented D1".

This is NOT `scripts/train/distill_pseudolabel.py`. That script transfers
knowledge through the DATASET (teacher's hard pseudo-labels, computed once,
offline, on unlabelled footage). This script transfers it through the LOSS
(teacher's soft class logits, computed every training step, on the SAME
labelled batch the student trains on) -- the two are complementary, not
substitutes, and a report should name them separately.

    python3 scripts/train/distill_kd.py --teacher yolo26s.pt --student yolo26n.pt \\
        --data data/processed/person/data.yaml --epochs 50 --imgsz 640 \\
        --temperature 2.0 --alpha 0.5

Requires teacher and student to share the same `nc` (same class set) --
this is a same-task capacity transfer, not a class-remapping tool. A
mismatch is a loud `RuntimeError` from `KDDetectionLoss`, not a silently
wrong number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--teacher", required=True, help="a LARGER checkpoint (e.g. yolo26s/m/l.pt)")
ap.add_argument("--student", required=True, help="the checkpoint actually deployed (e.g. yolo26n.pt)")
ap.add_argument("--data", required=True, help="YOLO dataset YAML (path/train/val/names)")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="0")
ap.add_argument("--temperature", type=float, default=2.0,
                 help="softens both teacher and student logits before the KD term; "
                      "higher = softer, more inter-class information transferred")
ap.add_argument("--alpha", type=float, default=0.5,
                 help="weight of the KD term vs the normal task loss, in [0, 1]. "
                      "0 = plain training (KD term computed but not optimised against -- "
                      "use this to sanity-check the two losses actually differ). "
                      "1 = ignore ground truth, match the teacher only.")
ap.add_argument("--project", default="runs/kd")
ap.add_argument("--name", default="kd")
a = ap.parse_args()

if not 0.0 <= a.alpha <= 1.0:
    sys.exit(f"--alpha {a.alpha} must be in [0, 1]")
if a.temperature <= 0:
    sys.exit(f"--temperature {a.temperature} must be > 0")

try:
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"missing dependency: {e.name}\n  pip install ultralytics")

from labsafe.models.kd_loss import KDDetectionLoss  # noqa: E402


def _attach_kd_criterion(teacher_module, temperature: float, alpha: float):
    """Callback fired on `on_train_start`, once the student trainer has
    finished building its model and attached `.args` to it -- attaching
    any earlier fails, because `v8DetectionLoss.__init__` (which
    `KDDetectionLoss` extends) reads `model.args` and `model.model[-1]`,
    neither of which exist before the trainer's own setup runs. Verified
    for real, 2026-09-26: `trainer.model` at `on_train_start` is the bare
    `DetectionModel` (not yet DDP-wrapped for a single-device run), and
    `trainer.model.criterion` does not exist yet, so setting it here is the
    first and only assignment `BaseModel.loss()` will see."""

    def _callback(trainer):
        device = trainer.device
        teacher = teacher_module.to(device)
        trainer.model.criterion = KDDetectionLoss(
            trainer.model, teacher, temperature=temperature, alpha=alpha
        )
        print(f"[KD] attached KDDetectionLoss (T={temperature}, alpha={alpha}) on {device}")

    return _callback


teacher_model = YOLO(a.teacher).model
student = YOLO(a.student)

t_nc = teacher_model.model[-1].nc
s_nc = student.model.model[-1].nc
if t_nc != s_nc:
    sys.exit(
        f"teacher has nc={t_nc}, student has nc={s_nc} -- KDDetectionLoss compares "
        f"per-anchor class logits directly and does not remap between class sets. "
        f"Train both on the same data.yaml, or use scripts/train/distill_pseudolabel.py "
        f"instead if the teacher and student genuinely target different class sets."
    )

student.add_callback("on_train_start", _attach_kd_criterion(teacher_model, a.temperature, a.alpha))

print(f"teacher: {a.teacher} (nc={t_nc})   student: {a.student} (nc={s_nc})   "
      f"T={a.temperature}   alpha={a.alpha}\n")

student.train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    project=a.project, name=a.name,
)

print(f"""
========================================================================
  student trained with KD from {a.teacher}
  weights: {a.project}/{a.name}/weights/best.pt
========================================================================

  This is a training loss change, not a proof the student got better.
  Before claiming a win:
    1. Validate student WITH KD vs a plain run at --alpha 0 (same seed,
       same data, same epochs) -- compare mAP per slice, not just overall.
    2. If kd_loss in the training log did not move across epochs, the
       teacher and student agreed from the start (or the KD term is not
       actually contributing) -- investigate before reporting a KD result.
""")
