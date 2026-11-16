"""Teacher-student knowledge distillation loss for a YOLO detection head.

This is the literal thing Hinton, Vinyals & Dean, *Distilling the Knowledge
in a Neural Network* (2015, arXiv:1503.02531) describe: the student's loss
is the task loss PLUS a term that pulls the student's own output toward the
teacher's output on the SAME input, computed every training step. That is
different from `scripts/distill_pseudolabel.py`, which transfers knowledge
through the DATASET (the teacher's hard labels become training targets,
once, offline) rather than through the loss (soft targets, every step,
online). Both are legitimate; they are not the same technique, and a
report should not call one the other.

Scope, stated plainly:

  * This distills the CLASSIFICATION branch only (per-anchor class logits),
    temperature-scaled. Box regression and intermediate feature maps are
    NOT distilled -- that would be Chen et al. 2017 (NeurIPS)'s fuller
    feature-mimicking KD for detectors, which requires matching backbone
    feature maps layer-by-layer between two possibly different
    architectures. Classification-only KD is the well-defined subset that
    does not need that extra machinery, and is the honest scope to claim.
  * Hinton's original form is softmax + KL-divergence, because the
    classifier it targets is single-label. YOLO's classification head is
    per-anchor MULTI-label (independent objectness per class, trained with
    `BCEWithLogitsLoss`, not softmax) -- there is no softmax distribution
    over classes to take a KL divergence of. The closest well-defined
    analogue for a multi-label, per-anchor head is a temperature-scaled
    BCE between sigmoid-softened teacher and student logits, which is what
    this class computes. This is a real adaptation choice, not the paper's
    literal formula -- documented here so nobody re-derives it and assumes
    a mismatch is a bug.

YOLO26's ONE-TO-ONE head, verified for real, 2026-09-26: a YOLO26 checkpoint
loaded straight from ultralytics (`yolo26n.pt`, `yolo26s.pt`, ...) has a
`one2one_cv2` module on its `Detect` head **built into the architecture
itself**, independent of any `end2end=` export flag. `init_criterion()`
picks `E2ELoss` over plain `v8DetectionLoss` whenever that module is
present, and `E2ELoss` computes two full v8-style losses per step -- one
against the "one2many" (dense, anchor-per-cell) branch, one against the
"one2one" (sparse, NMS-free) branch -- blended with a schedule that shifts
weight from one2many toward one2one over training. Assuming a flat
`v8DetectionLoss` here (this class's first draft did) crashes on the very
first real training batch with `KeyError: 'boxes'`, because `preds` is
`{"one2many": {...}, "one2one": {...}}`, not the flat dict `v8DetectionLoss`
expects.

This class distills the **one2many branch only** -- the branch this
project's own export pipeline actually ships (`export_rknn.py` forces
`end2end=False`; `KNOWN_ISSUES.md` #2 documents `end2end=True`/the one2one
path as an active RK3588 segfault risk). The one2one branch keeps training
against ground truth only, via ultralytics' own unmodified `E2ELoss`
weighting -- KD is not applied there because that branch is never deployed
in this project.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


class KDDetectionLoss:
    """Drop-in replacement for a `DetectionModel`'s `.criterion`.

    Usage: build a frozen teacher `DetectionModel`, then, once the
    student's own trainer has attached `.args` to its model (see
    `scripts/distill_kd.py`, which does this from an `on_train_start`
    callback -- before that point `model.args` does not exist yet and
    `v8DetectionLoss.__init__` will fail), set:

        trainer.model.criterion = KDDetectionLoss(trainer.model, teacher)

    Every subsequent `trainer.model.loss(batch)` call then transparently
    includes the KD term, for either a plain (`v8DetectionLoss`) or
    one2many/one2one (`E2ELoss`) student architecture -- detected
    automatically from whether the student's `Detect` head has a
    `one2one_cv2` module, the same test `init_criterion()` itself uses.
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.is_e2e = getattr(student_model.model[-1], "one2one_cv2", None) is not None
        self.base = E2ELoss(student_model) if self.is_e2e else v8DetectionLoss(student_model)
        # the v8DetectionLoss instance whose (stateless) parse_output/branch
        # layout matches the branch we actually distill (one2many when e2e).
        self._branch = self.base.one2many if self.is_e2e else self.base

        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher_is_e2e = getattr(teacher_model.model[-1], "one2one_cv2", None) is not None
        self.T = temperature
        self.alpha = alpha

    def _extract_one2many(self, preds):
        """Pull the one2many-branch (dense, anchor-per-cell) prediction dict
        out of a model's raw forward output, whether that output came from
        training mode (dict, possibly nested under 'one2many') or eval mode
        (`(decoded_tensor, dict)` tuple -- see `Detect.forward`)."""
        parsed = self._branch.parse_output(preds)
        return parsed["one2many"] if "one2many" in parsed else parsed

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_total, loss_items = self.base(preds, batch)  # E2ELoss or v8DetectionLoss, unmodified

        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])

        s_out = self._extract_one2many(preds)
        # the teacher's own parse_output only needs to unwrap eval-mode's
        # tuple; the "pull out one2many if nested" step is the same either way
        t_raw = self._branch.parse_output(teacher_preds) if not self.teacher_is_e2e else (
            teacher_preds[1] if isinstance(teacher_preds, tuple) else teacher_preds
        )
        t_out = t_raw["one2many"] if isinstance(t_raw, dict) and "one2many" in t_raw else t_raw

        s_logits = s_out["scores"]
        t_logits = t_out["scores"]
        if s_logits.shape != t_logits.shape:
            raise RuntimeError(
                f"KD classification-logit shape mismatch: student {tuple(s_logits.shape)} "
                f"vs teacher {tuple(t_logits.shape)}. Teacher and student must be run at the "
                f"same imgsz and share the same nc and stride layout -- this class does not "
                f"remap between different class sets or resolutions."
            )

        T = self.T
        s_soft = torch.sigmoid(s_logits / T)
        t_soft = torch.sigmoid(t_logits / T)
        kd_loss = F.binary_cross_entropy(s_soft, t_soft, reduction="mean") * (T * T)

        batch_size = s_out["boxes"].shape[0]
        # base_total is a length-3 (box/cls/dfl) vector for a plain
        # v8DetectionLoss student and ALSO a length-3 vector for an E2ELoss
        # student (o2m-weighted + o2o-weighted, still unsummed) -- .sum()
        # first in both cases so alpha cannot broadcast the scalar kd_loss
        # across those components by mistake.
        total = (1.0 - self.alpha) * base_total.sum() + self.alpha * kd_loss * batch_size

        loss_items = dict(loss_items)
        loss_items["kd_loss"] = kd_loss.detach()
        return total, loss_items

    def update(self) -> None:
        """Forwarded to `E2ELoss.update()` (the one2many/one2one blend
        decay schedule) when the student is an E2E model; `BaseTrainer`
        calls `criterion.update()` at the end of every epoch if the
        attribute exists (`hasattr` check), so this must exist whenever
        `self.base` is an `E2ELoss`, or that schedule silently never
        advances instead of erroring -- silently wrong, not absent, is
        exactly the failure mode this project exists to avoid."""
        if hasattr(self.base, "update"):
            self.base.update()
