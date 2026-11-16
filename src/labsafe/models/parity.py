"""Functional parity gate: does the converted model still see anything?

This is the single most important file in the RKNN work. A model can
convert without error, load without error, and run at full FPS while
detecting nothing (ultralytics#23753 -- see `docs/KNOWN_ISSUES.md` #3). Every
metric downstream of that -- latency, throughput, power -- is a number
about a dead model, and nothing about "the pipeline ran" would tell you.

`compare_detections()` is a pure function over `list[list[Detection]]` on
purpose: it never touches a backend, a file, or the network, so it is
testable with synthetic detections and never needs a board or a real
model to exercise every branch (`docs/experiment-ids.md` §2.6 of the
plan). `scripts/export/validate_conversion.py` is the only piece that talks to
real backends and real images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backends import Detection


@dataclass
class ParityThresholds:
    """Pass/fail knobs. Kept as CLI-settable parameters, not constants,
    because the "right" numbers are a research question, not a fact --
    see plan §6 "Điểm chưa chắc chắn nhưng KHÔNG chặn kế hoạch"."""

    min_det_ratio: float = 0.60
    # Upper bound and precision floor. Without them the gate is one-sided:
    # a candidate that reproduces every reference box AND invents twenty more
    # scores recall 1.0, blank_rate 0.0 and det_ratio 21 -- and PASSES, while
    # being useless in the field (every spurious box is a false alarm the
    # safety system acts on). Spurious low-confidence detections are as real
    # an INT8 failure mode as the zero-detection collapse this gate was
    # built for; they are just the other direction.
    max_det_ratio: float = 1.60
    min_precision: float = 0.70
    max_blank_rate: float = 0.30
    min_matched_iou: float = 0.70
    min_recall: float = 0.80
    max_mean_score_delta: float = 0.25
    match_iou: float = 0.50


@dataclass
class ParityReport:
    passed: bool
    failures: list[str]
    metrics: dict[str, Any]
    per_image: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": dict(self.metrics),
            "per_image": [dict(p) for p in self.per_image],
        }

    def as_markdown(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"## Parity verdict: {verdict}", ""]
        if self.failures:
            lines.append("**Failures:**")
            lines += [f"- `{f}`" for f in self.failures]
            lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, v in self.metrics.items():
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)


def _xyxy_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _match_one_image(
    ref: list[Detection], cand: list[Detection], match_iou: float
) -> list[tuple[int, int, float]]:
    """Greedy IoU matching, highest IoU first, never crossing class_id.

    Global-greedy (rather than per-class-then-greedy) makes no difference
    to the result here because a cross-class pair is never a candidate --
    filtering by `cls_id` up front keeps the sort short.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            if r.cls_id != c.cls_id:
                continue
            iou = _xyxy_iou(r, c)
            if iou >= match_iou:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda t: t[0], reverse=True)

    used_ref: set[int] = set()
    used_cand: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        matched.append((i, j, iou))
    return matched


def compare_detections(
    reference: list[list[Detection]],
    candidate: list[list[Detection]],
    thresholds: ParityThresholds = ParityThresholds(),
) -> ParityReport:
    """Compare a candidate model's detections against a reference model's,
    image for image, and decide whether the candidate is still alive.

    `failures` is empty if and only if `passed` is True -- there is no
    third state. Every failure has a machine-readable code so a report
    generator (or a human grepping a log) never has to parse English to
    find out what broke.
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"reference has {len(reference)} images but candidate has "
            f"{len(candidate)} -- refusing to zip them together silently. "
            "Both must be run over exactly the same, identically ordered "
            "image list."
        )

    n_images = len(reference)
    total_ref = sum(len(r) for r in reference)
    total_cand = sum(len(c) for c in candidate)

    failures: list[str] = []

    if total_ref == 0:
        # An empty reference set proves nothing -- a candidate that also
        # detects nothing would pass by accident. Refuse outright rather
        # than reporting a meaningless PASS.
        failures.append("reference_has_no_detections")
        metrics = {
            "n_images": n_images,
            "total_ref_dets": 0,
            "total_cand_dets": total_cand,
            "det_ratio": None,
            "precision": None,
            "blank_rate": None,
            "recall": None,
            "mean_matched_iou": None,
            "mean_score_delta": None,
        }
        return ParityReport(passed=False, failures=failures, metrics=metrics, per_image=[])

    if total_cand == 0:
        # Named separately, and placed first: this is exactly
        # ultralytics#23753 (docs/KNOWN_ISSUES.md #3), and it must never be
        # absorbed into a generic "low recall" bucket.
        failures.append("zero_detection_collapse")

    images_with_ref = sum(1 for r in reference if r)
    blank_images = sum(1 for r, c in zip(reference, candidate) if r and not c)
    blank_rate = blank_images / images_with_ref if images_with_ref else 0.0

    per_image: list[dict[str, Any]] = []
    all_ious: list[float] = []
    all_score_deltas: list[float] = []
    total_matched = 0

    for idx, (r, c) in enumerate(zip(reference, candidate)):
        matched = _match_one_image(r, c, thresholds.match_iou)
        ious = [iou for _, _, iou in matched]
        deltas = [abs(c[j].score - r[i].score) for i, j, _ in matched]
        total_matched += len(matched)
        all_ious.extend(ious)
        all_score_deltas.extend(deltas)
        per_image.append({
            "index": idx,
            "n_ref": len(r),
            "n_cand": len(c),
            "n_matched": len(matched),
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
            "blank": bool(r and not c),
        })

    det_ratio = total_cand / total_ref
    recall = total_matched / total_ref
    precision = total_matched / total_cand if total_cand else 0.0
    mean_matched_iou = sum(all_ious) / len(all_ious) if all_ious else None
    mean_score_delta = (
        sum(all_score_deltas) / len(all_score_deltas) if all_score_deltas else None
    )

    if det_ratio < thresholds.min_det_ratio:
        failures.append("low_detection_ratio")
    if det_ratio > thresholds.max_det_ratio:
        failures.append("excess_detections")
    if precision < thresholds.min_precision:
        failures.append("low_precision")
    if blank_rate > thresholds.max_blank_rate:
        failures.append("high_blank_rate")
    if recall < thresholds.min_recall:
        failures.append("low_recall")
    if mean_matched_iou is not None and mean_matched_iou < thresholds.min_matched_iou:
        failures.append("low_matched_iou")
    if mean_score_delta is not None and mean_score_delta > thresholds.max_mean_score_delta:
        failures.append("high_mean_score_delta")

    metrics = {
        "n_images": n_images,
        "total_ref_dets": total_ref,
        "total_cand_dets": total_cand,
        "det_ratio": round(det_ratio, 4),
        "blank_rate": round(blank_rate, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mean_matched_iou": round(mean_matched_iou, 4) if mean_matched_iou is not None else None,
        "mean_score_delta": round(mean_score_delta, 4) if mean_score_delta is not None else None,
    }

    return ParityReport(
        passed=(len(failures) == 0), failures=failures, metrics=metrics, per_image=per_image
    )
