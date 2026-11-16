#!/usr/bin/env python3
"""Parity gate: does the converted model still detect anything?

Run this after every conversion, before a single FPS number from that
model goes into a report. See `docs/rknn-pipeline.md` and
`KNOWN_ISSUES.md` #3 for why: a model can convert cleanly, load cleanly,
and run at full speed while detecting nothing, and nothing else in this
repo notices.

    python3 scripts/validate_conversion.py \\
        --reference-backend onnx    --reference-weights runs/export/yolo26n_640.onnx \\
        --candidate-backend  rknn   --candidate-weights  runs/export/yolo26n_fp16_640.rknn \\
        --images data/calib/ --imgsz 640 --conf 0.25 --iou 0.5 \\
        --names person \\
        --out runs/parity/N1_fp16

Exit codes (kept distinct on purpose, so `bench_matrix_npu.sh` can tell a
dead model apart from a missing dependency):
    0  PASS (or FAIL with --allow-fail)
    1  infrastructure error -- could not build a backend, no images, wrong
       host architecture for RKNN inference
    2  FAIL -- the candidate is a research result to report, not a bug in
       this script
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from labsafe.models.backends import build_backend  # noqa: E402
from labsafe.models.parity import ParityThresholds, compare_detections  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference-backend", required=True)
    ap.add_argument("--reference-weights", default=None)
    ap.add_argument("--candidate-backend", required=True)
    ap.add_argument("--candidate-weights", default=None)
    ap.add_argument("--images", required=True, help="directory of calibration/validation images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="backend detection confidence")
    ap.add_argument("--iou", type=float, default=0.50, help="backend NMS IoU")
    ap.add_argument("--reference-conf", type=float, default=None,
                     help="override --conf for the reference backend only")
    ap.add_argument("--candidate-conf", type=float, default=None,
                     help="override --conf for the candidate backend only "
                          "-- e.g. set higher than every stub score to rehearse "
                          "a dead-model FAIL without a real broken model")
    ap.add_argument("--names", required=True, help="comma-separated class names, in class-id order")
    ap.add_argument("--out", required=True, help="directory to write parity.json/parity.md/config.json")
    ap.add_argument("--allow-fail", action="store_true",
                     help="exit 0 even on FAIL (still prints FAIL, still writes passed:false) "
                          "-- for deliberately collecting evidence of a broken conversion")
    ap.add_argument("--core-mask", default="auto", help="RKNN NPU core mask, if candidate is rknn")
    # Thresholds are CLI-settable, not hardcoded: the "right" numbers are a
    # research question (plan §6), not a fact this script should assert.
    ap.add_argument("--min-det-ratio", type=float, default=ParityThresholds.min_det_ratio)
    ap.add_argument("--max-det-ratio", type=float, default=ParityThresholds.max_det_ratio,
                    help="upper bound on cand/ref detection count -- catches a "
                         "model that reproduces the real boxes AND invents more")
    ap.add_argument("--min-precision", type=float, default=ParityThresholds.min_precision,
                    help="floor on matched/total candidate detections")
    ap.add_argument("--max-blank-rate", type=float, default=ParityThresholds.max_blank_rate)
    ap.add_argument("--min-matched-iou", type=float, default=ParityThresholds.min_matched_iou)
    ap.add_argument("--min-recall", type=float, default=ParityThresholds.min_recall)
    ap.add_argument("--max-mean-score-delta", type=float, default=ParityThresholds.max_mean_score_delta)
    ap.add_argument("--match-iou", type=float, default=ParityThresholds.match_iou,
                     help="IoU above which a reference/candidate box pair counts as the same object")
    return ap


def load_images(images_dir: str) -> list[tuple[str, "cv2.Mat"]]:
    p = Path(images_dir)
    if not p.is_dir():
        return []
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    out = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            out.append((f.name, img))
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    names = [n.strip() for n in args.names.split(",") if n.strip()]

    if args.candidate_backend == "rknn" and platform.machine() != "aarch64":
        print(
            "candidate backend is 'rknn' but this host is not aarch64: RKNN "
            "inference only runs on the board. Convert on x86 with "
            "scripts/export_rknn.py, copy the .rknn + .card.json to the "
            "Orange Pi 5, then run this exact command there. See "
            "docs/rknn-pipeline.md §0.",
            file=sys.stderr,
        )
        return 1

    images = load_images(args.images)
    if not images:
        print(f"no readable images found in {args.images!r}", file=sys.stderr)
        return 1

    def spec(backend: str, weights: str | None, conf: float) -> dict:
        s: dict = {"backend": backend, "imgsz": args.imgsz, "conf": conf, "iou": args.iou}
        if weights:
            s["weights"] = weights
        if backend == "rknn":
            s["core_mask"] = args.core_mask
        return s

    ref_conf = args.reference_conf if args.reference_conf is not None else args.conf
    cand_conf = args.candidate_conf if args.candidate_conf is not None else args.conf

    try:
        reference = build_backend(spec(args.reference_backend, args.reference_weights, ref_conf), names)
        candidate = build_backend(spec(args.candidate_backend, args.candidate_weights, cand_conf), names)
    except Exception as e:  # noqa: BLE001 - infrastructure failure, not a parity verdict
        print(f"failed to build backend(s): {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    filenames = [name for name, _ in images]
    ref_dets, cand_dets = [], []
    try:
        for _, frame in images:
            ref_dets.append(reference(frame))
        for _, frame in images:
            cand_dets.append(candidate(frame))
    except Exception as e:  # noqa: BLE001 - a crash mid-inference is infrastructure, not FAIL
        print(f"inference failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        for backend in (reference, candidate):
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    thresholds = ParityThresholds(
        min_det_ratio=args.min_det_ratio,
        max_det_ratio=args.max_det_ratio,
        min_precision=args.min_precision,
        max_blank_rate=args.max_blank_rate,
        min_matched_iou=args.min_matched_iou,
        min_recall=args.min_recall,
        max_mean_score_delta=args.max_mean_score_delta,
        match_iou=args.match_iou,
    )
    report = compare_detections(ref_dets, cand_dets, thresholds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity.json").write_text(json.dumps(report.as_dict(), indent=2))
    (out_dir / "parity.md").write_text(report.as_markdown())
    config = {
        "reference_backend": args.reference_backend,
        "reference_weights": args.reference_weights,
        "candidate_backend": args.candidate_backend,
        "candidate_weights": args.candidate_weights,
        "images_dir": args.images,
        "image_files": filenames,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "reference_conf": ref_conf,
        "candidate_conf": cand_conf,
        "iou": args.iou,
        "names": names,
        "thresholds": vars(thresholds),
        "allow_fail": args.allow_fail,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(report.as_markdown())
    print(f"\nwritten: {out_dir}")

    if report.passed:
        return 0
    if args.allow_fail:
        print("\n--allow-fail set: exiting 0 despite FAIL (see parity.json for the record)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
