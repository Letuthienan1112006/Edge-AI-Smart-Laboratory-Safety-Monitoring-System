#!/usr/bin/env python3
"""Convert a YOLO26 checkpoint to `.rknn` for the RK3588S NPU. x86 host only.

    python3 scripts/export_rknn.py yolo26n.pt --imgsz 640 --precision fp16 \\
            --target rk3588 --out runs/export/
    python3 scripts/export_rknn.py yolo26n.pt --imgsz 640 --precision int8 \\
            --target rk3588 --data data/processed/person/data.yaml --out runs/export/

See `docs/rknn-pipeline.md` for the full convert -> validate -> deploy
story. The export call signature below was verified against a real
`ultralytics==8.4.163` + `rknn-toolkit2==2.3.2` install on an x86 host,
2026-09-26 (see docs/rknn-pipeline.md "Verified" section for what changed
from the original, unverified draft of this script):

  * `--data` is a YOLO **dataset YAML** (`path:`/`train:`/`val:`/`names:`),
    the same file every other training/export script in this repo uses --
    NOT a bare directory of images. Ultralytics' own INT8 calibration path
    (`get_int8_calibration_dataloader` -> `check_det_dataset`) requires it;
    a directory was the wrong assumption in the original draft and would
    have failed with a confusing error the first time INT8 was attempted.
  * `int8=`/`end2end=` are accepted but DEPRECATED kwargs in this
    ultralytics version (silently mapped to `quantize=`/`nms=` with a
    warning printed on every run). This script uses the current names
    directly.

`ultralytics` is intentionally NOT imported at module load time: argument
validation (and `--help`) must work on a machine that has not installed
it yet, and `validate_args()` must be unit-testable without it.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.naming import (  # noqa: E402
    export_name,
    export_stem,
    onnx_reference_name,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights", help="a .pt YOLO26 checkpoint")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--precision", choices=["fp16", "int8"], required=True)
    ap.add_argument("--target", default="rk3588", help="RKNN target platform")
    ap.add_argument("--data", default=None,
                     help="YOLO dataset YAML (path/train/val/names) for INT8 calibration; "
                          "required for --precision int8. Same file your data.yaml already is "
                          "-- NOT a bare directory of images.")
    ap.add_argument("--calib-split", default="val", choices=["train", "val", "test"],
                     help="which split of --data to draw calibration images from")
    ap.add_argument("--min-calib-images", type=int, default=50,
                     help="same rule scripts/export_models.py already applies to TFLite INT8")
    ap.add_argument("--out", default="runs/export")
    ap.add_argument("--end2end", action="store_true",
                     help="export the end-to-end (NMS-free) head. SEGFAULTS at inference "
                          "on RK3588 (ultralytics#23753) -- requires --i-know-this-segfaults")
    ap.add_argument("--i-know-this-segfaults", action="store_true",
                     help="required acknowledgement to pass --end2end")
    also_onnx = ap.add_mutually_exclusive_group()
    also_onnx.add_argument("--also-onnx", dest="also_onnx", action="store_true", default=True,
                            help="also export an ONNX FP32 reference for the parity gate (default on)")
    also_onnx.add_argument("--no-also-onnx", dest="also_onnx", action="store_false",
                            help="skip the ONNX reference export")
    return ap


def _resolve_dataset_root(cfg: dict, yaml_path: Path) -> Path:
    """Mirror `ultralytics.data.utils.check_det_dataset`'s root resolution
    exactly -- verified by reading its source, 2026-09-26: a relative
    `path:` is tried AS-IS (i.e. relative to the current working directory)
    first, and only falls back to ultralytics' own configured
    `DATASETS_DIR` if that does not exist. It is never resolved relative to
    the yaml file's own directory.

    This project's own earlier version of this function got that wrong --
    it always resolved relative to the yaml's parent dir, which silently
    reports 0 calibration images for every dataset yaml that uses
    ultralytics' own standard convention (`path: <name>`, e.g. every stock
    coco*.yaml and, likely, this repo's own future data.yaml files) even
    though the real export call would have found the images fine. See
    KNOWN_ISSUES.md for the real run that caught this (yolo26n.pt + real
    coco128.yaml: this script said "0 images", the images were really
    there, at DATASETS_DIR/coco128/..., not next to the yaml).
    """
    root = Path(cfg.get("path", yaml_path.parent))
    if root.is_absolute():
        return root
    if root.exists():
        return root.resolve()
    try:
        from ultralytics.utils import DATASETS_DIR
        return (DATASETS_DIR / root).resolve()
    except ImportError:
        return root.resolve()


def _count_calib_images(data_yaml: str, split: str) -> tuple[int, str]:
    """Count images in `data_yaml`'s `split`, the way ultralytics itself
    resolves it: `path:` + the split's relative dir. Returns (count,
    resolved_dir) so a zero count can be reported with the path that was
    actually checked, not just the argument the user typed.

    This mirrors what `check_det_dataset` will do at export time (see
    docs/rknn-pipeline.md) closely enough to catch "wrong file", "missing
    split", and "empty split" before a slow export attempt -- it does not
    replace ultralytics' own validation, which also checks label files.
    """
    import yaml

    p = Path(data_yaml)
    if not p.is_file() or p.suffix.lower() not in {".yaml", ".yml"}:
        return 0, str(p)
    try:
        cfg = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError:
        return 0, str(p)
    split_rel = cfg.get(split)
    if not split_rel:
        return 0, f"{data_yaml} (no {split!r} key)"
    root = _resolve_dataset_root(cfg, p)
    split_dir = root / split_rel if not Path(split_rel).is_absolute() else Path(split_rel)
    if not split_dir.is_dir():
        return 0, str(split_dir)
    n = sum(1 for f in split_dir.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    return n, str(split_dir)


def validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argv AND enforce every hard rule before anything touches
    ultralytics or rknn-toolkit2. Exits the process (nonzero) on any
    violation -- this is deliberately not a pure function, so a bad
    invocation cannot fall through to a slow, expensive export attempt.
    """
    args = build_arg_parser().parse_args(argv)

    if platform.machine() == "aarch64":
        print(
            "RKNN export chỉ chạy trên host x86 Linux; trên board chỉ chạy "
            "inference. Convert on the x86 laptop with rknn-toolkit2, then "
            "copy the .rknn + .card.json to the board. See docs/rknn-pipeline.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.precision == "int8":
        if not args.data:
            print(
                "--precision int8 requires --data <dataset.yaml>. "
                "Quantising against no data (or random noise) produces an accuracy "
                "drop that is an artefact of the export, not of quantisation -- "
                "same rule as scripts/export_models.py.",
                file=sys.stderr,
            )
            sys.exit(1)
        if Path(args.data).suffix.lower() not in {".yaml", ".yml"}:
            print(
                f"--data {args.data!r} is not a .yaml file. RKNN INT8 calibration in "
                f"ultralytics reads a YOLO dataset YAML (path/train/val/names) via "
                f"check_det_dataset(), not a bare directory of images -- pass the same "
                f"data.yaml your training run used.",
                file=sys.stderr,
            )
            sys.exit(1)
        n, checked = _count_calib_images(args.data, args.calib_split)
        if n < args.min_calib_images:
            print(
                f"{args.calib_split!r} split of {args.data!r} has {n} image(s) at "
                f"{checked!r}, need at least {args.min_calib_images} for a usable "
                f"INT8 calibration set.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.end2end and not args.i_know_this_segfaults:
        print(
            "--end2end exports the NMS-free head, which SEGFAULTS at inference "
            "time on RK3588 (ultralytics issue #23753 -- see KNOWN_ISSUES.md #2). "
            "Pass --i-know-this-segfaults if you really mean to do this (e.g. to "
            "reproduce the bug for a report). The default (no --end2end) is the "
            "only supported path.",
            file=sys.stderr,
        )
        sys.exit(1)

    return args


def _toolkit_versions() -> tuple[str | None, str | None]:
    """Report versions; never assert a minimum. See docs/rknn-pipeline.md
    "Version gate: functional, not numeric" -- we do not know which
    ultralytics release first contains the fix for #23340/#23312, so the
    real gate is Phase 1 parity, not this number.

    Read from installed-package metadata, not `module.__version__`:
    `rknn-toolkit2` installs as the `rknn` namespace package, which has no
    `__version__` attribute at all (`import rknn; rknn.__version__` raises
    AttributeError-turned-None here, silently, on a real install -- this
    was wrong until verified against the real package on 2026-09-26, and
    would have left every ModelCard.toolkit_version empty despite a
    correctly installed toolkit, defeating the traceability this field
    exists for).
    """
    from importlib.metadata import PackageNotFoundError, version

    def _pkg_version(dist_name: str) -> str | None:
        try:
            return version(dist_name)
        except PackageNotFoundError:
            return None

    return _pkg_version("ultralytics"), _pkg_version("rknn-toolkit2")


def run_export(args: argparse.Namespace) -> Path:
    """The part that actually needs ultralytics + rknn-toolkit2 installed.
    Kept separate from `validate_args` so the latter is unit-testable
    without either package."""
    from ultralytics import YOLO

    from labsafe.models.modelcard import ModelCard

    ultra_version, toolkit_version = _toolkit_versions()
    print(f"ultralytics: {ultra_version or '(not detected)'}")
    print(f"rknn-toolkit2: {toolkit_version or '(not detected)'}")
    print(
        "NOTE: this script does not gate on either version number. Known "
        "upstream bugs: ultralytics#23340 (INT8 export, fixed by PR #23312) "
        "and ultralytics#23753 (end2end segfault / INT8 zero-detection on "
        "RK3588). The real gate is `scripts/validate_conversion.py` "
        "(functional parity), run AFTER this export. See KNOWN_ISSUES.md."
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    # NOT export kwargs -- ultralytics/rknn-toolkit2 bake this normalisation
    # into the graph themselves (there is no mean_values=/std_values= export
    # argument to set). These two lists only document, for the ModelCard,
    # what preprocessing the resulting .rknn expects: raw uint8 [0,255],
    # matching postprocess.letterbox(normalize=False) on the inference side.
    mean_values = [0.0, 0.0, 0.0]
    std_values = [255.0, 255.0, 255.0]

    export_kwargs = dict(
        format="rknn",
        imgsz=args.imgsz,
        name=args.target,
        # `quantize`/`nms` are the current parameter names (verified against
        # ultralytics==8.4.163 exporter.py, 2026-09-26). The older `int8=`/
        # `end2end=` kwargs still work via a deprecation shim but print a
        # warning on every run -- see docs/rknn-pipeline.md "Verified".
        quantize=(8 if args.precision == "int8" else 16),
        # end2end=True maps to nms=False (the risky NMS-free/Top-K head that
        # segfaults on RK3588, ultralytics#23753); the default path here
        # omits `nms` entirely, which is what end2end=False used to mean.
        **({"nms": False} if args.end2end else {}),  # ack already enforced above
    )
    if args.precision == "int8":
        export_kwargs["data"] = args.data
        export_kwargs["split"] = args.calib_split
    try:
        exported = model.export(**export_kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"model.export() rejected one of these kwargs: {export_kwargs}. "
            f"This signature was verified against ultralytics==8.4.163 on "
            f"2026-09-26 (see docs/rknn-pipeline.md 'Verified'), so a failure "
            f"here most likely means the installed ultralytics version has "
            f"changed its export API again -- open "
            f"site-packages/ultralytics/engine/exporter.py, find the 'rknn' "
            f"branch, and update export_kwargs here to match. "
            f"Original error: {e}"
        ) from e

    # The naming rule lives in labsafe.models.naming so producers,
    # consumers and tests all read it from one place -- see that module's
    # docstring for the failure that made it necessary.
    stem = export_stem(args.weights)
    dest = out_dir / export_name(stem, args.precision, args.imgsz)
    _collect_export(exported, dest)

    names = model.names
    class_names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    # Mirror the exact condition under which ultralytics wraps the model in
    # `_NormalizeCoords` before tracing (exporter.py, rknn + quantize=8 +
    # detect + not end2end; verified on 8.4.163). The parity gate is what
    # actually proves it -- decoding with the wrong convention makes every
    # box miss (KNOWN_ISSUES.md #24).
    try:
        from ultralytics.utils.export.engine import _NormalizeCoords  # noqa: F401
        normalizes = True
    except ImportError:
        normalizes = False
    box_coords = ("normalized" if args.precision == "int8" and not args.end2end and normalizes
                  else "pixels")

    card = ModelCard(
        format="rknn",
        precision=args.precision,
        imgsz=args.imgsz,
        layout="nhwc",
        input_dtype="uint8",
        mean_values=mean_values,
        std_values=std_values,
        end2end=args.end2end,
        target_platform=args.target,
        class_names=class_names,
        source_weights=str(args.weights),
        toolkit_version=toolkit_version,
        ultralytics_version=ultra_version,
        exported_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        box_coords=box_coords,
    )
    card.save(dest)
    print(f"\n-> {dest}")
    print(f"-> {ModelCard.sidecar_path(dest)}")

    if args.also_onnx:
        onnx_out = model.export(format="onnx", imgsz=args.imgsz)
        onnx_dest = out_dir / onnx_reference_name(stem, args.imgsz)
        _collect_export(onnx_out, onnx_dest, suffix=".onnx")
        print(f"-> {onnx_dest}  (FP32 reference for the parity gate)")

    print(
        "\nNext (parity gate, MUST run before this model's numbers go into a "
        "report):\n"
        f"  python3 scripts/validate_conversion.py \\\n"
        f"      --reference-backend onnx  --reference-weights {out_dir / onnx_reference_name(stem, args.imgsz)} \\\n"
        f"      --candidate-backend rknn  --candidate-weights {dest} \\\n"
        f"      --images data/calib/ --imgsz {args.imgsz} --names <your_classes> \\\n"
        f"      --out runs/parity/{stem}_{args.precision}\n"
        "  NOTE: the candidate side of that command only runs on the board "
        "(aarch64) -- RKNN inference does not run on this x86 host."
    )
    return dest


def _collect_export(exported, dest: Path, suffix: str = ".rknn") -> Path:
    """Move whatever `model.export()` produced to `dest`, or fail loudly.

    Ultralytics returns a FILE for some formats and a DIRECTORY for the
    packaged ones (`*_rknn_model/`) -- this repo already relies on that in
    scripts/export_models.py. The previous form here was
    `Path(exported).replace(dest) if Path(exported).is_file() else None`,
    which on the directory branch did nothing at all, silently: no move, no
    error, and then a model card written beside a path that does not exist
    and a success line printed for it. A conversion step must never report
    a destination it did not create.
    """
    src = Path(exported)
    if not src.exists():
        raise RuntimeError(
            f"export reported success but produced nothing at {src!r}. "
            f"Do not trust the return value alone -- check the export log above."
        )
    if src.is_file() and src.suffix != suffix:
        raise RuntimeError(
            f"export returned {src.name!r}, which is not a *{suffix} file. "
            f"Renaming it to {dest.name!r} would produce an artefact whose "
            f"extension lies about its format."
        )
    if src.is_dir():
        candidates = sorted(src.rglob(f"*{suffix}"))
        if not candidates:
            raise RuntimeError(
                f"export produced directory {src} but it contains no *{suffix} "
                f"file. Contents: {[p.name for p in sorted(src.iterdir())][:10]}"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"export produced {len(candidates)} *{suffix} files in {src}: "
                f"{[p.name for p in candidates]}. Refusing to guess which one "
                f"is the model."
            )
        src = candidates[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    # NOT Path.replace(): that is a bare rename() and raises
    # "Invalid cross-device link" the moment `--out` resolves to a
    # different filesystem/mount than wherever ultralytics wrote its
    # output (verified against a real export, 2026-09-26 -- ultralytics
    # writes beside the source checkpoint, ignoring our --out entirely,
    # so collecting across a mount boundary is the NORMAL case here, not
    # an edge case). shutil.move() falls back to copy+delete when rename()
    # cannot be used atomically.
    shutil.move(str(src), str(dest))
    if not dest.is_file():
        raise RuntimeError(f"move to {dest} did not produce a file")
    return dest


def main(argv: list[str] | None = None) -> int:
    args = validate_args(argv)
    run_export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
