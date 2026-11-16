"""Every consumer of an exported artefact name must agree with the producer.

The bug this guards: the export filename gained an `imgsz` and the
experiment configs were updated, but the benchmark matrix, one config and
the documented command were not. The dangerous case is not the crash --
it is both files existing, so the parity gate validates
`model_fp16.rknn` while the benchmark loads `model_fp16_640.rknn`, and
the report carries numbers from a model the gate never checked.

These tests read the real config and script files and check every
`runs/export/...` path they mention against `labsafe.models.naming` --
the function the exporter itself calls. Asserting against a regex copied
into the test would agree with a stale consumer just as happily.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labsafe.models.naming import export_name, onnx_reference_name  # noqa: E402

EXPORT_PATH = re.compile(r"runs/export/[A-Za-z0-9_.${}]+\.(?:rknn|onnx)")


def _producible(path: str) -> bool:
    """Could `scripts/export_rknn.py` ever have written this name?"""
    name = Path(path).name
    if "$" in name:            # shell-interpolated, checked separately below
        return True
    for stem in ("yolo26n", "yolo26s", "yolo26m"):
        for imgsz in (320, 416, 512, 640, 960):
            if name == onnx_reference_name(stem, imgsz):
                return True
            for precision in ("fp16", "int8", "fp32"):
                if name == export_name(stem, precision, imgsz):
                    return True
    return False


def test_naming_rule_is_what_the_configs_encode():
    """Every weights path in an N* config must be a name the exporter produces."""
    bad = []
    for path in sorted((ROOT / "configs" / "exp").glob("N*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        for stage, block in cfg.items():
            if not isinstance(block, dict):
                continue
            w = block.get("weights")
            if isinstance(w, str) and w.startswith("runs/export/") and not _producible(w):
                bad.append(f"{path.name}:{stage} -> {w}")
    assert not bad, (
        "these configs point at names scripts/export_rknn.py never writes:\n  "
        + "\n  ".join(bad)
    )


def test_matrix_script_does_not_hardcode_export_paths():
    """The matrix must derive artefact paths from variables, not retype them.

    Retyped paths are what drifted last time: the configs were updated and
    the script was not, and nothing failed until the board was in front of
    someone.
    """
    text = (ROOT / "scripts" / "bench_matrix_npu.sh").read_text()
    literals = [m for m in EXPORT_PATH.findall(text) if "$" not in m]
    assert not literals, (
        "hardcoded export paths in bench_matrix_npu.sh: "
        f"{sorted(set(literals))}. Derive them from STEM/BASE_IMGSZ instead."
    )


def test_documented_export_command_matches_the_real_cli():
    """`weights` is positional. A documented `--weights` errors out."""
    offenders = []
    for path in list((ROOT / "configs" / "exp").glob("*.yaml")) + [
        ROOT / "scripts" / "bench_matrix_npu.sh",
        ROOT / "docs" / "rknn-pipeline.md",
    ]:
        if "export_rknn.py --weights" in path.read_text():
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} document `export_rknn.py --weights`, but the argument is "
        "positional -- the command as printed fails with 'unrecognized arguments'."
    )


def test_export_name_requires_imgsz_to_differ_per_resolution():
    names = {export_name("m", "fp16", sz) for sz in (320, 416, 512, 640)}
    assert len(names) == 4, f"resolutions collide on one filename: {names}"


def test_onnx_reference_is_shared_across_precisions_but_not_sizes():
    assert onnx_reference_name("m", 640) == onnx_reference_name("m", 640)
    assert onnx_reference_name("m", 640) != onnx_reference_name("m", 320)
    assert onnx_reference_name("m", 640) != export_name("m", "fp16", 640)
