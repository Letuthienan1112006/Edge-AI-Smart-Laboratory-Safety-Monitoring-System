"""One place that decides what an exported artefact is called.

This exists because of a real failure: the export filename gained an
`imgsz` (RKNN bakes the input shape in at convert time, so a resolution
sweep needs one file per point), the experiment configs were updated to
match -- and the CONSUMERS of that name were not. The benchmark matrix
still gated parity on the old path, one config still pointed at the old
ONNX reference, and the documented command still produced the old name.

The dangerous shape of that mismatch is not the crash. It is the case
where both files exist: the parity gate validates `model_fp16.rknn` while
the benchmark run loads `model_fp16_640.rknn`, and the report then
carries numbers from a model the gate never checked. A gate that verifies
a different artefact than the one measured is worse than no gate, because
it is believed.

So the rule lives in code, is imported by every producer, and is what the
tests assert against -- never a regex copied into a test file, which
agrees with a stale consumer just as happily as with a fresh one.
"""

from __future__ import annotations

from pathlib import Path


def export_stem(weights: str | Path) -> str:
    """The base name an export derives from its source checkpoint."""
    return Path(weights).stem


def export_name(stem: str, precision: str, imgsz: int, suffix: str = ".rknn") -> str:
    """Name of a converted model: `<stem>_<precision>_<imgsz><suffix>`.

    `imgsz` is mandatory and not optional-with-a-default on purpose: every
    caller must state it, because omitting it is exactly the bug this
    module was written for.
    """
    return f"{stem}_{precision}_{int(imgsz)}{suffix}"


def onnx_reference_name(stem: str, imgsz: int) -> str:
    """Name of the FP32 ONNX the parity gate compares against.

    No precision segment: there is only ever one reference per size, and
    calling it `_fp32_` would invite a second one.
    """
    return f"{stem}_{int(imgsz)}.onnx"
