"""Model card sidecar: pins convert-time assumptions to runtime checks.

An RKNN (or TFLite) model's correct behaviour depends on facts that live
nowhere in the model file itself -- the mean/std used at convert time, the
imgsz it was traced with, whether it is end-to-end. Get any of those wrong
at load time and the failure mode is not an error, it is a model that
quietly detects nothing (`docs/KNOWN_ISSUES.md` #2, #4). The card is the single
place those facts are written once, at convert time, and read once, at
load time, so the two can never drift apart without the code refusing to
run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelCard:
    format: str                        # "rknn" | "onnx" | "tflite" | ...
    precision: str                     # "fp32" | "fp16" | "int8"
    imgsz: int
    layout: str                        # "nhwc" | "nchw"
    input_dtype: str                   # "uint8" | "float32"
    mean_values: list[float]
    std_values: list[float]
    end2end: bool
    target_platform: str | None
    class_names: list[str]
    source_weights: str
    toolkit_version: str | None
    ultralytics_version: str | None
    exported_at: str
    parity: dict | None = field(default=None)
    # "pixels" (letterbox-image pixels) or "normalized" (divided by the
    # input size). Ultralytics' RKNN INT8 export divides box coordinates by
    # imgsz so boxes and class scores fit one per-tensor INT8 scale
    # (`_NormalizeCoords`, ultralytics==8.4.163 exporter.py). Decoding those
    # as pixels does not error -- it yields boxes ~640x too small that match
    # nothing (docs/KNOWN_ISSUES.md #24). `None` = card written before this field
    # existed; RKNNBackend refuses to guess for INT8.
    box_coords: str | None = field(default=None)

    def as_dict(self) -> dict:
        """Plain dict, for embedding in metrics.json / bench metadata --
        `json.dumps(asdict(card))` everywhere else is more typing than
        this deserves."""
        return asdict(self)

    @staticmethod
    def sidecar_path(weights: str | Path) -> Path:
        """`<weights>.card.json`, always next to the weights file."""
        return Path(str(weights) + ".card.json")

    def save(self, weights: str | Path) -> Path:
        path = self.sidecar_path(weights)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, weights: str | Path) -> "ModelCard | None":
        """`None` when there is no sidecar -- an old model without one must
        still load (with a loud warning from the caller), not crash."""
        path = cls.sidecar_path(weights)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
