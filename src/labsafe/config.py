"""Config loading. One source of truth, no scattered defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Config:
    """Pipeline config plus the class taxonomy it depends on."""

    pipeline: dict[str, Any]
    classes: dict[str, Any]
    zones: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        pipeline: str | Path = "configs/pipeline.yaml",
        classes: str | Path = "configs/classes.yaml",
    ) -> "Config":
        pl = load_yaml(pipeline)
        cl = load_yaml(classes)
        zones: dict[str, Any] = {}
        zcfg = pl.get("zones", {})
        if zcfg.get("enabled") and zcfg.get("file"):
            zones = load_yaml(zcfg["file"])
        return cls(pipeline=pl, classes=cl, zones=zones)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.pipeline
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node
