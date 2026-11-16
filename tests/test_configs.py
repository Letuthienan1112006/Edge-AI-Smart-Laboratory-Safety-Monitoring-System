"""Every `configs/exp/*.yaml` file must parse and must not silently sneak
dashboard or GPIO on into a benchmark run -- both are cost the harness
would then attribute to the model instead of to itself.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

EXP_DIR = Path(__file__).resolve().parents[1] / "configs" / "exp"
EXP_FILES = sorted(EXP_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", EXP_FILES, ids=lambda p: p.name)
def test_every_exp_config_parses(path: Path):
    cfg = yaml.safe_load(path.read_text())
    assert "stage_a" in cfg, f"{path.name} missing stage_a"
    assert "backend" in cfg["stage_a"], f"{path.name} stage_a missing backend"
    assert "imgsz" in cfg["stage_a"], f"{path.name} stage_a missing imgsz"


@pytest.mark.parametrize("path", EXP_FILES, ids=lambda p: p.name)
def test_every_exp_config_disables_dashboard_and_gpio(path: Path):
    cfg = yaml.safe_load(path.read_text())
    assert cfg.get("dashboard", {}).get("enabled") is False, (
        f"{path.name}: dashboard must be off in every bench config -- JPEG "
        f"encode cost would be attributed to the model instead of to the dashboard"
    )
    assert cfg.get("alerts", {}).get("gpio") == "off", (
        f"{path.name}: gpio must be 'off' in every bench config"
    )


@pytest.mark.parametrize("path", [p for p in EXP_FILES if p.name.startswith("N")], ids=lambda p: p.name)
def test_rknn_configs_declare_engine_npu(path: Path):
    cfg = yaml.safe_load(path.read_text())
    if cfg["stage_a"].get("backend") == "rknn":
        assert cfg["stage_a"].get("engine") == "npu", (
            f"{path.name}: an rknn-backed stage must declare engine: npu "
            f"(see src/labsafe/optimize/engine_map.py)"
        )


# -- resolution sweep: each point needs its own export --------------------
#
# Regression guard: N3a/b/c once declared imgsz 320/416/512 while all three
# pointed at `yolo26n_fp16.rknn`, the file N1 exported at 640. RKNN bakes the
# input shape in at convert time, so on the board the model-card guard would
# raise -- mid-matrix, after N0/N1/N2 had already burned hours, and before
# the report step ever ran. Nothing caught it on the laptop because the
# models do not exist here.

import re  # noqa: E402


def _rknn_stage_a_configs():
    out = []
    for path in sorted(EXP_DIR.glob("N*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        a = (cfg or {}).get("stage_a", {})
        if str(a.get("backend", "")).lower() in {"rknn", "npu"} and a.get("weights"):
            out.append((path.name, str(a["weights"]), int(a["imgsz"])))
    return out


def test_every_rknn_config_points_at_a_model_exported_at_its_own_imgsz():
    configs = _rknn_stage_a_configs()
    assert configs, "no RKNN configs found -- has the N* matrix moved?"
    for name, weights, imgsz in configs:
        m = re.search(r"_(\d+)\.rknn$", weights)
        assert m, (
            f"{name}: weights {weights!r} has no imgsz in its filename. Without "
            f"it, a resolution sweep overwrites one file and every point "
            f"benchmarks the last export."
        )
        assert int(m.group(1)) == imgsz, (
            f"{name}: declares imgsz {imgsz} but points at {weights!r}, which "
            f"was exported at {m.group(1)}. RKNN fixes the input shape at "
            f"convert time -- this cannot run."
        )


def test_resolution_sweep_points_use_distinct_models():
    sweep = {n: w for n, w, _ in _rknn_stage_a_configs() if n.startswith("N3")}
    assert len(sweep) >= 3, f"expected N3a/b/c, found {sorted(sweep)}"
    assert len(set(sweep.values())) == len(sweep), (
        f"resolution sweep points share a model file: {sweep}"
    )
