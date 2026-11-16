"""Task x engine resolution (O8, the central contribution). Every test
here runs on a synthetic CPUInfo -- no board required, per plan §2.6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_optimize import _rk3588s_info, pi4  # noqa: E402 - shared fixture helpers

from labsafe.optimize.engine_map import (  # noqa: E402
    KNOWN_ENGINES,
    resolve,
    resolve_stage_engines,
)


def test_resolve_cpu_big_on_rk3588s_gives_4567():
    p = resolve("cpu_big", _rk3588s_info())
    assert p.cpus == [4, 5, 6, 7]


def test_resolve_cpu_little_gives_0123():
    p = resolve("cpu_little", _rk3588s_info())
    assert p.cpus == [0, 1, 2, 3]


def test_resolve_npu_carries_core_mask():
    cfg = {"stage_a": {"backend": "rknn", "engine": "npu", "core_mask": "all"}}
    placements = resolve_stage_engines(cfg, _rk3588s_info())
    assert placements["stage_a"].engine == "npu"
    assert placements["stage_a"].core_mask == "all"
    assert placements["stage_a"].cpus is None


def test_resolve_gpu_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        resolve("gpu", _rk3588s_info())


def test_unknown_engine_lists_the_known_ones():
    with pytest.raises(ValueError) as exc:
        resolve("quantum_annealer", _rk3588s_info())
    for name in KNOWN_ENGINES:
        assert name in str(exc.value)


def test_uniform_cpu_returns_no_mask_with_explanation():
    p = resolve("cpu_big", pi4())
    assert p.cpus is None
    assert "uniform" in p.note


def test_resolve_stage_engines_defaults_to_cpu_any_when_absent():
    cfg = {"stage_a": {"backend": "stub", "imgsz": 320}}  # no "engine" key at all
    placements = resolve_stage_engines(cfg, _rk3588s_info())
    assert placements["stage_a"].engine == "cpu_any"
    assert placements["stage_a"].cpus is None


def test_resolve_stage_engines_ignores_non_stage_keys():
    cfg = {"stage_a": {"engine": "cpu_any"}, "affinity": {"cpus": "big"}, "note": "x"}
    placements = resolve_stage_engines(cfg, _rk3588s_info())
    assert set(placements) == {"stage_a"}
