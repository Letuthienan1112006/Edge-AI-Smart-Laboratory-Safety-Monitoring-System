"""The bench loop must stop when the work runs out, not spin to the clock.

Regression guard for a real reproduction: a 2-second video clip benchmarked
with `--duration 6` reported `fps_sustained: 723174` over 4,383,050
"iterations" while the router had processed 60 frames. The loop only ever
checked wall-clock time, so once the file ended it kept calling a step that
returned immediately and counted each no-op as an iteration.

That number is not an outlier a reader spots in context -- it is the
headline figure of the run, and it is fabricated. This project's whole
premise is refusing to emit numbers like it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.app import _spin_warning  # noqa: E402
from labsafe.bench.runner import BenchConfig, BenchRunner  # noqa: E402


def _runner(tmp_path, **kw):
    cfg = BenchConfig(duration_s=5.0, warmup_s=0.0, sample_interval_s=10.0,
                      sustained_window_s=5.0, out_dir=str(tmp_path), exp_id="T", **kw)
    return BenchRunner(cfg)


def test_runner_stops_when_step_returns_false(tmp_path):
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        return False if calls["n"] >= 5 else None

    result = _runner(tmp_path).run(step)

    assert calls["n"] == 5, "loop kept calling step after it said the source was gone"
    assert result["stop_reason"] == "source_exhausted"
    assert result["ended_early"] is True
    assert result["short_by_s"] > 0


def test_runner_records_duration_reached_when_nothing_stops_it(tmp_path):
    cfg = BenchConfig(duration_s=0.3, warmup_s=0.0, sample_interval_s=10.0,
                      sustained_window_s=1.0, out_dir=str(tmp_path), exp_id="T")
    result = BenchRunner(cfg).run(lambda: None)
    assert result["stop_reason"] == "duration_reached"
    assert result["ended_early"] is False


def test_returning_none_keeps_the_old_contract(tmp_path):
    """Existing callers return None and must not be stopped by it."""
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        return None

    cfg = BenchConfig(duration_s=0.3, warmup_s=0.0, sample_interval_s=10.0,
                      sustained_window_s=1.0, out_dir=str(tmp_path), exp_id="T")
    BenchRunner(cfg).run(step)
    assert calls["n"] > 1


# -- the independent cross-check -----------------------------------------

def test_spin_warning_fires_when_iterations_dwarf_frames():
    """The two counters come from different places (runner vs router) and
    already sat side by side in metrics.json; nothing was comparing them."""
    w = _spin_warning({"iterations_total": 4_383_050, "router": {"frames": 60},
                       "stop_reason": "duration_reached"})
    assert w is not None
    assert "4383050" in w and "60" in w


def test_spin_warning_silent_when_counters_agree():
    assert _spin_warning({"iterations_total": 900, "router": {"frames": 880}}) is None


def test_spin_warning_silent_on_runs_too_short_to_judge():
    assert _spin_warning({"iterations_total": 10, "router": {"frames": 1}}) is None


def test_spin_warning_silent_without_router_stats():
    """A bench run of a bare backend has no router; absence is not a finding."""
    assert _spin_warning({"iterations_total": 5000}) is None
