"""`labsafe.bench.report` is the last line of defence against a dead model
sneaking a shiny FPS number into the results table (`KNOWN_ISSUES.md` #3).
These tests build synthetic `runs/<exp>/metrics.json` files -- no bench
run, no model, no board -- and check that:

  * the two new telemetry columns (`A det/call`, `A blank`) actually show
    up with the right numbers, not just that the code does not crash.
  * a run where stage A was called a lot and detected nothing gets called
    out BY NAME in the "detected nothing" warning.
  * a run where stage A was never invoked at all (e.g. `stage_a` disabled,
    or a config that never reached the loop) is NOT swept into that
    warning just because its detection count also happens to be 0 --
    `stage_calls.a == 0` and `stage_detections.a == 0` mean two very
    different things and must not collapse into the same warning.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.bench.report import collect, dig, to_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = "/usr/bin/python3"


def _run_report(runs_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [PY, "-m", "labsafe.bench.report", str(runs_dir)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


def _write_run(runs_dir: Path, exp_id: str, *, stage_calls_a: int, stage_detections_a: int,
                result_valid=None, ended_early=None, short_by_s=None) -> None:
    run_dir = runs_dir / f"{exp_id}-fake"
    run_dir.mkdir(parents=True)
    metrics = {
        "exp_id": exp_id,
        "mode": "parallel",
        "backend": "stub",
        "imgsz": 320,
        "latency_ms": {"p50": 10.0, "p90": 12.0, "p99": 15.0},
        "fps_sustained": 30.0,
        **({"ended_early": ended_early} if ended_early is not None else {}),
        **({"short_by_s": short_by_s} if short_by_s is not None else {}),
        "system": {"rss_mb": {"max": 100.0}, "temp_c": {"max": 40.0},
                   **({"result_valid": result_valid} if result_valid is not None else {})},
        "router": {
            "occupancy": 0.5,
            "stage_calls_per_frame": {"b": 0.1, "c": 0.05},
            "stage_calls": {"a": stage_calls_a, "b": 0, "c": 0},
            "stage_detections": {"a": stage_detections_a, "b": 0, "c": 0},
            "stage_detections_per_call": {
                "a": round(stage_detections_a / stage_calls_a, 3) if stage_calls_a else 0.0
            },
            "stage_blank_rate": {
                "a": 0.0 if stage_detections_a else (1.0 if stage_calls_a else 0.0)
            },
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_markdown_table_has_the_new_telemetry_columns(tmp_path):
    _write_run(tmp_path, "HEALTHY", stage_calls_a=100, stage_detections_a=100)
    rows = collect(tmp_path)
    md = to_markdown(rows)
    assert "A det/call" in md
    assert "A blank" in md
    assert "1.0" in md  # HEALTHY's det/call


def test_dig_reads_nested_router_paths():
    row = {"router": {"stage_detections_per_call": {"a": 0.5}}}
    assert dig(row, "router.stage_detections_per_call.a") == 0.5
    assert dig(row, "router.stage_detections_per_call.b") is None
    assert dig(row, "does.not.exist") is None


def test_cli_flags_a_dead_model_run_by_name(tmp_path):
    """End-to-end through the actual `__main__` block (subprocess, not
    imported), because the dead-model warning logic lives there, not in a
    separately importable function."""
    _write_run(tmp_path, "DEAD_ONE", stage_calls_a=6000, stage_detections_a=0)
    _write_run(tmp_path, "ALIVE_ONE", stage_calls_a=6000, stage_detections_a=6000)
    out = _run_report(tmp_path)
    assert out.returncode == 0, out.stderr
    assert "DEAD_ONE" in out.stdout
    assert "detected nothing" in out.stdout
    # ALIVE_ONE must not be swept into the same warning line.
    warn_line = next(l for l in out.stdout.splitlines() if "detected nothing" in l)
    assert "ALIVE_ONE" not in warn_line


def test_cli_does_not_flag_a_stage_that_was_never_called(tmp_path):
    """`stage_calls.a == 0` (stage never invoked) must not be reported as
    "detected nothing" -- that phrase means "ran and saw nothing", not
    "did not run". The report script's own guard is
    `dig(r, "router.stage_calls.a")` (truthy check), which this pins."""
    _write_run(tmp_path, "NEVER_RAN", stage_calls_a=0, stage_detections_a=0)
    out = _run_report(tmp_path)
    assert out.returncode == 0, out.stderr
    assert "detected nothing" not in out.stdout


def test_cli_warns_when_a_run_ended_before_its_duration(tmp_path):
    """A truncated run must say so in the report, not only in metrics.json.

    With `DURATION=900` against a short clip every row of the matrix is a
    truncated run whose sustained FPS is not comparable with a full-length
    one. The flag existed in the file and no reader of the table saw it.
    """
    _write_run(tmp_path, "SHORTCLIP", stage_calls_a=100, stage_detections_a=250,
               ended_early=True, short_by_s=840.0)
    out = _run_report(tmp_path)
    assert out.returncode == 0, out.stderr
    assert "ended before its duration" in out.stdout
    assert "SHORTCLIP" in out.stdout and "840.0" in out.stdout


def test_cli_silent_about_early_end_for_a_full_length_run(tmp_path):
    _write_run(tmp_path, "FULL", stage_calls_a=100, stage_detections_a=250,
               ended_early=False, short_by_s=0.0)
    out = _run_report(tmp_path)
    assert "ended before its duration" not in out.stdout


def test_invalid_warning_no_longer_names_only_throttling(tmp_path):
    """`result_valid=false` now has three causes; the line must not claim one."""
    _write_run(tmp_path, "BAD", stage_calls_a=100, stage_detections_a=250,
               result_valid=False)
    out = _run_report(tmp_path)
    assert "marked invalid" in out.stdout
    assert "throttled/under-voltage during" not in out.stdout
