"""Regression tests for `scripts/board_sampler.sh`.

Runs on a laptop, no root and no board needed: `board_sampler.sh` only runs
its `main()` body (root check, sysfs discovery, sampling loop) when
*executed* directly -- see the `BASH_SOURCE` guard at the bottom of the
script. Sourcing it here just defines `parse_npu_load` / `csv_field_count` /
`row_matches_header` as plain functions, which this test calls with sample
strings.

Positive control, run once by hand (not part of the suite, since the buggy
code no longer exists in the repo to import): the original implementation
piped through a second `grep -oE '[0-9]+'` that also matched the digit in
the literal label "Core0" itself --

    $ load_line="NPU load:  Core0:  44%, Core1: 12%, Core2: 7%,"
    $ echo "$load_line" | grep -oE 'Core0:[[:space:]]*[0-9]+%' | grep -oE '[0-9]+'
    0
    44

-- two lines instead of one ("0" from the label, "44" from the value). That
embedded newline is exactly what corrupted a real captured CSV
(KNOWN_ISSUES.md #22, see `scripts/board_sampler.sh`'s comment above
`parse_npu_load`). `test_parse_npu_load_returns_one_line_not_two` below is a
direct regression guard for that: it would fail against the old grep-based
code (multi-line output) and passes against the current `sed -nE`
implementation.

`test_row_matches_header_with_old_buggy_ne_logic_would_have_kept_the_bad_row`
is a positive control run *inside* the suite (not by hand) for a second,
distinct bug (danh-gia.md Vong 2, R2-1): the pre-write guard used to compare
`csv_field_count`'s raw output with `[ -ne ]` directly. When a row contains
an embedded newline, `csv_field_count` itself prints multiple lines (awk
prints one NF per input line), `[ -ne ]` on that multi-line string errors
with "integer expression expected", `if` treats that non-zero *error* exit
as "condition is false", and the malformed row falls through to the KEEP
branch -- silently defeating the very guard meant to drop it. That test
recreates the old `[ -ne ]` comparison directly (it no longer exists in the
script to import) and asserts it wrongly KEEPS the corrupted row, which is
exactly the defect `row_matches_header` (checking for an embedded newline
first, before ever comparing field counts) fixes.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "board_sampler.sh"


def _call(fn: str, *args: str) -> str:
    """Source board_sampler.sh (only defines functions, does not run main)
    and call `fn` with the given args, returning raw stdout. Asserts the
    call exits 0 -- use `_call_status` instead for functions whose return
    status (not stdout) is the thing under test.

    Args are passed as real subprocess argv entries (via bash's "$@"), not
    interpolated into a shell string -- so a literal newline or any other
    shell-special byte inside an arg survives intact instead of being
    reinterpreted by the shell.
    """
    cmd = ["bash", "-c", f'source {SCRIPT!s} && {fn} "$@"', "_", *args]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, (
        f"{fn}({args!r}) exited {result.returncode}\nstderr: {result.stderr}"
    )
    return result.stdout


def _call_status(fn: str, *args: str) -> int:
    """Like `_call`, but returns the exit status instead of asserting it is
    0 -- for boolean-decision functions like `row_matches_header`."""
    cmd = ["bash", "-c", f'source {SCRIPT!s} && {fn} "$@"', "_", *args]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=10,
    )
    return result.returncode


def test_parse_npu_load_zero_percent_all_cores():
    out = _call("parse_npu_load", "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,")
    assert out == "0,0,0\n"


def test_parse_npu_load_real_sample_line():
    out = _call("parse_npu_load", "NPU load:  Core0: 44%, Core1:  0%, Core2:  7%,")
    assert out == "44,0,7\n"


def test_parse_npu_load_returns_one_line_not_two():
    """Regression guard for the grep-double-match bug (KNOWN_ISSUES.md #22):
    the label "Core0" itself contains a digit, so a second unscoped
    `grep -oE '[0-9]+'` matched both the label's digit and the value,
    printing two lines. Exactly one line must come out."""
    out = _call("parse_npu_load", "NPU load:  Core0: 44%, Core1: 12%, Core2: 7%,")
    lines = out.splitlines()
    assert len(lines) == 1, f"expected exactly one line, got: {lines!r}"
    assert lines[0] == "44,12,7"


def test_parse_npu_load_missing_cores_fall_back_to_na():
    out = _call("parse_npu_load", "some unrelated debugfs content")
    assert out == "NA,NA,NA\n"


def test_parse_npu_load_empty_line_falls_back_to_na():
    out = _call("parse_npu_load", "")
    assert out == "NA,NA,NA\n"


def test_csv_field_count_matches_header_for_a_well_formed_row():
    header = "t,temp_soc_c,freq_policy0_mhz,npu_core0_pct,npu_core1_pct,npu_core2_pct,mem_available_mb,loadavg1"
    row = "1712000000.123,45.0,1800.0,44,0,7,3200,0.42"
    header_n = int(_call("csv_field_count", header).strip())
    row_n = int(_call("csv_field_count", row).strip())
    assert header_n == row_n == 8


# -- row_matches_header: the actual keep/drop decision (danh-gia.md R2-1) --
#
# These tests call `row_matches_header` itself -- the exact function the
# sampling loop uses to decide whether to write a row -- not just the lower
# level `csv_field_count` helper. A test that only inspects
# `csv_field_count`'s raw output (as the previous round's test did) can pass
# even when the surrounding `[ -ne ]` comparison is broken; only asserting
# on `row_matches_header`'s return status actually exercises the decision.

HEADER = "t,npu_core0_pct,npu_core1_pct,npu_core2_pct"


def test_row_matches_header_true_for_a_well_formed_row():
    row = "1712000000.123,44,0,7"
    assert _call_status("row_matches_header", HEADER, row) == 0


def test_row_matches_header_false_for_a_row_with_wrong_field_count():
    row = "1712000000.123,44,0"  # missing the third npu core column
    assert _call_status("row_matches_header", HEADER, row) == 1


def test_row_matches_header_false_for_a_row_with_an_embedded_newline():
    row = "1712000000.123,0\n44,0,7"
    assert _call_status("row_matches_header", HEADER, row) == 1


def test_row_matches_header_false_for_a_row_built_from_the_old_buggy_parse_output():
    """Simulates exactly what the old double-grep parse_npu_load produced
    for one field: npu_vals = $'0\\n44' (KNOWN_ISSUES.md #22), spliced into
    a row the way board_sampler.sh's sampling loop does (`row="${row},${npu_vals}"`)."""
    npu_vals = "0\n44,0,7"  # old bug's multi-line output, one field's worth
    row = f"1712000000.123,{npu_vals}"
    assert _call_status("row_matches_header", HEADER, row) == 1


def test_row_matches_header_with_old_buggy_ne_logic_would_have_kept_the_bad_row():
    """Positive control: proves the *previous round's* guard -- comparing
    `csv_field_count`'s raw output with `[ "$row_fields" -ne "$header_fields" ]`
    -- is exactly the bug `row_matches_header` fixes. Recreates that old
    comparison directly (it no longer exists in the script to import) and
    shows it wrongly KEEPS a newline-corrupted row, unlike
    `row_matches_header` (tested above), which correctly drops it."""
    row = "1712000000.123,0\n44,0,7"
    script = f'''
    source {SCRIPT!s}
    row_fields="$(csv_field_count "$1")"
    header_fields="$(csv_field_count "$2")"
    if [ "$row_fields" -ne "$header_fields" ] 2>/dev/null; then
      echo DROPPED
    else
      echo KEPT
    fi
    '''
    result = subprocess.run(
        ["bash", "-c", script, "_", row, HEADER],
        capture_output=True, text=True, timeout=10,
    )
    assert "KEPT" in result.stdout, (
        f"expected the old buggy `[ -ne ]` comparison to WRONGLY keep the "
        f"newline-corrupted row (that is precisely the bug row_matches_header "
        f"fixes); got stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_script_has_bash_source_guard_so_it_is_safe_to_source_for_testing():
    text = SCRIPT.read_text()
    assert 'BASH_SOURCE[0]' in text
    assert "main \"$@\"" in text


# -- main()'s cleanup trap (danh-gia.md R2-2) -------------------------------
#
# `main` requires real root (`id -u` == 0) before it does anything else, so
# it cannot be exercised end-to-end as a genuine non-root user. To test the
# actual bug -- the `trap cleanup EXIT` in `main` referencing `OUT`/`META`/
# `_owner`, which used to be declared `local` to `main` and were therefore
# out of scope (raising `_owner: unbound variable` under `set -u`) by the
# time the trap fires after `main` returns -- this test fakes a root
# environment by prepending a fake `id` (reports uid 0) and a fake `chown`
# (just logs its arguments instead of touching real ownership) onto PATH,
# then runs the real script end to end with a 1-second duration.
#
# Limitation: this does not use real root privileges or a real chown -- it
# cannot prove the *file ownership* changes on disk, only that `cleanup`
# reaches the `chown` call at all, with the right arguments, instead of
# aborting on an unbound variable first. Proving the on-disk ownership
# change is deferred to a real board run (danh-gia.md board condition B4);
# this test's job is to catch the "cleanup crashes before it can even try"
# regression, which is a laptop-testable bug independent of real root.
def test_main_cleanup_trap_runs_chown_without_unbound_variable_error(tmp_path):
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()

    fake_id = fake_bin / "id"
    fake_id.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-u" ]; then echo 0; exit 0; fi\n'
        'exec /usr/bin/id "$@"\n'
    )
    fake_id.chmod(0o755)

    chown_log = tmp_path / "chown.log"
    fake_chown = fake_bin / "chown"
    fake_chown.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "chown called with: $*" >> {chown_log!s}\n'
        "exit 0\n"
    )
    fake_chown.chmod(0o755)

    out_csv = tmp_path / "out.csv"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["SUDO_USER"] = "testuser"

    result = subprocess.run(
        ["bash", str(SCRIPT), str(out_csv), "1", "1"],
        capture_output=True, text=True, timeout=15, env=env,
    )

    assert result.returncode == 0, (
        f"expected a clean exit, got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    assert "unbound variable" not in result.stderr, (
        f"cleanup crashed on an out-of-scope local variable before it could "
        f"chown -- this is exactly the R2-2 regression: stderr={result.stderr!r}"
    )
    assert chown_log.exists(), (
        "cleanup never reached the chown call at all -- expected a call "
        "logged by the fake chown"
    )
    logged = chown_log.read_text()
    assert "testuser:testuser" in logged
    assert str(out_csv) in logged
    assert f"{out_csv}.meta.txt" in logged
