#!/usr/bin/env bash
# Samples thermal/frequency/NPU-load/memory telemetry on the board while a
# live run is in progress -- the RK3588 side of the picture that
# src/labsafe/bench/sysmon.py cannot provide (it has no throttle verdict on
# this SoC yet; see docs/KNOWN_ISSUES.md and plan §2.7, follow-up not fixed
# here). This script deliberately does NOT touch labsafe's own code path:
# it is a separate root-only sampler, run alongside `labsafe.app --bench`,
# never instead of it.
#
# Usage:
#   sudo bash scripts/board/board_sampler.sh <out.csv> <duration_s> [interval_s=1]
#
# Discovers what the board actually has (thermal zones, cpufreq policies,
# NPU devfreq node) at start, rather than hardcoding RK3588-specific paths
# that may differ across kernel/board revisions.
#
# Testable without root or a board: `parse_npu_load`, `csv_field_count` and
# `row_matches_header` below are plain functions with no side effects. This
# script can be `source`d (its body only runs when *executed* directly --
# see the `BASH_SOURCE` guard at the bottom) so a laptop test suite can call
# those functions directly with sample strings. `main`'s exit-time `chown`
# can also be exercised end to end without real root, by faking `id`/`chown`
# on PATH -- see tests/test_board_sampler.py.

set -u  # deliberately NOT `set -e`: one unreadable sysfs file must not
        # kill a 15-minute sampling run; a bad read becomes "NA" instead.

# Force the C locale for the whole script. Verified for real on this
# machine: with LC_NUMERIC set to a locale that uses a comma decimal
# separator (e.g. vi_VN), `awk 'BEGIN{printf "%.1f", 27800/1000}'` prints
# "27,8", not "27.8" -- silently corrupting every temperature/frequency
# column in the CSV for any downstream reader (Python's csv/pandas both
# expect '.'). This does not error, does not warn, and would only show up
# as "some column is unparseable" days later when someone runs T6. Whether
# the board happens to share this shell's locale is exactly the kind of
# thing not worth trusting.
export LC_ALL=C

# -- parse_npu_load: pure function, no I/O, no side effects ----------------
#
# rknpu/load looks like: "NPU load:  Core0:  12%, Core1:  0%, Core2:  0%,"
# (exact spacing varies by driver version).
#
# BUG FIXED HERE (docs/KNOWN_ISSUES.md): the original implementation piped
# through `grep -oE 'Core0:[[:space:]]*[0-9]+%'` and then a SECOND
# `grep -oE '[0-9]+'` to strip the label. That second grep matches ANY
# run of digits in what survived the first match -- including the digit in
# the literal label "Core0" itself -- so it printed two numbers ("0" and
# "44") on two separate lines instead of one. That embedded newline then
# propagated into $row and corrupted the CSV: a single logical row was
# split across several physical lines, exactly what was found in a real
# captured file (docs/KNOWN_ISSUES.md, board_sampler.sh entry). `sed -nE
# 's/.../\1/p'` captures the digits directly with one command and one
# match, so there is nothing left over to re-match.
#
# Prints "core0,core1,core2" (each "NA" if not found) on exactly one line.
parse_npu_load() {
  local load_line="$1"
  local c0="NA" c1="NA" c2="NA"
  if [ -n "$load_line" ]; then
    local c0v c1v c2v
    c0v="$(echo "$load_line" | sed -nE 's/.*Core0:[[:space:]]*([0-9]+)%.*/\1/p')"
    c1v="$(echo "$load_line" | sed -nE 's/.*Core1:[[:space:]]*([0-9]+)%.*/\1/p')"
    c2v="$(echo "$load_line" | sed -nE 's/.*Core2:[[:space:]]*([0-9]+)%.*/\1/p')"
    [ -n "$c0v" ] && c0="$c0v"
    [ -n "$c1v" ] && c1="$c1v"
    [ -n "$c2v" ] && c2="$c2v"
  fi
  echo "${c0},${c1},${c2}"
}

# -- csv_field_count: number of comma-separated fields in one line ---------
#
# WARNING: `awk` prints one NF value PER INPUT LINE. If `$1` itself contains
# an embedded newline (exactly the corruption a bad parse can produce -- see
# parse_npu_load above), this prints *multiple* lines, e.g. $'2\n3'. Comparing
# that multi-line string with `[ "$x" -ne "$y" ]` raises "integer expression
# expected" (exit 2), which `if` treats as false -- so callers must not
# compare this function's raw output with `-eq`/`-ne` directly. Use
# `row_matches_header` below instead of doing that comparison inline.
csv_field_count() {
  awk -F',' '{print NF}' <<<"$1"
}

# -- row_matches_header: the ONLY place that decides keep vs. drop ---------
#
# A row must be dropped if either:
#   (a) it contains an embedded newline -- this is precisely the corruption
#       the old double-`grep` bug in parse_npu_load produced (docs/KNOWN_ISSUES.md
#       #22): one logical row split across several physical lines. Checking
#       this FIRST, before any field-count comparison, is what makes this
#       function safe -- see the pitfall explained below.
#   (b) its comma-separated field count differs from the header's.
#
# This function exists because comparing csv_field_count's output with
# `[ -ne ]` inline (the previous approach) is NOT safe: for a row with an
# embedded newline, `csv_field_count` itself prints multiple lines (awk
# prints one NF per input line), so `[ "$row_fields" -ne "$header_fields" ]`
# gets a multi-line left-hand side, `[ ]` errors with "integer expression
# expected", `if` reads that non-zero *error* exit status as "condition is
# false", and execution falls into the KEEP branch -- silently writing the
# very corrupted row this whole check exists to catch. Rejecting embedded
# newlines up front, before ever calling `[ -ne ]` on anything derived from
# the row, sidesteps that trap entirely: by the time field counts are
# compared, `$row` is guaranteed to be a single line, so csv_field_count's
# output is guaranteed to be a single number.
#
# Returns 0 (keep) or 1 (drop). No output, no side effects.
row_matches_header() {
  local header="$1" row="$2"
  case "$row" in
    *$'\n'*) return 1 ;;
  esac
  [ "$(csv_field_count "$row")" = "$(csv_field_count "$header")" ]
}

main() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "board_sampler.sh must run as root (NPU load lives under" >&2
    echo "/sys/kernel/debug, which needs root to read/mount). Re-run with:" >&2
    echo "  sudo bash scripts/board/board_sampler.sh <out.csv> <duration_s> [interval_s]" >&2
    exit 1
  fi

  # OUT, META and _owner are deliberately NOT `local`: `cleanup` below is a
  # plain (global) bash function -- bash has no lexical scoping for nested
  # function *definitions*, only for variables -- and it is invoked by
  # `trap cleanup EXIT` after `main` has already `return`ed (the normal
  # completion path: the sampling loop ends, `main` returns, the script
  # falls off the end, and only THEN does the process actually exit and run
  # the EXIT trap). If OUT/META/_owner were `local` to `main`, they would
  # already be unset by the time `cleanup` runs, and under `set -u` that is
  # not silently-empty -- it is a hard `_owner: unbound variable` error,
  # `cleanup` aborts before reaching `chown`, and the output files are left
  # owned by root. Verified: this is not hypothetical, it reproduced with
  # exactly this pattern (see tests/test_board_sampler.py). Keeping these
  # three as ordinary (script-global) variables means they are still set
  # when the EXIT trap fires, on every exit path (normal completion,
  # `exit 1`, or an interrupting signal).
  local DURATION_S INTERVAL_S
  OUT="${1:?usage: board_sampler.sh <out.csv> <duration_s> [interval_s]}"
  DURATION_S="${2:?usage: board_sampler.sh <out.csv> <duration_s> [interval_s]}"
  INTERVAL_S="${3:-1}"
  META="${OUT}.meta.txt"

  # chown output back to the invoking user on exit, whatever happens --
  # sudo-created files owned by root are a routine annoyance otherwise.
  _owner="${SUDO_USER:-root}"
  cleanup() {
    chown "${_owner}:${_owner}" "$OUT" "$META" 2>/dev/null || true
  }
  trap cleanup EXIT

  # -- discovery -----------------------------------------------------------

  local THERMAL_ZONES=()   # "zoneN:type"
  local zdir zn ztype
  for zdir in /sys/class/thermal/thermal_zone*; do
    [ -d "$zdir" ] || continue
    zn="$(basename "$zdir")"
    ztype="$(cat "$zdir/type" 2>/dev/null || echo unknown)"
    THERMAL_ZONES+=("${zn}:${ztype}")
  done

  local CPUFREQ_POLICIES=()  # "policyN"
  local pdir
  for pdir in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -d "$pdir" ] || continue
    CPUFREQ_POLICIES+=("$(basename "$pdir")")
  done

  local NPU_DEVFREQ=""
  local d
  for d in /sys/class/devfreq/*npu*; do
    [ -d "$d" ] && NPU_DEVFREQ="$d" && break
  done

  local NPU_LOAD_PATH="${NPU_LOAD_PATH:-/sys/kernel/debug/rknpu/load}"
  if [ ! -r "$NPU_LOAD_PATH" ]; then
    mount -t debugfs none /sys/kernel/debug 2>/dev/null || true
  fi

  # -- meta file: static facts, written once --------------------------------

  {
    echo "# board_sampler.sh meta -- $(date -Is)"
    echo "uname -r: $(uname -r)"
    echo
    echo "## dmesg | grep -i rknpu"
    dmesg 2>/dev/null | grep -i rknpu || echo "(none)"
    if [ -r /sys/kernel/debug/rknpu/version ]; then
      echo
      echo "## /sys/kernel/debug/rknpu/version"
      cat /sys/kernel/debug/rknpu/version 2>&1
    fi
    local entry tp p f
    for entry in "${THERMAL_ZONES[@]}"; do
      zn="${entry%%:*}"; ztype="${entry##*:}"
      zdir="/sys/class/thermal/${zn}"
      echo
      echo "## ${zn} (type=${ztype})"
      for tp in "$zdir"/trip_point_*_type "$zdir"/trip_point_*_temp; do
        [ -e "$tp" ] || continue
        echo "$(basename "$tp"): $(cat "$tp" 2>&1)"
      done
    done
    for p in "${CPUFREQ_POLICIES[@]}"; do
      pdir="/sys/devices/system/cpu/cpufreq/${p}"
      echo
      echo "## ${p}"
      for f in related_cpus cpuinfo_max_freq scaling_governor; do
        [ -r "${pdir}/${f}" ] && echo "${f}: $(cat "${pdir}/${f}" 2>&1)"
      done
    done
    if [ -n "$NPU_DEVFREQ" ]; then
      echo
      echo "## NPU devfreq: ${NPU_DEVFREQ}"
      for f in max_freq available_frequencies governor; do
        [ -r "${NPU_DEVFREQ}/${f}" ] && echo "${f}: $(cat "${NPU_DEVFREQ}/${f}" 2>&1)"
      done
    else
      echo
      echo "## NPU devfreq: not found (glob /sys/class/devfreq/*npu*)"
    fi
  } > "$META"

  # -- CSV header ------------------------------------------------------------

  local header="t"
  for entry in "${THERMAL_ZONES[@]}"; do
    ztype="${entry##*:}"
    header="${header},temp_${ztype}_c"
  done
  for p in "${CPUFREQ_POLICIES[@]}"; do
    header="${header},freq_${p}_mhz"
  done
  if [ -n "$NPU_DEVFREQ" ]; then
    header="${header},npu_freq_mhz"
  fi
  header="${header},npu_core0_pct,npu_core1_pct,npu_core2_pct,mem_available_mb,loadavg1"
  echo "$header" > "$OUT"
  local header_fields
  header_fields="$(csv_field_count "$header")"

  # -- sampling loop ----------------------------------------------------------

  local end_t
  end_t=$(( $(date +%s) + DURATION_S ))
  local t row raw load_line npu_vals mem la reason
  while [ "$(date +%s)" -lt "$end_t" ]; do
    t="$(date +%s.%3N)"
    row="$t"

    for entry in "${THERMAL_ZONES[@]}"; do
      zn="${entry%%:*}"
      raw="$(cat "/sys/class/thermal/${zn}/temp" 2>/dev/null)"
      if [ -n "$raw" ]; then
        row="${row},$(awk -v v="$raw" 'BEGIN{printf "%.1f", v/1000}')"
      else
        row="${row},NA"
      fi
    done

    for p in "${CPUFREQ_POLICIES[@]}"; do
      raw="$(cat "/sys/devices/system/cpu/cpufreq/${p}/scaling_cur_freq" 2>/dev/null)"
      if [ -n "$raw" ]; then
        row="${row},$(awk -v v="$raw" 'BEGIN{printf "%.1f", v/1000}')"
      else
        row="${row},NA"
      fi
    done

    if [ -n "$NPU_DEVFREQ" ]; then
      raw="$(cat "${NPU_DEVFREQ}/cur_freq" 2>/dev/null)"
      if [ -n "$raw" ]; then
        row="${row},$(awk -v v="$raw" 'BEGIN{printf "%.1f", v/1000000}')"
      else
        row="${row},NA"
      fi
    fi

    load_line="$(cat "$NPU_LOAD_PATH" 2>/dev/null)"
    npu_vals="$(parse_npu_load "$load_line")"
    row="${row},${npu_vals}"

    mem="$(awk '/MemAvailable/{printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null)"
    row="${row},${mem:-NA}"

    la="$(awk '{print $1}' /proc/loadavg 2>/dev/null)"
    row="${row},${la:-NA}"

    # Safety catch, independent of the parse_npu_load fix above: whatever
    # the reason (a future sysfs surprise, a stray embedded newline from
    # some other field), a row must have exactly as many comma-separated
    # fields as the header before it is trusted into the CSV. A mismatched
    # row is exactly what corrupted `runs/board/samples/N1L_sustained_*.csv`
    # -- writing it anyway is worse than dropping one sample, because a
    # silently-broken CSV is discovered only much later by a reader that
    # cannot line up columns to rows. See row_matches_header's own comment
    # for why the decision must go through that one function rather than an
    # inline `[ -ne ]` comparison.
    if row_matches_header "$header" "$row"; then
      echo "$row" >> "$OUT"
    else
      case "$row" in
        *$'\n'*) reason="embedded newline" ;;
        *) reason="$(csv_field_count "$row") fields, expected ${header_fields}" ;;
      esac
      echo "board_sampler.sh: WARNING: dropping malformed row (${reason}): ${row}" >&2
    fi

    sleep "$INTERVAL_S"
  done
}

# Only run `main` when this script is *executed*, not when it is `source`d
# for testing -- lets a test suite load `parse_npu_load`/`csv_field_count`
# in isolation without needing root, a duration, or real sysfs paths.
if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
  main "$@"
fi
