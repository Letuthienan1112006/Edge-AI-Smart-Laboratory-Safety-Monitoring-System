#!/usr/bin/env bash
# Started by labsafe-ramlog.service at boot: one CSV per boot, numbered from the files already there (the clock is wrong after a power cut).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/ramlog
n=$(ls runs/ramlog 2>/dev/null | sed -n 's/^ram-boot0*\([0-9][0-9]*\)\.csv$/\1/p' | sort -n | tail -1)
exec python3 scripts/ram_logger.py --interval 60 --smaps-every 900 --out "runs/ramlog/$(printf 'ram-boot%03d.csv' $(( ${n:-0} + 1 )))"
