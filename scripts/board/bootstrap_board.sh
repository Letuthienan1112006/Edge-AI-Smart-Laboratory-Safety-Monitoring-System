#!/usr/bin/env bash
# One-shot setup for a freshly-flashed Orange Pi 5, run ON THE BOARD over SSH.
# Idempotent: safe to re-run if a step failed partway.
#
# What it does NOT do, on purpose:
#   - does not install torch/tensorflow/ultralytics (training runs on the
#     x86 laptop; the board only ever loads already-converted models)
#   - does not touch configs/pipeline.yaml's GPIO pin numbers (those are
#     board-specific -- run `gpioinfo` yourself and edit them, see
#     docs/board-setup.md step 7)
#   - does not flash the OS or partition anything
#
# Usage (on the board, after `git clone` and `cd` into the repo):
#   bash scripts/board/bootstrap_board.sh
set -euo pipefail

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[1;31m!! %s\033[0m\n' "$1" >&2; exit 1; }

[ -f "Makefile" ] && [ -d "src/labsafe" ] || \
  fail "run this from the repo root (Makefile and src/labsafe/ not found here)"

ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || \
  fail "expected aarch64, got '$ARCH'. This script is for the Orange Pi board, not the training laptop. See docs/board-setup.md."

log "1/6  apt packages"
sudo apt update
sudo apt install -y \
    python3-pip python3-venv python3-opencv python3-numpy python3-yaml \
    python3-psutil python3-libgpiod gpiod git build-essential

log "2/6  Python edge dependencies (NCNN / tflite-runtime / RKNN inference / GPIO)"
pip install --break-system-packages -r requirements/edge.txt || {
  printf '\n%s\n' \
    "pip install failed -- this is the step most likely to break on a new" \
    "board image, usually rknn-toolkit-lite2 vs the board's librknnrt.so." \
    "See docs/KNOWN_ISSUES.md and docs/rknn-pipeline.md section 0 before retrying."
  exit 1
}

log "3/6  data/runs directories (kept out of git, needed for the pipeline to write to)"
mkdir -p data/raw data/interim data/processed runs

log "4/6  hardware identity check"
python3 -m labsafe.optimize.arm_probe || fail "arm_probe crashed -- report the traceback above"

log "5/6  big.LITTLE topology"
python3 -m labsafe.optimize.affinity || fail "affinity crashed -- report the traceback above"

log "6/6  NPU / accelerator device nodes"
if ls /dev/rknpu* /dev/dri/renderD* 2>/dev/null | grep -q .; then
  echo "found: $(ls /dev/rknpu* /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')"
else
  printf '\n%s\n' \
    "WARNING: no /dev/rknpu* or /dev/dri/renderD* found. The NPU driver may" \
    "not be loaded. Check: dmesg | grep -i rknpu" \
    "You can still continue with CPU-only backends (ncnn/tflite), but RKNN" \
    "inference (scripts/export/validate_conversion.py --candidate-backend rknn," \
    "and any N1/N2/N5 bench config) will fail until this is resolved."
fi

cat <<'EOF'

========================================================================
Bootstrap done. Next:
  1. python3 -m labsafe.optimize.roofline
       -- confirms compute vs memory bound on THIS board (not a guess)
  2. gpioinfo
       -- find the real line offsets for your buzzer/LED wiring, then
          edit configs/pipeline.yaml's `alerts:` block (see
          docs/board-setup.md step 7 -- Pi 4 pin numbers do NOT apply here)
  3. python3 -m labsafe.app --source 0 --no-display
       -- first live run, stub backend, confirms camera -> pipeline works
          end to end on real hardware before any trained model exists
========================================================================
EOF
