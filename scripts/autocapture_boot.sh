#!/usr/bin/env bash
# Started by labsafe-capture.service at boot: unattended RAW capture, one new session per boot
# (counter, because the board clock is unreliable). Needs no network.
#
# OFF SWITCH (no root needed):  touch ~/AI_Challenge/.no_autocapture   -> boots without recording
#                               rm    ~/AI_Challenge/.no_autocapture   -> records again
# Tuning:  CAPTURE_HOURS (default 12)  CAPTURE_EVERY seconds (default 4)  CAPTURE_MIN_DIFF (default 3)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -e .no_autocapture ]; then
  echo "autocapture disabled by .no_autocapture"; exit 0
fi
# Session name = highest existing bootNNN + 1, read from the directories themselves. (A separate counter
# file was fragile: a power cut right after writing it leaves a file of NUL bytes, the next boot then
# picks an existing name and refuses to run -- seen on the board.)
sd=data/raw/lab_sessions
mkdir -p "$sd"
last=$(ls "$sd" 2>/dev/null | sed -n 's/^boot0*\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)
session="$(printf 'boot%03d' $(( ${last:-0} + 1 )))"
exec python3 scripts/capture_interval.py --session "$session" \
  --every "${CAPTURE_EVERY:-4}" --hours "${CAPTURE_HOURS:-12}" --min-diff "${CAPTURE_MIN_DIFF:-3}" \
  --note "autostart at boot (${session}); natural lab scene, not annotated"
