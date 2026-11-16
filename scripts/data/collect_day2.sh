#!/usr/bin/env bash
# Laptop side, collection day 2: guided 30 FPS videos on the board (configs/collection_plan_L2.yaml), then the
# all-day unattended capture. Resumable: Ctrl+C any time, run again, finished scenes are kept.
#   make collect-day2            (or: bash scripts/data/collect_day2.sh [host])
set -euo pipefail
HOST="${1:-orangepi-lan}"
. "$(dirname "$0")/../board/_board_env.sh"
cd "$(dirname "$0")/../.."
ssh -o ConnectTimeout=8 "$HOST" true || { echo "Cannot reach $HOST: check the LAN cable / power."; exit 1; }
bash "$(dirname "$0")/../board/sync_board_clock.sh" "$HOST"
rsync -a --exclude __pycache__ scripts configs src "$HOST":AI_Challenge/
# the camera must be free: stop the capture service, keep it off until we are done
ssh "$HOST" "touch ~/AI_Challenge/.no_autocapture; echo '$BOARD_SUDO_PASS' | sudo -S systemctl stop labsafe-capture.service 2>/dev/null; true" >/dev/null 2>&1 || true
echo "Tip: run 'make camera' first to check the angle (same angle as before, or redraw configs/zones_lab.yaml)."
ssh -t "$HOST" "cd ~/AI_Challenge && python3 scripts/data/collect_guided.py --plan configs/collection_plan_L2.yaml" || true
echo
read -r -p "Start the all-day unattended capture now? [y/N] " yn
if [[ "$yn" =~ ^[yY]$ ]]; then bash scripts/board/capture_ctl.sh start; else echo "Not started. Later: make start-capture"; fi
