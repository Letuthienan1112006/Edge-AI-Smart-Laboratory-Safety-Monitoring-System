#!/usr/bin/env bash
# Laptop side: run the lab safety system on the board and open its web dashboard here.
#   make dashboard            (or: bash scripts/dashboard.sh [host])
# Copies the code + lab model to the board, frees the camera if the auto-capture service holds it, starts the app
# with configs/live/dashboard_lab_board.yaml, and opens http://localhost:8000 through an SSH tunnel (the page is
# never exposed on the network). Ctrl+C stops the app on the board and releases the camera.
set -euo pipefail
HOST="${1:-orangepi-lan}"; PORT="${PORT:-8000}"
. "$(dirname "$0")/_board_env.sh"
cd "$(dirname "$0")/.."
echo "Connecting to $HOST ..."
ssh -o ConnectTimeout=8 "$HOST" true || { echo "Cannot reach $HOST: check the LAN cable / power."; exit 1; }
bash "$(dirname "$0")/sync_board_clock.sh" "$HOST"
echo "Copying code and models ..."
rsync -a --exclude __pycache__ src scripts configs "$HOST":AI_Challenge/
ssh "$HOST" 'mkdir -p ~/AI_Challenge/runs/export'
rsync -a runs/export/lab_v6_split_int8_640.rknn runs/export/lab_v6_split_int8_640.rknn.card.json \
  runs/export/stageC_fire_y26n_best_int8_416.rknn runs/export/stageC_fire_y26n_best_int8_416.rknn.card.json \
  "$HOST":AI_Challenge/runs/export/
# the capture service owns /dev/video0 for hours after each boot
ssh "$HOST" "touch ~/AI_Challenge/.no_autocapture; echo '$BOARD_SUDO_PASS' | sudo -S systemctl stop labsafe-capture.service 2>/dev/null; true" >/dev/null 2>&1 || true
( sleep 8; command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || echo "Open http://localhost:$PORT in a browser" ) &
echo "Starting the system on the board (Ctrl+C to stop) ..."
ssh -t -L "$PORT:127.0.0.1:$PORT" "$HOST" \
  "cd ~/AI_Challenge && PYTHONPATH=src python3 -m labsafe.app --config configs/live/dashboard_lab_board.yaml --classes configs/classes_lab_v3.yaml --no-display --gpio off" || true
echo
echo "Stopped. Auto-capture stays OFF until you run: make start-capture"
