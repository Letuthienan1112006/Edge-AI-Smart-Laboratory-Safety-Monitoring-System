#!/usr/bin/env bash
# Laptop side: show the board camera live in the browser to check the placement/angle.
#   make camera            (or: bash scripts/board/camera_view.sh [host])
# Opens an SSH tunnel (the stream is never exposed on the network), frees the camera if the
# auto-capture service holds it, starts the viewer on the board and opens http://localhost:8081.
# Ctrl+C stops the viewer and releases the camera.
set -euo pipefail
HOST="${1:-orangepi-lan}"; PORT="${PORT:-8081}"
. "$(dirname "$0")/_board_env.sh"
echo "Connecting to $HOST ..."
ssh -o ConnectTimeout=8 "$HOST" true || { echo "Cannot reach $HOST: check the LAN cable / power."; exit 1; }
bash "$(dirname "$0")/sync_board_clock.sh" "$HOST"
ssh "$HOST" "echo '$BOARD_SUDO_PASS' | sudo -S systemctl stop labsafe-capture.service 2>/dev/null; true" >/dev/null 2>&1 || true
( sleep 4; command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || echo "Open http://localhost:$PORT in a browser" ) &
ssh -t -L "$PORT:127.0.0.1:$PORT" "$HOST" "cd ~/AI_Challenge && python3 scripts/board/camera_view.py --port $PORT" || true
echo
echo "Camera released. If the framing is OK, start capturing with:  make start-capture"
