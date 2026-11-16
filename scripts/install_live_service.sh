#!/usr/bin/env bash
# Laptop side: install (or remove) the live system as a systemd service on the board. Asks for the board's sudo password (ssh -t).
#   bash scripts/install_live_service.sh            # install + enable + start
#   bash scripts/install_live_service.sh remove     # stop + disable + delete the unit
# The auto-capture service and the live service both need /dev/video0, so install turns auto-capture off (.no_autocapture);
# `make start-capture` / `make collect-day2` turn it back on after you stop this service.
set -euo pipefail
HOST="${HOST:-orangepi-lan}"
cd "$(dirname "$0")/.."
if [ "${1:-install}" = remove ]; then
  ssh -t "$HOST" 'sudo systemctl disable --now labsafe-live.service; sudo rm -f /etc/systemd/system/labsafe-live.service; sudo systemctl daemon-reload'
  exit 0
fi
rsync -aq --exclude __pycache__ src scripts configs deploy "$HOST":AI_Challenge/
rsync -aq runs/export/lab_v6_split_int8_640.rknn runs/export/lab_v6_split_int8_640.rknn.card.json \
  runs/export/stageC_fire_y26n_best_int8_416.rknn runs/export/stageC_fire_y26n_best_int8_416.rknn.card.json "$HOST":AI_Challenge/runs/export/
ssh -t "$HOST" 'cd ~/AI_Challenge && touch .no_autocapture &&
  sudo install -m 644 deploy/labsafe-live.service /etc/systemd/system/labsafe-live.service &&
  systemd-analyze verify /etc/systemd/system/labsafe-live.service &&
  sudo systemctl daemon-reload && sudo systemctl stop labsafe-capture.service ; sudo systemctl enable labsafe-live.service && sudo systemctl reset-failed labsafe-live.service; sudo systemctl restart labsafe-live.service &&
  sleep 8 && systemctl --no-pager status labsafe-live.service | head -12'
