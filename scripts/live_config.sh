#!/usr/bin/env bash
# Laptop side: which config does the live service on the board run, and switch it.
#   bash scripts/live_config.sh            # show the running config
#   bash scripts/live_config.sh demo       # configs/live/demo_lab_board.yaml (coat threshold 0.5, coat memory 3 s)
#   bash scripts/live_config.sh standard   # configs/live/dashboard_lab_board.yaml (0.6, 2 s; the measured default)
#   bash scripts/live_config.sh v5         # configs/live/lab_v5_board.yaml (trial: lab_v5 model, coat 0.3, memory 3 s; KNOWN_ISSUES #52)
# Switching restarts the service: asks the board's sudo password once (ssh -t). Needs the unit installed after 23/10 (reads live.env).
set -euo pipefail
HOST="${HOST:-orangepi-lan}"
running() { ssh "$HOST" "pgrep -af 'labsafe.app' | grep -o -- '--config [^ ]*' | head -1" || true; }
case "${1:-show}" in
  show) echo "running: $(running)"; ssh "$HOST" 'cat ~/AI_Challenge/live.env 2>/dev/null || echo "(no live.env: standard)"'; exit 0;;
  demo) cfg=configs/live/demo_lab_board.yaml;;
  standard) cfg=configs/live/dashboard_lab_board.yaml;;
  v5) cfg=configs/live/lab_v5_board.yaml;;
  *) echo "usage: $0 [show|demo|standard|v5]"; exit 2;;
esac
ssh "$HOST" "systemctl cat labsafe-live.service | grep -q LABSAFE_CONFIG" \
  || { echo "the installed service predates live.env: run 'make install-live-service' once, then retry"; exit 1; }
ssh "$HOST" "test -f ~/AI_Challenge/$cfg && echo LABSAFE_CONFIG=$cfg > ~/AI_Challenge/live.env"
ssh -t "$HOST" 'sudo systemctl restart labsafe-live.service'
sleep 8
now=$(running)
echo "running: ${now:-NOT RUNNING}"
[ "$now" = "--config $cfg" ] || { echo "!! the service is not running $cfg (see: ssh $HOST tail -30 AI_Challenge/runs/labsafe-live.log)"; exit 1; }
