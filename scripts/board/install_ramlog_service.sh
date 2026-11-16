#!/usr/bin/env bash
# Laptop side: install the RAM logger as a boot-time service on the board (asks the board's sudo password). `remove` uninstalls it.
set -euo pipefail
HOST="${HOST:-orangepi-lan}"
cd "$(dirname "$0")/../.."
if [ "${1:-install}" = remove ]; then
  ssh -t "$HOST" 'sudo systemctl disable --now labsafe-ramlog.service; sudo rm -f /etc/systemd/system/labsafe-ramlog.service; sudo systemctl daemon-reload'; exit 0
fi
rsync -aq scripts deploy "$HOST":AI_Challenge/
# the manual logger started earlier would write a second CSV: stop it (it is owned by the board user, no sudo needed)
ssh "$HOST" 'test -f /tmp/ramlog.pid && kill $(cat /tmp/ramlog.pid) 2>/dev/null; true'
ssh -t "$HOST" 'cd ~/AI_Challenge && sudo install -m 644 deploy/labsafe-ramlog.service /etc/systemd/system/ &&
  systemd-analyze verify /etc/systemd/system/labsafe-ramlog.service ; sudo systemctl daemon-reload &&
  sudo systemctl enable labsafe-ramlog.service && sudo systemctl restart labsafe-ramlog.service && sleep 4 &&
  systemctl --no-pager status labsafe-ramlog.service | head -8; ls -l runs/ramlog'
