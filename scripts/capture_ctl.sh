#!/usr/bin/env bash
# Laptop side: start / stop / check the board's unattended capture.
#   make start-capture    make capture-status    make stop-capture      (HOST=orangepi to override)
# `start` runs the capture as a board service, so it KEEPS RUNNING after you unplug the cable or close
# the terminal; it stops by itself after 12 h, on `stop`, or when power is cut. Needs no network.
set -euo pipefail
cmd="${1:-status}"; HOST="${2:-${HOST:-orangepi-lan}}"
. "$(dirname "$0")/_board_env.sh"
ssh -o ConnectTimeout=8 "$HOST" true 2>/dev/null || { echo "Cannot reach $HOST: check the LAN cable / power."; exit 1; }
sudo_b() { ssh "$HOST" "echo '$BOARD_SUDO_PASS' | sudo -S $* 2>/dev/null"; }
status() {
  ssh "$HOST" 'cd ~/AI_Challenge/data/raw/lab_sessions 2>/dev/null || { echo "no sessions yet"; exit 0; }
    echo "service: $(systemctl is-active labsafe-capture.service)   board time: $(date +%H:%M:%S)"
    for d in */; do d=${d%/}; n=$(ls $d/frames 2>/dev/null | wc -l)
      last=$(ls -t $d/frames 2>/dev/null | head -1); age="-"
      [ -n "$last" ] && age="$(( $(date +%s) - $(stat -c %Y $d/frames/$last) )) s ago"
      echo "  $d: $n frames, last frame $age"; done'
}
case "$cmd" in
  start)
    bash "$(dirname "$0")/sync_board_clock.sh" "$HOST"
    # "[c]amera_view.py": a plain `pkill -f camera_view.py` also matches the `bash -c '...'` that ssh runs this line
    # in, kills it, and the rm below never ran -> after make dashboard / collect-day2 (which set .no_autocapture)
    # the service started but recorded nothing (KNOWN_ISSUES #36).
    ssh "$HOST" 'pkill -f "[c]amera_view.py" 2>/dev/null; rm -f ~/AI_Challenge/.no_autocapture; true' >/dev/null 2>&1 || true
    sudo_b systemctl restart labsafe-capture.service
    sleep 12; status
    ssh "$HOST" 'systemctl is-active labsafe-capture.service' | grep -q '^active$' \
      || { echo "!! capture did not start; log:"; ssh "$HOST" 'tail -5 ~/AI_Challenge/data/raw/autocapture.log'; exit 1; }
    echo; echo "Capturing (1 frame / 4 s, stops after 12 h). You can unplug the LAN cable now; do NOT cut the power."
    echo "Later: plug the cable back in -> make capture-status -> make fetch-sessions";;
  stop)  sudo_b systemctl stop labsafe-capture.service; sleep 2; status;;
  status) status;;
  *) echo "usage: capture_ctl.sh start|stop|status [host]"; exit 2;;
esac
