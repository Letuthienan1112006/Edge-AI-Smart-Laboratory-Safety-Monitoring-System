#!/usr/bin/env bash
# Laptop side: ONE command for the final-round demo. Waits for the board to boot, syncs its clock, copies the current code, restarts the
# live service, waits for frames, prints the preflight, opens the SSH tunnel and the browser in presentation mode.
#   make demo-day                  (or: bash scripts/demo_day.sh [--net] [--no-deploy] [--no-open])
#     --net        also give the board internet through this laptop (Zalo messages): asks the laptop's sudo + the board's password
#     --no-deploy  skip the file copy + service restart (board already has this code; use after a plain power-on)
#     --no-open    do not open the browser
# The board's sudo password: secrets/board.env (BOARD_SUDO_PASS=...). Without it you are asked for it by ssh.
# After a plain power-on the live service starts BY ITSELF (enabled, retries every 10 s while the camera is missing): `make demo-open` is enough.
set -uo pipefail
HOST="${HOST:-orangepi-lan}"; PORT="${PORT:-8000}"; WAIT_S="${WAIT_S:-180}"
cd "$(dirname "$0")/.."
. scripts/_board_env.sh
NET=0; DEPLOY=1; OPEN=1
for a in "$@"; do case "$a" in --net) NET=1;; --no-deploy) DEPLOY=0;; --no-open) OPEN=0;; *) echo "unknown option $a"; exit 2;; esac; done
step() { printf '\n== %s\n' "$*"; }
ok()   { printf '  OK  %s\n' "$*"; }
bad()  { printf '  !!  %s\n' "$*"; }

step "1/6 board reachable (power it on now; it needs about a minute to boot)"
t0=$(date +%s)
until ssh -o ConnectTimeout=4 -o BatchMode=yes "$HOST" true 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -ge "$WAIT_S" ] && { bad "board not reachable after ${WAIT_S}s: LAN cable between board and laptop, board power (5 V / 4 A), then run again"; exit 1; }
  printf '.'; sleep 3
done
ok "reachable ($HOST) after $(( $(date +%s) - t0 )) s"

step "2/6 clock (the board has no RTC battery: it boots with an old date; Zalo messages and photos use it)"
bash scripts/sync_board_clock.sh "$HOST" | sed 's/^/      /'
skew=$(( $(ssh "$HOST" date +%s) - $(date +%s) )); [ "${skew#-}" -le 5 ] && ok "clock within ${skew} s of the laptop" || bad "board clock off by ${skew} s (set BOARD_SUDO_PASS in secrets/board.env or run: ssh -t $HOST 'sudo date -s @$(date +%s)')"

restart() {
  if [ -n "$BOARD_SUDO_PASS" ]; then ssh "$HOST" "echo '$BOARD_SUDO_PASS' | sudo -S systemctl restart labsafe-live 2>/dev/null"
  else ssh -t "$HOST" 'sudo systemctl restart labsafe-live'; fi
}
if [ "$DEPLOY" = 1 ]; then
  step "3/6 copy the code and models, restart the service"
  rsync -a --exclude __pycache__ scripts src configs deploy "$HOST":AI_Challenge/ && ok "src, configs, scripts, deploy copied"
  ssh "$HOST" 'mkdir -p ~/AI_Challenge/runs/export'
  rsync -a runs/export/lab_v6_split_int8_640.rknn runs/export/lab_v6_split_int8_640.rknn.card.json "$HOST":AI_Challenge/runs/export/ && ok "model copied/up to date"
  if [ -n "${DEMO_VIDEO:-}" ]; then
    [ -f "$DEMO_VIDEO" ] || { bad "video not found: $DEMO_VIDEO"; exit 1; }
    rsync -a "$DEMO_VIDEO" "$HOST":AI_Challenge/data/demo_lab.mp4 && ok "video demo: $DEMO_VIDEO -> board data/demo_lab.mp4"
    ssh "$HOST" 'echo LABSAFE_CONFIG=configs/live/video_demo_board.yaml > ~/AI_Challenge/live.env' && ok "service set to the VIDEO demo profile"
  else
    ssh "$HOST" 'grep -q video_demo ~/AI_Challenge/live.env 2>/dev/null && echo LABSAFE_CONFIG=configs/live/dashboard_lab_board.yaml > ~/AI_Challenge/live.env && echo "  OK  back to the camera profile (was the video demo)"; true'
  fi
  restart && ok "labsafe-live restarted" || bad "could not restart the service (password?)"
else
  step "3/6 deploy skipped (--no-deploy)"
fi

step "4/6 waiting for live frames (camera plugged? the service retries every 10 s)"
good=0
for i in $(seq 1 45); do
  s=$(ssh -o ConnectTimeout=4 "$HOST" 'curl -s -m 3 http://127.0.0.1:8000/status' 2>/dev/null)
  if [ -n "$s" ] && python3 - "$s" <<'PY'
import json, sys
j = json.loads(sys.argv[1]); y = j.get("sys") or {}
ok = j.get("fps", 0) >= 10 and j.get("frame_age_s", 99) < 2
print(f"  {'OK ' if ok else '.. '} {j.get('fps')} FPS, frame age {j.get('frame_age_s')} s, {j.get('summary', {}).get('persons', 0)} person(s); CPU {y.get('cpu_pct')}%  NPU {y.get('npu_pct')}% ({y.get('npu_src')})  {y.get('temp_c')} C")
sys.exit(0 if ok else 1)
PY
  then good=1; break; fi
  sleep 2
done
[ "$good" = 1 ] || bad "no live frames: plug the camera (/dev/video0), wait 10 s, run: bash scripts/demo_day.sh --no-deploy"

if [ "$NET" = 1 ]; then
  step "5/6 internet for the board (Zalo)"
  bash scripts/board_internet.sh on
else
  step "5/6 internet for the board skipped (add --net if Zalo messages are wanted)"
fi

step "6/6 preflight, tunnel, browser"
bash scripts/lab_preflight.sh 2>&1 | grep -E "^==|!!|READY|item" | sed 's/^/  /'
if curl -s -o /dev/null -m 2 "http://localhost:$PORT/status"; then ok "tunnel already open"
else
  (setsid nohup ssh -N -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:8000" "$HOST" >/dev/null 2>&1 </dev/null & echo $! > /tmp/labsafe-tunnel.pid)
  sleep 2; curl -s -o /dev/null -m 3 "http://localhost:$PORT/status" && ok "tunnel open (close: kill \$(cat /tmp/labsafe-tunnel.pid))" || bad "tunnel did not open"
fi
[ "$OPEN" = 1 ] && command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT/#present" >/dev/null 2>&1
[ "$good" = 1 ] && printf '\nREADY' || printf '\nNOT READY (no live frames yet)'
printf ': http://localhost:%s/#present   keys: P presentation, F full screen, V draw the zone, A acknowledge, M sound\n' "$PORT"
printf 'Zone drawn for THIS camera position? Press V on the page, draw, save. Test: one person in a lab coat alone in front of the camera must get a green box.\n'
