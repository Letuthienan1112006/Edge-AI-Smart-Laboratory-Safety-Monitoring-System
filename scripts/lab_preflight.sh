#!/usr/bin/env bash
# Laptop side: one command before testing / filming at the lab. Checks everything that has gone wrong before, prints OK / !! per line.
#   bash scripts/lab_preflight.sh          # check only
#   bash scripts/lab_preflight.sh --open   # also open the dashboard tunnel (background) and the browser
# Read-only on the board except the clock (sync_board_clock.sh). Never restarts anything.
set -uo pipefail
HOST="${HOST:-orangepi-lan}"
cd "$(dirname "$0")/.."
MODEL=runs/export/lab_v6_split_int8_640.rknn
bad=0
ok()   { printf '  OK  %s\n' "$*"; }
warn() { printf '  !!  %s\n' "$*"; bad=$((bad + 1)); }

echo "== board"
ssh -o ConnectTimeout=6 "$HOST" true 2>/dev/null || { warn "board unreachable over $HOST: check the LAN cable and power, wait ~60 s after power-on"; exit 1; }
ok "reachable ($HOST)"
bash scripts/sync_board_clock.sh "$HOST" | sed 's/^/      /'
skew=$(( $(ssh "$HOST" date +%s) - $(date +%s) )); [ ${skew#-} -le 5 ] && ok "clock within ${skew} s of the laptop" || warn "board clock off by ${skew} s"

echo "== code and model on the board = this laptop"
diff=$(rsync -nac --out-format='%n' --exclude __pycache__ src configs scripts deploy "$HOST":AI_Challenge/ 2>/dev/null | grep -v '/$')
[ -z "$diff" ] && ok "src/configs/scripts/deploy identical" || warn "differs from laptop: $(echo $diff | head -c 300) -> make deploy-board, then restart the service"
l=$(sha256sum "$MODEL" | cut -c1-16); r=$(ssh "$HOST" "sha256sum AI_Challenge/$MODEL 2>/dev/null" | cut -c1-16)
[ "$l" = "$r" ] && ok "model $MODEL ($l)" || warn "model differs/missing on the board (laptop $l, board ${r:-none})"
ssh "$HOST" 'python3 -c "import PIL"' 2>/dev/null && ok "Pillow present (Vietnamese labels)" || warn "no Pillow on the board: labels without diacritics"

echo "== live service"
# one property per call: `systemctl show -p A -p B --value` prints in systemd's own order, not the order asked (misread on 23/10)
st=$(ssh "$HOST" 'for p in ActiveState NRestarts ExecMainStartTimestamp; do systemctl show labsafe-live -p $p --value; done | paste -sd"|"')
IFS='|' read -r act nre since <<<"$st"
[ "$act" = active ] && ok "labsafe-live active, $nre restart(s), started $since" \
  || warn "labsafe-live $act -> camera plugged? ssh -t $HOST 'sudo systemctl restart labsafe-live'"
fpsc=$(ssh "$HOST" 'v4l2-ctl -d /dev/video0 --get-ctrl=exposure_dynamic_framerate 2>/dev/null' | grep -o '[01]$')
[ "$fpsc" = 1 ] && printf '      note: the camera lowers its frame rate in dim light (exposure_dynamic_framerate=1): <30 FPS on a dark scene is the camera\n'
cfg=$(ssh "$HOST" "pgrep -af 'labsafe.app' | grep -o -- '--config [^ ]*' | head -1")
ok "running ${cfg:-?}"
ssh "$HOST" "systemctl cat labsafe-live.service | grep -q LABSAFE_CONFIG" && ok "unit can switch config (scripts/live_config.sh)" \
  || warn "installed unit is older than 23/10: make install-live-service once (needed for live_config.sh demo|standard)"
ssh "$HOST" 'ls /dev/video0 >/dev/null 2>&1' && ok "camera /dev/video0 present" || warn "no /dev/video0: plug the camera (the service retries every 10 s)"
s=$(ssh "$HOST" 'curl -s -m 3 http://127.0.0.1:8000/status' 2>/dev/null)
if [ -n "$s" ]; then
  python3 - "$s" <<'PY'
import json, sys
j = json.loads(sys.argv[1]); s = j.get("summary", {}); c = j.get("stage_calls", {})
fps, age = j.get("fps", 0), j.get("frame_age_s", 1e9)
print(("  OK  " if fps >= 15 and age < 1 else "  !!  ") + f"/status: {fps} FPS, frame age {age} s, {s.get('persons', 0)} person(s), {s.get('wearing', 0)} with coat")
print(("  OK  " if not c.get("c") else "  !!  ") + f"fire/smoke model calls: {c.get('c', 0)} (must be 0: switched off)")
PY
else warn "/status does not answer (service starting, or camera missing)"; fi

echo "== Zalo notification (optional)"
if ssh "$HOST" 'test -s AI_Challenge/secrets/zalo.json'; then
  ok "Zalo secret on the board"
  ssh "$HOST" 'curl -s -o /dev/null -m 5 https://bot-api.zaloplatforms.com' && ok "board reaches the Zalo API (internet OK)" \
    || warn "board has no internet: phone USB tethering into the board (Zalo messages are dropped meanwhile)"
else printf '      Zalo not set up (make zalo-setup TOKEN=... then make zalo-push): alarms stay on the dashboard only\n'; fi

echo "== LED + OLED panel (optional)"
pl=$(ssh "$HOST" 'grep -a "labsafe.panel" AI_Challenge/runs/labsafe-live.log 2>/dev/null | tail -3 | cut -c1-200')
case "$pl" in
  *"LEDs on, OLED on"*) ok "LEDs and OLED running";;
  *"LEDs on"*) warn "LEDs run, OLED not: $(echo "$pl" | tail -1 | cut -c1-150)  -> make panel-setup, reboot the board";;
  "") printf '      panel: no log line yet (not installed or the service predates it): make install-live-service\n';;
  *) warn "panel not running: $(echo "$pl" | tail -1 | cut -c1-160)  -> make panel-setup (once), reboot the board, make panel-test";;
esac

echo "== zone and disk"
z=$(ssh "$HOST" 'f=AI_Challenge/data/zones_live.yaml; [ -f $f ] && echo "$(date -r $f "+%d/%m %H:%M") $(grep -cE "^ *- (- |\\[)" $f) points" || echo none')
[ "$z" = none ] && warn "no zone drawn on site: the default configs/zones_lab.yaml is used -> draw the zone on the dashboard (V)" \
  || ok "zone drawn on site: $z  (drawn at HOME before 23/10? redraw it here)"
df=$(ssh "$HOST" "df -BG --output=avail ~ | tail -1 | tr -dc 0-9"); [ "${df:-0}" -ge 3 ] && ok "board disk free ${df} GB" || warn "board disk free ${df} GB (< 3 GB)"
lf=$(df -BG --output=avail . | tail -1 | tr -dc 0-9); [ "${lf:-0}" -ge 5 ] && ok "laptop disk free ${lf} GB" || warn "laptop disk free ${lf} GB"
t=$(ssh "$HOST" 'cat /sys/class/thermal/thermal_zone0/temp'); ok "SoC $((t / 1000)) °C"

if [ "${1:-}" = --open ]; then
  echo "== dashboard"
  if curl -s -o /dev/null -m 2 http://localhost:8000/status; then ok "tunnel already open"
  else
    (setsid nohup ssh -N -o ExitOnForwardFailure=yes -L 8000:127.0.0.1:8000 "$HOST" >/dev/null 2>&1 </dev/null & echo $! > /tmp/labsafe-tunnel.pid)
    sleep 2; curl -s -o /dev/null -m 3 http://localhost:8000/status && ok "tunnel open (close: kill \$(cat /tmp/labsafe-tunnel.pid))" || warn "tunnel did not open"
  fi
  command -v xdg-open >/dev/null && xdg-open http://localhost:8000/ >/dev/null 2>&1
fi
echo; [ $bad -eq 0 ] && echo "READY" || echo "$bad item(s) to fix (!! above)"
