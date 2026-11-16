#!/usr/bin/env bash
# ON BOARD: run the live dashboard system for DURATION seconds with one browser-like MJPEG client attached,
# sample every 30 s: app RSS, CPU temperature, FPS and counts from /status. Output: runs/soak-<stamp>/soak.csv
#   bash scripts/board/soak_board.sh [DURATION_S] [CONFIG]
set -uo pipefail
cd "$(dirname "$0")/../.."
DUR="${1:-900}"; CFG="${2:-configs/live/dashboard_lab_board.yaml}"
OUT="runs/soak-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
PYTHONPATH=src timeout "$DUR" python3 -m labsafe.app --config "$CFG" --classes configs/classes_lab_v3.yaml \
  --no-display --gpio off --run-dir "$OUT/app" > "$OUT/app.log" 2>&1 &
APP=$!
sleep 15
timeout $((DUR - 20)) curl -s localhost:8000/stream -o /dev/null &   # a viewer with the page open
echo "t_s,rss_mb,temp_c,fps,persons,not_wearing,violating,alarms_saved" > "$OUT/soak.csv"
t0=$(date +%s)
while kill -0 "$APP" 2>/dev/null; do
  pid=$(pgrep -n -f "^python3 -m labsafe.app --config $CFG")
  rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  st=$(curl -s -m 2 localhost:8000/status | python3 -c 'import json,sys
try:
    j = json.load(sys.stdin); s = j.get("summary", {})
    print(",".join(str(v) for v in (j.get("fps", 0), s.get("persons", 0), s.get("not_wearing", 0),
                                     s.get("violating", 0), j.get("evidence_saved"))))
except Exception:
    print(",,,,")')
  echo "$(( $(date +%s) - t0 )),$rss,$temp,$st" >> "$OUT/soak.csv"
  sleep 30
done
echo "$OUT"
