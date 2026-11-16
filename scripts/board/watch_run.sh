#!/usr/bin/env bash
# Live view of a running labsafe bench on the board: one line every 5 s.
# Usage: bash scripts/board/watch_run.sh [log-file]   (default: newest runs/soak_*.log)
cd "$(dirname "$0")/../.."
log="${1:-$(ls -t runs/soak_*.log 2>/dev/null | head -1)}"
[ -n "$log" ] || { echo "no log found"; exit 1; }
echo "log: $log   (Ctrl+C to stop watching; the run itself keeps going)"
printf '%-9s %-10s %-9s %-9s %s\n' "time" "progress" "RAM_MB" "temp_C" "frames"
while true; do
  pid=$(pgrep -f "labsafe.app --bench" | while read p; do
          grep -q "labsafe.app" /proc/$p/cmdline 2>/dev/null && ! grep -q "bash" /proc/$p/cmdline && echo $p; done | head -1)
  ram=$( [ -n "$pid" ] && awk '/VmRSS/ {printf "%.0f", $2/1024}' /proc/$pid/status || echo "-")
  temp=$(awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp)
  last=$(tail -c 200 "$log" | tr '\r' '\n' | grep -E '[0-9]+s / [0-9]+s' | tail -1)
  prog=$(echo "$last" | grep -oE '[0-9]+s / [0-9]+s')
  frames=$(echo "$last" | grep -oE 'frames=[0-9]+')
  printf '%-9s %-10s %-9s %-9s %s\n' "$(date +%H:%M:%S)" "${prog:--}" "$ram" "$temp" "$frames"
  [ -z "$pid" ] && { echo "run finished"; break; }
  sleep 5
done
