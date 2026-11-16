#!/usr/bin/env bash
# ON BOARD, as root:  sudo bash scripts/bench/ddr_ab_board.sh [SECONDS_PER_RUN=240] [ROUNDS=3]
# End-to-end A/B of the DDR governor: the deployed pipeline (adaptive, split INT8, A/B/C async) on the replay video, paced at 30 FPS,
# default `dmc_ondemand` vs `dmc` pinned to `performance` (everything else stays default), interleaved, same video.
# Stops labsafe-live for the duration (it would share the NPU) and restores both the DDR governor and the service at the end.
set -uo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
cd "$(dirname "$0")/../.."
DUR="${1:-240}"; ROUNDS="${2:-3}"; DMC=/sys/class/devfreq/dmc/governor
ORIG=$(cat $DMC); WAS_ACTIVE=$(systemctl is-active labsafe-live.service || true)
restore() { echo "$ORIG" > $DMC; [ "$WAS_ACTIVE" = active ] && systemctl start labsafe-live.service; echo "restored: dmc=$(cat $DMC) service=$(systemctl is-active labsafe-live.service)"; }
trap restore EXIT
systemctl stop labsafe-live.service
OUT=runs/ddr_ab-$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"; chown orangepi: "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for mode in default pinned; do
    [ $mode = pinned ] && echo performance > $DMC || echo dmc_ondemand > $DMC
    sleep 20   # let the board settle after the previous run (temperature, governor)
    echo "round $r $mode (dmc=$(cat $DMC)) ..."
    sudo -u orangepi env PYTHONPATH=src python3 -m labsafe.app --bench --replay --exp ddr_${mode}_r$r --mode adaptive \
      --config configs/exp/L3_lab_v5_full_async_int8_640.yaml --classes configs/classes_lab_v5_full.yaml --source data/lab_replay.mp4 \
      --duration "$DUR" --warmup 10 --no-display --gpio off --run-dir "$OUT/${mode}_r$r" > "$OUT/${mode}_r$r.log" 2>&1
    grep -q . "$OUT/${mode}_r$r/metrics.json" 2>/dev/null && echo "  ok" || echo "  FAILED: see $OUT/${mode}_r$r.log"
  done
done
echo "$OUT"
