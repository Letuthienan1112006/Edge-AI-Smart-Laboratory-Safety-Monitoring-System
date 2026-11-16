#!/usr/bin/env bash
# Laptop side: 640 vs 512 latency, with default governors and with ALL clocks pinned to performance, same models, same images.
#   bash scripts/bench_pinned.sh            # asks for the board's sudo password once (ssh -t); governors are restored at the end
# Output: runs/lat_pinned/{default,pinned}.json (+ printed). The camera/app must not be running on the board.
set -euo pipefail
HOST="${HOST:-orangepi-lan}"
cd "$(dirname "$0")/.."
# A benchmark sharing the NPU core / big CPUs with the live app measures the contention, not the model.
busy=$(ssh "$HOST" 'systemctl is-active labsafe-live.service 2>/dev/null; ps -eo cmd | grep -E "[l]absafe.app|[c]amera_view|[c]apture_interval" | head -3' || true)
if echo "$busy" | grep -q -E "^active|labsafe.app|camera_view|capture_interval"; then
  echo "REFUSING TO MEASURE: something else is using the NPU/camera on the board:"; echo "$busy"
  echo "stop it first, e.g.:  ssh -t $HOST 'sudo systemctl stop labsafe-live'   (start it again afterwards)"; exit 3
fi
M640=runs/export/split/split640_s0_base_300_int8_640.rknn; M512=runs/export/split/split512_s0_base_300_int8_512.rknn
rsync -aq scripts src "$HOST":AI_Challenge/
rsync -aq "$M640" "$M640.card.json" "$M512" "$M512.card.json" "$HOST":AI_Challenge/runs/export/split/
RUN="cd ~/AI_Challenge && mkdir -p runs/lat_pinned && PYTHONPATH=src taskset -c 4-7 python3 scripts/bench_backend_latency.py --models $M640:640 $M512:512"
ssh "$HOST" "$RUN --label default --out runs/lat_pinned/default.json" 2>&1 | { grep -v -E "^[IW] RKNN|^W rknn-toolkit|^W Query|onnxruntime" || true; }
trap 'ssh -t "$HOST" "sudo bash ~/AI_Challenge/scripts/pin_clocks.sh restore"' EXIT
ssh -t "$HOST" "sudo bash ~/AI_Challenge/scripts/pin_clocks.sh pin"
ssh "$HOST" "$RUN --label pinned --out runs/lat_pinned/pinned.json" 2>&1 | { grep -v -E "^[IW] RKNN|^W rknn-toolkit|^W Query|onnxruntime" || true; }
mkdir -p runs/lat_pinned && rsync -aq "$HOST":AI_Challenge/runs/lat_pinned/ runs/lat_pinned/
