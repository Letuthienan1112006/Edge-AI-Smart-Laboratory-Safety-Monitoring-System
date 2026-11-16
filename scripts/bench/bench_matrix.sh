#!/usr/bin/env bash
# The experiment matrix from docs/research-plan.md, one run per line.
# Run ON THE PI. Each config repeats 3x; the harness handles warmup and
# the sustained window. Between runs the board cools down -- a back-to-back
# second run starts hot and measures a different machine.
set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE="${SOURCE:-data/processed/bench_clip.mp4}"
DURATION="${DURATION:-900}"
COOLDOWN="${COOLDOWN:-180}"
REPEATS="${REPEATS:-3}"

run () {   # run <exp_id> <mode> <extra args...>
  local exp="$1"; local mode="$2"; shift 2
  for i in $(seq 1 "$REPEATS"); do
    echo "=== ${exp} repeat ${i}/${REPEATS} (mode=${mode}) ==="
    python3 -m labsafe.app --bench --exp "${exp}r${i}" --mode "$mode" \
      --source "$SOURCE" --duration "$DURATION" --no-display "$@"
    echo "--- cooling down ${COOLDOWN}s ---"
    sleep "$COOLDOWN"
  done
}

export PYTHONPATH="src:${PYTHONPATH:-}"

# Edit configs/pipeline.yaml between groups to change backend/imgsz, or keep
# per-experiment config files under configs/exp/ and pass --config.
run E1 parallel --config configs/exp/E1_torch_640.yaml
run E2 parallel --config configs/exp/E2_ncnn_640.yaml
run E3 parallel --config configs/exp/E3_tflite_int8_640.yaml
run E4 parallel --config configs/exp/E4_tflite_int8_320.yaml
run E5 adaptive --config configs/exp/E4_tflite_int8_320.yaml

python3 -m labsafe.bench.report runs
