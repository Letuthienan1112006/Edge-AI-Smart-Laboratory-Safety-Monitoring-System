#!/usr/bin/env bash
# The N* (NPU) experiment matrix -- see docs/experiment-ids.md. Run ON THE
# BOARD (Orange Pi 5 / RK3588S): RKNN inference only works on aarch64.
# Converting the .rknn files themselves happens on an x86 host first, with
# scripts/export/export_rknn.py -- see docs/rknn-pipeline.md.
#
# Unlike scripts/bench/bench_matrix.sh (the old E* CPU-runtime matrix), every
# rknn run here is gated by the parity check FIRST. A model that fails
# parity is a research result (docs/KNOWN_ISSUES.md #3), not something worth
# spending 15 minutes benchmarking -- so its bench run is SKIPPED, not
# aborted, and the matrix keeps going.
set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE="${SOURCE:-data/processed/bench_clip.mp4}"
DURATION="${DURATION:-900}"
COOLDOWN="${COOLDOWN:-180}"
REPEATS="${REPEATS:-3}"
CALIB_IMAGES="${CALIB_IMAGES:-data/calib}"

# Artefact names are derived ONCE here and nowhere else in this file. They
# must match labsafe.models.naming.export_name(); tests/test_naming.py pins
# that. A hand-copied path here is how the parity gate ends up validating a
# different file than the benchmark loads -- see docs/KNOWN_ISSUES.md #9.
STEM="${STEM:-yolo26n}"
BASE_IMGSZ="${BASE_IMGSZ:-640}"
REF_ONNX="runs/export/${STEM}_${BASE_IMGSZ}.onnx"
MODEL_FP16="runs/export/${STEM}_fp16_${BASE_IMGSZ}.rknn"
MODEL_INT8="runs/export/${STEM}_int8_${BASE_IMGSZ}.rknn"
NAMES="${NAMES:-person}"

export PYTHONPATH="src:${PYTHONPATH:-}"

SKIP_LOG="runs/skipped.log"
mkdir -p runs

# parity_gate <exp_id> <rknn_weights> <onnx_reference>
# Exit code 2 (FAIL) -> skip this run, log why, keep going.
# Exit code 1 (infra: missing rknnlite, wrong host arch, bad images) -> stop
# the whole matrix, because that means the ENVIRONMENT is broken, not just
# one model.
parity_gate () {
  local exp="$1"; local rknn="$2"; local onnx="$3"
  echo "=== parity gate: ${exp} (${rknn}) ==="
  set +e
  python3 scripts/export/validate_conversion.py \
    --reference-backend onnx --reference-weights "$onnx" \
    --candidate-backend rknn --candidate-weights "$rknn" \
    --images "$CALIB_IMAGES" --names "$NAMES" \
    --out "runs/parity/${exp}"
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    return 0
  elif [ "$rc" -eq 2 ]; then
    echo "$(date -Iseconds)  ${exp}  SKIPPED  parity FAIL (see runs/parity/${exp}/parity.json)" \
      >> "$SKIP_LOG"
    echo "!! ${exp}: parity FAILED -- skipping this run, see runs/parity/${exp}/parity.md"
    return 1
  else
    echo "!! ${exp}: parity gate hit an infrastructure error (exit ${rc}) -- stopping the matrix."
    exit 1
  fi
}

run () {   # run <exp_id> <mode> <config> <extra args...>
  local exp="$1"; local mode="$2"; local config="$3"; shift 3
  for i in $(seq 1 "$REPEATS"); do
    echo "=== ${exp} repeat ${i}/${REPEATS} (mode=${mode}) ==="
    python3 -m labsafe.app --bench --exp "${exp}r${i}" --mode "$mode" \
      --source "$SOURCE" --duration "$DURATION" --no-display --config "$config" "$@"
    echo "--- cooling down ${COOLDOWN}s ---"
    sleep "$COOLDOWN"
  done
}

# N0: CPU/ONNX baseline -- no NPU, no parity gate needed.
run N0 parallel configs/exp/N0_cpu_onnx_640.yaml

# N1 / N1b / N2: gated by parity first.
if parity_gate N1 "$MODEL_FP16" "$REF_ONNX"; then
  run N1 parallel configs/exp/N1_rknn_fp16_640.yaml
fi
if parity_gate N1b "$MODEL_FP16" "$REF_ONNX"; then
  run N1b parallel configs/exp/N1b_rknn_fp16_640_core_all.yaml
fi
if parity_gate N2 "$MODEL_INT8" "$REF_ONNX"; then
  run N2 parallel configs/exp/N2_rknn_int8_640.yaml
else
  echo "N2 (INT8) failing parity is itself a reportable result -- see docs/KNOWN_ISSUES.md #3 " \
       "and docs/experiment-ids.md. Not a reason to stop the matrix."
fi

# N3a/b/c: resolution sweep. Each point needs its OWN export -- RKNN fixes the
# input shape at convert time, so these are NOT N1's file at a different imgsz.
# A missing export must skip that point, never abort the matrix: the runs above
# already cost hours and the report at the end must still be produced.
for pt in N3a:320 N3b:416 N3c:512; do
  exp="${pt%%:*}"; sz="${pt##*:}"
  model="runs/export/${STEM}_fp16_${sz}.rknn"
  if [ -f "$model" ]; then
    ref_sz="runs/export/${STEM}_${sz}.onnx"
    if [ -f "$ref_sz" ] && parity_gate "$exp" "$model" "$ref_sz"; then
      run "$exp" parallel "configs/exp/${exp}_rknn_fp16_${sz}.yaml" || \
        echo "$exp failed -- continuing the matrix"
    else
      echo "SKIP $exp: parity gate not satisfied (needs $ref_sz alongside $model)."
      echo "  Each sweep point is its own conversion, so it needs its own gate;"
      echo "  N1's gate says nothing about this file."
    fi
  else
    echo "SKIP $exp: $model not found. Export it first:"
    echo "  python3 scripts/export/export_rknn.py <yolo26n.pt> --imgsz $sz --precision fp16 --target rk3588"
  fi
done

# N5: adaptive vs N1's parallel baseline, same model.
run N5 adaptive configs/exp/N5_adaptive_fp16_640.yaml

python3 -m labsafe.bench.report runs
