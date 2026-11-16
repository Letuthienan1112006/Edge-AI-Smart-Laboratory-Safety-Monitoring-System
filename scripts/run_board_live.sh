#!/usr/bin/env bash
# Standard Orange Pi live run: always keeps logs, video, summary and snapshots.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

stamp="$(date +%Y%m%d-%H%M%S)"
run_dir="runs/board-live-${stamp}"
mkdir -p "$run_dir"

echo "Starting LabSafe; press Ctrl+C to finish cleanly."
echo "Artifacts will be in: ${run_dir}"
set +e
# Ctrl+C is delivered to every process in the foreground pipeline.  Keep the
# wrapper and tee alive; labsafe.app installs its own SIGINT handler and stops
# after the current NPU frame, then closes MP4/JSON cleanly.
trap '' INT
python3 -m labsafe.app \
  --source 0 \
  --no-display \
  --gpio off \
  --config configs/exp/N1L_live_rknn_fp16_640.yaml \
  --classes configs/classes_coco80.yaml \
  --run-dir "${run_dir}" \
  --record-video \
  --record-fps "${LABSAFE_RECORD_FPS:-20}" \
  --save-every "${LABSAFE_SAVE_EVERY:-3}" \
  --save-max "${LABSAFE_SAVE_MAX:-100}" \
  2>&1 | tee "${run_dir}/console.log"
app_status=${PIPESTATUS[0]}
trap - INT
set -e

echo "Done: ${run_dir}"
exit "$app_status"
