#!/usr/bin/env bash
# Laptop side: pull recorded sessions from the board, cut them into frames, draft labels.
#   bash scripts/fetch_and_prepare.sh                 # host orangepi-lan, fps 2, with lab_coat drafts
#   HOST=orangepi FPS=1 COAT=0 bash scripts/fetch_and_prepare.sh
# Safe to re-run: sessions already extracted / pre-labelled are skipped (delete their folder to redo).
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${HOST:-orangepi-lan}"; FPS="${FPS:-2}"; COAT="${COAT:-1}"; PY=".venv/bin/python"
mkdir -p data/raw/lab_sessions
echo "== pull sessions from $HOST"
[ "$HOST" = local ] || rsync -a --info=stats1 "$HOST:AI_Challenge/data/raw/lab_sessions/" data/raw/lab_sessions/
for s in data/raw/lab_sessions/*/; do
  name="$(basename "$s")"
  [ -f "$s/video.avi" ] || { echo "-- $name: no video.avi (interval session or incomplete), skipped"; continue; }
  if [ ! -f "data/interim/lab_frames/$name/extract.json" ]; then
    "$PY" scripts/extract_frames.py "$s" --fps "$FPS"
  fi
  if [ ! -f "data/interim/prelabel/$name/prelabel.json" ]; then
    extra=""; [ "$COAT" = 1 ] && extra="--lab-coat"
    "$PY" scripts/prelabel.py "data/interim/lab_frames/$name" --out "data/interim/prelabel/$name" --copy-images $extra
  fi
done
echo; echo "Next: open data/interim/prelabel/<session>/ in CVAT/Label Studio (images/ + labels/ + obj.names),"
echo "correct the boxes (review.csv: least confident first), export YOLO to data/labeled/<session>/{images,labels},"
echo "then: make build-lab-dataset"
