#!/usr/bin/env bash
# Laptop side: record the backup demo video while the live system runs (needs the dashboard tunnel: make dashboard-tunnel).
#   bash scripts/record_demo.sh                    # both: the annotated camera stream + the dashboard window (click it)
#   bash scripts/record_demo.sh stream [name]      # only the annotated camera video from the board (what the AI sees)
#   bash scripts/record_demo.sh screen [name]      # only the dashboard window, with the alarm beep from the laptop speakers
#   GEOMETRY=1280x720+0+0 bash scripts/record_demo.sh screen   # fixed screen region instead of clicking a window
# Ctrl+C stops and closes the files cleanly. Output: runs/demo_video/<time>_<name>_{stream,screen}.mp4
# The board encodes each JPEG once for all viewers, so recording the stream does not slow the board down.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-both}"; NAME="${2:-demo}"
URL="${URL:-http://localhost:8000}"
case "$MODE" in stream|screen|both) ;; *) echo "usage: $0 [stream|screen|both] [name]"; exit 2;; esac
mkdir -p runs/demo_video
base="runs/demo_video/$(date +%Y%m%d-%H%M%S)_${NAME}"
pids=()

if [ "$MODE" != screen ]; then
  curl -s -o /dev/null -m 3 "$URL/status" || { echo "dashboard not reachable at $URL (run: make dashboard-tunnel)"; exit 1; }
  # MJPEG has no timestamps: stamp frames on arrival and resample to a constant 15 FPS (the dashboard's video rate)
  # -reconnect*: a dropped tunnel / service restart does not end the recording
  setsid ffmpeg -hide_banner -loglevel error -nostdin -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 \
    -use_wallclock_as_timestamps 1 -f mpjpeg -i "$URL/stream" \
    -vsync cfr -r 15 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -movflags +faststart "${base}_stream.mp4" &
  pids+=($!)
  echo "recording the camera stream -> ${base}_stream.mp4"
fi

if [ "$MODE" != stream ]; then
  if [ -z "${GEOMETRY:-}" ]; then
    echo "click the browser window that shows the dashboard ..."
    info=$(xwininfo) || exit 1
    g() { printf '%s\n' "$info" | sed -n "s/^ *$1: *\([0-9-]*\).*/\1/p"; }
    GEOMETRY="$(g Width)x$(g Height)+$(g 'Absolute upper-left X')+$(g 'Absolute upper-left Y')"
  fi
  w=${GEOMETRY%%x*}; rest=${GEOMETRY#*x}; h=${rest%%+*}; xy=${rest#*+}; x=${xy%%+*}; y=${xy#*+}
  w=$(( w / 2 * 2 )); h=$(( h / 2 * 2 ))          # H.264 4:2:0 needs even sizes
  audio=()
  sink=$(pactl get-default-sink 2>/dev/null) && [ -n "$sink" ] && audio=(-f pulse -i "${sink}.monitor")   # what the laptop plays (the beep)
  setsid ffmpeg -hide_banner -loglevel error -nostdin -f x11grab -framerate 30 -video_size "${w}x${h}" -i "${DISPLAY:-:0}+${x},${y}" \
    "${audio[@]}" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p ${audio:+-c:a aac -b:a 128k} -movflags +faststart \
    "${base}_screen.mp4" &
  pids+=($!)
  echo "recording the screen ${w}x${h}+${x}+${y}${audio:+ with sound} -> ${base}_screen.mp4"
fi

# ffmpeg runs under setsid, out of the terminal's process group: Ctrl+C reaches only this script, which sends each ffmpeg exactly ONE
# SIGINT. A second one (terminal + trap) makes ffmpeg abort without writing the MP4 index -> unplayable file (seen in the self-test).
stop() { kill -INT "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; }
trap 'stop' INT TERM
echo "Ctrl+C to stop"
wait "${pids[@]}"
trap - INT TERM
for f in "${base}"_*.mp4; do
  [ -s "$f" ] || continue
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  LC_NUMERIC=C printf '%s  %.0f s  %s\n' "$f" "${d:-0}" "$(du -h "$f" | cut -f1)"
done
