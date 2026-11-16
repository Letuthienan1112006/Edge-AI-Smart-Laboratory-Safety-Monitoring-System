#!/usr/bin/env bash
# Grab ONE frame from the camera and open it, to check framing/lighting before recording.
#   bash scripts/board/peek_camera.sh              # the board  (ssh orangepi-lan)
#   bash scripts/board/peek_camera.sh local        # this laptop's camera (testing)
set -euo pipefail
host="${1:-orangepi-lan}"
dst="${PEEK_OUT:-/tmp/peek.jpg}"
grab='import cv2,time,sys
c=cv2.VideoCapture(0); c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*"MJPG")); c.set(3,640); c.set(4,480)
t=time.time()
while time.time()-t<3: c.read()          # auto-exposure warm-up (KNOWN_ISSUES #19)
ok,f=c.read(); cv2.imwrite(sys.argv[1],f); print("std=%.1f"%f.std())'
if [ "$host" = local ]; then
  "$(dirname "$0")/../../.venv/bin/python" -c "$grab" "$dst"
else
  ssh "$host" "python3 -c '$grab' /tmp/peek.jpg" && scp -q "$host:/tmp/peek.jpg" "$dst"
fi
echo "saved $dst"
command -v xdg-open >/dev/null && [ -n "${DISPLAY:-}" ] && xdg-open "$dst" >/dev/null 2>&1 || true
