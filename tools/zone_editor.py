#!/usr/bin/env python3
"""Click a restricted-zone polygon onto a live frame, write configs/zones.yaml.

  left click  add point      right click  undo
  n           next zone      s            save
  q / ESC     quit
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--out", default="configs/zones.yaml")
ap.add_argument("--image", default=None, help="use a still image instead of a camera")
a = ap.parse_args()

if a.image:
    frame = cv2.imread(a.image)
    if frame is None:
        sys.exit(f"cannot read {a.image}")
else:
    dev = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(dev)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read a frame from {a.source}")

H, W = frame.shape[:2]
zones: list[dict] = []
current: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and current:
        current.pop()


cv2.namedWindow("zones")
cv2.setMouseCallback("zones", on_mouse)
print(__doc__)

while True:
    vis = frame.copy()
    for z in zones:
        pts = [(int(px * W), int(py * H)) for px, py in z["polygon"]]
        for i in range(len(pts)):
            cv2.line(vis, pts[i], pts[(i + 1) % len(pts)], (200, 120, 255), 2)
        cv2.putText(vis, z["name"], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 120, 255), 1)
    for i, p in enumerate(current):
        cv2.circle(vis, p, 4, (60, 220, 60), -1)
        if i:
            cv2.line(vis, current[i - 1], p, (60, 220, 60), 2)
    cv2.putText(vis, f"zone {len(zones)+1}: {len(current)} pts | n=next s=save q=quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.imshow("zones", vis)

    k = cv2.waitKey(20) & 0xFF
    if k in (27, ord("q")):
        break
    if k == ord("n") and len(current) >= 3:
        name = input("zone name: ").strip() or f"zone_{len(zones)+1}"
        sev = input("severity [low/medium/high] (medium): ").strip() or "medium"
        # normalised, so the zones survive a resolution change
        zones.append({"name": name, "severity": sev,
                      "polygon": [[round(x / W, 4), round(y / H, 4)] for x, y in current]})
        current.clear()
    if k == ord("s"):
        if len(current) >= 3:
            zones.append({"name": f"zone_{len(zones)+1}", "severity": "medium",
                          "polygon": [[round(x / W, 4), round(y / H, 4)] for x, y in current]})
            current.clear()
        Path(a.out).write_text(yaml.safe_dump({"zones": zones}, sort_keys=False))
        print(f"wrote {len(zones)} zones -> {a.out}")
        break

cv2.destroyAllWindows()
