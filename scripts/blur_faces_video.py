"""Blur the heads in a video (privacy for anything that leaves the lab: poster, paper, slides).

A head is the top part of every person box (lab model, low threshold so more people are caught) plus every face the OpenCV Haar cascades
find (frontal, profile). Neither is perfect: ALWAYS look at the output before publishing (scripts/blur_faces_video.py writes a contact sheet).

  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/blur_faces_video.py data/interim/demo/boot014_demo.mp4 data/interim/demo/boot014_demo_blur.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = "/mnt/big/AGV_ANLEE/weights/lab_v6_best.pt"


def pixelate(img, x1, y1, x2, y2, cells=6):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = img[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(2, (x2 - x1) // cells), max(2, (y2 - y1) // cells)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    cas = [cv2.CascadeClassifier(cv2.data.haarcascades + f) for f in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    sheet = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        r = model.predict(fr, conf=0.15, imgsz=640, verbose=False, device="cpu")[0]
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            if int(c) != 0:
                continue
            x1, y1, x2, y2 = b
            bh = y2 - y1
            pixelate(fr, x1 - 0.06 * (x2 - x1), y1 - 0.03 * bh, x2 + 0.06 * (x2 - x1), y1 + 0.30 * bh)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        for cc in cas:
            for (x, y, fw, fh) in cc.detectMultiScale(gray, 1.1, 3, minSize=(14, 14)):
                pixelate(fr, x - 0.25 * fw, y - 0.3 * fh, x + 1.25 * fw, y + 1.3 * fh)
        out.write(fr)
        if n % 75 == 0:
            sheet.append(cv2.resize(fr, (320, 240)))
        n += 1
        if n % 100 == 0:
            print("frames", n, flush=True)
    out.release()
    rows = [np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet) - len(sheet) % 4, 4)]
    if rows:
        cv2.imwrite(str(Path(dst).with_suffix(".contact.jpg")), np.vstack(rows))
    print("wrote", dst, n, "frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
