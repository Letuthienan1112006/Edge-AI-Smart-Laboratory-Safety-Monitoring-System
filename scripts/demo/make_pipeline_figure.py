"""Real pipeline figure for the paper/poster: ONE real frame, four stages, produced by the project's own code.

  (a) camera frame   (b) stage A output (person + lab-coat boxes with scores)   (c) tracker + coat assignment + foot point + zone
  (d) the final overlay the dashboard shows (green = coat, red = violation, grey = clothes not visible).

Faces are shown as they are (the people gave their permission). Pass --blur to blur heads (top 26 % of every person box) when the
material has no such permission; the detector can miss a person, so check the output by eye.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/demo/make_pipeline_figure.py --video data/interim/demo/boot014_demo.mp4 \
      --out runs/figures/pipeline_real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from labsafe.app import build_system                      # noqa: E402
from labsafe.config import Config                         # noqa: E402
from labsafe.viz import COLOR, _put_labels, draw          # noqa: E402


def blur_heads(img: np.ndarray, boxes, frac: float = 0.26) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        bh = y2 - y1
        ya, yb = int(max(0, y1 - 0.02 * bh)), int(min(h, y1 + frac * bh))
        xa, xb = int(max(0, x1 - 4)), int(min(w, x2 + 4))
        if yb - ya < 4 or xb - xa < 4:
            continue
        roi = out[ya:yb, xa:xb]
        small = cv2.resize(roi, (max(2, roi.shape[1] // 9), max(2, roi.shape[0] // 9)), interpolation=cv2.INTER_LINEAR)
        out[ya:yb, xa:xb] = cv2.GaussianBlur(cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST), (0, 0), 3)
    return out


def render(a, router, frame, now, alarming, out_dir):
    h, w = frame.shape[:2]
    dets = router.stage_a(frame)                                           # panel (b): the raw stage A output of this frame
    heads = [(t.det.x1, t.det.y1, t.det.x2, t.det.y2) for t in router.tracks] + [(d.x1, d.y1, d.x2, d.y2) for d in dets if d.cls_name == "person"]
    base = blur_heads(frame, heads) if a.blur else frame.copy()
    out_dir.mkdir(parents=True, exist_ok=True)

    pa = base.copy()
    _put_labels(pa, [("(a) Ảnh từ camera", (6, 28), (60, 60, 60), 22)])

    pb = base.copy()
    labels = []
    for d in dets:
        col = COLOR["ok"] if d.cls_name != "person" else (255, 190, 60)
        cv2.rectangle(pb, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
        labels.append((f"{'áo' if d.cls_name != 'person' else 'người'} {str(round(d.score, 2)).replace('.', ',')}", (int(d.x1), int(d.y1)), col, 16))
    _put_labels(pb, labels + [("(b) Tầng A: người + áo", (6, 28), (60, 60, 60), 22)])

    pc = base.copy()
    for z in router.zones:
        pts = np.array(z.pixel_polygon(w, h), dtype=np.int32)
        ov = pc.copy()
        cv2.fillPoly(ov, [pts], COLOR["zone"])
        cv2.addWeighted(ov, 0.18, pc, 0.82, 0, pc)
        cv2.polylines(pc, [pts], True, COLOR["zone"], 2, cv2.LINE_AA)
    labels = []
    for tr in router.tracks:
        d = tr.det
        inside = any(z.contains_foot(d, w, h) for z in router.zones)
        wears = tr.wears("lab_coat", now, router.attire_memory_s)
        col = COLOR["ok"] if wears else (COLOR["unk"] if tr.attire_unknown else COLOR["crit"] if inside else COLOR["warn"])
        cv2.rectangle(pc, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
        fx, fy = d.foot
        cv2.circle(pc, (int(fx), int(min(fy, h - 4))), 7, (255, 255, 255), -1)
        cv2.circle(pc, (int(fx), int(min(fy, h - 4))), 5, (30, 30, 230) if inside else (130, 130, 130), -1)
        txt = "áo" if wears else ("chưa thấy áo" if tr.attire_unknown else "không áo")
        labels.append((f"#{tr.track_id} {txt}{' · chân trong' if inside else ''}", (int(d.x1), int(d.y1)), col, 16))
    _put_labels(pc, labels + [("(c) Bộ bám, gán áo, điểm chân", (6, 28), (60, 60, 60), 22)])

    pd = draw(base, router.tracks, router.zones, alarming, "", coat_scores=getattr(router, "_coat_scores", {}), memory_s=router.attire_memory_s)
    _put_labels(pd, [("(d) Kết quả: cảnh báo", (6, 28), (60, 60, 60), 22)])

    for name, img in zip("abcd", (pa, pb, pc, pd)):
        cv2.imwrite(str(out_dir / f"panel_{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", out_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live/dashboard_lab_pc.yaml")
    ap.add_argument("--classes", default="configs/classes_lab_v3.yaml")
    ap.add_argument("--video", default="data/interim/demo/boot014_demo.mp4")
    ap.add_argument("--out", default="runs/figures/pipeline_real")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--min-wearing", type=int, default=3)
    ap.add_argument("--min-persons", type=int, default=5)
    ap.add_argument("--blur", action="store_true", help="blur heads (off by default)")
    ap.add_argument("--max-persons", type=int, default=99)
    ap.add_argument("--pick", type=int, default=1, help="write this many candidate frames (each with an alarm and enough people) to --out/f<frame>/")
    a = ap.parse_args()

    cfg = Config.load(a.config, a.classes)
    router, _, _ = build_system(cfg, "adaptive")
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    i, found = a.start, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("done:", found, "frame(s) written")
            return 0 if found else 1
        i += 1
        if (i - a.start) % a.stride:
            continue
        now = i / fps
        events, alarming = router.process(frame, now=now)
        wearing = sum(1 for tr in router.tracks if tr.wears("lab_coat", now, router.attire_memory_s))
        if alarming and a.min_persons <= len(router.tracks) <= a.max_persons and wearing >= a.min_wearing:
            found += 1
            print("frame", i, "persons", len(router.tracks), "wearing", wearing, "alarming", [e.track_id for e in alarming])
            render(a, router, frame, now, alarming, Path(a.out) / f"f{i:04d}")
            if found >= a.pick:
                return 0


if __name__ == "__main__":
    sys.exit(main())
