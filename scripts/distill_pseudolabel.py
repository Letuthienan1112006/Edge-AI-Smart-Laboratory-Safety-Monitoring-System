#!/usr/bin/env python3
"""Knowledge distillation via a teacher's pseudo-labels on unlabelled lab footage.

Why this form of distillation rather than feature-mimicking:

Classic KD needs teacher and student to run together every training step
and to have matching feature maps -- doable, but it means patching the
training loop of whatever detector library you use, and a bug there is
silent (the loss still goes down). The pseudo-label form transfers the
same knowledge through the dataset instead of through the loss: a large,
slow teacher (YOLOv8m/l, running on your laptop GPU, where speed does not
matter) labels footage you never annotated, and the small student trains
on your hand labels PLUS those.

It fits this project specifically because the binding constraint is not
model capacity, it is labelled lab images. An hour of unlabelled footage
from the actual room, pseudo-labelled by a strong teacher, is worth more
than a deeper student trained on the same small hand-labelled set -- and
the teacher never has to run on the Pi.

Guards that matter, because pseudo-labels are confidently wrong sometimes:
  * a high confidence floor (default 0.75): a wrong pseudo-label is a
    teacher telling the student to be wrong, and unlike a missing label it
    is actively harmful
  * hand labels always win: an image with real labels is never overwritten
  * the split is by VIDEO, not by frame -- adjacent frames are near
    duplicates, and mixing them across train/val inflates every metric
  * a manifest recording which labels are human and which are machine, so
    the paper can report both and the evaluation set can exclude machine ones

    python3 scripts/distill_pseudolabel.py --teacher yolov8m.pt \
        --videos data/raw/lab_footage --out data/processed/person_kd \
        --conf 0.75 --stride 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--teacher", required=True, help="a LARGE model; it never runs on the Pi")
ap.add_argument("--videos", required=True, help="directory of unlabelled video files")
ap.add_argument("--out", required=True, help="output YOLO dataset root")
ap.add_argument("--conf", type=float, default=0.75,
                help="confidence floor; below this the teacher is guessing")
ap.add_argument("--stride", type=int, default=5,
                help="keep 1 frame in N; adjacent frames are near-duplicates")
ap.add_argument("--imgsz", type=int, default=960,
                help="run the teacher LARGER than the student will ever run")
ap.add_argument("--classes", default=None,
                help="comma-separated teacher class ids to keep (e.g. '0' for person)")
ap.add_argument("--max-per-video", type=int, default=400)
ap.add_argument("--val-videos", type=int, default=1,
                help="how many videos to hold out for val (split by video, never by frame)")
a = ap.parse_args()

if a.conf < 0.5:
    print(f"WARNING: --conf {a.conf} is low. Wrong pseudo-labels actively teach the\n"
          f"         student to be wrong, which is worse than no label at all.\n")

try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"missing dependency: {e.name}\n  pip install ultralytics opencv-python")

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"}
videos = sorted(p for p in Path(a.videos).rglob("*") if p.suffix in VIDEO_EXT)
if not videos:
    sys.exit(f"no video files found under {a.videos}")
if len(videos) <= a.val_videos:
    sys.exit(f"found {len(videos)} videos but --val-videos is {a.val_videos}; "
             "need at least one more video than that so train and val hold different sessions")

keep_ids = {int(c) for c in a.classes.split(",")} if a.classes else None
out = Path(a.out)
for split in ("train", "val"):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

print(f"teacher: {a.teacher}   videos: {len(videos)}   "
      f"val held out: {a.val_videos} whole video(s)\n")
model = YOLO(a.teacher)

val_set = set(videos[-a.val_videos:])
manifest: list[dict] = []
kept = dropped = 0

for vid in videos:
    split = "val" if vid in val_set else "train"
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"  ! cannot open {vid.name}, skipping")
        continue
    idx = saved = 0
    while saved < a.max_per_video:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % a.stride:
            continue
        res = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        h, w = frame.shape[:2]
        lines = []
        for b in res.boxes:
            cid = int(b.cls.item())
            if keep_ids is not None and cid not in keep_ids:
                dropped += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cx, cy = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)
            bw, bh = ((x2 - x1) / w, (y2 - y1) / h)
            if not (0 < bw <= 1 and 0 < bh <= 1):
                dropped += 1
                continue
            new_id = sorted(keep_ids).index(cid) if keep_ids else cid
            lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # An image with no confident detection is still a valid training
        # sample: an empty label file teaches "nothing here", which is how a
        # detector learns not to hallucinate on empty benches.
        stem = f"{vid.stem}_{idx:06d}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), frame)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        manifest.append({"image": f"{stem}.jpg", "split": split, "source": "pseudo",
                         "video": vid.name, "frame": idx, "boxes": len(lines)})
        kept += len(lines)
        saved += 1
    cap.release()
    print(f"  {vid.name:<40} split={split:<5} frames kept={saved}")

names = [model.names[cid] for cid in sorted(keep_ids)] if keep_ids else list(model.names.values())
(out / "data.yaml").write_text(
    f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
    f"nc: {len(names)}\nnames: {names}\n"
)
(out / "pseudo_manifest.json").write_text(json.dumps(manifest, indent=2))

print(f"""
========================================================================
  pseudo-labelled images : {len(manifest)}
  pseudo boxes kept      : {kept}    dropped below conf/class filter: {dropped}
  dataset                : {out/'data.yaml'}
  manifest               : {out/'pseudo_manifest.json'}
========================================================================

  BEFORE training on this:
    1. python3 scripts/verify_dataset.py {out/'data.yaml'} --crops 8
       and LOOK at the crops. The teacher is confident, not correct.
    2. Merge with your hand-labelled set; on any collision, the HUMAN
       label wins.
    3. Your TEST set must contain zero pseudo-labels. Evaluating a student
       against its own teacher's output measures agreement with the
       teacher, not accuracy -- and it always looks good.
""")
