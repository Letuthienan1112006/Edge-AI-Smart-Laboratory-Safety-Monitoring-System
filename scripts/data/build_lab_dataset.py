#!/usr/bin/env python3
"""Assemble corrected lab labels into one YOLO dataset, split BY SESSION.

    python3 scripts/data/build_lab_dataset.py                    # data/labeled/<session>/{images,labels}
    python3 scripts/data/build_lab_dataset.py --test L1_15,L1_16 --val L1_14

Split comes from the `split` field of configs/collection_plan.yaml (override with --train/--val/--test
session lists). Every image of a session goes to exactly ONE split, so train/test never share a
recording (adjacent frames are near-duplicates; KNOWN_ISSUES #17/#29). Refuses to build when the test
split is empty, when an image has no label file, or when a label uses a class id outside configs/lab_classes.yaml.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASSES = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "lab_classes.yaml").read_text())["names"]
IMG = {".jpg", ".jpeg", ".png", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("--labeled", default="data/labeled")
ap.add_argument("--plan", nargs="+", default=["configs/collection_plan.yaml", "configs/collection_plan_L2.yaml"],
                help="collection plans whose `split` hints decide train/val/test per session (several allowed)")
ap.add_argument("--out", default="data/processed/lab_v1")
for k in ("train", "val", "test"):
    ap.add_argument(f"--{k}", default="", help=f"comma-separated session ids forced into {k}")
ap.add_argument("--pilot-time-holdout", type=float, default=0.0, metavar="FRAC",
                help="PILOT ONLY, for when there is a single session: the LAST FRAC of each session (by frame order, a "
                     "contiguous block) becomes test, with a 5%% gap dropped before it. Same people, same scene -> the test "
                     "numbers are OPTIMISTIC and must not be reported as generalisation (KNOWN_ISSUES #17/#29).")
a = ap.parse_args()

hint: dict[str, str] = {}
for plan_path in a.plan:
    plan = yaml.safe_load(Path(plan_path).read_text())
    hint.update({f"{plan['batch']}_{sc['id']}": sc["split"] for sc in plan["scenes"]})
force = {s: k for k in ("train", "val", "test") for s in getattr(a, k).split(",") if s}
sessions = sorted(p for p in Path(a.labeled).iterdir() if (p / "images").is_dir()) if Path(a.labeled).is_dir() else []
if not sessions:
    sys.exit(f"no labeled sessions under {a.labeled}/<session>/images + labels")

out = Path(a.out)
if out.exists():
    shutil.rmtree(out)
split_of: dict[str, str] = {}
stats = {k: {"sessions": [], "images": 0, "boxes": Counter()} for k in ("train", "val", "test")}
problems: list[str] = []
for sd in sessions:
    name = sd.name
    base_split = force.get(name) or hint.get(name) or "train"
    split = base_split
    split_of[name] = base_split
    all_imgs = sorted(p for p in (sd / "images").iterdir() if p.suffix.lower() in IMG)
    n_all = len(all_imgs)
    cut = int(n_all * (1 - a.pilot_time_holdout)) if a.pilot_time_holdout > 0 else n_all
    gap = int(n_all * 0.05) if a.pilot_time_holdout > 0 else 0
    for idx, img in enumerate(all_imgs):
        if a.pilot_time_holdout > 0:
            if cut - gap <= idx < cut:
                continue                       # gap between train and the held-out block
            split = "test" if idx >= cut else ("val" if not force.get(name) and False else base_split)
        lbl = sd / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            problems.append(f"{name}/{img.name}: no label file (an empty file means 'nothing here'; a missing one is an error)")
            continue
        rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
        bad = [r for r in rows if len(r) != 5 or not r[0].isdigit() or int(r[0]) >= len(CLASSES)]
        if bad:
            problems.append(f"{name}/{lbl.name}: bad line(s) {bad[:2]}")
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        pre = "" if img.stem.startswith(f"{name}__") else f"{name}__"      # do not double the session prefix
        shutil.copy2(img, out / "images" / split / f"{pre}{img.name}")
        shutil.copy2(lbl, out / "labels" / split / f"{pre}{img.stem}.txt")
        stats[split]["images"] += 1
        stats[split]["boxes"].update(CLASSES[int(r[0])] for r in rows)
    for sp_ in ({base_split, "test"} if a.pilot_time_holdout > 0 else {base_split}):
        if name not in stats[sp_]["sessions"]:
            stats[sp_]["sessions"].append(name)

if problems:
    print("PROBLEMS (dataset not trusted until fixed):", *problems[:20], sep="\n  ")
    shutil.rmtree(out, ignore_errors=True)
    sys.exit(1)
if stats["test"]["images"] == 0:
    shutil.rmtree(out, ignore_errors=True)
    sys.exit("test split is empty -- label at least one session marked split: test (or pass --test <session>)")
if a.pilot_time_holdout <= 0:
    overlap = [s for s in split_of if sum(s in stats[k]["sessions"] for k in stats) > 1]
    assert not overlap, overlap      # a session can only be in one split by construction
(out / "data.yaml").write_text(yaml.safe_dump({
    "path": str(out.resolve()), "train": "images/train", "val": "images/val" if stats["val"]["images"] else "images/train",
    "test": "images/test", "names": dict(enumerate(CLASSES))}, sort_keys=False))
(out / "split.json").write_text(json.dumps({k: {**v, "boxes": dict(v["boxes"])} for k, v in stats.items()}, indent=2))
for k, v in stats.items():
    print(f"{k:5s} {v['images']:5d} images  sessions={len(v['sessions'])}  boxes={dict(v['boxes'])}")
if not stats["val"]["images"]:
    print("NOTE: no val sessions -> val points at train (early-stopping numbers will be optimistic); label a val session.")
if a.pilot_time_holdout > 0:
    print("!! PILOT SPLIT: test = last block of the SAME session (same scene, same people). Scores are optimistic;")
    print("   do NOT report them as generalisation. Use a different session / different people for a real test.")
print(f"-> {out}/data.yaml   next: python3 scripts/data/verify_dataset.py {out}/data.yaml --crops 8")
