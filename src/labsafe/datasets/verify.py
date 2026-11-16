"""Verify a YOLO dataset before training it.

The failure this exists to catch: a converter's `names:` list is a CLAIM
about what the integer labels mean, and nothing enforces it. Export a
dataset from one tool, convert it with another, and the class order can
shift silently -- training then optimises a model to call goggles
"gloves", the loss curve looks perfectly healthy, and the mistake only
surfaces as unexplainable behaviour at deployment.

So: count labels per class, show them next to the names, and force a look
at real crops per class before any training run. Cheap now, expensive later.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _split_dirs(root: Path, data: dict, split: str) -> Path | None:
    val = data.get(split)
    if not val:
        return None
    p = Path(val)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p if p.exists() else None


def verify_dataset(data_yaml: str | Path, sample_crops: int = 0, out_dir: str | Path | None = None) -> dict:
    """Return per-split, per-class counts plus every structural problem found."""
    data_yaml = Path(data_yaml).resolve()
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    report: dict = {"data_yaml": str(data_yaml), "names": names, "splits": {}, "problems": []}

    for split in ("train", "val", "test"):
        d = _split_dirs(root, data, split)
        if d is None:
            continue
        images = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
        counts: Counter[int] = Counter()
        boxes_per_image: list[int] = []
        empty, missing, malformed = 0, 0, 0

        for img in images:
            lp = _label_path(img)
            if not lp.exists():
                missing += 1
                continue
            lines = [ln for ln in lp.read_text().splitlines() if ln.strip()]
            if not lines:
                empty += 1
            n_ok = 0
            for ln in lines:
                parts = ln.split()
                try:
                    cid = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:]]
                except (ValueError, IndexError):
                    malformed += 1
                    continue
                if len(coords) < 4 or any(not (-0.001 <= c <= 1.001) for c in coords[:4]):
                    malformed += 1
                    continue
                if not (0 <= cid < len(names)):
                    report["problems"].append(
                        f"{split}: class id {cid} in {lp.name} is outside names (0..{len(names)-1})"
                    )
                    malformed += 1
                    continue
                counts[cid] += 1
                n_ok += 1
            boxes_per_image.append(n_ok)

        per_class = {names[i]: counts.get(i, 0) for i in range(len(names))}
        report["splits"][split] = {
            "dir": str(d),
            "images": len(images),
            "labels_missing": missing,
            "labels_empty": empty,
            "malformed_lines": malformed,
            "boxes": sum(counts.values()),
            "boxes_per_image_mean": round(
                sum(boxes_per_image) / len(boxes_per_image), 2
            ) if boxes_per_image else 0,
            "per_class": per_class,
        }

        for cname, n in per_class.items():
            if n == 0:
                report["problems"].append(
                    f"{split}: class '{cname}' has ZERO labels -- either the name list is "
                    f"wrong, or this class cannot be learned"
                )
        nonzero = [v for v in per_class.values() if v > 0]
        if nonzero:
            hi, lo = max(nonzero), min(nonzero)
            if hi / lo > 20:
                report["problems"].append(
                    f"{split}: class imbalance {hi}:{lo} (>20x) -- report per-class AP, "
                    f"never a single mAP, or the rare class hides behind the common one"
                )
        if missing:
            report["problems"].append(f"{split}: {missing} images have no label file")

    if sample_crops and out_dir:
        report["crops"] = _dump_crops(root, data, names, sample_crops, Path(out_dir))
    return report


def _dump_crops(root: Path, data: dict, names: list[str], n: int, out_dir: Path) -> str:
    """Write n real crops per class. Looking at them is the actual verification."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    d = _split_dirs(root, data, "train")
    if d is None:
        return "no train split"
    images = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
    random.shuffle(images)
    want = {i: n for i in range(len(names))}

    for img in images:
        if not any(want.values()):
            break
        lp = _label_path(img)
        if not lp.exists():
            continue
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for ln in lp.read_text().splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if want.get(cid, 0) <= 0:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            cls_dir = out_dir / f"{cid:02d}_{names[cid]}"
            cls_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(cls_dir / f"{img.stem}_{want[cid]}.jpg"), crop)
            want[cid] -= 1
    return str(out_dir)


def format_report(report: dict) -> str:
    lines = [f"dataset: {report['data_yaml']}", f"names:   {report['names']}", ""]
    for split, s in report["splits"].items():
        lines.append(
            f"[{split}] {s['images']} images, {s['boxes']} boxes, "
            f"{s['boxes_per_image_mean']} boxes/img, "
            f"missing={s['labels_missing']} empty={s['labels_empty']} "
            f"malformed={s['malformed_lines']}"
        )
        width = max((len(k) for k in s["per_class"]), default=4)
        total = max(1, s["boxes"])
        for cname, n in sorted(s["per_class"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {cname:<{width}}  {n:>7}  {100*n/total:5.1f}%")
        lines.append("")
    if report["problems"]:
        lines.append("PROBLEMS")
        lines += [f"  ! {p}" for p in report["problems"]]
    else:
        lines.append("no structural problems found")
    lines += [
        "",
        "Counts alone do not prove the names are right. Run with --crops N and",
        "LOOK at data/interim/crops/: if folder '01_goggles' contains gloves,",
        "the class order is shifted and every metric downstream is meaningless.",
    ]
    return "\n".join(lines)
