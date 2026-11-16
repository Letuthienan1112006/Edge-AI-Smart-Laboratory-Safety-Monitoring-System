"""Letterbox + YOLO head decoding + NMS, in numpy only.

Kept dependency-free on purpose: the Pi runs this, and pulling torch onto
the Pi just to decode boxes would distort every power measurement.
"""

from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np


def letterbox(
    frame: np.ndarray,
    imgsz: int,
    nchw: bool = True,
    pad_value: int = 114,
    normalize: bool = True,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio, pad to square. Returns (blob, scale, (padx, pady)).

    `normalize=False` returns raw uint8 NHWC. NPU runtimes (RKNN) fold the
    mean/std division into the model at conversion time, so normalising here
    as well would scale the input twice -- which does not crash, it just
    makes the model detect nothing.
    """
    h, w = frame.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    padx, pady = (imgsz - nw) / 2.0, (imgsz - nh) / 2.0
    top, left = int(round(pady - 0.1)), int(round(padx - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if not normalize:
        return rgb[None, ...], scale, (padx, pady)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1) if nchw else blob
    return blob[None, ...], scale, (padx, pady)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Plain greedy NMS over xyxy boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def assemble_split_outputs(outs, expect_nc: int | None = None) -> np.ndarray:
    """Six raw head tensors -> one (1, 4+nc, N) array in letterbox pixels (cx, cy, w, h) + class scores.

    The "split" export keeps the head's per-scale outputs un-concatenated and un-decoded, in the order
    box_P3, cls_P3, box_P4, cls_P4, box_P5, cls_P5:
      box_i : (1, 4, H_i, W_i)  distances left/top/right/bottom from the cell centre, in units of the stride
      cls_i : (1, nc, H_i, W_i) class probabilities (sigmoid already applied in the graph)
    Why: with one concatenated decoded tensor an INT8 NPU must share ONE 8-bit scale between box pixels (0..~690) and
    class scores (0..1), so a box edge is only known to ~2.5 px (INT8 mAP50-95 collapses by ~19 points, KNOWN_ISSUES #42).
    Raw distances have a small range per scale, so their 8-bit step is a fraction of a pixel; the cheap decode
    (anchor +- distance) is done here in float.
    """
    if len(outs) % 2 or len(outs) < 2:
        raise ValueError(f"split head needs box/cls pairs, got {len(outs)} tensors")
    boxes, scores = [], []
    for i in range(0, len(outs), 2):
        b = np.asarray(outs[i], dtype=np.float32)
        c = np.asarray(outs[i + 1], dtype=np.float32)
        b, c = b.reshape(b.shape[-3:]), c.reshape(c.shape[-3:])
        if b.shape[0] != 4:
            raise ValueError(f"box tensor {i} has {b.shape[0]} channels, expected 4 (ltrb); a DFL head (64) is not supported")
        if expect_nc is not None and c.shape[0] != expect_nc:
            raise ValueError(f"cls tensor {i + 1} has {c.shape[0]} channels, config declares {expect_nc} class(es)")
        h, w = b.shape[1:]
        boxes.append(b)
        scores.append(c)
    # stride of each scale: the largest map is the smallest stride; the input is square, so stride = imgsz / H, and imgsz
    # is taken from the P3 map (stride 8).
    imgsz = boxes[0].shape[1] * 8.0
    outb, outc = [], []
    for b, c in zip(boxes, scores):
        h, w = b.shape[1:]
        stride = imgsz / h
        gx = np.broadcast_to(np.arange(w, dtype=np.float32) + 0.5, (h, w))
        gy = np.broadcast_to((np.arange(h, dtype=np.float32) + 0.5)[:, None], (h, w))
        x1, y1 = (gx - b[0]) * stride, (gy - b[1]) * stride
        x2, y2 = (gx + b[2]) * stride, (gy + b[3]) * stride
        outb.append(np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]).reshape(4, -1))
        outc.append(c.reshape(c.shape[0], -1))
    return np.concatenate([np.concatenate(outb, axis=1), np.concatenate(outc, axis=1)], axis=0)[None]


def yolo_decode(
    raw: np.ndarray,
    scale: float,
    pad: tuple[float, float],
    conf_thr: float,
    iou_thr: float,
    name_of: Callable[[int], str],
    expect_nc: int | None = None,
    box_scale: float = 1.0,
    class_subset: Sequence[int] | None = None,
):
    """Decode an anchor-free YOLOv8/YOLO26-style head: (1, 4+nc, N) or (1, N, 4+nc).

    `box_scale` multiplies the 4 box channels back to letterbox pixels. It is
    1.0 for any output already in pixels, and `imgsz` for an RKNN INT8 export,
    whose box coordinates Ultralytics normalises by the input size
    (ModelCard.box_coords == "normalized", KNOWN_ISSUES.md #24).

    This accepts output that is ALREADY through DFL and already decoded to
    per-pixel cxcywh in letterbox-image coordinates -- i.e. Ultralytics
    export with `end2end=False`. It does NOT accept `rknn_model_zoo`-style
    raw per-stride branches (pre-DFL); feeding those in used to silently
    produce garbage boxes instead of an error (KNOWN_ISSUES.md #5), which
    is exactly what the shape guards below exist to stop.

    `expect_nc`, when given, catches a mismatched checkpoint: a model
    exported with a different class count than the running config still
    produces a plausible-looking array, and `argmax` over the wrong number
    of classes fails silently -- wrong labels, not an error.
    """
    from .backends import Detection

    if raw.ndim == 3:
        arr = np.squeeze(raw, axis=0)
    else:
        arr = raw
    if arr.ndim != 2:
        raise ValueError(
            f"yolo_decode expected a (1, C, N)/(1, N, C) or (C, N)/(N, C) array, "
            f"got shape {raw.shape} (ndim={raw.ndim} after squeeze). This usually "
            f"means the model was exported with multiple output branches "
            f"(rknn_model_zoo style) rather than Ultralytics end2end=False -- "
            f"see KNOWN_ISSUES.md #5."
        )
    # Orient to (N, 4+nc): the attribute axis is the short one.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 5:
        raise ValueError(
            f"yolo_decode got {arr.shape[1]} attribute channels per box after "
            f"orienting to (N, C); need at least 5 (4 box coords + >=1 class). "
            f"Shape was {raw.shape} before orienting."
        )
    if expect_nc is not None and (arr.shape[1] - 4) != expect_nc:
        raise ValueError(
            f"model output has {arr.shape[1] - 4} class channels but the running "
            f"config declares {expect_nc} class(es) -- this is almost always a "
            f"converted checkpoint that does not match configs/classes.yaml. "
            f"Shape was {raw.shape}."
        )
    boxes_cxcywh, cls_scores = arr[:, :4], arr[:, 4:]
    if cls_scores.size == 0:
        return []
    if class_subset is not None:
        # Only the classes the caller keeps: on a COCO-80 head with
        # keep_classes=[person] this turns an argmax over 8400x80 into a read
        # of one column (measured 0.9 ms float32 / 3.9 ms float16 -> 0.02 ms;
        # ~10% of the INT8 frame budget on the board). Semantics differ
        # slightly from "argmax over all classes, then filter": a box whose
        # best class is a car but whose person score is >= conf is now kept
        # (and NMS'd) as a person -- the right behaviour for a person detector.
        sub = np.asarray(class_subset, dtype=np.intp)
        if sub.size == 0 or sub.min() < 0 or sub.max() >= cls_scores.shape[1]:
            raise ValueError(
                f"class_subset {list(class_subset)} out of range for {cls_scores.shape[1]} class channels"
            )
        cls_scores = cls_scores[:, sub]
    cls_ids = cls_scores.argmax(axis=1) if cls_scores.shape[1] > 1 else np.zeros(cls_scores.shape[0], dtype=np.intp)
    scores = cls_scores.max(axis=1) if cls_scores.shape[1] > 1 else cls_scores[:, 0]
    if class_subset is not None:
        cls_ids = sub[cls_ids]
    keep_mask = scores >= conf_thr
    if not keep_mask.any():
        return []
    boxes_cxcywh, scores, cls_ids = (
        boxes_cxcywh[keep_mask],
        scores[keep_mask],
        cls_ids[keep_mask],
    )
    cx, cy, w, h = boxes_cxcywh.T * box_scale
    padx, pady = pad
    xyxy = np.stack(
        [
            (cx - w / 2 - padx) / scale,
            (cy - h / 2 - pady) / scale,
            (cx + w / 2 - padx) / scale,
            (cy + h / 2 - pady) / scale,
        ],
        axis=1,
    )
    out = []
    for cid in np.unique(cls_ids):
        sel = np.where(cls_ids == cid)[0]
        for k in nms(xyxy[sel], scores[sel], iou_thr):
            i = int(sel[k])
            out.append(
                Detection(
                    *(float(v) for v in xyxy[i]),
                    score=float(scores[i]),
                    cls_id=int(cid),
                    cls_name=name_of(int(cid)),
                )
            )
    return out
