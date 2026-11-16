"""Split-head decoding (box ltrb in stride units + class probabilities per scale), see KNOWN_ISSUES #42."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labsafe.models.postprocess import assemble_split_outputs, yolo_decode  # noqa: E402


def make_outs(imgsz=64, nc=2, box=(20.0, 24.0, 44.0, 52.0), cls_id=1, score=0.9, scale_i=1):
    """One object (xyxy in input pixels) placed on scale `scale_i`; everything else zero."""
    outs = []
    for i, stride in enumerate((8, 16, 32)):
        h = imgsz // stride
        b = np.zeros((1, 4, h, h), np.float32)
        c = np.zeros((1, nc, h, h), np.float32)
        if i == scale_i:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            gx, gy = int(cx // stride), int(cy // stride)
            ax, ay = (gx + 0.5) * stride, (gy + 0.5) * stride
            b[0, :, gy, gx] = [(ax - box[0]) / stride, (ay - box[1]) / stride, (box[2] - ax) / stride, (box[3] - ay) / stride]
            c[0, cls_id, gy, gx] = score
        outs += [b, c]
    return outs


def test_split_outputs_decode_to_the_original_box():
    outs = make_outs()
    arr = assemble_split_outputs(outs, expect_nc=2)
    assert arr.shape == (1, 6, 8 * 8 + 4 * 4 + 2 * 2)
    dets = yolo_decode(arr, 1.0, (0.0, 0.0), 0.5, 0.5, lambda i: ["person", "lab_coat"][i], expect_nc=2)
    assert len(dets) == 1 and dets[0].cls_id == 1 and dets[0].cls_name == "lab_coat"
    assert [round(v, 4) for v in (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2)] == [20.0, 24.0, 44.0, 52.0]
    assert dets[0].score == pytest.approx(0.9)


def test_object_on_each_scale_decodes_the_same():
    for si in range(3):
        arr = assemble_split_outputs(make_outs(scale_i=si, box=(8.0, 8.0, 56.0, 56.0)), expect_nc=2)
        d = yolo_decode(arr, 1.0, (0.0, 0.0), 0.5, 0.5, str, expect_nc=2)
        assert len(d) == 1 and (d[0].x1, d[0].y1, d[0].x2, d[0].y2) == (8.0, 8.0, 56.0, 56.0)


def test_wrong_channel_counts_are_rejected_not_guessed():
    outs = make_outs()
    with pytest.raises(ValueError, match="expected 4"):
        assemble_split_outputs([np.zeros((1, 64, 8, 8), np.float32)] + outs[1:], expect_nc=2)   # a DFL head
    with pytest.raises(ValueError, match="class"):
        assemble_split_outputs(outs, expect_nc=80)
    with pytest.raises(ValueError, match="pairs"):
        assemble_split_outputs(outs[:5], expect_nc=2)
