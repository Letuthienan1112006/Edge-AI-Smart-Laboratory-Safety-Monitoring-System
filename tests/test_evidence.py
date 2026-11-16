"""Photos of the person behind an alarm: right person, raw pixels, numbered files, disk cap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.io.evidence import ViolationRecorder, crop_person  # noqa: E402
from labsafe.models.backends import Detection  # noqa: E402
from labsafe.pipeline.decision import SafetyEvent  # noqa: E402
from labsafe.pipeline.tracker import Track  # noqa: E402


def track(tid, x1, y1, x2, y2):
    return Track(tid, Detection(x1, y1, x2, y2, 0.9, 0, "person"), 0.0, 0.0)


def test_crop_is_the_person_with_margin_and_clipped():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[100:300, 200:300] = 255
    c = crop_person(frame, Detection(200, 100, 300, 300, 0.9, 0, "person"), padding=0.1)
    assert c.shape[:2] == (240, 120)                     # 200x100 box + 10% each side
    edge = crop_person(frame, Detection(600, 400, 640, 480, 0.9, 0, "person"), padding=0.5)
    assert edge.shape[0] > 0 and edge.shape[1] > 0       # clipped, never negative


def test_records_only_person_alarms_and_aligns_output(tmp_path):
    rec = ViolationRecorder(tmp_path)
    raw = np.full((480, 640, 3), 40, np.uint8)
    raw[100:400, 100:200] = (0, 0, 255)                  # the violator, red in the RAW frame
    tracks = [track(5, 100, 100, 200, 400)]
    events = [SafetyEvent("zone_ppe_violation", "high", 5, detail="san_phong"),
              SafetyEvent("fire", "critical"),                                   # not about a person
              SafetyEvent("zone_ppe_violation", "high", 9, detail="san_phong")]  # track already gone
    out = rec.record(raw, raw.copy(), events, tracks)
    assert [o is not None for o in out] == [True, False, False]
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["000001_zone_ppe_violation_p5_crop.jpg", "000001_zone_ppe_violation_p5_frame.jpg",
                     "violations.jsonl"]
    crop = cv2.imread(str(tmp_path / files[0]))
    assert crop[crop.shape[0] // 2, crop.shape[1] // 2][2] > 200   # centre of the crop is the red person
    line = json.loads((tmp_path / "violations.jsonl").read_text().splitlines()[0])
    assert line["track_id"] == 5 and line["detail"] == "san_phong"


def test_numbering_continues_after_restart_and_cap_removes_oldest(tmp_path):
    raw = np.zeros((480, 640, 3), np.uint8)
    tr = [track(1, 10, 10, 100, 200)]
    ev = [SafetyEvent("zone_ppe_violation", "high", 1)]
    rec = ViolationRecorder(tmp_path, max_files=3)
    for _ in range(2):
        rec.record(raw, raw, ev, tr)
    rec2 = ViolationRecorder(tmp_path, max_files=3)       # app restarted
    for _ in range(3):
        rec2.record(raw, raw, ev, tr)
    crops = sorted(p.name for p in tmp_path.glob("*_crop.jpg"))
    assert crops == [f"{n:06d}_zone_ppe_violation_p1_crop.jpg" for n in (3, 4, 5)]
    assert sorted(p.name for p in tmp_path.glob("*_frame.jpg")) == [c.replace("_crop", "_frame") for c in crops]
