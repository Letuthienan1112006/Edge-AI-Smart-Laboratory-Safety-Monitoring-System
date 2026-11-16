"""The video overlay is what a viewer reads: colour and words must follow the same rule as the alarms."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.backends import Detection  # noqa: E402
from labsafe.pipeline.decision import SafetyEvent  # noqa: E402
from labsafe.pipeline.tracker import Track  # noqa: E402
from labsafe.pipeline.zones import Zone  # noqa: E402
from labsafe.viz import COLOR, ascii_fold, draw, person_label  # noqa: E402


def track(i, seen=None):
    t = Track(track_id=i, det=Detection(100, 100, 200, 400, 0.9, 0, "person"), first_seen=0.0, last_seen=10.0)
    if seen is not None:
        t.attire_seen["lab_coat"] = seen
    return t


def test_labels_follow_violation_then_coat_state():
    assert person_label(track(1), True, False, None) == ("#1 VI PHẠM · không áo lab", "crit")
    assert person_label(track(2), False, True, 0.912) == ("#2 Có áo lab 0,91", "ok")
    assert person_label(track(3), False, False, None) == ("#3 Không áo lab", "warn")   # no coat but outside the zone: amber, not green


def test_box_colour_uses_the_rules_coat_memory():
    """Coat last seen 2.5 s ago: with the rule's 3 s memory the box is green; with a hard-coded 2 s it turned amber (old bug)."""
    frame = np.zeros((480, 640, 3), np.uint8)
    t = track(4, seen=7.5)
    img3 = draw(frame, [t], [], [], memory_s=3.0)
    img2 = draw(frame, [t], [], [], memory_s=2.0)
    assert tuple(int(c) for c in img3[250, 100]) == COLOR["ok"]
    assert tuple(int(c) for c in img2[250, 100]) == COLOR["warn"]


def test_violation_box_is_red():
    frame = np.zeros((480, 640, 3), np.uint8)
    z = Zone("san_phong", "high", [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)], ("lab_coat",))
    img = draw(frame, [track(5)], [z], [SafetyEvent("zone_ppe_violation", "high", 5, "san_phong")])
    assert tuple(int(c) for c in img[250, 100]) == COLOR["crit"] or tuple(int(c) for c in img[250, 101]) == COLOR["crit"]


def test_ascii_fallback_strips_vietnamese_marks():
    assert ascii_fold("VI PHẠM · không áo lab · Đỗ") == "VI PHAM - khong ao lab - Do"


def test_a_person_whose_clothes_cannot_be_seen_gets_no_verdict():
    t = track(6)
    t.attire_unknown = True
    assert person_label(t, False, False, None) == ("#6 Chưa thấy rõ áo", "unk")
