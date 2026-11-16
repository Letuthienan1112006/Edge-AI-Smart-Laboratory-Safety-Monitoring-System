"""Zones that only forbid people MISSING something (zone `requires: [lab_coat]`).

The rule the user asked for: walking into the chemical area in a lab coat is normal, without one it is a violation.
Stage A is a 2-class model (person, lab_coat); coat boxes must be attached to people, never tracked as people.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.backends import Detection  # noqa: E402
from labsafe.pipeline.fall import FallConfig  # noqa: E402
from labsafe.pipeline.router import AdaptiveRouter  # noqa: E402
from labsafe.pipeline.zones import Zone, load_zones  # noqa: E402

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
# bottom half of the frame; a standing person's foot (y2) at 400 is inside
ZONE_COAT = Zone("chem", "high", [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)], ("lab_coat",))
ZONE_ANY = Zone("cabinet", "high", [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)])


def person(x1, y1, x2, y2):
    return Detection(x1, y1, x2, y2, 0.9, 0, "person")


def coat(x1, y1, x2, y2):
    return Detection(x1, y1, x2, y2, 0.8, 1, "lab_coat")


class ScriptedA:
    """Stage A stand-in that returns the detections it is told to, frame by frame."""

    name = "scripted"
    imgsz = 640
    last_latency_ms = 1.0

    def __init__(self):
        self.next: list[Detection] = []

    def __call__(self, frame):
        return list(self.next)


def make(zones, memory_s=2.0, min_conf=0.0):
    a = ScriptedA()
    r = AdaptiveRouter(a, None, None, zones, FallConfig(), mode="adaptive", attire_memory_s=memory_s,
                       attire_min_conf=min_conf)
    return a, r


def kinds(events):
    return [(e.kind, e.detail) for e in events]


def test_person_with_coat_in_coat_zone_is_allowed():
    a, r = make([ZONE_COAT])
    a.next = [person(100, 100, 200, 400), coat(105, 150, 195, 330)]
    events, _ = r.process(FRAME, now=0.0)
    assert events == []
    assert len(r.tracks) == 1, "the coat box must not become a second person"


def test_person_without_coat_in_coat_zone_is_violation():
    a, r = make([ZONE_COAT])
    a.next = [person(100, 100, 200, 400)]
    events, _ = r.process(FRAME, now=0.0)
    assert kinds(events) == [("zone_ppe_violation", "chem")]
    assert events[0].extra["missing"] == ["lab_coat"]


def test_person_outside_zone_without_coat_is_fine():
    a, r = make([ZONE_COAT])
    a.next = [person(100, 10, 200, 200)]          # foot y=200 is above the zone
    events, _ = r.process(FRAME, now=0.0)
    assert events == []


def test_coat_memory_bridges_short_dropouts_only():
    a, r = make([ZONE_COAT], memory_s=2.0)
    a.next = [person(100, 100, 200, 400), coat(105, 150, 195, 330)]
    r.process(FRAME, now=0.0)
    a.next = [person(101, 100, 201, 400)]          # coat not detected for a moment
    events, _ = r.process(FRAME, now=1.5)
    assert events == [], "1.5 s without a coat box is inside the 2 s memory"
    events, _ = r.process(FRAME, now=3.0)
    assert kinds(events) == [("zone_ppe_violation", "chem")]


def test_one_coat_is_given_to_one_person_only():
    """Two people side by side, only the left one wears a coat: the right one must still be flagged."""
    a, r = make([ZONE_COAT])
    a.next = [person(100, 100, 220, 400), person(200, 100, 320, 400), coat(105, 150, 205, 330)]
    events, _ = r.process(FRAME, now=0.0)
    flagged = {e.track_id for e in events}
    by_x = sorted(r.tracks, key=lambda t: t.det.x1)
    assert flagged == {by_x[1].track_id}


def test_coat_outside_every_person_is_ignored():
    a, r = make([ZONE_COAT])
    a.next = [person(100, 100, 200, 400), coat(400, 150, 500, 330)]   # a coat hanging elsewhere
    events, _ = r.process(FRAME, now=0.0)
    assert kinds(events) == [("zone_ppe_violation", "chem")]


def test_zone_without_requires_still_forbids_everyone():
    a, r = make([ZONE_ANY])
    a.next = [person(100, 100, 200, 400), coat(105, 150, 195, 330)]
    events, _ = r.process(FRAME, now=0.0)
    assert kinds(events) == [("zone_intrusion", "cabinet")]


def test_coat_is_never_a_person_even_without_coat_zones():
    a, r = make([ZONE_ANY])
    a.next = [person(100, 10, 200, 200), coat(105, 50, 195, 150)]
    r.process(FRAME, now=0.0)
    assert len(r.tracks) == 1


def test_violation_alarms_only_after_debounce():
    a, r = make([ZONE_COAT])
    a.next = [person(100, 100, 200, 400)]
    _, alarming = r.process(FRAME, now=0.0)
    assert alarming == []
    _, alarming = r.process(FRAME, now=3.5)
    assert kinds(alarming) == [("zone_ppe_violation", "chem")]


def test_load_zones_reads_requires():
    zs = load_zones({"zones": [
        {"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]], "requires": "lab_coat"},
        {"name": "b", "polygon": [[0, 0], [1, 0], [1, 1]], "requires": ["lab_coat", "gloves"]},
        {"name": "c", "polygon": [[0, 0], [1, 0], [1, 1]]},
    ]})
    assert [z.requires for z in zs] == [("lab_coat",), ("lab_coat", "gloves"), ()]


def test_person_cut_off_by_bottom_of_frame_is_inside_a_zone_reaching_the_edge():
    """Found on real frames: people close to the camera have y2 == frame height and were never judged."""
    a, r = make([ZONE_COAT])
    a.next = [person(100, 100, 200, 480)]
    events, _ = r.process(FRAME, now=0.0)
    assert kinds(events) == [("zone_ppe_violation", "chem")]


def test_unsure_coat_does_not_let_a_person_through():
    a, r = make([ZONE_COAT], min_conf=0.6)
    a.next = [person(100, 100, 200, 400), Detection(105, 150, 195, 330, 0.45, 1, "lab_coat")]
    events, _ = r.process(FRAME, now=0.0)
    assert kinds(events) == [("zone_ppe_violation", "chem")]
    a.next = [person(100, 100, 200, 400), Detection(105, 150, 195, 330, 0.85, 1, "lab_coat")]
    events, _ = r.process(FRAME, now=0.1)
    assert events == []


def test_coat_score_is_exposed_even_below_the_attire_threshold():
    """The dashboard must be able to say 'coat seen at 0.45, rule needs 0.60' - otherwise a coat that is never accepted looks like no coat at all."""
    import numpy as np
    from labsafe.models.backends import Detection
    from labsafe.pipeline.fall import FallConfig
    from labsafe.pipeline.router import AdaptiveRouter

    class Scripted:
        name, last_latency_ms, last_timing = "s", 0.0, {}

        def __call__(self, frame):
            return [Detection(200, 100, 400, 460, 0.9, 0, "person"), Detection(220, 200, 380, 400, 0.45, 1, "lab_coat")]

    r = AdaptiveRouter(Scripted(), None, None, [ZONE_COAT], FallConfig(), mode="adaptive", attire_min_conf=0.6)
    ev, _ = r.process(np.zeros((480, 640, 3), np.uint8), now=0.0)
    p = r.people_summary(ev)["people"][0]
    assert p["coat_score"] == 0.45 and p["wearing"] is False     # seen, but below the rule's threshold -> still a violation


def test_head_only_box_at_the_bottom_edge_is_not_zone_checked_when_partial_min_h_is_set():
    import numpy as np
    from labsafe.models.backends import Detection
    from labsafe.pipeline.fall import FallConfig
    from labsafe.pipeline.router import AdaptiveRouter

    class Head:
        name, last_latency_ms, last_timing = "s", 0.0, {}

        def __call__(self, frame):      # a head at the bottom-right corner: the box touches the bottom edge and is only 90 px tall
            return [Detection(500, 390, 600, 480, 0.8, 0, "person")]

    def run(min_h):
        r = AdaptiveRouter(Head(), None, None, [ZONE_COAT], FallConfig(), mode="adaptive", attire_min_conf=0.6, partial_min_h=min_h)
        ev, _ = r.process(np.zeros((480, 640, 3), np.uint8), now=0.0)
        return [e.kind for e in ev]

    assert run(0.0) == ["zone_ppe_violation"]      # off: the clamped foot point lands inside the zone (current behaviour)
    assert run(0.3) == []                          # on: 90 px < 30% of 480 -> a partial person, not a zone violator


def _router_with(dets, vis):
    import numpy as np
    from labsafe.pipeline.fall import FallConfig
    from labsafe.pipeline.router import AdaptiveRouter

    class S:
        name, last_latency_ms, last_timing = "s", 0.0, {}

        def __call__(self, frame):
            return dets

    r = AdaptiveRouter(S(), None, None, [ZONE_COAT], FallConfig(), mode="adaptive", attire_min_conf=0.6, visibility=vis)
    ev, _ = r.process(np.zeros((480, 640, 3), np.uint8), now=0.0)
    return r, ev


VIS = {"min_h": 0.22, "tall_aspect": 2.0, "min_w_px": 24, "max_occlusion": 0.4}


def test_a_face_only_box_in_the_zone_is_not_a_violation_when_visibility_is_on():
    from labsafe.models.backends import Detection
    face = [Detection(300, 300, 345, 350, 0.7, 0, "person")]            # 45x50 px: a face, no clothes visible
    assert [e.kind for e in _router_with(face, None)[1]] == ["zone_ppe_violation"]   # old behaviour
    r, ev = _router_with(face, VIS)
    assert ev == [] and r.people_summary(ev)["unknown"] == 1 and r.people_summary(ev)["not_wearing"] == 0


def test_a_far_full_body_and_a_visible_person_are_still_judged():
    from labsafe.models.backends import Detection
    far = [Detection(300, 260, 330, 360, 0.8, 0, "person")]             # 30x100: small but tall-shaped = a far full body
    assert [e.kind for e in _router_with(far, VIS)[1]] == ["zone_ppe_violation"]
    near = [Detection(250, 150, 380, 470, 0.9, 0, "person")]
    assert [e.kind for e in _router_with(near, VIS)[1]] == ["zone_ppe_violation"]


def test_a_person_mostly_hidden_behind_someone_closer_gets_no_verdict():
    from labsafe.models.backends import Detection
    behind = Detection(260, 250, 360, 420, 0.8, 0, "person")            # feet at 420
    front = Detection(240, 260, 380, 475, 0.9, 0, "person")             # feet lower = closer, covers most of `behind`
    coat = Detection(250, 300, 370, 460, 0.9, 1, "lab_coat")            # the front person wears a coat
    r, ev = _router_with([behind, front, coat], VIS)
    assert ev == []                                                     # front wears a coat, behind is unknown


def test_a_coat_does_not_follow_an_id_onto_a_different_person():
    """Regression (#51): a track re-matched only by centre distance (no box overlap) may be another person; the coat memory of
    the previous wearer must not make them 'wearing'."""
    from labsafe.models.backends import Detection
    from labsafe.pipeline.tracker import IOUTracker
    tk = IOUTracker(iou_threshold=0.3, max_age_s=3.0)
    t = tk.update([Detection(100, 100, 200, 400, 0.9, 0, "person")], now=0.0)[0]
    t.attire_seen["lab_coat"] = 0.0
    t2 = tk.update([Detection(205, 100, 305, 400, 0.9, 0, "person")], now=0.1)   # no overlap, close enough for the centre fallback
    assert t2 and t2[0].track_id == t.track_id and "lab_coat" not in t2[0].attire_seen
    t3 = tk.update([Detection(210, 100, 310, 400, 0.9, 0, "person")], now=0.2)   # normal IoU match keeps memory semantics
    assert t3[0].track_id == t.track_id
