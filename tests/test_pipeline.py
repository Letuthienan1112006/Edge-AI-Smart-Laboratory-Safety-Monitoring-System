"""Logic tests that need neither a camera nor a trained model.

These cover the parts where a silent bug would corrupt the paper's
numbers rather than crash: gating counts, debounce windows, the fall
state machine, and zone geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.backends import Detection, build_backend  # noqa: E402
from labsafe.pipeline.decision import Debouncer, SafetyEvent, worst_severity  # noqa: E402
from labsafe.pipeline.fall import FallConfig, FallDetector  # noqa: E402
from labsafe.pipeline.router import AdaptiveRouter  # noqa: E402
from labsafe.pipeline.tracker import IOUTracker  # noqa: E402
from labsafe.pipeline.zones import Zone, load_zones, point_in_polygon  # noqa: E402


def det(x1, y1, x2, y2, name="person", score=0.9):
    return Detection(x1, y1, x2, y2, score, 0, name)


# -- zones ---------------------------------------------------------------

def test_point_in_polygon_square():
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon(0.5, 0.5, sq)
    assert not point_in_polygon(1.5, 0.5, sq)


def test_zone_uses_foot_not_centre():
    """A person standing below a high zone must not count as inside it."""
    z = Zone("upper", "high", [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)])
    standing = det(100, 100, 200, 400)          # centre y=250 (in), foot y=400 (out)
    assert not z.contains_foot(standing, 640, 480)


def test_load_zones_rejects_pixel_coordinates():
    with pytest.raises(ValueError, match="0..1"):
        load_zones({"zones": [{"name": "z", "polygon": [[0, 0], [640, 0], [640, 480]]}]})


def test_load_zones_rejects_a_self_intersecting_polygon():
    with pytest.raises(ValueError, match="self-intersects"):
        load_zones({"zones": [{"name": "z", "polygon": [[0, 0], [1, 1], [0, 1], [1, 0]]}]})


# -- tracker -------------------------------------------------------------

def test_tracker_keeps_identity_across_frames():
    tr = IOUTracker(iou_threshold=0.3, max_age_s=1.0)
    a = tr.update([det(100, 100, 200, 300)], now=0.0)
    b = tr.update([det(110, 100, 210, 300)], now=0.1)
    assert a[0].track_id == b[0].track_id


def test_tracker_expires_stale_tracks():
    tr = IOUTracker(max_age_s=0.5)
    tr.update([det(100, 100, 200, 300)], now=0.0)
    assert tr.update([], now=2.0) == []
    assert tr.tracks == {}


def test_two_tracks_cannot_claim_one_box():
    tr = IOUTracker(iou_threshold=0.1)
    tr.update([det(0, 0, 100, 200), det(300, 0, 400, 200)], now=0.0)
    out = tr.update([det(10, 0, 110, 200)], now=0.1)
    assert len(out) == 1


# -- fall ----------------------------------------------------------------

def _fall_sequence(fd, tracker, frame_h=480):
    """Upright, then a fast drop to a wide box that stays put."""
    tracker.update([det(200, 50, 260, 400)], now=0.0)
    fd.update(list(tracker.tracks.values()), frame_h, now=0.0)
    for i, t in enumerate([0.1, 0.2, 0.3]):
        tracker.update([det(200, 50 + i * 20, 260, 400)], now=t)
    down = det(150, 380, 330, 440)              # wide, centre dropped
    events = []
    for t in (0.5, 1.0, 2.0, 3.0, 4.0):
        tracks = tracker.update([down], now=t)
        events += fd.update(tracks, frame_h, now=t)
    return events


def test_fall_requires_staying_down():
    fd = FallDetector(FallConfig(still_duration_s=2.5, fps_estimate=10))
    events = _fall_sequence(fd, IOUTracker(iou_threshold=0.1))
    assert events, "a genuine fall that stays down must alarm"


def test_crouch_and_stand_is_not_a_fall():
    fd = FallDetector(FallConfig(still_duration_s=2.5, fps_estimate=10))
    tracker = IOUTracker(iou_threshold=0.1)
    tracker.update([det(200, 50, 260, 400)], now=0.0)
    fd.update(list(tracker.tracks.values()), 480, now=0.0)
    wide = det(150, 380, 330, 440)
    for t in (0.5, 1.0):                        # briefly wide...
        fd.update(tracker.update([wide], now=t), 480, now=t)
    tall = det(200, 50, 260, 400)               # ...then upright again
    events = []
    for t in (1.5, 2.0, 3.0, 4.0):
        events += fd.update(tracker.update([tall], now=t), 480, now=t)
    assert not events, "standing back up within the window is not a fall"


# -- debounce ------------------------------------------------------------

def test_debounce_suppresses_single_frame_events():
    db = Debouncer(debounce_s=3.0, cooldown_s=10.0)
    ev = [SafetyEvent("fall", "critical", 1, "x")]
    assert db.filter(ev, now=0.0) == []
    assert db.filter(ev, now=1.0) == []
    assert len(db.filter(ev, now=3.1)) == 1


def test_debounce_window_restarts_after_a_gap():
    """A flicker must not accumulate credit across minutes."""
    db = Debouncer(debounce_s=3.0, cooldown_s=10.0)
    ev = [SafetyEvent("fall", "critical", 1, "x")]
    db.filter(ev, now=0.0)
    db.filter([], now=1.0)                      # event disappeared
    assert db.filter(ev, now=2.0) == []         # window restarted, not 2s in
    assert db.filter(ev, now=4.0) == []
    assert len(db.filter(ev, now=5.1)) == 1


def test_grace_keeps_credit_over_a_short_dropout_but_not_a_long_one():
    ev = [SafetyEvent("zone_ppe_violation", "high", 1, "z")]
    db = Debouncer(debounce_s=3.0, cooldown_s=10.0, grace_s=1.0)
    db.filter(ev, now=0.0)
    db.filter([], now=0.8)                      # a dropout shorter than the grace
    db.filter(ev, now=1.2)
    assert len(db.filter(ev, now=3.1)) == 1     # credit from t=0 survived
    db2 = Debouncer(debounce_s=3.0, cooldown_s=10.0, grace_s=1.0)
    db2.filter(ev, now=0.0)
    db2.filter([], now=0.5)
    db2.filter([], now=1.6)                     # absent for more than the grace: window restarts
    assert db2.filter(ev, now=2.0) == []
    assert db2.filter(ev, now=4.0) == []
    assert len(db2.filter(ev, now=5.1)) == 1


def test_cooldown_blocks_repeat_alarms():
    db = Debouncer(debounce_s=0.0, cooldown_s=10.0)
    ev = [SafetyEvent("zone_intrusion", "high", 2, "cabinet")]
    assert len(db.filter(ev, now=0.0)) == 1
    assert db.filter(ev, now=5.0) == []
    assert len(db.filter(ev, now=11.0)) == 1


def test_worst_severity():
    assert worst_severity([]) == "none"
    assert worst_severity([
        SafetyEvent("a", "low"), SafetyEvent("b", "critical"), SafetyEvent("c", "medium")
    ]) == "critical"


# -- the gating claim ----------------------------------------------------

def _router(mode, stage_c_interval=1.0):
    from labsafe.pipeline.ppe import PPEChecker, PPEConfig

    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 0.3}, ["person"])
    b = PPEChecker(
        build_backend({"backend": "stub", "imgsz": 64, "conf": 0.3}, ["goggles"]),
        PPEConfig(required=["goggles"], recheck_interval_s=2.0),
    )
    c = build_backend({"backend": "stub", "imgsz": 64, "conf": 0.3}, ["none", "smoke", "fire"])
    return AdaptiveRouter(a, b, c, [], FallConfig(), mode=mode,
                          stage_c_interval_s=stage_c_interval)


def test_adaptive_calls_stage_b_far_less_than_parallel():
    """The core quantitative claim of the design. If this ever fails, the
    energy numbers in the paper are not explained by gating."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    adaptive, parallel = _router("adaptive"), _router("parallel")
    for i in range(60):
        t = i * 0.05                            # 3s at 20 fps
        adaptive.process(frame, now=t)
        parallel.process(frame, now=t)
    assert adaptive.stats.stage_calls["b"] < parallel.stats.stage_calls["b"]
    assert adaptive.stats.stage_calls["c"] < parallel.stats.stage_calls["c"]


def test_stage_c_respects_its_cadence():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = _router("adaptive", stage_c_interval=1.0)
    for i in range(100):
        r.process(frame, now=i * 0.05)          # 5 seconds of frames
    assert r.stats.stage_calls["c"] <= 6, "1 Hz cadence must not run every frame"


def test_occupancy_is_recorded():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = _router("adaptive")
    for i in range(10):
        r.process(frame, now=i * 0.05)
    assert 0.0 <= r.stats.occupancy <= 1.0
    assert r.stats.summary()["frames"] == 10


# -- telemetry: a live call count is not the same as a live model --------

def test_router_counts_detections_not_just_calls():
    """A stub always returns exactly one detection per call -- so after 10
    frames, stage A's detection count must equal its call count, not just
    be nonzero."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = _router("parallel")
    for i in range(10):
        r.process(frame, now=i * 0.05)
    assert r.stats.stage_calls["a"] == 10
    assert r.stats.stage_detections["a"] == 10


def test_liveness_warning_fires_on_a_dead_stage_a():
    """A stage A backend that always returns [] (e.g. a model that
    converted broken -- ultralytics#23753) must trip the warning after
    enough calls, and the warning must name the issue so it is
    actionable, not just alarming."""
    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 1.1}, ["person"])
    r = AdaptiveRouter(a, None, None, [], FallConfig(), mode="parallel")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(150):
        r.process(frame, now=i * 0.05)
    warning = r.stats.liveness_warning()
    assert warning is not None
    assert "23753" in warning


def test_liveness_warning_silent_when_model_works():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = _router("parallel")
    for i in range(150):
        r.process(frame, now=i * 0.05)
    assert r.stats.liveness_warning() is None


def test_liveness_warning_silent_before_call_threshold():
    """A dead stage A must not be flagged before 100 calls -- fewer calls
    is not enough evidence to distinguish "the object was not in frame
    yet" from "the model is dead", and crying wolf on every short smoke
    test would train people to ignore the warning."""
    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 1.1}, ["person"])
    r = AdaptiveRouter(a, None, None, [], FallConfig(), mode="parallel")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(99):
        r.process(frame, now=i * 0.05)
    assert r.stats.stage_calls["a"] == 99
    assert r.stats.liveness_warning() is None


def test_blank_rate_is_between_0_and_1():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r = _router("parallel")
    for i in range(30):
        r.process(frame, now=i * 0.05)
    rate = r.stats.summary()["stage_blank_rate"]["a"]
    assert 0.0 <= rate <= 1.0


def test_stage_b_blank_rate_uses_the_same_denominator_as_stage_calls():
    """blank_rate must be blanks/calls, not blanks-per-frame over calls.

    Stage B can run several times in one frame (one per tracked person).
    `stage_calls["b"]` counts each call; the blank counter used to add 1 per
    FRAME. With 5 calls in a frame and zero detections between them the rate
    came out 0.2 when the truth is 1.0 -- a dead PPE stage reading as 80%
    healthy in the file the report is written from.
    """
    import numpy as np

    from labsafe.models.backends import build_backend
    from labsafe.pipeline.fall import FallConfig
    from labsafe.pipeline.ppe import PPEChecker, PPEConfig
    from labsafe.pipeline.router import AdaptiveRouter

    class _NeverDetects(PPEChecker):
        """Stands in for a converted model that runs but finds nothing."""

        CALLS_PER_FRAME = 5      # as if five people were checked this frame

        def check(self, frame, tracks, now=None, force=False):
            # The bug only shows when stage B runs MORE than once per frame.
            # With one call per frame the old and new accounting agree, so a
            # single-track fixture is a test that passes either way.
            self.calls += self.CALLS_PER_FRAME
            self.last_detection_count = 0
            return []

    a = build_backend({"backend": "stub", "imgsz": 64, "conf": 0.3}, ["person"])
    b = _NeverDetects(
        build_backend({"backend": "stub", "imgsz": 32, "conf": 0.3}, ["goggles"]),
        PPEConfig(required=["goggles"], recheck_interval_s=0.0),
    )
    r = AdaptiveRouter(a, b, None, [], FallConfig(), mode="adaptive")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(10):
        r.process(frame, now=i * 0.1)

    summary = r.stats.summary()
    calls = summary["stage_calls"]["b"]
    assert calls == 10 * _NeverDetects.CALLS_PER_FRAME, (
        f"fixture must produce several calls per frame or the bug is "
        f"invisible; got {calls}"
    )
    assert summary["stage_blank_rate"]["b"] == 1.0, (
        f"every one of {calls} calls detected nothing, so the blank rate must "
        f"be 1.0, got {summary['stage_blank_rate']['b']}"
    )


# -- long live runs must not grow without bound (KNOWN_ISSUES #37) -------

def test_live_stats_window_bounds_every_series():
    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 0.3}, ["person"])
    r = AdaptiveRouter(a, None, None, [], FallConfig(), mode="parallel", stats_window=50)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(300):
        r.process(frame, now=i * 0.05)
    assert r.stats.frames == 300                       # counters still count everything
    assert len(r.stats.frame_latency_ms) == 50
    assert all(len(v) <= 50 for v in r.stats.stage_latency_ms.values())
    assert all(len(v) <= 50 for v in r.stats.stage_a_breakdown_ms.values())
    assert r.stats.summary()["frame_latency_ms"]["n"] == 50


def test_bench_stats_keep_every_sample():
    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 0.3}, ["person"])
    r = AdaptiveRouter(a, None, None, [], FallConfig(), mode="parallel")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(300):
        r.process(frame, now=i * 0.05)
    assert len(r.stats.frame_latency_ms) == 300


def test_debouncer_forgets_people_after_their_cooldown():
    d = Debouncer(debounce_s=0.0, cooldown_s=10.0)
    for tid in range(100):                              # 100 different people, each alarmed once
        d.filter([SafetyEvent("zone_ppe_violation", "high", tid, detail="z")], now=float(tid))
    d.filter([], now=500.0)
    assert len(d._last_fired) == 0 and len(d._first_seen) == 0
    # and a person still in violation keeps the cooldown
    ev = SafetyEvent("zone_ppe_violation", "high", 7, detail="z")
    assert d.filter([ev], now=600.0) and not d.filter([ev], now=605.0)


def test_fall_detection_can_be_switched_off_by_config():
    """`fall: {enabled: false}` used to be silently ignored (FallConfig had no such field)."""
    cfg = FallConfig.from_dict({"enabled": False, "aspect_ratio_threshold": 1.15})
    assert cfg.enabled is False
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 0.3}, ["person"])
    r = AdaptiveRouter(a, None, None, [], cfg, mode="adaptive")
    kinds = set()
    for i in range(120):
        ev, _ = r.process(frame, now=i * 0.1)
        kinds |= {e.kind for e in ev}
    assert "fall" not in kinds
