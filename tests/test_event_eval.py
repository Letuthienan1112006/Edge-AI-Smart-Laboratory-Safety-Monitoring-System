import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.bench.event_eval import fall_times, merge, score, zone_intervals  # noqa: E402


def cue(t, typ, who, **kw):
    return {"t": t, "event": {"type": typ, "who": who, **kw}}


def test_intervals_split_at_coat_changes():
    cues = [cue(10, "enter", "B", coat=False), cue(40, "wear", "B"), cue(70, "unwear", "B"), cue(100, "leave", "B")]
    iv = zone_intervals(cues, 120)
    assert [(i.start, i.end, i.kind) for i in iv] == [(10, 40, "violation"), (40, 70, "covered"), (70, 100, "violation")]


def test_open_interval_is_closed_at_the_end_and_other_cue_types_are_ignored():
    iv = zone_intervals([cue(5, "enter", "A", coat=True), cue(8, "sit", "C"), cue(9, "occlude", "A,B")], 60)
    assert [(i.who, i.start, i.end, i.kind) for i in iv] == [("A", 5, 60, "covered")]


def test_score_counts_detected_missed_false_and_duplicate_alarms():
    cues = [cue(10, "enter", "B", coat=False), cue(35, "leave", "B"),     # long violation: must alarm
            cue(55, "enter", "B", coat=False), cue(57, "leave", "B"),     # 2 s crossing: must stay silent
            cue(80, "enter", "B", coat=False), cue(110, "leave", "B"),    # long violation: will be missed
            cue(60, "enter", "A", coat=True), cue(90, "leave", "A")]      # covered: must stay silent
    iv = zone_intervals(cues, 120)
    alarms = [{"t": 14, "kind": "zone_ppe_violation"}, {"t": 20, "kind": "zone_ppe_violation"},   # one event, alarmed twice
              {"t": 70, "kind": "zone_ppe_violation"}]                                            # nobody should alarm here
    s = score(alarms, iv, [], 120, debounce_s=3.0)
    assert (s.expected, s.detected, s.missed, s.false_alarms, s.duplicates) == (2, 1, 1, 1, 1)
    assert s.delays == [4.0]
    assert round(s.row()["fa_per_hour"]) == 30
    m = merge([s, s])
    assert (m.expected, m.false_alarms) == (4, 2)


def test_falls_are_matched_in_a_window_and_extra_ones_are_false():
    cues = [cue(10, "fall", "C"), cue(40, "fall", "C")]
    assert fall_times(cues) == [10.0, 40.0]
    s = score([{"t": 13, "kind": "fall"}, {"t": 25, "kind": "fall"}], [], fall_times(cues), 60, 3.0)
    assert (s.falls_expected, s.falls_detected, s.fall_false_alarms) == (2, 1, 1)


# --- the whole chain on scripted detections: does `grace_s` change the event-level outcome the way it should? ---

def _replay(grace, dropout_every=60, dropout_len=1):
    """30 fps, 40 s. B stands in the zone WITHOUT a coat from t=10 to t=35, but the detector loses them for `dropout_len` frame(s) every
    `dropout_every` frames. Returns the alarms the real router + debouncer raise."""
    import numpy as np
    from labsafe.models.backends import Detection
    from labsafe.pipeline.decision import Debouncer  # noqa: F401  (the router builds its own from alerts_cfg)
    from labsafe.pipeline.fall import FallConfig
    from labsafe.pipeline.router import AdaptiveRouter
    from labsafe.pipeline.zones import Zone

    zone = Zone("floor", "high", [(0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)], ("lab_coat",))

    class Scripted:
        name, last_latency_ms, last_timing = "scripted", 0.0, {}

        def __init__(self):
            self.i = 0

        def __call__(self, frame):
            i, self.i = self.i, self.i + 1
            t = i / 30.0
            lost = (i % dropout_every) < dropout_len
            if 10.0 <= t < 35.0 and not lost:
                return [Detection(250, 150, 390, 460, 0.9, 0, "person")]
            return []

    r = AdaptiveRouter(Scripted(), None, None, [zone], FallConfig(), mode="adaptive",
                       tracker_cfg={"iou_match_threshold": 0.3, "max_age_s": 3.0},
                       alerts_cfg={"debounce_s": 3.0, "cooldown_s": 60.0, "grace_s": grace}, attire_min_conf=0.6)
    frame = np.zeros((480, 640, 3), np.uint8)
    alarms = []
    for i in range(30 * 40):
        _, al = r.process(frame, now=i / 30.0)
        alarms += [{"t": i / 30.0, "kind": e.kind} for e in al]
    return alarms


def test_without_grace_a_dropout_every_two_seconds_means_no_alarm_with_grace_it_alarms_after_about_three_seconds():
    iv = zone_intervals([cue(10, "enter", "B", coat=False), cue(35, "leave", "B")], 40)
    s0 = score(_replay(0.0), iv, [], 40, 3.0)
    s1 = score(_replay(1.0), iv, [], 40, 3.0)
    assert (s0.expected, s0.detected) == (1, 0), "a 1-frame dropout every 2 s must keep restarting the 3 s window when grace_s=0"
    assert (s1.detected, s1.false_alarms) == (1, 0)
    assert 3.0 <= s1.delays[0] <= 4.5
