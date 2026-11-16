"""Event-level scoring of the alarm stream against scripted ground truth (the cues of scripts/data/collect_guided.py).

Frame-level accuracy says little about an alarm system: what matters is whether each real violation raised ONE alarm soon enough, and how
many alarms were raised when nothing was wrong. Ground truth comes from the cues the actors were told at known seconds of the recording.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class Interval:
    who: str
    start: float
    end: float
    kind: str                 # "violation" (in the zone without a coat) | "covered" (in the zone with a coat)

    @property
    def duration(self) -> float:
        return self.end - self.start


def zone_intervals(cues: list[dict], total_s: float) -> list[Interval]:
    """Per person, the stretches spent in the zone, split at coat on/off. enter/leave/wear/unwear cues; others are ignored."""
    state: dict[str, dict] = {}
    out: list[Interval] = []

    def close(who: str, t: float) -> None:
        s = state[who]
        if s["in"] and t > s["since"]:
            out.append(Interval(who, s["since"], t, "covered" if s["coat"] else "violation"))

    for c in sorted(cues, key=lambda c: c["t"]):
        ev = c.get("event") or {}
        typ, who, t = ev.get("type"), ev.get("who"), float(c["t"])
        if typ not in ("enter", "leave", "wear", "unwear") or who is None:
            continue
        s = state.setdefault(who, {"in": False, "coat": False, "since": 0.0})
        if typ == "enter":
            if s["in"]:
                close(who, t)
            s.update({"in": True, "coat": bool(ev.get("coat", False)), "since": t})
        elif typ == "leave":
            close(who, t)
            s["in"] = False
        elif typ in ("wear", "unwear") and s["in"]:
            close(who, t)
            s.update({"coat": typ == "wear", "since": t})
    for who, s in state.items():
        if s["in"]:
            close(who, total_s)
    return out


def fall_times(cues: list[dict]) -> list[float]:
    return sorted(float(c["t"]) for c in cues if (c.get("event") or {}).get("type") == "fall")


@dataclass
class Score:
    expected: int = 0                     # violations that last long enough to be alarmed (>= debounce + margin)
    detected: int = 0
    missed: int = 0
    false_alarms: int = 0                 # zone alarms outside every expected window
    duplicates: int = 0                   # extra alarms inside an already-alarmed window
    delays: list[float] = field(default_factory=list)
    hours: float = 0.0
    falls_expected: int = 0
    falls_detected: int = 0
    fall_false_alarms: int = 0

    def row(self) -> dict:
        return {
            "expected": self.expected, "detected": self.detected, "missed": self.missed,
            "recall": round(self.detected / self.expected, 3) if self.expected else None,
            "false_alarms": self.false_alarms, "fa_per_hour": round(self.false_alarms / self.hours, 1) if self.hours else None,
            "median_delay_s": round(median(self.delays), 2) if self.delays else None,
            "dup_per_event": round(self.duplicates / self.detected, 2) if self.detected else None,
            "falls": f"{self.falls_detected}/{self.falls_expected}", "fall_false_alarms": self.fall_false_alarms,
        }


def score(alarms: list[dict], intervals: list[Interval], falls: list[float], total_s: float, debounce_s: float,
          end_tol_s: float = 2.0, min_margin_s: float = 1.0, fall_window_s: float = 8.0) -> Score:
    """alarms: [{"t": seconds since the start of the recording, "kind": "zone_ppe_violation" | "fall" | ...}].

    An expected window is [violation start, violation end + end_tol_s]; it needs the violation to last at least debounce_s + min_margin_s,
    otherwise a correct system stays silent. Alarms of a zone kind outside every expected window are false alarms.
    Delay is measured from the cue time, so it includes the actor's own reaction (about 1-2 s).
    """
    sc = Score(hours=total_s / 3600.0)
    windows = [(iv.start, iv.end + end_tol_s) for iv in intervals if iv.kind == "violation" and iv.duration >= debounce_s + min_margin_s]
    sc.expected = len(windows)
    zone = sorted(a["t"] for a in alarms if a["kind"] in ("zone_ppe_violation", "zone_intrusion"))
    used = [0] * len(windows)
    first = [None] * len(windows)
    stray = 0
    for t in zone:
        hit = [i for i, (a, b) in enumerate(windows) if a <= t <= b]
        if hit:
            i = hit[0]
            used[i] += 1
            first[i] = t if first[i] is None else first[i]
        else:
            stray += 1
    sc.false_alarms = stray
    for i, (a, _) in enumerate(windows):
        if used[i]:
            sc.detected += 1
            sc.duplicates += used[i] - 1
            sc.delays.append(first[i] - a)
    sc.missed = sc.expected - sc.detected
    fall_alarms = sorted(a["t"] for a in alarms if a["kind"] == "fall")
    sc.falls_expected = len(falls)
    matched = set()
    for t in fall_alarms:
        j = next((j for j, f in enumerate(falls) if f <= t <= f + fall_window_s and j not in matched), None)
        if j is None:
            sc.fall_false_alarms += 1
        else:
            matched.add(j)
    sc.falls_detected = len(matched)
    return sc


def merge(scores: list[Score]) -> Score:
    m = Score()
    for s in scores:
        m.expected += s.expected; m.detected += s.detected; m.missed += s.missed; m.false_alarms += s.false_alarms
        m.duplicates += s.duplicates; m.delays += s.delays; m.hours += s.hours
        m.falls_expected += s.falls_expected; m.falls_detected += s.falls_detected; m.fall_false_alarms += s.fall_false_alarms
    return m
