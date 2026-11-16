"""Safety events, debouncing and alarm state.

Two knobs, both there for the same reason: a single frame is never
evidence of anything. `debounce_s` stops a one-frame detection from
sounding a buzzer; `cooldown_s` stops one real event from sounding it
forty times. Both add latency, and both are reported, not hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class SafetyEvent:
    kind: str                       # ppe_violation | fall | zone_intrusion | fire | smoke
    severity: str
    track_id: int | None = None
    detail: str = ""
    t: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Identity for debouncing: same kind, same subject."""
        return f"{self.kind}:{self.track_id}:{self.detail}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "track_id": self.track_id,
            "detail": self.detail,
            "t": round(self.t, 3),
            **({"extra": self.extra} if self.extra else {}),
        }


class Debouncer:
    """An event must persist `debounce_s` before it fires, then goes quiet
    for `cooldown_s`. Returns only the events that should actually alarm."""

    def __init__(self, debounce_s: float = 3.0, cooldown_s: float = 10.0, grace_s: float = 0.0):
        self.debounce_s = debounce_s
        self.cooldown_s = cooldown_s
        # An event missing for less than `grace_s` keeps its debounce credit: one missed detection or a coat score
        # dipping under its threshold for a frame would otherwise restart the 3 s window and the alarm never fires.
        self.grace_s = grace_s
        self._last_seen: dict[str, float] = {}
        self._first_seen: dict[str, float] = {}
        self._last_fired: dict[str, float] = {}

    def filter(self, events: list[SafetyEvent], now: float | None = None) -> list[SafetyEvent]:
        now = time.time() if now is None else now
        fired: list[SafetyEvent] = []
        present = {e.key for e in events}

        for ev in events:
            self._last_seen[ev.key] = now
            first = self._first_seen.setdefault(ev.key, now)
            if now - first < self.debounce_s:
                continue
            last = self._last_fired.get(ev.key, -1e9)
            if now - last < self.cooldown_s:
                continue
            self._last_fired[ev.key] = now
            ev.extra["debounced_s"] = round(now - first, 2)
            fired.append(ev)

        # An event that stopped occurring must restart its debounce window,
        # otherwise a flicker accumulates credit across minutes and fires.
        for key in [k for k in self._first_seen if k not in present and now - self._last_seen.get(k, now) >= self.grace_s]:
            del self._first_seen[key]
            self._last_seen.pop(key, None)
        # Forget fire times once their cooldown is over: the key holds a track id, so over a day this dict would
        # otherwise keep one entry per person ever seen.
        for key in [k for k, t in self._last_fired.items() if now - t >= self.cooldown_s and k not in present]:
            del self._last_fired[key]

        return fired


def worst_severity(events: list[SafetyEvent]) -> str:
    if not events:
        return "none"
    return max((e.severity for e in events), key=lambda s: SEVERITY_ORDER.get(s, 0))
