"""Minimal IoU tracker.

Deliberately not DeepSORT: an appearance model would cost more CPU than
the detector we are trying to save, and every downstream consumer here
(fall state, PPE re-check interval, zone dwell time) only needs identity
continuity over a couple of seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from ..models.backends import Detection


def iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    det: Detection
    first_seen: float
    last_seen: float
    history: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Downstream per-track state, kept here so the router stays stateless.
    last_ppe_check: float = 0.0
    ppe_status: dict[str, bool] = field(default_factory=dict)
    fall_state: str = "upright"      # upright | falling | down
    fall_since: float = 0.0
    zone_names: list[str] = field(default_factory=list)
    zone_since: dict[str, float] = field(default_factory=dict)
    # attire class (e.g. "lab_coat") -> last time a box of it was matched to this person
    attire_seen: dict[str, float] = field(default_factory=dict)
    attire_unknown: bool = False     # clothes not visible enough to judge this frame (router.attire_visible)

    def wears(self, item: str, now: float, memory_s: float) -> bool:
        t = self.attire_seen.get(item)
        return t is not None and now - t <= memory_s

    @property
    def age_s(self) -> float:
        return self.last_seen - self.first_seen


class IOUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age_s: float = 1.5,
        center_cost_threshold: float = 1.2,
    ):
        self.iou_threshold = iou_threshold
        self.max_age_s = max_age_s
        # Centre distance in units of the mean box diagonal. 1.2 allows a
        # box to move a bit more than its own size between two frames --
        # generous at 30 FPS, about right at 8.
        self.center_cost_threshold = center_cost_threshold
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, dets: Iterable[Detection], now: float | None = None) -> list[Track]:
        now = time.time() if now is None else now
        dets = list(dets)
        unmatched = set(range(len(dets)))

        # Greedy match: best IoU first, so two tracks cannot claim one box.
        pairs = sorted(
            (
                (iou(tr.det, d), tid, di)
                for tid, tr in self.tracks.items()
                for di, d in enumerate(dets)
            ),
            key=lambda p: -p[0],
        )
        claimed_tracks: set[int] = set()
        for score, tid, di in pairs:
            if score < self.iou_threshold:
                break
            if tid in claimed_tracks or di not in unmatched:
                continue
            tr = self.tracks[tid]
            tr.history.append((tr.det.x1, tr.det.y1, tr.det.x2, tr.det.y2))
            if len(tr.history) > 90:
                del tr.history[:-90]
            tr.det, tr.last_seen = dets[di], now
            claimed_tracks.add(tid)
            unmatched.discard(di)

        # Fallback: IoU is zero between two boxes that do not overlap at all,
        # which is precisely what a fall looks like at 7-15 FPS -- the box
        # jumps from standing to lying with no overlap, the track dies, and
        # the fall state machine restarts on a fresh track that has no
        # history to compare against. An IoU threshold tuned while watching
        # a 30 FPS laptop preview fails silently at the frame rate the Pi
        # actually delivers. So unmatched pairs get a second chance on
        # centre distance, scaled by box size.
        if unmatched:
            free = [t for t in self.tracks if t not in claimed_tracks
                    and self.tracks[t].last_seen != now]
            cand = sorted(
                (
                    (self._center_cost(self.tracks[tid].det, dets[di]), tid, di)
                    for tid in free
                    for di in unmatched
                ),
                key=lambda p: p[0],
            )
            for cost, tid, di in cand:
                if cost > self.center_cost_threshold:
                    break
                if tid in claimed_tracks or di not in unmatched:
                    continue
                tr = self.tracks[tid]
                tr.history.append((tr.det.x1, tr.det.y1, tr.det.x2, tr.det.y2))
                if len(tr.history) > 90:
                    del tr.history[:-90]
                tr.det, tr.last_seen = dets[di], now
                # A match without box overlap may be ANOTHER person taking this id (crowds, low frame rates): a lab coat seen on
                # the previous wearer must not carry over to them (KNOWN_ISSUES #51). The coat is re-attached this frame if seen.
                tr.attire_seen.clear()
                claimed_tracks.add(tid)
                unmatched.discard(di)

        for di in sorted(unmatched):
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, dets[di], now, now)

        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > self.max_age_s]:
            del self.tracks[tid]

        return [tr for tr in self.tracks.values() if tr.last_seen == now]

    @staticmethod
    def _center_cost(a: Detection, b: Detection) -> float:
        ax, ay = a.center
        bx, by = b.center
        aw, ah = a.wh
        bw, bh = b.wh
        scale = ((aw**2 + ah**2) ** 0.5 + (bw**2 + bh**2) ** 0.5) / 2.0
        if scale <= 0:
            return float("inf")
        return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / scale
