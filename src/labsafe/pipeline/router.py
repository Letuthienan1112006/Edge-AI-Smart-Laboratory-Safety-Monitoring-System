"""The adaptive event router -- the part of this system the paper is about.

Two modes, one code path, switched by a single flag so the ablation is
honest (no second implementation to differ in unrelated ways):

  mode="parallel"  (E1 baseline) every stage runs on every frame.
  mode="adaptive"  (E5)          stage B runs only when stage A found a
                                 person AND that person is due a re-check;
                                 stage C runs on its own wall-clock cadence.

What it measures, because the research questions need it and nothing else
in the system can supply it:
  * per-stage invocation counts   -> how much compute the gating actually saved
  * per-stage latency samples     -> p50/p90/p99, never just the mean
  * occupancy (frames with >=1 person) -> the x-axis of the RQ2 curve
  * event timestamps              -> detection latency vs ground truth
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..models.backends import Backend, Detection
from .async_stage import AsyncPPEChecker, AsyncStageC
from .decision import Debouncer, SafetyEvent
from .fall import FallConfig, FallDetector
from .ppe import PPEChecker, PPEConfig, missing_items
from .tracker import IOUTracker, Track
from .zones import Zone


@dataclass
class RouterStats:
    frames: int = 0
    frames_with_person: int = 0
    stage_calls: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "c": 0})
    stage_latency_ms: dict[str, list[float]] = field(
        default_factory=lambda: {"a": [], "b": [], "c": []}
    )
    # A call count alone cannot tell "200 FPS of a healthy model" apart
    # from "200 FPS of a model that converted broken and sees nothing"
    # (docs/KNOWN_ISSUES.md #3). These two counters make that distinction
    # visible in every summary and every report row.
    stage_detections: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "c": 0})
    stage_blank_frames: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "c": 0})
    frame_latency_ms: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    # Per-stage-A timing breakdown (pre/infer/post, see Backend.last_timing).
    # A stub backend never sets last_timing, so this stays {} for E1..E5/N5
    # runs -- absence here means "no breakdown available", not "zero cost".
    stage_a_breakdown_ms: dict[str, list[float]] = field(default_factory=dict)
    # None (benchmarks): keep every sample, the report needs them all. An int (live / dashboard runs that last hours):
    # keep only the last `window` samples per series and the last 1000 events. Unbounded lists added ~7 floats per
    # frame forever = ~0.4 MB/min at 30 FPS, a real leak for an all-day run (KNOWN_ISSUES #37).
    window: int | None = None

    def __post_init__(self) -> None:
        if self.window:
            self.stage_latency_ms = {k: deque(v, maxlen=self.window) for k, v in self.stage_latency_ms.items()}
            self.frame_latency_ms = deque(self.frame_latency_ms, maxlen=self.window)
            self.events = deque(self.events, maxlen=1000)

    def series(self, table: dict, key: str):
        if key not in table:
            table[key] = deque(maxlen=self.window) if self.window else []
        return table[key]

    @property
    def occupancy(self) -> float:
        return self.frames_with_person / self.frames if self.frames else 0.0

    def liveness_warning(self) -> str | None:
        """A stage called a lot with zero detections is not "quiet", it is
        dead. 100 calls is enough to rule out "the object just wasn't in
        frame yet" -- see ultralytics#23753 / docs/KNOWN_ISSUES.md #3."""
        if self.stage_calls["a"] >= 100 and self.stage_detections["a"] == 0:
            return (
                "stage A called >=100 times with 0 detections -- the model may "
                "have died after conversion (see docs/KNOWN_ISSUES.md #3 / "
                "ultralytics issue #23753). Do not trust the FPS/latency numbers "
                "in this run until this is ruled out with "
                "scripts/export/validate_conversion.py."
            )
        return None

    def summary(self) -> dict:
        def pct(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0}
            a = np.asarray(xs, dtype=np.float64)
            return {
                "n": int(a.size),
                "mean": round(float(a.mean()), 2),
                "p50": round(float(np.percentile(a, 50)), 2),
                "p90": round(float(np.percentile(a, 90)), 2),
                "p99": round(float(np.percentile(a, 99)), 2),
                "max": round(float(a.max()), 2),
            }

        return {
            "frames": self.frames,
            "occupancy": round(self.occupancy, 4),
            "stage_calls": dict(self.stage_calls),
            "stage_calls_per_frame": {
                k: round(v / self.frames, 3) if self.frames else 0.0
                for k, v in self.stage_calls.items()
            },
            "stage_latency_ms": {k: pct(v) for k, v in self.stage_latency_ms.items()},
            "stage_detections": dict(self.stage_detections),
            "stage_detections_per_call": {
                k: round(self.stage_detections[k] / v, 3) if v else 0.0
                for k, v in self.stage_calls.items()
            },
            "stage_blank_rate": {
                k: round(self.stage_blank_frames[k] / v, 4) if v else 0.0
                for k, v in self.stage_calls.items()
            },
            "frame_latency_ms": pct(self.frame_latency_ms),
            "stage_a_breakdown_ms": {k: pct(v) for k, v in self.stage_a_breakdown_ms.items()},
            "events": len(self.events),
        }


class AdaptiveRouter:
    def __init__(
        self,
        stage_a: Backend,
        stage_b: PPEChecker | None,
        stage_c: Backend | None,
        zones: list[Zone],
        fall_cfg: FallConfig,
        *,
        mode: str = "adaptive",
        tracker_cfg: dict | None = None,
        alerts_cfg: dict | None = None,
        stage_c_interval_s: float = 1.0,
        async_stages: bool = False,
        attire_memory_s: float = 2.0,
        attire_min_conf: float = 0.0,
        stats_window: int | None = None,
        partial_min_h: float = 0.0,
        visibility: dict | None = None,
    ):
        if mode not in {"adaptive", "parallel"}:
            raise ValueError("mode must be 'adaptive' or 'parallel'")
        self.mode = mode
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.stage_c = stage_c
        # async_stages: stage B / C model calls run on their own threads (one NPU core each,
        # see pipeline/async_stage.py). Off by default: the parallel/adaptive ablation must
        # compare like with like, and async trades a bounded result lag for frame time.
        self.async_stages = bool(async_stages)
        self._async_c: AsyncStageC | None = None
        if self.async_stages:
            if stage_b is not None and not isinstance(stage_b, AsyncPPEChecker):
                self.stage_b = AsyncPPEChecker(stage_b.backend, stage_b.cfg)
            if stage_c is not None:
                self._async_c = AsyncStageC(stage_c)
        self.zones = zones
        # Items some zone requires (e.g. lab_coat). Stage A boxes of any class other than `person` are things a
        # person wears, never people: they are attached to the person box that contains them (_attach_attire).
        self.attire_classes = frozenset(r for z in zones for r in z.requires)
        # A coat box can drop out for a few frames (occlusion, turning around); remember it this long.
        self.attire_memory_s = float(attire_memory_s)
        # A coat box only counts when the model is this sure. A false coat lets a violator through (the dangerous
        # direction), so this is set higher than stage A's person threshold; measured trade-off in KNOWN_ISSUES #35.
        self.attire_min_conf = float(attire_min_conf)
        # A person whose box touches the bottom edge of the frame has no visible feet (the foot point is clamped to the edge, KNOWN_ISSUES #35 fix). If the box is also
        # SHORT (just a head or shoulders at the frame corner, e.g. the operator at the laptop) the clamp puts a phantom foot inside the zone: such boxes are not zone-checked.
        self.partial_min_h = float(partial_min_h)
        # "No coat" is only a verdict when the clothes can be SEEN: a face at the frame edge, a head between other people or a sliver
        # of a person is 'attire unknown' -> no zone_ppe_violation (it still counts for zone_intrusion). KNOWN_ISSUES #50.
        #   min_h: box height / frame height below which a box must be tall-shaped (h/w >= tall_aspect: a far full body) to be judged
        #   min_w_px: thinner boxes are never judged;  max_occlusion: share of the box covered by people closer to the camera
        v = visibility or {}
        self.vis_min_h = float(v.get("min_h", 0.0))
        self.vis_tall_aspect = float(v.get("tall_aspect", 2.0))
        self.vis_min_w_px = float(v.get("min_w_px", 0.0))
        self.vis_max_occlusion = float(v.get("max_occlusion", 1.0))
        self.stage_c_interval_s = stage_c_interval_s
        tracker_cfg = tracker_cfg or {}
        self.tracker = IOUTracker(
            iou_threshold=tracker_cfg.get("iou_match_threshold", 0.3),
            max_age_s=tracker_cfg.get("max_age_s", 1.5),
        )
        self.fall = FallDetector(fall_cfg)
        alerts_cfg = alerts_cfg or {}
        self.debouncer = Debouncer(
            debounce_s=alerts_cfg.get("debounce_s", 3.0),
            cooldown_s=alerts_cfg.get("cooldown_s", 10.0),
            grace_s=alerts_cfg.get("grace_s", 0.0),
        )
        self.stats = RouterStats(window=stats_window)
        self._last_stage_c = 0.0
        self._last_fire: tuple[str, float] = ("none", 0.0)
        self.tracks: list[Track] = []

    # -- the frame loop ---------------------------------------------------

    def process(self, frame: np.ndarray, now: float | None = None) -> tuple[list[SafetyEvent], list[SafetyEvent]]:
        """Returns (all_active_events, alarming_events_after_debounce)."""
        now = time.time() if now is None else now
        t_frame = time.perf_counter()
        h, w = frame.shape[:2]
        self.stats.frames += 1

        # Stage A -- always on. This is the only unconditional model call.
        dets_a = self.stage_a(frame)
        persons = [d for d in dets_a if d.cls_name == "person"]
        attire = [d for d in dets_a if d.cls_name != "person" and d.score >= self.attire_min_conf]
        self.stats.stage_calls["a"] += 1
        self.stats.stage_latency_ms["a"].append(self.stage_a.last_latency_ms)
        self.stats.stage_detections["a"] += len(persons)
        if not persons:
            self.stats.stage_blank_frames["a"] += 1
        if persons:
            self.stats.frames_with_person += 1
        for k, v in getattr(self.stage_a, "last_timing", {}).items():
            self.stats.series(self.stats.stage_a_breakdown_ms, k).append(float(v))

        self.tracks = self.tracker.update(persons, now=now)
        if attire:
            self._attach_attire(self.tracks, attire, now)
        # Best coat score per person BEFORE the attire_min_conf filter (the dashboard shows it: "the coat was seen at 0.45, the rule needs 0.60")
        self._coat_scores = self._best_attire_scores(self.tracks, [d for d in dets_a if d.cls_name != "person"])
        events: list[SafetyEvent] = []

        # Zones and fall are geometry over stage A's output: no extra model,
        # so they run in both modes and cost nothing worth gating.
        events += self._zone_events(self.tracks, w, h, now)
        if self.fall.cfg.enabled:
            events += self._fall_events(self.tracks, h, now)

        # Stage B -- gated in adaptive mode.
        if self.stage_b is not None:
            run_b = bool(self.tracks) if self.mode == "adaptive" else True
            if run_b:
                before = self.stage_b.calls
                violating = self.stage_b.check(
                    frame,
                    self.tracks if self.mode == "adaptive" else self._all_tracks(),
                    now=now,
                    force=(self.mode == "parallel"),
                )
                made = self.stage_b.calls - before
                if made:
                    self.stats.stage_calls["b"] += made
                    self.stats.stage_latency_ms["b"].append(self.stage_b.backend.last_latency_ms)
                    self.stats.stage_detections["b"] += self.stage_b.last_detection_count
                    if self.stage_b.last_detection_count == 0:
                        # `made` calls happened this frame and produced zero
                        # detections between them, so every one of them was
                        # blank. Counting 1 here while stage_calls counts
                        # `made` puts a different denominator under the
                        # numerator and reports 0.2 where the truth is 1.0.
                        self.stats.stage_blank_frames["b"] += made
                for tr in violating:
                    events.append(
                        SafetyEvent(
                            "ppe_violation", "medium", tr.track_id,
                            detail=",".join(sorted(missing_items(tr))),
                        )
                    )

        # Stage C -- independent cadence. Fire does not need 30 Hz; in
        # parallel mode it does run every frame, which is the point of E1.
        if self.stage_c is not None:
            due = (now - self._last_stage_c) >= self.stage_c_interval_s
            if self._async_c is not None:
                done = self._async_c.poll()
                if self.mode == "parallel" or due:
                    self._last_stage_c = now
                    self._async_c.submit(frame)
                if done is not None:
                    dets, lat_ms = done
                    self.stats.stage_calls["c"] += 1
                    self.stats.stage_latency_ms["c"].append(lat_ms)
                    self.stats.stage_detections["c"] += len(dets)
                    if not dets:
                        self.stats.stage_blank_frames["c"] += 1
                    self._last_fire = self._fire_state(dets, now)
            elif self.mode == "parallel" or due:
                self._last_stage_c = now
                dets = self.stage_c(frame)
                self.stats.stage_calls["c"] += 1
                self.stats.stage_latency_ms["c"].append(self.stage_c.last_latency_ms)
                self.stats.stage_detections["c"] += len(dets)
                if not dets:
                    self.stats.stage_blank_frames["c"] += 1
                self._last_fire = self._fire_state(dets, now)
            kind, score = self._last_fire
            if kind in {"fire", "smoke"}:
                events.append(
                    SafetyEvent(
                        kind, "critical" if kind == "fire" else "high",
                        detail=f"score={score:.2f}",
                    )
                )

        alarming = self.debouncer.filter(events, now=now)
        for ev in alarming:
            self.stats.events.append(ev.as_dict())

        self.stats.frame_latency_ms.append((time.perf_counter() - t_frame) * 1000.0)
        return events, alarming

    def async_stats(self) -> dict:
        """Queue health of the async stages ({} when running synchronously)."""
        out: dict = {}
        if isinstance(self.stage_b, AsyncPPEChecker):
            out["b"] = self.stage_b.worker.stats()
        if self._async_c is not None:
            out["c"] = self._async_c.worker.stats()
        return out

    def close(self) -> None:
        if isinstance(self.stage_b, AsyncPPEChecker):
            self.stage_b.close()
        if self._async_c is not None:
            self._async_c.close()

    def set_zones(self, zones: list[Zone]) -> None:
        """Swap the zone set while running. One attribute assignment: `process()` runs on another thread and reads
        `self.zones` once per use, so it sees either the old list or the new one, never a mix."""
        self.zones = list(zones)
        self.attire_classes = frozenset(r for z in self.zones for r in z.requires)

    def people_summary(self, events: list[SafetyEvent], item: str = "lab_coat") -> dict:
        """What the dashboard shows for the current frame: who is here, who wears `item`, who is where.

        `wearing` uses the same memory window as the zone rule, so the two can never disagree on screen.
        """
        tracks = self.tracks
        now = max((t.last_seen for t in tracks), default=0.0)
        wearing = [t for t in tracks if t.wears(item, now, self.attire_memory_s)]
        violators = {e.track_id for e in events if e.kind == "zone_ppe_violation"}
        intruders = {e.track_id for e in events if e.kind == "zone_intrusion"}
        return {
            "persons": len(tracks),
            "wearing": len(wearing),
            "not_wearing": sum(1 for t in tracks if t not in wearing and not getattr(t, "attire_unknown", False)),
            "unknown": sum(1 for t in tracks if t not in wearing and getattr(t, "attire_unknown", False)),
            "in_zone": sum(1 for t in tracks if t.zone_names),
            "violating": len(violators | intruders),
            "people": [
                {"id": t.track_id, "wearing": t in wearing, "unknown": t not in wearing and getattr(t, "attire_unknown", False),
                 "zones": list(t.zone_names),
                 "coat_score": (round(self._coat_scores[t.track_id], 2) if t.track_id in getattr(self, "_coat_scores", {}) else None),
                 "violating": t.track_id in violators or t.track_id in intruders,
                 "fall": t.fall_state}
                for t in tracks
            ],
        }

    # -- helpers ----------------------------------------------------------

    def _all_tracks(self) -> list[Track]:
        return list(self.tracker.tracks.values())

    @staticmethod
    def _coat_owner(g: Detection, tracks: list[Track], min_inside: float = 0.5):
        """The ONE person box that contains most of the attire box (>= min_inside of its area). A torso-fit rule was tried on 21/10
        and made coat wearers MORE often flagged on the lab test sets (góc 2: 17 -> 26 of 684), so containment stays (KNOWN_ISSUES #51)."""
        ga = max(1e-6, (g.x2 - g.x1) * (g.y2 - g.y1))
        best, best_f = None, min_inside
        for tr in tracks:
            d = tr.det
            iw = max(0.0, min(g.x2, d.x2) - max(g.x1, d.x1))
            ih = max(0.0, min(g.y2, d.y2) - max(g.y1, d.y1))
            f = iw * ih / ga
            if f >= best_f:
                best, best_f = tr, f
        return best

    @classmethod
    def _best_attire_scores(cls, tracks: list[Track], attire: list[Detection], min_inside: float = 0.5) -> dict[int, float]:
        """track id -> highest score of an attire box owned by that person (same owner rule as _attach_attire)."""
        out: dict[int, float] = {}
        for g in attire:
            best = cls._coat_owner(g, tracks, min_inside)
            if best is not None:
                out[best.track_id] = max(out.get(best.track_id, 0.0), float(g.score))
        return out

    @classmethod
    def _attach_attire(cls, tracks: list[Track], attire: list[Detection], now: float, min_inside: float = 0.5) -> None:
        """Give each attire box to ONE person (`_coat_owner`): one coat never counts for two people standing close together."""
        for g in attire:
            best = cls._coat_owner(g, tracks, min_inside)
            if best is not None:
                best.attire_seen[g.cls_name] = now

    def attire_visible(self, tr: Track, tracks: list[Track], h: int) -> bool:
        """Can this person's clothes be judged in this frame? (see `visibility` in __init__)"""
        d = tr.det
        bw, bh = max(1e-6, d.x2 - d.x1), max(1e-6, d.y2 - d.y1)
        if bw < self.vis_min_w_px:
            return False
        if bh < self.vis_min_h * h and bh / bw < self.vis_tall_aspect:
            return False
        if self.vis_max_occlusion < 1.0:
            covered = 0.0
            for o in tracks:
                if o is tr or o.det.y2 <= d.y2:      # only people whose feet are lower = closer to the camera can hide this one
                    continue
                iw = max(0.0, min(d.x2, o.det.x2) - max(d.x1, o.det.x1))
                ih = max(0.0, min(d.y2, o.det.y2) - max(d.y1, o.det.y1))
                covered += iw * ih
            if covered / (bw * bh) > self.vis_max_occlusion:
                return False
        return True

    def _zone_events(self, tracks: list[Track], w: int, h: int, now: float) -> list[SafetyEvent]:
        out: list[SafetyEvent] = []
        for tr in tracks:
            tr.attire_unknown = not self.attire_visible(tr, tracks, h)
            partial = self.partial_min_h > 0 and tr.det.y2 >= h - 2 and (tr.det.y2 - tr.det.y1) < self.partial_min_h * h
            inside = [] if partial else [z for z in self.zones if z.contains_foot(tr.det, w, h)]
            tr.zone_names = [z.name for z in inside]
            for z in inside:
                tr.zone_since.setdefault(z.name, now)
                dwell = round(now - tr.zone_since[z.name], 2)
                if not z.requires:
                    out.append(SafetyEvent("zone_intrusion", z.severity, tr.track_id, detail=z.name,
                                           extra={"dwell_s": dwell}))
                    continue
                missing = [r for r in z.requires if not tr.wears(r, now, self.attire_memory_s)]
                if missing and not tr.attire_unknown:     # a coat seen recently still counts; an unseen one is not 'missing'
                    out.append(SafetyEvent("zone_ppe_violation", z.severity, tr.track_id, detail=z.name,
                                           extra={"dwell_s": dwell, "missing": missing}))
            for gone in [n for n in tr.zone_since if n not in tr.zone_names]:
                del tr.zone_since[gone]
        return out

    def _fall_events(self, tracks: list[Track], h: int, now: float) -> list[SafetyEvent]:
        fallen = self.fall.update(tracks, h, now=now)
        out = [SafetyEvent("fall", "critical", tr.track_id, detail="fall_detected") for tr in fallen]
        # A person still on the ground keeps the event active, so debounce
        # sees a persistent condition rather than one isolated frame.
        for tr in tracks:
            if tr.fall_state == "down" and not any(e.track_id == tr.track_id for e in out):
                out.append(SafetyEvent("fall", "critical", tr.track_id, detail="still_down"))
        return out

    @staticmethod
    def _fire_state(dets: list[Detection], now: float) -> tuple[str, float]:
        best: Detection | None = None
        for d in dets:
            if d.cls_name in {"fire", "smoke"} and (best is None or d.score > best.score):
                best = d
        return (best.cls_name, best.score) if best else ("none", 0.0)
