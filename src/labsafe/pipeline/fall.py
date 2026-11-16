"""Fall detection as a temporal state machine over person tracks.

Why not a single aspect-ratio threshold (see docs/research-plan.md 2.3):
crouching to pick something up widens the box exactly like a fall does,
and a fall towards the camera barely changes the box at all. The three
conditions below are what separate them, and the third one -- staying
down -- is what kills most false positives, at zero compute cost.

  1. the box becomes wide relative to tall,   AND
  2. its centre dropped fast within a window, AND
  3. it then stays put for `still_duration_s`.

Condition 3 costs latency by construction: you cannot know someone stayed
down without waiting. That is a real trade-off and belongs in the paper,
not hidden -- report `still_duration_s` alongside every fall metric.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .tracker import Track


@dataclass
class FallConfig:
    enabled: bool = True     # `fall: {enabled: false}` in a config turns the detector off (it used to be ignored)
    aspect_ratio_threshold: float = 1.15
    center_drop_ratio: float = 0.18
    drop_window_s: float = 1.0
    still_duration_s: float = 2.5
    still_iou_threshold: float = 0.70
    fps_estimate: float = 15.0

    @classmethod
    def from_dict(cls, d: dict) -> "FallConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


class FallDetector:
    """Updates `track.fall_state` in place; returns tracks that just fell."""

    def __init__(self, cfg: FallConfig):
        self.cfg = cfg

    def update(self, tracks: list[Track], frame_h: int, now: float | None = None) -> list[Track]:
        now = time.time() if now is None else now
        fallen: list[Track] = []
        window_frames = max(2, int(self.cfg.drop_window_s * self.cfg.fps_estimate))

        for tr in tracks:
            w, h = tr.det.wh
            ratio = w / h if h > 0 else 0.0
            wide = ratio >= self.cfg.aspect_ratio_threshold

            dropped = False
            if len(tr.history) >= 2:
                past = tr.history[-min(window_frames, len(tr.history))]
                past_cy = (past[1] + past[3]) / 2.0
                dropped = (tr.det.center[1] - past_cy) / frame_h >= self.cfg.center_drop_ratio

            if tr.fall_state == "upright":
                if wide and dropped:
                    tr.fall_state, tr.fall_since = "falling", now
            elif tr.fall_state == "falling":
                if not wide:
                    tr.fall_state = "upright"          # stood back up: not a fall
                elif self._is_still(tr) and now - tr.fall_since >= self.cfg.still_duration_s:
                    tr.fall_state = "down"
                    fallen.append(tr)
            elif tr.fall_state == "down":
                if not wide:
                    tr.fall_state, tr.fall_since = "upright", 0.0

        return fallen

    def _is_still(self, tr: Track) -> bool:
        """Boxes keep overlapping across frames = the person is not moving."""
        if len(tr.history) < 2:
            return False
        px1, py1, px2, py2 = tr.history[-1]
        d = tr.det
        ix1, iy1 = max(d.x1, px1), max(d.y1, py1)
        ix2, iy2 = min(d.x2, px2), min(d.y2, py2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        prev_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        union = d.area + prev_area - inter
        return (inter / union if union > 0 else 0.0) >= self.cfg.still_iou_threshold
