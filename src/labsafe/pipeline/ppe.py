"""PPE compliance by association, not by a `no_goggles` class.

The detector reports PPE items that are present. Compliance is then a set
difference against `required`. This keeps stage B's label space small and
its training data portable across PPE datasets, which disagree wildly on
how absence is annotated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..models.backends import Backend, Detection
from .tracker import Track


@dataclass
class PPEConfig:
    required: list[str]
    recheck_interval_s: float = 2.0
    crop_padding: float = 0.12
    min_person_px: int = 64          # below this a crop carries no usable signal

    @classmethod
    def from_dict(cls, d: dict, required: list[str]) -> "PPEConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d and f != "required"}
        return cls(required=list(required), **known)


class PPEChecker:
    def __init__(self, backend: Backend, cfg: PPEConfig):
        self.backend = backend
        self.cfg = cfg
        self.calls = 0
        self.skipped = 0
        # Raw detection count from the most recent check() -- lets the
        # router tell "stage B ran but the PPE model saw nothing" apart
        # from "stage B ran and everyone was compliant" (see router.py
        # RouterStats.stage_detections).
        self.last_detection_count = 0

    def due(self, tr: Track, now: float) -> bool:
        """PPE state changes on the scale of seconds, not frames."""
        return now - tr.last_ppe_check >= self.cfg.recheck_interval_s

    def check(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        now: float | None = None,
        force: bool = False,
    ) -> list[Track]:
        """Run stage B on the tracks that are due. Returns tracks now violating.

        `force` bypasses the re-check interval. The baseline arm of the
        ablation MUST pass it: the interval is one of the optimisations
        being measured, and a control arm that quietly keeps it makes the
        optimisation look worthless in exactly the comparison meant to
        prove it works.
        """
        now = time.time() if now is None else now
        violating: list[Track] = []
        self.last_detection_count = 0

        for tr in tracks:
            if not force and not self.due(tr, now):
                self.skipped += 1
                if self._violates(tr):
                    violating.append(tr)
                continue

            crop, ok = self._crop(frame, tr.det)
            tr.last_ppe_check = now
            if not ok:
                continue
            self.calls += 1
            dets = self.backend(crop)
            self.last_detection_count += len(dets)
            items = {d.cls_name for d in dets if d.score >= self.backend.conf}
            tr.ppe_status = {req: (req in items) for req in self.cfg.required}
            if self._violates(tr):
                violating.append(tr)

        return violating

    @staticmethod
    def _violates(tr: Track) -> bool:
        return bool(tr.ppe_status) and not all(tr.ppe_status.values())

    def _crop(self, frame: np.ndarray, det: Detection) -> tuple[np.ndarray, bool]:
        h, w = frame.shape[:2]
        bw, bh = det.wh
        if min(bw, bh) < self.cfg.min_person_px:
            return frame, False
        px, py = bw * self.cfg.crop_padding, bh * self.cfg.crop_padding
        x1 = max(0, int(det.x1 - px))
        y1 = max(0, int(det.y1 - py))
        x2 = min(w, int(det.x2 + px))
        y2 = min(h, int(det.y2 + py))
        if x2 <= x1 or y2 <= y1:
            return frame, False
        return frame[y1:y2, x1:x2], True


def missing_items(tr: Track) -> list[str]:
    return [k for k, present in tr.ppe_status.items() if not present]
