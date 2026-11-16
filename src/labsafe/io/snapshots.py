"""Save frames with their overlay + a JSON sidecar, on a schedule.

Why this exists: the dashboard only streams MJPEG (never persisted, and
`--bench` turns it off entirely to avoid attributing JPEG-encode cost to
the model), so there was no way to look at what the model actually saw on
a run that just finished, short of re-running it live with `--no-display`
off. For a first live run on real hardware, "does detection look sane on
an actual frame" is exactly what needs checking by eye (KNOWN_ISSUES.md
#3: a healthy-looking FPS number and a dead model look identical without
this).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FrameSaver:
    """Saves at most `max_n` frames, no more often than every `every_s`
    seconds. `every_s <= 0` disables saving entirely (the default in most
    configs -- this is opt-in via `--save-every`)."""

    def __init__(self, out_dir: Path, every_s: float, max_n: int = 30):
        self.out_dir = Path(out_dir)
        self.every_s = float(every_s)
        self.max_n = int(max_n)
        self.saved = 0
        self._last_saved_t: float | None = None

    def due(self, now: float) -> bool:
        if self.every_s <= 0:
            return False
        if self.saved >= self.max_n:
            return False
        if self._last_saved_t is None:
            return True
        return (now - self._last_saved_t) >= self.every_s

    def save(self, now: float, frame_vis: np.ndarray, meta: dict[str, Any]) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{self.saved:04d}"
        img_path = self.out_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), frame_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        payload = {"t": now, **meta}
        (self.out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
        self.saved += 1
        self._last_saved_t = now
        return img_path
