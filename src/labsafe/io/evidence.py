"""Save a photo of the person behind each alarm (e.g. no lab coat inside a danger zone), as evidence on disk.

Per alarm: `<n>_<kind>_p<track>_crop.jpg` (the person, cut from the RAW frame with some margin) and
`<n>_<kind>_p<track>_frame.jpg` (the annotated full frame, for context), plus one line in `violations.jsonl`.
Files are numbered, not named by wall time: the board clock is wrong after every power loss (no RTC battery). The
jsonl line still records the board's time and the monotonic age since start.

These are photos of identifiable people. Keep them on the board, tell the people in the lab, delete when no longer
needed; `max_files` caps disk use (the oldest pairs are removed first).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..models.backends import Detection

DEFAULT_KINDS = ("zone_ppe_violation", "zone_intrusion")


def crop_person(frame: np.ndarray, det: Detection, padding: float = 0.15) -> np.ndarray:
    """Person box grown by `padding` of its size on every side, clipped to the frame."""
    h, w = frame.shape[:2]
    bw, bh = det.x2 - det.x1, det.y2 - det.y1
    x1 = int(max(0, det.x1 - padding * bw))
    y1 = int(max(0, det.y1 - padding * bh))
    x2 = int(min(w, det.x2 + padding * bw))
    y2 = int(min(h, det.y2 + padding * bh))
    if x2 <= x1 or y2 <= y1:
        return frame[0:1, 0:1].copy()
    return frame[y1:y2, x1:x2].copy()


class ViolationRecorder:
    def __init__(self, out_dir: str | Path, kinds: tuple[str, ...] = DEFAULT_KINDS, padding: float = 0.15,
                 max_files: int = 2000, save_frame: bool = True, jpeg_quality: int = 90):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kinds = set(kinds)
        self.padding = float(padding)
        self.max_files = int(max_files)
        self.save_frame = bool(save_frame)
        self.quality = int(jpeg_quality)
        self.saved = 0
        self._t0 = time.monotonic()
        existing = sorted(self.dir.glob("*_crop.jpg"))
        self._n = (int(existing[-1].name.split("_", 1)[0]) + 1) if existing else 1

    def record(self, raw: np.ndarray, annotated: np.ndarray | None, events: list, tracks: list) -> list[bytes | None]:
        """For each alarming event: save crop (+frame) if it is a recorded kind about a visible person.

        Returns one entry per event, aligned with `events`: the crop as JPEG bytes (for the dashboard), or None.
        """
        by_id = {t.track_id: t for t in tracks}
        out: list[bytes | None] = []
        for ev in events:
            tr = by_id.get(ev.track_id)
            if ev.kind not in self.kinds or tr is None:
                out.append(None)
                continue
            crop = crop_person(raw, tr.det, self.padding)
            ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                out.append(None)
                continue
            stem = f"{self._n:06d}_{ev.kind}_p{ev.track_id}"
            self._n += 1
            (self.dir / f"{stem}_crop.jpg").write_bytes(buf.tobytes())
            if self.save_frame and annotated is not None:
                cv2.imwrite(str(self.dir / f"{stem}_frame.jpg"), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            line = {"file": f"{stem}_crop.jpg", **ev.as_dict(), "box": tr.det.as_dict(),
                    "since_start_s": round(time.monotonic() - self._t0, 2)}
            with (self.dir / "violations.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.saved += 1
            out.append(buf.tobytes())
        self._enforce_cap()
        return out

    def _enforce_cap(self) -> None:
        crops = sorted(self.dir.glob("*_crop.jpg"))
        for old in crops[: max(0, len(crops) - self.max_files)]:
            old.unlink(missing_ok=True)
            (self.dir / old.name.replace("_crop.jpg", "_frame.jpg")).unlink(missing_ok=True)
