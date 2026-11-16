"""Overlay drawing. Kept out of the pipeline so it can be switched off.

What a viewer (or a jury) must read at a glance on the video: who wears a lab coat (green), who does not (amber), who is a
violation right now (red), and where the zone is. Labels are Vietnamese with diacritics, drawn with a bundled TTF font
(OpenCV's Hershey fonts cannot draw them); without Pillow or the font the same labels are drawn without diacritics.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import cv2
import numpy as np

from .pipeline.decision import SafetyEvent
from .pipeline.tracker import Track
from .pipeline.zones import Zone

# BGR
COLOR = {
    "ok": (90, 200, 70),        # wears a lab coat
    "warn": (40, 170, 255),     # no lab coat, not (yet) a violation
    "crit": (50, 50, 230),      # violation now (no coat in a zone that requires one), or a person down
    "unk": (150, 150, 150),     # clothes not visible enough to judge (face only, hidden behind someone): no verdict
    "zone": (200, 120, 255),
    "text": (255, 255, 255),
}
FONT_PATH = Path(__file__).resolve().parent / "assets" / "DejaVuSans-Bold.ttf"
_font_cache: dict[int, object] = {}


def _font(size: int):
    """The bundled TTF at `size`, or None when Pillow / the font is unavailable (then labels fall back to ASCII)."""
    if size not in _font_cache:
        try:
            from PIL import ImageFont

            _font_cache[size] = ImageFont.truetype(str(FONT_PATH), size)
        except Exception:  # noqa: BLE001 -- no Pillow, no font file: degrade, never break the overlay
            _font_cache[size] = None
    return _font_cache[size]


def ascii_fold(s: str) -> str:
    """'VI PHẠM · Không áo lab' -> 'VI PHAM - Khong ao lab' for the Hershey fallback."""
    s = s.replace("đ", "d").replace("Đ", "D").replace("·", "-")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def _num(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def person_label(tr: Track, violating: bool, wearing: bool, coat_score: float | None) -> tuple[str, str]:
    """(text, colour key) for one person box."""
    if tr.fall_state == "down":
        return f"#{tr.track_id} CÓ THỂ ĐÃ NGÃ", "crit"
    if violating:
        return f"#{tr.track_id} VI PHẠM · không áo lab", "crit"
    if wearing:
        return f"#{tr.track_id} Có áo lab" + (f" {_num(coat_score)}" if coat_score else ""), "ok"
    if getattr(tr, "attire_unknown", False):
        return f"#{tr.track_id} Chưa thấy rõ áo", "unk"
    missing = [k for k, v in tr.ppe_status.items() if not v]
    return f"#{tr.track_id} Không áo lab" + (f" · thiếu {', '.join(missing)}" if missing else ""), "warn"


def zone_label(z: Zone) -> str:
    return "VÙNG CẤM · cần áo lab" if "lab_coat" in z.requires else ("VÙNG CẤM" if not z.requires else f"VÙNG CẤM · cần {', '.join(z.requires)}")


def draw(
    frame: np.ndarray,
    tracks: list[Track],
    zones: list[Zone],
    events: list[SafetyEvent],
    hud: str = "",
    coat_scores: dict[int, float] | None = None,
    memory_s: float = 2.0,
    item: str = "lab_coat",
) -> np.ndarray:
    """`coat_scores`: best coat score seen on each person THIS frame (router._coat_scores); `memory_s` must be the same coat memory
    the zone rule uses (router.attire_memory_s), otherwise the box can say "Có áo lab" while the rule raises a violation."""
    out = frame.copy()
    h, w = out.shape[:2]
    coat_scores = coat_scores or {}
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int], int]] = []   # text, anchor (x, bottom y), colour, size

    for z in zones:
        pts = np.array(z.pixel_polygon(w, h), dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], COLOR["zone"])
        cv2.addWeighted(overlay, 0.16, out, 0.84, 0, out)
        cv2.polylines(out, [pts], True, COLOR["zone"], 2, cv2.LINE_AA)
    # zone names as a legend in the top-left corner: next to the polygon they collide with the people standing on it
    for i, z in enumerate(zones):
        labels.append((zone_label(z), (4, 30 + 26 * i), COLOR["zone"], 15))

    violators = {e.track_id for e in events if e.kind in ("zone_ppe_violation", "zone_intrusion")}
    for tr in tracks:
        d = tr.det
        wearing = tr.wears(item, tr.last_seen, memory_s)
        text, key = person_label(tr, tr.track_id in violators, wearing, coat_scores.get(tr.track_id))
        col = COLOR[key]
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 3 if key == "crit" else 2)
        fx, fy = d.foot
        cv2.circle(out, (int(fx), int(min(fy, h - 3))), 4, col, -1)
        labels.append((text, (int(d.x1), int(d.y1)), col, 16))

    _put_labels(out, labels)
    if hud:   # engineering line: only for the local window / recordings, never on the dashboard video (app.py decides)
        for i, line in enumerate(hud.split("\n")):
            cv2.putText(out, line, (8, h - 12 - 18 * (len(hud.split(chr(10))) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, COLOR["text"], 1, cv2.LINE_AA)
    return out


def _put_labels(out: np.ndarray, labels: list) -> None:
    """Filled tag + white text above each anchor; TTF with diacritics when available, Hershey ASCII otherwise."""
    if not labels:
        return
    h, w = out.shape[:2]
    if _font(16) is not None:
        from PIL import Image, ImageDraw

        img = Image.fromarray(out)                     # channel order is irrelevant: colours below are given in BGR
        dr = ImageDraw.Draw(img)
        for text, (x, y), col, size in labels:
            f = _font(size)
            x0, y0, x1, y1 = dr.textbbox((0, 0), text, font=f)
            tw, th = x1 - x0, y1 - y0
            bx = int(min(max(x, 0), max(0, w - tw - 8)))
            by = int(max(y - th - 8, 0))
            dr.rectangle([bx, by, bx + tw + 8, by + th + 8], fill=tuple(col))
            dr.text((bx + 4 - x0, by + 4 - y0), text, font=f, fill=COLOR["text"])
        out[:] = np.asarray(img)
        return
    for text, (x, y), col, size in labels:
        t = ascii_fold(text)
        (tw, th), base = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bx, by = int(min(max(x, 0), max(0, w - tw - 8))), int(max(y - th - 10, 0))
        cv2.rectangle(out, (bx, by), (bx + tw + 8, by + th + 10), col, -1)
        cv2.putText(out, t, (bx + 4, by + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR["text"], 1, cv2.LINE_AA)
