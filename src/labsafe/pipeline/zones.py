"""Restricted-zone test. Pure geometry -- costs nothing, so it always runs.

A zone either forbids everyone (no `requires`: any foot inside is a `zone_intrusion`) or forbids only people who
are missing something (`requires: [lab_coat]`: a person inside WITHOUT a lab coat is a `zone_ppe_violation`, a
person wearing one is allowed). What a person "wears" comes from attire boxes of stage A, see router._attach_attire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..models.backends import Detection


_EDGE_SNAP = 0.01
_GEOMETRY_EPS = 1e-9


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return (abs(_cross(a, b, p)) <= _GEOMETRY_EPS
            and min(a[0], b[0]) - _GEOMETRY_EPS <= p[0] <= max(a[0], b[0]) + _GEOMETRY_EPS
            and min(a[1], b[1]) - _GEOMETRY_EPS <= p[1] <= max(a[1], b[1]) + _GEOMETRY_EPS)


def _segments_intersect(a: tuple[float, float], b: tuple[float, float],
                        c: tuple[float, float], d: tuple[float, float]) -> bool:
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > _GEOMETRY_EPS and ab_d < -_GEOMETRY_EPS)
            or (ab_c < -_GEOMETRY_EPS and ab_d > _GEOMETRY_EPS)):
        if ((cd_a > _GEOMETRY_EPS and cd_b < -_GEOMETRY_EPS)
                or (cd_a < -_GEOMETRY_EPS and cd_b > _GEOMETRY_EPS)):
            return True
    return (_on_segment(a, b, c) or _on_segment(a, b, d)
            or _on_segment(c, d, a) or _on_segment(c, d, b))


def _validated_polygon(raw: Sequence[Sequence[float]], name: object) -> list[tuple[float, float]]:
    """Return a safe canonical polygon for config files and dashboard input.

    Canvas clicks just shy of an image edge are snapped to that edge. This avoids
    sub-pixel strips where a bottom-clipped person's foot falls outside. A simple
    polygon is required because ray casting has unintuitive holes when edges cross.
    """
    if not 3 <= len(raw) <= 60:
        raise ValueError(f"zone {name!r}: polygon needs 3-60 points")
    poly: list[tuple[float, float]] = []
    for p in raw:
        x, y = float(p[0]), float(p[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"zone {name!r} has coordinates outside 0..1 -- "
                "zones are normalised so they survive a resolution change"
            )
        x = 0.0 if x <= _EDGE_SNAP else 1.0 if x >= 1.0 - _EDGE_SNAP else x
        y = 0.0 if y <= _EDGE_SNAP else 1.0 if y >= 1.0 - _EDGE_SNAP else y
        if not poly or poly[-1] != (x, y):     # two clicks into the same snapped corner are one vertex, not an error
            poly.append((x, y))
    if len(poly) > 1 and poly[0] == poly[-1]:  # the same across the closing edge
        poly.pop()
    if len(poly) < 3:
        raise ValueError(f"zone {name!r}: polygon needs 3-60 distinct points")

    if len(set(poly)) != len(poly):
        raise ValueError(f"zone {name!r}: polygon has repeated points")
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"zone {name!r}: polygon self-intersects (các cạnh cắt nhau)")
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
    if abs(area2) <= _GEOMETRY_EPS:
        raise ValueError(f"zone {name!r}: polygon has zero area")
    return poly


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """Ray casting. `poly` is a list of (x, y) in the same units as (x, y)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if x < x_at_y:
                inside = not inside
    return inside


@dataclass
class Zone:
    name: str
    severity: str
    polygon: list[tuple[float, float]]  # normalised 0..1
    requires: tuple[str, ...] = ()      # empty: nobody may enter; else: only people missing one of these are flagged

    def contains_foot(self, det: Detection, frame_w: int, frame_h: int) -> bool:
        fx, fy = det.foot
        # A person cut off by the bottom of the frame has foot y == frame_h, i.e. exactly on a zone edge drawn at
        # 1.0, and ray casting counts a point on that edge as OUTSIDE. Pull the point just inside the frame.
        eps = 1e-6
        nx = min(max(fx / frame_w, eps), 1.0 - eps)
        ny = min(max(fy / frame_h, eps), 1.0 - eps)
        return point_in_polygon(nx, ny, self.polygon)

    def pixel_polygon(self, frame_w: int, frame_h: int) -> list[tuple[int, int]]:
        return [(int(px * frame_w), int(py * frame_h)) for px, py in self.polygon]


def load_zones(zones_cfg: dict) -> list[Zone]:
    out: list[Zone] = []
    for z in (zones_cfg or {}).get("zones", []) or []:
        poly = _validated_polygon(z["polygon"], z.get("name"))
        req = z.get("requires") or []
        if isinstance(req, str):
            req = [req]
        out.append(Zone(z["name"], z.get("severity", "medium"), poly, tuple(str(r) for r in req)))
    return out


SEVERITIES = ("low", "medium", "high", "critical")


def zone_to_dict(z: Zone) -> dict:
    d = {"name": z.name, "severity": z.severity, "polygon": [[round(x, 4), round(y, 4)] for x, y in z.polygon]}
    if z.requires:
        d["requires"] = list(z.requires)
    return d


def validate_zone_dicts(raw: object, allowed_requires: tuple[str, ...] = (), max_zones: int = 8) -> list[Zone]:
    """Zones coming from OUTSIDE the process (the dashboard editor): check everything, raise ValueError with a
    message a person can act on. Never trust the shape of `raw`."""
    if not isinstance(raw, list) or len(raw) > max_zones:
        raise ValueError(f"zones must be a list of at most {max_zones} items")
    seen: set[str] = set()
    for z in raw:
        if not isinstance(z, dict):
            raise ValueError("each zone must be an object")
        name = z.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 40:
            raise ValueError("zone name must be 1-40 characters")
        if name in seen:
            raise ValueError(f"duplicate zone name {name!r}")
        seen.add(name)
        if z.get("severity", "medium") not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        poly = z.get("polygon")
        if not isinstance(poly, list) or not 3 <= len(poly) <= 60:
            raise ValueError(f"zone {name!r}: polygon needs 3-60 points")
        for p in poly:
            if (not isinstance(p, (list, tuple)) or len(p) != 2
                    or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in p)):
                raise ValueError(f"zone {name!r}: each point must be [x, y] numbers")
        req = z.get("requires") or []
        if not isinstance(req, list) or any(r not in allowed_requires for r in req):
            raise ValueError(f"zone {name!r}: requires must be a subset of {list(allowed_requires)}")
    return load_zones({"zones": raw})


class ZoneStore:
    """The live zone set of one running router, editable at runtime (dashboard) and remembered across restarts.

    The live file is separate from the shipped `configs/zones_lab.yaml` on purpose: `make deploy-board` copies
    configs/ over the board's, which would silently undo a zone drawn on site. `reset()` deletes the live file and
    goes back to the shipped zones.
    """

    def __init__(self, router, live_file, default_zones: list[Zone], allowed_requires: tuple[str, ...]):
        from pathlib import Path

        self.router = router
        self.live_file = Path(live_file)
        self.default = list(default_zones)
        self.allowed = tuple(allowed_requires)

    def current(self) -> list[dict]:
        return [zone_to_dict(z) for z in self.router.zones]

    def apply(self, raw: object) -> list[dict]:
        import os

        import yaml

        zones = validate_zone_dicts(raw, self.allowed)
        self.router.set_zones(zones)
        self.live_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.live_file.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump({"zones": [zone_to_dict(z) for z in zones]}, sort_keys=False, allow_unicode=True))
        os.replace(tmp, self.live_file)          # atomic: a power cut never leaves half a file
        return self.current()

    def reset(self) -> list[dict]:
        self.live_file.unlink(missing_ok=True)
        self.router.set_zones(list(self.default))
        return self.current()


def load_live_zones(live_file, allowed_requires: tuple[str, ...] = ()) -> list[Zone] | None:
    """Zones saved by the editor, or None if there is no (valid) live file. A broken file must not stop the monitor
    from starting: log and fall back to the shipped zones."""
    import logging
    from pathlib import Path

    import yaml

    p = Path(live_file)
    if not p.is_file():
        return None
    try:
        return validate_zone_dicts((yaml.safe_load(p.read_text()) or {}).get("zones", []), allowed_requires)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("labsafe").warning("ignoring unreadable live zones %s: %s", p, e)
        return None
