"""Zones drawn on site: validation of untrusted input, live file, hot swap in the router, HTTP guards."""
from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.io.dashboard import Dashboard  # noqa: E402
from labsafe.models.backends import Detection, build_backend  # noqa: E402
from labsafe.pipeline.fall import FallConfig  # noqa: E402
from labsafe.pipeline.router import AdaptiveRouter  # noqa: E402
from labsafe.pipeline.zones import Zone, ZoneStore, load_live_zones, validate_zone_dicts  # noqa: E402

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
SELF_CROSSING_FLOOR = [
    [0.0467, 0.9994], [0.0112, 0.8272], [0.105, 0.472],
    [0.9042, 0.9476], [0.9156, 0.9903], [0.0455, 0.9949],
    [0.9407, 0.9552],
]


def good(**kw):
    return {"name": "z1", "severity": "high", "polygon": SQUARE, **kw}


@pytest.mark.parametrize("bad", [
    "not a list", [1], [good(name="")], [good(name="x" * 41)], [good(), good()],
    [good(severity="extreme")], [good(polygon=[[0, 0], [1, 1]])], [good(polygon=[[0, 0], [1, 0], [2, 1]])],
    [good(polygon=[[0, 0], [1, 0], ["a", 1]])], [good(polygon=[[0, 0], [1, 0], [True, 1]])],
    [good(requires=["gloves"])], [good(name=f"z{i}") for i in range(9)],
])
def test_validate_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_zone_dicts(bad, allowed_requires=("lab_coat",))


def test_validate_accepts_good_input():
    zs = validate_zone_dicts([good(requires=["lab_coat"])], allowed_requires=("lab_coat",))
    assert zs[0].requires == ("lab_coat",) and len(zs[0].polygon) == 4


def test_validate_rejects_the_self_crossing_floor_polygon_from_the_dashboard():
    """Regression: the saved zone crossed itself at the bottom and made membership flip with bbox jitter."""
    with pytest.raises(ValueError, match="self-intersects"):
        validate_zone_dicts([good(polygon=SELF_CROSSING_FLOOR)], allowed_requires=("lab_coat",))


def test_near_frame_edge_is_snapped_so_a_bottom_clipped_person_stays_in_zone():
    """A click 0.06% above the canvas edge must mean the edge, not leave a sub-pixel escape strip."""
    zs = validate_zone_dicts([good(polygon=[[0.005, 0.5], [0.995, 0.5], [0.995, 0.9994], [0.005, 0.9994]])])
    clipped = Detection(100, 100, 200, 480, 0.9, 0, "person")
    assert zs[0].contains_foot(clipped, 640, 480)
    assert max(y for _, y in zs[0].polygon) == 1.0


def test_two_clicks_snapped_into_the_same_corner_become_one_vertex():
    """Regression: a drawing that starts and ends near the bottom-left corner (both clicks snap to (0, 1)) was rejected as
    'repeated points' after edge snapping was added; the home floor zone drawn on the board on 17/10 is exactly this shape."""
    zs = validate_zone_dicts([good(polygon=[[0.0029, 0.99], [0.25, 0.69], [0.64, 0.73], [0.69, 0.99], [0.1048, 0.9936], [0.0052, 0.9936]])],
                             allowed_requires=("lab_coat",))
    assert zs[0].polygon[0] == (0.0, 1.0) and zs[0].polygon.count((0.0, 1.0)) == 1 and len(zs[0].polygon) == 5


def test_a_vertex_repeated_far_apart_is_still_rejected():
    with pytest.raises(ValueError, match="repeated points|self-intersects"):
        validate_zone_dicts([good(polygon=[[0.2, 0.2], [0.8, 0.2], [0.5, 0.5], [0.8, 0.8], [0.5, 0.5], [0.2, 0.8]])])


def make_router(zones):
    a = build_backend({"backend": "stub", "imgsz": 128, "conf": 0.3}, ["person"])
    return AdaptiveRouter(a, None, None, zones, FallConfig(), mode="adaptive")


def test_store_hot_swaps_persists_and_resets(tmp_path):
    default = [Zone("old", "high", [(0, 0), (1, 0), (1, 1)], ("lab_coat",))]
    r = make_router(default)
    live = tmp_path / "zones_live.yaml"
    st = ZoneStore(r, live, default, ("lab_coat",))
    out = st.apply([good(requires=["lab_coat"])])
    assert [z.name for z in r.zones] == ["z1"] and out[0]["name"] == "z1"
    assert r.attire_classes == frozenset({"lab_coat"})
    again = load_live_zones(live, ("lab_coat",))             # what the next start would load
    assert again is not None and again[0].name == "z1"
    st.reset()
    assert [z.name for z in r.zones] == ["old"] and not live.exists()
    assert load_live_zones(live, ("lab_coat",)) is None


def test_bad_apply_changes_nothing(tmp_path):
    default = [Zone("old", "high", [(0, 0), (1, 0), (1, 1)])]
    r = make_router(default)
    st = ZoneStore(r, tmp_path / "z.yaml", default, ())
    with pytest.raises(ValueError):
        st.apply([good(polygon=[[0, 0]])])
    assert [z.name for z in r.zones] == ["old"] and not (tmp_path / "z.yaml").exists()


def test_broken_live_file_does_not_stop_startup(tmp_path):
    p = tmp_path / "z.yaml"
    p.write_text("zones: [{{{ not yaml")
    assert load_live_zones(p, ()) is None


def test_new_zone_is_used_by_the_very_next_frame():
    r = make_router([])
    frame = np.zeros((480, 640, 3), np.uint8)
    r.stage_a = type("A", (), {"name": "s", "imgsz": 640, "last_latency_ms": 0.0,
                               "__call__": lambda s, f: [Detection(100, 100, 200, 400, 0.9, 0, "person")]})()
    ev, _ = r.process(frame, now=0.0)
    assert ev == []
    r.set_zones(validate_zone_dicts([good()], ()))
    ev, _ = r.process(frame, now=0.1)
    assert [e.kind for e in ev] == ["zone_intrusion"]


# ---- HTTP -----------------------------------------------------------------

def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _post(url, body, headers=None, ctype="application/json"):
    h = {"Content-Type": ctype, "X-Requested-With": "labsafe", **(headers or {})}
    h = {k: v for k, v in h.items() if v is not None}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, (json.loads(e.read() or b"{}") if e.headers.get("Content-Type", "").startswith("application/json") else {})


def test_http_get_post_and_guards(tmp_path):
    r = make_router([])
    st = ZoneStore(r, tmp_path / "z.yaml", [], ("lab_coat",))
    port = _port()
    d = Dashboard("127.0.0.1", port, zones=st)
    base = f"http://127.0.0.1:{port}"
    try:
        got = json.loads(urllib.request.urlopen(base + "/zones", timeout=5).read())
        assert got["zones"] == [] and got["allowed_requires"] == ["lab_coat"]
        code, j = _post(base + "/zones", {"zones": [good(requires=["lab_coat"])]})
        assert code == 200 and j["ok"] and r.zones[0].name == "z1"
        code, j = _post(base + "/zones", {"zones": [good(polygon=[[0, 0]])]})
        assert code == 400 and not j["ok"] and "polygon" in j["error"]
        code, j = _post(base + "/zones", {"zones": [good(polygon=SELF_CROSSING_FLOOR)]})
        assert code == 400 and not j["ok"] and "self-intersects" in j["error"]
        # a foreign page cannot do this: wrong content type, missing marker header, foreign Host
        assert _post(base + "/zones", {"zones": []}, ctype="text/plain")[0] == 403
        assert _post(base + "/zones", {"zones": []}, headers={"X-Requested-With": None})[0] == 403
        assert _post(base + "/zones", {"zones": []}, headers={"Host": "evil.example"})[0] == 403
        assert [z.name for z in r.zones] == ["z1"]                      # none of the refused calls changed anything
        code, j = _post(base + "/zones/reset", {})
        assert code == 200 and r.zones == []
        assert _post(base + "/nope", {})[0] == 404
    finally:
        d.close()


def test_without_a_store_the_editor_is_off(tmp_path):
    port = _port()
    d = Dashboard("127.0.0.1", port)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/zones", timeout=5)
        assert e.value.code == 404
        assert _post(f"http://127.0.0.1:{port}/zones", {"zones": []})[0] == 404
    finally:
        d.close()
