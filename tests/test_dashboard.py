"""Dashboard HTTP endpoints, exercised over a real socket on a free port (no camera, no model)."""
from __future__ import annotations

import json
import socket
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.io.dashboard import Dashboard, PAGE  # noqa: E402
from labsafe.pipeline.decision import SafetyEvent  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def test_status_events_snapshot_timeline_and_page():
    port = _free_port()
    d = Dashboard("127.0.0.1", port)
    base = f"http://127.0.0.1:{port}"
    try:
        frame = np.zeros((480, 640, 3), np.uint8)
        summary = {"persons": 3, "wearing": 1, "not_wearing": 2, "in_zone": 2, "violating": 1, "people": []}
        ev = SafetyEvent("zone_ppe_violation", "high", 7, detail="san_phong", extra={"missing": ["lab_coat"]})
        _, person_jpg = __import__("cv2").imencode(".jpg", np.full((200, 80, 3), 128, np.uint8))
        d.publish(frame, {"severity": "high", "fps": 9.5, "summary": summary, "events": [ev.as_dict()]}, [ev],
                  [person_jpg.tobytes()])
        d.publish(frame, {"severity": "high", "fps": 9.5, "summary": summary, "events": []}, [])

        _, ctype, body = _get(base + "/status")
        st = json.loads(body)
        assert "json" in ctype and st["summary"]["not_wearing"] == 2 and st["frame_age_s"] < 5 and "server_now" in st

        evs = json.loads(_get(base + "/events")[2])["events"]
        assert [e["kind"] for e in evs] == ["zone_ppe_violation"] and evs[0]["id"] == 1
        assert json.loads(_get(base + "/events?since=1")[2])["events"] == []

        code, ctype, jpg = _get(base + "/snap/1")
        assert code == 200 and ctype == "image/jpeg" and jpg[:2] == b"\xff\xd8"
        code, ctype, person = _get(base + "/crop/1")
        assert code == 200 and person == person_jpg.tobytes() and evs[0]["has_crop"] is True

        samples = json.loads(_get(base + "/timeline")[2])["samples"]
        assert len(samples) == 1 and samples[0][1:] == [3, 2, 1]

        _, ctype, page = _get(base + "/")
        assert "text/html" in ctype and "Không mặc áo lab" in page.decode()
    finally:
        d.close()


def test_snapshot_of_unknown_alarm_is_404():
    port = _free_port()
    d = Dashboard("127.0.0.1", port)
    try:
        try:
            _get(f"http://127.0.0.1:{port}/snap/99")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        d.close()


def test_page_backs_off_failed_api_polling_without_overlapping_requests():
    assert "function backoffPoll" in PAGE
    assert "backoffPoll(status,500,5000)" in PAGE
    assert "backoffPoll(events,1000,5000)" in PAGE
    assert "backoffPoll(timeline,5000,15000)" in PAGE
    assert "setInterval(status" not in PAGE
    assert "setInterval(events" not in PAGE
    assert "setInterval(timeline" not in PAGE


def test_status_carries_system_gauges_and_the_page_has_the_performance_panel():
    port = _free_port()
    d = Dashboard("127.0.0.1", port)
    try:
        sysd = {"cpu_pct": 41.5, "ram_pct": 30.0, "temp_c": 52.3, "npu_pct": 75.0, "npu_src": "estimate", "infer_ms": 25.1}
        d.publish(np.zeros((480, 640, 3), np.uint8), {"severity": "none", "fps": 29.4, "summary": {"persons": 0, "people": []},
                                                      "events": [], "sys": sysd}, [])
        st = json.loads(_get(f"http://127.0.0.1:{port}/status")[2])
        assert st["sys"]["temp_c"] == 52.3 and st["sys"]["npu_src"] == "estimate"
        page = _get(f"http://127.0.0.1:{port}/")[2].decode()
        assert 'id="perfList"' in page and "Hiệu năng hệ thống" in page
        assert "estimate" not in page or "ước tính" in page            # the page labels an estimated NPU figure as such
    finally:
        d.close()
