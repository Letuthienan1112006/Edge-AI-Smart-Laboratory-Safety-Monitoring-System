"""Zalo phone notification: against a local fake of the Bot API (no internet, no real token)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from labsafe.io.notify import Notifier, NotifyError, ZaloBot, build_notifier, format_message, upload_temp
from labsafe.pipeline.decision import SafetyEvent

TOKEN = "123:secret-token"


@pytest.fixture
def api():
    got, mode = [], {"status": 200, "ok": True, "delay": 0.0, "upload": "https://litter.example/Ab12Cd.jpg", "upload_delay": 0.0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/upload":                     # the temporary file host
                time.sleep(mode["upload_delay"])
                got.append(("/upload", raw))
                out = mode["upload"].encode()
                self.send_response(200 if out else 500)
                self.end_headers()
                self.wfile.write(out)
                return
            body = json.loads(raw)
            time.sleep(mode["delay"])
            got.append((self.path, body))
            out = json.dumps({"ok": True, "result": {"message_id": "m1"}} if mode["ok"] else
                             {"ok": False, "error_code": 400, "description": "bad chat"}).encode()
            self.send_response(mode["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", got, mode
    srv.shutdown()


def secret(tmp_path):
    p = tmp_path / "zalo.json"
    p.write_text(json.dumps({"token": TOKEN, "chat_id": "chat42"}))
    return p


def ev(tid, kind="zone_ppe_violation", zone="san_phong"):
    return SafetyEvent(kind, "high", tid, detail=zone)


def wait_for(cond, s=3.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < s:
        time.sleep(0.02)
    return cond()


def test_alarm_sends_one_zalo_message(api, tmp_path):
    url, got, _ = api
    n = Notifier(secret(tmp_path), base_url=url)
    n.on_alarm([ev(5)])
    assert wait_for(lambda: n.sent == 1)
    path, body = got[0]
    assert path == f"/bot{TOKEN}/sendMessage" and body["chat_id"] == "chat42"
    assert "#5" in body["text"] and "KHÔNG mặc áo lab" in body["text"] and "Sàn phòng" in body["text"] and "san_phong" not in body["text"]
    n.close()


def test_alarms_within_interval_are_folded_not_flooded(api, tmp_path):
    url, got, _ = api
    t = [1000.0]
    n = Notifier(secret(tmp_path), base_url=url, min_interval_s=10, clock=lambda: t[0])
    n.on_alarm([ev(1)])
    for i in range(2, 5):                     # three more alarms 1-3 s later: held back
        t[0] += 1
        n.on_alarm([ev(i)])
    n.on_alarm([])
    assert wait_for(lambda: n.sent == 1) and len(got) == 1
    t[0] += 10                                # interval over: the held-back ones go out together, even on a frame with no alarm
    n.on_alarm([])
    assert wait_for(lambda: n.sent == 2)
    assert all(f"#{i}" in got[1][1]["text"] for i in (2, 3, 4))
    n.close()


def test_other_kinds_and_empty_frames_send_nothing(api, tmp_path):
    url, got, _ = api
    n = Notifier(secret(tmp_path), base_url=url)
    n.on_alarm([ev(1, kind="fall")])
    n.on_alarm([])
    time.sleep(0.2)
    assert got == [] and n.sent == 0
    n.close()


def test_api_error_or_no_network_never_raises_into_the_frame_loop(api, tmp_path):
    url, _, mode = api
    mode["ok"] = False
    n = Notifier(secret(tmp_path), base_url=url, min_interval_s=0)
    n.on_alarm([ev(1)])
    assert wait_for(lambda: n.failed == 1)
    dead = Notifier(secret(tmp_path), base_url="http://127.0.0.1:9", min_interval_s=0)   # nothing listens on port 9
    dead.on_alarm([ev(1)])
    assert wait_for(lambda: dead.failed == 1)
    n.close(), dead.close()


def test_slow_api_does_not_block_on_alarm(api, tmp_path):
    url, _, mode = api
    mode["delay"] = 1.0
    n = Notifier(secret(tmp_path), base_url=url, min_interval_s=0)
    t0 = time.perf_counter()
    for i in range(50):
        n.on_alarm([ev(i)])
    assert time.perf_counter() - t0 < 0.1     # queued, sent off-thread; a full queue drops instead of waiting
    assert n.dropped > 0
    n.close(timeout=0.1)


def test_missing_secret_is_off_and_picked_up_later_without_restart(api, tmp_path):
    url, got, _ = api
    t = [0.0]
    p = tmp_path / "zalo.json"
    n = Notifier(p, base_url=url, min_interval_s=0, recheck_s=30, clock=lambda: t[0])
    n.on_alarm([ev(1)])
    assert n._pending == [] and got == []     # nowhere to send: dropped, not piled up
    p.write_text(json.dumps({"token": TOKEN, "chat_id": "c"}))
    t[0] += 31
    n.on_alarm([ev(2)])
    assert wait_for(lambda: n.sent == 1)
    n.close()


def test_errors_never_contain_the_token():
    bot = ZaloBot(TOKEN, "c", base_url="http://127.0.0.1:9")
    with pytest.raises(NotifyError) as e:
        bot.send_text("x")
    assert "secret-token" not in str(e.value)


def test_build_notifier_off_unless_enabled():
    assert build_notifier(None) is None and build_notifier({"enabled": False}) is None
    with pytest.raises(ValueError):
        build_notifier({"enabled": True, "channel": "sms"})


def test_message_names_people_and_folded_count():
    m = format_message([ev(3), ev(7)], folded=2, now=0)
    assert "2 người (#3, #7)" in m and "+2" in m


JPEG = b"\xff\xd8\xff\xe0fake-person-crop\xff\xd9"


def photo_notifier(url, tmp_path, **kw):
    return Notifier(secret(tmp_path), base_url=url, min_interval_s=0, photo=True, upload_url=url + "/upload", **kw)


def test_photo_is_uploaded_then_sent_to_zalo_after_the_text(api, tmp_path):
    url, got, _ = api
    n = photo_notifier(url, tmp_path)
    n.on_alarm([ev(5)], [JPEG])
    assert wait_for(lambda: n.photos_sent == 1 and n.sent == 1)
    paths = [g[0] for g in got]
    assert paths[0].endswith("/sendMessage") and paths[1] == "/upload" and paths[2].endswith("/sendPhoto")   # text first, never behind the photo
    assert JPEG in got[1][1] and b'name="time"' in got[1][1] and b"1h" in got[1][1]
    photo = got[2][1]
    assert photo["photo"] == "https://litter.example/Ab12Cd.jpg" and photo["chat_id"] == "chat42" and "#5" in photo["caption"]
    n.close()


def test_failed_upload_costs_only_the_photo(api, tmp_path):
    url, got, mode = api
    mode["upload"] = ""                                   # the file host answers 500
    n = photo_notifier(url, tmp_path)
    n.on_alarm([ev(5)], [JPEG])
    assert wait_for(lambda: n.sent == 1 and n.photos_failed == 1)
    assert n.photos_sent == 0 and not any(g[0].endswith("/sendPhoto") for g in got)
    n.close()


def test_host_reply_that_is_not_an_https_url_is_refused(api):
    url, _, mode = api
    mode["upload"] = "Error: file too large"
    with pytest.raises(NotifyError):
        upload_temp(JPEG, url + "/upload")
    mode["upload"] = "http://insecure.example/a.jpg"      # Zalo refuses anything but https
    with pytest.raises(NotifyError):
        upload_temp(JPEG, url + "/upload")


def test_no_photo_without_crop_or_when_photo_is_off(api, tmp_path):
    url, got, _ = api
    n = photo_notifier(url, tmp_path)
    n.on_alarm([ev(1)], [None])                           # evidence disabled / person not visible: text only
    assert wait_for(lambda: n.sent == 1)
    off = Notifier(secret(tmp_path), base_url=url, min_interval_s=0, photo=False, upload_url=url + "/upload")
    off.on_alarm([ev(2)], [JPEG])
    assert wait_for(lambda: off.sent == 1)
    time.sleep(0.2)
    assert not any(g[0] in ("/upload",) or g[0].endswith("/sendPhoto") for g in got)
    n.close(), off.close()


def test_slow_photo_upload_never_delays_the_next_text_or_the_frame_loop(api, tmp_path):
    url, got, mode = api
    mode["upload_delay"] = 1.0
    n = photo_notifier(url, tmp_path)
    t0 = time.perf_counter()
    n.on_alarm([ev(1)], [JPEG])
    assert time.perf_counter() - t0 < 0.05
    assert wait_for(lambda: n.sent == 1)                  # the text is out while the photo is still uploading
    assert n.photos_sent == 0
    assert wait_for(lambda: n.photos_sent == 1, s=5)
    n.close()
