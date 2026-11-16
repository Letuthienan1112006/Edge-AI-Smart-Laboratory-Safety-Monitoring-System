"""Phone notification of alarms through a Zalo Bot (bot.zaloplatforms.com): the phone buzzes even when the dashboard is closed.

Only alarms that the decision layer already let through (debounce + cooldown) are sent, and several within `min_interval_s` are folded
into one message, so a crowded zone does not flood the phone. Sending runs on its own thread with a bounded queue: no network, a slow
API or a revoked token never delays the frame loop; a message that cannot be sent is dropped and counted.

The token + chat id live in a JSON file outside git (`secrets/zalo.json`, written by scripts/board/zalo_setup.py). The file is re-checked
while it is missing, so dropping it on the board turns notifications on without a service restart. The token is part of the API URL:
never log URLs or request objects here.

Photo: Zalo's sendPhoto accepts ONLY a public https URL (verified: file upload, base64 and local paths are all refused with 400), so the
person's crop is first uploaded to a temporary file host (default litterbox.catbox.moe, deleted after `photo_expiry`, 1 h) and the URL is
handed to Zalo. That puts a photo of a real person on a third party's server for that hour, reachable by anyone with the link: the owner
chose this on 23/10 (`notify.photo: true`); set it to false for text only. The text goes out first and never waits for the photo; a failed
upload or sendPhoto costs only the photo.

    PYTHONPATH=src python3 -m labsafe.io.notify --test [secrets/zalo.json]     # send a test message (checks internet + token)
"""

from __future__ import annotations

import json
import logging
import uuid
import queue
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger("labsafe.notify")
ZALO_API = "https://bot-api.zaloplatforms.com"


class NotifyError(RuntimeError):
    pass


class ZaloBot:
    def __init__(self, token: str, chat_id: str = "", base_url: str = ZALO_API, timeout: float = 8.0):
        self._token, self.chat_id, self.base_url, self.timeout = token.strip(), str(chat_id), base_url.rstrip("/"), timeout

    def call(self, method: str, payload: dict, timeout: float | None = None) -> dict:
        req = urllib.request.Request(f"{self.base_url}/bot{self._token}/{method}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                j = json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:          # str(e) carries the status only, never the URL (= the token)
            raise NotifyError(f"{method}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise NotifyError(f"{method}: no connection ({getattr(e, 'reason', e)})") from None
        except ValueError:
            raise NotifyError(f"{method}: reply is not JSON") from None
        if not j.get("ok"):
            raise NotifyError(f"{method}: error {j.get('error_code')} {j.get('description', '')}".strip())
        return j

    def send_text(self, text: str) -> dict:
        return self.call("sendMessage", {"chat_id": self.chat_id, "text": text[:2000]})

    def send_photo(self, photo_url: str, caption: str = "") -> dict:
        return self.call("sendPhoto", {"chat_id": self.chat_id, "photo": photo_url, "caption": caption[:2000]}, timeout=30)


UPLOAD_URL = "https://litterbox.catbox.moe/resources/internals/api.php"


def upload_temp(jpeg: bytes, url: str = UPLOAD_URL, expiry: str = "1h", timeout: float = 20.0) -> str:
    """Upload a JPEG to the temporary file host, return its public https URL (plain-text reply of the litterbox API)."""
    b = "----labsafe" + uuid.uuid4().hex
    parts = b""
    for k, v in (("reqtype", "fileupload"), ("time", expiry)):
        parts += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    parts += (f'--{b}\r\nContent-Disposition: form-data; name="fileToUpload"; filename="alarm.jpg"\r\n'
              f"Content-Type: image/jpeg\r\n\r\n").encode() + jpeg + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url, data=parts, headers={"Content-Type": f"multipart/form-data; boundary={b}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.read().decode(errors="replace").strip()
    except urllib.error.HTTPError as e:
        raise NotifyError(f"upload: HTTP {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise NotifyError(f"upload: no connection ({getattr(e, 'reason', e)})") from None
    if not out.startswith("https://"):
        raise NotifyError(f"upload: unexpected reply {out[:80]!r}")
    return out


def load_secret(path: str | Path) -> dict | None:
    p = Path(path).expanduser()
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("token") and d.get("chat_id") else None


ZONE_NAMES = {"san_phong": "Sàn phòng"}          # the same display names as the web page (dashboard.py ZNAME)


def zone_label(name: str) -> str:
    return ZONE_NAMES.get(name, str(name).replace("_", " "))


def format_message(events: list, folded: int = 0, now: float | None = None) -> str:
    """Vietnamese, readable on a lock screen: what, who, where, when."""
    when = datetime.fromtimestamp(now or time.time()).strftime("%H:%M:%S %d/%m")
    ids = ", ".join(f"#{e.track_id}" for e in events if e.track_id is not None) or "?"
    zones = ", ".join(sorted({zone_label(e.detail) for e in events if e.detail})) or "vùng cấm"
    n = len(events)
    head = "⚠️ BÁO ĐỘNG AN TOÀN PHÒNG LAB"
    body = (f"{'Có người' if n == 1 else f'{n} người'} ({ids}) KHÔNG mặc áo lab trong vùng cấm ({zones}).")
    more = f"\n(+{folded} cảnh báo khác cùng lúc)" if folded else ""
    return f"{head}\n{body}{more}\n🕒 {when}\nẢnh bằng chứng: xem trên trang giám sát."


class Notifier:
    def __init__(self, secret_file: str | Path, kinds=("zone_ppe_violation",), min_interval_s: float = 10.0,
                 base_url: str = ZALO_API, max_queue: int = 20, recheck_s: float = 30.0, clock: Callable[[], float] = time.time,
                 photo: bool = False, upload_url: str = UPLOAD_URL, photo_expiry: str = "1h"):
        self.secret_file, self.kinds, self.min_interval_s = Path(secret_file).expanduser(), set(kinds), float(min_interval_s)
        self.base_url, self.recheck_s, self.clock = base_url, recheck_s, clock
        self.photo, self.upload_url, self.photo_expiry = bool(photo), upload_url, photo_expiry
        self.sent = self.failed = self.dropped = self.photos_sent = self.photos_failed = 0
        self._last_photo_err = ""
        self._bot: ZaloBot | None = None
        self._last_check = -1e18
        self._last_sent_t = -1e18
        self._pending: list = []            # alarms held back by min_interval_s, folded into the next message
        self._said_missing = False
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._last_err = ""
        self._thread = threading.Thread(target=self._run, name="notify", daemon=True)
        self._thread.start()
        self._ensure_bot()

    def _ensure_bot(self) -> ZaloBot | None:
        if self._bot is None and self.clock() - self._last_check >= self.recheck_s:
            self._last_check = self.clock()
            s = load_secret(self.secret_file)
            if s is not None:
                self._bot = ZaloBot(s["token"], s["chat_id"], self.base_url)
                log.info("notify: Zalo notifications ON (%s)", self.secret_file)
            elif not self._said_missing:
                self._said_missing = True
                log.info("notify: %s not found -> phone notifications off until it appears (scripts/board/zalo_setup.py)", self.secret_file)
        return self._bot

    def on_alarm(self, events: list, crops: list | None = None) -> None:
        """Call EVERY frame with that frame's alarms (possibly none) and, aligned with them, the person crops as JPEG bytes (or None).
        Held-back alarms go out once min_interval_s has passed. Never blocks."""
        crops = crops or [None] * len(events)
        self._pending += [(e, c) for e, c in zip(events, crops) if e.kind in self.kinds]
        if not self._pending or self.clock() - self._last_sent_t < self.min_interval_s:
            return
        if self._ensure_bot() is None:
            self._pending.clear()           # nowhere to send: do not let alarms pile up for hours
            return
        now = self.clock()
        evs = [e for e, _ in self._pending]
        # one message: up to 5 people named, the rest counted; the photo is of the first person that has one
        msg = format_message(evs[:5], folded=max(0, len(evs) - 5), now=now)
        shot = next(((e, c) for e, c in self._pending if c), None)
        self._pending, self._last_sent_t = [], now
        try:
            self._q.put_nowait(("text", msg))
            if self.photo and shot is not None:
                self._q.put_nowait(("photo", shot[1], f"Người #{shot[0].track_id} — không mặc áo lab, trong vùng cấm"))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            kind = job[0]
            try:
                if kind == "text":
                    self._bot.send_text(job[1])
                    self.sent += 1
                else:
                    url = upload_temp(job[1], self.upload_url, self.photo_expiry)
                    self._bot.send_photo(url, job[2])
                    self.photos_sent += 1
            except Exception as e:  # noqa: BLE001 -- a phone message must never take the monitoring loop down
                if kind == "text":
                    self.failed += 1
                    last = "_last_err"
                else:
                    self.photos_failed += 1
                    last = "_last_photo_err"
                if str(e) != getattr(self, last):        # one line per new error, not one per alarm
                    log.warning("notify: %s not sent: %s", "message" if kind == "text" else "photo", e)
                    setattr(self, last, str(e))

    def close(self, timeout: float = 5.0) -> None:
        self._q.put(None)
        self._thread.join(timeout)


def build_notifier(cfg: dict | None) -> Notifier | None:
    cfg = cfg or {}
    if not cfg.get("enabled"):
        return None
    if cfg.get("channel", "zalo") != "zalo":
        raise ValueError(f"notify.channel {cfg.get('channel')!r}: only 'zalo' is implemented")
    return Notifier(cfg.get("secret_file", "secrets/zalo.json"), kinds=cfg.get("kinds", ["zone_ppe_violation"]),
                    min_interval_s=float(cfg.get("min_interval_s", 10.0)), base_url=cfg.get("base_url", ZALO_API),
                    photo=bool(cfg.get("photo", False)), upload_url=cfg.get("upload_url", UPLOAD_URL),
                    photo_expiry=str(cfg.get("photo_expiry", "1h")))


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="?", const="secrets/zalo.json", help="secret file; sends one test message")
    ap.add_argument("--test-photo", nargs="?", const="secrets/zalo.json",
                    help="secret file; uploads a generated NON-person test image to the temporary host and sends it (checks the photo path)")
    a = ap.parse_args()
    s = load_secret(a.test or a.test_photo or "")
    if not (a.test or a.test_photo):
        ap.error("give --test or --test-photo")
    if s is None:
        sys.exit(f"{a.test or a.test_photo}: missing or without token/chat_id (run scripts/board/zalo_setup.py on the laptop)")
    if a.test_photo:
        import cv2
        import numpy as np

        img = np.full((300, 400, 3), (70, 70, 190), np.uint8)
        cv2.putText(img, "ANH THU", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        ok, buf = cv2.imencode(".jpg", img)
        try:
            url = upload_temp(buf.tobytes())
            ZaloBot(s["token"], s["chat_id"]).send_photo(url, "Ảnh thử (không có người)")
        except NotifyError as e:
            sys.exit(f"photo NOT sent: {e}")
        print(f"photo sent via {url.split('/')[2]} (deleted by the host after about an hour): check Zalo")
        sys.exit(0)
    try:
        ZaloBot(s["token"], s["chat_id"]).send_text(f"✅ Thiết bị giám sát lab gửi thử lúc {datetime.now():%H:%M:%S %d/%m}: kết nối Zalo OK.")
    except NotifyError as e:
        sys.exit(f"NOT sent: {e}")
    print("sent: check Zalo on the phone")
