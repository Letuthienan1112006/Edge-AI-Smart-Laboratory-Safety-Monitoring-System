#!/usr/bin/env python3
"""Laptop side: connect the lab safety system to YOUR Zalo (phone message on every alarm). Needs internet on the laptop.

    python3 scripts/board/zalo_setup.py --token-file ~/zalo_token.txt   # 1) find your chat id, 2) write secrets/zalo.json, 3) test message
    python3 scripts/board/zalo_setup.py --push                          # copy secrets/zalo.json to the board + test FROM the board

Before step 1: in Zalo, search the OA "Zalo Bot Manager" -> "Tạo bot" -> name starting with "Bot" (e.g. "Bot An Toan Lab"); Zalo sends
you the Bot Token in a message. Save it alone in a text file (do not paste it into chats or commit it). While this script waits, open
your bot in Zalo and send it any message: that message tells the script which chat to notify.
The token is never printed. secrets/ is git-ignored; the file is created readable by you only.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# load notify.py by path: importing the labsafe.io package pulls in OpenCV, which the laptop's system python3 does not have
_spec = importlib.util.spec_from_file_location("notify", Path(__file__).resolve().parents[2] / "src/labsafe/io/notify.py")
_notify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_notify)
ZALO_API, NotifyError, ZaloBot, load_secret = _notify.ZALO_API, _notify.NotifyError, _notify.ZaloBot, _notify.load_secret

SECRET = Path("secrets/zalo.json")


def chats(result) -> list[tuple[str, str]]:
    """(chat_id, display name) from a getUpdates result: one update object or a list of them."""
    out = []
    for u in result if isinstance(result, list) else [result]:
        m = (u or {}).get("message") or {}
        cid = (m.get("chat") or {}).get("id")
        if cid:
            out.append((str(cid), (m.get("from") or {}).get("display_name", "?")))
    return out


def setup(token_file: str, wait_s: float, base_url: str = ZALO_API) -> None:
    token = Path(token_file).expanduser().read_text().strip()
    if ":" not in token:
        sys.exit(f"{token_file} does not look like a Zalo Bot token (expected '<number>:<secret>')")
    bot = ZaloBot(token, base_url=base_url)
    try:                                    # a wrong token (pasted twice, cut short) must fail HERE, not loop on getUpdates for minutes
        me = bot.call("getMe", {}).get("result") or {}
    except NotifyError as e:
        sys.exit(f"token rejected by Zalo ({e}). It is probably pasted twice or cut short: copy ONLY the 'number:string' once.")
    print(f"token OK: bot '{me.get('display_name', '?')}'")
    print(f"Mở Zalo, vào chat với bot của bạn và gửi một tin bất kỳ (ví dụ 'hi'). Đang chờ tối đa {wait_s:.0f} s ...")
    found, t0 = [], time.time()
    while not found and time.time() - t0 < wait_s:
        try:
            found = chats(bot.call("getUpdates", {"timeout": "20"}, timeout=30).get("result"))
        except NotifyError as e:
            if any(c in str(e) for c in ("401", "403", "404")):
                sys.exit(f"Zalo rejected the request ({e}): check the token (copy it again from Zalo's message)")
            if "408" not in str(e) and "no connection" not in str(e):
                print(f"  ({e}; retrying)")
            time.sleep(2)
    if not found:
        sys.exit("no message received: send a message to the bot in Zalo and run again")
    chat_id, name = found[-1]
    SECRET.parent.mkdir(exist_ok=True)
    os.chmod(SECRET.parent, 0o700)
    fd = os.open(SECRET, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"token": token, "chat_id": chat_id, "name": name}, f)
    print(f"chat found: {name}  -> {SECRET} (only you can read it)")
    ZaloBot(token, chat_id, base_url).send_text("✅ Đã kết nối hệ thống giám sát an toàn phòng lab. "
                                      "Khi có người KHÔNG mặc áo lab vào vùng cấm, bạn sẽ nhận tin ở đây.")
    print("test message sent from the laptop: check Zalo. Next: python3 scripts/board/zalo_setup.py --push (board plugged in)")


def push(host: str) -> None:
    if load_secret(SECRET) is None:
        sys.exit(f"{SECRET} missing: run with --token-file first")
    subprocess.run(["ssh", host, "mkdir -p AI_Challenge/secrets && chmod 700 AI_Challenge/secrets"], check=True)
    subprocess.run(["scp", "-q", str(SECRET), f"{host}:AI_Challenge/secrets/zalo.json"], check=True)
    subprocess.run(["ssh", host, "chmod 600 AI_Challenge/secrets/zalo.json"], check=True)
    print("copied to the board. Test message FROM the board (needs internet on the board):")
    r = subprocess.run(["ssh", host, "cd AI_Challenge && PYTHONPATH=src timeout 20 python3 -m labsafe.io.notify --test"])
    if r.returncode != 0:
        print("!! the board could not send: it has no internet. Plug the phone into the board's USB and turn on USB tethering\n"
              "   (Điểm phát sóng -> Chia sẻ Internet qua USB), then run --push again. The live service picks the file up by itself.")
        sys.exit(1)
    print("OK: the running service starts sending alarms to Zalo within ~30 s, no restart needed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--host", default="orangepi-lan")
    ap.add_argument("--wait", type=float, default=120)
    ap.add_argument("--base-url", default=ZALO_API, help=argparse.SUPPRESS)   # tests: a local fake of the API
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    if a.token_file:
        setup(a.token_file, a.wait, a.base_url)
    if a.push:
        push(a.host)
    if not (a.token_file or a.push):
        ap.print_help()
