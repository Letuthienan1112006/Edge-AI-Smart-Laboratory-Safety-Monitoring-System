#!/usr/bin/env python3
"""Live camera view with framing checks -- run on the board, look at it in a browser on the laptop.

    make camera                      # on the laptop: opens an SSH tunnel and the browser
    python3 scripts/board/camera_view.py   # on the board (what `make camera` runs); Ctrl+C frees the camera

Serves ONLY on 127.0.0.1 (reach it through the SSH tunnel): a live camera feed must not be open to
the network. Shows the MJPEG stream with a rule-of-thirds grid, the people stage A finds, and live
advice computed from the picture:
  * backlight  : share of blown-out (>=250) pixels -- a bright window behind people hides them
  * too dark   : mean brightness
  * person size: tallest person's height as a share of the frame (a goggle is a few pixels when the
                 person is small; the number lets you compare camera positions)
Thresholds are rules of thumb, not measured limits; the point is to compare placements.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="camera index or a video file (looped, for testing)")
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--detector", default="rknn", choices=["rknn", "onnx", "none"])
ap.add_argument("--weights", default=None)
ap.add_argument("--jpeg-quality", type=int, default=75)
a = ap.parse_args()

HTML = """<!doctype html><meta charset=utf-8><title>Camera check</title>
<style>body{margin:0;background:#111;color:#eee;font:15px system-ui;display:flex;flex-wrap:wrap;gap:16px;padding:16px}
img{max-width:min(100%,960px);border:2px solid #444}#p{min-width:320px;max-width:520px}
.ok{color:#6f6}.warn{color:#fc4}.bad{color:#f66}h3{margin:.2em 0}a{color:#8cf}</style>
<div><img src=/stream></div><div id=p><h3>Kiểm tra góc camera</h3><div id=s>đang đo ...</div>
<p><a href=/snap.jpg download=camera_check.jpg>Lưu ảnh này</a> &middot; Ctrl+C ở terminal để tắt (nhả camera)</p>
<small>Lưới chia 3 phần. Ngưỡng là quy tắc kinh nghiệm để so sánh vị trí đặt camera, không phải giới hạn đã đo.</small></div>
<script>async function t(){try{const r=await (await fetch('/stats')).json();
document.getElementById('s').innerHTML=r.lines.map(l=>'<div class='+l[0]+'>'+l[1]+'</div>').join('')}catch(e){}}
setInterval(t,1000);t()</script>"""

state = {"frame": None, "jpeg": None, "stats": {"lines": [["warn", "đang khởi động camera ..."]]}, "dets": []}
lock = threading.Lock()
stop = threading.Event()


def advise(img: np.ndarray, dets: list) -> dict:
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blown = float((g >= 250).mean() * 100)
    dark = float((g <= 20).mean() * 100)
    mean = float(g.mean())
    L: list[list[str]] = []
    L.append(["ok" if blown < 5 else "warn" if blown < 15 else "bad",
              f"Điểm ảnh cháy sáng: {blown:.1f}% " + ("(ổn)" if blown < 5 else "(ngược sáng nhẹ)" if blown < 15
                                                  else "(NGƯỢC SÁNG: người sẽ tối và khó nhận diện, đổi hướng/che cửa sổ)")])
    L.append(["ok" if 70 < mean < 190 else "warn", f"Độ sáng trung bình: {mean:.0f}/255 " + ("(ổn)" if 70 < mean < 190 else "(tối quá hoặc chói quá)")])
    if dark > 40:
        L.append(["warn", f"{dark:.0f}% khung hình gần đen"])
    if not dets:
        L.append(["warn", "Không thấy người trong khung (đứng vào khung để thử)"])
    else:
        for i, d in enumerate(sorted(dets, key=lambda d: -(d[3] - d[1])), 1):
            x1, y1, x2, y2, s = d
            hp = (y2 - y1) / h * 100
            cut = y1 <= 2 or y2 >= h - 2 or x1 <= 2 or x2 >= w - 2
            verdict = ("ngưỡng: quá nhỏ, kính/găng chỉ còn vài điểm ảnh" if hp < 25 else
                       "ổn cho kiểm tra PPE" if hp < 90 else "quá sát/chiếm gần hết khung")
            L.append(["bad" if hp < 25 or hp > 95 else "ok",
                      f"Người {i}: cao {hp:.0f}% khung, tin cậy {s:.2f}" + (" - BỊ CẮT MÉP KHUNG" if cut else "") + f" ({verdict})"])
            if i >= 4:
                break
    L.append(["ok", f"FPS hiển thị: {state.get('fps', 0):.0f}"])
    return {"lines": L, "blown_pct": round(blown, 1), "mean": round(mean), "people": len(dets)}


def draw(img: np.ndarray, dets: list) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for k in (1, 2):
        cv2.line(out, (w * k // 3, 0), (w * k // 3, h), (255, 255, 255), 1)
        cv2.line(out, (0, h * k // 3), (w, h * k // 3), (255, 255, 255), 1)
    for x1, y1, x2, y2, s in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, f"{(y2-y1)/h*100:.0f}%", (int(x1) + 4, int(y1) + 18), 0, 0.6, (0, 255, 0), 2)
    return out


def camera_loop() -> None:
    src = int(a.source) if a.source.isdigit() else a.source
    try:
        if isinstance(src, int):
            from labsafe.io.camera import CameraSource
            cam = CameraSource(src)
            read = lambda: cam.read(wait_new=True)
        else:
            cap = cv2.VideoCapture(src)

            def read():
                ok, f = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                time.sleep(1 / 25)
                return ok, f
    except RuntimeError as e:
        state["stats"] = {"lines": [["bad", f"Không mở được camera: {e}. Camera đang bị dịch vụ chụp tự động chiếm? "
                                           "Chạy `make camera` (nó tự dừng dịch vụ) hoặc kiểm tra cáp USB."]]}
        return
    n, t0 = 0, time.time()
    last_ok = time.time()
    while not stop.is_set():
        ok, f = read()
        if not ok or f is None:
            if time.time() - last_ok > 5:       # never sit on "starting..." silently
                with lock:
                    state["stats"] = {"lines": [["bad", "Không nhận được hình từ camera/nguồn trong 5 giây. Kiểm tra cáp USB, "
                                                       "camera có đang bị chương trình khác dùng không, hoặc đường dẫn video."]]}
            time.sleep(0.05)
            continue
        last_ok = time.time()
        with lock:
            dets = list(state["dets"])
        img = draw(f, dets)
        okj, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        n += 1
        if time.time() - t0 > 1:
            state["fps"] = n / (time.time() - t0); n, t0 = 0, time.time()
        with lock:
            state["frame"] = f
            if okj:
                state["jpeg"] = buf.tobytes()
            state["stats"] = advise(f, dets)


def detect_loop() -> None:
    if a.detector == "none":
        return
    try:
        import yaml
        from labsafe.models.backends import build_backend
        names = yaml.safe_load(open("configs/classes_coco80.yaml"))["stage_a"]["names"]
        spec = {"backend": a.detector, "imgsz": 640, "conf": 0.35, "iou": 0.5, "keep_classes": ["person"],
                "weights": a.weights or ("runs/export/yolo26n_int8_640.rknn" if a.detector == "rknn" else "runs/export/yolo26n_640.onnx")}
        be = build_backend(spec, names)
    except Exception as e:   # the view still works without detection
        print(f"detector disabled: {type(e).__name__}: {e}")
        return
    while not stop.is_set():
        with lock:
            f = None if state["frame"] is None else state["frame"].copy()
        if f is None:
            time.sleep(0.1)
            continue
        d = [(x.x1, x.y1, x.x2, x.y2, x.score) for x in be(f)]
        with lock:
            state["dets"] = d
        time.sleep(0.05)


class H(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_GET(self):
        if self.path == "/":
            b = HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path.startswith("/stats"):
            with lock:
                b = json.dumps(state["stats"]).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path.startswith("/snap.jpg"):
            with lock:
                b = state["jpeg"]
            if b is None:
                self.send_error(503); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            last = None
            try:
                while not stop.is_set():
                    with lock:
                        b = state["jpeg"]
                    if b is not None and b is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b) + b + b"\r\n")
                        last = b
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=detect_loop, daemon=True).start()
srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
srv.daemon_threads = True
print(f"camera view on http://127.0.0.1:{a.port}  (Ctrl+C to stop and free the camera)", flush=True)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    stop.set()
    srv.server_close()
