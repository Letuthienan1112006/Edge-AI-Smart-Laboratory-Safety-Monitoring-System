#!/usr/bin/env python3
"""Laptop side: measure the board's energy with an inline USB-C meter (e.g. KEWEISI KWS-1902C) that shows
cumulative Wh, for: idle, adaptive, parallel (same L3 config and replay video as the 3x15-min benchmark, #39).

The meter has no PC link, so the person reads the mWh counter and types it in; the board side is scripted (ssh).
The workload keeps running UNTIL the end reading is typed, and the duration is measured between the two typed readings,
so a late answer cannot mix idle energy into a run (first version did: a 95-minute late answer turned adaptive into
"6.8 W", KNOWN_ISSUES #41). Average power = delta mWh / time between the readings. A mid-phase instantaneous W reading
is asked as a cross-check. No meter reset is needed.

    python3 scripts/bench/energy_session.py              # 1 pair, ~50 min
    python3 scripts/bench/energy_session.py --pairs 2    # repeat adaptive/parallel twice

Output: runs/energy/<stamp>/readings.csv (+ the board's bench folders under runs/board/energy/).
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="orangepi-lan")
ap.add_argument("--pairs", type=int, default=1)
ap.add_argument("--idle-s", type=int, default=600)
ap.add_argument("--run-s", type=int, default=900)
ap.add_argument("--product-s", type=int, default=600, help="seconds for the PRODUCT phase (live config, stage A only, dashboard on); 0 = skip")
ap.add_argument("--config", default="configs/exp/L3_lab_v5_full_async_int8_640.yaml")
ap.add_argument("--classes", default="configs/classes_lab_v5_full.yaml")
ap.add_argument("--cool-to-c", type=float, default=42.0)
a = ap.parse_args()

out = Path("runs/energy") / datetime.now().strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True, exist_ok=True)
rows: list[dict] = []


def ssh(cmd: str, check: bool = True) -> str:
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", a.host, cmd], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"ssh failed: {cmd}\n{r.stderr}")
    return r.stdout.strip()


def beep():
    print("\a", end="", flush=True)


def read_wh(label: str) -> float:
    """The KWS-1902C shows cumulative energy as mWh (e.g. 039637mWh); the person types that number."""
    beep()
    while True:
        s = input(f"\n>>> [{label}] Gõ số mWh đang hiện trên đồng hồ (dòng có chữ mWh, vd 039637): ").strip()
        try:
            return float(s.replace(",", ".").lower().replace("mwh", "")) / 1000.0
        except ValueError:
            print("   Không đọc được số, gõ lại.")


def temp_c() -> float:
    return int(ssh("cat /sys/class/thermal/thermal_zone0/temp")) / 1000.0


def read_w(label: str) -> float | None:
    beep()
    s = input(f"\n>>> [{label}] (kiểm tra chéo) Gõ số W đang hiện trên đồng hồ, hoặc Enter để bỏ qua: ").strip()
    try:
        return float(s.replace(",", ".").lower().replace("w", "")) if s else None
    except ValueError:
        return None


PIDFILE = "~/AI_Challenge/runs/energy_app.pid"


def app_running() -> bool:
    return ssh(f"p=$(cat {PIDFILE} 2>/dev/null) && kill -0 $p 2>/dev/null && echo yes || echo no", check=False) == "yes"


def stop_app():
    # by PID, never by name: `pkill -f <name>` sent over ssh also matches the ssh shell's own command line (#36)
    ssh(f"p=$(cat {PIDFILE} 2>/dev/null) || exit 0; kill -INT $p 2>/dev/null; for i in 1 2 3 4 5 6 7 8; do "
        f"kill -0 $p 2>/dev/null || break; sleep 1; done; kill -9 $p 2>/dev/null; rm -f {PIDFILE}; true", check=False)


def phase(name: str, seconds: int, board_cmd: str | None):
    if board_cmd is None:
        print(f"\n== {name}: board để NGHỈ (không chạy gì), đo ~{seconds // 60} phút.")
    else:
        print(f"\n== {name}: khởi động trên board, chờ 60 s cho ổn định rồi đo ~{seconds // 60} phút.")
        # the inner shell writes ITS OWN pid ($$) and then `exec`s the app, so the file holds the app's PID ($! of
        # setsid would be a different process: setsid forks). sshd keeps this session open while the background job
        # lives, so the call is cut after a few seconds on purpose -- the job is detached (setsid+nohup) and keeps
        # running; app_running() below is the real check. (`ssh -f` did NOT start the job at all on this board.)
        try:
            subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=8", a.host,
                            f"cd ~/AI_Challenge && setsid nohup bash -c 'echo $$ > {PIDFILE}; exec {board_cmd}' "
                            "> /dev/null 2>&1 < /dev/null &"], timeout=8)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(60)
        if not app_running():
            sys.exit(f"{name}: the app did not start on the board (see runs/energy_*.log there)")
    wh0 = read_wh(f"{name} - BẮT ĐẦU")
    t0 = time.time()
    time.sleep(seconds / 2)
    w_mid = read_w(f"{name} - GIỮA CHỪNG")
    rest = seconds - (time.time() - t0)
    if rest > 0:
        time.sleep(rest)
    wh1 = read_wh(f"{name} - KẾT THÚC")          # the workload is STILL running here
    t1 = time.time()
    temp = temp_c()
    if board_cmd is not None:
        stop_app()
    dur = t1 - t0
    w = (wh1 - wh0) / (dur / 3600.0) if dur > 0 else float("nan")
    row = {"phase": name, "start_wh": wh0, "end_wh": wh1, "t_start": round(t0, 1), "t_end": round(t1, 1),
           "seconds": round(dur, 1), "avg_w": round(w, 3), "mid_w_display": w_mid, "temp_end_c": temp}
    rows.append(row)
    with (out / "readings.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row))
        wr.writeheader()
        wr.writerows(rows)
    chk = "" if w_mid is None else f" (đồng hồ giữa chừng hiện {w_mid:.2f} W)"
    print(f"   -> {name}: {(wh1 - wh0) * 1000:.0f} mWh trong {dur / 60:.1f} phút = TRUNG BÌNH {w:.2f} W{chk}")


def cool_down():
    t = temp_c()
    while t > a.cool_to_c:
        print(f"   chờ board nguội: {t:.1f} °C > {a.cool_to_c} °C", end="\r", flush=True)
        time.sleep(15)
        t = temp_c()
    print(f"   board {t:.1f} °C, tiếp tục.                    ")


subprocess.run(["bash", "scripts/board/sync_board_clock.sh", a.host], check=False)
ssh("touch ~/AI_Challenge/.no_autocapture; sudo -n systemctl stop labsafe-capture.service 2>/dev/null; true", check=False)
if ssh("pgrep -c -f '^python3 -m [l]absafe.app' || true", check=False) not in ("", "0"):
    sys.exit("something is already running on the board (labsafe.app) -- stop it first (Ctrl+C make dashboard)")
print(f"""
Đo năng lượng board. Đồng hồ USB phải nằm GIỮA nguồn và board (nguồn -> đồng hồ -> board).
Mỗi khi máy kêu bíp và hỏi, đọc con số mWh trên màn hình đồng hồ rồi gõ vào. Không cần reset đồng hồ.
Trình tự: nghỉ {a.idle_s // 60} phút, SẢN PHẨM {a.product_s // 60} phút, rồi {a.pairs} cặp (adaptive {a.run_s // 60} phút, parallel {a.run_s // 60} phút). Kết quả: {out}/readings.csv
""")
cool_down()
stop_app()
phase("idle", a.idle_s, None)
# The PRODUCT: what runs at the competition (configs/live/dashboard_lab_board.yaml: stage A only, dashboard, panel). Replayed video (--replay: looped
# at its own FPS, and it never sends Zalo messages), so the load is identical on every run. This is the number the poster needs.
if a.product_s > 0:
    cool_down()
    phase("product", a.product_s,
          "env PYTHONPATH=src python3 -m labsafe.app --config configs/live/dashboard_lab_board.yaml --classes configs/classes_lab_v3.yaml "
          "--source data/lab_replay.mp4 --replay --no-display --gpio off --run-dir runs/energy_product > runs/energy_product.log 2>&1")
# live (not --bench) run so a SIGINT stops it cleanly at the end reading; same config, video pacing and loop as the
# 3x15-min benchmark (--demo-video = paced at the file FPS + looped)
bench = ("env PYTHONPATH=src python3 -m labsafe.app --mode {mode} "
         f"--config {a.config} --classes {a.classes} --source data/lab_replay.mp4 --demo-video "
         "--no-display --gpio off --run-dir runs/energy_{mode}_p{p} > runs/energy_{mode}_p{p}.log 2>&1")
for p in range(1, a.pairs + 1):
    for mode in ("adaptive", "parallel"):
        cool_down()
        phase(f"{mode}_p{p}", a.run_s, bench.format(mode=mode, p=p))

idle = next(r["avg_w"] for r in rows if r["phase"] == "idle")
print("\n================ KẾT QUẢ ================")
for r in rows:
    extra = "" if r["phase"] == "idle" else f"  (trên mức nghỉ: {r['avg_w'] - idle:+.2f} W)"
    print(f"{r['phase']:12s} {r['avg_w']:6.2f} W{extra}")
print(f"\nĐã lưu {out}/readings.csv. Báo Claude để tính năng lượng/khung và ghi vào báo cáo.")
