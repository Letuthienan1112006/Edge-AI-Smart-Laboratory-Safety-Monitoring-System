#!/usr/bin/env python3
"""Laptop side: is the USB-C meter trustworthy? Times the readings with the laptop clock (a person typing and a chat round trip are NOT 1 minute).
  1) baseline: you type the mWh counter at two beeps exactly 60 s apart; the script computes the average W and compares it with the W the meter shows.
  2) load: the script runs 8 busy CPU processes on the board for ~70 s; at the beep you type A and W shown. They must rise clearly.
    python3 scripts/meter_check.py            # board running, meter between the charger and the board
Needs nothing stopped on the board (the live service keeps running; it is part of the baseline).
"""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="orangepi-lan")
ap.add_argument("--seconds", type=int, default=60)
a = ap.parse_args()

BURN = """
import multiprocessing as mp, time
def burn(t):
    e = time.time() + t
    while time.time() < e: pass
if __name__ == "__main__":
    ps = [mp.Process(target=burn, args=(70,)) for _ in range(8)]
    [p.start() for p in ps]; [p.join() for p in ps]
"""


def beep():
    print("\a", end="", flush=True)


def ask(prompt, cast=float):
    beep()
    while True:
        s = input(prompt).strip().replace(",", ".").lower()
        for suffix in ("mwh", "mah", "w", "a", "v"):
            s = s.replace(suffix, "")
        try:
            return cast(s)
        except ValueError:
            print("   không đọc được số, gõ lại.")


print("== 1/2 BASELINE ==  (board chạy bình thường, đừng động vào board/cáp)")
input("Nhìn vào màn hình đồng hồ. Khi sẵn sàng đọc số mWh thì nhấn Enter, rồi gõ NGAY số mWh đang hiện... ")
w_disp = ask("   W đang hiện (dòng W, vd 3.05): ")
v_disp = ask("   V đang hiện: ")
i_disp = ask("   A đang hiện: ")
t0 = time.time(); m0 = ask("   >>> mWh ngay bây giờ (gõ rồi Enter): ")
t0 = time.time()      # the clock starts when the number is typed
print(f"Đợi {a.seconds} giây... (đồng hồ bắt đầu từ lúc bạn gõ số mWh)")
while time.time() - t0 < a.seconds - 8:
    time.sleep(1)
print("Chuẩn bị: sẽ có tiếng bíp sau vài giây, lúc đó đọc số mWh và gõ.")
while time.time() - t0 < a.seconds:
    time.sleep(0.1)
m1 = ask("   >>> BÍP: gõ số mWh ngay bây giờ: ")
t1 = time.time()
dt = t1 - t0
w_avg = (m1 - m0) / 1000.0 / (dt / 3600.0)
print(f"\n   tăng {m1 - m0:.0f} mWh trong {dt:.0f} s  =>  trung bình {w_avg:.2f} W")
print(f"   đồng hồ hiện {w_disp:.2f} W; V x A = {v_disp * i_disp:.2f} W")
rel = abs(w_avg - w_disp) / max(w_disp, 1e-6)
print("   " + ("KHỚP (lệch %.0f%%): bộ đếm mWh đáng tin." % (rel * 100) if rel < 0.15 else
                "LỆCH %.0f%%: bộ đếm hoặc cách bạn gõ có vấn đề; báo Claude." % (rel * 100)))

print("\n== 2/2 TẢI CPU ==  (board sẽ chạy 8 tiến trình 70 giây)")
input("Nhấn Enter để bắt đầu tải, rồi nhìn đồng hồ... ")
proc = subprocess.Popen(["ssh", a.host, "python3", "-"], stdin=subprocess.PIPE, text=True)
proc.stdin.write(BURN); proc.stdin.close()
time.sleep(25)
w_load = ask("   >>> BÍP (đang tải): gõ W đang hiện: ")
i_load = ask("   >>> A đang hiện: ")
proc.wait()
print(f"\n   nghỉ -> tải: {w_disp:.2f} W -> {w_load:.2f} W ({w_load - w_disp:+.2f} W), {i_disp:.2f} A -> {i_load:.2f} A")
print("   " + ("ĐỒNG HỒ PHẢN ỨNG VỚI TẢI: tốt." if w_load - w_disp > 0.8 else
                "Gần như không đổi: đồng hồ đang không đo dòng của board (nối sai chỗ?). Báo Claude."))
