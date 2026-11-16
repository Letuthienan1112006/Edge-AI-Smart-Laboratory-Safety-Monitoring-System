#!/usr/bin/env python3
"""Does the live system recognise YOUR lab coats here? Guided check against the running service (dashboard /status on the board).

    python3 scripts/eval/coat_check.py                        # no coat, then 'coat 1', 'coat 2' (names asked for)
    python3 scripts/eval/coat_check.py --labels "khong ao" "ao 1" "ao 2" "ao 1 mo khuy" --seconds 20

For each label you stand alone in front of the camera (one person), press Enter, get a 5 s countdown, and the script polls the service for N seconds.
Reports, over the frames where exactly one person is seen: the share of polls in which that person is `wearing` a lab coat (coat box with
score >= attire_min_conf attached to the person, remembered 2 s). A good coat reads close to 100%, 'no coat' close to 0%.
Needs the board reachable by SSH and labsafe-live running (it owns the camera).
"""
import argparse
import json
import subprocess
import sys
import time

REMOTE = r'''
import json, sys, time, urllib.request
dur = float(sys.argv[1]); t0 = time.time()
while time.time() - t0 < dur:
    try:
        j = json.load(urllib.request.urlopen("http://127.0.0.1:8000/status", timeout=2)); s = j.get("summary", {})
        ppl = s.get("people") or []
        print(json.dumps([s.get("persons", 0), s.get("wearing", 0), j.get("fps"), (ppl[0].get("coat_score") if len(ppl) == 1 else None)]), flush=True)
    except Exception:
        print("null", flush=True)
    time.sleep(0.5)
'''

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="orangepi-lan")
ap.add_argument("--labels", nargs="+", default=["khong mac ao", "ao lab 1", "ao lab 2"])
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--out", default=None)
a = ap.parse_args()

res = {}
for lab in a.labels:
    print(f"\n=== {lab}: đứng MỘT MÌNH trước camera, thấy toàn thân hoặc nửa trên, đứng/đi nhẹ.")
    input("    Sẵn sàng thì nhấn Enter: ")
    for n in range(5, 0, -1):
        print(f"    bắt đầu sau {n} ...", end="\r", flush=True)
        time.sleep(1)
    print("    ĐANG ĐO ...                  ")
    p = subprocess.run(["ssh", a.host, "python3", "-", str(a.seconds)], input=REMOTE, capture_output=True, text=True, timeout=a.seconds + 30)
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.strip() and l.strip() != "null"]
    one = [r for r in rows if r[0] == 1]
    wear = sum(1 for r in one if r[1] >= 1)
    sc = [r[3] for r in one if len(r) > 3 and r[3] is not None]
    res[lab] = {"polls": len(rows), "one_person": len(one), "wearing": wear, "share": round(wear / len(one), 3) if one else None,
                "coat_seen": len(sc), "coat_score_mean": round(sum(sc) / len(sc), 2) if sc else None, "coat_score_min": min(sc) if sc else None,
                "coat_score_max": max(sc) if sc else None,
                "fps_mean": round(sum(r[2] for r in rows if r[2]) / max(1, sum(1 for r in rows if r[2])), 1)}
    r = res[lab]
    pct = "n/a" if r["share"] is None else "%.0f%%" % (100 * r["share"])
    print("    %d lần đọc, %d lần thấy đúng 1 người, trong đó 'đang mặc áo lab': %d (%s), FPS %s" % (r["polls"], r["one_person"], r["wearing"], pct, r["fps_mean"]))
    print("    thấy khung áo ở %d/%d lần; điểm áo trung bình %s (thấp nhất %s, cao nhất %s); luật cần >= 0,60" % (
        r["coat_seen"], r["one_person"], r["coat_score_mean"], r["coat_score_min"], r["coat_score_max"]))
print("\n=== tóm tắt")
for lab, r in res.items():
    pct = "n/a" if r["share"] is None else "%.0f%%" % (100 * r["share"])
    print("%-22s %6s  (%d/%d)   thấy khung áo %d lần, điểm TB %s" % (lab, pct, r["wearing"], r["one_person"], r["coat_seen"], r["coat_score_mean"]))
if a.out:
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
