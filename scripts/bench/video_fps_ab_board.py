#!/usr/bin/env python3
"""ON BOARD, as root:  sudo python3 scripts/bench/video_fps_ab_board.py [SECONDS=120] [ROUNDS=3]
Controlled A/B of `dashboard.video_fps` (0 = draw + JPEG-encode the page video on every frame, 15 = at most 15x/s) on the SAME replay video, live
mode (dashboard on, one viewer attached like a browser), DDR pinned for both. Measures sustained FPS (from the app's own /status) and the
process CPU. Stops labsafe-live for the duration and restores the DDR governor and the service at the end.
"""
import json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 120
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DMC = Path("/sys/class/devfreq/dmc/governor")
assert os.geteuid() == 0, "run with sudo"
os.chdir(Path(__file__).resolve().parents[2])
orig = DMC.read_text().strip()
was_active = subprocess.run(["systemctl", "is-active", "labsafe-live.service"], capture_output=True, text=True).stdout.strip() == "active"
base = Path("configs/live/dashboard_lab_board.yaml").read_text()
out = Path(f"runs/video_fps_ab-{time.strftime('%Y%m%d-%H%M%S')}"); out.mkdir(parents=True)
os.chown(out, 1000, 1000)


def cfg(vfps: int) -> str:
    t = re.sub(r"device: 0,", "device: data/lab_replay.mp4,", base)
    t = re.sub(r"dashboard: \{enabled: true,", f"dashboard: {{enabled: true, video_fps: {vfps},", t)
    assert f"video_fps: {vfps}" in t and "lab_replay.mp4" in t
    p = Path(f"/tmp/vfps_{vfps}.yaml"); p.write_text(t); return str(p)


def ticks(pid):
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])   # utime + stime


def status():
    try:
        return json.load(urllib.request.urlopen("http://127.0.0.1:8000/status", timeout=2))
    except Exception:
        return None


def run(vfps: int, tag: str):
    # run as the board user directly (no sudo layer): the app stops cleanly on SIGINT, and a signal sent to a sudo wrapper may not reach it
    app = subprocess.Popen(["python3", "-m", "labsafe.app", "--config", cfg(vfps),
                            "--classes", "configs/classes_lab_v3.yaml", "--no-display", "--gpio", "off", "--replay", "--run-dir", str(out / tag)],
                           stdout=open(out / f"{tag}.log", "w"), stderr=subprocess.STDOUT, user=1000, group=1000,
                           env={**os.environ, "PYTHONPATH": "src", "HOME": "/home/orangepi"})
    time.sleep(20)
    viewer = subprocess.Popen(["curl", "-s", "http://127.0.0.1:8000/stream", "-o", "/dev/null"])   # a browser with the page open
    pid = app.pid
    t0, c0 = time.time(), ticks(pid)
    fps = []
    while time.time() - t0 < DUR:
        time.sleep(5)
        s = status()
        if s: fps.append(s["fps"])
    cpu = (ticks(pid) - c0) / os.sysconf("SC_CLK_TCK") / (time.time() - t0) * 100
    viewer.terminate()
    app.send_signal(signal.SIGINT)
    try:
        app.wait(timeout=20)
    except subprocess.TimeoutExpired:
        app.kill(); app.wait()
    return sum(fps) / len(fps), min(fps), cpu


results = {0: [], 15: []}
try:
    subprocess.run(["systemctl", "stop", "labsafe-live.service"])
    DMC.write_text("performance\n")
    for r in range(1, ROUNDS + 1):
        for vfps in (0, 15):
            time.sleep(15)
            res = run(vfps, f"v{vfps}_r{r}")
            results[vfps].append(res)
            print(f"round {r} video_fps={vfps:2d}: FPS mean {res[0]:.2f} (min {res[1]:.1f})  CPU {res[2]:.0f}%", flush=True)
finally:
    DMC.write_text(orig + "\n")
    if was_active:
        subprocess.run(["systemctl", "start", "labsafe-live.service"])
    print(f"restored dmc={DMC.read_text().strip()}")
for k, v in results.items():
    if v:
        print(f"video_fps={k:2d}: FPS {sum(x[0] for x in v)/len(v):.2f}  CPU {sum(x[2] for x in v)/len(v):.0f}%  per-round FPS {[round(x[0],2) for x in v]}")
(out / "result.json").write_text(json.dumps(results))
print(out)
