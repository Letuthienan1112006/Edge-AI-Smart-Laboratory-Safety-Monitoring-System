#!/usr/bin/env python3
"""ON THE BOARD: test the status LEDs + OLED by themselves, in the order a person can check by eye. Stop the live service first (it owns the pins):
    make panel-test        (from the laptop: stops the service, runs this, starts the service again; asks the sudo password)
Steps: 1) each LED alone with its colour name (is the wiring right?)  2) the four states of the panel as the live system will show them.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import yaml  # noqa: E402

from labsafe.io.panel import Gpio, Panel, find_i2c_bus, parse_pin  # noqa: E402
from labsafe.pipeline.decision import SafetyEvent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/live/dashboard_lab_board.yaml")
cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())["panel"]

print("== pins")
for name, pin in cfg["leds"].items():
    print(f"  {name:6s} {pin} -> chip label/line {parse_pin(pin)}")
try:
    print(f"  OLED on /dev/i2c-{find_i2c_bus(str(cfg['oled'].get('i2c_controller', 'fead0000')), cfg['oled'].get('i2c_bus'))}, address {cfg['oled'].get('address')}")
except Exception as e:  # noqa: BLE001
    print(f"  OLED bus NOT available: {e}")

print("\n== 1/2 each LED alone (1.5 s): is the colour right?")
try:
    g = Gpio(cfg["leds"], bool(cfg.get("active_high", True)))
except Exception as e:  # noqa: BLE001
    sys.exit(f"cannot open the LED pins: {type(e).__name__}: {e}\n(stop labsafe-live first; permissions: scripts/board/setup_panel_board.sh)")
for name in cfg["leds"]:
    print(f"  -> only the {name.upper()} LED should be lit")
    g.set(name, True)
    time.sleep(1.5)
    g.set(name, False)
g.close()

print("\n== 2/2 the panel states (LEDs + OLED)")
p = Panel(cfg)
if p.gpio is None and p.oled is None:
    sys.exit("neither LEDs nor OLED could start (see the warnings above)")
p._t0 -= 5
quiet = {"persons": 0, "wearing": 0, "not_wearing": 0, "violating": 0}
steps = [
    ("AN TOÀN: green steady; OLED 'AN TOÀN'", dict(summary={**quiet, "persons": 2, "wearing": 2}, events=[]), 4),
    ("CHÚ Ý: yellow steady (someone without a coat outside the zone)", dict(summary={**quiet, "persons": 1, "not_wearing": 1}, events=[]), 4),
    ("BÁO ĐỘNG: RED BLINKS, OLED flashes", dict(summary={**quiet, "persons": 1, "not_wearing": 1, "violating": 1},
                                                events=[SafetyEvent("zone_ppe_violation", "high", 1, detail="san_phong")]), 5),
]
for text, kw, secs in steps:
    print(f"  {text}")
    t0 = time.time()
    while time.time() - t0 < secs:
        p.update(kw["summary"], kw["events"], 29.0)
        time.sleep(0.1)
print("  MẤT TÍN HIỆU: yellow blinks slowly (no frames any more); OLED 'MẤT TÍN HIỆU'")
time.sleep(6)
p.close()
print("\ndone: all LEDs off, OLED off.")
