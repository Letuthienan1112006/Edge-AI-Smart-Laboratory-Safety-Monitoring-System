"""Status panel on the board: three LEDs (green / yellow / red) and a small I2C OLED, showing the same four states as the web page.

    AN TOÀN       green   camera sends frames, nobody violates
    CHÚ Ý         yellow  someone without a coat outside the zone, a violation being confirmed, or an alarm in the last `hold_s` seconds
    BÁO ĐỘNG      red     blinking; the OLED flashes (inverted) with the number of people in violation
    MẤT TÍN HIỆU  yellow  slow blink; no frame for `stale_s` seconds. Green is NEVER shown without fresh frames.
    (on start all three LEDs light for 1.5 s: wiring check)

Pins are given as in the Orange Pi pinout, e.g. `GPIO4_B2` (bank 4, group B, index 2 -> chip label "gpio4", line 10); the chip is found by its
label, because /dev/gpiochipN numbering is not stable. Needs libgpiod (python3-gpiod 1.x or 2.x) and read/write access to /dev/gpiochip* and
/dev/i2c-N (deploy/99-labsafe-io.rules + scripts/setup_panel_board.sh). The OLED is driven directly through /dev/i2c-N (SSD1306 / SH1106,
128x64) with Pillow for the Vietnamese text: no extra Python package.

Never touches the frame loop's timing: update() stores a few numbers; a thread does all the GPIO / I2C work. Any failure (no permission,
no wiring, bus error) is logged once per device and that device is skipped; the monitoring keeps running.
"""

from __future__ import annotations

import fcntl
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("labsafe.panel")

I2C_SLAVE = 0x0703
ALARM_KINDS = {"zone_ppe_violation", "zone_intrusion", "ppe_violation", "fall", "fire", "smoke"}   # = the web page's KIND list
WORD = {"safe": "AN TOÀN", "watch": "CHÚ Ý", "alarm": "BÁO ĐỘNG", "lost": "MẤT TÍN HIỆU", "start": "ĐANG KHỞI ĐỘNG"}
FONT = Path(__file__).resolve().parents[1] / "assets" / "DejaVuSans-Bold.ttf"


# --------------------------------------------------------------------------- state
def decide_state(summary: dict | None, events: list, fresh: bool, since_alarm_s: float | None, hold_s: float = 10.0) -> str:
    """The web page's decision (dashboard.py `status()`), minus its acknowledge button, plus a hold time after an alarm ends."""
    if not fresh:
        return "lost"
    s = summary or {}
    if any(getattr(e, "kind", None) in ALARM_KINDS for e in events):
        return "alarm"
    if since_alarm_s is not None and since_alarm_s < hold_s:
        return "watch"
    if (s.get("violating") or 0) > 0 or (s.get("not_wearing") or 0) > 0:
        return "watch"
    return "safe"


def led_pattern(state: str, t: float) -> dict[str, bool]:
    """Which LEDs are on at time t (seconds)."""
    off = {"green": False, "yellow": False, "red": False}
    if state == "safe":
        return {**off, "green": True}
    if state == "watch":
        return {**off, "yellow": True}
    if state == "alarm":
        return {**off, "red": int(t * 4) % 2 == 0}             # 2 Hz blink
    if state == "lost":
        return {**off, "yellow": int(t) % 2 == 0}              # 0.5 Hz blink
    return {"green": True, "yellow": True, "red": True}        # start: lamp test


# --------------------------------------------------------------------------- GPIO
def parse_pin(spec) -> tuple[str, int]:
    """'GPIO4_B2' -> ('gpio4', 10); also accepts (label, line)."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return str(spec[0]), int(spec[1])
    m = re.fullmatch(r"GPIO([0-4])_([A-D])([0-7])", str(spec).strip().upper())
    if not m:
        raise ValueError(f"pin {spec!r}: expected a name like GPIO4_B2")
    return f"gpio{m.group(1)}", "ABCD".index(m.group(2)) * 8 + int(m.group(3))


class Gpio:
    """Output lines found by chip label. libgpiod 1.x and 2.x."""

    def __init__(self, pins: dict[str, object], active_high: bool = True):
        import gpiod

        self.active_high = active_high
        self._lines: dict[str, object] = {}
        self._reqs: list = []
        wanted = {name: parse_pin(p) for name, p in pins.items()}
        chips = {}
        for i in range(0, 12):
            path = f"/dev/gpiochip{i}"
            if not os.path.exists(path):
                continue
            try:
                c = gpiod.Chip(path)
                label = c.label() if hasattr(c, "label") and callable(c.label) else c.get_info().label
            except Exception as e:  # noqa: BLE001
                if isinstance(e, PermissionError) or "ermission" in str(e):
                    raise PermissionError(f"{path}: no permission (run scripts/setup_panel_board.sh once, then re-login/reboot)") from None
                continue
            chips[label] = (c, path)
        self._v2 = hasattr(gpiod, "request_lines")
        for name, (label, line) in wanted.items():
            if label not in chips:
                raise RuntimeError(f"no GPIO chip labelled {label!r} (found {sorted(chips)})")
            chip, path = chips[label]
            if self._v2:
                from gpiod.line import Direction, Value

                req = gpiod.request_lines(path, consumer="labsafe-panel",
                                          config={line: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)})
                self._reqs.append(req)
                self._lines[name] = (req, line)
            else:
                ln = chip.get_line(line)
                ln.request(consumer="labsafe-panel", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0 if active_high else 1])
                self._lines[name] = ln
        self._gpiod = gpiod

    def set(self, name: str, on: bool) -> None:
        level = on if self.active_high else not on
        ln = self._lines[name]
        if self._v2:
            from gpiod.line import Value

            ln[0].set_value(ln[1], Value.ACTIVE if level else Value.INACTIVE)
        else:
            ln.set_value(1 if level else 0)

    def close(self) -> None:
        for n in list(self._lines):
            try:
                self.set(n, False)
            except Exception:  # noqa: BLE001
                pass
        for r in self._reqs:
            try:
                r.release()
            except Exception:  # noqa: BLE001
                pass
        if not self._v2:
            for ln in self._lines.values():
                try:
                    ln.release()
                except Exception:  # noqa: BLE001
                    pass


# --------------------------------------------------------------------------- OLED
def find_i2c_bus(controller: str = "fead0000", bus: int | None = None) -> int:
    """/dev/i2c-N number of the controller (RK3588 I2C5 = fead0000: the pins 3 and 5 of the 26-pin header), from sysfs."""
    if bus is not None:
        return int(bus)
    for p in sorted(Path("/sys/class/i2c-dev").glob("i2c-*")):
        if controller in os.path.realpath(p / "device"):
            return int(p.name.split("-")[1])
    raise RuntimeError(f"I2C controller {controller} is not enabled (add overlays=i2c5-m3 to /boot/orangepiEnv.txt and reboot)")


class Oled:
    """128x64 SSD1306 / SH1106 over /dev/i2c-N in page-addressing mode. `write` is injectable for tests."""

    W, H = 128, 64

    def __init__(self, bus: int, address: int = 0x3C, driver: str = "ssd1306", rotate: int = 0,
                 write: Callable[[bytes], None] | None = None):
        self.col = 2 if driver.lower() == "sh1106" else 0
        self.rotate = int(rotate) % 360
        self._fd = None
        if write is None:
            self._fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
            fcntl.ioctl(self._fd, I2C_SLAVE, address)
            write = lambda b: os.write(self._fd, b)  # noqa: E731
        self._w = write
        self._last: bytes | None = None
        self._inverted = False
        flip = (0xA0, 0xC0) if self.rotate == 180 else (0xA1, 0xC8)
        self._cmd(0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x02, flip[0], flip[1],
                  0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF)

    def _cmd(self, *cmds: int) -> None:
        self._w(b"".join(bytes((0x80, c)) for c in cmds))      # 0x80 = control byte: one command follows (Co=1, D/C#=0)

    def show(self, img) -> None:
        """img: PIL image, mode '1' or 'L', 128x64. Skipped when identical to what is on the glass."""
        import numpy as np

        a = np.asarray(img.convert("1").resize((self.W, self.H))) > 0                        # True = lit pixel
        pages = a.reshape(8, 8, self.W)                                                      # page, row in page, column
        data = ((pages * (1 << np.arange(8))[None, :, None]).sum(axis=1)).astype(np.uint8)    # bit 0 = top row of the page
        raw = data.tobytes()
        if raw == self._last:
            return
        self._last = raw
        for p in range(8):
            self._cmd(0xB0 + p, self.col & 0x0F, 0x10 | (self.col >> 4))
            row = raw[p * self.W:(p + 1) * self.W]
            for i in range(0, self.W, 31):                      # one I2C message <= 32 bytes: 1 control byte + 31 data bytes
                self._w(b"\x40" + row[i:i + 31])

    def invert(self, on: bool) -> None:
        if on != self._inverted:
            self._inverted = on
            self._cmd(0xA7 if on else 0xA6)

    def close(self) -> None:
        try:
            self._cmd(0xAE)                                      # display off
        except Exception:  # noqa: BLE001
            pass
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def render(state: str, lines: list[str], size=(128, 64)):
    """The OLED picture: the state word large, then up to two small lines."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("1", size, 0)
    d = ImageDraw.Draw(img)
    word = WORD.get(state, state)
    big = 22
    while big > 11:                                              # shrink until the word fits the width ("MẤT TÍN HIỆU" is the long one)
        f = ImageFont.truetype(str(FONT), big)
        if d.textlength(word, font=f) <= size[0] - 2:
            break
        big -= 1
    d.text((1, 0), word, font=f, fill=1)
    y = 30
    for ln in lines[:2]:
        sz = 13
        while sz > 8:                                            # each small line shrinks until it fits the width
            f2 = ImageFont.truetype(str(FONT), sz)
            if d.textlength(ln, font=f2) <= size[0] - 2:
                break
            sz -= 1
        while len(ln) > 1 and d.textlength(ln, font=f2) > size[0] - 2:      # still too wide at the smallest size: cut it
            ln = ln[:-2] + "…"
        d.text((1, y), ln, font=f2, fill=1)
        y += 17
    return img


# --------------------------------------------------------------------------- the panel
class Panel:
    def __init__(self, cfg: dict | None = None, gpio_factory=None, oled_factory=None, clock=time.time):
        cfg = cfg or {}
        self.clock = clock
        self.hold_s = float(cfg.get("hold_s", 10.0))
        self.stale_s = float(cfg.get("stale_s", 3.0))
        self.state = "start"
        self.lines: list[str] = ["", ""]
        self._summary: dict = {}
        self._events: list = []
        self._fps = 0.0
        self._last_frame_t: float | None = None
        self._last_alarm_t: float | None = None
        self._t0 = self.clock()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.gpio = self.oled = None
        pins = cfg.get("leds")
        if pins:
            try:
                self.gpio = (gpio_factory or (lambda: Gpio(pins, bool(cfg.get("active_high", True)))))()
            except Exception as e:  # noqa: BLE001 -- no wiring / no permission must not stop the monitoring
                log.warning("panel: LEDs unavailable (%s: %s)", type(e).__name__, e)
        o = cfg.get("oled") or {}
        if o.get("enabled", True) and (oled_factory or o):
            try:
                self.oled = (oled_factory or (lambda: Oled(find_i2c_bus(str(o.get("i2c_controller", "fead0000")), o.get("i2c_bus")),
                                                           int(str(o.get("address", 0x3C)), 0), str(o.get("driver", "ssd1306")),
                                                           int(o.get("rotate", 0)))))()
            except Exception as e:  # noqa: BLE001
                log.warning("panel: OLED unavailable (%s: %s)", type(e).__name__, e)
        self._thread = None
        if self.gpio is not None or self.oled is not None:
            log.info("panel: LEDs %s, OLED %s", "on" if self.gpio else "off", "on" if self.oled else "off")
            self._thread = threading.Thread(target=self._run, name="panel", daemon=True)
            self._thread.start()

    # called from the frame loop: stores numbers, nothing else
    def update(self, summary: dict | None, events: list, fps: float = 0.0) -> None:
        now = self.clock()
        with self._lock:
            self._summary, self._events, self._fps, self._last_frame_t = summary or {}, list(events), fps, now
            if any(getattr(e, "kind", None) in ALARM_KINDS for e in events):
                self._last_alarm_t = now

    def _snapshot(self) -> tuple[str, list[str]]:
        now = self.clock()
        with self._lock:
            fresh = self._last_frame_t is not None and now - self._last_frame_t < self.stale_s
            since = None if self._last_alarm_t is None else now - self._last_alarm_t
            s, ev, fps = self._summary, self._events, self._fps
        if self._last_frame_t is None and now - self._t0 < 1.5:
            return "start", ["", ""]
        st = decide_state(s, ev, fresh, since, self.hold_s)
        if st == "lost":
            lines = ["Mất hình camera", "Kiểm tra cáp camera"]
        elif st == "alarm":
            lines = [f"{s.get('violating', 0)} người vi phạm", "KHÔNG MẶC ÁO LAB"]
        else:
            lines = [f"{s.get('persons', 0)} người · {s.get('wearing', 0)} áo lab", f"{time.strftime('%H:%M:%S', time.localtime(now))}  {int(fps)} FPS"]
        return st, lines

    def _run(self) -> None:
        last_pic = None
        while not self._stop.wait(0.05):
            st, lines = self._snapshot()
            self.state, self.lines = st, lines
            now = self.clock()
            if self.gpio is not None:
                try:
                    for name, on in led_pattern(st, now).items():
                        self.gpio.set(name, on)
                except Exception as e:  # noqa: BLE001
                    log.warning("panel: LED write failed (%s: %s); LEDs off from now on", type(e).__name__, e)
                    self.gpio = None
            if self.oled is not None:
                try:
                    key = (st, tuple(lines))
                    if key != last_pic:
                        last_pic = key
                        self.oled.show(render(st, lines))
                    self.oled.invert(st == "alarm" and int(now * 2) % 2 == 0)
                except Exception as e:  # noqa: BLE001
                    log.warning("panel: OLED write failed (%s: %s); OLED off from now on", type(e).__name__, e)
                    self.oled = None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        for dev in (self.gpio, self.oled):
            if dev is not None:
                try:
                    dev.close()
                except Exception:  # noqa: BLE001
                    pass


def build_panel(cfg: dict | None) -> Panel | None:
    return Panel(cfg) if cfg and cfg.get("enabled") else None
