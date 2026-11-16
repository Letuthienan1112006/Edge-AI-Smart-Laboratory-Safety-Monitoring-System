"""LED + OLED status panel: states, pin names, the SSD1306 byte stream and the thread, all with fake devices."""
import time

import numpy as np
import pytest

from labsafe.io.panel import Oled, Panel, decide_state, led_pattern, parse_pin, render
from labsafe.pipeline.decision import SafetyEvent


def ev(kind="zone_ppe_violation"):
    return SafetyEvent(kind, "high", 1, detail="san_phong")


def test_pin_names_follow_the_orange_pi_pinout():
    assert parse_pin("GPIO4_B2") == ("gpio4", 10)      # header pin 11
    assert parse_pin("GPIO4_B3") == ("gpio4", 11)      # pin 13
    assert parse_pin("GPIO0_D4") == ("gpio0", 28)      # pin 15
    assert parse_pin(("gpio1", 5)) == ("gpio1", 5)
    for bad in ("GPIO9_A1", "PIN11", "GPIO4_E1", "GPIO4_B8"):
        with pytest.raises(ValueError):
            parse_pin(bad)


def test_state_matches_the_web_page_and_green_needs_fresh_frames():
    quiet = {"persons": 0, "not_wearing": 0, "violating": 0}
    assert decide_state(quiet, [], True, None) == "safe"
    assert decide_state(quiet, [], False, None) == "lost"                    # never green without frames
    assert decide_state(quiet, [ev()], False, None) == "lost"
    assert decide_state({**quiet, "not_wearing": 1}, [], True, None) == "watch"
    assert decide_state({**quiet, "violating": 1}, [], True, None) == "watch"   # still being confirmed
    assert decide_state(quiet, [ev()], True, None) == "alarm"
    assert decide_state(quiet, [], True, 4.0, hold_s=10) == "watch"          # just after an alarm
    assert decide_state(quiet, [], True, 12.0, hold_s=10) == "safe"
    assert decide_state(quiet, [ev("fall")], True, None) == "alarm"


def test_led_patterns():
    assert led_pattern("safe", 0) == {"green": True, "yellow": False, "red": False}
    assert led_pattern("watch", 0) == {"green": False, "yellow": True, "red": False}
    reds = {led_pattern("alarm", t / 10)["red"] for t in range(10)}
    assert reds == {True, False}                                            # blinks
    assert not any(led_pattern("alarm", t / 10)["green"] for t in range(10))
    assert {led_pattern("lost", t)["yellow"] for t in (0.0, 1.0)} == {True, False}
    assert not any(led_pattern("lost", t)["green"] for t in range(4))
    assert all(led_pattern("start", 0).values())


def test_render_fits_and_draws():
    for st in ("safe", "watch", "alarm", "lost"):
        img = render(st, ["2 người · 1 có áo lab", "10:20:30 · 29 FPS"])
        assert img.size == (128, 64) and np.asarray(img).any()
    # the long word still fits the 128 px width
    a = np.asarray(render("lost", ["", ""]))
    assert a[:26, :].any() and not a[:26, 127].any()
    b = np.asarray(render("safe", ["x" * 60, "y" * 60]))                   # an over-long line shrinks instead of running off the glass
    assert not b[28:, 127].any()


def test_oled_stream_is_valid_ssd1306():
    sent = []
    o = Oled(5, write=sent.append)
    assert sent[0][0] == 0x80 and sent[0][1] == 0xAE and sent[0][-1] == 0xAF         # init: display off ... display on
    assert all(len(m) <= 32 or m[0] == 0x80 for m in sent)
    sent.clear()
    img = render("safe", ["a", "b"])
    o.show(img)
    data = [m for m in sent if m[0] == 0x40]
    assert sum(len(m) - 1 for m in data) == 1024                                     # 8 pages x 128 columns
    assert all(len(m) <= 32 for m in sent)
    n = len(sent)
    o.show(img)
    assert len(sent) == n                                                            # an identical picture is not re-sent
    sent.clear()
    o.invert(True), o.invert(True), o.invert(False)
    assert [m[1] for m in sent] == [0xA7, 0xA6]


def test_oled_pixel_layout_top_left_pixel_is_bit0_of_page0():
    from PIL import Image

    sent = []
    o = Oled(5, write=sent.append)
    sent.clear()
    img = Image.new("1", (128, 64), 0)
    img.putpixel((0, 0), 1)
    img.putpixel((3, 9), 1)                                                           # page 1, row 1
    o.show(img)
    data = b"".join(m[1:] for m in sent if m[0] == 0x40)
    assert data[0] == 0b00000001 and data[128 + 3] == 0b00000010 and sum(data) == 3


def test_sh1106_uses_a_two_column_offset():
    sent = []
    o = Oled(5, driver="sh1106", write=sent.append)
    sent.clear()
    from PIL import Image

    o.show(Image.new("1", (128, 64), 1))
    assert bytes((0x80, 0x02, 0x80, 0x10)) in sent[0]                                 # column low = 2


class FakeGpio:
    def __init__(self):
        self.log, self.closed = [], False

    def set(self, name, on):
        self.log.append((name, on))

    def close(self):
        self.closed = True


class FakeOled:
    def __init__(self):
        self.pics, self.inv, self.closed = [], [], False

    def show(self, img):
        self.pics.append(img)

    def invert(self, on):
        self.inv.append(on)

    def close(self):
        self.closed = True


def wait(cond, s=2.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < s:
        time.sleep(0.02)
    return cond()


def test_panel_follows_frames_and_goes_lost_when_they_stop():
    g, o = FakeGpio(), FakeOled()
    p = Panel({"leds": {"green": "GPIO4_B2"}, "stale_s": 0.4}, gpio_factory=lambda: g, oled_factory=lambda: o)
    p._t0 -= 5                                                                       # skip the start-up lamp test
    p.update({"persons": 0, "not_wearing": 0, "violating": 0}, [], 29.0)
    assert wait(lambda: p.state == "safe") and ("green", True) in g.log
    bad = {"persons": 1, "wearing": 0, "not_wearing": 1, "violating": 1}
    p.update(bad, [ev()], 29.0)
    assert wait(lambda: p.state == "alarm") and "1 người vi phạm" in p.lines[0]

    def alarm_frames():                                                              # frames keep arriving during a real alarm
        p.update(bad, [ev()], 29.0)
        return True in o.inv
    assert wait(alarm_frames, s=3.0)                                                 # the OLED flashes in an alarm
    assert wait(lambda: p.state == "lost", s=2.0)                                    # no more frames -> never stays green/red
    p.close()
    assert g.closed and o.closed and g.log[-1][1] in (True, False)


def test_panel_never_raises_into_the_loop_when_a_device_fails():
    class Broken(FakeGpio):
        def set(self, name, on):
            raise OSError("bus error")

    o = FakeOled()
    p = Panel({"leds": {"green": "GPIO4_B2"}}, gpio_factory=Broken, oled_factory=lambda: o)
    p._t0 -= 5
    p.update({"persons": 0}, [], 29.0)
    assert wait(lambda: p.gpio is None)                                              # LEDs dropped, OLED keeps going
    assert wait(lambda: len(o.pics) >= 1)
    p.close()


def test_missing_devices_are_skipped_not_fatal():
    def nope():
        raise PermissionError("/dev/gpiochip4")

    p = Panel({"leds": {"green": "GPIO4_B2"}, "oled": {"enabled": True}}, gpio_factory=nope, oled_factory=nope)
    assert p.gpio is None and p.oled is None and p._thread is None
    p.update({"persons": 0}, [], 29.0)                                               # harmless
    p.close()
