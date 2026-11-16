"""Buzzer / LED output, with a working no-GPIO fallback.

The fallback is not a convenience: developing against a stub GPIO on the
laptop is what lets the alert logic be tested before the Pi exists, and
`gpio: off` is also the correct setting for a benchmark run, where GPIO
chatter would add jitter to the numbers.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("labsafe.alerts")


class _RPiGpio:
    """Adapter for the RPi.GPIO-style API (also OPi.GPIO)."""

    def __init__(self, gpio):
        self._g = gpio

    def set(self, pin: int, high: bool) -> None:
        self._g.output(pin, self._g.HIGH if high else self._g.LOW)

    def cleanup(self) -> None:
        self._g.cleanup()


class _Gpiod2:
    """Adapter for libgpiod 2.x."""

    def __init__(self, request):
        self._r = request

    def set(self, pin: int, high: bool) -> None:
        from gpiod.line import Value

        self._r.set_value(pin, Value.ACTIVE if high else Value.INACTIVE)

    def cleanup(self) -> None:
        self._r.release()


class _Gpiod1:
    """Adapter for libgpiod 1.x."""

    def __init__(self, chip, lines):
        self._chip, self._lines = chip, lines

    def set(self, pin: int, high: bool) -> None:
        self._lines[pin].set_value(1 if high else 0)

    def cleanup(self) -> None:
        for line in self._lines.values():
            line.release()
        self._chip.close()

SEVERITY_PATTERN = {          # (beeps, on_s, off_s)
    "low": (1, 0.08, 0.0),
    "medium": (2, 0.10, 0.10),
    "high": (3, 0.15, 0.10),
    "critical": (5, 0.25, 0.10),
}


class AlertSink:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.buzzer_pin = cfg.get("buzzer_pin", 17)
        self.led_r_pin = cfg.get("led_r_pin", 27)
        self.led_g_pin = cfg.get("led_g_pin", 22)
        # libgpiod addresses (chip, line offset). On a Pi the chip is
        # gpiochip0 and offsets happen to equal BCM numbers; on an Orange Pi
        # they do not. Run `gpioinfo` and read the offsets off the board.
        self.gpio_chip = str(cfg.get("gpio_chip", "gpiochip0"))
        mode = str(cfg.get("gpio", "auto")).lower()
        self.gpio = None
        if mode != "off":
            self.gpio = self._try_gpio(required=(mode == "on"))
        self.fired = 0

    def _try_gpio(self, required: bool):
        """Try libgpiod first, then the board-specific libraries.

        RPi.GPIO does not exist on non-Raspberry boards, and its BCM pin
        numbers mean nothing there. libgpiod is the kernel's own interface:
        it addresses lines as (chip, offset) and works on any board exposing
        /dev/gpiochip*, so the same code runs on a Pi and an Orange Pi.

        Find the offsets on the board with `gpioinfo`. They are NOT BCM
        numbers -- copying a Pi pinout onto another board silently drives
        the wrong line, which looks exactly like a dead buzzer.
        """
        for attempt in (self._try_gpiod, self._try_rpi_gpio, self._try_opi_gpio):
            backend = attempt()
            if backend is not None:
                return backend
        if required:
            raise RuntimeError(
                "gpio: on, but no GPIO library available. Install one:\n"
                "  sudo apt install python3-libgpiod   (any board, preferred)\n"
                "  pip install RPi.GPIO               (Raspberry Pi only)\n"
                "  pip install OPi.GPIO               (Orange Pi)"
            )
        log.info("no GPIO library available; alerts will be logged only")
        return None

    def _try_gpiod(self):
        try:
            import gpiod
        except ImportError:
            return None
        pins = [self.buzzer_pin, self.led_r_pin, self.led_g_pin]
        try:
            if hasattr(gpiod, "request_lines"):          # libgpiod 2.x
                from gpiod.line import Direction, Value

                req = gpiod.request_lines(
                    f"/dev/{self.gpio_chip}",
                    consumer="labsafe",
                    config={p: gpiod.LineSettings(direction=Direction.OUTPUT,
                                                  output_value=Value.INACTIVE)
                            for p in pins},
                )
                return _Gpiod2(req)
            chip = gpiod.Chip(self.gpio_chip)            # libgpiod 1.x
            lines = {p: chip.get_line(p) for p in pins}
            for p, line in lines.items():
                line.request(consumer="labsafe",
                             type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
            return _Gpiod1(chip, lines)
        except Exception as e:
            log.warning("libgpiod present but unusable (%s: %s); trying other backends",
                        type(e).__name__, e)
            return None

    def _try_rpi_gpio(self):
        try:
            import RPi.GPIO as GPIO  # type: ignore
        except ImportError:
            return None
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (self.buzzer_pin, self.led_r_pin, self.led_g_pin):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        return _RPiGpio(GPIO)

    def _try_opi_gpio(self):
        try:
            import OPi.GPIO as GPIO  # type: ignore
        except ImportError:
            return None
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (self.buzzer_pin, self.led_r_pin, self.led_g_pin):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        return _RPiGpio(GPIO)

    def alert(self, severity: str, message: str = "") -> None:
        """Never blocks: the beep pattern runs off-thread.

        Sleeping through a 1.7s critical pattern inside the frame loop
        would land in the p99 latency column and make the alarm look like
        an inference cost. Keep hardware timing out of the measurement.
        """
        self.fired += 1
        log.warning("ALERT [%s] %s", severity.upper(), message)
        if self.gpio is None:
            return
        threading.Thread(target=self._beep, args=(severity,), daemon=True).start()

    def _beep(self, severity: str) -> None:
        beeps, on_s, off_s = SEVERITY_PATTERN.get(severity, SEVERITY_PATTERN["medium"])
        self.gpio.set(self.led_g_pin, False)
        self.gpio.set(self.led_r_pin, True)
        for i in range(beeps):
            self.gpio.set(self.buzzer_pin, True)
            time.sleep(on_s)
            self.gpio.set(self.buzzer_pin, False)
            if i < beeps - 1:
                time.sleep(off_s)

    def clear(self) -> None:
        if self.gpio is None:
            return
        self.gpio.set(self.led_r_pin, False)
        self.gpio.set(self.led_g_pin, True)

    def close(self) -> None:
        if self.gpio is not None:
            self.gpio.cleanup()
            self.gpio = None
