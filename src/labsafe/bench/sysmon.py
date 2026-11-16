"""System sampler: RAM, CPU, temperature, CPU frequency, throttle flags.

The throttle flag is the one that silently ruins Raspberry Pi results.
`vcgencmd get_throttled` returns a bitmask; bit 0 = under-voltage now,
bit 1 = frequency capped now, bit 2 = throttled now, bits 16-18 = the
same conditions having occurred at any point since boot. A benchmark that
ends with a non-zero mask measured a thermally limited machine, and
reporting its FPS as the platform's capability is wrong. So the value is
sampled, stored, and surfaced in the report rather than left for someone
to remember to check.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil

THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_since_boot",
    17: "arm_freq_capped_since_boot",
    18: "throttled_since_boot",
    19: "soft_temp_limit_since_boot",
}


def read_temp_c() -> float | None:
    """CPU temperature, via sysfs first (cheap) then vcgencmd."""
    for path in (
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ):
        try:
            with open(path) as fh:
                return int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
        ).stdout
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def read_throttled() -> dict | None:
    """Decoded `vcgencmd get_throttled`, or None off a Pi."""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        raw = int(out.split("=")[1], 16)
    except Exception:
        return None
    return {
        "raw": hex(raw),
        "clean": raw == 0,
        **{name: bool(raw & (1 << bit)) for bit, name in THROTTLE_BITS.items()},
    }


def read_cpu_freq_mhz() -> float | None:
    try:
        f = psutil.cpu_freq()
        return float(f.current) if f else None
    except Exception:
        return None


@dataclass
class Sample:
    t: float
    rss_mb: float
    cpu_percent: float
    temp_c: float | None
    freq_mhz: float | None


@dataclass
class SystemMonitor:
    """Background sampler. Start before the run, stop after, keep the samples."""

    interval_s: float = 1.0
    samples: list[Sample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    throttled_before: dict | None = None
    throttled_after: dict | None = None

    def start(self) -> "SystemMonitor":
        self.throttled_before = read_throttled()
        proc = psutil.Process(os.getpid())
        proc.cpu_percent(None)  # prime: the first call always returns 0.0

        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                try:
                    self.samples.append(
                        Sample(
                            t=time.time(),
                            rss_mb=proc.memory_info().rss / 1e6,
                            cpu_percent=proc.cpu_percent(None),
                            temp_c=read_temp_c(),
                            freq_mhz=read_cpu_freq_mhz(),
                        )
                    )
                except Exception:  # a sampler must never kill the run
                    continue

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "SystemMonitor":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 2)
        self.throttled_after = read_throttled()
        return self

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0, "throttled_after": self.throttled_after}
        rss = [s.rss_mb for s in self.samples]
        cpu = [s.cpu_percent for s in self.samples]
        temps = [s.temp_c for s in self.samples if s.temp_c is not None]
        freqs = [s.freq_mhz for s in self.samples if s.freq_mhz is not None]
        out = {
            "samples": len(self.samples),
            "rss_mb": {"mean": round(sum(rss) / len(rss), 1), "max": round(max(rss), 1)},
            "cpu_percent": {"mean": round(sum(cpu) / len(cpu), 1), "max": round(max(cpu), 1)},
            "throttled_before": self.throttled_before,
            "throttled_after": self.throttled_after,
        }
        if temps:
            out["temp_c"] = {
                "start": round(temps[0], 1),
                "end": round(temps[-1], 1),
                "max": round(max(temps), 1),
            }
        if freqs:
            out["cpu_freq_mhz"] = {
                "start": round(freqs[0], 0),
                "end": round(freqs[-1], 0),
                "min": round(min(freqs), 0),
            }
        # The headline verdict, so nobody has to decode the bitmask by hand.
        after = self.throttled_after
        if after is not None:
            out["result_valid"] = bool(
                not after["throttled_now"]
                and not after["arm_freq_capped_now"]
                and not after["under_voltage_now"]
            )
            if not out["result_valid"]:
                out["warning"] = (
                    "board was throttled or under-voltage during this run -- "
                    "discard these numbers or report them as thermally limited"
                )
        return out

    def rows(self) -> list[dict]:
        return [
            {
                "t": round(s.t, 3),
                "rss_mb": round(s.rss_mb, 2),
                "cpu_percent": round(s.cpu_percent, 1),
                "temp_c": s.temp_c,
                "freq_mhz": s.freq_mhz,
            }
            for s in self.samples
        ]
