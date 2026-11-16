"""Live system gauges for the dashboard: CPU, RAM, temperature, NPU busy %, NPU clock.

Read straight from /proc and /sys (no psutil, no root), at most once a second, so calling `update()` every frame costs
nothing. Anything the machine does not expose comes back as None and the page shows a dash.

NPU load: the kernel's own counter (/sys/kernel/debug/rknpu/load) is root-only on the board, and the service runs as an
ordinary user. When it cannot be read the figure is ESTIMATED from the time the model spends in the NPU per second
(inference ms x frames/s): that is the busy share of the one NPU core this program uses, and `npu_src` says which of the
two it is, so the page can label it honestly.
"""

from __future__ import annotations

import glob
import re
import time
from pathlib import Path

_CORE = re.compile(r"Core(\d+):\s*(\d+)\s*%")


def parse_proc_stat(text: str) -> tuple[int, int] | None:
    """(busy_ticks, total_ticks) of the aggregate `cpu` line of /proc/stat."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            v = [int(x) for x in line.split()[1:]]
            if len(v) < 4:
                return None
            idle = v[3] + (v[4] if len(v) > 4 else 0)      # idle + iowait
            total = sum(v[:8]) if len(v) >= 8 else sum(v)  # guest ticks are already counted in user/nice
            return total - idle, total
    return None


def parse_meminfo(text: str) -> tuple[float, float] | None:
    """(used_pct, used_mb) from /proc/meminfo; used = MemTotal - MemAvailable."""
    kb = {}
    for line in text.splitlines():
        k, _, rest = line.partition(":")
        if k in ("MemTotal", "MemAvailable"):
            kb[k] = int(rest.split()[0])
    if "MemTotal" not in kb or "MemAvailable" not in kb or kb["MemTotal"] <= 0:
        return None
    used = kb["MemTotal"] - kb["MemAvailable"]
    return 100.0 * used / kb["MemTotal"], used / 1024.0


def parse_npu_load(text: str) -> float | None:
    """Busiest NPU core in percent, from "NPU load:  Core0: 12%, Core1: 0%, Core2: 0%,"."""
    loads = [int(m.group(2)) for m in _CORE.finditer(text)]
    return float(max(loads)) if loads else None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except (OSError, ValueError):
        return None


class SysStats:
    def __init__(self, interval_s: float = 1.0, npu_load_path: str = "/sys/kernel/debug/rknpu/load",
                 proc_stat: str = "/proc/stat", meminfo: str = "/proc/meminfo", proc_status: str = "/proc/self/status",
                 thermal: str = "/sys/class/thermal/thermal_zone0/temp",
                 npu_freq_glob: str = "/sys/class/devfreq/*npu*/cur_freq", clock=time.monotonic) -> None:
        self.interval_s = float(interval_s)
        self._p = dict(npu=npu_load_path, stat=proc_stat, mem=meminfo, status=proc_status, thermal=thermal, freq=npu_freq_glob)
        self._clock = clock
        self._last_t = -1e9
        self._prev_cpu: tuple[int, int] | None = None
        self._infer_ema: float | None = None
        self.snapshot: dict = {}

    def update(self, fps: float, infer_ms: float) -> dict:
        """Call every frame. Cheap: the files are re-read once per `interval_s`; the latest figures are returned in between."""
        if infer_ms and infer_ms > 0:
            self._infer_ema = infer_ms if self._infer_ema is None else 0.9 * self._infer_ema + 0.1 * infer_ms
        now = self._clock()
        if now - self._last_t < self.interval_s and self.snapshot:
            return self.snapshot
        self._last_t = now
        self.snapshot = self._read_all(fps)
        return self.snapshot

    def _read_all(self, fps: float) -> dict:
        out: dict = {"cpu_pct": None, "ram_pct": None, "ram_mb": None, "rss_mb": None, "temp_c": None,
                     "npu_pct": None, "npu_src": None, "npu_mhz": None, "infer_ms": None}
        t = _read(self._p["stat"])
        cur = parse_proc_stat(t) if t else None
        if cur and self._prev_cpu and cur[1] > self._prev_cpu[1]:
            out["cpu_pct"] = round(100.0 * (cur[0] - self._prev_cpu[0]) / (cur[1] - self._prev_cpu[1]), 1)
        self._prev_cpu = cur or self._prev_cpu
        t = _read(self._p["mem"])
        mem = parse_meminfo(t) if t else None
        if mem:
            out["ram_pct"], out["ram_mb"] = round(mem[0], 1), round(mem[1])
        t = _read(self._p["status"])
        m = re.search(r"VmRSS:\s*(\d+)\s*kB", t or "")
        if m:
            out["rss_mb"] = round(int(m.group(1)) / 1024.0)
        t = _read(self._p["thermal"])
        try:
            out["temp_c"] = round(int(t.strip()) / 1000.0, 1) if t else None
        except ValueError:
            pass
        for f in glob.glob(self._p["freq"]):
            t = _read(f)
            try:
                out["npu_mhz"] = round(int(t.strip()) / 1e6) if t else None
            except ValueError:
                pass
            break
        t = _read(self._p["npu"])
        k = parse_npu_load(t) if t else None
        if k is not None:
            out["npu_pct"], out["npu_src"] = k, "kernel"
        elif self._infer_ema is not None and fps and fps > 0:
            out["npu_pct"], out["npu_src"] = round(min(100.0, self._infer_ema * fps / 10.0), 1), "estimate"
        if self._infer_ema is not None:
            out["infer_ms"] = round(self._infer_ema, 1)
        return out
