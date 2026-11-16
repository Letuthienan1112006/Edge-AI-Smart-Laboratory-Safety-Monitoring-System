"""Benchmark runner: one experiment -> one directory of artefacts.

Design rules it enforces, from docs/measurement-protocol.md:
  * warmup is excluded from statistics, never merely "ignored later"
  * latency is reported as percentiles; a lone mean is not accepted output
  * sustained FPS = the final `sustained_window_s`, not the whole run
  * throttle state is recorded before and after and printed as a verdict
  * every run writes the full config it ran with, so a number in the paper
    can always be traced back to the exact configuration that produced it
"""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .sysmon import SystemMonitor


@dataclass
class BenchConfig:
    duration_s: float = 900.0
    warmup_s: float = 30.0
    sample_interval_s: float = 1.0
    sustained_window_s: float = 300.0
    out_dir: str = "runs"
    exp_id: str = "E0"
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "BenchConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 2),
        "std": round(float(a.std()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "max": round(float(a.max()), 2),
    }


class BenchRunner:
    """Drives a callable `step() -> None` and times it.

    Keeping the measured unit a plain callable means the same harness times
    the full pipeline, a single backend, or a no-op control run that
    measures capture-and-overhead alone.
    """

    def __init__(self, cfg: BenchConfig, extra_meta: dict | None = None,
                 out: str | Path | None = None):
        self.cfg = cfg
        self.extra_meta = extra_meta or {}
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out = Path(out) if out is not None else Path(cfg.out_dir) / f"{cfg.exp_id}-{stamp}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.iter_ms: list[float] = []
        self.iter_t: list[float] = []
        self.stop_reason = "not_started"

    def run(
        self,
        step: Callable[[], bool | None],
        on_tick: Callable[[int, float], None] | None = None,
    ) -> dict:
        """Drive `step` for `duration_s`, or until it says the work is gone.

        `step` returning False means "there is nothing left to process" --
        a video file that ran out of frames, a closed capture device. The
        loop MUST honour that. A wall-clock-only loop keeps calling a step
        that returns immediately, counts those no-ops as iterations, and
        reports a sustained FPS in the hundreds of thousands. That number
        is not an outlier a reader can spot in context; it is the headline
        figure of the run, and it is fabricated. Returning None (the old
        contract) still means "keep going", so existing callers are
        unaffected.
        """
        mon = SystemMonitor(interval_s=self.cfg.sample_interval_s).start()
        t_start = time.time()
        warm_until = t_start + self.cfg.warmup_s
        end = t_start + self.cfg.duration_s
        n = 0
        self.stop_reason = "duration_reached"
        try:
            while time.time() < end:
                t0 = time.perf_counter()
                alive = step()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                now = time.time()
                if alive is False:
                    self.stop_reason = "source_exhausted"
                    break
                if now >= warm_until:        # warmup excluded at source
                    self.iter_ms.append(dt_ms)
                    self.iter_t.append(now)
                n += 1
                if on_tick:
                    on_tick(n, now - t_start)
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            print("\ninterrupted -- writing partial results")
        finally:
            mon.stop()

        result = self._summarise(mon, t_start, time.time(), n)
        self._write(result, mon)
        return result

    def _summarise(self, mon: SystemMonitor, t_start: float, t_end: float, iters: int) -> dict:
        measured_s = max(1e-9, t_end - (t_start + self.cfg.warmup_s))
        sustained = self._sustained_fps(t_end)
        short = self.cfg.duration_s - (t_end - t_start)
        return {
            "exp_id": self.cfg.exp_id,
            "note": self.cfg.note,
            "stop_reason": self.stop_reason,
            "ended_early": self.stop_reason != "duration_reached",
            "short_by_s": round(short, 1) if short > 1.0 else 0.0,
            "started": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
            "wall_s": round(t_end - t_start, 1),
            "warmup_s": self.cfg.warmup_s,
            "iterations_total": iters,
            "iterations_measured": len(self.iter_ms),
            "latency_ms": percentiles(self.iter_ms),
            "fps_mean_measured": round(len(self.iter_ms) / measured_s, 2),
            "fps_sustained": sustained,
            "sustained_window_s": self.cfg.sustained_window_s,
            "system": mon.summary(),
            "host": {
                "machine": platform.machine(),
                "processor": platform.processor() or platform.machine(),
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            **self.extra_meta,
        }

    def _sustained_fps(self, t_end: float) -> float | None:
        """Mean FPS over the final window -- what the device actually holds."""
        if not self.iter_t:
            return None
        cut = t_end - self.cfg.sustained_window_s
        tail = [t for t in self.iter_t if t >= cut]
        if len(tail) < 2:
            return None
        span = tail[-1] - tail[0]
        return round(len(tail) / span, 2) if span > 0 else None

    def _write(self, result: dict, mon: SystemMonitor) -> None:
        (self.out / "metrics.json").write_text(json.dumps(result, indent=2))
        (self.out / "config.json").write_text(
            json.dumps({"bench": asdict(self.cfg), **self.extra_meta}, indent=2, default=str)
        )
        with (self.out / "latency.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "iter_ms"])
            w.writerows(zip([round(t, 4) for t in self.iter_t], [round(v, 3) for v in self.iter_ms]))
        rows = mon.rows()
        if rows:
            with (self.out / "system.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"\nresults -> {self.out}")
