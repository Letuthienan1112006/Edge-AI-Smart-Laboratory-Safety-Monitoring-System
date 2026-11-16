"""CPU affinity on a big.LITTLE SoC -- the optimisation with no accuracy cost.

On a heterogeneous chip (RK3588S: 4x Cortex-A76 @2.4GHz + 4x Cortex-A55
@1.8GHz) the Linux scheduler decides which cores your inference thread
runs on, and it optimises for whole-system energy and fairness, not for
your p99 latency. It will migrate a hot thread onto a LITTLE core, and
the only symptom is latency variance that looks like noise.

That makes affinity three things at once:

  1. A speedup. Pinning to the big cluster costs nothing in accuracy,
     nothing in model size, and no training time.
  2. A measurement-validity requirement. An unpinned benchmark measures
     the scheduler's mood as much as the model. Two runs of the "same"
     configuration can differ by the frequency ratio and the
     microarchitecture gap between core types.
  3. An experimental result. big-only vs LITTLE-only vs unpinned, on the
     same model and the same frames, is a clean three-arm comparison that
     almost no edge-AI paper reports -- because most of them ran on
     uniform hardware where the question does not exist.

There is a real trade-off to report rather than hide: pinning to big
cores raises power and heat, which on a passively cooled board brings
thermal throttling forward. Measure energy per inference, not just
latency, and watch the throttle flags.

    python3 -m labsafe.optimize.affinity                # show the topology
    python3 -m labsafe.optimize.affinity --benchmark    # measure the spread
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Callable, Iterable

from .arm_probe import CPUInfo, probe_cpu


def supported() -> bool:
    """Affinity control exists on Linux; elsewhere the pin is a no-op."""
    return hasattr(os, "sched_setaffinity")


def current_affinity() -> set[int]:
    return os.sched_getaffinity(0) if supported() else set(range(os.cpu_count() or 1))


@contextmanager
def pinned(cpu_ids: Iterable[int] | None):
    """Pin this process to `cpu_ids` for the duration of the block.

    Restores the previous mask on exit, including on exception -- a
    benchmark that leaves the process pinned silently changes every
    measurement that follows it in the same run.
    """
    if cpu_ids is None or not supported():
        yield False
        return
    ids = set(int(c) for c in cpu_ids)
    if not ids:
        yield False
        return
    previous = current_affinity()
    try:
        os.sched_setaffinity(0, ids)
        yield True
    finally:
        try:
            os.sched_setaffinity(0, previous)
        except OSError:
            pass


def cluster_masks(info: CPUInfo | None = None) -> dict[str, list[int]]:
    """Named CPU sets worth benchmarking against each other."""
    info = info or probe_cpu()
    masks: dict[str, list[int]] = {"all": sorted(current_affinity())}
    if not info.clusters:
        return masks
    for c in info.clusters:
        key = "big" if c.is_big else "little"
        # Several LITTLE clusters would collide on one key; suffix them.
        if key in masks:
            key = f"{key}_{c.core_name.lower().replace('cortex-', '')}"
        masks[key] = list(c.cpu_ids)
    return masks


def measure(
    step: Callable[[], None],
    cpu_ids: list[int] | None,
    iterations: int = 60,
    warmup: int = 10,
) -> dict:
    """Time `step` under one affinity mask. Warmup runs inside the pin.

    The warmup must be inside: the first iterations after a migration pay
    for cold caches on the new core, and letting them into the sample
    measures the migration, not the steady state.
    """
    import numpy as np

    with pinned(cpu_ids) as ok:
        for _ in range(warmup):
            step()
        lat = []
        t0 = time.perf_counter()
        for _ in range(iterations):
            t = time.perf_counter()
            step()
            lat.append((time.perf_counter() - t) * 1000.0)
        wall = time.perf_counter() - t0

    a = np.asarray(lat)
    return {
        "cpus": cpu_ids,
        "pinned": ok,
        "iterations": iterations,
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p90_ms": round(float(np.percentile(a, 90)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "std_ms": round(float(a.std()), 2),
        "fps": round(iterations / wall, 2) if wall > 0 else None,
    }


def compare(step: Callable[[], None], iterations: int = 60, warmup: int = 10) -> dict:
    """Run `step` under every cluster mask and quantify the spread."""
    info = probe_cpu()
    masks = cluster_masks(info)
    results = {name: measure(step, ids, iterations, warmup) for name, ids in masks.items()}

    named = [(n, r) for n, r in results.items() if r["p50_ms"] > 0]
    best = min(named, key=lambda kv: kv[1]["p50_ms"])
    worst = max(named, key=lambda kv: kv[1]["p50_ms"])

    out = {
        "heterogeneous": info.heterogeneous,
        "clusters": [c.label for c in info.clusters],
        "results": results,
        "fastest": best[0],
        "slowest": worst[0],
        "spread": round(worst[1]["p50_ms"] / best[1]["p50_ms"], 2) if best[1]["p50_ms"] else None,
    }
    if info.heterogeneous and "all" in results and "big" in results:
        unpinned, big = results["all"], results["big"]
        out["pinning_gain"] = round(unpinned["p50_ms"] / big["p50_ms"], 2) if big["p50_ms"] else None
        # Variance, not the median, is what an unpinned run really costs:
        # the scheduler's migrations land in the tail.
        out["p99_reduction"] = round(unpinned["p99_ms"] / big["p99_ms"], 2) if big["p99_ms"] else None
    return out


def format_comparison(cmp: dict) -> str:
    lines = ["", f"{'mask':<12} {'cpus':<12} {'p50':>8} {'p90':>8} {'p99':>8} {'std':>8} {'fps':>8}",
             "-" * 68]
    for name, r in cmp["results"].items():
        cpus = ",".join(str(c) for c in (r["cpus"] or []))[:11] or "-"
        lines.append(f"{name:<12} {cpus:<12} {r['p50_ms']:>8.2f} {r['p90_ms']:>8.2f} "
                     f"{r['p99_ms']:>8.2f} {r['std_ms']:>8.2f} {r['fps'] or 0:>8.2f}")
    lines.append("-" * 68)
    if not cmp["heterogeneous"]:
        lines.append("uniform CPU: no big.LITTLE effect to measure here.")
        return "\n".join(lines)

    lines.append(f"fastest mask: {cmp['fastest']}   slowest: {cmp['slowest']}   "
                 f"spread: {cmp['spread']}x")
    if cmp.get("pinning_gain"):
        lines.append(f"pinning to the big cluster: {cmp['pinning_gain']}x on p50, "
                     f"{cmp.get('p99_reduction', '?')}x on p99")
        lines.append("")
        lines.append("  Report BOTH numbers. The p50 gain is the speedup; the p99 gain is")
        lines.append("  the measurement-validity argument -- an unpinned benchmark lets the")
        lines.append("  scheduler into your tail latency, and that is not the model's doing.")
        lines.append("  Then check power and temperature: pinning to big cores costs watts,")
        lines.append("  and on a passively cooled board it brings throttling forward.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--benchmark", action="store_true",
                    help="run a synthetic CPU load under each mask")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    info = probe_cpu()
    masks = cluster_masks(info)
    print(f"board       : {info.board or '(unknown)'}")
    print(f"topology    : {'heterogeneous' if info.heterogeneous else 'uniform'}")
    for c in info.clusters:
        freq = f"{c.max_freq_mhz:.0f} MHz" if c.max_freq_mhz else "? MHz"
        print(f"  {c.label:<28} cpus {c.taskset_mask:<10} {freq}")
    print(f"masks       : {json.dumps(masks)}")
    if not supported():
        print("\nsched_setaffinity unavailable on this platform: pinning is a no-op.")
        return 0

    if not a.benchmark:
        big = info.big_cluster
        if big:
            print(f"\npin a command to the big cluster:  taskset -c {big.taskset_mask} <cmd>")
            print(f"pin this pipeline:                 taskset -c {big.taskset_mask} "
                  f"python3 -m labsafe.app --bench ...")
        print("\nadd --benchmark to measure the spread with a synthetic load.")
        return 0

    import numpy as np

    # A matmul is a stand-in for the real detector: same shape of work
    # (dense FP GEMM), no model or runtime required, so the topology effect
    # can be measured on day one.
    m = np.random.rand(256, 256).astype(np.float32)

    def step() -> None:
        (m @ m).sum()

    cmp = compare(step, iterations=a.iterations)
    print(json.dumps(cmp, indent=2) if a.json else format_comparison(cmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
