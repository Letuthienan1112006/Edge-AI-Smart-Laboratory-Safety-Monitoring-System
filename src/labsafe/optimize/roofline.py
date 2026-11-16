"""Is this device compute-bound or memory-bound? The answer picks the technique.

Optimisations are not interchangeable. They attack different bottlenecks:

  * Fewer FLOPs (structured pruning, smaller input, cheaper architecture)
    helps a COMPUTE-bound model.
  * Fewer bytes moved (quantisation, smaller weights, better data layout)
    helps a MEMORY-bound model.

Apply the wrong one and the latency barely moves while the accuracy drops,
which is the most common way an optimisation chapter ends up with a table
full of changes that did nothing.

This module measures two device constants -- sustained memory bandwidth
and achievable FP32 throughput -- and computes the ridge point between
them: the arithmetic intensity (FLOPs per byte) above which a workload is
compute-bound. Then it places your model on that line.

Honest limits: numpy's BLAS is not NCNN's ARM kernels, so the absolute
GFLOPS is an upper bound on what a detector will reach. The RATIO, which
is what selects the technique, is far more stable than either number, and
the measurement is fully reproducible. Report it as device
characterisation, not as model performance.

    python3 -m labsafe.optimize.roofline
    python3 -m labsafe.optimize.roofline --gflops 1.2 --weights-mb 6.2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Roofline:
    bandwidth_gbs: float
    compute_gflops: float
    threads_hint: int

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte above which the device is compute-bound."""
        return self.compute_gflops / self.bandwidth_gbs if self.bandwidth_gbs else float("inf")


def measure_bandwidth(mb: int = 256, repeats: int = 5) -> float:
    """Sustained streaming bandwidth in GB/s, via a large out-of-cache copy.

    Sized well past the Pi 4's 1 MB shared L2 so the number reflects DRAM,
    not cache. A copy touches each byte twice (read + write), which is what
    the GB/s figure counts.
    """
    n = (mb * 1024 * 1024) // 8
    a = np.ones(n, dtype=np.float64)
    b = np.empty_like(a)
    np.copyto(b, a)                       # warm the pages: first touch is a page fault, not bandwidth
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.copyto(b, a)
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, (a.nbytes * 2) / dt / 1e9)
    return best


def measure_gflops(n: int = 512, repeats: int = 5) -> float:
    """Achievable FP32 GEMM throughput in GFLOP/s (2*n^3 FLOPs per matmul)."""
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b                                 # warmup: BLAS thread pool spin-up is not throughput
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        a @ b
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, (2.0 * n**3) / dt / 1e9)
    return best


def measure(mb: int = 256, n: int = 512) -> Roofline:
    import os

    return Roofline(
        bandwidth_gbs=measure_bandwidth(mb),
        compute_gflops=measure_gflops(n),
        threads_hint=os.cpu_count() or 1,
    )


def classify(rl: Roofline, model_gflops: float, weights_mb: float, activations_mb: float = 0.0) -> dict:
    """Place a model on the roofline and name the technique that will pay."""
    bytes_moved = (weights_mb + activations_mb) * 1e6
    flops = model_gflops * 1e9
    intensity = flops / bytes_moved if bytes_moved else float("inf")

    t_compute_ms = (flops / (rl.compute_gflops * 1e9)) * 1000 if rl.compute_gflops else float("inf")
    t_memory_ms = (bytes_moved / (rl.bandwidth_gbs * 1e9)) * 1000 if rl.bandwidth_gbs else float("inf")
    bound = "compute" if intensity > rl.ridge_point else "memory"

    advice = {
        "compute": [
            "structured channel pruning (removes FLOPs AND bytes)",
            "smaller input resolution (FLOPs scale ~quadratically with it)",
            "a cheaper backbone, or distillation into one",
            "INT8 ONLY if the CPU has a dot-product instruction -- run arm_probe first",
        ],
        "memory": [
            "shrink the weights: quantisation helps here even without a dot-product "
            "instruction, because the win is bytes moved, not integer maths",
            "structured pruning (also removes bytes)",
            "fewer/narrower feature maps; avoid wide early layers at high resolution",
            "reducing FLOPs alone will disappoint -- the device is waiting on memory",
        ],
    }[bound]

    return {
        "arithmetic_intensity_flops_per_byte": round(intensity, 2),
        "device_ridge_point": round(rl.ridge_point, 2),
        "bound_by": bound,
        "lower_bound_ms_compute": round(t_compute_ms, 2),
        "lower_bound_ms_memory": round(t_memory_ms, 2),
        "attainable_lower_bound_ms": round(max(t_compute_ms, t_memory_ms), 2),
        "what_will_actually_help": advice,
        "caveat": "numpy BLAS != NCNN ARM kernels; treat the absolute ms as an "
                  "optimistic floor and the bound_by classification as the finding",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mb", type=int, default=256, help="bandwidth test buffer size")
    ap.add_argument("--n", type=int, default=512, help="GEMM size for the compute test")
    ap.add_argument("--gflops", type=float, default=None, help="your model's GFLOPs per inference")
    ap.add_argument("--weights-mb", type=float, default=None, help="your model's weight size in MB")
    ap.add_argument("--activations-mb", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    print("measuring device (a few seconds)...", flush=True)
    rl = measure(a.mb, a.n)
    out: dict = {"device": asdict(rl), "ridge_point_flops_per_byte": round(rl.ridge_point, 2)}

    if a.gflops is not None and a.weights_mb is not None:
        out["model"] = classify(rl, a.gflops, a.weights_mb, a.activations_mb)

    if a.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"\n  sustained bandwidth : {rl.bandwidth_gbs:6.2f} GB/s")
    print(f"  FP32 GEMM throughput: {rl.compute_gflops:6.2f} GFLOP/s")
    print(f"  ridge point         : {rl.ridge_point:6.2f} FLOPs/byte")
    print("    -> a workload below this is memory-bound, above it is compute-bound")
    if "model" in out:
        m = out["model"]
        print(f"\n  your model          : {m['arithmetic_intensity_flops_per_byte']} FLOPs/byte"
              f"  ==> {m['bound_by'].upper()}-BOUND")
        print(f"  optimistic floor    : {m['attainable_lower_bound_ms']} ms"
              f"  (compute {m['lower_bound_ms_compute']} / memory {m['lower_bound_ms_memory']})")
        print("\n  what will actually help:")
        for tip in m["what_will_actually_help"]:
            print(f"    - {tip}")
        print(f"\n  caveat: {m['caveat']}")
    else:
        print("\n  pass --gflops and --weights-mb to place your model on this roofline.")
        print("  ultralytics prints GFLOPs in the model summary; weight size is the file size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
