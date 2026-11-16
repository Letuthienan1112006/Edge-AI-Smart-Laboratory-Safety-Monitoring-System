"""Ask the CPU what it can actually do before choosing an optimisation.

Why this file exists, and why it runs before anything is quantised:

INT8 inference is faster than FP32 only when the CPU has an instruction
that multiplies and accumulates 8-bit integers in one go (ARM SDOT/UDOT,
introduced as an option in ARMv8.2-A; x86 VNNI). Without it, the runtime
must widen INT8 to INT16/INT32, multiply, accumulate, then re-quantise --
and that extra work can cost more than the bandwidth saved. Measured
cross-platform: Cortex-A76 1.83x FASTER with INT8, Cortex-A78AE 2.11x
faster, but Cortex-A53 1.65x SLOWER and an x86 without VNNI 1.76x slower
(arXiv:2609.16085).

The answer is a property of the board in front of you, not of the plan
written before it was bought. So it is computed at runtime here rather
than asserted in prose somewhere -- a document that says "this chip lacks
the instruction" is wrong the moment the chip changes, and nothing tells
you. Run this, believe this.

Heterogeneous SoCs (RK3588S, Snapdragon, most modern ARM) add a second
question prose cannot answer either: the chip has big cores AND little
cores, and the OS scheduler is free to put your inference thread on
whichever it likes. That decision can cost more than quantisation buys.
So the probe reports the cluster topology too.

Run it on the target board: `python3 -m labsafe.optimize.arm_probe`
"""

from __future__ import annotations

import glob
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field

# /proc/cpuinfo "Features" flags that matter for inference throughput.
FEATURE_MEANING = {
    "asimd": "NEON/Advanced SIMD (128-bit vector FP and int)",
    "asimddp": "INT8 dot product (SDOT/UDOT) -- the flag INT8 speedup depends on",
    "asimdhp": "FP16 arithmetic (ARMv8.2 half precision)",
    "fphp": "FP16 scalar arithmetic",
    "i8mm": "INT8 matrix multiply (ARMv8.6) -- even faster INT8 GEMM",
    "sve": "Scalable Vector Extension",
    "crc32": "CRC32 (not inference-relevant; presence check only)",
}

# x86 equivalents, so the probe is honest on the training laptop too.
X86_FEATURES = {
    "avx2": "256-bit SIMD",
    "avx512f": "512-bit SIMD",
    "avx512_vnni": "INT8 dot product (VNNI) -- the x86 equivalent of asimddp",
    "avx_vnni": "INT8 dot product (VNNI, AVX encoding)",
}

# part id -> (name, ISA, has_int8_dotprod, is_big_core)
KNOWN_CORES = {
    "0xd03": ("Cortex-A53", "ARMv8-A", False, False),
    "0xd05": ("Cortex-A55", "ARMv8.2-A", True, False),
    "0xd08": ("Cortex-A72", "ARMv8-A", False, True),
    "0xd09": ("Cortex-A73", "ARMv8-A", False, True),
    "0xd0b": ("Cortex-A76", "ARMv8.2-A", True, True),
    "0xd0d": ("Cortex-A77", "ARMv8.2-A", True, True),
    "0xd41": ("Cortex-A78", "ARMv8.2-A", True, True),
    "0xd44": ("Cortex-X1", "ARMv8.2-A", True, True),
    "0xd4b": ("Cortex-A78C", "ARMv8.2-A", True, True),
}

# Where on-chip inference accelerators announce themselves in sysfs/devicetree.
ACCELERATOR_PROBES = {
    "rknpu": ("Rockchip NPU (RKNPU2)",
              ["/sys/class/devfreq/*npu*", "/proc/device-tree/npu*",
               "/sys/kernel/debug/rknpu", "/dev/rknpu*"]),
    "mali": ("ARM Mali GPU (OpenCL compute)",
             ["/sys/class/devfreq/*gpu*", "/proc/device-tree/gpu*", "/dev/mali*"]),
    "ethosu": ("ARM Ethos-U NPU", ["/proc/device-tree/ethosu*", "/dev/ethosu*"]),
    "edgetpu": ("Google Coral Edge TPU", ["/dev/apex_*"]),
}


@dataclass
class Cluster:
    """One set of identical cores. A uniform SoC has exactly one."""

    core_name: str
    isa: str
    part: str
    cpu_ids: list[int]
    max_freq_mhz: float | None
    is_big: bool
    has_int8_dot: bool

    @property
    def label(self) -> str:
        kind = "big" if self.is_big else "LITTLE"
        return f"{self.core_name} x{len(self.cpu_ids)} ({kind})"

    @property
    def taskset_mask(self) -> str:
        """The `taskset -c` list that pins a process to this cluster."""
        return ",".join(str(c) for c in self.cpu_ids)


@dataclass
class CPUInfo:
    arch: str
    model: str = ""
    cores: int = 0
    features: list[str] = field(default_factory=list)
    part: str = ""
    core_name: str = ""
    isa: str = ""
    max_freq_mhz: float | None = None
    board: str = ""
    clusters: list[Cluster] = field(default_factory=list)
    accelerators: list[str] = field(default_factory=list)

    @property
    def has_int8_dot(self) -> bool:
        return any(f in self.features for f in ("asimddp", "i8mm", "avx512_vnni", "avx_vnni"))

    @property
    def has_fp16(self) -> bool:
        return any(f in self.features for f in ("asimdhp", "fphp"))

    @property
    def heterogeneous(self) -> bool:
        return len(self.clusters) > 1

    @property
    def big_cluster(self) -> Cluster | None:
        big = [c for c in self.clusters if c.is_big]
        return max(big, key=lambda c: c.max_freq_mhz or 0) if big else None


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _cpu_max_freq(cpu_id: int) -> float | None:
    khz = _read(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/cpuinfo_max_freq").strip()
    return int(khz) / 1000.0 if khz.isdigit() else None


def _parse_cpuinfo(txt: str) -> tuple[set[str], dict[int, str], str]:
    """Return (features, {cpu_id: part}, model). Parsed per processor block."""
    feats: set[str] = set()
    parts: dict[int, str] = {}
    model = ""
    current = -1
    for line in txt.splitlines():
        if ":" not in line:
            continue
        key, _, val = (p.strip() for p in line.partition(":"))
        key = key.lower()
        if key == "processor" and val.isdigit():
            current = int(val)
        elif key in ("features", "flags"):
            feats.update(val.split())
        elif key == "cpu part" and current >= 0:
            # Each processor block carries its own part id -- which is the
            # only place big.LITTLE is visible. Reading just the first one
            # (the obvious thing) reports an 8-core chip as uniform.
            parts[current] = val.lower()
        elif key in ("model name", "hardware") and not model:
            model = val
    return feats, parts, model


def _build_clusters(parts: dict[int, str]) -> list[Cluster]:
    by_part: dict[str, list[int]] = {}
    for cpu_id, part in sorted(parts.items()):
        by_part.setdefault(part, []).append(cpu_id)

    clusters: list[Cluster] = []
    for part, ids in by_part.items():
        name, isa, dot, big = KNOWN_CORES.get(part, (f"unknown({part})", "?", False, True))
        freqs = [f for f in (_cpu_max_freq(i) for i in ids) if f is not None]
        clusters.append(Cluster(name, isa, part, ids, max(freqs) if freqs else None, big, dot))

    # "big" is relative: on a uniform chip every core is the big one, and on
    # a heterogeneous one the label must follow measured frequency, not the
    # core-model table alone.
    if len(clusters) > 1:
        top = max((c.max_freq_mhz or 0) for c in clusters)
        for c in clusters:
            c.is_big = (c.max_freq_mhz or 0) >= top
    return sorted(clusters, key=lambda c: -(c.max_freq_mhz or 0))


def _probe_accelerators() -> list[str]:
    found = []
    for _, (label, patterns) in ACCELERATOR_PROBES.items():
        if any(glob.glob(p) for p in patterns):
            found.append(label)
    return found


def probe_cpu() -> CPUInfo:
    info = CPUInfo(arch=platform.machine())
    feats, parts, model = _parse_cpuinfo(_read("/proc/cpuinfo"))

    info.model = model
    info.cores = len(parts) if parts else (len([
        l for l in _read("/proc/cpuinfo").splitlines() if l.lower().startswith("processor")
    ]) or 0)
    info.features = sorted(feats & (set(FEATURE_MEANING) | set(X86_FEATURES)))
    info.clusters = _build_clusters(parts)
    info.accelerators = _probe_accelerators()

    if info.clusters:
        top = info.clusters[0]
        info.part, info.core_name, info.isa = top.part, top.core_name, top.isa

    board = _read("/proc/device-tree/model").strip("\x00").strip()
    info.board = board or info.model

    khz = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").strip()
    if khz.isdigit():
        info.max_freq_mhz = int(khz) / 1000.0
    elif info.clusters:
        info.max_freq_mhz = info.clusters[0].max_freq_mhz
    return info


def int8_verdict(info: CPUInfo) -> dict:
    """Predict whether INT8 will speed this CPU up, slow it down, or neither."""
    if info.has_int8_dot:
        flags = [f for f in info.features if f in ("asimddp", "i8mm", "avx512_vnni", "avx_vnni")]
        return {
            "expect": "speedup",
            "confidence": "high",
            "reason": f"the CPU reports an INT8 dot-product instruction ({', '.join(flags)}), "
                      "so the runtime can use an integer MAC path",
            "reference_numbers": "Cortex-A76 1.83x faster, Cortex-A78AE 2.11x faster (arXiv:2609.16085)",
            "action": "quantise, and report the measured speedup",
        }
    return {
        "expect": "slowdown or no gain",
        "confidence": "high" if info.arch.startswith("aarch64") else "medium",
        "reason": "no INT8 dot-product instruction (asimddp/i8mm/VNNI) on this CPU. "
                  "The runtime must widen to INT16/INT32, accumulate, then re-quantise; "
                  "that overhead typically exceeds the bandwidth saved",
        "reference_numbers": "Cortex-A53 1.65x SLOWER, x86 without VNNI 1.76x slower (arXiv:2609.16085)",
        "action": "MEASURE INT8 against FP32 before building anything on it. Expect the "
                  "model file to shrink ~4x while latency stays flat or worsens. If so, "
                  "that is a publishable negative result -- report it, and spend the "
                  "optimisation effort on structured pruning and distillation instead",
    }


def topology_verdict(info: CPUInfo) -> dict:
    """On a big.LITTLE chip, which cores the inference thread lands on matters."""
    if not info.heterogeneous:
        return {
            "heterogeneous": False,
            "note": "uniform CPU: every core is equivalent, so thread COUNT is the only "
                    "scheduling knob. Sweep it anyway (scripts/sweep_runtime.py).",
        }
    big = info.big_cluster
    little = [c for c in info.clusters if not c.is_big]
    ratio = None
    if big and little and big.max_freq_mhz and little[0].max_freq_mhz:
        ratio = round(big.max_freq_mhz / little[0].max_freq_mhz, 2)
    return {
        "heterogeneous": True,
        "clusters": [c.label for c in info.clusters],
        "big_mask": big.taskset_mask if big else None,
        "little_mask": ",".join(c.taskset_mask for c in little) if little else None,
        "freq_ratio": ratio,
        "why_it_matters":
            "the OS scheduler may migrate the inference thread onto a LITTLE core "
            "mid-run. Latency then varies by roughly the frequency ratio above, plus "
            "the microarchitecture gap, with no change to your code -- which shows up "
            "as an inexplicably wide p99 rather than as an error.",
        "action":
            f"pin inference to the big cluster (`taskset -c {big.taskset_mask if big else '?'}`) "
            "and MEASURE the three cases: big-only, LITTLE-only, and unpinned. The spread "
            "is free performance and a genuine experimental result -- nearly nobody "
            "reports it. See labsafe.optimize.affinity.",
    }


def report(info: CPUInfo | None = None) -> str:
    info = info or probe_cpu()
    v = int8_verdict(info)
    t = topology_verdict(info)
    lines = [
        "=" * 72,
        "CPU capability probe -- run this BEFORE choosing an optimisation",
        "=" * 72,
        f"  board       : {info.board or '(unknown)'}",
        f"  arch        : {info.arch}   cores: {info.cores}",
    ]
    if info.clusters:
        lines.append("  clusters    :")
        for c in info.clusters:
            freq = f"{c.max_freq_mhz:.0f} MHz" if c.max_freq_mhz else "? MHz"
            lines.append(f"      {c.label:<28} cpus {c.taskset_mask:<10} {freq:>10}  {c.isa}")
    if info.accelerators:
        lines.append("  accelerators:")
        for acc in info.accelerators:
            lines.append(f"      [x] {acc}")
    else:
        lines.append("  accelerators: none detected (CPU-only device)")

    lines += ["", "  relevant instruction-set features:"]
    meanings = {**FEATURE_MEANING, **X86_FEATURES}
    for f in info.features:
        lines.append(f"    [x] {f:<14} {meanings.get(f, '')}")
    is_arm = info.arch.startswith("aarch64") or info.arch.startswith("arm")
    watch = ("asimd", "asimddp", "asimdhp", "i8mm") if is_arm else ("avx2", "avx_vnni", "avx512_vnni")
    for f in watch:
        if f not in info.features:
            lines.append(f"    [ ] {f:<14} {meanings.get(f, '')}  <- ABSENT")

    lines += [
        "",
        "-" * 72,
        f"  INT8 prediction : {v['expect'].upper()}  (confidence: {v['confidence']})",
        f"  why             : {v['reason']}",
        f"  measured elsewhere: {v['reference_numbers']}",
        f"  what to do      : {v['action']}",
        "-" * 72,
    ]
    if t["heterogeneous"]:
        lines += [
            f"  TOPOLOGY        : HETEROGENEOUS (big.LITTLE)"
            + (f", big/LITTLE freq ratio {t['freq_ratio']}x" if t["freq_ratio"] else ""),
            f"  why it matters  : {t['why_it_matters']}",
            f"  what to do      : {t['action']}",
            "-" * 72,
        ]
    lines += [
        "",
        "  FP16: " + (
            "supported in hardware -- worth benchmarking"
            if info.has_fp16
            else "NOT supported in hardware. FP16 here saves model size and memory "
                 "bandwidth only; the arithmetic is emulated or widened, so do not "
                 "expect a compute speedup."
        ),
        "",
        "  Reminder: pruning only speeds up dense CPU inference when it is",
        "  STRUCTURED (whole channels/filters removed). Zeroing individual",
        "  weights leaves the GEMM the same size and runs at the same speed.",
        "  See docs/optimization.md.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    info = probe_cpu()
    if "--json" in sys.argv:
        print(json.dumps({"cpu": asdict(info), "int8": int8_verdict(info),
                          "topology": topology_verdict(info)}, indent=2))
    else:
        print(report(info))
