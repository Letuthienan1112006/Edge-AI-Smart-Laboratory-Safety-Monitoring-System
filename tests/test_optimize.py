"""Tests for the optimisation advisor.

The advisor's whole value is that it says "do not quantise this chip"
BEFORE a week is spent quantising. A bug that flips the verdict is
therefore worse than useless, and silent -- the output is a plausible
paragraph either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.optimize.arm_probe import CPUInfo, int8_verdict, report  # noqa: E402
from labsafe.optimize.roofline import Roofline, classify  # noqa: E402


def pi4() -> CPUInfo:
    """Raspberry Pi 4: Cortex-A72, ARMv8.0-A, NEON but no dot product."""
    return CPUInfo(arch="aarch64", cores=4, features=["asimd", "crc32"],
                   part="0xd08", core_name="Cortex-A72", isa="ARMv8-A",
                   board="Raspberry Pi 4 Model B Rev 1.4")


def pi5() -> CPUInfo:
    """Raspberry Pi 5: Cortex-A76, ARMv8.2-A, has asimddp."""
    return CPUInfo(arch="aarch64", cores=4, features=["asimd", "asimddp", "asimdhp"],
                   part="0xd0b", core_name="Cortex-A76", isa="ARMv8.2-A")


def test_pi4_is_warned_against_int8():
    v = int8_verdict(pi4())
    assert v["expect"] == "slowdown or no gain"
    assert "dot-product" in v["reason"]
    assert "MEASURE" in v["action"]


def test_pi5_is_told_to_quantise():
    v = int8_verdict(pi5())
    assert v["expect"] == "speedup"
    assert "quantise" in v["action"]


def test_pi4_has_no_fp16_hardware():
    assert not pi4().has_fp16
    assert pi5().has_fp16


def test_x86_without_vnni_is_warned():
    cpu = CPUInfo(arch="x86_64", cores=8, features=["avx2"])
    assert int8_verdict(cpu)["expect"] == "slowdown or no gain"


def test_x86_with_vnni_is_cleared():
    cpu = CPUInfo(arch="x86_64", cores=12, features=["avx2", "avx_vnni"])
    assert int8_verdict(cpu)["expect"] == "speedup"


def test_report_names_the_absent_flag_on_pi4():
    """The absent flag must be visible, not merely missing from a list --
    a reader cannot notice something that was never printed."""
    text = report(pi4())
    assert "asimddp" in text and "ABSENT" in text
    assert "STRUCTURED" in text          # the pruning caveat travels with it


# -- roofline ------------------------------------------------------------

def test_memory_bound_model_is_told_to_shrink_bytes():
    rl = Roofline(bandwidth_gbs=4.0, compute_gflops=12.0, threads_hint=4)
    out = classify(rl, model_gflops=0.05, weights_mb=25.0)
    assert out["bound_by"] == "memory"
    assert any("quantisation helps here" in t for t in out["what_will_actually_help"])


def test_compute_bound_model_is_told_to_cut_flops():
    rl = Roofline(bandwidth_gbs=4.0, compute_gflops=12.0, threads_hint=4)
    out = classify(rl, model_gflops=8.0, weights_mb=6.0)
    assert out["bound_by"] == "compute"
    assert any("structured channel pruning" in t for t in out["what_will_actually_help"])


def test_compute_bound_advice_gates_int8_on_the_probe():
    """Never recommend INT8 without sending the reader to arm_probe first."""
    rl = Roofline(bandwidth_gbs=4.0, compute_gflops=12.0, threads_hint=4)
    out = classify(rl, model_gflops=8.0, weights_mb=6.0)
    int8_tips = [t for t in out["what_will_actually_help"] if "INT8" in t]
    assert int8_tips and all("arm_probe" in t for t in int8_tips)


def test_attainable_floor_is_the_worse_of_the_two_bounds():
    rl = Roofline(bandwidth_gbs=4.0, compute_gflops=12.0, threads_hint=4)
    out = classify(rl, model_gflops=8.0, weights_mb=6.0)
    assert out["attainable_lower_bound_ms"] == max(
        out["lower_bound_ms_compute"], out["lower_bound_ms_memory"]
    )


# -- heterogeneous SoCs (RK3588S / Orange Pi 5) --------------------------

RK3588_FEATURES = (
    "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp "
    "cpuid asimdrdm lrcpc dcpop asimddp"
)


def _rk3588s_cpuinfo() -> str:
    """4x Cortex-A55 on cpu0-3, 4x Cortex-A76 on cpu4-7."""
    blocks = []
    for i in range(8):
        part = "0xd05" if i < 4 else "0xd0b"
        blocks.append(
            f"processor\t: {i}\nFeatures\t: {RK3588_FEATURES}\n"
            f"CPU implementer\t: 0x41\nCPU part\t: {part}\n"
        )
    return "\n".join(blocks)


def _rk3588s_info(monkeypatch=None):
    from labsafe.optimize import arm_probe as ap

    feats, parts, _ = ap._parse_cpuinfo(_rk3588s_cpuinfo())
    real = ap._cpu_max_freq
    ap._cpu_max_freq = lambda i: 1800.0 if i < 4 else 2400.0
    try:
        clusters = ap._build_clusters(parts)
    finally:
        ap._cpu_max_freq = real
    known = set(ap.FEATURE_MEANING) | set(ap.X86_FEATURES)
    info = CPUInfo(arch="aarch64", cores=8, features=sorted(feats & known),
                   board="Orange Pi 5", clusters=clusters)
    info.part, info.core_name, info.isa = (
        clusters[0].part, clusters[0].core_name, clusters[0].isa
    )
    return info


def test_biglittle_is_detected_not_collapsed_to_one_core_type():
    """Reading only the first `CPU part` reports an 8-core big.LITTLE chip
    as uniform -- the failure this parser exists to avoid."""
    info = _rk3588s_info()
    assert info.heterogeneous
    assert len(info.clusters) == 2
    assert {c.core_name for c in info.clusters} == {"Cortex-A76", "Cortex-A55"}


def test_big_cluster_is_the_faster_one_and_lists_the_right_cpus():
    info = _rk3588s_info()
    big = info.big_cluster
    assert big is not None and big.core_name == "Cortex-A76"
    assert big.cpu_ids == [4, 5, 6, 7]
    assert big.taskset_mask == "4,5,6,7"


def test_rk3588s_is_cleared_for_int8():
    """A76 and A55 both carry asimddp, so the Pi 4 warning must NOT fire."""
    v = int8_verdict(_rk3588s_info())
    assert v["expect"] == "speedup"
    assert "asimddp" in v["reason"]


def test_topology_verdict_gives_an_actionable_taskset_command():
    from labsafe.optimize.arm_probe import topology_verdict

    t = topology_verdict(_rk3588s_info())
    assert t["heterogeneous"] is True
    assert t["big_mask"] == "4,5,6,7"
    assert t["little_mask"] == "0,1,2,3"
    assert t["freq_ratio"] == round(2400 / 1800, 2)


def test_uniform_cpu_reports_no_topology_work():
    from labsafe.optimize.arm_probe import topology_verdict

    t = topology_verdict(pi4())
    assert t["heterogeneous"] is False


def test_report_surfaces_the_npu_when_present():
    info = _rk3588s_info()
    info.accelerators = ["Rockchip NPU (RKNPU2)"]
    text = report(info)
    assert "RKNPU2" in text and "taskset -c 4,5,6,7" in text


# -- affinity ------------------------------------------------------------

def test_pinned_restores_the_previous_mask():
    """A benchmark that leaves the process pinned silently changes every
    measurement that follows it."""
    import os

    from labsafe.optimize.affinity import current_affinity, pinned, supported

    if not supported():
        return
    before = current_affinity()
    with pinned([sorted(before)[0]]):
        assert os.sched_getaffinity(0) == {sorted(before)[0]}
    assert current_affinity() == before


def test_pinned_restores_the_mask_even_on_exception():
    from labsafe.optimize.affinity import current_affinity, pinned, supported

    if not supported():
        return
    before = current_affinity()
    try:
        with pinned([sorted(before)[0]]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_affinity() == before


def test_pinned_with_none_is_a_noop():
    from labsafe.optimize.affinity import current_affinity, pinned

    before = current_affinity()
    with pinned(None) as ok:
        assert ok is False
    assert current_affinity() == before


def test_cluster_masks_names_big_and_little():
    from labsafe.optimize.affinity import cluster_masks

    masks = cluster_masks(_rk3588s_info())
    assert masks["big"] == [4, 5, 6, 7]
    assert masks["little"] == [0, 1, 2, 3]
    assert "all" in masks
