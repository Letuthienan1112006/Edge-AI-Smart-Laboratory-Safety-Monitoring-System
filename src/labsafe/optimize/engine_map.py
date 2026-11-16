"""Task x engine mapping -- the central contribution (O8 in docs/optimization.md).

Five safety tasks, up to four compute engines on RK3588S (NPU / CPU-big /
CPU-LITTLE / GPU). This module turns "which engine should stage X run on"
from a paragraph in a paper into one config line per stage, resolved
against the CPU topology the same way `affinity.py` already does.

Scope, on purpose (plan §5/§7):
  * This resolves and records a placement. It does NOT pin a per-stage
    thread to a cluster inside the single-threaded router loop -- that
    would be `sched_setaffinity` called per-stage inside one thread,
    which is meaningless (a syscall affects the whole calling thread, not
    "the next 5 lines of code"). Real per-stage pinning needs one process
    per stage, which is what `pipeline/pool.py` (Phase 6) opens the door
    to, but does not wire into the router this round.
  * GPU is refused with `NotImplementedError`, not fabricated: no backend
    in this repo runs on the Mali GPU yet (see docs/optimization.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .affinity import cluster_masks
from .arm_probe import CPUInfo, probe_cpu

KNOWN_ENGINES = {"npu", "cpu_big", "cpu_little", "cpu_any", "gpu"}

_CLUSTER_KEY = {"cpu_big": "big", "cpu_little": "little"}


@dataclass
class EnginePlacement:
    engine: str
    cpus: list[int] | None       # taskset mask, None if not pinned
    core_mask: str | None        # "auto" | "0" | "1" | "2" | "all" -- NPU only
    note: str


def resolve(engine: str, info: CPUInfo | None = None) -> EnginePlacement:
    """Resolve one engine name to a placement. Pure with respect to config
    -- `resolve_stage_engines` is what reads `core_mask` out of a stage's
    config block, because a bare engine name carries no such per-stage
    detail."""
    if engine not in KNOWN_ENGINES:
        raise ValueError(f"unknown engine {engine!r}; known engines: {sorted(KNOWN_ENGINES)}")

    if engine == "gpu":
        raise NotImplementedError(
            "engine='gpu' (Mali-G610/OpenCL) is not implemented in this round. "
            "No backend in this repo runs on it yet, and faking a placement "
            "would report a number nobody measured -- see docs/optimization.md §7 "
            "'KHÔNG làm trong lần này'."
        )

    if engine == "npu":
        return EnginePlacement(
            engine="npu", cpus=None, core_mask="auto",
            note="NPU core count is set per-stage via `core_mask` in config",
        )

    if engine == "cpu_any":
        return EnginePlacement(
            engine="cpu_any", cpus=None, core_mask=None,
            note="no CPU cluster pinning requested for this stage",
        )

    # cpu_big / cpu_little
    info = info or probe_cpu()
    masks = cluster_masks(info)
    key = _CLUSTER_KEY[engine]
    if not info.heterogeneous or key not in masks:
        return EnginePlacement(
            engine=engine, cpus=None, core_mask=None,
            note="uniform CPU, no big/LITTLE split available",
        )
    return EnginePlacement(
        engine=engine, cpus=list(masks[key]), core_mask=None,
        # "resolved", not "pinned": this returns the CPU set for the cluster,
        # it does not call sched_setaffinity. Per-stage pinning is out of
        # scope (the router runs one loop in one thread, so a per-stage mask
        # is not expressible). Saying "pinned" here would put a false claim
        # into metrics.json, which is the file the report is written from.
        note=f"resolved to the {key} cluster: cpus {masks[key]} "
             f"(recorded only; no per-stage affinity is applied)",
    )


def resolve_stage_engines(
    pipeline_cfg: dict[str, Any], info: CPUInfo | None = None
) -> dict[str, EnginePlacement]:
    """Resolve every `stage_*` block in a pipeline config.

    A stage with no `engine` key (every config written before this round)
    resolves to `cpu_any` -- i.e. behaves exactly as it always has. This
    is what lets `configs/pipeline.yaml` gain an `engine:` key without
    changing default behaviour for anyone who has not touched it.
    """
    info = info or probe_cpu()
    placements: dict[str, EnginePlacement] = {}
    for key, block in pipeline_cfg.items():
        if not (key.startswith("stage_") and isinstance(block, dict)):
            continue
        engine = str(block.get("engine", "cpu_any"))
        placement = resolve(engine, info)
        if engine == "npu":
            # `resolve()` only knows the generic default; the per-stage
            # core_mask lives in this stage's own config block.
            core_mask = str(block.get("core_mask", "auto"))
            placement = EnginePlacement(
                engine="npu", cpus=None, core_mask=core_mask, note=placement.note
            )
        placements[key] = placement
    return placements


def describe(placements: dict[str, EnginePlacement]) -> str:
    lines = ["stage        engine      cpus            core_mask   note", "-" * 78]
    for stage, p in placements.items():
        cpus = ",".join(str(c) for c in p.cpus) if p.cpus else "-"
        lines.append(
            f"{stage:<12} {p.engine:<11} {cpus:<15} {str(p.core_mask or '-'):<11} {p.note}"
        )
    return "\n".join(lines)
