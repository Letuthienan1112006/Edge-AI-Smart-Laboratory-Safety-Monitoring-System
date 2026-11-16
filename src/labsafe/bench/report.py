"""Collect run directories into the results table the paper needs."""

from __future__ import annotations

import json
from pathlib import Path

COLUMNS = [
    ("exp_id", "Exp"),
    ("mode", "Mode"),
    ("backend", "Runtime"),
    ("imgsz", "Input"),
    ("latency_ms.p50", "p50 ms"),
    ("latency_ms.p90", "p90 ms"),
    ("latency_ms.p99", "p99 ms"),
    ("fps_sustained", "Sustained FPS"),
    ("system.rss_mb.max", "RAM MB"),
    ("system.temp_c.max", "Tmax C"),
    ("router.occupancy", "Occup."),
    ("router.stage_calls_per_frame.b", "B/frame"),
    ("router.stage_calls_per_frame.c", "C/frame"),
    ("router.stage_detections_per_call.a", "A det/call"),
    ("router.stage_blank_rate.a", "A blank"),
    ("system.result_valid", "Valid"),
    ("ended_early", "Early"),
]


def dig(d: dict, path: str):
    node = d
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def collect(runs_dir: str | Path = "runs") -> list[dict]:
    out = []
    for m in sorted(Path(runs_dir).glob("*/metrics.json")):
        try:
            out.append(json.loads(m.read_text()))
        except json.JSONDecodeError:
            print(f"skipping unreadable {m}")
    return out


def to_markdown(rows: list[dict]) -> str:
    head = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [head, sep]
    for r in rows:
        vals = []
        for path, _ in COLUMNS:
            v = dig(r, path)
            vals.append("-" if v is None else ("yes" if v is True else "NO" if v is False else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    rows = collect(sys.argv[1] if len(sys.argv) > 1 else "runs")
    if not rows:
        print("no runs found")
    else:
        print(to_markdown(rows))
        invalid = [r["exp_id"] for r in rows if dig(r, "system.result_valid") is False]
        if invalid:
            # `result_valid` now carries three distinct causes (throttling, a
            # dead model, a loop that spun past its source), so this line
            # states that a run is untrustworthy and points at the field that
            # says why, instead of naming one cause for all three.
            print(f"\nWARNING: marked invalid -- see system.warning in "
                  f"metrics.json: {', '.join(invalid)}")
        # Separate from the invalid warning on purpose: a throttled run and
        # a dead model are two different reasons to distrust a row, and a
        # reader must not have to guess which one this exp_id hit.
        # `stage_calls.a` must be truthy: a stage that never ran did not
        # "detect nothing", it did not run.
        dead = [
            r["exp_id"] for r in rows
            if dig(r, "router.stage_detections.a") == 0 and dig(r, "router.stage_calls.a")
        ]
        if dead:
            print(
                f"\nWARNING: model detected nothing -- FPS below is meaningless for: "
                f"{', '.join(dead)} (see docs/KNOWN_ISSUES.md #3)"
            )
        early = [(r["exp_id"], r.get("short_by_s")) for r in rows if r.get("ended_early")]
        if early:
            listed = ", ".join(f"{e} (short by {s}s)" for e, s in early)
            print(
                f"\nWARNING: ended before its duration: {listed}. Sustained FPS "
                f"over a truncated window is not comparable with a full-length "
                f"run (see docs/KNOWN_ISSUES.md #6)"
            )
