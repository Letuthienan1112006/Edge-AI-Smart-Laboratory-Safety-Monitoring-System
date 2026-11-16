"""Timed cues for scripted recordings: tell the actors what to do at second T, and remember at which frame it was announced.

Ground truth for event-level evaluation (KNOWN_ISSUES #43 follow-up) is "who entered / left / put the coat on at second T"; announcing
the cue from the recorder's own loop ties that second to a frame index of the video that is being written.
"""
from __future__ import annotations


def due_cues(cues: list[dict], elapsed_s: float, fired: set[int]) -> list[tuple[int, dict]]:
    """Cues whose time has come and that were not announced yet, oldest first. `fired` is updated in place."""
    out = []
    for i, c in sorted(enumerate(cues), key=lambda ic: ic[1]["t"]):
        if i not in fired and elapsed_s >= float(c["t"]):
            fired.add(i)
            out.append((i, c))
    return out
