import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.io.cues import due_cues  # noqa: E402


def test_cues_fire_once_in_time_order_even_if_a_frame_is_late():
    cues = [{"t": 20, "say": "b"}, {"t": 5, "say": "a"}, {"t": 50, "say": "c"}]
    fired: set[int] = set()
    assert due_cues(cues, 4.9, fired) == []
    assert [c["say"] for _, c in due_cues(cues, 5.0, fired)] == ["a"]
    assert due_cues(cues, 5.1, fired) == []                                   # never twice
    assert [c["say"] for _, c in due_cues(cues, 60.0, fired)] == ["b", "c"]   # a stall past two cues: both, in order
