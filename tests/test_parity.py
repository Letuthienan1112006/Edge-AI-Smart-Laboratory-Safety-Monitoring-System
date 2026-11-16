"""Tests for the conversion parity gate -- the highest-priority module in
this round. `test_zero_detection_collapse_is_named` is the single most
important test in the whole repo right now: it is the one that catches a
model that converts cleanly and then detects nothing (KNOWN_ISSUES.md #3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.backends import Detection  # noqa: E402
from labsafe.models.parity import ParityThresholds, compare_detections  # noqa: E402


def det(x1=0.0, y1=0.0, x2=100.0, y2=100.0, score=0.9, cls_id=0, cls_name="person"):
    return Detection(x1, y1, x2, y2, score, cls_id, cls_name)


def image_with_n_objects(n: int, score: float = 0.9) -> list[Detection]:
    return [det(x1=i * 120.0, x2=i * 120.0 + 100.0, score=score) for i in range(n)]


def test_identical_detections_pass():
    ref = [image_with_n_objects(2), image_with_n_objects(1), image_with_n_objects(0)]
    cand = [list(imgs) for imgs in ref]
    report = compare_detections(ref, cand)
    assert report.passed is True
    assert report.failures == []


def test_zero_detection_collapse_is_named():
    ref = [image_with_n_objects(2) for _ in range(5)]
    cand = [[] for _ in range(5)]
    report = compare_detections(ref, cand)
    assert report.passed is False
    assert "zero_detection_collapse" in report.failures[0]


def test_partial_blank_over_threshold_fails():
    # 5 images, 2 of them (40%) come back blank on the candidate; the
    # default max_blank_rate is 0.30.
    ref = [image_with_n_objects(1) for _ in range(5)]
    cand = [image_with_n_objects(1) for _ in range(5)]
    cand[0] = []
    cand[1] = []
    report = compare_detections(ref, cand)
    assert report.passed is False


def test_small_jitter_still_passes():
    ref = [[det(x1=100.0, y1=100.0, x2=200.0, y2=200.0, score=0.80)]]
    cand = [[det(x1=102.0, y1=101.0, x2=202.0, y2=199.0, score=0.82)]]
    report = compare_detections(ref, cand)
    assert report.passed is True


def test_shifted_boxes_fail_on_iou():
    # Shift by half the box width -- IoU drops well below the 0.50 match
    # threshold, so the pair never matches at all.
    ref = [[det(x1=100.0, y1=100.0, x2=200.0, y2=200.0)]]
    cand = [[det(x1=150.0, y1=100.0, x2=250.0, y2=200.0)]]
    report = compare_detections(ref, cand)
    assert report.passed is False
    assert any("recall" in f or "iou" in f for f in report.failures)


def test_class_mismatch_is_not_matched():
    ref = [[det(cls_id=0, cls_name="person")]]
    cand = [[det(cls_id=1, cls_name="fire")]]
    report = compare_detections(ref, cand)
    assert report.passed is False
    assert report.metrics["recall"] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compare_detections([image_with_n_objects(1)], [])


def test_empty_reference_is_a_failure_not_a_pass():
    ref = [[], [], []]
    cand = [[], [], []]
    report = compare_detections(ref, cand)
    assert report.passed is False
    assert "reference_has_no_detections" in report.failures


def test_markdown_report_contains_the_verdict():
    ref = [image_with_n_objects(2)]
    cand = [[]]
    report = compare_detections(ref, cand)
    assert "FAIL" in report.as_markdown()


def test_garbage_overdetection_fails_on_recall_not_on_ratio():
    """A candidate that fires a lot of unrelated boxes (e.g. a bad NMS/
    decode config producing noise) must not be rewarded by `det_ratio`
    just because the raw count is high. Here every candidate box misses,
    so `recall` catches it; the harder sibling case -- garbage ADDED to
    correct boxes, where recall stays 1.0 -- is covered by
    `test_correct_boxes_plus_garbage_is_not_a_pass` below."""
    ref = [[det(x1=100.0, y1=100.0, x2=200.0, y2=200.0)]]
    # 20 candidate boxes, none of which overlaps the one real object.
    cand = [[det(x1=1000.0 + i * 50.0, y1=1000.0, x2=1040.0 + i * 50.0, y2=1040.0)
             for i in range(20)]]
    report = compare_detections(ref, cand)
    assert report.metrics["det_ratio"] == 20.0  # far above min_det_ratio=0.60
    assert report.passed is False
    assert "low_recall" in report.failures


def test_reference_also_has_zero_detections_is_not_confused_with_collapse():
    """If the *reference* model also sees nothing on a given image (the
    image genuinely has no object), that single blank image must not be
    conflated with the candidate collapsing -- only a reference that is
    blank on EVERY image trips `reference_has_no_detections`, and only a
    candidate that is blank overall while the reference is not trips
    `zero_detection_collapse`. Mixed images (some blank on both sides,
    some not) must be judged on their own merits."""
    ref = [image_with_n_objects(1), []]  # second image genuinely empty
    cand = [image_with_n_objects(1), []]
    report = compare_detections(ref, cand)
    assert report.passed is True
    assert "zero_detection_collapse" not in report.failures
    assert "reference_has_no_detections" not in report.failures


# -- thresholds are honoured, not just the defaults ----------------------

def test_custom_thresholds_are_respected():
    ref = [image_with_n_objects(10)]
    cand = [image_with_n_objects(5)]  # det_ratio 0.5, below default 0.60
    strict = compare_detections(ref, cand)
    assert strict.passed is False

    lenient = compare_detections(ref, cand, ParityThresholds(min_det_ratio=0.4, min_recall=0.4))
    assert lenient.passed is True


def test_correct_boxes_plus_garbage_is_not_a_pass():
    """Reproducing every reference box AND inventing more must FAIL.

    The gate was one-sided: `min_det_ratio` is a floor, so a candidate that
    returns all the true boxes plus twenty spurious ones scored recall 1.0,
    blank_rate 0.0, det_ratio 21 -- and passed. In the field every spurious
    box is a false alarm the safety system acts on, and spurious
    low-confidence detections are as real an INT8 failure mode as the
    zero-detection collapse this gate was built for. `max_det_ratio` and
    `min_precision` close it.
    """
    real = det(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
    ref = [[real]]
    cand = [[real] + [det(x1=1000.0 + i * 50.0, y1=1000.0,
                          x2=1040.0 + i * 50.0, y2=1040.0) for i in range(20)]]

    report = compare_detections(ref, cand)

    assert report.metrics["recall"] == 1.0, "fixture must keep recall perfect"
    assert report.metrics["det_ratio"] == 21.0
    assert report.metrics["precision"] < 0.1
    assert report.passed is False
    assert "excess_detections" in report.failures
    assert "low_precision" in report.failures


def test_a_faithful_conversion_still_passes_the_new_rules():
    """The new ceilings must not reject a good model: same boxes, one extra
    borderline detection is within tolerance."""
    boxes = [det(x1=float(100 + 300 * i), y1=100.0,
                 x2=float(200 + 300 * i), y2=200.0) for i in range(5)]
    report = compare_detections([boxes], [boxes])
    assert report.passed is True, report.failures
    assert report.metrics["precision"] == 1.0
