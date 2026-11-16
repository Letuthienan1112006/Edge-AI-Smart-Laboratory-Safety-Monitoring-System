"""Shape guards on `yolo_decode()` and the model-card hardening on
`RKNNBackend`. Both exist because their failure mode without a guard is
silent: wrong labels, or a segfault on a $80 board instead of a ValueError
on a laptop. `rknnlite` is mocked via `sys.modules` so the RKNNBackend
tests run on any x86 laptop, per the plan's "no code needs a board to be
tested" rule.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.models.modelcard import ModelCard  # noqa: E402
from labsafe.models.postprocess import yolo_decode  # noqa: E402


def _name_of(i: int) -> str:
    return f"cls_{i}"


def _raw(n_boxes: int, n_channels: int) -> np.ndarray:
    """A synthetic (1, 4+nc, N) head output, batch-first like Ultralytics.

    `n_boxes` must exceed `n_channels` (as it always does for a real
    anchor grid) so the "attribute axis is the short one" orientation
    logic in `yolo_decode` picks the same axis a real model would.
    """
    assert n_boxes > n_channels
    arr = np.zeros((n_channels, n_boxes), dtype=np.float32)
    arr[:4] = 100.0  # plausible box coords so downstream math does not NaN
    arr[4:] = 0.9    # every class channel "confident", easy to threshold
    return arr[None, ...]


# -- yolo_decode shape guards ---------------------------------------------

def test_yolo_decode_rejects_wrong_class_count():
    raw = _raw(n_boxes=50, n_channels=4 + 6)  # 6 classes in the array
    with pytest.raises(ValueError, match="class"):
        yolo_decode(raw, 1.0, (0.0, 0.0), 0.5, 0.5, _name_of, expect_nc=1)


def test_yolo_decode_rejects_non_2d_after_squeeze():
    raw = np.zeros((1, 1, 5, 10), dtype=np.float32)  # 4-D, still 3-D after squeeze
    with pytest.raises(ValueError):
        yolo_decode(raw, 1.0, (0.0, 0.0), 0.5, 0.5, _name_of)


def test_yolo_decode_still_works_without_expect_nc():
    raw = _raw(n_boxes=30, n_channels=4 + 1)
    out = yolo_decode(raw, 1.0, (0.0, 0.0), 0.5, 0.5, _name_of)
    assert len(out) >= 1


# -- ModelCard --------------------------------------------------------------

def _card(**overrides) -> ModelCard:
    base = dict(
        format="rknn", precision="fp16", imgsz=640, layout="nhwc",
        input_dtype="uint8", mean_values=[0.0, 0.0, 0.0], std_values=[255.0, 255.0, 255.0],
        end2end=False, target_platform="rk3588", class_names=["person"],
        source_weights="yolo26n.onnx", toolkit_version="2.0.0",
        ultralytics_version="8.3.0", exported_at="2026-09-24T00:00:00",
    )
    base.update(overrides)
    return ModelCard(**base)


def test_modelcard_roundtrip(tmp_path):
    weights = tmp_path / "model.rknn"
    weights.write_bytes(b"fake")
    card = _card()
    card.save(weights)
    loaded = ModelCard.load(weights)
    assert loaded is not None
    assert loaded.imgsz == 640
    assert loaded.end2end is False
    assert loaded.class_names == ["person"]


def test_modelcard_load_missing_returns_none(tmp_path):
    assert ModelCard.load(tmp_path / "nothing_here.rknn") is None


def test_modelcard_sidecar_path_is_next_to_weights(tmp_path):
    weights = tmp_path / "sub" / "model.rknn"
    p = ModelCard.sidecar_path(weights)
    assert p == Path(str(weights) + ".card.json")
    assert p.parent == weights.parent


# -- RKNNBackend guards, rknnlite mocked -------------------------------------

class _FakeRKNNLite:
    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1_2 = 7

    def load_rknn(self, path):
        return 0

    def init_runtime(self, core_mask=0):
        return 0

    def inference(self, inputs):
        return [_raw(n_boxes=30, n_channels=5)]

    def release(self):
        pass


@pytest.fixture()
def mock_rknnlite(monkeypatch):
    fake = types.ModuleType("rknnlite.api")
    fake.RKNNLite = _FakeRKNNLite
    monkeypatch.setitem(sys.modules, "rknnlite.api", fake)
    # Force a fresh import each test so a previously-imported real
    # `labsafe.models.backends` (with a real rknnlite) cannot leak through.
    monkeypatch.delitem(sys.modules, "labsafe.models.backends", raising=False)
    import labsafe.models.backends as backends_mod
    yield backends_mod


def test_rknn_backend_rejects_end2end_card(tmp_path, mock_rknnlite):
    weights = tmp_path / "bad.rknn"
    weights.write_bytes(b"fake")
    _card(end2end=True, imgsz=320).save(weights)
    with pytest.raises(ValueError, match="23753"):
        mock_rknnlite.RKNNBackend(names=["person"], imgsz=320, conf=0.4, weights=str(weights))


def test_rknn_backend_rejects_imgsz_mismatch(tmp_path, mock_rknnlite):
    weights = tmp_path / "mismatch.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=640).save(weights)
    with pytest.raises(ValueError, match="imgsz"):
        mock_rknnlite.RKNNBackend(names=["person"], imgsz=320, conf=0.4, weights=str(weights))


def test_rknn_backend_loads_fine_without_a_card(tmp_path, mock_rknnlite, caplog):
    weights = tmp_path / "no_card.rknn"
    weights.write_bytes(b"fake")
    be = mock_rknnlite.RKNNBackend(names=["person"], imgsz=320, conf=0.4, weights=str(weights))
    assert be.card is None


# -- class-name mismatch guard, KNOWN_ISSUES.md #20 -------------------------
#
# A COCO-80 checkpoint fed a single-class `configs/classes.yaml` used to
# die inside `yolo_decode(..., expect_nc=1)` at warmup, well after the NPU
# runtime was already opened. This guard raises in `__init__`, before
# `init_runtime` touches the NPU at all.

def test_rknn_backend_rejects_class_name_mismatch(tmp_path, mock_rknnlite):
    weights = tmp_path / "coco80.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=320, class_names=["person", "bicycle", "car"] + [f"c{i}" for i in range(77)]).save(weights)
    with pytest.raises(ValueError, match="classes_coco80"):
        mock_rknnlite.RKNNBackend(names=["person"], imgsz=320, conf=0.4, weights=str(weights))


def test_rknn_backend_loads_when_class_names_match(tmp_path, mock_rknnlite):
    weights = tmp_path / "matching.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=320, class_names=["person"]).save(weights)
    be = mock_rknnlite.RKNNBackend(names=["person"], imgsz=320, conf=0.4, weights=str(weights))
    assert be.card is not None


def test_rknn_backend_last_timing_has_three_keys_after_one_call(tmp_path, mock_rknnlite):
    weights = tmp_path / "timed.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=320, class_names=["person"]).save(weights)
    be = mock_rknnlite.RKNNBackend(names=["person"], imgsz=320, conf=0.4, weights=str(weights))
    be(np.zeros((480, 640, 3), dtype=np.uint8))
    assert set(be.last_timing) == {"pre_ms", "infer_ms", "post_ms"}
    assert all(v >= 0 for v in be.last_timing.values())


# -- INT8 normalised box coordinates, KNOWN_ISSUES.md #24 -------------------
#
# Ultralytics' RKNN INT8 export divides box coordinates by imgsz so boxes and
# class scores share one INT8 scale. Decoded as pixels, class and score stay
# correct and every box shrinks ~640x -- measured on the board: parity
# recall 0.0 with 49 "detections", no error anywhere.

def _one_box(cx: float, cy: float, w: float, h: float, n_boxes: int = 30) -> np.ndarray:
    """(1, 5, N) head output with a single confident box at anchor 0."""
    arr = np.zeros((5, n_boxes), dtype=np.float32)
    arr[:4, 0] = (cx, cy, w, h)
    arr[4, 0] = 0.9
    return arr[None, ...]


def test_yolo_decode_box_scale_maps_normalized_boxes_back_to_pixels():
    pixels = yolo_decode(_one_box(320, 320, 100, 200), 1.0, (0.0, 0.0), 0.5, 0.5, _name_of)
    norm = yolo_decode(_one_box(0.5, 0.5, 100 / 640, 200 / 640), 1.0, (0.0, 0.0), 0.5, 0.5,
                       _name_of, box_scale=640.0)
    assert len(pixels) == len(norm) == 1
    a, b = pixels[0], norm[0]
    assert (a.x1, a.y1, a.x2, a.y2) == pytest.approx((b.x1, b.y1, b.x2, b.y2), abs=1e-3)
    assert (a.x1, a.y1, a.x2, a.y2) == pytest.approx((270, 220, 370, 420), abs=1e-3)


def _normalized_backend(tmp_path, mock_rknnlite, **card_overrides):
    weights = tmp_path / "int8.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=640, precision="int8", **card_overrides).save(weights)
    be = mock_rknnlite.RKNNBackend(names=["person"], imgsz=640, conf=0.4, weights=str(weights))
    be.rknn.inference = lambda inputs: [_one_box(0.5, 0.5, 100 / 640, 200 / 640)]
    return be


def test_rknn_backend_rescales_boxes_for_a_normalized_int8_card(tmp_path, mock_rknnlite):
    be = _normalized_backend(tmp_path, mock_rknnlite, box_coords="normalized")
    assert be.box_scale == 640.0
    (det,) = be(np.zeros((640, 640, 3), dtype=np.uint8))
    assert (det.x1, det.y1, det.x2, det.y2) == pytest.approx((270, 220, 370, 420), abs=1e-3)


def test_normalized_output_decoded_as_pixels_gives_a_box_that_matches_nothing(tmp_path, mock_rknnlite):
    """The bug shape itself: a 'pixels' card on normalized output still
    returns a confident detection -- just a sub-pixel box in the corner."""
    be = _normalized_backend(tmp_path, mock_rknnlite, box_coords="pixels")
    (det,) = be(np.zeros((640, 640, 3), dtype=np.uint8))
    assert det.score == pytest.approx(0.9)
    assert det.x2 - det.x1 < 1.0 and det.y2 - det.y1 < 1.0


def test_rknn_backend_refuses_an_int8_card_without_box_coords(tmp_path, mock_rknnlite):
    weights = tmp_path / "old_int8.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=640, precision="int8").save(weights)  # box_coords defaults to None
    with pytest.raises(ValueError, match="box_coords"):
        mock_rknnlite.RKNNBackend(names=["person"], imgsz=640, conf=0.4, weights=str(weights))


def test_rknn_backend_fp16_card_without_box_coords_decodes_as_pixels(tmp_path, mock_rknnlite):
    weights = tmp_path / "old_fp16.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=640, precision="fp16").save(weights)
    be = mock_rknnlite.RKNNBackend(names=["person"], imgsz=640, conf=0.4, weights=str(weights))
    assert be.box_scale == 1.0


def test_rknn_backend_rejects_an_unknown_box_coords_value(tmp_path, mock_rknnlite):
    weights = tmp_path / "weird.rknn"
    weights.write_bytes(b"fake")
    _card(imgsz=640, box_coords="xyxy").save(weights)
    with pytest.raises(ValueError, match="box_coords"):
        mock_rknnlite.RKNNBackend(names=["person"], imgsz=640, conf=0.4, weights=str(weights))


def test_modelcard_written_before_box_coords_existed_loads_as_none(tmp_path):
    import json
    weights = tmp_path / "legacy.rknn"
    weights.write_bytes(b"fake")
    data = _card().as_dict()
    del data["box_coords"]
    ModelCard.sidecar_path(weights).write_text(json.dumps(data))
    assert ModelCard.load(weights).box_coords is None


# -- class_subset: decode only the classes keep_classes would keep ----------

def _two_class_boxes() -> np.ndarray:
    """(1, 4+3, N): anchor 0 = person 0.9; anchor 1 = car 0.9 (person 0.5); anchor 2 = car only."""
    arr = np.zeros((7, 40), dtype=np.float32)
    arr[:4, 0] = (100, 100, 40, 80); arr[4, 0] = 0.9
    arr[:4, 1] = (300, 300, 40, 80); arr[5, 1] = 0.9; arr[4, 1] = 0.5
    arr[:4, 2] = (500, 100, 40, 80); arr[5, 2] = 0.9
    return arr[None]


def test_class_subset_matches_full_decode_filtered_to_the_same_classes_on_clear_cases():
    raw = _two_class_boxes()
    full = [d for d in yolo_decode(raw, 1.0, (0., 0.), 0.4, 0.5, _name_of) if d.cls_id == 0]
    sub = yolo_decode(raw, 1.0, (0., 0.), 0.4, 0.5, _name_of, class_subset=[0])
    assert all(d.cls_id == 0 and d.cls_name == "cls_0" for d in sub)
    assert {(round(d.x1), round(d.y1)) for d in full} <= {(round(d.x1), round(d.y1)) for d in sub}


def test_class_subset_keeps_a_box_whose_best_class_is_another_but_person_score_passes():
    """The documented semantic difference: anchor 1 has car 0.9 > person 0.5."""
    raw = _two_class_boxes()
    full_person = [d for d in yolo_decode(raw, 1.0, (0., 0.), 0.4, 0.5, _name_of) if d.cls_id == 0]
    sub = yolo_decode(raw, 1.0, (0., 0.), 0.4, 0.5, _name_of, class_subset=[0])
    assert len(full_person) == 1 and len(sub) == 2


def test_class_subset_maps_ids_back_for_a_non_zero_class():
    raw = _two_class_boxes()
    out = yolo_decode(raw, 1.0, (0., 0.), 0.4, 0.5, _name_of, class_subset=[1])
    assert out and all(d.cls_id == 1 and d.cls_name == "cls_1" for d in out)


def test_class_subset_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        yolo_decode(_two_class_boxes(), 1.0, (0., 0.), 0.4, 0.5, _name_of, class_subset=[9])


def test_backend_passes_keep_classes_ids_to_the_decoder(tmp_path, mock_rknnlite):
    weights = tmp_path / "coco.rknn"
    weights.write_bytes(b"fake")
    names = [f"c{i}" for i in range(80)]; names[0] = "person"
    _card(imgsz=640, class_names=names).save(weights)
    be = mock_rknnlite.RKNNBackend(names=names, imgsz=640, conf=0.4, weights=str(weights), keep_classes=["person"])
    assert be.class_subset == (0,)
    assert mock_rknnlite.RKNNBackend(names=names, imgsz=640, conf=0.4, weights=str(weights)).class_subset is None
