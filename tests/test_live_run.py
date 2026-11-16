"""Tests for the board-first-live-run additions:

  * `keep_classes` filtering on `Backend` (COCO-80 checkpoint, project
    cares about `person` only -- docs/KNOWN_ISSUES.md #20)
  * per-stage timing breakdown (`Backend.last_timing`,
    `RouterStats.stage_a_breakdown_ms`)
  * `CameraSource.duplicate_reads` (a relative of docs/KNOWN_ISSUES.md #6)
  * `FrameSaver`
  * `app.camera_stats` / `app._duplicate_warning`
  * `app.main` end to end with a fake camera (bench save-frames path, and
    the live-mode KeyboardInterrupt -> JSON-summary fix, docs/KNOWN_ISSUES.md
    plan §2.5)

None of this needs a board, a real camera, or `rknnlite` -- the RKNN-card
guard test lives in `tests/test_backends_guard.py`, which already has the
`mock_rknnlite` fixture.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labsafe.app import _duplicate_warning, camera_stats, main  # noqa: E402
from labsafe.io.camera import CameraSource  # noqa: E402
from labsafe.io.snapshots import FrameSaver  # noqa: E402
from labsafe.models.backends import Backend, Detection, StubBackend, build_backend  # noqa: E402
from labsafe.pipeline.fall import FallConfig  # noqa: E402
from labsafe.pipeline.router import AdaptiveRouter  # noqa: E402


def det(name="person", score=0.9, x1=0, y1=0, x2=10, y2=10):
    return Detection(x1, y1, x2, y2, score, 0, name)


# -- keep_classes ----------------------------------------------------------

class _FakeMultiClassBackend(Backend):
    name = "fake_multiclass"

    def _infer(self, frame):
        return [det("person"), det("chair"), det("person")]


def test_keep_classes_filters_to_the_named_classes():
    be = _FakeMultiClassBackend(names=["person", "chair"], imgsz=320, conf=0.4,
                                 keep_classes=["person"])
    out = be(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 2
    assert all(d.cls_name == "person" for d in out)
    assert be.last_latency_ms > 0


def test_keep_classes_rejects_unknown_name():
    with pytest.raises(ValueError, match="not in names"):
        _FakeMultiClassBackend(names=["person"], imgsz=320, conf=0.4,
                                keep_classes=["dog"])


def test_keep_classes_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        _FakeMultiClassBackend(names=["person"], imgsz=320, conf=0.4, keep_classes=[])


def test_keep_classes_omitted_does_not_filter():
    be = _FakeMultiClassBackend(names=["person", "chair"], imgsz=320, conf=0.4)
    out = be(np.zeros((4, 4, 3), dtype=np.uint8))
    assert len(out) == 3


def test_stub_backend_accepts_keep_classes_via_build_backend():
    be = build_backend(
        {"backend": "stub", "keep_classes": ["person"]}, ["person"],
    )
    out = be(np.zeros((10, 10, 3), dtype=np.uint8))
    assert isinstance(out, list)  # did not raise; keep_classes did not leak into StubBackend kwargs


# -- router breakdown --------------------------------------------------------

class _TimedBackend(Backend):
    name = "timed"

    def _infer(self, frame):
        self.last_timing = {"pre_ms": 1.0, "infer_ms": 2.0, "post_ms": 0.5}
        return []


def _router(stage_a) -> AdaptiveRouter:
    return AdaptiveRouter(stage_a, None, None, [], FallConfig(), mode="parallel")


def test_router_records_stage_a_breakdown():
    r = _router(_TimedBackend(names=["person"], imgsz=320, conf=0.4))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(5):
        r.process(frame)
    summary = r.stats.summary()
    assert summary["stage_a_breakdown_ms"]["infer_ms"]["n"] == 5
    assert summary["stage_a_breakdown_ms"]["pre_ms"]["n"] == 5


def test_router_breakdown_empty_for_stub():
    r = _router(StubBackend(names=["person"], imgsz=320, conf=0.4))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    r.process(frame)
    assert r.stats.summary()["stage_a_breakdown_ms"] == {}


# -- camera duplicate_reads --------------------------------------------------

def _bare_camera() -> CameraSource:
    cam = CameraSource.__new__(CameraSource)
    cam.threaded = True
    import threading
    cam._lock = threading.Lock()
    cam._cond = threading.Condition(cam._lock)
    cam._frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cam.frames_read = 1
    cam.duplicate_reads = 0
    cam._last_returned_seq = -1
    cam.stall_s = 3.0
    cam.last_frame_t = time.time()
    return cam


def test_camera_duplicate_reads_counts_repeated_frame():
    cam = _bare_camera()
    ok1, _ = cam.read()
    ok2, _ = cam.read()
    assert ok1 and ok2
    assert cam.duplicate_reads == 1, "second read of the same frame must be a duplicate"

    cam.frames_read += 1
    cam.read()
    assert cam.duplicate_reads == 1, "a genuinely new frame must not be counted as a duplicate"


def test_camera_read_with_no_frame_yet():
    cam = _bare_camera()
    cam._frame = None
    ok, frame = cam.read()
    assert ok is False and frame is None


# -- FrameSaver --------------------------------------------------------------

def test_frame_saver_respects_interval_and_max(tmp_path):
    saver = FrameSaver(tmp_path / "frames", every_s=3, max_n=2)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert saver.due(0.0) is True
    saver.save(0.0, frame, {"tracks": [{"track_id": 1, "score": 0.77}]})

    assert saver.due(1.0) is False       # too soon
    assert saver.due(3.1) is True
    saver.save(3.1, frame, {"tracks": [{"track_id": 1, "score": 0.80}]})

    assert saver.due(6.5) is False       # hit max_n
    assert saver.saved == 2

    jpgs = sorted((tmp_path / "frames").glob("*.jpg"))
    jsons = sorted((tmp_path / "frames").glob("*.json"))
    assert len(jpgs) == 2
    assert len(jsons) == 2
    payload = json.loads(jsons[0].read_text())
    assert payload["tracks"][0]["score"] == 0.77


def test_frame_saver_disabled_when_every_s_zero(tmp_path):
    saver = FrameSaver(tmp_path / "frames", every_s=0, max_n=5)
    assert saver.due(0.0) is False
    assert saver.due(1000.0) is False


# -- _duplicate_warning -------------------------------------------------------

def test_duplicate_warning_fires_above_tolerance():
    w = _duplicate_warning({"duplicate_ratio": 0.2, "capture_fps": 14.0})
    assert w is not None
    assert "duplicate_ratio" in w


def test_duplicate_warning_silent_below_tolerance():
    assert _duplicate_warning({"duplicate_ratio": 0.01, "capture_fps": 30.0}) is None


# -- camera_stats -------------------------------------------------------------

class _FakeCamForStats:
    def __init__(self, frames_read=10, duplicate_reads=2):
        self.frames_read = frames_read
        self.duplicate_reads = duplicate_reads
        self.t_open = time.time() - 1.0

    @property
    def resolution(self):
        return (640, 480)


def test_camera_stats_shape():
    cs = camera_stats(_FakeCamForStats(), frames_processed=10, read_ms=[1.0, 2.0, 3.0])
    assert cs["resolution"] == [640, 480]
    assert cs["duplicate_reads"] == 2
    assert cs["duplicate_ratio"] == 0.2
    assert cs["read_ms"]["n"] == 3
    assert cs["capture_fps"] is not None


# -- app.main integration, fake camera ---------------------------------------

class _FakeLiveCamera:
    """Stands in for CameraSource: same surface `app.py` touches."""

    def __init__(self, device, width, height, fps, threaded=None, raise_at=None):
        self.is_file = False
        self.frames_read = 0
        self.duplicate_reads = 0
        self.t_open = time.time()
        self._raise_at = raise_at
        self._w, self._h = width, height

    def read(self, wait_new=False, timeout_s=0.25):
        self.frames_read += 1
        if self._raise_at is not None and self.frames_read >= self._raise_at:
            raise KeyboardInterrupt
        frame = np.random.randint(0, 255, (self._h, self._w, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        pass

    @property
    def resolution(self):
        return (self._w, self._h)


def _write_minimal_configs(tmp_path: Path) -> tuple[Path, Path]:
    pipeline = {
        "source": {"device": 0, "width": 640, "height": 480, "fps": 30},
        "stage_a": {"backend": "stub", "imgsz": 320, "conf": 0.4},
        "stage_b": {"enabled": False},
        "stage_c": {"enabled": False},
        "fall": {"enabled": True},
        "zones": {"enabled": False},
        "tracker": {"max_age_s": 1.5, "iou_match_threshold": 0.3},
        "affinity": {"cpus": "none"},
        "alerts": {"gpio": "off", "debounce_s": 3.0, "cooldown_s": 10.0},
        "dashboard": {"enabled": False, "host": "0.0.0.0", "port": 8000, "jpeg_quality": 70},
    }
    classes = {
        "stage_a": {"names": ["person"]},
        "stage_b": {"names": [], "required": []},
        "stage_c": {"names": []},
    }
    p_path = tmp_path / "pipeline.yaml"
    c_path = tmp_path / "classes.yaml"
    p_path.write_text(yaml.safe_dump(pipeline))
    c_path.write_text(yaml.safe_dump(classes))
    return p_path, c_path


def test_app_main_bench_saves_frames_and_camera_stats(tmp_path, monkeypatch, capsys):
    p_path, c_path = _write_minimal_configs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "labsafe.app.CameraSource",
        lambda device, width, height, fps, threaded=None: _FakeLiveCamera(
            device, width, height, fps, threaded
        ),
    )

    rc = main([
        "--bench", "--exp", "T_LIVE", "--config", str(p_path), "--classes", str(c_path),
        "--duration", "2", "--warmup", "0", "--no-display", "--gpio", "off",
        "--save-every", "0.5", "--save-max", "3",
    ])
    assert rc == 0

    run_dirs = sorted((tmp_path / "runs").glob("T_LIVE-*"))
    assert run_dirs, "bench run directory was not created"
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text())
    assert "camera" in metrics
    assert metrics["camera"]["resolution"] == [640, 480]
    # Tightened per review: a 2s run with --save-every 0.5 --save-max 3 must
    # actually save something. `<= 3` alone would also pass on 0 saved
    # frames (a broken FrameSaver that never fires), which is exactly the
    # failure mode this test exists to catch.
    assert 1 <= metrics["frames_saved"] <= 3

    frame_files = sorted((run_dirs[0] / "frames").glob("*.jpg"))
    assert 1 <= len(frame_files) <= 3
    assert len(frame_files) == metrics["frames_saved"]


def test_app_main_live_prints_summary_on_keyboardinterrupt(tmp_path, monkeypatch, capsys):
    p_path, c_path = _write_minimal_configs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "labsafe.app.CameraSource",
        lambda device, width, height, fps, threaded=None: _FakeLiveCamera(
            device, width, height, fps, threaded, raise_at=5
        ),
    )

    run_dir = tmp_path / "captured-live"
    rc = main([
        "--config", str(p_path), "--classes", str(c_path),
        "--no-display", "--gpio", "off", "--record-video",
        "--record-fps", "10", "--run-dir", str(run_dir),
    ])
    assert rc == 0

    out = capsys.readouterr().out
    summary = json.loads(out)
    assert "frames" in summary
    assert "camera" in summary
    assert summary["frames"] >= 1
    assert (run_dir / "annotated.mp4").stat().st_size > 0
    assert (run_dir / "run.log").exists()
    assert (run_dir / "command.json").exists()
    persisted = json.loads((run_dir / "summary.json").read_text())
    assert persisted["video"]["frames_written"] >= 1
    assert persisted["video"]["error"] is None


# -- --gpio CLI override (plan §2.6) ------------------------------------------

def test_gpio_off_flag_overrides_config_auto(tmp_path, monkeypatch):
    """`configs/pipeline.yaml` ships `alerts.gpio: auto` with Raspberry-Pi
    BCM pin numbers baked in (plan §2.6). `--gpio off` must win over a
    config that still says `auto`, or a first board run could drive an
    unconfirmed GPIO line. Every other integration test in this file uses a
    config that already says `gpio: off`, so none of them can distinguish
    "the CLI override worked" from "the config already said off" -- this
    test starts from `auto` and asserts `AlertSink._try_gpio` (the only
    thing that would touch real hardware) is never even called.
    """
    from labsafe.io.alerts import AlertSink

    p_path, c_path = _write_minimal_configs(tmp_path)
    pipeline = yaml.safe_load(p_path.read_text())
    pipeline["alerts"]["gpio"] = "auto"
    p_path.write_text(yaml.safe_dump(pipeline))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "labsafe.app.CameraSource",
        lambda device, width, height, fps, threaded=None: _FakeLiveCamera(
            device, width, height, fps, threaded, raise_at=3
        ),
    )

    def _fail_if_called(self, required):
        raise AssertionError(
            "AlertSink._try_gpio was called -- --gpio off did not override "
            "the config's alerts.gpio: auto"
        )

    monkeypatch.setattr(AlertSink, "_try_gpio", _fail_if_called)

    rc = main([
        "--config", str(p_path), "--classes", str(c_path),
        "--no-display", "--gpio", "off",
    ])
    assert rc == 0


# -- configs: classes_coco80.yaml + N1L/N0L (plan §8 mục 10) ------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLASSES_COCO80 = _REPO_ROOT / "configs" / "classes_coco80.yaml"
_CLASSES_YAML = _REPO_ROOT / "configs" / "classes.yaml"

# Independently retyped here (not copied from configs/classes_coco80.yaml
# itself, which would make this test agree with a corrupted file just as
# happily as with a correct one) -- Ultralytics' own coco.yaml order, id
# 0..79. Cross-checked against the real installed `ultralytics` package
# below when available, and against the real board's `*.rknn.card.json`
# at authoring time (see classes_coco80.yaml's header comment).
_COCO80_CANONICAL_ORDER = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def test_classes_coco80_has_80_unique_names_person_first():
    cfg = yaml.safe_load(_CLASSES_COCO80.read_text())
    names = cfg["stage_a"]["names"]
    assert len(names) == 80
    assert len(set(names)) == 80, "duplicate class name(s) in classes_coco80.yaml"
    assert names[0] == "person"


def test_classes_coco80_matches_canonical_coco_order():
    cfg = yaml.safe_load(_CLASSES_COCO80.read_text())
    assert cfg["stage_a"]["names"] == _COCO80_CANONICAL_ORDER


def test_classes_coco80_matches_installed_ultralytics_coco_yaml_when_available():
    ultralytics = pytest.importorskip("ultralytics")
    coco_yaml = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco.yaml"
    if not coco_yaml.exists():
        pytest.skip("installed ultralytics has no cfg/datasets/coco.yaml")
    real = yaml.safe_load(coco_yaml.read_text())
    real_names = [real["names"][i] for i in sorted(real["names"])]
    cfg = yaml.safe_load(_CLASSES_COCO80.read_text())
    assert cfg["stage_a"]["names"] == real_names


def test_classes_coco80_stage_b_c_match_classes_yaml():
    coco80 = yaml.safe_load(_CLASSES_COCO80.read_text())
    product = yaml.safe_load(_CLASSES_YAML.read_text())
    assert coco80["stage_b"] == product["stage_b"]
    assert coco80["stage_c"] == product["stage_c"]


@pytest.mark.parametrize(
    "exp_name", ["N1L_live_rknn_fp16_640.yaml", "N0L_live_cpu_onnx_640.yaml"],
)
def test_live_exp_configs_keep_classes_subset_of_coco80_and_stage_bc_off(exp_name):
    coco80_names = set(yaml.safe_load(_CLASSES_COCO80.read_text())["stage_a"]["names"])
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "exp" / exp_name).read_text())
    keep = cfg["stage_a"].get("keep_classes")
    assert keep, f"{exp_name}: stage_a.keep_classes must be set for a COCO-80 checkpoint"
    assert set(keep) <= coco80_names, (
        f"{exp_name}: keep_classes {keep} is not a subset of classes_coco80.yaml"
    )
    assert cfg["stage_b"]["enabled"] is False
    assert cfg["stage_c"]["enabled"] is False


# -- CameraSource.read(wait_new=True): one inference per camera frame ----------

class _FakeCap:
    """A 'camera' producing a new noisy frame every `period` seconds."""

    def __init__(self, *_a, period=0.03, **_k):
        self.period = period
        self._rng = np.random.default_rng(0)

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 0.0

    def read(self, wait_new=False, timeout_s=0.25):
        import time
        time.sleep(self.period)
        return True, self._rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)

    def release(self):
        pass


def _fake_camera(monkeypatch, period=0.03):
    import cv2
    from labsafe.io.camera import CameraSource
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: _FakeCap(period=period))
    return CameraSource(0, threaded=True)


def test_wait_new_returns_each_camera_frame_once(monkeypatch):
    import time
    cam = _fake_camera(monkeypatch)
    cam.read(wait_new=True)                     # sync to the stream
    base_dups = cam.duplicate_reads
    t0 = time.perf_counter()
    for _ in range(15):
        ok, f = cam.read(wait_new=True)
        assert ok and f is not None
    elapsed = time.perf_counter() - t0
    assert cam.duplicate_reads == base_dups     # no stale frame was handed out
    assert 0.3 < elapsed < 1.2                  # paced by the 30 ms camera, not spinning
    cam.release()


def test_default_read_still_spins_and_counts_duplicates(monkeypatch):
    cam = _fake_camera(monkeypatch)
    for _ in range(50):
        cam.read()                              # old behaviour: no waiting
    assert cam.duplicate_reads > 20
    cam.release()


def test_wait_new_times_out_instead_of_hanging_when_the_camera_stalls(monkeypatch):
    import time
    cam = _fake_camera(monkeypatch, period=0.03)
    cam.read(wait_new=True)
    cam.cap.period = 5.0                        # the grabber is now stuck in a long read
    time.sleep(0.1)
    cam.read(wait_new=True)                     # consume what is there
    t0 = time.perf_counter()
    ok, f = cam.read(wait_new=True, timeout_s=0.1)
    assert ok and (time.perf_counter() - t0) < 0.5      # returned the stale frame at the timeout
    cam.stop_for_test = True
    cam._stop.set()
