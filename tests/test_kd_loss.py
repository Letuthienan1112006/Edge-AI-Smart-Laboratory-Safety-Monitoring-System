"""Real unit tests for KDDetectionLoss against real torch/ultralytics
objects -- skipped automatically where those are not installed (this
repo's default dev environment does not carry them; see KNOWN_ISSUES.md
#12's note on that). Models are built from ultralytics' own packaged YAML
configs with random init (`DetectionModel(cfg=..., nc=...)`), never a
downloaded checkpoint, so these tests need no network and are fast.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402

from labsafe.models.kd_loss import KDDetectionLoss  # noqa: E402

IMGSZ = 64  # divisible by the largest stride (32) with two more halvings to spare


def _build_model(cfg: str, nc: int = 80) -> DetectionModel:
    torch.manual_seed(hash((cfg, nc)) % (2**31))  # distinct-but-deterministic init per (cfg, nc)
    m = DetectionModel(cfg=cfg, nc=nc, verbose=False)
    m.args = get_cfg()
    m.train()
    return m


def _tiny_batch(bs: int, imgsz: int = IMGSZ) -> dict:
    return {
        "img": torch.rand(bs, 3, imgsz, imgsz),
        "batch_idx": torch.arange(bs, dtype=torch.float32),
        "cls": torch.zeros(bs, dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]] * bs, dtype=torch.float32),
    }


def test_alpha_zero_matches_plain_task_loss_exactly():
    """alpha=0 must reduce to exactly the ordinary (non-KD) task loss --
    the KD term is still computed (for logging) but must not move total."""
    student = _build_model("yolo11n.yaml")
    teacher = _build_model("yolo11n.yaml")
    batch = _tiny_batch(2)
    preds = student(batch["img"])

    kd = KDDetectionLoss(student, teacher, temperature=2.0, alpha=0.0)
    total, items = kd(preds, batch)

    plain = v8DetectionLoss(student)
    base_total, _ = plain(preds, batch)

    assert total.item() == pytest.approx(base_total.sum().item(), rel=1e-5)
    assert "kd_loss" in items
    assert items["kd_loss"].item() > 0  # independently-initialised teacher/student disagree


def test_alpha_one_matches_kd_loss_only():
    """alpha=1 must ignore ground truth entirely and optimise the KD term alone."""
    student = _build_model("yolo11n.yaml")
    teacher = _build_model("yolo11n.yaml")
    batch = _tiny_batch(2)
    preds = student(batch["img"])

    kd = KDDetectionLoss(student, teacher, temperature=2.0, alpha=1.0)
    total, items = kd(preds, batch)

    batch_size = batch["img"].shape[0]
    expected = items["kd_loss"] * batch_size
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)


def test_gradient_flows_from_kd_term_into_student_only():
    """The whole point of this class is a LOSS-based transfer -- if the KD
    term were accidentally detached from the student's graph, this would
    silently train nothing while still printing a plausible-looking
    kd_loss number every step. Verify real gradients land on the student
    and never touch the frozen teacher."""
    student = _build_model("yolo11n.yaml")
    teacher = _build_model("yolo11n.yaml")
    batch = _tiny_batch(1)
    preds = student(batch["img"])

    kd = KDDetectionLoss(student, teacher, temperature=2.0, alpha=1.0)  # KD-only
    total, _ = kd(preds, batch)
    total.backward()

    student_grad = sum(p.grad.abs().sum().item() for p in student.parameters() if p.grad is not None)
    assert student_grad > 0

    assert all(p.grad is None for p in teacher.parameters())


def test_classification_shape_mismatch_raises_loudly():
    """Different nc between teacher and student must fail loudly, not
    silently broadcast/misalign class logits."""
    student = _build_model("yolo11n.yaml", nc=80)
    teacher = _build_model("yolo11n.yaml", nc=5)
    batch = _tiny_batch(2)
    preds = student(batch["img"])

    kd = KDDetectionLoss(student, teacher, temperature=2.0, alpha=0.5)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        kd(preds, batch)


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": -1}])
def test_invalid_temperature_rejected(kwargs):
    student = _build_model("yolo11n.yaml")
    teacher = _build_model("yolo11n.yaml")
    with pytest.raises(ValueError):
        KDDetectionLoss(student, teacher, **kwargs)


@pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"alpha": 1.1}])
def test_invalid_alpha_rejected(kwargs):
    student = _build_model("yolo11n.yaml")
    teacher = _build_model("yolo11n.yaml")
    with pytest.raises(ValueError):
        KDDetectionLoss(student, teacher, **kwargs)


def test_yolo26_one2one_architecture_is_handled_via_one2many_branch():
    """YOLO26's default architecture carries a one2one_cv2 head, which
    means ultralytics uses E2ELoss (not plain v8DetectionLoss) and `preds`
    is nested as {"one2many": ..., "one2one": ...} -- verified for real,
    2026-09-26: assuming the flat v8DetectionLoss shape here crashed with
    KeyError('boxes') on the first real training batch. This test is the
    regression guard for that."""
    student = _build_model("yolo26n.yaml")
    teacher = _build_model("yolo26n.yaml")
    assert getattr(student.model[-1], "one2one_cv2", None) is not None  # confirms the premise

    batch = _tiny_batch(2)
    preds = student(batch["img"])
    assert isinstance(preds, dict) and "one2many" in preds and "one2one" in preds

    kd = KDDetectionLoss(student, teacher, temperature=2.0, alpha=0.5)
    assert kd.is_e2e is True
    total, items = kd(preds, batch)
    assert torch.isfinite(total)
    assert "kd_loss" in items

    kd.update()  # must delegate to E2ELoss.update() without raising
