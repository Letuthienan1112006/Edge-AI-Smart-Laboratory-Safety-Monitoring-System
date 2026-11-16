"""Inference backends behind one interface.

The whole point of the experiment matrix (E1-E4) is swapping the runtime
without touching the pipeline. So the pipeline only ever sees `Backend`.

`stub` exists so the pipeline, the tracker, the alert logic and the bench
harness can all be built and validated before a single model is trained.
It is not a placeholder to delete later: it is also the control condition
that measures the cost of everything that is *not* inference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger("labsafe.models.backends")


@dataclass
class Detection:
    """One detection in pixel coordinates of the frame it came from."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls_id: int
    cls_name: str

    @property
    def wh(self) -> tuple[float, float]:
        return self.x2 - self.x1, self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre: where a standing person touches the floor.

        Zone membership uses this, not the box centre -- a person standing
        just outside a zone has a box centre that drifts over it.
        """
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def area(self) -> float:
        w, h = self.wh
        return max(0.0, w) * max(0.0, h)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)],
            "score": round(self.score, 3),
            "cls": self.cls_name,
        }


class Backend:
    """Base class. Subclasses implement `_infer`."""

    name = "base"

    def __init__(self, names: Sequence[str], imgsz: int, conf: float, **kw: Any):
        self.names = list(names)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        keep = kw.pop("keep_classes", None)
        self.keep_classes: set[str] | None
        if keep is None:
            self.keep_classes = None
        else:
            keep = [str(x) for x in keep]
            if not keep:
                raise ValueError(
                    "keep_classes is empty -- every detection would be dropped. "
                    "Omit keep_classes entirely if you do not want filtering."
                )
            unknown = sorted(set(keep) - set(self.names))
            if unknown:
                raise ValueError(
                    f"keep_classes {unknown} not in names {self.names} -- these "
                    f"classes can never be produced by this backend's decode step. "
                    f"If this is a COCO-pretrained checkpoint, pass "
                    f"names=configs/classes_coco80.yaml's stage_a.names, not the "
                    f"project's own single-class configs/classes.yaml."
                )
            self.keep_classes = set(keep)
        # Class ids the decoder may look at, so it never scores the classes
        # that keep_classes would drop anyway (see yolo_decode `class_subset`).
        self.class_subset: tuple[int, ...] | None = (
            None if self.keep_classes is None
            else tuple(sorted(self.names.index(k) for k in self.keep_classes))
        )
        self.kw = kw
        self.last_latency_ms = 0.0
        self.last_timing: dict[str, float] = {}

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        self.last_timing = {}
        dets = self._infer(frame)
        if self.keep_classes is not None:
            dets = [d for d in dets if d.cls_name in self.keep_classes]
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return dets

    def _infer(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover
        raise NotImplementedError

    def warmup(self, n: int = 3, shape: tuple[int, int] = (480, 640)) -> None:
        """First inferences are always slow. Never let them into a measurement."""
        dummy = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        for _ in range(n):
            self(dummy)

    def _name_of(self, cls_id: int) -> str:
        if 0 <= cls_id < len(self.names):
            return self.names[cls_id]
        return f"cls_{cls_id}"


class StubBackend(Backend):
    """Deterministic fake detector with a configurable CPU cost.

    `sim_latency_ms` lets you rehearse a target device's timing on the
    laptop: set it to a measured Pi 4 latency and the whole pipeline
    behaves as it will on the Pi, minus the inference itself.
    """

    name = "stub"

    def __init__(self, names, imgsz, conf, sim_latency_ms: float = 0.0, **kw):
        super().__init__(names, imgsz, conf, **kw)
        self.sim_latency_ms = float(sim_latency_ms)
        self._t0 = time.time()

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        if self.sim_latency_ms > 0:
            end = time.perf_counter() + self.sim_latency_ms / 1000.0
            while time.perf_counter() < end:
                pass  # busy-wait: models burn CPU, sleep() does not
        h, w = frame.shape[:2]
        # A single object tracing a slow horizontal path, so the tracker,
        # the zone test and the fall state machine all get something moving.
        t = time.time() - self._t0
        cx = (0.5 + 0.35 * np.sin(t * 0.45)) * w
        bw, bh = 0.16 * w, 0.55 * h
        cy = 0.60 * h
        score = 0.90
        if score < self.conf:
            # Every real backend thresholds by `conf` inside its decode
            # step; the stub must too, or it cannot be used to rehearse a
            # "model detects nothing" scenario (parity gate, liveness
            # telemetry) without a second, separate fake backend class.
            return []
        return [
            Detection(
                x1=cx - bw / 2, y1=cy - bh / 2, x2=cx + bw / 2, y2=cy + bh / 2,
                score=score, cls_id=0, cls_name=self._name_of(0),
            )
        ]


class UltralyticsBackend(Backend):
    """PyTorch / YOLO. Training machine and experiment E1 only."""

    name = "ultralytics"

    def __init__(self, names, imgsz, conf, weights: str, iou: float = 0.5, **kw):
        super().__init__(names, imgsz, conf, **kw)
        from ultralytics import YOLO  # imported lazily: never needed on the Pi

        self.model = YOLO(weights)
        self.iou = iou
        if not self.names:
            self.names = [self.model.names[i] for i in sorted(self.model.names)]

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )[0]
        out: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cid = int(b.cls.item())
            out.append(
                Detection(x1, y1, x2, y2, float(b.conf.item()), cid, self._name_of(cid))
            )
        return out


class ONNXBackend(Backend):
    """onnxruntime. Useful as a portable middle point between torch and NCNN."""

    name = "onnx"

    def __init__(self, names, imgsz, conf, weights: str, iou: float = 0.5, **kw):
        super().__init__(names, imgsz, conf, **kw)
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(kw.get("threads", 4))
        self.sess = ort.InferenceSession(weights, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.iou = iou

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        from .postprocess import assemble_split_outputs, letterbox, yolo_decode

        t0 = time.perf_counter()
        blob, scale, pad = letterbox(frame, self.imgsz)
        t1 = time.perf_counter()
        outs = self.sess.run(None, {self.input_name: blob})
        raw = assemble_split_outputs(outs, len(self.names)) if len(outs) > 1 else outs[0]
        t2 = time.perf_counter()
        dets = yolo_decode(raw, scale, pad, self.conf, self.iou, self._name_of,
                            expect_nc=len(self.names), class_subset=self.class_subset)
        t3 = time.perf_counter()
        self.last_timing = {
            "pre_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
        }
        return dets


class NCNNBackend(Backend):
    """NCNN via ultralytics' exported `*_ncnn_model` directory.

    Ultralytics recommends NCNN for Raspberry Pi; this keeps the export and
    the runtime consistent so E2 differs from E1 in exactly one variable.
    """

    name = "ncnn"

    def __init__(self, names, imgsz, conf, weights: str, iou: float = 0.5, **kw):
        super().__init__(names, imgsz, conf, **kw)
        from ultralytics import YOLO

        self.model = YOLO(str(Path(weights)), task=kw.get("task", "detect"))
        self.iou = iou

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )[0]
        out: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cid = int(b.cls.item())
            out.append(
                Detection(x1, y1, x2, y2, float(b.conf.item()), cid, self._name_of(cid))
            )
        return out


class TFLiteBackend(Backend):
    """LiteRT / TFLite, including INT8. Experiment E3."""

    name = "tflite"

    def __init__(self, names, imgsz, conf, weights: str, iou: float = 0.5, **kw):
        super().__init__(names, imgsz, conf, **kw)
        try:
            from tflite_runtime.interpreter import Interpreter  # Pi
        except ImportError:  # laptop
            from tensorflow.lite import Interpreter  # type: ignore

        self.interp = Interpreter(model_path=weights, num_threads=int(kw.get("threads", 4)))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.iou = iou

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        from .postprocess import letterbox, yolo_decode

        blob, scale, pad = letterbox(frame, self.imgsz, nchw=False)
        if self.inp["dtype"] == np.int8:
            s, z = self.inp["quantization"]
            blob = (blob / s + z).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], blob)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            s, z = self.out["quantization"]
            raw = (raw.astype(np.float32) - z) * s
        return yolo_decode(raw, scale, pad, self.conf, self.iou, self._name_of,
                            expect_nc=len(self.names), class_subset=self.class_subset)


class RKNNBackend(Backend):
    """Rockchip NPU (RKNPU2) -- RK3588/RK3588S, RK3566, RK3568.

    The NPU is a fixed-function INT8 engine, so two things differ from a
    CPU backend and both are silent when wrong:

      * The model is converted offline (rknn-toolkit2, on an x86 host) and
        the mean/std normalisation is baked into the .rknn. The runtime
        therefore wants raw uint8 NHWC, not a normalised float tensor --
        feeding it normalised input produces a model that simply detects
        nothing, with no error.
      * RK3588 has THREE NPU cores. `core_mask` chooses how many. One core
        is the default; all three raise throughput and power together. That
        is a knob worth sweeping, and reporting: cores-vs-latency-vs-watts
        is a result, not a configuration detail.

    Ops the converter cannot map fall back to the CPU. A model that looks
    "NPU accelerated" can be spending most of its time on fallbacks, so
    always compare against the CPU backend rather than assuming a win.
    """

    name = "rknn"

    def __init__(self, names, imgsz, conf, weights: str, iou: float = 0.5,
                 core_mask: str = "auto", **kw):
        super().__init__(names, imgsz, conf, **kw)
        try:
            from rknnlite.api import RKNNLite  # on-device runtime
        except ImportError as e:  # pragma: no cover - device only
            raise ImportError(
                "rknnlite not installed. On the board:\n"
                "  pip install rknn-toolkit-lite2\n"
                "and convert the model on an x86 host with rknn-toolkit2 "
                "(ONNX -> .rknn). The two packages are different and their "
                "versions must match the librknnrt.so on the board."
            ) from e

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(str(weights)) != 0:
            raise RuntimeError(f"failed to load rknn model: {weights}")

        # The card is the only place that says whether this specific file
        # is safe to run at all -- checked BEFORE init_runtime touches the
        # NPU, because end2end=True is a segfault on RK3588, not a wrong
        # answer (KNOWN_ISSUES.md #2).
        from .modelcard import ModelCard

        self.card = ModelCard.load(weights)
        self.box_scale = 1.0
        if self.card is not None:
            self.box_scale = self._resolve_box_scale(self.card, weights)
            if self.card.imgsz != self.imgsz:
                raise ValueError(
                    f"model card says imgsz={self.card.imgsz} but this backend was "
                    f"built with imgsz={self.imgsz} -- these must match exactly, "
                    f"RKNN input shape is fixed at convert time"
                )
            if self.card.input_dtype != "uint8":
                raise ValueError(
                    f"model card says input_dtype={self.card.input_dtype!r}, but "
                    f"this backend always feeds letterbox(normalize=False) uint8 "
                    f"NHWC. That preprocessing is only correct for a model whose "
                    f"mean/std were baked in at convert time (input_dtype=uint8) "
                    f"-- see KNOWN_ISSUES.md #4."
                )
            if self.card.end2end:
                raise ValueError(
                    f"model card says end2end=True for {weights} -- this segfaults "
                    f"on RK3588 at inference time (ultralytics issue #23753, see "
                    f"KNOWN_ISSUES.md #2). Re-export with end2end=False before "
                    f"using this model on the NPU."
                )
            if self.card.class_names and list(self.card.class_names) != list(self.names):
                raise ValueError(
                    f"model card for {weights} was exported with "
                    f"{len(self.card.class_names)} class(es) "
                    f"{list(self.card.class_names)[:3]}..., but the running config "
                    f"declares {len(self.names)} class(es) {list(self.names)[:3]}... "
                    f"-- yolo_decode() would either raise on the channel-count "
                    f"mismatch or, worse, silently argmax over the wrong classes. "
                    f"If this is a COCO-pretrained checkpoint that has not been "
                    f"fine-tuned for this project's own classes yet, run with "
                    f"`--classes configs/classes_coco80.yaml` and set "
                    f"`stage_a.keep_classes: [person]` in the pipeline config "
                    f"(see KNOWN_ISSUES.md #20)."
                )
        else:
            log.warning(
                "no model card next to %s; preprocessing assumptions "
                "(imgsz, mean/std baked in, end2end=False) are unverified", weights
            )

        masks = {
            "auto": getattr(RKNNLite, "NPU_CORE_AUTO", 0),
            "0": getattr(RKNNLite, "NPU_CORE_0", 1),
            "1": getattr(RKNNLite, "NPU_CORE_1", 2),
            "2": getattr(RKNNLite, "NPU_CORE_2", 4),
            "all": getattr(RKNNLite, "NPU_CORE_0_1_2", 7),
        }
        if str(core_mask) not in masks:
            raise ValueError(f"core_mask must be one of {sorted(masks)}")
        self.core_mask_name = str(core_mask)
        if self.rknn.init_runtime(core_mask=masks[str(core_mask)]) != 0:
            raise RuntimeError("failed to init rknn runtime")
        self.iou = iou

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        from .postprocess import assemble_split_outputs, letterbox, yolo_decode

        t0 = time.perf_counter()
        blob, scale, pad = letterbox(frame, self.imgsz, nchw=False, normalize=False)
        t1 = time.perf_counter()
        # `infer_ms` times the RKNNLite.inference() call as a whole -- input
        # transfer to the NPU, the NPU run itself, and the fp16->fp32
        # conversion of the output tensor on the way back. It is NOT the
        # time spent computing on the NPU alone; there is no separate hook
        # here for that.
        outputs = self.rknn.inference(inputs=[blob])
        t2 = time.perf_counter()
        if len(outputs) == 1:
            dets = yolo_decode(outputs[0], scale, pad, self.conf, self.iou, self._name_of,
                                expect_nc=len(self.names), box_scale=self.box_scale,
                                class_subset=self.class_subset)
        elif len(outputs) % 2 == 0 and self.box_scale == 1.0:
            # "split" export: raw per-scale (box ltrb, class prob) tensors, decoded here (see assemble_split_outputs)
            dets = yolo_decode(assemble_split_outputs(outputs, len(self.names)), scale, pad, self.conf, self.iou,
                                self._name_of, expect_nc=len(self.names), class_subset=self.class_subset)
        else:
            shapes = [getattr(o, "shape", None) for o in outputs]
            raise RuntimeError(
                f"unsupported RKNN outputs: {len(outputs)} tensors {shapes} with box_scale={self.box_scale}. Supported: one "
                f"(1, 4+nc, N) tensor (card box_coords normalized|pixels), or an even number of split box/cls tensors "
                f"with card box_coords='pixels' (scripts/export_split.py). A DFL/raw head is rejected, see KNOWN_ISSUES.md #5."
            )
        t3 = time.perf_counter()
        self.last_timing = {
            "pre_ms": (t1 - t0) * 1000.0,
            "infer_ms": (t2 - t1) * 1000.0,
            "post_ms": (t3 - t2) * 1000.0,
        }
        return dets

    @staticmethod
    def _resolve_box_scale(card, weights) -> float:
        """Factor that brings the model's box channels back to letterbox pixels.

        Ultralytics' RKNN INT8 export divides box coordinates by the input
        size; FP16 exports do not. Getting this wrong is silent: an INT8
        model decoded as pixels still reports correct classes and scores,
        but every box is ~640x too small (KNOWN_ISSUES.md #24). A card that
        predates `box_coords` is refused for INT8 rather than guessed.
        """
        if card.box_coords == "normalized":
            return float(card.imgsz)
        if card.box_coords == "pixels":
            return 1.0
        if card.box_coords is None:
            if card.precision == "int8":
                raise ValueError(
                    f"model card for {weights} is INT8 but has no `box_coords` field. "
                    f"Ultralytics' RKNN INT8 export normalises box coordinates by the "
                    f"input size, so decoding them as pixels gives boxes that match "
                    f"nothing (KNOWN_ISSUES.md #24). Re-export with "
                    f"scripts/export_rknn.py, which records box_coords."
                )
            return 1.0
        raise ValueError(
            f"model card for {weights} has box_coords={card.box_coords!r}; "
            f"expected 'pixels' or 'normalized'"
        )

    def canary(self, images: list[np.ndarray]) -> int:
        """Total detections across `images` -- a startup smoke check.

        Call this once after building the backend, on a handful of known
        frames, before trusting the model for a full bench run. Zero here
        on frames known to contain the target class is the cheapest
        possible early warning for KNOWN_ISSUES.md #3.
        """
        return sum(len(self(img)) for img in images)

    def close(self) -> None:
        if getattr(self, "rknn", None) is not None:
            self.rknn.release()


_BACKENDS: dict[str, type[Backend]] = {
    "stub": StubBackend,
    "ultralytics": UltralyticsBackend,
    "torch": UltralyticsBackend,
    "onnx": ONNXBackend,
    "ncnn": NCNNBackend,
    "tflite": TFLiteBackend,
    "litert": TFLiteBackend,
    "rknn": RKNNBackend,
    "npu": RKNNBackend,
}


def build_backend(spec: dict[str, Any], names: Sequence[str]) -> Backend:
    """Instantiate a backend from a config block."""
    kind = str(spec.get("backend", "stub")).lower()
    if kind not in _BACKENDS:
        raise ValueError(f"unknown backend {kind!r}; known: {sorted(_BACKENDS)}")
    kw = {k: v for k, v in spec.items() if k not in {"backend", "imgsz", "conf", "enabled"}}
    weights = kw.get("weights")
    if kind != "stub" and not weights:
        raise ValueError(f"backend {kind!r} needs `weights` in its config block")
    if kind == "stub":
        kw.pop("weights", None)
    return _BACKENDS[kind](
        names=names,
        imgsz=spec.get("imgsz", 320),
        conf=spec.get("conf", 0.4),
        **kw,
    )
