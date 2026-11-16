# Known issues — bugs Coder/Tester/Reviewer must not rediscover the hard way

This file exists so a week is not spent re-finding a bug that upstream (or
we, in this repo) already hit once. Each entry: Triệu chứng (symptom) /
Nguyên nhân (root cause) / Biện pháp trong repo (mitigation here) / File
liên quan (related files). New entries go at the end, same format.

---

## 1. ultralytics#23340 — YOLO26 RKNN INT8 export segfaults

**Triệu chứng:** `model.export(format="rknn", int8=True)` on a YOLO26
checkpoint crashes the Python process (segfault) during conversion or on
first inference, with no Python traceback to act on.

**Nguyên nhân:** an interaction between YOLO26's end-to-end head and the
RKNN INT8 quantisation path in early ultralytics releases; fixed upstream
by PR ultralytics/ultralytics#23312.

**Biện pháp trong repo:** do not pin a minimum `ultralytics` version to
"solve" this — we do not know exactly which release first contains
#23312, and hardcoding a guess would give false confidence. Instead:
`scripts/export/export_rknn.py` prints the installed `ultralytics.__version__`
and explicitly names issues #23340/#23753 in its output, and states that
the real gate is Phase 1's functional parity check
(`scripts/export/validate_conversion.py`), not a version comparison. If export
segfaults, that is the signal to check for an ultralytics update, not to
edit a version pin.

**File liên quan:** `scripts/export/export_rknn.py`, `docs/rknn-pipeline.md`.

---

## 2. ultralytics#23753 — `end2end=True` segfaults at inference on RK3588

**Triệu chứng:** a model exported with YOLO26's default end-to-end
(NMS-free) head loads fine on the RKNN runtime but segfaults the moment
`rknn.inference()` is called on-device.

**Nguyên nhân:** the end-to-end head's internal NMS/decode ops are not
representable in the RKNN op set the way `rknn-toolkit2` lowers them for
this architecture; the runtime crashes rather than falling back to CPU.

**Biện pháp trong repo:** `end2end=False` is mandatory for every RKNN
export. `scripts/export/export_rknn.py` forces it and refuses `--end2end`
unless the caller also passes `--i-know-this-segfaults`. `ModelCard`
records `end2end`, and `RKNNBackend.__init__` (`backends.py`) raises
*before* touching the NPU if a loaded model's card says `end2end=True` —
segfaults must be caught on the laptop reading a JSON sidecar, not on the
board.

**File liên quan:** `src/labsafe/models/backends.py`,
`src/labsafe/models/modelcard.py`, `scripts/export/export_rknn.py`.

---

## 3. ultralytics#23753 — INT8 RKNN convert succeeds, model detects nothing

**Triệu chứng:** `.rknn` INT8 conversion returns exit code 0, the model
loads on-device, inference runs at a normal FPS, and every frame reports
zero detections. Nothing errors. A dashboard or FPS-only report looks
identical to a healthy run.

**Nguyên nhân:** open upstream issue at time of writing — INT8
quantisation of the YOLO26 head can collapse output scores below any
usable threshold on RK3588. Root cause not fully isolated upstream yet.

**Biện pháp trong repo:** this is the reason Phase 1
(`src/labsafe/models/parity.py`, `scripts/export/validate_conversion.py`)
exists and is the highest-priority deliverable in this plan. Every
conversion must be run through `validate_conversion.py` before its
numbers go into any report. The specific failure mode has its own
machine-readable code, `zero_detection_collapse`, so it never gets
silently absorbed into a generic "low recall" bucket.
`RouterStats.liveness_warning()` (`src/labsafe/pipeline/router.py`) is a
second, independent tripwire at bench time: 100+ calls to stage A with
zero detections raises a warning that names this issue.

**File liên quan:** `src/labsafe/models/parity.py`,
`scripts/export/validate_conversion.py`, `src/labsafe/pipeline/router.py`.

---

## 4. Double normalisation with RKNN → zero detections, no error

**Triệu chứng:** an RKNN model converted correctly still detects nothing,
even though the same ONNX model detects normally.

**Nguyên nhân:** RKNN bakes `mean_values`/`std_values` into the model at
convert time, so the runtime expects raw `uint8` input in `[0, 255]`. If
the caller also normalises the frame to `float32` in `[0, 1]` before
calling `rknn.inference()`, the input is scaled twice. This does not
raise — floating point math on out-of-range values just produces useless
scores below any sane confidence threshold.

**Biện pháp trong repo:** `letterbox(..., normalize=False)` in
`src/labsafe/models/postprocess.py` returns raw `uint8` NHWC, and
`RKNNBackend._infer` always calls it that way. `ModelCard.input_dtype`
records what the model actually expects; `RKNNBackend.__init__` asserts
`card.input_dtype == "uint8"` and raises with an explanation if not,
instead of letting a mismatched card through.

**File liên quan:** `src/labsafe/models/postprocess.py`,
`src/labsafe/models/backends.py`, `src/labsafe/models/modelcard.py`.

---

## 5. `yolo_decode()` does not accept `rknn_model_zoo`-style raw-branch output

**Triệu chứng:** a model converted following Rockchip's `rknn_model_zoo`
YOLOv8 example (3 or 9 raw regression/classification branches per stride,
pre-DFL) is fed into `yolo_decode()`. The function does not crash — it
squeezes and transposes the raw array as if it were already
DFL-decoded, and produces plausible-looking but meaningless boxes.

**Nguyên nhân:** `yolo_decode()` was written for the shape ultralytics
produces when exporting with `end2end=False`: `(1, 4+nc, N)` or
`(1, N, 4+nc)`, already through DFL and already decoded to per-pixel
`cxcywh` in letterbox-image coordinates. It has no way to distinguish
"wrong shape, still technically 2-D after squeeze" from "right shape".

**Biện pháp trong repo:** the official conversion path in this repo is
Ultralytics' own RKNN export (`end2end=False`), never the
`rknn_model_zoo` cut-before-head approach — `docs/optimization.md` §5 and
`docs/rknn-pipeline.md` say so explicitly. `yolo_decode()` also gained a
shape guard: it rejects non-2-D input after squeeze, rejects fewer than 5
attribute channels, and accepts an optional `expect_nc` that raises if
the channel count does not match the configured class list — instead of
silently `argmax`-ing over the wrong number of classes.

**File liên quan:** `src/labsafe/models/postprocess.py`,
`docs/optimization.md`, `docs/rknn-pipeline.md`.

---

## 6. Bench loop spins to the clock after a video source ends — fabricated FPS

**Triệu chứng:** a `--bench` run against a video file shorter than
`--duration` reports a wildly high `fps_sustained`. Reproduced: a 2-second
clip with `--duration 6` gave `fps_sustained: 723174` over `iterations_total:
4383050`, while `router.frames` was 60. No flag was raised: `result_valid`
stayed true, because the model was alive — it was the source that had died.

**Nguyên nhân:** `BenchRunner.run()` looped on `while time.time() < end`
only. `app.py`'s `step()` set `state["stop"]` when a file source was
exhausted, but the runner never read it, so every remaining iteration was a
no-op that still counted. The reported rate measured the loop, not the model.

**Biện pháp trong repo:** `step()` now returns `False` when the source is
exhausted and `BenchRunner.run()` breaks on it, recording `stop_reason`,
`ended_early` and `short_by_s` in `metrics.json` (returning `None` keeps the
old contract for callers that never end). A second, independent guard —
`app._spin_warning()` — compares the runner's `iterations_total` against the
router's `frames` and sets `result_valid: false` when they diverge by more
than 2x; the two counters come from different places and already sat side by
side in the file, with nothing reading them against each other.

**File liên quan:** `src/labsafe/bench/runner.py`, `src/labsafe/app.py`,
`tests/test_bench_runner_stop.py`.

---

## 7. Ultralytics export returns a DIRECTORY for packaged formats

**Triệu chứng:** `scripts/export/export_rknn.py` printed a success line and a
destination path, wrote a model card next to it, printed the parity-gate
command — and no `.rknn` file existed at that path. Exit code 0.

**Nguyên nhân:** the script moved the artefact with
`Path(exported).replace(dest) if Path(exported).is_file() else None`.
Ultralytics returns a directory (`*_rknn_model/`) for packaged formats, so
the `else None` branch ran and did nothing, silently. The repo already knew
this: `scripts/export/export_models.py` handles the directory case with `is_dir()`.

**Biện pháp trong repo:** `export_rknn._collect_export()` accepts a file or a
directory, unwraps a single `*.rknn` from the directory, and raises with a
specific message when the export produced nothing, produced a directory with
no model, or produced several models. A conversion step must never report a
destination it did not create.

**File liên quan:** `scripts/export/export_rknn.py`, `tests/test_export_rknn_cli.py`.

---

## 8. RKNN fixes the input shape at convert time — a resolution sweep needs N exports

**Triệu chứng:** configs `N3a/N3b/N3c` declared `imgsz: 320/416/512` while all
three pointed at `runs/export/yolo26n_fp16.rknn`, the model N1 exported at
640. On the board the model-card guard raises mid-matrix — after N0/N1/N2
have already run for hours — and `bench_matrix_npu.sh` (`set -euo pipefail`)
aborts before the report step.

**Nguyên nhân:** two mistakes compounding. The export filename carried no
`imgsz`, so four resolutions would overwrite one file; and the sweep configs
assumed one model could serve every resolution, which RKNN does not allow.

**Biện pháp trong repo:** exports are named `{stem}_{precision}_{imgsz}.rknn`;
each `N3*` config points at its own model and carries a header comment with
the export command; `bench_matrix_npu.sh` skips a sweep point whose model is
missing instead of aborting the matrix; and
`tests/test_configs.py` asserts that every RKNN config's declared `imgsz`
matches the one encoded in its weights filename, and that the sweep points
use distinct models.

**File liên quan:** `scripts/export/export_rknn.py`, `configs/exp/N3*.yaml`,
`scripts/bench/bench_matrix_npu.sh`, `tests/test_configs.py`.

---

## 9. Renaming an artefact without renaming its consumers — the gate checks a different file

**Triệu chứng:** after `#8` gave exports an `imgsz` in their filename, the
`N*` configs were updated but `bench_matrix_npu.sh`, `configs/exp/N0*.yaml`,
`docs/rknn-pipeline.md` and the `validate_conversion.py` docstring still
named the old paths. `run N0` dies immediately under `set -euo pipefail`.

**Nguyên nhân:** a half-rename. The producer's naming rule lived in one
script and every consumer had the name retyped by hand, so there was no
single thing to change and nothing that disagreed loudly.

The crash is the harmless outcome. The dangerous one is both files
existing: someone follows the stale doc, produces `model_fp16.rknn`, the
parity gate PASSES on it, and `run N1` then benchmarks
`model_fp16_640.rknn`. **The gate validates model A while the report
measures model B** — a gate that checks a different artefact than the one
measured is worse than no gate, because it is believed.

**Biện pháp trong repo:** the rule now lives in
`src/labsafe/models/naming.py` (`export_name`, `onnx_reference_name`) and is
imported by the exporter. `bench_matrix_npu.sh` derives every path from
`STEM`/`BASE_IMGSZ` instead of retyping them. `tests/test_naming.py` reads
the real configs and scripts and checks each `runs/export/...` path against
those functions — never against a regex copied into the test, which would
agree with a stale consumer just as happily as with a fresh one. It also
pins that the documented export command matches the real CLI (`weights` is
positional; the printed `--weights` form failed with "unrecognized
arguments").

**File liên quan:** `src/labsafe/models/naming.py`, `scripts/export/export_rknn.py`,
`scripts/bench/bench_matrix_npu.sh`, `configs/exp/N*.yaml`,
`docs/rknn-pipeline.md`, `tests/test_naming.py`.

---

## 10. `rknn` is a namespace package with no `__version__` — reading it directly reports "not installed" even when it is

**Triệu chứng:** `_toolkit_versions()` in `scripts/export/export_rknn.py` returned
`(None, None)` for rknn-toolkit2, and would have written every
`ModelCard.toolkit_version` as empty, even on a machine with a correctly
installed, importable toolkit (`from rknn.api import RKNN` worked fine).

**Nguyên nhân:** the code did `import rknn; getattr(rknn, "__version__",
None)`. `rknn-toolkit2` installs as the `rknn` **namespace package** — the
real content lives in `rknn.api`, and the top-level `rknn` module has no
`__version__` attribute, no `__file__`, nothing. `getattr(..., None)`
never raises, so the absence looked exactly like a normal successful
lookup returning "not set." This was unverifiable in this repo's own dev
environment (neither `ultralytics` nor `rknn-toolkit2` were installed) and
only surfaced once the real package was installed for the first time,
2026-09-26, on the x86 training machine — see `docs/board-setup.md` and
`docs/rknn-pipeline.md` for why that install happens there, not on the
board.

**Biện pháp trong repo:** `_toolkit_versions()` now reads
`importlib.metadata.version("rknn-toolkit2")` / `("ultralytics")` — the
installed-**distribution** name, not the imported **module** name.
`tests/test_export_rknn_cli.py` pins this by asserting the function never
imports `rknn` or `ultralytics` at all, only reads package metadata; a
positive control (reverting the fix) confirmed the test fails against the
old code.

**Nguyên tắc chung:** `getattr(module, "__version__", None)` is not a safe
"is this installed" check for any namespace package, and Python has no
warning for silently returning the default. When a library's own docs or
`pip show` don't loudly say "this exposes `__version__`", prefer
`importlib.metadata.version()` keyed by the PyPI distribution name.

**File liên quan:** `scripts/export/export_rknn.py`, `tests/test_export_rknn_cli.py`.

---

## 11. `Path.replace()` fails moving an export across filesystems — the normal case, not the edge case

**Triệu chứng:** `scripts/export/export_rknn.py` crashed with `OSError: [Errno 18]
Invalid cross-device link` right after `rknn-toolkit2` finished building a
real model (~11s of real conversion work wasted per attempt).

**Nguyên nhân:** `_collect_export()` used `src.replace(dest)`
(`Path.replace`), which is a bare `rename()` and only works within one
filesystem. Ultralytics writes its export beside the **source checkpoint**
(`yolo11n.pt` → `./yolo11n_rknn_model/`), completely ignoring this
script's `--out`. So collecting the result across a mount boundary — `--out`
on a different disk, a tmpfs, an external SSD, a mounted network share —
is not a rare edge case here, it is the default shape of the operation
whenever the checkpoint and `--out` are not on the same filesystem.
Reproduced for real: converting with `--out /tmp/...` (tmpfs) from a repo
on a regular disk, 2026-09-26.

**Biện pháp trong repo:** `_collect_export()` uses `shutil.move()`, which
falls back to copy+delete when an atomic rename is impossible.
`tests/test_export_rknn_cli.py::test_collect_export_survives_a_cross_device_move`
monkeypatches `Path.replace` to raise the exact `OSError(18, ...)` and
confirms the move still succeeds (positive-controlled: reverting to
`Path.replace()` makes the test fail).

**Nguyên tắc chung:** `Path.replace()`/`os.rename()` silently assume same
filesystem and are correct on the same filesystem, which is exactly the
condition a local dev machine and a small test satisfy and a real
multi-disk or containerised deployment does not. Prefer `shutil.move()`
whenever the source and destination directories are not guaranteed to be
on the same device — which for any script accepting an `--out`/`--dest`
argument from the user is essentially always.

**File liên quan:** `scripts/export/export_rknn.py`, `tests/test_export_rknn_cli.py`.

---

## 12. Structured pruning ran, reported "0.0% removed", and exited 0 — the dependency tracer silently saw almost nothing

**Triệu chứng:** `scripts/train/prune_structured.py` against a real `yolo11n.pt`
(torch 2.14.0+cu130, torch_pruning 1.6.1, x86, 2026-09-26) printed a
plausible-looking report —

```
before pruning: 2.62 M params, 6.55 GFLOPs
protected 25 conv layers in the detection head
pruning 30% of channels over 2 steps (16.3% per step), fine-tuning 0 epochs between steps
[step 1/2] 2.62 M params (100% of original), 6.55 GFLOPs (100%)
[step 2/2] 2.62 M params (100% of original), 6.55 GFLOPs (100%)
params : 2.62 M -> 2.62 M (0.0% removed)
```

— with no exception, no warning, and exit code 0. Nothing in the output
looks wrong at a glance; the table format is identical to a real run that
pruned nothing because `--ratio` was set too low, which is a legitimate
(if unlikely) outcome, not an error. This is the same shape as the RKNN
INT8 zero-detection collapse (#3): the tool ran, produced numbers, and the
numbers were a lie.

**Nguyên nhân:** `torch_pruning.dependency.DependencyGraph.build_dependency()`
traces the model with forward hooks and needs to walk every module actually
invoked. Debugged directly (not guessed): `pruner.DG.module2node` had
**1 entry**, against 88 `Conv2d` layers in the real model. Root cause
traced one level further: `model(example)` in eval mode returns
`(Tensor, dict)`, not a plain `Tensor` — YOLO's `Detect` head packs extra
metadata into the second element unless its `.export` flag is set (the
same flag ultralytics itself sets before ONNX/NCNN/RKNN export). Setting
`model.model[-1].export = True` fixes the `(Tensor, dict)` output shape,
but did **not** fix the trace count (still 1 node) — so a second,
unresolved cause remains. The most likely remaining candidate, not yet
confirmed: `torch_pruning==1.6.1` (the latest release on PyPI at the time)
against `torch==2.14.0+cu130` is a version combination nobody has
necessarily tested — ultralytics' own isolated build environment for RKNN
pins `torch==2.4` specifically (`exporter.py`'s `isolated-rknn` entry),
several major torch releases behind what this repo's venv resolved to
with no upper bound.

**Chưa xác minh / KHÔNG được đoán tiếp:** whether pinning `torch==2.4`
actually fixes the trace count. Not attempted in this session: doing so
in the same venv that now has a verified-working RKNN export pipeline
risked breaking that (rknn-toolkit2's own dependency resolution was
against the currently-installed torch), and pruning is explicitly a
later-phase optimisation in this project's own priority order
(`docs/optimization.md`: INT8/NPU/pipeline first, pruning/KD only "if
time permits"). The cheapest next diagnostic step is a **separate** venv
with `torch==2.4` + `torch_pruning==1.6.1` + a plain (non-YOLO) CNN, to
isolate "torch version" from "YOLO's forward pass" as the variable.

**Biện pháp trong repo:** `scripts/train/prune_structured.py` now refuses to
report a completed step in two independent ways, so a future
compatibility break (fixed torch pin, upgraded library, different model)
cannot regress silently:
1. Right after each `MagnitudePruner` is constructed, `len(pruner.DG.module2node)`
   is compared against the count of prunable `Conv2d` layers; fewer traced
   modules than prunable layers aborts immediately with the exact numbers
   and this issue's diagnosis.
2. After every `pruner.step()`, real parameter count before/after is
   compared; `p_after >= p_before` aborts. This is deliberately
   independent of guard 1 — it also catches a future silent-no-op whose
   cause is not "the tracer barely ran" (e.g. a pruning-ratio-rounds-to-
   zero edge case, a different library bug entirely).
Both guards fire with exit code 1 and a message that includes the actual
numbers, not just "something went wrong" — verified for real: the fixed
script now exits 1 with a clear diagnosis on the exact model that
previously exited 0 with a false "0.0% removed".

**Chưa có test hồi quy:** `scripts/train/prune_structured.py` is a flat script
(module-level code executed on import, no `validate_args()`-style
separation like `scripts/export/export_rknn.py` has). Unit-testing the two new
guards would require either refactoring the script into importable
functions (a larger change, out of scope for this fix) or a slow
subprocess-based integration test that itself needs `torch`+`ultralytics`+
`torch_pruning` installed, which this repo's default dev environment does
not have. The guards are simple, self-contained comparisons
(`traced < total_prunable`, `p_after >= p_before`); the risk they carry is
lower than the risk they close.

**File liên quan:** `scripts/train/prune_structured.py`.

## 13. `_count_calib_images()` resolved a relative `path:` against the wrong directory — false "0 images", real export would have worked

**Triệu chứng, xác minh thật, 2026-09-26:** chạy INT8 export thật cho
**YOLO26n** (không phải model thế thân) với `--data coco128.yaml` (dataset
chuẩn của ultralytics, tự tải qua `check_det_dataset('coco128.yaml')`,
128 ảnh thật tại `/mnt/big/datasets/coco128/images/train2017`):

```
'val' split of '.../ultralytics/cfg/datasets/coco128.yaml' has 0 image(s)
at '.../ultralytics/cfg/datasets/coco128/images/train2017', need at least
50 for a usable INT8 calibration set.
```

Script từ chối chạy — nhưng dataset có thật 128 ảnh, ở đúng chỗ ultralytics
sẽ tìm khi thật sự export. Đây là **false negative do chính script tự viết
lại (reimplement) sai một phần logic path resolution của ultralytics**,
không phải lỗi của dataset hay của ultralytics.

**Nguyên nhân, đọc trực tiếp source `ultralytics.data.utils.check_det_dataset`:**

```python
path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)
if not path.exists() and not path.is_absolute():
    path = (DATASETS_DIR / path).resolve()
```

Tức là: một `path:` tương đối được thử **y nguyên theo cwd trước**, và
**chỉ fallback về `DATASETS_DIR`** (thư mục dataset toàn cục của
ultralytics, cấu hình qua `ultralytics.utils.SETTINGS`) khi path đó không
tồn tại. `coco128.yaml` khai `path: coco128` — một cái tên tương đối trần,
đúng convention chuẩn của MỌI dataset yaml có sẵn của ultralytics
(`coco.yaml`, `coco128.yaml`, `coco8.yaml`, ...) và rất có thể là convention
mà `data/processed/*/data.yaml` của chính project này cũng sẽ theo, vì đó
là cách `yolo train data=...` mặc định hoạt động.

`_count_calib_images()` (thêm trong lần sửa lỗi #1-4 hôm nay) lại resolve
`path:` tương đối **luôn theo thư mục chứa file YAML** —
`(p.parent / root).resolve()` — không bao giờ thử theo cwd, không bao giờ
fallback `DATASETS_DIR`. Với `coco128.yaml` nằm trong
`site-packages/ultralytics/cfg/datasets/`, điều này resolve ra
`.../cfg/datasets/coco128/images/train2017` — một thư mục không tồn tại —
thay vì `/mnt/big/datasets/coco128/images/train2017` (nơi ảnh thật nằm).

Đây là bug **trong chính reimplementation của project**, không phải hiểu
sai API ultralytics như #1-4 — cùng một bài học ("đừng đoán lại logic của
thư viện, đọc source hoặc verify thật") áp dụng ngược lại cho code của
chính mình.

**Biện pháp:** thêm `_resolve_dataset_root(cfg, yaml_path)` tái tạo đúng
thứ tự thử của `check_det_dataset` (absolute → giữ nguyên; tồn tại theo cwd
→ giữ nguyên; ngược lại → fallback `ultralytics.utils.DATASETS_DIR`), và
`_count_calib_images()` gọi hàm này thay vì tự resolve theo `p.parent`.
Import `ultralytics.utils.DATASETS_DIR` chỉ xảy ra trong nhánh fallback,
bọc `try/except ImportError`, để giữ đúng ràng buộc thiết kế đã có từ đầu
(`validate_args()` phải test được khi chưa cài `ultralytics`).

**Verify thật sau khi sửa:**
```
count: 128 checked: /mnt/big/datasets/coco128/images/train2017
```
và chạy lại đúng lệnh INT8 export YOLO26n ở trên: exit 0, ra
`yolo26n_int8_640.rknn` (4.6 MB) thật, có `.card.json` đi kèm.
150 test vẫn xanh sau khi sửa (path tuyệt đối dùng trong
`tests/test_export_rknn_cli.py::_make_data_yaml` không bị ảnh hưởng).

**Chưa xác minh:** exit code 0 + file `.rknn` ra đời **không** chứng minh
model detect đúng — đó vẫn là việc của `validate_conversion.py` (cổng
parity), và nhánh `--candidate-backend rknn` của nó **chỉ chạy được trên
board** (aarch64). Rủi ro #23753 (INT8 RKNN detect ra 0 vật thể trên
RK3588) chưa được loại trừ hay xác nhận cho YOLO26n cụ thể — chỉ mới xác
nhận bước *convert* không lỗi, chưa xác nhận bước *inference* đúng.

**File liên quan:** `scripts/export/export_rknn.py` (`_resolve_dataset_root`,
`_count_calib_images`).

## 14. Trying TFLite export silently downgraded `torch` in the SAME venv the verified RKNN pipeline depends on

**Mức độ:** cao hơn mọi bug khác trong file này về phạm vi ảnh hưởng — đây
không phải lỗi trong một script, mà là một hành động tưởng như vô hại (thử
export TFLite) **âm thầm thay đổi môi trường dùng chung** cho toàn bộ
pipeline RKNN đã xác minh trước đó trong cùng phiên làm việc.

**Triệu chứng, xác minh thật, 2026-09-26:** chạy
`scripts/export/export_models.py yolo26n.pt --formats tflite --int8 --data
coco128.yaml` trong đúng venv (`.venv/`) đã dùng để xác minh toàn bộ
pipeline RKNN (bug #1-#13 ở trên). Ultralytics tự phát hiện thiếu
`litert-torch`/`ai-edge-litert` (backend TFLite mới của nó, thay cho
`tensorflow` cổ điển) và tự chạy `pip install` (`AutoUpdate`) — **49 gói,
366 giây, tải 502 MB cho riêng `torch`**. Sau đó:

```
Trước:  torch 2.14.0+cu130   (dùng để xác minh export_rknn.py, #1-#13)
Sau:    torch 2.13.0+cu130   (+ torchao 0.18.0 mới xuất hiện)
```

Không có cảnh báo, không hỏi xác nhận, không có gì trong output nói rõ
"sắp thay đổi torch của bạn". Việc downgrade này xảy ra hoàn toàn như một
side-effect của dependency resolution khi `pip` cài `litert-torch`.

Bản thân lần chạy ĐÓ (cùng process vừa kích hoạt AutoUpdate) còn **crash**
ngay sau khi cài xong:

```
File ".../litert_torch/backend/composite/mark_tensor.py", line 59
File ".../torch/_dynamo/... torch/distributed/tensor/_ops/_math_ops.py", line 1462
AttributeError: '_OpNamespace' 'aten' object has no attribute 'cholesky'
```

**Đã sửa chẩn đoán sau khi kiểm tra thêm** (chẩn đoán ban đầu — "xung đột
vĩnh viễn giữa litert-torch và torch 2.13" — SAI, đã tự phát hiện và sửa
lại ở đây thay vì để lại thông tin sai): chạy lại **đúng lệnh TFLite y hệt
trong một process Python mới** (không có AutoUpdate nào chạy lần này, vì
gói đã có sẵn từ lần trước) — **thành công**, ra `yolo26n_int8.tflite`
(2.9 MB), exit 0. Vậy `litert-torch 0.9.4` với `torch 2.13.0+cu130` **tương
thích với nhau khi cả hai đã ổn định trên đĩa**. Nguyên nhân crash thật sự:
`pip install` đã **thay thế file `torch` trên đĩa NGAY TRONG LÚC** tiến
trình Python đang chạy đã import một phần `torch` (bản 2.14 cũ, còn nằm
trong `sys.modules`); khi `litert_torch` sau đó import tiếp
`torch._dynamo` (một submodule chưa từng được nạp), Python đọc file MỚI
(2.13) từ đĩa trong khi phần lõi `torch` (extension C, `aten` namespace)
vẫn là bản 2.14 đã nạp sẵn trong bộ nhớ của tiến trình đó — một trạng thái
"nửa 2.14 nửa 2.13" không nhất quán chỉ tồn tại trong CHÍNH tiến trình vừa
kích hoạt AutoUpdate. Tiến trình mới, khởi động sau khi việc cài đặt đã
hoàn tất, không gặp lại vấn đề này.

**Đã kiểm tra ngay việc downgrade này có phá pipeline RKNN không** (vì đây
là câu hỏi quan trọng nhất): chạy lại `export_rknn.py --precision fp16`
cho `yolo26n.pt` với `torch 2.13.0` mới — **vẫn exit 0, vẫn ra `.rknn` thật
(7.4 MB)**; chạy lại toàn bộ 150 test — vẫn xanh. Không có bằng chứng
pipeline RKNN bị hỏng bởi lần downgrade này, nhưng đó là may mắn của lần
này, không phải điều được đảm bảo cho lần sau.

**Bài học, quan trọng hơn bug cụ thể này:** cái nguy hiểm không phải
"TFLite không tương thích với torch 2.13" (nó tương thích) mà là:
(a) `ultralytics`' AutoUpdate có thể **âm thầm đổi phiên bản `torch` của cả
venv** — không hỏi, không cảnh báo — chỉ vì thử một `--formats` mới lần
đầu, kể cả khi format đó không phải đích cạnh tranh chính (RK3588S/RKNN);
và (b) **lần chạy đúng lúc AutoUpdate xảy ra sẽ crash** vì tự mâu thuẫn
phiên bản trong chính tiến trình đó, khiến người chạy dễ kết luận nhầm
"format này bị lỗi/không tương thích" trong khi thực ra chỉ cần chạy lại
là được — một chẩn đoán sai (giống hệt điều vừa xảy ra ở đây) có thể khiến
tốn hàng giờ đi sửa một thứ không hỏng.

**Biện pháp:**
- KHÔNG downgrade lại `torch` — đã xác minh 2.13.0 vẫn chạy tốt với RKNN
  (export lại `yolo26n.pt` fp16 thật, exit 0) và với TFLite (chạy lại ở
  tiến trình mới, exit 0); không có lý do quay lại 2.14.0.
- **Nếu một lần export vừa kích hoạt AutoUpdate rồi crash ngay sau đó**:
  đừng vội kết luận hai thư viện xung đột — chạy lại y hệt lệnh đó một lần
  nữa (tiến trình mới) trước khi kết luận bất cứ điều gì. Đây chính xác là
  điều đã xảy ra ở đây.
- Sau bất kỳ lần nào AutoUpdate âm thầm đổi `torch`/gói lõi khác, **re-run
  `pytest` + một lần export RKNN thật** trước khi tin số liệu RKNN tiếp
  theo — đã làm điều này ở đây (150 test xanh, RKNN fp16 export lại thành
  công) nên pipeline RKNN vẫn được coi là đáng tin sau sự cố này.
- `scripts/export/export_models.py` giờ bọc try/except quanh từng format
  (xem code) — một format lỗi không còn làm mất khả năng thấy kết quả các
  format khác trong cùng lần chạy, và script exit khác 0 nếu có format
  lỗi thay vì im lặng dừng giữa chừng.
- ONNX và NCNN không kích hoạt AutoUpdate nặng này (không cần
  `tensorflow`/`litert-torch`), nên rủi ro này chỉ áp dụng cho TFLite.

**File liên quan:** `scripts/export/export_models.py`; không có mã nguồn nào
trong repo trực tiếp gây ra bug này — nó nằm ở hành vi AutoUpdate của
`ultralytics` + xung đột `litert-torch`/`torch` build cụ thể của venv này.

## 15. `distill_pseudolabel.py --classes` wrote placeholder names (`class_0`) into `data.yaml`, discarding the real class name it already had

**Triệu chứng, xác minh thật, 2026-09-26:** chạy full pipeline thật (2 video
tổng hợp, teacher `yolo26n.pt`, `--classes 0`) — script chạy sạch, exit 0,
ra dataset + label file đúng định dạng YOLO. Nhưng `data.yaml` sinh ra ghi:

```yaml
nc: 1
names: ['class_0']
```

trong khi lớp thật đang giữ lại là "person" (id 0 của COCO) — và teacher
model đã biết tên thật này (`model.names[0] == "person"`), script chỉ đơn
giản không dùng nó khi `--classes` được truyền.

**Nguyên nhân:**
```python
names = [f"class_{i}" for i in range(len(keep_ids))] if keep_ids else list(model.names.values())
```
Nhánh không lọc (`else`) dùng đúng tên thật từ `model.names`. Nhánh có lọc
(`--classes`) lại bỏ qua `model.names` hoàn toàn và tự sinh tên giả theo
thứ tự — có vẻ như một chỗ để-làm-sau bị bỏ quên, không phải cố ý.

**Ảnh hưởng:** không crash, không cảnh báo — nhưng bất kỳ lần train/verify
nào dùng dataset lọc theo `--classes` sau này sẽ thấy "class_0" thay vì
"person" trong mọi biểu đồ, log, và số liệu report — đúng dạng "chạy được,
số liệu vẫn ra, nhưng sai một cách âm thầm" mà cả file này tồn tại để bắt.

**Biện pháp:**
```python
names = [model.names[cid] for cid in sorted(keep_ids)] if keep_ids else list(model.names.values())
```
`sorted(keep_ids)` khớp chính xác với thứ tự remap id đã dùng ở chỗ khác
trong script (`new_id = sorted(keep_ids).index(cid)`), nên tên và id luôn
đồng bộ. Verify thật: `--classes 0` → `names: ['person']`; `--classes
"5,0"` (thứ tự ngược, để kiểm tra không phụ thuộc thứ tự nhập) →
`names: ['person', 'bus']`, và nhãn thật của khung hình có xe bus in ra
đúng id `1` — khớp đúng vị trí trong danh sách `names`.

**Chưa có test hồi quy:** cùng lý do như #12 — đây là script flat, không
có hàm tách riêng để test mà không cần một model + video thật. Đã verify
thủ công bằng chạy thật với 2 video tổng hợp + `yolo26n.pt`, không thêm
test tự động.

**File liên quan:** `scripts/train/distill_pseudolabel.py`.

## 16. YOLO26 luôn có kiến trúc one2one built-in — giả định `v8DetectionLoss` phẳng sai ngay ở batch đầu tiên

**Bối cảnh:** thêm KD thật kiểu Hinton (D1) — `src/labsafe/models/kd_loss.py`,
`scripts/train/distill_kd.py` — theo yêu cầu trả lời câu hỏi "đã áp dụng bài báo
distillation chưa" (trước đó `distill_pseudolabel.py` chỉ làm pseudo-label,
không phải KD thật qua loss). Bug này tìm được khi verify KD thật lần đầu.

**Triệu chứng, xác minh thật, 2026-09-26:** bản nháp đầu của `KDDetectionLoss`
kế thừa thẳng `v8DetectionLoss`, giả định `preds` (output thật của model lúc
training) là dict phẳng `{"boxes","scores","feats"}` — đúng như code đọc
được trong `get_assigned_targets_and_loss`. Chạy training thật (teacher
`yolo26s.pt`, student `yolo26n.pt`, `coco128.yaml`, CPU) crash ngay batch
đầu tiên:
```
File ".../ultralytics/utils/loss.py", line 484, in loss
    batch_size = preds["boxes"].shape[0]
KeyError: 'boxes'
```

**Nguyên nhân, xác minh bằng in trực tiếp `preds` thật:** `preds` thực ra là
`{"one2many": {...}, "one2one": {...}}` — một dict LỒNG, không phẳng.
YOLO26 (khác YOLO11) có module `one2one_cv2` **built vào kiến trúc mặc
định**, độc lập với cờ `end2end=` lúc export (`end2end=False` chỉ ảnh hưởng
INFERENCE/NMS lúc export, không xóa nhánh one2one khỏi model lúc training).
`init_criterion()` của ultralytics tự chọn `E2ELoss` (không phải
`v8DetectionLoss`) bất cứ khi nào `one2one_cv2` tồn tại — nghĩa là **fine-tune
bình thường (không KD) một checkpoint YOLO26 mặc định cũng đã luôn dùng
`E2ELoss`** (hai loss v8 riêng cho nhánh one2many/one2one, trộn theo lịch
decay `o2m: 0.8→0.1`), một chi tiết không hề rõ ràng nếu chỉ đọc
`v8DetectionLoss` một mình.

**Biện pháp:** `KDDetectionLoss` giờ tự phát hiện `one2one_cv2` (đúng điều
kiện `init_criterion()` dùng), dùng `E2ELoss` nội bộ khi cần, và **chỉ áp
dụng KD lên nhánh one2many** — nhánh duy nhất project này thực sự export/
deploy (`export_rknn.py` luôn `end2end=False`; nhánh one2one là rủi ro
segfault đã ghi ở mục #2, không bao giờ lên board). Nhánh one2one vẫn train
bình thường với ground truth qua `E2ELoss` gốc, không bị KD đụng vào.
Cũng phải forward `update()` cho lịch decay o2m/o2o (`BaseTrainer` gọi
`criterion.update()` cuối mỗi epoch qua `hasattr` — thiếu nó thì lịch decay
âm thầm không bao giờ chạy, không lỗi, không cảnh báo).

**Verify thật sau khi sửa:** training thật chạy hết 1 epoch thật, exit 0,
`results.csv` có cột `train/kd_loss` với giá trị thật (0.01846, khác 0,
không NaN). Thêm 9 unit test thật (`tests/test_kd_loss.py`, dùng model
random-init từ yaml có sẵn của ultralytics — không cần tải checkpoint,
không cần mạng): xác nhận `alpha=0` khớp CHÍNH XÁC task loss thuần,
`alpha=1` khớp CHÍNH XÁC KD loss thuần, gradient thật chảy vào student mà
KHÔNG chạm vào teacher (`teacher.parameters()` có `.grad is None` sau
backward), nc lệch giữa teacher/student raise `RuntimeError` ngay thay vì
lệch trục âm thầm, và trường hợp YOLO26 one2one chạy đúng không crash +
`update()` không raise. 159 test xanh trong venv có torch (150 + 9 mới),
150 vẫn xanh + 1 skip đúng trong venv chuẩn không có torch.

**File liên quan:** `src/labsafe/models/kd_loss.py`, `scripts/train/distill_kd.py`,
`tests/test_kd_loss.py`.

## 17. mAP50 = 0.995 với Precision = 0.006 — train thật trên 10 ảnh internet, số liệu đẹp nhưng model vô dụng

**Bối cảnh:** để trả lời "train và chạy thử với ảnh/video lấy trên mạng",
đã build một dataset thật (không phải mock): 10 ảnh từ Wikimedia Commons
(giấy phép rõ ràng, tải trực tiếp qua `Special:FilePath`), 3 lớp
(`person`, `ppe`, `fire_smoke`). Bbox người dùng **detect thật** từ
`yolo26n.pt` (COCO, đã verify), bbox ppe/fire_smoke tự gán bằng mắt sau khi
xem từng ảnh qua tool đọc ảnh, verify lại bằng cách vẽ box lên ảnh và xem
lại. Train thật 60 epoch trên GPU (RTX 3050), `data/processed/lab_safety_demo/`.

**Triệu chứng, xác minh thật, 2026-09-28:** bảng validation cuối training in ra:
```
Class     Images  Instances  P        R    mAP50  mAP50-95
person         1          1  0.00621  1    0.995  0.995
ppe            1          1  0.00556  1    0.0663 0.0332
fire_smoke     1          1  0.00386  1    0.111  0.0276
```
`mAP50=0.995` cho lớp person — nhìn qua tưởng model học tốt. Nhưng chạy
`model.predict()` thật trên **2 ảnh hoàn toàn mới** (chưa từng thấy, tải
riêng sau khi train xong) ở `conf=0.1`, `conf=0.01`, thậm chí `conf=0.001`:
**0 detection ở tất cả ngưỡng hợp lý**; ở `conf=0.001` có ra 300 box
nhưng box tự tin nhất chỉ đạt confidence **0.021** — thấp hơn bất kỳ
ngưỡng hiển thị thực tế nào (0.25 mặc định, hay cả 0.1). So sánh trực
tiếp: model **gốc** (`yolo26n.pt`, chưa fine-tune) detect đúng cả 4 người
trong ảnh mới đó với confidence 0.46–0.84 — tức là **fine-tune đã phá mất
khả năng detect người vốn đã hoạt động tốt**, không phải "chưa học được gì".

**Nguyên nhân — không phải bug, là hệ quả thật của quá ít dữ liệu:**
`Precision=0.006` ở đúng epoch/checkpoint có `mAP50=0.995` giải thích toàn
bộ mâu thuẫn: **mAP là metric xếp hạng (ranking), không phải metric hiệu
chỉnh (calibration).** `mAP50` chỉ cần box đúng nằm ĐÂU ĐÓ trong danh sách
box được xếp hạng theo confidence — với validator dùng ngưỡng quét rất
thấp (mặc định ~0.001) để dựng đường cong precision-recall, model chỉ cần
gán cho box đúng một confidence **cao hơn nhiễu một chút** là đủ để
`mAP50` gần tuyệt đối, kể cả khi confidence tuyệt đối của box đó (0.006–
0.02) thấp đến mức không AI nào đặt threshold hiển thị ở đó. Với 8 ảnh
train / ~11 box person, model không có đủ tín hiệu để học confidence
*hiệu chỉnh đúng biên độ* — chỉ học được thứ hạng tương đối.

**Đây chính xác là dạng lỗi mà cả `docs/KNOWN_ISSUES.md` này tồn tại để bắt**:
số liệu (mAP) chạy ra, trông đẹp, và sai lệch hoàn toàn với hành vi thực tế
của model khi dùng thật (deploy với ngưỡng conf cố định, như
`export_rknn.py`/`app.py` đều làm). `docs/research-plan.md` §2.5 đã cảnh
báo trước một dạng liên quan (benchmark thiếu điều kiện khó làm tối ưu hoá
trông miễn phí) — đây là biến thể của cùng nguyên tắc: **không bao giờ tin
một con số aggregate (mAP) mà không kiểm tra hành vi ở ngưỡng vận hành
thật.**

**Không phải bug cần sửa trong code** — đây là hệ quả toán học đúng của
train trên 8-10 ảnh, không phải dấu hiệu lỗi trong `export_rknn.py`,
`kd_loss.py`, hay pipeline train. Bài học cho khi có dataset PPE thật:
1. **Luôn báo cáo Precision/Recall ở ngưỡng conf THẬT SẼ DÙNG khi deploy**,
   không chỉ mAP tổng — đúng nguyên tắc đã có trong
   `docs/measurement-protocol.md` cho slice, giờ áp dụng thêm cho
   confidence threshold.
2. **Luôn test trên ảnh KHÔNG nằm trong train/val**, không chỉ tin
   `results.csv` — một model có thể báo mAP hoàn hảo trên val (chỉ 1-2
   ảnh) và vô dụng trên ảnh thật đầu tiên gặp phải.
3. Dataset thật cần **hàng trăm/nghìn ảnh mỗi lớp**, không phải chục ảnh —
   đây là lý do `docs/research-plan.md` xếp việc thu thập dataset PPE là
   việc tốn công nhất của cả đồ án, không phải việc phụ.

**Dataset dùng cho lần verify này:** `data/processed/lab_safety_demo/`
(10 ảnh Wikimedia Commons + `bus.jpg` từ ultralytics assets, 3 lớp) —
**không commit vào git** (đúng quy ước sẵn có của repo, `.gitignore` loại
trừ toàn bộ `data/processed/*`); giữ lại trong working tree như một fixture
nhỏ để tái hiện phát hiện này hoặc dùng làm smoke-test cho pipeline train,
không phải dữ liệu thi đấu thật.

**File liên quan:** không có file code nào cần sửa; phát hiện thuộc về
phương pháp đo lường, không phải lỗi phần mềm.

## 18. Image OS "sạch hơn, ổn định hơn" nhưng không có driver NPU — flash nhầm 1 lần, mất ~30 phút

**Triệu chứng, xác minh thật, 2026-09-28:** setup Orange Pi 5 thật lần
đầu. Ban đầu khuyên dùng image cộng đồng "Ubuntu Rockchip" (Joshua-Riek,
`ubuntu-24.10-preinstalled-server-arm64-orangepi-5.img.xz`, kernel
`6.11.0-1006-rockchip`) vì lý do chung chung "ổn định hơn, ít lỗi driver
hơn bản vendor". Flash thật, boot thật, SSH vào được, chạy đủ 3 bước
kiểm tra phần cứng (`docs/board-setup.md` §3) — CPU/RAM/nhiệt độ đều
đúng — nhưng:
```
ls /dev/rknpu*        -> No such file
dmesg | grep -i rknpu -> (không có dòng nào)
modprobe rknpu         -> Module rknpu not found in directory ...
find /lib/modules/.../  -> không có file nào tên rknpu
```
Không phải "chưa load" — module **không tồn tại trong kernel này**.
Tra cứu thêm: repo Joshua-Riek/ubuntu-rockchip đã **archive (ngừng phát
triển)**, và yêu cầu thêm RKNPU driver đã bị từ chối chính thức ("not
planned", issue #1093). Phải flash lại từ đầu bằng image chính thức
Orange Pi (`Orangepi5_x.x.x_ubuntu_jammy_server_linux6.1.99.7z`, kernel
BSP `6.1.99-rockchip-rk3588` của chính Rockchip/Orange Pi) — image này
có sẵn `rknpu 0.9.8` (bản mới nhất) load đúng ngay từ boot đầu tiên,
xác nhận qua `dmesg`: `[drm] Initialized rknpu 0.9.8 20240828`.

**Nguyên nhân:** tiêu chí chọn image ("cộng đồng thường sạch/ổn định
hơn vendor") là lời khuyên đúng cho mục đích chung (Docker host, NAS,
dev box thường) nhưng **sai cho project này cụ thể** — vì tiêu chí đó
không hề tính đến yêu cầu cứng "phải có driver RKNPU", và không có bước
nào kiểm tra trước xem nguồn image có driver đó không trước khi khuyên
dùng. Một khuyến nghị hạ tầng chung chung áp dụng nhầm vào một ràng buộc
rất cụ thể của đúng project này.

**Bài học:** khi một thành phần hạ tầng (OS image, base container, SDK
version...) được chọn thay cho người dùng, **liệt kê rõ ràng NHỮNG YÊU
CẦU CỨNG của project trước khi so sánh các lựa chọn theo tiêu chí chung
chung** ("ổn định hơn", "mới hơn", "nhẹ hơn") — với project này, "có
driver RKNPU" là yêu cầu cứng phải xếp trên "ổn định hơn". Nếu không
chắc một lựa chọn có đáp ứng yêu cầu cứng, **nói rõ điều đó chưa được
xác minh** trước khi đề xuất, thay vì đề xuất trước rồi phát hiện thiếu
sau khi đã tốn công flash + boot + verify.

**Phát hiện phụ, cũng xác minh thật:** trên kernel BSP `6.1.99` (image
đúng), RKNPU đăng ký như **DRM device** (`/dev/dri/renderD1xx`), không
tạo `/dev/rknpu*` — nên bản thân bước kiểm tra "NPU có driver chưa" nếu
chỉ nhìn `/dev/rknpu*` (cách viết cũ trong `docs/board-setup.md`) SẼ
BÁO SAI ngay cả trên image đúng. Cách kiểm tra đúng: tìm dòng `[drm]
Initialized rknpu <version>` trong `dmesg`, không phải tìm tên file cụ
thể. `docs/board-setup.md` §3 đã cập nhật lại.

**File liên quan:** `docs/board-setup.md` §1, §3.

## 19. `CameraSource._await_first()` coi frame đen lúc khởi động là "đã sẵn sàng" — và một buổi debug sai hướng hoàn toàn vì tin nhầm vào giả thuyết phần cứng

**Triệu chứng, xác minh thật, 2026-09-28:** camera Logitech C270 cắm vào
Orange Pi 5, chụp thử qua `v4l2-ctl`/OpenCV ở 640×480 YUYV, đọc 5-10
frame liên tiếp ngay sau khi mở device — **luôn luôn** ra ảnh đen tuyệt
đối, `std=0` chính xác (`Y=16, U=V=128` lặp lại y hệt trên toàn khung
hình, qua nhiều lần chụp không dao động dù 1 bit). Đã tốn nhiều thời
gian đi theo hướng phần cứng: đổi 3 cổng USB (2 cổng EHCI + 1 cổng
xHCI), tắt USB autosuspend, kiểm tra dmesg tìm dấu hiệu reset/undervolt,
hỏi lại thông số sạc (5V/4A) — **tất cả đều đúng hướng chẩn đoán hợp
lý, nhưng SAI**, vì bằng chứng cốt lõi (frame tĩnh tuyệt đối qua nhiều
lần chụp độc lập) đã bị hiểu sai ý nghĩa.

**Nguyên nhân thật, tìm ra sau khi người dùng verify bằng công cụ
khác** (chụp 1280×720 MJPEG + đợi đủ ~3 giây thời gian thực → ra ảnh
màu thật): mọi webcam UVC cần một khoảng thời gian thực (~2-3 giây) để
auto-exposure hội tụ sau khi mở device — trong khoảng đó, camera trả về
một frame "trống" đồng nhất tuyệt đối, không phải dữ liệu rác hay lỗi.
Cách tôi chụp (`v4l2-ctl --stream-count=5` ở 640×480, hoặc đọc 10 frame
liên tiếp qua OpenCV ngay lập tức) hoàn thành trong **chưa tới 1 giây
thời gian thực** — không đủ để auto-exposure hội tụ, bất kể cổng USB
hay nguồn điện nào. Đổi cổng/tắt autosuspend/khởi động lại board không
thay đổi kết quả vì chúng không ảnh hưởng đến biến thực sự quan trọng
(thời gian thực trôi qua), nên mọi phép thử đều "xác nhận" nhầm giả
thuyết phần cứng.

**Bài học phương pháp luận:** dữ liệu "tĩnh tuyệt đối qua nhiều lần
thử" (std=0, hoặc giá trị lặp lại y hệt) dễ bị đọc thành "lỗi nhất
quán, chắc chắn có nguyên nhân hệ thống" — nhưng cũng chính là dấu hiệu
kinh điển của "chưa đủ thời gian cho một quá trình hội tụ tự nhiên"
(auto-exposure, auto-gain, cảm biến khởi động, cache chưa nóng...).
Trước khi đổ cho phần cứng vì kết quả "nhất quán", tự hỏi: phép thử có
để đủ THỜI GIAN THỰC trôi qua chưa, hay chỉ nhanh chóng hỏi lại nhiều
lần trong tích tắc?

**Bug thật tìm được nhờ vụ này, đã sửa:**
`src/labsafe/io/camera.py`'s `_await_first()` chỉ kiểm tra
`frame is not None` — với luồng đọc nền (`threaded=True`, mặc định cho
camera sống) chạy `cap.read()` liên tục ngay từ `__init__`, điều kiện
này gần như chắc chắn đúng chỉ sau vài mili-giây, ngay cả khi frame đó
là frame đen lúc khởi động. Nghĩa là **pipeline thật (`labsafe.app
--source 0`) có thể bắt đầu suy luận trên frame rác đầu tiên mà không
báo lỗi gì** — đúng dạng "chạy được, kết quả sai âm thầm" mà cả file
này tồn tại để bắt.

**Biện pháp:** `_await_first()` giờ kiểm tra thêm `frame.std() >=
min_std` (mặc định 2.0) trước khi coi frame là dùng được — một frame
suy biến/đồng nhất tuyệt đối có std ≈ 0, ảnh thật (kể cả phòng tối) thì
không. Không dùng một khoảng `sleep` cố định (phụ thuộc camera/ánh sáng
cụ thể, không tổng quát) mà dùng điều kiện đo được trực tiếp trên chính
dữ liệu.

**Verify thật:** chạy `CameraSource(0)` thật trên board — trước khi sửa
sẽ trả về gần như ngay lập tức; sau khi sửa, đo được mất **2.02 giây**
thật để có frame hợp lệ (`std=54.93`, `mean=144.58` — khớp với ảnh màu
thật đã xác nhận bằng mắt). 150 test vẫn xanh.

**File liên quan:** `src/labsafe/io/camera.py` (`_await_first`).

## 20. COCO-80 checkpoint + `configs/classes.yaml` (`stage_a.names: [person]`): now refused at load by a guard (verified on the board); the pre-guard crash at warmup is inferred from code, never observed

**Triệu chứng, mức xác minh cụ thể (không gộp chung một mức cho cả đoạn):**
chạy `python3 -m labsafe.app` hoặc
`--bench --config configs/exp/N1_rknn_fp16_640.yaml` với
`runs/export/yolo26n_fp16_640.rknn` (checkpoint COCO gốc, 80 lớp, chưa
fine-tune) và `configs/classes.yaml` mặc định (`stage_a.names: [person]`,
1 lớp), **nếu không có guard**, không crash ngay lúc `RKNNBackend.__init__`
-- nó crash **sau đó**, trong `router.stage_a.warmup()`, với
`ValueError: model output has 80 class channels but the running config
declares 1 class(es)` từ `yolo_decode(..., expect_nc=1)`. Mức xác minh của
câu này: suy ra từ đọc code + xác nhận bằng unit test trên laptop (mock
`rknnlite`), **không** tái hiện được thật trên board, vì guard mới (xem
Biện pháp bên dưới) chặn trước khi `warmup()` chạy tới -- guard đó, và chỉ
guard đó, **đã xác minh thật trên board** ở bước T2a (2026-09-28,
`.bangiao/ket-qua-test.md`): board raise đúng `ValueError` của guard, đúng
như kỳ vọng, trước khi mở NPU runtime.

Mọi config `configs/exp/N0..N5*.yaml` hiện có, và `scripts/bench/bench_matrix_npu.sh`
(mặc định `NAMES=person`), sẽ chết theo cách này (nay là qua guard, không
còn qua `yolo_decode`) nếu chạy với đúng checkpoint COCO-80 này. Suy ra từ
đó: lần chạy parity PASS trên board hẳn đã dùng `--names` đủ 80 lớp, không
phải `configs/classes.yaml` -- nhưng đây vẫn là suy luận, chưa phải bằng
chứng trực tiếp: `ket-qua-test.md` T0 không ghi lại `len(names)` của config
dùng cho lần parity đó, và board offline lúc viết mục này nên không kiểm
lại được. Ai xác minh lại nên dán output `len(c["names"])` của
`runs/parity/*/config.json` vào `ket-qua-test.md` T0 để khép suy luận này
lại thành bằng chứng.

**Nguyên nhân:** `configs/classes.yaml` khai `stage_a.names: [person]` vì
đó là taxonomy SẢN PHẨM của project (một model fine-tune cho đúng lớp này
sẽ ra output 1+4 kênh). Nhưng checkpoint hiện có (`yolo26n_fp16_640.rknn`)
là COCO-pretrained nguyên bản, chưa fine-tune -- output của nó luôn là
4+80 kênh bất kể config nào được truyền vào. `yolo_decode(expect_nc=...)`
đúng theo thiết kế (mục #5 ở trên: KHÔNG được gỡ guard này, nó tồn tại
chính xác để bắt kiểu lệch số lớp này thay vì âm thầm argmax sai) -- lỗi
không nằm ở guard, mà ở việc không có gì kiểm tra sớm hơn, trước khi tốn
thời gian mở NPU runtime rồi mới chết ở bước warmup.

**Biện pháp trong repo:**
- `configs/classes_coco80.yaml` mới: khai đủ 80 tên COCO, đối chiếu trực
  tiếp với `*.rknn.card.json` thật trên board (không đoán lại thứ tự từ
  tài liệu Ultralytics).
- `Backend.__init__` (`src/labsafe/models/backends.py`) nhận thêm
  `keep_classes` -- router chỉ thấy các lớp được liệt kê (ví dụ
  `[person]`), dù model output đủ 80 lớp. Không lọc thì ghế/laptop/chai...
  sẽ bị coi là người, và tracker/zone/fall/occupancy đều sai.
- `RKNNBackend.__init__` thêm guard so khớp `card.class_names` với
  `names` của config đang chạy, raise `ValueError` **ngay khi backend
  được tạo** (trước `warmup()`, trước khi mở NPU runtime) nếu hai danh
  sách khác nhau, kèm gợi ý dùng `configs/classes_coco80.yaml` +
  `keep_classes`.
- Chưa sửa `scripts/bench/bench_matrix_npu.sh` và các config `N0..N5*.yaml`
  hiện có trong lần này -- chúng vẫn sẽ crash theo cách cũ (lúc warmup,
  không phải lúc load) nếu chạy nguyên trạng với checkpoint COCO-80. Việc
  cần làm tiếp: cho `bench_matrix_npu.sh` một cờ `--classes`/`NAMES` trỏ
  đúng `classes_coco80.yaml` khi ma trận N* thật sự chạy với checkpoint
  chưa fine-tune này.

**File liên quan:** `src/labsafe/models/backends.py`,
`configs/classes_coco80.yaml`, `configs/exp/N1L_live_rknn_fp16_640.yaml`,
`configs/exp/N0L_live_cpu_onnx_640.yaml`, `scripts/bench/bench_matrix_npu.sh`
(chưa sửa, xem trên).

## 21. `awk printf "%.1f"` silently swaps `.` for `,` under a comma-decimal locale — a CSV column that looks fine and parses wrong

**Triệu chứng, xác minh thật, 2026-09-28, trên máy laptop dùng để viết
`scripts/board/board_sampler.sh`:** `awk -v v=27800 'BEGIN{printf "%.1f", v/1000}'`
in ra `27,8`, không phải `27.8`. Không có lỗi, không có cảnh báo -- output
trông hoàn toàn hợp lệ (một số thập phân), chỉ sai dấu phân cách. Một
script Python đọc CSV này bằng `csv.DictReader` + `float(...)` (đúng cách
`.bangiao/ke-hoach.md` §6 T6 dùng để tổng hợp số liệu) sẽ ném
`ValueError: could not convert string to float: '27,8'` -- hoặc tệ hơn,
nếu code dùng `try/except` nuốt lỗi rồi coi cột đó là "NA", việc mất dữ
liệu nhiệt độ sẽ không ai để ý cho tới khi đọc báo cáo cuối thấy thiếu số.

**Nguyên nhân:** `awk`'s `printf "%.1f"` tôn trọng `LC_NUMERIC` của
locale hệ thống. Máy này có `LC_NUMERIC=vi_VN` (locale tiếng Việt dùng
dấu phẩy làm dấu thập phân) trong khi `LANG=en_US.UTF-8` -- một tổ hợp
locale hỗn hợp hoàn toàn hợp lệ và phổ biến (ngôn ngữ hệ thống tiếng Anh,
định dạng số/ngày kiểu Việt), không phải cấu hình lỗi. `date`, `grep`
không bị ảnh hưởng (chúng không định dạng số thập phân qua `printf` kiểu
này), nên dễ tưởng "locale không liên quan gì tới một script bash đơn
giản" cho tới khi cụ thể `awk printf "%.1f"` được dùng.

**Biện pháp trong repo:** `scripts/board/board_sampler.sh` đặt `export LC_ALL=C`
ngay đầu script (sau `set -u`), trước bất kỳ lời gọi `awk`/`date` nào --
đảm bảo mọi cột số trong CSV luôn dùng `.` làm dấu thập phân bất kể locale
thật của board là gì, không cần biết trước board được cấu hình locale nào.

**Nguyên tắc chung:** bất kỳ script shell nào định dạng số bằng
`printf`/`awk printf` và ghi ra một file sẽ được **một chương trình khác
đọc lại bằng `.` làm dấu thập phân** (CSV, JSON, hầu hết mọi format trao
đổi dữ liệu) nên đặt `LC_ALL=C` (hoặc ít nhất `LC_NUMERIC=C`) ở đầu script
-- không giả định locale của máy chạy nó trùng với locale máy viết nó,
và không giả định "chỉ ảnh hưởng ngày tháng" khi chưa kiểm tra riêng từng
lệnh định dạng số.

**File liên quan:** `scripts/board/board_sampler.sh`.

## 22. A second unscoped `grep -oE '[0-9]+'` re-matched the digit inside its own label — one NPU-load sample corrupted every CSV row after it

**Triệu chứng, xác minh thật (đọc lại review độc lập, `.bangiao/danh-gia.md`
mục 1, và tái hiện lại bằng tay ở đây, 2026-09-29):**
`scripts/board/board_sampler.sh`'s original NPU-load parser was two chained
`grep -oE` calls:

```bash
c0v="$(echo "$load_line" | grep -oE 'Core0:[[:space:]]*[0-9]+%' | grep -oE '[0-9]+')"
```

Với `load_line="NPU load:  Core0: 44%, Core1: 12%, Core2: 7%,"`, lệnh này
in ra **hai dòng**: `0` (từ chữ số trong chính nhãn `Core0`) rồi `44` (giá
trị thật), thay vì một số duy nhất. Không có gì báo lỗi -- `grep -oE` vẫn
exit 0, và `$(...)` vẫn gán thành công một chuỗi (có chứa một newline ẩn
bên trong). Chuỗi có newline đó được nối thẳng vào biến `row` rồi
`echo "$row" >> "$OUT"`, khiến **một dòng logic bị ghi thành nhiều dòng vật
lý** trong file CSV. Bằng chứng thật, không phải suy đoán: file đã thu thập
trên board, `runs/board/samples/N1L_sustained_20260928-221918.csv`, có
header 17 cột nhưng mỗi dòng dữ liệu logic bị bẻ thành 4 dòng vật lý (ví
dụ `...,1000.0,0` / `0,1` / `0,2` / `0,7633,0.70`) -- đúng khớp cơ chế lỗi
này. Reviewer độc lập tái hiện lại được chính xác trên laptop trước khi
Coder sửa (mục 1 của `.bangiao/danh-gia.md`).

**Nguyên nhân:** grep thứ nhất đã trích đúng đoạn con `Core0:  44%`, nhưng
grep thứ hai (`[0-9]+`, không giới hạn phạm vi) khớp **mọi** dãy chữ số
trong chuỗi đó -- kể cả chữ số `0` nằm ngay trong tên nhãn "Core0" -- không
chỉ giá trị phần trăm phía sau dấu `:`. Bài kiểm tra tay ban đầu của lần
sửa trước (`.bangiao/thay-doi.md` vòng 1, mục "Diễn giải thêm") không dùng
đúng chuỗi mẫu thật dạng `CoreN: NN%` nên không bắt được lỗi này -- một lần
kiểm tra tay không thay thế được test hồi quy chạy với đúng dữ liệu mẫu
thật.

**Biện pháp trong repo:**
- Thay hai `grep -oE` bằng một `sed -nE
  's/.*Core0:[[:space:]]*([0-9]+)%.*/\1/p'` -- capture group trích thẳng
  đúng nhóm số nằm sau `Core0:` và trước `%`, không còn bước thứ hai nào có
  thể khớp nhầm chữ số của nhãn.
- Thêm một chốt an toàn độc lập với bug cụ thể này: trước khi ghi `row` vào
  CSV, script hỏi hàm `row_matches_header` xem `row` có nên được giữ hay
  không; nếu không, in cảnh báo ra stderr và **không ghi dòng đó**, thay vì
  ghi một dòng hỏng rồi để người đọc CSV sau này tự phát hiện.
  **Lưu ý (danh-gia.md Vòng 2, R2-1):** phiên bản đầu của chốt này (vòng
  sửa 1) tự nó có bug tương tự về bản chất — nó so `row_fields="$(csv_field_count
  "$row")"` với `header_fields` bằng `[ "$row_fields" -ne "$header_fields" ]`
  ngay trong vòng lặp. Khi `row` chứa newline nhúng (đúng kiểu hỏng mà chính
  bug này sinh ra), `csv_field_count` in ra **nhiều dòng** (awk in một NF
  cho mỗi dòng input), nên `[ -ne ]` nhận một chuỗi nhiều dòng, báo lỗi
  `integer expression expected` (exit 2), và `if` coi exit khác 0 đó là
  "false" — rơi vào nhánh **giữ lại** dòng hỏng, chính xác là loại hỏng chốt
  này được viết ra để chặn. Đã sửa bằng cách tách quyết định thành hàm thuần
  `row_matches_header(header, row)`: hàm này kiểm tra newline nhúng trước
  tiên (`case "$row" in *$'\n'*) return 1;; esac`) và chỉ so số cột **sau
  khi** đã chắc chắn `row` là một dòng — không bao giờ gọi `[ -ne ]` trên
  một chuỗi có thể nhiều dòng. Hàm này **không tự động bắt được mọi nguyên
  nhân tương lai gây lệch số cột** — nó bắt được đúng hai điều đã liệt kê
  (newline nhúng, và số cột khác header); một dạng hỏng khác chưa hình dung
  ra vẫn có thể lọt qua nếu nó không rơi vào một trong hai điều kiện này.
- Tách `parse_npu_load` và `csv_field_count` thành các hàm bash thuần
  (không I/O ngoài đối số/stdout), và bọc phần thân chạy thật (root check,
  discovery, vòng lấy mẫu) trong `main()`, chỉ gọi khi script được **thực
  thi** (`BASH_SOURCE[0] == $0`), không gọi khi bị `source`. Nhờ vậy
  `tests/test_board_sampler.py` gọi được `parse_npu_load` với chuỗi mẫu
  thật trên laptop, không cần root/board/sysfs giả.

**Verify thật:** `tests/test_board_sampler.py::test_parse_npu_load_returns_one_line_not_two`
dùng đúng chuỗi mẫu `"NPU load:  Core0: 44%, Core1: 12%, Core2: 7%,"`.
Positive control xác nhận bằng tay: chạy lại đúng hai lệnh `grep -oE` cũ
với chuỗi này cho ra `$'0\n44'` (2 dòng) -- test này sẽ FAIL nếu dán code
cũ vào; với `sed -nE` mới, ra đúng `44` (1 dòng), test PASS.

Cho riêng `row_matches_header` (chốt sau khi sửa R2-1):
`tests/test_board_sampler.py::test_row_matches_header_false_for_a_row_with_an_embedded_newline`,
`::test_row_matches_header_false_for_a_row_built_from_the_old_buggy_parse_output`
gọi thẳng hàm quyết định (không chỉ `csv_field_count`) với một row có
newline nhúng và assert nó bị drop.
`::test_row_matches_header_with_old_buggy_ne_logic_would_have_kept_the_bad_row`
là positive control chạy trong chính bộ test (không chỉ bằng tay): dựng lại
đúng phép so sánh `[ -ne ]` cũ trên cùng dữ liệu và assert nó **giữ lại**
dòng hỏng -- chứng minh hàm mới sửa đúng lỗi này, không phải sửa một lỗi
khác trông giống.

**Nguyên tắc chung:** một pipe `grep -oE PATTERN1 | grep -oE PATTERN2` mà
`PATTERN2` không neo (`^`/`$`/ranh giới rõ ràng) vào đúng phần đã được
`PATTERN1` cô lập sẽ khớp lại bất kỳ phần nào của `PATTERN1`'s output
trùng hình dạng với `PATTERN2` -- kể cả chữ số/ký tự nằm trong chính literal
label dùng để định vị giá trị cần lấy. Dùng `sed`/`grep -P` với capture
group (`\1`) để trích đúng một nhóm, thay vì hai lượt lọc rộng rồi thu hẹp
dần, luôn an toàn hơn khi label và value có thể trùng "hình dạng" (ở đây:
cùng là chữ số).

**File liên quan:** `scripts/board/board_sampler.sh`, `tests/test_board_sampler.py`.

## 23. `trap cleanup EXIT` chạy sau khi `main()` đã `return` — biến `local` bên trong `main` đã hết phạm vi, `cleanup` chết vì `_owner: unbound variable`, `chown` không bao giờ chạy

**Triệu chứng, xác minh thật (danh-gia.md Vòng 2, R2-2; tái hiện lại ở đây
bằng cách giả lập môi trường root, 2026-09-29):** `scripts/board/board_sampler.sh`
(bản vòng sửa 1) khai `OUT`, `META`, `_owner` là `local` bên trong `main()`,
định nghĩa `cleanup()` (đọc ba biến đó) ngay sau, rồi `trap cleanup EXIT`.
Nhìn qua tưởng là an toàn: `cleanup` được định nghĩa **trong cùng phạm vi**
với các biến nó dùng, nên có vẻ như lúc nào gọi cũng thấy được chúng.
**Nhưng thật ra không phải vậy** — hai điều bị nhầm lẫn với nhau:

- (A, tưởng là đúng) "`cleanup` được định nghĩa lồng trong `main`, nên nó
  luôn thấy biến `local` của `main`."
- (B, sự thật) Bash không có scope kiểu lexical cho **định nghĩa hàm** —
  `foo() { ...}` viết bên trong một hàm khác vẫn tạo ra một hàm **toàn cục**
  ngay khi dòng đó được thực thi. Biến mà hàm đó tham chiếu được tra cứu
  **tại thời điểm gọi**, không phải tại thời điểm định nghĩa. Khi vòng lặp
  lấy mẫu chạy xong bình thường, `main` `return`, các biến `local` của nó
  bị hủy ngay lập tức — nhưng script còn tiếp tục chạy hết (không có gì sau
  lệnh `main "$@"`) rồi mới thật sự `exit`, và **chỉ lúc đó** `trap ... EXIT`
  mới kích hoạt `cleanup`. Tại thời điểm này, `main` đã return từ lâu,
  `OUT`/`META`/`_owner` không còn tồn tại. Dưới `set -u`, đọc một biến đã bị
  hủy không phải là chuỗi rỗng — là lỗi cứng `_owner: unbound variable`,
  `cleanup` dừng ngay dòng đó, **`chown` không bao giờ được gọi**.

Tái hiện được bằng cách giả lập root (không cần sudo/root thật): thêm một
`id` giả vào đầu `PATH` (báo uid 0) và một `chown` giả (chỉ ghi log thay vì
đổi owner thật), chạy thẳng `bash scripts/board/board_sampler.sh out.csv 1 1`.
Với bản có `local OUT DURATION_S INTERVAL_S META` + `local _owner=...`:
stderr in ra `line N: _owner: unbound variable`, log `chown` không có gì.
Với bản đã sửa (bỏ `local` cho ba biến này, các biến khác trong `main`
không liên quan tới `cleanup` vẫn giữ `local` như cũ): stderr rỗng, log
`chown` có đúng `testuser:testuser <out.csv> <out.csv>.meta.txt`.

**Nguyên nhân:** nhầm "hàm lồng nhau thấy được biến `local` của hàm cha, y
như một closure" (đúng ở nhiều ngôn ngữ khác) với hành vi thật của bash —
tầm vực biến `local` gắn với **lời gọi hàm đang chạy** (call frame), không
gắn với vị trí hàm được viết ra trong mã nguồn. Một `trap ... EXIT` được
đăng ký bên trong một hàm sẽ chạy **sau khi hàm đó đã return xong** (trừ khi
chính hàm đó gọi `exit` bên trong, như nhánh root-check ở trên nó), tức là
chạy ở đúng call frame **ngoài cùng** của script — nơi mọi biến `local` của
`main` đã biến mất.

**Hậu quả nếu không phát hiện:** mọi lần chạy thật trên board (không phải
nhánh lỗi non-root, mà là nhánh chạy xong bình thường — đúng nhánh sẽ dùng
nhiều nhất) đều kết thúc bằng một dòng lỗi khó hiểu trên stderr, và CSV/meta
luôn bị bỏ lại với owner `root` (trái với đúng mục đích comment của
`cleanup`) vì `sudo bash ...` chạy tới `chown` thì đã crash trước đó.

**Biện pháp trong repo:** bỏ `local` cho đúng ba biến mà `cleanup` cần thấy
(`OUT`, `META`, `_owner`) — để chúng là biến toàn cục thường (gán trong
`main`, nhưng không khai `local`), nên chúng còn tồn tại ở mọi nhánh thoát
(chạy xong bình thường, `exit 1`, hay bị ngắt bởi tín hiệu) vì `main` trong
script này chỉ được gọi đúng một lần ở top-level. Các biến khác của `main`
(`DURATION_S`, `INTERVAL_S`, `header`, `row`, v.v. — không được `cleanup`
tham chiếu) vẫn giữ nguyên `local` như cũ, không có gì cần đổi ở đó.

**Verify thật:**
`tests/test_board_sampler.py::test_main_cleanup_trap_runs_chown_without_unbound_variable_error`
chạy thẳng script thật (`bash scripts/board/board_sampler.sh <out> 1 1`, không
`source`) với `id`/`chown` giả trong `PATH` để mô phỏng root mà không cần
quyền root thật, rồi assert: không có `"unbound variable"` trong stderr, và
`chown` giả có được gọi với đúng owner/đường dẫn. Đã tự tái hiện bằng tay
(script ở trên) rằng test này FAIL đúng kiểu (`unbound variable`, không có
log `chown`) với bản `local` cũ, và PASS với bản đã sửa.

**Giới hạn đã biết:** test này KHÔNG dùng root/chown thật — nó không chứng
minh được quyền sở hữu file trên đĩa thực sự đổi đúng khi chạy trên board
thật với `sudo`. Nó chỉ chứng minh `cleanup` **chạy tới được** lệnh `chown`
với đúng đối số, thay vì chết trước đó vì biến không tồn tại — đây là phần
lỗi test được trên laptop. Phần "chown thật có đổi owner trên đĩa hay
không" cần xác nhận trên board thật (xem điều kiện B4 trong
`.bangiao/danh-gia.md` Vòng 2: `ls -l` file CSV/meta sau khi chạy thật phải
cho owner là user gọi `sudo`, không phải `root`).

**Nguyên tắc chung:** không giả định một `trap` đăng ký bên trong một hàm
sẽ chạy trong khi hàm đó (hoặc call frame của nó) vẫn còn "đang sống". Nếu
`cleanup` cần đọc biến, các biến đó phải hoặc (a) không phải `local` ở bất
kỳ hàm nào có thể return trước khi trap kích hoạt, hoặc (b) được đưa thẳng
vào chính lệnh `trap` dưới dạng giá trị đã expand tại thời điểm đặt trap
(`trap "chown '$OUT'..." EXIT`), hoặc (c) `cleanup` được gọi tường minh
ngay trước khi hàm return, không dựa vào `trap EXIT` cho đường thoát bình
thường (chỉ giữ `trap` cho đường thoát bất thường/bị ngắt).

**File liên quan:** `scripts/board/board_sampler.sh`, `tests/test_board_sampler.py`.

---

## 24. RKNN INT8 xuất tọa độ khung đã chuẩn hóa (chia cho imgsz) — decode như pixel thì mọi khung trượt, không có lỗi nào

**Triệu chứng, đo thật trên Orange Pi 5 (2026-09-29):** `yolo26n_int8_640.rknn`
(ultralytics 8.4.163, rknn-toolkit2 2.3.2) chạy parity gate so với ONNX FP32
trên 22 ảnh (20 frame lab có người + bus/zidane): **49 phát hiện, 0 khớp**
(`recall=0.0`, `precision=0.0`, `blank_rate=0.0`). Không phải
`zero_detection_collapse` (#3) — số phát hiện gần như bằng bản gốc (49 vs 47).
So tensor thô trên `bus.jpg`: cùng anchor, cùng lớp, điểm 0.910 (INT8) vs
0.903 (ONNX), nhưng khung là `[0.5, 0.45, 0.73, 0.48]` thay vì
`[320, 290, 469, 306]` — đúng bằng pixel chia 640.

**Nguyên nhân:** `ultralytics/engine/exporter.py` (8.4.163), khi
`format == "rknn"` + `quantize == 8` + task detect + không end2end, bọc model
trong `_NormalizeCoords` trước khi trace ONNX: tọa độ khung bị chia cho kích
thước input để khung (0–640) và điểm lớp (0–1) dùng chung một scale INT8
per-tensor mà điểm không bị làm tròn về 0. Đây là hành vi cố ý của upstream
(comment trong exporter: "INT8 graphs hold normalized coordinates"), nhưng
`yolo_decode()` của repo vẫn hiểu 4 kênh đầu là pixel letterbox. FP16 không
bị bọc nên không bị — parity FP16 cùng bộ ảnh: PASS, IoU 0.997.

**Biện pháp trong repo:**
- `ModelCard.box_coords` (`"pixels"` | `"normalized"`), do
  `scripts/export/export_rknn.py` ghi theo đúng điều kiện của exporter (có
  `_NormalizeCoords` trong ultralytics đã cài + int8 + không end2end).
- `yolo_decode(..., box_scale=...)`; `RKNNBackend` truyền `box_scale=imgsz`
  khi card nói `normalized`.
- Card INT8 cũ **không có** `box_coords` → `RKNNBackend` raise `ValueError`,
  không đoán. Card FP16 không có trường này → pixel (đúng với mọi export FP16
  đã qua parity).

**Verify thật (board, cùng 22 ảnh, sau khi sửa):** INT8 parity **PASS** —
recall 1.0, precision 0.959, IoU trung bình 0.867 (thấp hơn FP16 vì tọa độ
chỉ còn 256 mức ≈ 2.5 px/mức). Live 60 s mỗi bản, C270 640x480: infer NPU
FP16 47.8 ms → INT8 25.1 ms (1.9x); INT8 xử lý hết ~30 FPS của camera
(duplicate_ratio 0.17 = pipeline nhanh hơn camera). Test:
`tests/test_backends_guard.py` (box_scale, card normalized/pixels/thiếu
trường); gỡ `* box_scale` trong `yolo_decode` thì 2 test FAIL.

**Bài học:** parity gate đo **khớp khung**, không chỉ đếm phát hiện — một
kiểm tra kiểu "có ra detection không" (canary, liveness, FPS) đã cho qua bản
INT8 hỏng này.

**File liên quan:** `src/labsafe/models/modelcard.py`,
`src/labsafe/models/postprocess.py`, `src/labsafe/models/backends.py`,
`scripts/export/export_rknn.py`, `configs/exp/N2L_live_rknn_int8_640.yaml`.

---

## 25. Fine-tune trên PPE công trường: mAP50 0.80 trên test nhưng bỏ sót người ở lab mà COCO gốc bắt được 20/20

**Triệu chứng, đo thật (2026-09-30):** `yolo26n.pt` fine-tune 60 lượt trên
`data/processed/ppe_public` (subset 3 lớp person/goggles/gloves của
HB1204/PPE_Detection, 2612 ảnh train, RTX 3050). Test (244 ảnh, 732 khung):
mAP50 0.796, mAP50-95 0.377; AP50 person 0.864, goggles 0.725, gloves 0.798.
Trên 20 frame lab thật có người (`runs/board/N1L_withperson/`, cùng ngưỡng
conf): bản fine-tune thấy người ở **16/20** (conf 0.25) và **15/20** (0.4);
`yolo26n.pt` COCO gốc thấy **20/20** cả hai ngưỡng. Bị sót: frame 0000, 0003,
0004, 0018 (người ngồi gần, chỉ thấy nửa thân).

**Nguyên nhân (giả thuyết, chưa tách biến):** dataset toàn ảnh công trường,
người đứng xa toàn thân; fine-tune làm mất khả năng bắt người ngồi/che khuất
gần camera của COCO. Chưa chứng minh bằng thí nghiệm riêng (ví dụ trộn ảnh
COCO person, đóng băng backbone).

**Điểm tốt:** 0 goggles/gloves giả trên 20 frame lab có người + 30 frame bàn
(tay trần, không đeo) — nhưng đó chỉ là ca âm, chưa có ca dương ở lab.

**Bài học:** mAP trên tập cùng miền không dự báo được miền đích; phải so với
baseline COCO **trên ảnh lab thật** trước khi thay Stage A. Chưa dùng model
này thay Stage A.

**File liên quan:** `scripts/train/train_baseline.py` (đã sửa lỗi `ap50 or []` trên
numpy array làm crash dòng in cuối), `data/processed/ppe_public/`.

---

## 26. Thầy open-vocab zero-shot (YOLOE-11s) gán nhãn kính bảo hộ gần như vô dụng; găng tay yếu — không dùng nhãn tự động thô để train

**Triệu chứng, đo thật (2026-09-30):** `yoloe-11s-seg.pt` + text prompt, so với
nhãn người gán trên 244 ảnh test của `data/processed/ppe_public` (AP50 tự
tính bằng `scripts/eval/eval_teacher_openvocab.py`, IoU 0.5):

| prompt (person / kính / găng) | person AP50 | goggles AP50 (P, R @0.25) | gloves AP50 (P, R @0.25) |
|---|---|---|---|
| person / safety goggles / gloves | 0.776 | 0.093 (0.22, 0.13) | 0.395 (0.62, 0.39) |
| person / goggles / work gloves | 0.774 | 0.073 (0.27, 0.08) | 0.394 (0.65, 0.43) |
| person / protective glasses / hand wearing glove | 0.776 | 0.133 (0.28, 0.17) | 0.408 (0.59, 0.49) |

Cùng tập test, học trò đã fine-tune (#25): goggles 0.725, gloves 0.798.
Nhãn tự động với precision kính ~0.25 nghĩa là ~3/4 nhãn "kính" sai — huấn
luyện trên đó sẽ dạy model sai.

**Thử thêm (cùng tập test, prompt "protective glasses" / "hand wearing glove"):**

| model, imgsz | person AP50 | goggles AP50 (P, R) | gloves AP50 (P, R) |
|---|---|---|---|
| 11s, 640 | 0.776 | 0.133 (0.28, 0.17) | 0.408 (0.59, 0.49) |
| 11s, 1280 | 0.472 | 0.086 (0.19, 0.10) | 0.304 (0.48, 0.37) |
| **11l, 640** | 0.764 | **0.199 (0.31, 0.30)** | **0.456 (0.58, 0.56)** |
| 11l, 1280 | không đo xong (GPU 6 GB hết bộ nhớ, bỏ) | | |

Model lớn hơn cải thiện kính 0.13 -> 0.20 nhưng vẫn kém xa học trò fine-tune
(0.73); tăng độ phân giải làm **tệ đi** (lượt 1280 chạy khi GPU gần hết bộ
nhớ nên con số có thể bị ảnh hưởng, không dùng làm kết luận).

**Giới hạn của phép đo:** chỉ thử YOLOE 11s/11l, 3 cách đặt tên, ảnh công
trường (kính nhỏ, xa) — chưa thử Grounding DINO hay chạy thầy trên crop
người. Chưa có ca dương nào ở lab (chưa có ảnh đeo kính/
găng/áo blouse thật). Không kết luận "open-vocab vô dụng nói chung".

**Bẫy vận hành:** `YOLOE.set_classes()` tải `mobileclip_blt.ts` (572 MB) vào
**thư mục làm việc hiện tại**; chạy từ thư mục khác thì tải lại từ đầu, treo
20+ phút không báo lỗi, CPU ~0%. Chạy trong `/mnt/big/AGV_ANLEE/weights/ovd`.

**File liên quan:** `scripts/eval/eval_teacher_openvocab.py`.

---

## 27. RAM của app tăng đều ~6 MB/phút trên board (lượt 50 phút: 157 -> 543 MB) — rò thật, chưa tìm ra nguồn

**Triệu chứng, đo thật (2026-09-30, Orange Pi 5, RKNN FP16, C270, ~19,5 FPS, 58 775 khung):**
RSS của tiến trình `labsafe.app --bench` tăng gần tuyến tính suốt 50 phút, **không
có bình nguyên**: 157 MB (đầu) -> 543 MB (giây 3000). Độ dốc theo từng đoạn:
+8,3 (1-5 phút), +7,6, +7,2, +5,6, +2,8, +6,5 MB/phút; 20 phút cuối +5,2
MB/phút. Khoảng 6,6 KB mỗi khung hình. Nhiệt độ tối đa 54,5 °C, p99 71 ms,
không lỗi. Ở tốc độ này 8 GB RAM đủ khoảng 20 giờ. (Lượt 15 phút trước đó,
KNOWN "RSS chưa bão hòa", giờ đã được xác nhận là rò.)

**Đã loại trừ (mỗi mục nêu mức chắc chắn):**
- *Chắc chắn* — camera + letterbox: 4 669 khung, RSS đứng yên 65,1 MB.
- *Chắc chắn, nhưng nhỏ* — phần Python của pipeline (danh sách thống kê, sự kiện, tracker,
  router): chạy detector giả trên laptop, 4,36 triệu khung: +0,20 MB/1000 khung,
  tức ~3% mức rò trên board (6,57 MB/1000 khung). Các danh sách thống kê vẫn
  không có giới hạn (vô hại ở vài giờ). Tracker/router có giới hạn (lịch sử 90
  điểm, track quá hạn bị xóa).
- *Không kết luận được* — NPU thuần (4000 lần gọi, sau ~55 MB cấp phát ban đầu RSS
  dao động 220-236 MB) và camera + backend RKNN đầy đủ (8 116 khung, RSS
  dao động 200-266 MB): độ nhiễu ±50 MB **bằng đúng mức rò cần phát hiện**
  (6,6 KB x 8 000 = 54 MB), nên hai phép thử này không đủ nhạy.
- *Chỉ giải thích một phần* — phân mảnh bộ nhớ: `MALLOC_ARENA_MAX=1
  MALLOC_TRIM_THRESHOLD_=131072`, 10 phút: RSS trung bình thấp hơn ~100 MB
  (192-242 so với ~300 MB) nhưng độ dốc vẫn ~+6 MB/phút, và chậm hơn
  (17,5 so với 19,5 FPS, p50 56 so với 50,5 ms). Không dùng làm cách sửa.
- `tracemalloc` trong app thật: bộ nhớ Python truy vết được không tăng đơn điệu
  (83 -> 114 -> 56 MB) trong khi RSS tăng +59 MB -> phần tăng nằm ngoài
  bộ cấp phát Python (thư viện native hoặc ánh xạ bộ nhớ khác).
  (Danh sách dòng code chi tiết không lưu được: bộ lọc in sai, chưa chạy lại.)

**Chưa biết:** nguồn rò. Ứng viên còn lại: runtime NPU/`rknnlite` khi có
đầu ra lớn theo từng khung; đường đầu ra float của RKNN; OpenCV/luồng camera
khi phối hợp với NPU.

**Bước tiếp theo đề xuất (chưa làm):** (1) so sánh `/proc/PID/smaps` đầu và cuối
để biết vùng nhớ nào tăng (heap, anon, hay ánh xạ NPU); (2) vòng gọi NPU thuần
dài 20 000+ lần, đo bằng smaps thay vì RSS tổng; (3) `LD_PRELOAD` trình theo dõi
malloc hoặc `valgrind --tool=massif` (chậm).

**Cách tránh cho demo:** đặt lịch khởi động lại app định kỳ (ví dụ mỗi 6 giờ ~
+2 GB tối đa); theo dõi RSS bằng `scripts/board/watch_run.sh`. Đừng báo cáo "chạy ổn
định nhiều ngày" khi chưa sửa.

**File liên quan:** `runs/board/soak/` (số liệu thô, gitignored),
`scripts/board/watch_run.sh`, `scripts/board/board_sampler.sh`.

---

## 28. Stage B huấn luyện trên crop người: không chính xác hơn model toàn khung — lợi ích chỉ là chi phí (320 thay vì 640)

**Giả thuyết ban đầu (sai):** train kính/găng trên crop người sẽ cải thiện AP kính vì vật to ra trong crop.
**Đo thật (2026-09-30, `data/processed/ppe_crops`, 4 140/188/404 crop train/val/test, tách theo
nguồn ảnh của HB1204, YOLO26n imgsz 320, dừng sớm lượt 72):**

| tập test crop (404) | kính AP50 | găng AP50 |
|---|---|---|
| Stage B (train trên crop, 320) | 0.653 | 0.752 |
| model toàn khung `ppe_public_y26n` (640) chấm trên đúng các crop này | 0.691 | 0.739 |

Ngang nhau về độ chính xác (kính thấp hơn 0.04, găng cao hơn 0.01; một tập test 404
crop nên chênh dưới ~0.04 không đủ để kết luận). Lợi ích thật: mỗi lần gọi Stage B chỉ 1/4
số điểm ảnh; chi phí tỉ lệ theo số người thay vì cố định mỗi khung. Chưa đo chi phí thật
trên NPU (board offline).

**Giới hạn:** dữ liệu là ảnh công trường (#25), chưa có ảnh lab; hai model chấm bằng hai
đường khác nhau (`model.val` cho Stage B, tính AP tay cho model toàn khung).

**File liên quan:** `scripts/data/make_person_crops.py`, `data/processed/ppe_crops/` (gitignored).

---

## 29. Bộ lửa/khói Roboflow: cùng một ảnh gốc nằm ở cả train và val/test (điểm val/test sẽ cao giả)

**Triệu chứng, đo thật (2026-09-30, `betasecond/jimei-fire-smoke-yolo-dataset`):** chia gốc
train/valid/test = 9 156 / 872 / 435 ảnh. Tên ảnh dạng `<stem>_jpg.rf.<hash>.jpg`: Roboflow
nhân mỗi ảnh gốc thành ~3,8 bản tăng cường (2 419 ảnh gốc khác nhau trong train). Nhóm theo
`<stem>`: **290 ảnh gốc xuất hiện ở cả train và valid, 151 ở cả train và test**. Không có
lỗi nào báo ra; mọi công cụ (verify_dataset, train, val) đều xanh.

**Biện pháp:** `scripts/data/regroup_split.py` gộp ba tập, nhóm theo ảnh gốc, chia nguyên nhóm
80/10/10 (seed 0). Kiểm lại: 0 ảnh gốc chung giữa mọi cặp tập
(`data/processed/fire_smoke`: 8 380 / 1 045 / 1 038 ảnh; 2 530 / 316 / 316 nhóm).

**Kiểm chứng ngược:** bộ PPE `HB1204` (cũng Roboflow) **không** bị: val/test là ảnh gốc
chưa tăng cường, 0 ảnh gốc chung với train (871/112/244 ảnh gốc), nên các số PPE ở
#25/#28 không bị thổi phồng vì lý do này.

**Mức thổi phồng đo được:** Stage C (YOLO26n, 416, 50 lượt, train trên chia lại) chấm trên tập
test sạch: **mAP50 0,721** (lửa 0,771, khói 0,671; P 0,78, R 0,66). Chấm cùng model trên tập
test **gốc** của bộ dữ liệu (435 ảnh, nay nằm lẫn trong train mới, 151 ảnh gốc trùng train): mAP50
**0,837** (lửa 0,881, khói 0,792) -- cao hơn ~0,12 vì đã "thấy" các ảnh này.
Số 0,72 là con số được phép trích; khói khó hơn lửa. Bộ dữ liệu là ảnh thật nhưng không
phải phòng lab (tin tức, bật lửa, lò sưởi, cháy ngoài trời) và đã tăng cường mạnh.

**Giới hạn:** nhóm theo tên gốc chỉ bắt được bản tăng cường của cùng file; các khung gần
giống nhau từ cùng một video (tên khác nhau) vẫn có thể nằm ở hai tập.

**File liên quan:** `scripts/data/regroup_split.py`.

---

## 30. Stage B/C chạy tuần tự trong vòng lặp khung: trung bình rẻ nhưng gây cú giật độ trễ — bất đồng bộ trên nhân NPU riêng cắt đuôi p99

**Triệu chứng, đo thật (2026-10-01, Orange Pi 5, tất cả INT8):** mỗi lần gọi một mình trên NPU
(1 nhân): A (người, 640) 30,5 ms; B (kính/găng trên crop 320, MỖI NGƯỜI) 12,5 ms; C (lửa/khói,
416) 17,8 ms. Model nhỏ không rẻ theo số điểm ảnh: B chỉ nhỏ hơn A ~2,4 lần dù ảnh nhỏ 4 lần
(chi phí cố định ~10 ms mỗi lần gọi). Trong router cũ A, B, C gọi nối tiếp cùng một luồng nên dù
mỗi stage gán nhân NPU riêng (`core_mask`) chúng không bao giờ chạy cùng lúc.

**Đo hệ thống đầy đủ** (`configs/exp/L1_*` tuần tự so với `L2_*` bất đồng bộ, A/B/C trên nhân
0/1/2; phát lại cùng một video 1 800 khung, 1 người, occupancy 98%):

| | L1 tuần tự | L2 bất đồng bộ |
|---|---|---|
| FPS | 30,3 | 29,6 |
| độ trễ khung p50 | 29,4 ms | 29,5 ms |
| độ trễ khung p99 | **61,2 ms** | **49,9 ms** |
| độ trễ lớn nhất | 78,6 ms | 58,3 ms |

Trung bình gần như không đổi: nhờ gating (B mỗi 2 s/người, C mỗi 1 s) chi phí trung bình của
B+C chỉ khoảng 0,8 ms/khung. Cái bị ảnh hưởng là **đuôi**: khung nào trùng lượt B hoặc C phải
trả thêm 11-14 ms (hai cái trùng nhau: ~25 ms). Bất đồng bộ cắt p99 18% và độ trễ cực đại 26%.

**Biện pháp:** `src/labsafe/pipeline/async_stage.py` (`AsyncWorker`, `AsyncPPEChecker`,
`AsyncStageC`), bật bằng `async_stages: true` trong pipeline config (mặc định tắt để so sánh
song song/thích ứng trong cùng điều kiện). Hàng đợi có giới hạn, quá tải thì bỏ job và đếm
(tránh lặp lại kiểu tăng RAM không giới hạn của #27); lỗi worker được đếm và hiện trong
`router.async_stats()`, không nuốt im. Kết quả B/C trễ tối đa một chu kỳ kiểm tra (PPE 2 s,
lửa/khói 1 s). 10 test (`tests/test_async_stage.py`): submit không chờ model, kết quả đến sau,
track biến mất, khoảng kiểm lại, hàng đợi đầy, lỗi worker, router không cộng độ trễ B/C vào khung.

**Giới hạn:** 1 người, 20 khung ảnh lab lặp lại (có khung overlay vẽ sẵn), không phải dữ liệu
mới quay; B/C chưa được kiểm độ chính xác ở lab; chưa đo năng lượng. Với nhiều người, B cộng
~0,18 ms/khung mỗi người (12,5 ms / (2 s x 30 FPS)) -- chưa đo thực nghiệm. Khung "kính" trong
các crop này gần như không có đồ bảo hộ nên số phát hiện B không dùng để đánh giá B.

**Bài học đo lường:** lần chạy đầu tôi dùng video phát lại **không có người** (occupancy 0,
0 phát hiện) -- B không chạy lần nào và so sánh sạch sẽ nhưng vô nghĩa. Phải kiểm
`occupancy`/`stage_detections` trước khi tin phép so sánh; và đã loại nghi vấn tối ưu giải mã
(`class_subset`) gây mất người bằng cách đếm trực tiếp (15=15, 16=16 người trên 12 ảnh).

**File liên quan:** `src/labsafe/pipeline/async_stage.py`, `src/labsafe/pipeline/router.py`,
`configs/classes_lab_v1.yaml`, `configs/exp/L1_*`, `L2_*`, `scripts/bench/profile_pipeline.py`.

---

## 31. Camera: 20% vòng lặp suy luận lại trên ảnh cũ (FPS thổi phồng 37 so với 29,6 thật) — sửa; và RAM tăng nằm trong heap, chưa tìm ra nguồn (cập nhật #27)

**A. FPS thổi phồng (đã sửa).** Hệ thống đầy đủ L2, camera C270 thật, 5 phút: `fps_sustained` 37,0 nhưng
camera chỉ cấp 29,97 hình/giây và `duplicate_ratio` 20,5% -> FPS thật ~29,4; 20% công NPU/điện
tiêu vào việc nhận diện lại một ảnh không đổi. Sửa: `CameraSource.read(wait_new=True)` chờ (tối đa
0,25 s, rồi trả ảnh cũ thay vì treo) cho tới khi có ảnh mới; `app.py` dùng nó. Đo lại: `duplicate_ratio`
**0,0%**, FPS **29,6** khớp camera. Lưu ý: độ trễ vòng lặp (`latency_ms`) giờ gồm cả thời gian chờ
camera (~33 ms); độ trễ xử lý thật là `router.frame_latency_ms`. 3 test mới.

**B. RAM tăng khi chạy hệ thống đầy đủ (6,4-13 MB/phút tùy FPS, ~5 KB/khung): nguồn CHƯA tìm ra.**
Đã làm, đã đo (board, L2, 5-15 phút mỗi lượt):
- `smaps` giây 60 -> 270: **phần tăng nằm trong `[heap]`** (+29,6 MB heap, +2,6 MB anon); không phải vùng
  nhớ NPU, không phải thư viện, không phải mảng lớn mmap.
- Không phải code Python: router + tracker + vùng cấm + thống kê với model ONNX thật và 4 000 khung
  có người trên laptop, `tracemalloc`: +0,5 MB tổng (+0,72 MB/1000 khung, đúng các danh sách thống kê).
- NPU thuần 5 000 lần gọi: +6,5 MB (~1,6 KB/lần, trong nhiễu). Camera + letterbox 4 669 khung: 0 MB.
- Giảm thiểu bộ cấp phát KHÔNG hiệu quả: `MALLOC_MMAP_THRESHOLD_` 4 MB và 2 MB, `malloc_trim` mỗi 5 s
  (đã thử rồi gỡ bỏ): RAM vẫn tăng. Ngưỡng 128 KB trông "bị chặn" ở lượt 5 phút (răng cưa 185-273 MB)
  nhưng lượt 15 phút cho thấy vẫn tăng: RAM trung bình 218 -> 257 -> 307 MB, mức thấp nhất 141 -> 163
  -> 215 MB, và chậm hơn ~4 ms trung vị. Bài học: một cửa sổ 5 phút trên đường RAM răng cưa không đủ
  để kết luận "đã hết rò".
**Kết luận hiện tại:** rò nằm trong heap native của tiến trình khi camera + NPU chạy cùng nhau; chưa
cô lập được thành phần. Cách tránh: khởi động lại app định kỳ (~9 MB/phút -> 8 GB đủ ~15 giờ, khuyến
nghị mỗi 4-6 giờ). Bước tiếp theo: `valgrind --tool=massif`/`heaptrack` trên board, hoặc thử bản
`librknnrt`/`rknn-toolkit-lite2` mới hơn 2.3.2.

**Hiệu năng đo cuối (board, INT8, A+B+C bất đồng bộ, camera C270 640x480, có người 94% thời gian,
5 phút, `wait_new`):** 29,6 FPS thật (camera 29,97 là trần), độ trễ xử lý p50 25,5 / p90 35,3 /
p99 43,6 / max 49,0 ms; NPU nhân 0 (Stage A) tải 72,5% -> còn dư ~27% (năng lực thô ~37 FPS nếu
camera nhanh hơn); nhiệt độ tối đa 47 °C; CPU tiến trình 37% (một nhân = 100%); 0 job B/C bị bỏ, 0
lỗi worker; 295 lần chạy Stage C, 0 báo động lửa/khói giả; B chạy 248 lần.

**File liên quan:** `src/labsafe/io/camera.py`, `src/labsafe/app.py`, `tests/test_live_run.py`.

---

## 32. Gợi ý nhãn bằng model của chính project (`scripts/data/prelabel.py`): đúng ~75% trên ảnh công trường; gợi ý nhầm tay trần thành găng ở ảnh lab

**Mục đích:** thay hướng "thầy open-vocab tự gán nhãn rồi train thẳng" (#26, không đạt) bằng gán
nhãn bán tự động: máy vẽ khung gợi ý ngưỡng thấp, người sửa/xóa. `scripts/data/prelabel.py` chạy Stage A
(COCO person) + Stage B trên crop người đúng như hệ thống chạy (crop pad 10%, 320) + tùy chọn
`--lab-coat` (YOLOE-11l zero-shot), xuất nhãn YOLO + `review.csv` (ảnh kém tin cậy lên đầu) +
`scores.json` + `data.yaml`/`obj.names` để nhập CVAT/Label Studio.

**Đo thật** (244 ảnh test công trường chưa từng thấy; IoU >= 0,5; ngưỡng person 0,35, PPE 0,20):

| lớp | GT | gợi ý | đúng | precision | recall | cần xóa | cần vẽ thêm |
|---|---|---|---|---|---|---|---|
| person | 409 | 409 | 321 | 0,78 | 0,78 | 88 | 88 |
| goggles | 78 | 83 | 58 | 0,70 | 0,74 | 25 | 20 |
| gloves | 246 | 204 | 181 | 0,89 | 0,74 | 23 | 65 |

So với thầy zero-shot cùng tập (#26: kính precision 0,28 / recall 0,30), gợi ý từ model đã train tốt
hơn rõ rệt. Khoảng 25% khung cần sửa thay vì vẽ 100% từ đầu.

**Ca khó ở lab** (20 ảnh `runs/board/N1L_withperson`, người mặc áo thun trắng, tay trần, không đeo gì,
`--lab-coat`): person 20/20 đúng; **8 khung "gloves" sai** (tay trần), **2 khung "lab_coat" sai** (áo thun
trắng, conf 0,28-0,36), 0 goggles. Người sửa sẽ phải xóa các khung này; chưa có ca dương nào của lab
(chưa có ảnh đeo kính/găng/áo blouse) nên chưa biết gợi ý áo blouse thật đúng bao nhiêu.

**Giới hạn:** số trên là ảnh công trường, không phải lab; Stage B được train trên train-split của cùng
bộ dữ liệu (test là phần chưa thấy). Đo lại trên lô lab đầu tiên đã sửa xong trước khi tin tỉ lệ 75%.

**File liên quan:** `scripts/data/prelabel.py`.

---

## 33. Lô gán nhãn đầu tiên: toàn bộ khung `gloves` là một đôi găng nằm trên bàn; lỗi chọn ảnh của tôi làm lệch bộ

**Phát hiện (03/10/2026, `data/labeled/boot006`, 194 ảnh gán qua Roboflow):** 141 khung `gloves`
trong 141 ảnh, **91% nằm trong ô 4% quanh cùng một điểm (x 0,80, y 0,40)**, 13 khung còn lại ở
(0,72-0,76; 0,44) cùng mặt bàn: một đôi găng đặt trên bàn, không ai đeo. Lớp `gloves` định nghĩa là
"đang đeo", nên lô này **không có một ví dụ găng đeo nào**; huấn luyện sẽ dạy "găng = vật ở chỗ này trên
bàn". **Quyết định:** bỏ `gloves` khỏi lớp lab (`configs/lab_classes.yaml`, importer
`--drop-classes gloves`), thêm lại ở CUỐI danh sách khi có găng đeo thật (thu thứ Hai).
`goggles` cũng đã bỏ (lab không ai đeo; phiên đầu chỉ có 10 khung gợi ý).

**Lỗi chọn mẫu của tôi:** `prelabel.py` gợi ý găng chủ yếu trên đôi găng bàn này; khi chọn 197 ảnh để
gán nhãn tôi ưu tiên "ảnh có găng (lớp hiếm)" nên kéo vào nhiều khung cùng một cảnh. Bài học:
ưu tiên lớp hiếm theo gợi ý của máy chỉ đúng khi gợi ý đó không phải một vật tĩnh; cần kiểm vị trí khung
theo thời gian (độ phân tán) trước khi coi một lớp là "đa dạng".

**Tình trạng nhãn sau khi bỏ găng:** person 537, lab_coat 495 (sau khi gộp 158 mảnh áo mở), 83% người
có đúng 1 khung áo, 11% không áo. **25 khung áo (20 ảnh) nằm ngoài mọi khung người**: người ở mép/trên
cùng ảnh, bị cắt hoặc rất nhỏ, thiếu khung `person`
(`data/interim/coat_without_person.txt`). Chưa sửa; một model gộp (person + lab_coat) sẽ bị phạt oan ở các
người đó. Nhãn do một người gán, chưa có đối chiếu giữa hai người.

**File liên quan:** `scripts/data/import_roboflow.py`, `configs/lab_classes.yaml`.

---

## 34. Pilot `lab_coat` (146 ảnh, 1 phiên): AP50 0,91 trên tập test cùng phiên nhưng gọi áo thun trắng là áo blouse ở 18/20 ảnh

**Đo thật (03/10/2026, YOLO26n 60 lượt trên `data/processed/lab_v1`, person + lab_coat, train 146 ảnh, test = khối
cuối 39 ảnh của CÙNG phiên `boot006` có khoảng đệm):**

| phép thử | kết quả |
|---|---|
| test cùng phiên, `lab_coat` AP50 | **0,911** (P 0,94, R 0,87 ở conf 0,4; 71 khung) |
| test cùng phiên, `person` AP50 | 0,950 (COCO gốc 0,969; ngang nhau, fine-tune bắt nhiều hơn: R 0,96 so với 0,90) |
| **âm tính: người mặc áo thun trắng, không áo blouse**, ngày/góc/người khác (20 ảnh `runs/board/N1L_withperson`) | **18/20 ảnh bị báo `lab_coat`** (32 khung, conf 0,52-0,92) |
| phòng trống/tay-bàn phím (40 ảnh) | 3 ảnh báo `lab_coat` |
| góc cũ boot001-005 (37 ảnh, KHÔNG có nhãn) | 28/37 báo `lab_coat`; không đánh giá được đúng/sai |

**Kết luận:** điểm 0,91 là con số cùng phòng, cùng người, cùng ánh sáng nên **không phải bằng chứng model nhận
ra áo blouse**. Phép thử âm tính (áo thun trắng) cho thấy model học "quần áo trắng trên người" (hoặc nhớ
cảnh), không phải "áo blouse". Dữ liệu train có 11% người không áo blouse nhưng chủ yếu mặc đồ tối, và chỉ vài
người; không có ví dụ "áo trắng nhưng không phải áo blouse".

**Việc cần để sửa (chưa làm):** (1) ảnh âm tính khó: người mặc áo trắng/sáng không phải áo blouse (áo thun,
sơ mi trắng), (2) nhiều người khác nhau, (3) một phiên khác làm test. Thứ Hai quay đúng các cảnh đó. Trước khi
có chúng KHÔNG dùng model này để ra quyết định "có mặc áo blouse"; chỉ dùng làm gợi ý nhãn và biết nó thiên về báo dư.

**Bài học đo lường (lặp lại #17/#25):** tập test lấy từ cùng phiên cho điểm cao giả; phải kèm phép thử âm
tính từ miền khác trước khi tin một lớp. Phép thử này tốn 2 phút và đảo ngược kết luận.

**File liên quan:** `data/processed/lab_v1` (gitignored), `/mnt/big/AGV_ANLEE/weights/lab_v1_pilot_best.pt`.

## 35. Vùng nguy hiểm "phải mặc áo lab" (`requires: [lab_coat]`): gần như không báo nhầm người mặc áo, nhưng BỎ SÓT 58% người không mặc áo

**Tính năng (09/10):** vùng trong `configs/zones_lab.yaml` có `requires: [lab_coat]` -> người đứng trong vùng mà không có
áo là `zone_ppe_violation`; có áo thì được phép. Stage A là model 2 lớp `lab_v2` (person, lab_coat); khung áo gán cho
MỘT người chứa ≥50% diện tích áo (`router._attach_attire`), nhớ 2 s (`zones.attire_memory_s`). Chỉ lớp `person` là người.
Test: `tests/test_zone_attire.py` (lỗi tìm ra khi viết test: khung áo từng bị đếm thành người nếu không vùng nào cần áo).

**Đo trên 150 ảnh test thật `boot014test` (nhãn người gán, mỗi người xét như đang ở trong vùng, từng ảnh riêng):**
người KHÔNG mặc áo thật 354 -> báo đúng 150, bỏ sót 204 (recall vi phạm 0,42); người CÓ áo thật 170 -> báo nhầm 1
(precision 0,99). Ngoài ra sót 120 người, 104 khung người thừa.

**Vì sao:** chính lỗi #34 — model gọi áo trắng/đồng phục là `lab_coat`, nên người không mặc áo bị coi là có áo. Với
an toàn đây là chiều NGUY HIỂM (bỏ sót vi phạm), không phải chiều phiền (báo nhầm). Bộ nhớ 2 s làm bỏ sót nặng thêm khi
chạy live (một khung nhầm áo cho qua 2 s). Không chỉnh được bằng ngưỡng (đã thử, #34). Cách sửa: dữ liệu âm tính áo trắng
(lô `boot014_b2` chọn đúng các ảnh này), sau đó đo lại bằng đúng phép đo này.

**Chưa làm:** xuất RKNN `lab_v2` + parity, chạy live trên board; vùng vẽ theo góc camera boot014, đổi góc phải vẽ lại.

**Cập nhật 09/10 (`lab_v3`, +250 ảnh khó `boot014_b2`, 941 ảnh train, cùng tập test):** lab_coat precision 0,52 -> 0,76,
AP50 0,80 -> 0,82, recall 0,91 -> 0,88; luật vùng cấm bắt người không áo 42% -> 57% (207/364), vẫn báo nhầm 1/171 người
có áo. Áo thun trắng (N1L_withperson) vẫn 19/20 -> lô khó giúp thật nhưng chưa đủ. Trọng số `/mnt/big/AGV_ANLEE/weights/lab_v3_best.pt`.
Lưu ý đo: `model.val()` chạy trong heredoc có lúc treo ở dataloader worker (0% CPU) -> dùng `workers=0`.

**Lỗi tìm ra khi xem ảnh (09/10):** người bị mép dưới khung hình cắt ngang có chân y == chiều cao khung, nằm đúng trên
cạnh vùng vẽ tới 1,0 -> ray casting coi là NGOÀI vùng, nên người đứng gần camera không bao giờ bị xét. Sửa trong
`Zone.contains_foot` (kéo điểm vào trong khung), test `test_person_cut_off_by_bottom_of_frame_...`. Đo lại:
`lab_v2` bắt 53% (187/354), báo nhầm 3/170; `lab_v3` bắt **66%** (240/364), báo nhầm **4/171**. Các số 42%/57% ở trên
bị lỗi này làm thấp đi. Vùng `san_phong` thật vẽ tới 1,0 nên lỗi này cũng có trên board.

**Ngưỡng tin cậy riêng cho áo (`zones.attire_min_conf`, 09/10), `lab_v3`, cùng 150 ảnh test:**

| ngưỡng áo | bắt người không áo | báo nhầm người có áo |
|---|---|---|
| 0,25 | 57% | 2% |
| 0,40 | 66% | 2% |
| 0,50 | 72% | 2% |
| 0,60 | 78% | 4% |
| 0,70 | 82% | 5% |
| 0,80 | 89% | 15% |

Khuyến nghị 0,6–0,7. CẢNH BÁO: ngưỡng chọn trên chính tập test -> số này lạc quan; phải xác nhận trên một phiên khác
(chưa có tập val riêng). Khác với #34 (ngưỡng chung không giúp): ở đây ngưỡng chỉ áp cho khung áo trong luật vùng.

**`lab_v4` (10/10, +250 ảnh khó `boot014_b3`, 1.191 ảnh train) KHÔNG tốt hơn `lab_v3`** (cùng 150 ảnh test, `scripts/eval/eval_zone_rule.py`):
lab_coat precision 0,76 -> 0,69, AP50 0,82 -> 0,83; luật vùng ở ngưỡng áo 0,6 bắt 77% -> 69%, báo nhầm 4% -> 4%; áo thun 19/20 -> 20/20.
Lô khó thứ 2 lấy từ CÙNG phiên, cùng người, cùng áo -> lợi ích đã bão hòa (lô khó 1 cho +0,24 precision). Chưa đo độ dao
động giữa các lần train (seed): với 176 áo trong tập test, chênh vài điểm có thể là nhiễu. Giữ `lab_v3` cho board.

**Độ dao động giữa các lần train (10/10, seed 0/1/2, cùng dữ liệu, cùng 150 ảnh test, log `runs/logs/seed_eval.txt`):**

| | lab_coat precision | lab_coat AP50 | bắt người không áo @0,6 | báo nhầm @0,6 | bắt @0,7 | báo nhầm @0,7 |
|---|---|---|---|---|---|---|
| lab_v3 (941 ảnh) | 0,72 (0,68–0,76) | 0,83 (0,82–0,85) | 75% (72–77) | 5% (4–6) | 82% (80–83) | 7% (5–9) |
| lab_v4 (1.191 ảnh) | 0,70 (0,66–0,76) | 0,85 (0,83–0,89) | 71% (69–74) | 4% (4–5) | 78% (75–82) | 6% (6–6) |

Kết luận: chỉ đổi seed đã lệch ~±4 điểm; hai bộ dữ liệu NGANG NHAU trong phạm vi nhiễu (v4 bắt kém ~4 điểm nhưng báo
nhầm ít hơn ~1 điểm — một điểm khác trên cùng đường đánh đổi). Lô khó thứ 2 cùng phiên không đem lại gì đo được.
Áo thun trắng: 19–20/20 ở MỌI lần -> lỗi này là do thiếu dữ liệu, không phải do may rủi khi train.
Lưu ý: `lab_v3` seed 0 là lần tốt nhất trong 3 -> chọn nó theo tập test là lạc quan; con số nên báo cáo là TRUNG BÌNH.
Từ nay so sánh hai cấu hình phải dùng ≥3 seed, chênh dưới ~4 điểm thì không kết luận.

**`lab_v3` trên board (12/10):** RKNN INT8 (`lab_v3_best_int8_640.rknn`, calibration = ảnh train lab) qua cổng parity với ONNX
trên 40 ảnh test: PASS, det_ratio 1,06, recall 0,92, precision 0,87, IoU khớp 0,82, lệch điểm 0,025 (`runs/parity/lab_v3_int8`).
Chạy hệ thống đầy đủ (`configs/live/dashboard_lab_board.yaml`, camera thật, 60 s): 27,1 FPS, RAM 298 MB, trang web phục vụ
được, có cảnh báo + ảnh vi phạm ghi trên board. LƯU Ý: camera lúc đó KHÔNG đặt ở góc lab (chĩa vào bàn/chân người), nên
số người/cảnh báo trong lần chạy này vô nghĩa; chưa đo độ chính xác trên board với đúng góc lab. Parity trên 40 ảnh
là bộ kiểm nhỏ, vẫn chưa phải đo accuracy (#24).

## 36. `pkill -f <pattern>` chạy qua ssh tự giết shell của chính nó -> `make start-capture` không xóa công tắc tắt

**Triệu chứng (tìm ra 12/10, trước ngày thu 2):** `ssh host 'pkill -f camera_view.py; rm -f ~/AI_Challenge/.no_autocapture'`
trả mã 255, lệnh `rm` không bao giờ chạy. sshd chạy chuỗi lệnh trong `bash -c '...'`, dòng lệnh của bash đó CHỨA
`camera_view.py`, `pkill -f` khớp và giết nó (pkill chỉ tự loại chính nó, không loại tiến trình cha).
**Hậu quả:** sau `make dashboard` / `make collect-day2` (hai lệnh này tạo `.no_autocapture` để giữ camera), `make
start-capture` khởi động lại dịch vụ nhưng `autocapture_boot.sh` thoát ngay ("disabled") -> board KHÔNG chụp gì cả ngày.
Cũng là lỗi tôi gặp 2 lần trong phiên (`pkill -f "labsafe.app"`, `pkill -f soak_board.sh` giết chính lệnh của tôi).
**Sửa:** mẫu `"[c]amera_view.py"` (regex vẫn khớp tên thật, nhưng chuỗi lệnh chứa `[c]` nên không tự khớp). Đã kiểm trên
board: lệnh sau `pkill` chạy tiếp. Quy tắc: mọi `pkill -f`/`pgrep -f` gửi qua ssh hay `bash -c` phải dùng mẹo `[x]`.

**Kèm theo:** `scripts/board/sync_board_clock.sh` đặt giờ board theo laptop (board không pin RTC, không internet: lệch 10 ngày),
gọi trong `make camera`, `make dashboard`, `make collect-day2`, `make start-capture`.

## 37. Chạy dài trên board (12/10): cảnh báo lặp, danh sách thống kê không giới hạn, thử librknnrt 2.3.2

Bài thử `scripts/board/soak_board.sh` (hệ thống đầy đủ `configs/live/dashboard_lab_board.yaml`, camera thật, một client MJPEG
mở suốt, mẫu mỗi 30 s; log `runs/board/soak/`). Camera KHÔNG ở góc lab (chĩa vào bàn) -> số người/cảnh báo chỉ để so
tương đối giữa các lượt, không phải accuracy.

**A. Cảnh báo lặp (đã sửa):** 122 cảnh báo/14,5 phút; một người ngồi lì bị báo 39 lần (mỗi 10 s cooldown), mỗi lần mất
dấu vài khung (max_age 1,5 s) lại thành người mới (24 id cho 2-4 người). Sửa trong config board: `cooldown_s` 60,
`tracker.max_age_s` 3 (tracker chỉ trả người thấy ở khung hiện tại nên không đếm sai). Đo lại: 24 cảnh báo/9,5 phút,
10 id. Bảng điều khiển vẫn hiện tức thời (không chờ cooldown).

**B. Danh sách thống kê không giới hạn (đã sửa):** mỗi khung thêm ~7 số (độ trễ, đọc camera) + mỗi cảnh báo một dict,
không bao giờ xóa; `Debouncer._last_fired` giữ một khóa cho mỗi người từng xuất hiện. Benchmark cần đủ mẫu nên giữ
nguyên; chạy live giờ giữ cửa sổ 18000 mẫu (~10 phút) và 1000 sự kiện cuối (`stats_window`), debouncer quên người
hết cooldown. 3 test mới.

**C. librknnrt 2.3.2 (thay 2.3.0, khớp toolkit export 2.3.2):** `rknnlite` nạp CỨNG `/usr/lib/librknnrt.so` (bỏ qua
LD_LIBRARY_PATH; LD_PRELOAD nạp cả hai bản và vẫn dùng 2.3.0) -> phải chép đè. Bản cũ sao lưu ở
`/usr/lib/librknnrt.so.2.3.0.bak` (khôi phục: `sudo cp /usr/lib/librknnrt.so.2.3.0.bak /usr/lib/librknnrt.so`).
Parity `lab_v3` INT8 giống hệt từng chữ số; RAM tăng 1,89 (2.3.0) vs 1,81 MB/phút (2.3.2): KHÔNG sửa được rò RAM.
Giữ 2.3.2 (khớp phiên bản export). File tải từ github.com/airockchip/rknn-toolkit2 tag v2.3.2, `third_party/` (gitignored).

**Số đo các lượt (cùng board, 640x480, INT8, async C):** FPS trung bình 29,0-29,4 (camera là trần), nhiệt độ tối đa
47-49 °C, RAM tiến trình 155-201 MB.

**Lượt 30 phút sau khi giới hạn danh sách (12/10, librknnrt 2.3.2, giờ board đã đặt đúng):** RAM 159 -> 213 MB; độ dốc
0-10 phút 3,29 MB/phút (khởi động + cửa sổ đang đầy), **15-30 phút 1,15 MB/phút**; 29,0 FPS; tối đa 49,9 °C; 51 cảnh báo.
Chưa có lượt 30 phút KHÔNG giới hạn cùng cấu hình để so trực tiếp, nên không tách được phần do danh sách và phần do
khởi động. Phần còn lại (~1,1 MB/phút ≈ 70 MB/giờ, ~0,6 GB cho 8 giờ trên 7,9 GB) là heap native chưa rõ nguồn (#31):
chấp nhận được cho một ngày; vẫn khuyến nghị khởi động lại app mỗi ngày.

## 38. Vùng nguy hiểm chỉ vẽ được trong file -> đổi góc camera là vùng lệch, phải sửa file và chạy lại (đã sửa: vẽ ngay trên trang web)

**Triệu chứng (13/10):** camera đổi sang khu vực khác nhưng `configs/zones_lab.yaml` vẫn vẽ theo cảnh cũ -> vùng đè lên
tủ/bàn, người trên sàn thật không nằm trong vùng (hoặc bị xét nhầm).
**Sửa:** nút "Vẽ lại vùng nguy hiểm" trên trang web: bấm các điểm trên video đang chạy, "Chốt vùng", "Lưu và áp dụng"; áp dụng
ngay ở khung kế tiếp, không khởi động lại. Lưu ở `data/zones_live.yaml` (RIÊNG với file trong repo, vì `make deploy-board`
chép đè `configs/`); khởi động lại vẫn dùng bản đã vẽ; nút "Vùng mặc định" xóa file này. Kiểm tra mọi đầu vào từ trình duyệt
(`validate_zone_dicts`: tên, mức độ, 3-60 điểm số trong 0..1, `requires` chỉ lớp stage A thật sự xuất ra, tối đa 8 vùng).
POST chỉ nhận từ chính trang này: `Content-Type: application/json` + header `X-Requested-With` + Host là localhost
(trang web lạ không gửi được nếu không qua CORS preflight bị từ chối; chống DNS rebinding). Thử thật trên trình duyệt:
bấm 4 điểm -> `/zones` trả đúng tọa độ, vùng hiện trên video, khởi động lại vẫn còn, reset về mặc định, POST thiếu header -> 403.
Lỗi tìm ra khi thử: `pkill -f "[x]..."` vẫn tự giết shell nếu CHÍNH dòng lệnh chứa chuỗi đó ở chỗ khác (#36): dùng `fuser -k PORT/tcp`.

**Góc camera thứ 2 + `lab_v5` (14/10).** `boot018` (13/10) là góc KHÁC `boot014` (nhìn rộng, sàn lớn). Tập test riêng
`boot018test` (149 ảnh, 1.080 người, 969 áo, chỉ ~80 người không áo). `lab_v5` = boot006 + boot014 + 209 ảnh boot018
(1.400 ảnh). 3 seed mỗi bên, ngưỡng áo 0,6, log `runs/logs/eval_v5.txt`:

| | góc cũ: bắt không-áo | góc cũ: báo nhầm | góc cũ: áo AP50 | góc mới: người AP50 | góc mới: bắt không-áo | góc mới: báo nhầm |
|---|---|---|---|---|---|---|
| lab_v3 (chưa thấy góc mới) | 75% (72-77) | 5% | 0,83 | 0,94 (0,93-0,95) | 87% (85-88) | 9% |
| lab_v5 | 78% (75-82) | 4% | 0,87 | **0,96 (0,96-0,96)** | 90% (89-91) | 10% |

- `lab_v3` CHƯA học góc mới mà vẫn tốt ở đó (áo AP50 0,94): model không gãy khi đổi góc vừa phải.
- `lab_v5` ≥ `lab_v3` ở mọi cột; chắc chắn nhất là bắt người ở góc mới (+2, ba seed không chồng nhau); các chênh khác
  nằm trong nhiễu ±4 điểm. Không làm kém góc cũ.
- Áo thun trắng vẫn 20/20 ở mọi lần: lỗi dữ liệu chưa sửa được (boot018 ít người áo trắng không phải blouse).
- Góc mới nhiều người mặc áo -> báo nhầm ~10% ở ngưỡng 0,6 (ở 0,4: ~4%); ngưỡng phải chọn theo chỗ đặt camera.
Dùng `lab_v5` (seed 0) cho board: RKNN INT8 đã xuất (`runs/export/lab_v5_best_int8_640.rknn`), CHƯA chạy parity trên board
(board offline lúc xuất); `make dashboard` đã trỏ sang lab_v5 -> phải chạy parity trước khi tin số của board.

## 39. Adaptive vs parallel trên board: với bộ điều xung mặc định, adaptive TRÔNG chậm hơn — do DVFS, không phải do thuật toán

**Đo 15/10** (`configs/exp/L3_lab_v5_full_async_int8_640.yaml`: A = lab_v5 INT8 + luật vùng, B = kiểm tra trên crop người,
C = lửa/khói; async, NPU nhân 0/1/2; cùng video `data/lab_replay.mp4` 1.800 khung, occupancy 0,95; mỗi lượt ~1 phút;
log `runs/board/L3/`). "NPU-ms/khung" = tổng (số lần gọi mỗi stage / khung × thời gian mỗi lần gọi).

| Xung nhịp | Chế độ | FPS | trễ p50 / p99 (ms) | A mỗi lần | NPU-ms/khung | CPU | T max |
|---|---|---|---|---|---|---|---|
| mặc định (ondemand), 2 lượt | adaptive | 31,6 | 27,0 / 45,7 | 30,1 ms | 31,6 | 38% | 36-42 °C |
| mặc định, 2 lượt | parallel | 37,0 | 25,1 / 35,4 | 25,5 ms | 66,2 | 80% | 45-47 °C |
| chỉ NPU cố định 1 GHz | adaptive | 32,7 | 26,8 / 45,3 | 29,3 ms | 30,7 | 37% | 40,7 °C |
| chỉ NPU cố định 1 GHz | parallel | 36,6 | 25,5 / 37,2 | 25,8 ms | 67,1 | 80% | 48,1 °C |
| **CPU + DDR + NPU cố định max** | **adaptive** | **42,0** | **23,8 / 24,9** | 23,3 ms | **24,0** | **23%** | 45,3 °C |
| **CPU + DDR + NPU cố định max** | **parallel** | 40,1 | 24,5 / 25,9 | 23,7 ms | 60,9 | 58% | 51,8 °C |

**Đọc:** adaptive gọi B 0,04-0,06 lần/khung và C 0,02-0,03 lần/khung (parallel: 2,4-2,6 và 1,0) -> **ít hơn ~2,5 lần công
NPU và ~2,5 lần CPU**. Nhưng với bộ điều xung mặc định, tải nhẹ làm CPU và **bộ nhớ DDR** (`dmc_ondemand`, 528 MHz khi
nhàn, 2.112 MHz khi max) hạ xung, nên MỖI lần suy luận chậm hơn (A 30 ms so với 25 ms) và adaptive trông như chậm hơn
(31,6 so với 37 FPS). Cố định chỉ NPU không đủ; cố định cả CPU + DDR + NPU thì adaptive nhanh hơn (42 so với 40 FPS) và
đuôi trễ thấp hơn. Bài học đo lường: **so sánh tốc độ giữa hai chế độ tải khác nhau phải cố định xung** (hoặc đo năng
lượng); FPS dưới ondemand đo "bộ điều xung", không đo thuật toán. Việc xung hạ khi tải nhẹ có thể chính là nơi
adaptive tiết kiệm điện — chưa chứng minh được khi chưa có đồng hồ đo công suất.
Ghi chú: ở parallel + async, hàng đợi B bỏ ~1/3 job (~2.150/6.600) vì không theo kịp -> parallel thật sự còn tốn hơn bảng.
Lượt 1 phút: nhiệt độ chưa bão hòa, chỉ dùng để so tương đối. Bộ điều xung đã trả về mặc định sau khi đo.

**Đo dài 15/10 (đúng giao thức: 15 phút/lượt, khởi động 30 s, 3 lượt mỗi chế độ xen kẽ, bộ điều xung mặc định, video phát
đúng 30 FPS và lặp `--replay`; 2 lần cúp điện, lượt dở bị bỏ và chạy lại; log `runs/board/L3long/`):**

| (TB 3 lượt, khoảng min-max) | adaptive | parallel |
|---|---|---|
| FPS | 28,2 (28,0-28,3) | 29,8 (29,8-29,9) |
| trễ p50 / p99 | 26,9 / 45,5 ms | 25,6 / 36,4 ms |
| gọi B / C mỗi khung | 0,07 / 0,03 | 3,1 / 1,0 |
| công NPU | **32,2 ms/khung** | 74,3 ms/khung (x2,3) |
| CPU tiến trình | **35%** | 73% (x2,1) |
| **nhiệt độ SoC 5 phút cuối (bão hòa)** | **41,5 °C (41,4-41,6)** | **54,3 °C (53,3-55,0)** |

Ba lượt mỗi bên gần như trùng nhau -> khác biệt là thật. **Chênh 12,8 °C ở trạng thái ổn định** là bằng chứng gián tiếp
(nhiệt = điện năng tỏa ra) rằng adaptive tiêu thụ ít điện hơn; số watt vẫn cần đồng hồ đo. Cái giá: dưới bộ điều xung
mặc định adaptive bỏ ~6% khung (28,2 so với 30 FPS) và đuôi trễ dài hơn (p99 45 so với 36 ms) vì DDR/CPU hạ xung khi
tải nhẹ (A 30,5 ms so với 25,8 ms mỗi lần). Hướng thử tiếp: chỉ khóa xung DDR (`dmc` performance) cho adaptive rồi đo
lại cả nhiệt độ lẫn FPS — khi có đồng hồ đo công suất thì quyết định được đánh đổi này.

## 40. Lỗi "áo thun trắng = áo lab" sửa được bằng 238 ảnh COCO người mặc áo trắng (âm tính), không hại điểm ở lab

**Cách làm (15/10):** `scripts/data/coco_negatives.py` lấy từ COCO val2017 những ảnh có người (không có vùng đám đông, chú thích
không nhắc nghề mặc áo choàng: doctor, nurse, lab, chef...) mà thân trên ≥45% điểm ảnh trắng/sáng (trong mặt nạ người);
ghi tất cả người là `person`, KHÔNG có `lab_coat`, phiên `cocow` (chỉ để train). 2.693 ảnh có người -> 238 ảnh, 1.072 người.
Xem 60 mẫu: áo thun, sơ mi, đồ tennis, vài vest trắng; không thấy áo lab. `lab_v6` = dữ liệu lab_v5 + cocow (1.638 ảnh).

**Kết quả (3 seed mỗi bên, log `runs/logs/eval_v6.txt`):**

| | áo thun trắng bị gọi áo lab | góc 1: bắt không-áo @0,6 | góc 1: báo nhầm | góc 2: bắt | góc 2: báo nhầm | áo AP50 góc 1 / 2 |
|---|---|---|---|---|---|---|
| lab_v5 | **20/20, 20/20, 20/20** | 78% | 4% | 90% | 10% | 0,87 / 0,94 |
| lab_v6 | **0/20, 2/20, 4/20** | 79% | 7% | 88% | 10% | 0,87 / 0,94 |

- Lỗi tồn tại qua mọi model từ 03/10 (#34) giảm từ 20/20 xuống 0-4/20. Điểm ở hai tập test lab không đổi trong mức nhiễu
  ±4; báo nhầm góc 1 nhích lên ~3 điểm (chưa chắc là thật).
- Bài áo thun chỉ có 20 khung từ MỘT phiên/người: bằng chứng mạnh về chiều hướng, yếu về con số. Việc sửa lỗi này KHÔNG
  làm tăng tỉ lệ bắt người không áo ở hai tập test lab -> các lần bỏ sót ở đó phần lớn là lỗi khác (người che nhau,
  khung gộp hai người, áo bị che), không phải áo thun trắng.
- Trái với #25 (ảnh công trường làm kém model), ở đây ảnh ngoài chỉ thêm ÂM TÍNH cho một lớp và người vẫn được gán
  `person` -> không hại khả năng bắt người (AP50 người 0,82-0,83 / 0,96-0,97 như cũ).
RKNN INT8 `runs/export/lab_v6_best_int8_640.rknn`: parity trên board PASS (78 ảnh: recall 0,98, precision 0,93, IoU 0,84).
`make dashboard` và demo đã chuyển sang lab_v6.

## 41. Đo năng lượng lần đầu sai: đồng hồ đếm cả lúc board nghỉ chờ người gõ số (lỗi script của tôi)

**15/10:** `make energy` bản đầu chạy board 15 phút rồi mới hỏi số mWh kết thúc, và chia mWh cho đúng 15 phút. Người
đo trả lời chậm 95 phút sau lượt adaptive (board nghỉ suốt lúc đó, đồng hồ vẫn đếm) -> adaptive ra "6,8 W", cao hơn
parallel 3,5 W trong khi board kết thúc lượt adaptive chỉ 32 °C. Lượt nghỉ cũng lệch ~8 phút. Cả bộ số bị loại
(`runs/energy/20261015-132252_INVALID_late_readings`).
**Sửa:** tải chạy TIẾP cho tới khi gõ số kết thúc; thời gian tính giữa hai lần gõ; thêm số W tức thời giữa chừng để đối
chiếu. Khởi động tải bằng PID (`echo $$; exec`), dừng bằng SIGINT theo PID, không theo tên (#36). Hai bẫy gặp khi thử:
`$!` của `setsid` không phải PID của app (setsid fork); `ssh -f` không khởi động được job, còn ssh thường bị treo vì
sshd giữ phiên khi còn job nền -> cắt ssh sau 8 s rồi kiểm tra bằng PID. Đã chạy thử trên board: app lên, dừng gọn sau
SIGINT, có summary.json, không Traceback.

## 42. INT8 mất ~19 điểm mAP50-95 (khung kém khít) vì một tensor đầu ra dùng chung 1 thang 8 bit -> tách đầu ra giảm còn ~1 điểm

**Phát hiện (16/10):** parity và mọi báo cáo trước đo "INT8 khớp FP32" (recall 0,98) nhưng KHÔNG đo độ chính xác so với nhãn tay.
`scripts/eval/eval_ap.py` (một bộ chấm chung cho mọi chính xác, chạy qua đúng backend triển khai) trên 2 tập test khác phiên:

| YOLO26n `lab_v6` seed 0 | Góc 1 mAP50 / mAP50-95 | Góc 2 mAP50 / mAP50-95 | trễ/ảnh (NPU lõi 0) |
|---|---|---|---|
| FP32 (PyTorch/ONNX) | 0,864 / 0,661 | 0,976 / 0,791 | --- |
| FP16 trên NPU | 0,864 / 0,663 | 0,976 / 0,789 | 52 ms |
| **INT8 1 tensor (Ultralytics, đang dùng)** | 0,828 / **0,481** | 0,950 / **0,590** | 32 ms |
| INT8 1 tensor, hiệu chuẩn 300 ảnh | 0,847 / 0,482 | 0,951 / 0,585 | 31 ms |
| INT8 `kl_divergence` / `hybrid1` / `auto` | 0,836-0,848 / 0,477-0,483 | 0,953-0,957 / 0,591-0,596 | 30-50 ms |
| **INT8 đầu ra TÁCH, 640** | **0,862 / 0,652** | **0,975 / 0,779** | 33 ms |
| **INT8 đầu ra TÁCH, 512** | **0,876 / 0,650** | **0,969 / 0,765** | 20-22 ms |

- **Nguyên nhân (đã xác minh):** FP16 không mất gì -> mất do lượng tử hóa INT8. Mã Ultralytics ghi rõ chia tọa độ khung cho imgsz
  "để thang INT8 trên từng tensor giữ được điểm lớp" -> khung và điểm lớp dùng CHUNG một thang 8 bit: mỗi cạnh khung chỉ biết
  tới ~1/255 ảnh (~2,7 px). Đo: lệch cạnh trung bình INT8 so với FP32 là **5,3 px** (trung vị 4,4; IoU 0,84).
- **Các núm chỉnh bộ lượng tử hóa KHÔNG sửa được:** thuật toán KL, hybrid 1, hybrid tự động, số ảnh hiệu chuẩn 300 so với 1.400
  đều ra cùng ~0,48 mAP50-95 (hybrid còn chậm hơn 30-60%). Hiệu chuẩn 300 ảnh đủ (xuất nhanh gấp 5).
- **Sửa:** `scripts/export/export_split.py` xuất 6 tensor thô (mỗi scale: khung ltrb theo đơn vị stride + xác suất lớp), CPU tính
  `anchor ± khoảng cách` (`postprocess.assemble_split_outputs`, backend ONNX/RKNN tự nhận khi có 6 đầu ra, thẻ `box_coords: pixels`).
  Khoảng giá trị khung mỗi scale chỉ ~12-15 đơn vị stride nên bước 8 bit chỉ 0,5-2 px. FP32 split giải mã trùng khít bản gốc
  (328 khung, lệch 0,0000 px). Mất mAP50 còn **0,2 điểm**, mAP50-95 còn **0,9-1,2 điểm** so với FP32.
- **Độ phân giải:** (FP32, 1 seed) 640/512/416 cho mAP50 ngang nhau (0,864-0,877) và recall người xa 0,80/0,80/0,78 (khoảng tin cậy
  chồng nhau). Trên NPU 512 giảm trễ từ ~31 xuống ~20 ms. Cần 3 seed để khẳng định.
- **Cảnh báo:** các con số độ chính xác trong báo cáo/`docs/results.md` §5 là FP32; bản INT8 cũ thấp hơn 3-5 điểm AP50.

**Cập nhật cuối #42 (16/10, đã triển khai):** đã đo 3 seed x 3 độ phân giải trên NPU (`runs/split_eval/`, bảng ở `docs/results.md` 5d). INT8 đầu ra tách 640:
mAP50-95 0,657 / 0,784 (FP32 0,661 / 0,791). 416 kém rõ; 512 ngang 640 về mAP50, kém ~1,4 điểm mAP50-95 ở góc 2. Độ trễ cùng một model dao động 5-7 ms
(bộ điều tốc xung NPU/DDR hạ xung khi tải nhẹ; board không có sudo không mật khẩu để khóa xung) nên CHƯA kết luận 512 nhanh hơn 640.
Hệ thống đã chuyển sang `runs/export/lab_v6_split_int8_640.rknn` (parity PASS: recall 0,983, precision 0,998, IoU 0,967); luật vùng với INT8 thật khớp FP32.

## 43. Một khung vắng sự kiện làm hết "tín dụng" chống rung 3 s — cảnh báo không bao giờ nổ (16/10)

**Triệu chứng:** chạy hệ thống trên video phát lại (`--replay`), người vi phạm ở trong vùng 60 s liền, dashboard hiện sự kiện đang hoạt động nhưng
lịch sử cảnh báo trống, `evidence_saved` = 0. Model cũ (một tensor) và mới (đầu ra tách) giống hệt nhau, nên không phải do model.
**Nguyên nhân:** `Debouncer` xóa mốc "thấy lần đầu" ngay khi sự kiện vắng dù chỉ 1 khung. Video ngắn lặp lại làm người nhảy vị trí mỗi 2-4 s,
sự kiện vắng một khung, cửa sổ 3 s khởi động lại mãi. Trên camera thật cùng cơ chế có thể xảy ra khi điểm áo tụt dưới 0,6 hoặc model bỏ sót một khung.
**Sửa:** `Debouncer(grace_s=...)` (mặc định 0 = hành vi cũ): sự kiện vắng ngắn hơn `grace_s` giữ nguyên tín dụng. Cấu hình `alerts.grace_s: 1.0` ở
`configs/live/dashboard_lab_board.yaml`. Test `test_grace_keeps_credit_over_a_short_dropout_but_not_a_long_one`. Kiểm trên board: cảnh báo nổ sau 3 s, có ảnh bằng chứng, cooldown 60 s giữ.
**Chưa đo:** ảnh hưởng lên số cảnh báo giả/độ trễ trên video thật 30 FPS (cần video từ camera).

## 44. Camera rút/treo: hệ thống phân tích mãi khung cuối và không báo gì (16/10)

`CameraSource._loop` gặp `read()` thất bại thì ngủ 5 ms rồi thử lại mãi; `read()` vẫn trả khung cuối cùng. Rút cáp camera giữa chừng thì người dùng thấy
trang vẫn "sống" với người cuối cùng thấy, không cảnh báo, không lỗi. **Sửa:** `CameraSource(stall_s=3.0)`: sau 3 s không có khung, luồng bắt hình mở lại thiết bị
(`reopens` đếm số lần, log WARNING) và `read()` trả "không có khung" thay vì khung cũ; cắm lại thì tự chạy tiếp. Test `tests/test_camera_stall.py` (thiết bị giả).
**Chưa kiểm trên board thật bằng cách rút cáp** (chỉ kiểm không hồi quy: 30 FPS, 0 lần mở lại).

## 45. Dashboard vẽ và nén JPEG ở MỌI khung, vòng lặp đơn luồng rớt dưới 30 FPS (16/10)

Dịch vụ live (camera thật, nhiều người di chuyển) chỉ đạt 24-25 FPS dù đã khóa DDR (replay không dashboard: 29,0 FPS); CPU của tiến trình 74% so với ~50% ở bench.
Vòng lặp `step()` vẽ khung (`draw`) và `imencode` JPEG cho trang web mỗi khung, vài ms, trong cùng luồng với suy luận. **Sửa:** `dashboard.video_fps` (mặc định 15, 0 = mọi khung):
hình trên trang chỉ vẽ/nén tối đa ngần ấy lần mỗi giây; trạng thái, sự kiện, cảnh báo vẫn cập nhật mọi khung; khung có cảnh báo, hiển thị, ghi video, lưu mẫu luôn được vẽ.
`Dashboard.publish(frame=None, ...)` chỉ cập nhật trạng thái. **Chưa đo hiệu quả trên camera thật** (cần khởi động lại dịch vụ, cần sudo).
**Đã đo (16/10, docs/results.md 6d):** replay + dashboard, DDR khóa: `video_fps` 0 -> 15 làm FPS 24,0 -> 25,9 (+8%), CPU 62,8% -> 56,4%. Vẫn dưới 29 FPS của bench không dashboard; phần còn lại chưa profile.

## 46. Restart theo `RuntimeMaxSec` tốn 3 lần khởi động; nhật ký journald quá ngắn để biết vì sao (17/10)

Qua đêm: ở mốc 6 giờ dịch vụ tự restart đúng, nhưng `NRestarts` 0 -> 3 trong ~30 s (đủ 5 lần/5 phút thì systemd bỏ cuộc). Journal của board chỉ giữ ~40 dòng cuối, nên không đọc được lỗi. Nghi: camera hoặc cổng 8000 chưa được trả lại.
**Đã làm:** `CameraSource(open_retry_s=20)` thử mở camera lại tới 20 s (test `test_open_is_retried_until_the_device_appears`). **Nguyên nhân mất nhật ký (đã tìm ra 17/10):** `/var/log` của board là zram 188 MB, journal chỉ ~18 MB, và `hostapd` lặp lỗi "Failed to set up interface" ~80 dòng/phút
(cùng ~5.600 dòng của systemd) nên đẩy hết dòng của dịch vụ ra ngoài. **Đã làm:** dịch vụ ghi log riêng `runs/labsafe-live.log` (`StandardOutput=append:`); `RuntimeMaxSec` 24h (RAM đã phẳng, docs/results.md 6e).
**Chưa làm / hỏi người dùng:** tắt `hostapd` (đang lặp lỗi, có thể đang không dùng); kiểm restart thật trên board.

## 47. Người chỉ lộ cái đầu ở mép dưới khung bị tính là đứng trong vùng (17/10)

Bản vá #35 (điểm chân bị cắt ở mép dưới khung được ghim vào mép) làm một hộp chỉ chứa đầu/vai ở góc dưới khung (người điều khiển ngồi cạnh máy, camera nhìn thấy đỉnh đầu) có "chân" nằm trong đa giác nếu vùng chạm mép dưới.
Thấy trong video quay ở nhà khi `grace_s` > 0 (các lần chớp ngắn được nối thành 3 s). **Sửa:** `zones.partial_min_h` (0 = tắt): hộp chạm mép dưới và thấp hơn tỉ lệ này của chiều cao khung bị bỏ qua khi xét vùng; bật 0,3 ở config live. Test `test_head_only_box_at_the_bottom_edge_is_not_zone_checked_when_partial_min_h_is_set`.
Giá phải trả: người thật đứng sát camera chỉ lộ nửa trên thấp hơn 30% khung vẫn không bị xét vùng (chưa gặp trong dữ liệu). Chưa kiểm ở phòng lab.
**Kiểm chứng 18/10:** lần restart `RuntimeMaxSec=24h` đầu tiên với bản sửa chỉ tốn 1 lần khởi động (`NRestarts=1`), nhật ký riêng ghi đủ diễn biến, khoảng trống giám sát ~9 s (docs/results.md 6g). **#46 coi như đã sửa**; nguyên nhân gốc của 3 lần khởi động trước vẫn chưa xác minh (nhật ký lúc đó bị mất).

## 48. Dashboard mất kết nối tạo bão request/lỗi console (21/10)

**Triệu chứng:** khi board hoặc HTTP server mất kết nối, trang vẫn gọi
`/status` mỗi 0,5 s, `/events` mỗi 1 s và `/timeline` mỗi 5 s bằng ba
`setInterval` độc lập. Trong vài giây DevTools có hàng chục lỗi mạng; một request
chậm còn có thể chồng lên request kế tiếp của cùng endpoint.

**Nguyên nhân:** polling có nhịp cố định, không xét kết quả request và không có
cờ chống chồng. Trạng thái `MẤT TÍN HIỆU` đã đúng, nhưng lớp vận chuyển không
giảm tải khi server vắng mặt.

**Sửa:** thay ba interval bằng `backoffPoll(fn, base, max)`. Mỗi endpoint chỉ
có tối đa một request đang chạy; khi lỗi, nhịp tăng gấp đôi tới 5 s cho status/
events và 15 s cho timeline; lần thành công đầu tiên đưa nhịp về bình thường.
`tickAges` vẫn chạy cục bộ mỗi giây. Test
`test_page_backs_off_failed_api_polling_without_overlapping_requests` khóa cấu
trúc này. Chưa kiểm bằng trình duyệt với board thật bị ngắt kết nối.

## 49. Polygon vùng live tự cắt và hở dưới 1 pixel làm người sát camera lúc trong/lúc ngoài vùng (21/10)

**Triệu chứng:** dashboard nhận ra người không mặc áo nhưng có lúc hiện
`1 Không mặc áo lab / 0 Đang vi phạm`. Hai người áo màu ở sát camera chạm đáy
khung; banner từng báo đỏ rồi chuyển vàng "Vừa có cảnh báo, đã dừng" dù họ vẫn
đứng trong phần sàn được tô.

**Nguyên nhân đã tái hiện:** vùng lưu tại `data/zones_live.yaml` có các điểm
cuối sai thứ tự, làm nhiều cạnh tự cắt ở dải đáy. Điểm cao nhất của mép dưới
chỉ là `y=0.9994`, trong khi `Zone.contains_foot` ghim chân người bị cắt ở đáy
vào `y=0.999999`; ray casting vì vậy trả ngoài vùng. Bbox rung quanh mép làm
membership bật/tắt. Phép kiểm trực tiếp trước sửa: với x=0,08--0,80,
y=0,999999 đều `False`, còn y=0,98 là `True`.

**Sửa:** mọi polygon đi qua `load_zones`/dashboard giờ phải là đa giác đơn:
từ chối điểm lặp, diện tích bằng 0 và các cạnh không kề nhau giao nhau. Tọa độ
cách mép ảnh tối đa 1% được chuẩn hóa về đúng 0 hoặc 1, tránh một dải hở dưới
pixel do thao tác trên canvas. Polygon live hiện tại cũng đã được sắp lại điểm
và khép đáy tại `y=1.0`; bản sạch 21 điểm được đưa vào cả
`configs/zones_lab.yaml` có version control. File live bị gitignore theo thiết
kế; nếu board còn file cũ, bản code mới sẽ từ chối nó và an toàn quay về đúng
polygon sạch trong `configs/zones_lab.yaml` thay vì dùng hình tự cắt.

**Verify:** ba test hồi quy ban đầu đều đỏ, sau sửa đều xanh; đường HTTP POST
trả 400 cho đúng polygon lỗi thật. Targeted zone/pipeline/attire: 61 passed;
toàn suite: 309 passed. Chạy lại đoạn video 17 s quanh cảnh lỗi: người thiếu
áo có `zones=['san_phong']`, `violating=True`; run tạo 7 cảnh báo high trong
353 frame. Vấn đề gán nhầm áo giữa người đứng sát nhau (#35) chưa đổi vì chưa
có video có nhãn đủ để chứng minh một luật association mới tốt hơn.

**File liên quan:** `src/labsafe/pipeline/zones.py`,
`tests/test_zone_editor.py`, `tests/test_pipeline.py`, `data/zones_live.yaml`.
**Bổ sung #49 (21/10, Claude Code):** quy tắc dính mép 1% làm HAI lần bấm sát cùng một góc (thường gặp khi vẽ khép về góc) thành hai điểm trùng nhau -> cả polygon bị từ chối "repeated points".
Vùng vẽ ở nhà ngày 17/10 (`data/zones_live.yaml` trên board) bị đúng lỗi này, nên với code deploy 21/10 board âm thầm quay về vùng mặc định (chỉ có dòng WARNING trong log).
Sửa: bỏ điểm trùng LIỀN KỀ sau khi dính mép (kể cả qua cạnh khép), điểm trùng không liền kề vẫn bị từ chối. Test `test_two_clicks_snapped_into_the_same_corner_become_one_vertex`,
`test_a_vertex_repeated_far_apart_is_still_rejected`; chấm sự kiện ở nhà chạy lại cho kết quả giống hệt (7/7, 3 báo giả, trễ 5,24 s). Chưa đưa lên board.

## 50. "Không thấy áo" bị coi là "không mặc áo": báo vi phạm người chỉ lộ mặt hoặc bị che (21/10)

Người dùng thấy trong lịch sử cảnh báo của bản demo (video lab đông người) nhiều ảnh chỉ có khuôn mặt hoặc cái đầu bị gắn "Không mặc áo lab".
Kiểm 30 ảnh cảnh báo gần nhất: ~5 đúng (áo đen/xanh, thấy rõ thân), phần còn lại là mặt/đầu (31x35 đến 76x93 px), người ngồi bị người phía trước che, hoặc một dải mảnh ở mép khung;
nhiều người trong số đó đang mặc áo lab mà áo không lọt vào khung của họ. Luật cũ: thiếu khung áo gắn với người = vi phạm, bất kể có nhìn thấy quần áo hay không.
**Sửa:** `zones.visibility` (router `attire_visible`): người KHÔNG được phán áo (khung xám "Chưa thấy rõ áo", không báo `zone_ppe_violation`, vẫn tính `zone_intrusion`) khi
khung thấp hơn `min_h` x chiều cao ảnh mà không có dáng người đứng xa (h/w < `tall_aspect`), hoặc hẹp hơn `min_w_px`, hoặc bị người gần camera hơn (chân thấp hơn) che quá `max_occlusion` diện tích.
Bật {min_h 0,22, tall_aspect 2, min_w_px 24, max_occlusion 0,4} ở cả 3 config live.
**Đo:** demo `boot014_demo.mp4`, cùng 170 s: 31 cảnh báo -> 6, cả 6 là người áo đen/xanh thật sự không áo. Tập test lab (FP32 lab_v6, ngưỡng áo 0,6, người có nhãn không đủ thấy cũng bị loại khỏi chấm):
góc 1: bắt 78% -> 78%, báo nhầm 7% -> 6% (chấm 644 -> 518 người); góc 2 (đông): người mặc áo bị báo nhầm 79 -> 16 (8% -> 2%), người không áo bị bỏ sót 12/88 -> 6/18
(70 "người không áo" theo nhãn bị loại vì không thấy rõ thân: phần lớn là người mặc áo bị che, nhãn không vẽ áo vì không thấy). Chấm sự kiện ở nhà không đổi (7/7, 3 báo giả).
**Giá phải trả:** người không áo đứng sau người khác hoặc chỉ lộ đầu sẽ không bị báo cho tới khi lộ thân; chưa đo trên video có đáp án với người thật bị che (cảnh 07 kịch bản demo).

## 51. Người áo đen/xanh mang nhãn "Có áo lab": áo của khung gộp hai người + trí nhớ áo trên video tua nhanh (21/10)

Người dùng thấy trên bản demo laptop người áo xanh và người áo đen được ghi "Có áo lab". Truy vết track #35 (khung 465-472 của `boot014_demo.mp4`): model trả về MỘT khung
bao cả cô gái áo đen lẫn người mặc áo lab bên cạnh (khung 0..253 px), áo lab (0,85-0,88) được gán cho khung đó; khung tách ra ở khung 469 nhưng vẫn cùng track (IoU cao) nên
"trí nhớ áo" (2 s) giữ cô gái áo đen là "Có áo lab". Video demo là ảnh tua nhanh (1 ảnh/4 s phát 10 hình/giây): 2 s ở đây = khoảng 80 s thời gian thật.
Đã thử rồi BỎ: gán áo theo vừa thân người (torso) -> người mặc áo bị báo nhầm nhiều hơn trên tập test (góc 2: 17 -> 26/684). Giữ quy tắc chứa (containment).
**Sửa:** (1) config demo laptop `dashboard_lab_pc.yaml`: `attire_memory_s: 0.3` (đúng với video tua nhanh; board vẫn 2,0 s); (2) tracker: khi một track chỉ khớp được bằng
khoảng cách tâm (không chồng khung) thì xóa trí nhớ áo (có thể là người khác nhận id), test `test_a_coat_does_not_follow_an_id_onto_a_different_person`.
Kiểm lại 4 khung có người áo đen/xanh: cô gái áo đen thành "VI PHẠM · không áo lab", người bị che thành "Chưa thấy rõ áo".
**Còn lại trên board (30 FPS):** khung gộp hai người vẫn có thể cho người không áo "mượn" áo tối đa 2 s (trễ báo động tối đa 2 s), chưa đo trên video thật có đám đông.


## 52. Góc camera mới ở lab: người mặc áo lab trắng KHÔNG có khung `lab_coat` nào, bị gọi là "Không áo lab" (23/10)

Người dùng chụp trang web trên camera thật tại lab (board, 28 FPS): người #48 mặc áo lab trắng đứng nghiêng, camera đặt cao nhìn xuống, hiện khung cam "Không áo lab". `/status`: `coat_score: None`.
Kiểm bằng cùng model (lab_v6, ONNX FP32 `split640_s0`) ở ngưỡng 0,05 (board dùng 0,4) trên 3 khung lấy từ luồng: **person** 0,56-0,86 nhưng **không có khung `lab_coat` nào** ở bất kỳ ngưỡng nào
(chỉ 4-6 khung `person` điểm thấp). Vậy không phải do ngưỡng `attire_min_conf` 0,6 hay `conf` 0,4: model không nhận ra áo này ở góc này.
Giả thuyết (CHƯA tách riêng từng yếu tố): góc camera mới (cao, nhìn xuống, người đứng nghiêng; dữ liệu huấn luyện boot006/014/018 là các góc khác) và/hoặc áo trắng sát áo trắng của người đứng sau (khung gộp, #51).
**Hệ quả với demo:** lỗi nghiêm trọng nhất cho luật sản phẩm (người ĐÚNG bị báo vi phạm). Ngưỡng và config không cứu được vì không có khung nào để gán.
**Cách sửa đúng:** quay video ở ĐÚNG vị trí camera cuối (kịch bản `configs/collection_plan_demo.yaml`, cảnh 02/03/08), gán nhãn ~200-300 ảnh (Roboflow), huấn luyện lại, xuất RKNN INT8 split, kiểm parity trên board.
**Chưa đo:** áo có được nhận khi người đứng chính diện/thấp hơn? tỷ lệ nhận áo ở góc này (cần `scripts/eval/coat_check.py` với người đứng một mình).

**Cập nhật 23/10 (cùng ngày), kết quả đo và xác nhận của người dùng:** (1) trên 122 khung liên tiếp ở camera lab, người mặc áo: lab_v6 nhận áo 0% ở MỌI ngưỡng (cả ONNX chuẩn lẫn INT8 trên board);
lab_v5 20% ở 0,6 / 60% ở 0,3 (89% theo thời gian nếu nhớ áo 3 s), người khác bị nhầm có áo 3/336 ở 0,3; lab_v3 17% / 50%, nhầm 15/412. Ước lượng thô: 1 người áo, 1 cảnh, 8 s liên tiếp.
(2) Người dùng xác nhận: góc camera hôm nay KHÁC góc quay dữ liệu huấn luyện. Ảnh so sánh: dữ liệu train là người đứng xa, thấy trọn thân, áo trắng sáng, nền tủ gỗ; hôm nay camera rất gần, cắt mất đầu, nhìn từ trên xuống, ánh sáng xám.
(3) Đã xuất `runs/export/lab_v5_split_int8_640.rknn` (INT8 tách, 300 ảnh hiệu chuẩn, cùng định dạng thẻ model với v6; SHA khớp board) và config thử `configs/live/lab_v5_board.yaml` (`make live-config CONFIG=v5`).
CHƯA kiểm: v5 INT8 trên board (parity/`coat_check`), v5 với áo thun trắng (#34/#40 từng gọi áo thun trắng là áo lab 19-20/20).
**Kết quả sau khi đặt camera về góc cũ (23/10, board thật, config chuẩn lab_v6 + 0,6 + 2 s):** 25 s, 6 người trong khung. 4 người mặc áo lab (#130, #127, #92, #134): điểm áo thấy ở 97-100% số lần thăm dò
(0,56-0,94), "đang mặc áo" 98-100%. Người không áo lab (#102, #103, #125): điểm áo tối đa 0,55, không lần nào >= 0,6; áo thun/polo trắng (#104, #131): không có khung áo. Tức là ở góc này mô hình tách được áo lab với áo trắng thường
trong mẫu này (1 phiên, 25 s, 6 người: chưa phải độ chính xác). **Kết luận: lỗi #52 là do góc camera, không phải do mô hình hỏng; KHÔNG chuyển sang lab_v5** (v5 từng gọi áo thun trắng là áo lab).
**Còn lại (#51):** người đứng sát người mặc áo "mượn" áo của người bên cạnh: #103 (ngồi sát, không áo) được tính "mặc áo" 48% số lần, #125 (trong vùng, không áo) 32% và vì thế chưa bị báo trong 8 s quan sát.
Khi demo: để từng người vào vùng riêng, không đứng sát người mặc áo.
