# RKNN pipeline: convert → validate → deploy

Two different machines, two different Python packages, and a gate in
between that must never be skipped. Read this before running any of
`scripts/export_rknn.py`, `scripts/validate_conversion.py`, or
`scripts/bench_matrix_npu.sh`.

## 0. Two machines, two packages — do not confuse them

| | Runs on | Package | Job |
|---|---|---|---|
| **Convert** | x86 Linux laptop | `rknn-toolkit2` | ONNX → `.rknn` |
| **Inference** | Orange Pi 5 (aarch64) | `rknn-toolkit-lite2` | load `.rknn`, run NPU |

They are **different pip packages** with independent version numbers, and
both must be compatible with the `librknnrt.so` baked into the board's
image. A version mismatch produces errors that do not name the version as
the cause — see `KNOWN_ISSUES.md` for the general shape of these
failures. `ModelCard.toolkit_version` records what converted the model so
this is traceable after the fact; it is not a substitute for checking
compatibility before converting.

`scripts/export_rknn.py` refuses to run on `aarch64` (convert is x86-only).
`scripts/validate_conversion.py --candidate-backend rknn` refuses to run
anywhere except `aarch64` (RKNN inference only happens on the board). This
means the full convert→validate cycle for a real `.rknn` file always spans
two machines and two separate commands — there is no single-command path,
and there should not be one.

## 1. Convert (x86 laptop)

```bash
python3 scripts/export_rknn.py yolo26n.pt --imgsz 640 --precision fp16 \
        --target rk3588 --out runs/export/

python3 scripts/export_rknn.py yolo26n.pt --imgsz 640 --precision int8 \
        --target rk3588 --data data/processed/person/data.yaml --out runs/export/
```

Non-negotiable properties of every export produced this way:

- **`end2end=False`, always.** YOLO26 defaults to an end-to-end (NMS-free)
  head; on RK3588 that segfaults at inference (`KNOWN_ISSUES.md` #2). The
  script forces `end2end=False` and only accepts the opposite behind an
  explicit `--i-know-this-segfaults` acknowledgement flag.
- **INT8 requires a real YOLO dataset YAML**, `--data <path/to/data.yaml>`
  (`path:`/`train:`/`val:`/`names:` — the same file every training run in
  this repo already produces), with at least 50 images in the resolved
  split (`--calib-split`, default `val`). **Not a bare directory of
  images** — ultralytics' own INT8 calibration path
  (`get_int8_calibration_dataloader` → `check_det_dataset`) requires the
  YAML and reads the split from it; a directory fails deep inside
  ultralytics with a confusing error. This differs from
  `scripts/export_models.py`'s TFLite path only in what "the calibration
  data" means for each backend — TFLite there also takes a `data.yaml`,
  so the convention is consistent, the earlier draft of this doc was not.
- **A `.card.json` sidecar is written next to the `.rknn` file**
  (`src/labsafe/models/modelcard.py`), recording the mean/std actually
  used, `input_dtype`, `imgsz`, `end2end`, `target_platform`, and both the
  `ultralytics` and toolkit versions seen at convert time. The runtime
  backend reads this file and refuses to run if its assumptions do not
  match what the model was actually converted with.
- **An ONNX FP32 reference is exported alongside it** (`--also-onnx`,
  default on) so the parity gate below has something to compare against.
  A candidate with no reference is not a validated candidate.

### mean/std and input dtype

RKNN bakes normalisation into the model at convert time. The runtime then
expects raw `uint8` NHWC input — feeding it a normalised `float32` tensor
does not error, it just produces a model that detects nothing
(`KNOWN_ISSUES.md` #4). `export_rknn.py` sets `mean_values=[[0,0,0]]`,
`std_values=[[255,255,255]]` to match `letterbox(..., normalize=False)`
in `src/labsafe/models/postprocess.py`, and writes exactly those values
into the model card — not a hand-typed default that could drift from what
was actually used.

### Version gate: functional, not numeric

We do not know precisely which `ultralytics` release first contains the
fix for issue #23340/#23312. **Do not hardcode a minimum version number
anywhere in this pipeline.** `export_rknn.py` prints the installed
`ultralytics.__version__` and the `rknn-toolkit2` version (if importable)
for traceability, and states plainly that the real gate is step 2 below —
functional parity — not a version comparison. If conversion or inference
segfaults, that is the signal to check for an ultralytics update, not to
edit a version pin in this repo.

### Verified — real export call signature, 2026-09-26

Confirmed against a real install (`ultralytics==8.4.163`,
`rknn-toolkit2==2.3.2`, x86 laptop) by reading
`ultralytics/engine/exporter.py`'s `rknn` branch directly, and then
running a real conversion end-to-end (`yolo11n.pt` → `.rknn`, 7.3 MB,
`rknn-toolkit2` actually built the model). Three things the original,
unverified draft of this script got wrong:

1. **`int8=`/`end2end=` are deprecated kwargs**, not the current API.
   Ultralytics now uses `quantize=8` (INT8) / `quantize=16` (FP16) and
   `nms=`. The deprecated names still work — `_handle_deprecation()`
   maps them onto `quantize=`/`nms=` — but print a warning on every run.
   `export_rknn.py` now calls `model.export(format="rknn", imgsz=<n>,
   name=<target>, quantize=(8 if int8 else 16), nms=False if
   end2end else <omitted>)` directly. `end2end=True` → `nms=False` is
   what actually sets `model.end2end = True` (the risky NMS-free/Top-K
   head that segfaults on RK3588, #23753); the default path omits `nms`
   entirely, which is what `end2end=False` always meant.
2. **`--data` for INT8 must be a dataset YAML, not a directory** — see
   above. `model.export(data=..., split=...)` feeds
   `check_det_dataset()` internally, exactly like the TFLite INT8 path
   elsewhere in this repo.
3. **`mean_values`/`std_values` are not export kwargs at all.** RKNN
   bakes normalisation in automatically; the two lists in
   `export_rknn.py` only *document*, for the model card, what the
   resulting `.rknn` expects (raw uint8 — see below). Nothing passes
   them to `model.export()`.

A fourth bug surfaced only by running a real export end-to-end, not by
reading source: **`_collect_export()` used `Path.replace()`, which raises
`OSError: [Errno 18] Invalid cross-device link` the moment `--out` is on a
different filesystem/mount than wherever ultralytics wrote its output**
— which is the *normal* case, since ultralytics writes beside the source
checkpoint and ignores `--out` entirely. Fixed with `shutil.move()`. See
`KNOWN_ISSUES.md` for the full writeup and the regression tests (all four
of the above have a positive-controlled test in
`tests/test_export_rknn_cli.py` — reverted the fix, confirmed the test
goes red, restored it).

`export_rknn.py` does not swallow a `TypeError` from a wrong keyword: it
re-raises with the attempted call spelled out, so if a future
`ultralytics` release changes this signature again, that shows up as a
loud crash naming the exact rejected kwargs, not a quiet no-op export.

### Verified against the real YOLO26n checkpoint, 2026-09-26

Everything above was first verified with `yolo11n.pt` (a lighter stand-in).
Re-ran both precisions against the actual `yolo26n.pt` this project targets:

- **fp16**: exit 0, real `.rknn` (7.1 MB) + ONNX reference + `.card.json`
  (`end2end: false`, `input_dtype: uint8`, mean/std as documented above).
- **int8**: exit 0 against a real dataset yaml (`coco128.yaml`, 128 real
  calibration images), real `.rknn` (4.6 MB).

A fifth bug surfaced only by using a *real* dataset yaml with the standard
ultralytics `path: <name>` convention (not a synthetic test fixture with an
absolute `path:`): `_count_calib_images()`'s own path resolution disagreed
with `check_det_dataset()`'s real resolution order and reported "0 images"
for a split that genuinely had 128. Fixed in `_resolve_dataset_root()` —
full root-cause writeup in `KNOWN_ISSUES.md` #13.

**Not yet verified by any of this:** whether the exported `.rknn` actually
*detects* anything correctly. That is exactly what the parity gate below
exists to check, and its RKNN candidate side only runs on the board. A
clean, zero-error export is necessary but not sufficient — see
`KNOWN_ISSUES.md` #3 for the documented case (ultralytics#23753) where an
INT8 RKNN export on RK3588 converts and runs fine while detecting nothing.

**Verified on the board (2026-09-29):** the first INT8 export failed the
gate in a different way than #3 — as many detections as the reference, zero
of them matching — because ultralytics' RKNN INT8 export emits box
coordinates normalised by imgsz. The card now records `box_coords` and the
decoder rescales; after that INT8 passes the gate (recall 1.0, precision
0.96, IoU 0.87) at 25 ms NPU infer vs 48 ms for FP16. Full record:
`KNOWN_ISSUES.md` #24.

## 2. Validate (parity gate) — mandatory before any number is reported

```bash
python3 scripts/validate_conversion.py \
    --reference-backend onnx    --reference-weights runs/export/yolo26n_640.onnx \
    --candidate-backend  rknn   --candidate-weights  runs/export/yolo26n_fp16_640.rknn \
    --images data/calib/ --imgsz 640 --conf 0.25 --iou 0.5 \
    --names person \
    --out runs/parity/N1_fp16
```

**No `.rknn` model's FPS, latency, or detection numbers may go into a
report until this command has been run against it and printed PASS.** A
model that converts cleanly and runs at full speed while detecting
nothing (`KNOWN_ISSUES.md` #3) looks identical to a healthy model in every
metric except this one.

- Reference and candidate are both built through `build_backend()` — no
  separate ad-hoc model-loading code path.
- Exit code **0** = PASS, **2** = FAIL (model likely broken —
  `zero_detection_collapse` is the failure to look for first), **1** =
  infrastructure error (missing backend, unreadable images, wrong host
  architecture for RKNN inference).
- `--allow-fail` still runs and still writes `passed: false` to
  `parity.json`; it only changes the process exit code to 0, for the
  specific case of deliberately collecting evidence that INT8 failed (see
  `KNOWN_ISSUES.md` #3 and the N2 experiment in `docs/experiment-ids.md`).
- RKNN inference (the candidate side, when `--candidate-backend rknn`)
  only runs on `aarch64` — i.e. on the board. Converting on x86 and
  validating on the board are two separate steps on two separate
  machines; there is no way around that with the current toolkits.

## 3. Deploy / benchmark (board)

Only after step 2 has printed PASS for a given `.rknn` file:

```bash
bash scripts/bench_matrix_npu.sh
```

The matrix script re-runs the parity gate before each `N1`/`N2` entry and
skips (not aborts) a run whose model fails parity, logging why to
`runs/skipped.log`. Infrastructure failures (exit code 1) stop the whole
matrix instead of skipping, since they mean the environment itself is
broken, not just one model.
