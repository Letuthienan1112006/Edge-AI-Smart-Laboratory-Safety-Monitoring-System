# Experiment ID mapping — `N*` (NPU matrix) vs `E*` (old runtime matrix) vs `O*` (optimisation report)

This repo already has two experiment-ID namespaces in use before this plan:

- `E1..E5` in `configs/exp/*.yaml` and `scripts/bench_matrix.sh` — the
  original CPU-runtime comparison (torch / ncnn / tflite-int8, `--mode`
  parallel vs adaptive). `E1` there means "PyTorch baseline"; it has
  nothing to do with the user's new "E1 = RKNN FP16" request.
- `O0..O9` in `docs/optimization.md` §7 — the optimisation *narrative*
  (probe, affinity, sweep, quantise, prune, distil, NPU cores, engine
  mapping, adaptive+engine combined). These are report sections, not run
  IDs with their own config files.

The user's NPU experiment request (E0 CPU baseline … E6 knowledge
distillation) collides on symbol with the first table but means something
different. Reusing `E1..E5` for it would put two different tables both
labelled "E3" in the same final report — a silent, report-level bug.

**Decision: the new NPU matrix uses the `N` prefix (N = NPU matrix) and is
never abbreviated back to `E*`.**

## Mapping table

| `N*` id | Content | User's request | Related `O*` section |
|---|---|---|---|
| `N0` | CPU baseline, ONNX FP32, imgsz 640 | E0 | O2 (runtime sweep baseline) |
| `N1` | RKNN **FP16** on NPU, imgsz 640, `core_mask: auto` | E1 | O7 (NPU core count) |
| `N1b` | Same as N1, `core_mask: all` (1 core vs 3 cores) | — (new) | O7 |
| `N2` | RKNN **INT8** on NPU, imgsz 640 | E2 | O3/O4 (quantisation) |
| `N3a`..`N3d` | Resolution sweep: 320 / 416 / 512 / 640 (FP16) | E3 | O2 |
| `N4` | Worker/async pool sweep, 1..N workers | E4 | — (new, see `scripts/sweep_workers.py`) |
| `N5` | Adaptive vs always-on, reusing `--mode` | E5 | O9 (adaptive + engine mapping) |
| `N6` | Knowledge distillation YOLO26s → YOLO26n | E6 | O6 — **not built this round**, see plan §7 |
| `N1L` | Live smoke/evaluation on real hardware (camera + RKNN FP16), stage A only, `keep_classes: [person]` on the un-fine-tuned COCO-80 checkpoint | "first live run" request, not the original E0..E6 list | — (not in `bench_matrix_npu.sh`) |
| `N0L` | CPU control for N1L (ONNX FP32), same live-run purpose | same as N1L | — (not in `bench_matrix_npu.sh`) |
| `N2L` | INT8 counterpart of N1L (RKNN INT8, lab+coco128 calibration), identical otherwise | INT8 speed-up check, KNOWN_ISSUES.md #24 | — (not in `bench_matrix_npu.sh`) |
| `L1` | Full system A+B+C (all INT8 RKNN), stages called serially, `classes_lab_v1.yaml` | first full-system measurement, KNOWN_ISSUES.md #30 | — |
| `L2` | L1 with `async_stages: true` and A/B/C on NPU cores 0/1/2 | same | — |

`N1L`/`N0L` are **not** part of the N-matrix or `bench_matrix_npu.sh` --
they exist for the one-off "run and evaluate on the board for the first
time" request (`.bangiao/ke-hoach.md`, board-first-live-run), use
`configs/classes_coco80.yaml` instead of `configs/classes.yaml`, and disable
stage B/C entirely rather than running their stubs. Do not fold their
numbers into an N-matrix report table without saying so explicitly.

`O8` (task × engine mapping, the central contribution) is not a single run
ID: it is produced by comparing several `N*` runs against each other once
`src/labsafe/optimize/engine_map.py` has resolved each stage's engine, and
by re-running the same `N1`/`N5` configs with different `engine:` values
per stage in `configs/pipeline.yaml`.

## Rule for anyone adding a new experiment later

- If it belongs to the NPU/RKNN matrix, prefix it `N`.
- If it belongs to the old CPU-runtime matrix (torch/ncnn/tflite), keep
  using `E*` — do not renumber the existing files, that breaks
  `scripts/bench_matrix.sh` and any report that already cites them.
- Never let a report table mix `E3` (tflite INT8, 640) and `N3` (RKNN FP16
  resolution sweep item) under the same header without saying which
  matrix it is from.
