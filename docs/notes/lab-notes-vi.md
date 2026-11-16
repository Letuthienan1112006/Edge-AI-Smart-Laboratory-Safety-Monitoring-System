# Lab Safety Edge AI

**Heterogeneous Multi-Task Edge AI for Real-Time Laboratory Safety Monitoring**
Orange Pi 5 / RK3588S · person + lab coat + restricted zone · experimental fall + fire/smoke

> **Phạm vi đã kiểm chứng:** sản phẩm hiện tại được đánh giá đầy đủ nhất cho
> luật *người không mặc áo lab trong vùng nguy hiểm*. Té ngã và lửa/khói đã
> được tích hợp vào pipeline nhưng chưa có đánh giá thực địa đủ để công bố độ
> chính xác; kính/găng không thuộc cấu hình live. Xem
> `docs/competition/competition-claims.md` trước khi lấy số liệu làm slide hoặc thuyết trình.

Target: 4× Cortex-A76 @2.4GHz + 4× Cortex-A55 @1.8GHz + Mali-G610 GPU +
RKNPU2 (6 TOPS, 3 cores). Four compute engines, five tasks — the mapping
between them is the research question.

> Đọc theo thứ tự:
> 1. `docs/research-plan.md` — đánh giá hướng đề tài, RQ, ma trận thí nghiệm.
> 2. `docs/optimization.md` — **25% điểm "tối ưu hóa/tăng tốc"**: vì sao INT8
>    có thể làm Pi 4 CHẬM đi, vì sao pruning không cấu trúc không tăng tốc,
>    và tài liệu tham khảo cho từng kỹ thuật.
> 3. `docs/measurement-protocol.md` — giao thức đo, quyết định chất lượng số liệu.
> 4. `docs/KNOWN_ISSUES.md` — bug đã biết (YOLO26 × RKNN × RK3588S) và cách repo
>    này tránh chúng. Đọc trước khi đụng vào `models/backends.py` hay
>    `scripts/export/export_rknn.py`.
> 5. `docs/rknn-pipeline.md` — quy trình convert → validate → deploy cho
>    NPU RKNN: hai máy, hai gói, và cổng kiểm định bắt buộc trước khi một
>    con số nào của model RKNN được đưa vào báo cáo.

## Chạy TRƯỚC KHI tối ưu bất cứ thứ gì

```bash
make probe        # cluster nào, tần số bao nhiêu, có asimddp không, có NPU không
make affinity     # topology big.LITTLE + lệnh taskset
make roofline ARGS="--gflops 1.2 --weights-mb 6.2"
```

Ba lệnh này **đo** thiết bị chứ không giả định nó. Khi phần cứng đổi (đã đổi
một lần rồi: Pi 4 → Orange Pi 5), công cụ vẫn đúng còn tài liệu thì sai —
nên tin công cụ.

Với RK3588S: A76 và A55 đều có `asimddp`, nên **INT8 tăng tốc thật** (~1.8×),
khác hẳn kết luận cho Cortex-A72 trước đây. Chi tiết: `docs/optimization.md` §1.

## Chạy thử ngay (chưa cần model, chưa cần Pi)

```bash
export PYTHONPATH=src
python3 -m labsafe.app --source 0            # webcam + overlay
make demo                                    # tương đương
make test                                    # 296 test pass gần nhất, không cần camera (21/10/2026)
make bench                                   # smoke benchmark 60s
```

Mặc định mọi stage dùng `backend: stub` — một detector giả sinh ra một "người"
di chuyển. Nhờ vậy toàn bộ tracker, zone, fall, debounce, alert, dashboard và
bench harness chạy và kiểm chứng được **trước khi** có model. `sim_latency_ms`
của stub cho phép mô phỏng đúng độ trễ của Pi 4 ngay trên laptop.

## Kiến trúc

```
Camera ──> Stage A: person detector (mọi frame, model nhỏ nhất)
              │
              ├─> Tracker (IoU + centre-distance fallback)
              │      ├─> Zone check      hình học, 0 chi phí
              │      └─> Fall state      hình học + thời gian, 0 chi phí
              │
              └─> Stage B: tải thử nghiệm trên crop người
                          (tắt trong cấu hình live hiện tại)
Stage C: fire/smoke toàn khung, cadence riêng 1 Hz (đã tích hợp, chưa đánh giá trong lab)
              │
              v
       Debounce + cooldown ──> Buzzer/LED + Dashboard
```

Lý do tách tầng và tại sao **không** dùng một YOLO đa lớp: `docs/research-plan.md` §2.2.

Hai chế độ, **cùng một code path**, khác nhau đúng một flag:
- `--mode parallel` = E1, baseline: mọi stage chạy mọi frame, không gating, không
  recheck interval.
- `--mode adaptive` = E5, hệ thống đề xuất.

Đo thử với stub giả lập Pi 4 (A=55ms, B=30ms, C=40ms), occupancy 100% (tình
huống **tệ nhất** cho gating vì luôn có người trong khung):

| mode | FPS | stage B calls/frame | stage C calls/frame | p50 latency |
|---|---|---|---|---|
| parallel | 8.0 | 1.00 | 1.00 | 125 ms |
| adaptive | 17.2 | 0.03 | 0.06 | 55 ms |

Đây là số của detector giả — nó chứng minh **kiến trúc**, không phải accuracy.

## Cấu trúc

```
configs/           classes.yaml (taxonomy), pipeline.yaml, zones.yaml,
                   exp/E1..E4 (CPU runtimes), exp/N0..N5 (NPU matrix),
                   export/yolo26_rk3588.yaml (RKNN convert defaults)
src/labsafe/
  models/          backends.py (stub|ultralytics|onnx|ncnn|tflite|rknn),
                   postprocess.py, parity.py (conversion parity gate),
                   modelcard.py (convert-time <-> runtime contract)
  pipeline/        router.py (adaptive core), tracker, fall, ppe, zones,
                   decision, pool.py (worker pool, N4 sweep)
  io/              camera, alerts (GPIO), dashboard (MJPEG)
  bench/           runner.py, sysmon.py (nhiệt độ + throttle), report.py
  optimize/        arm_probe.py (ISA + big.LITTLE + NPU), affinity.py,
                   roofline.py, engine_map.py (task x engine, O8)
  datasets/        verify.py  <- chạy TRƯỚC mọi lần train
scripts/           verify_dataset, train_baseline, export_models, bench_matrix.sh
                   prune_structured.py, distill_pseudolabel.py, sweep_runtime.py
                   export_rknn.py (x86 only), validate_conversion.py (parity gate),
                   sweep_workers.py (N4), bench_matrix_npu.sh (N* matrix, board only)
tools/             zone_editor.py (click polygon -> zones.yaml)
tests/             143 test cho phần logic dễ sai âm thầm (parity, model card,
                   engine mapping, worker pool, telemetry, config ...)
```

## Quy trình

**1. Dataset (Phase 2)**
```bash
python3 scripts/data/verify_dataset.py data/processed/ppe/data.yaml --crops 8
```
Script sẽ in số nhãn theo từng lớp và dump crop thật ra `data/interim/crops/`.
**Phải mở thư mục đó ra xem.** Nếu `01_goggles/` chứa ảnh găng tay thì thứ tự
lớp đã lệch, và mọi con số sau đó vô nghĩa. Script exit code 1 nếu có vấn đề.

**2. Train trên laptop (Phase 3)**
```bash
python3 scripts/train/train_baseline.py --data .../data.yaml --name stage_a --imgsz 640 --batch 16
```
Script **từ chối train** nếu verifier báo lỗi (dùng `--force` để ghi đè).

**3. Export (Phase 4)**
```bash
python3 scripts/export/export_models.py runs/train/stage_a/weights/best.pt \
        --imgsz 320 --formats onnx,ncnn,tflite --int8 --data .../data.yaml
```
`--int8` bắt buộc có `--data`: quantize không có ảnh calibration thật sẽ tạo ra
một mức rớt accuracy giả, rồi bị viết vào bài như "INT8 mất 9% mAP".

**3b. Tối ưu (Phase 4) — theo thứ tự rẻ-trước**
```bash
make sweep BACKEND=ncnn ARGS="--weights ... --imgsz 256,320,416"   # miễn phí
python3 scripts/train/prune_structured.py --weights ... --ratio 0.4 --steps 4
python3 scripts/train/distill_pseudolabel.py --teacher yolov8m.pt --videos ...
```
`prune_structured.py` từ chối `--unstructured` và giải thích tại sao: prune
không cấu trúc không làm CPU ARM chạy nhanh hơn, nó chỉ làm file nhỏ hơn.

**4. Đo trên Pi 4 (Phase 4–5)**
```bash
SOURCE=data/processed/bench_clip.mp4 DURATION=900 bash scripts/bench/bench_matrix.sh
python3 -m labsafe.bench.report runs
```
Harness tự ghi nhiệt độ, RSS, tần số CPU và `vcgencmd get_throttled` trước/sau.
Nếu board bị throttle, `result_valid: false` và `metrics.json` ghi rõ cảnh báo —
lần đo đó **bỏ**, không được đưa vào bài.

## Cài đặt

**Board mới nhận, chưa setup gì:** làm theo `docs/board-setup.md` từ đầu —
flash OS, xác minh phần cứng, rồi `bash scripts/board/bootstrap_board.sh` cài mọi
gói cần thiết trong một lệnh. Đọc tài liệu đó, đừng chỉ chạy script — mỗi
bước có lệnh xác minh mà script không tự kiểm tra hộ được (VD: line offset
GPIO thật của board bạn).

Laptop (train + convert RKNN): `pip install -r requirements/train.txt`
(cài torch CUDA trước; `rknn-toolkit2` cũng chạy ở đây, không chạy trên board).

Board (deploy): `pip install -r requirements/edge.txt` — **không cài torch hay
tensorflow lên board**, đó là lý do tồn tại của đường NCNN/tflite-runtime/RKNN.

GPIO dùng **libgpiod** (`sudo apt install python3-libgpiod gpiod`), không dùng
`RPi.GPIO` — thư viện đó không tồn tại trên Orange Pi và số chân BCM của Pi
không có ý nghĩa ở đó. Tìm line offset bằng `gpioinfo`, điền vào
`configs/pipeline.yaml` mục `alerts`.

## Việc còn lại trước khi thi

- Chạy lại dashboard mới trên **board + camera + màn hình/máy chiếu thật**;
  kiểm ba trạng thái: an toàn, vi phạm áo lab trong vùng và mất tín hiệu.
- Mở rộng đánh giá theo sự kiện: nhiều người, nhiều phiên, trong lab; tách
  train/validation/test để ngưỡng áo không được chọn trên tập test.
- Đo công suất khi có camera thật và lặp nhiều cặp (29/10 đã đo 1 cặp trên video
  phát lại: nghỉ 0,86 W, sản phẩm 2,20 W, thích nghi 1,98 W, chạy hết 3,34 W; xem
  `docs/energy-2026-10-29/`).
- Thu và chấm video té ngã thật an toàn; thử lửa/khói trong bối cảnh lab trước
  khi trình bày hai module này như tính năng đã xác thực.
- Thu ca thiếu sáng/ngược sáng và thêm người xa, vì recall người xa là lát cắt
  yếu nhất đã đo.

Nguồn số liệu dùng khi làm hồ sơ thi: `docs/competition/competition-claims.md`. Kết quả
chi tiết và giới hạn nằm trong `docs/results.md`. Khi chuẩn bị demo dùng
`docs/competition/competition-demo-checklist.md`; trước phần hỏi đáp đọc
`docs/competition/judge-questions.md`.
