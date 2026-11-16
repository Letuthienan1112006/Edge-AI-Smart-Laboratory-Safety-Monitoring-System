# Tối ưu AI trên RK3588S (Orange Pi 5) — 25% điểm

> **Cập nhật 24/09/2026.** Phần cứng đổi từ Raspberry Pi 4 (Cortex-A72) sang
> Orange Pi 5 (RK3588S). Tài liệu này đã viết lại. Phần kết luận cũ về
> "INT8 sẽ chậm hơn" **không còn đúng** với thiết bị mới — xem §1.

Thể lệ chấm 4 phần bằng nhau, trong đó **25% là "Mức độ tối ưu hóa/tăng tốc
triển khai trên phần cứng"**, nêu đích danh quantization, pruning, knowledge
distillation. Với RK3588S bạn có thêm một trục mà Pi 4 không có, và nó
chính là trục mạnh nhất: **tăng tốc bằng phần cứng chuyên dụng trên chip**.

---

## 0. Phần cứng bạn đang có

| Thành phần | Thông số | Ý nghĩa cho đề tài |
|---|---|---|
| CPU big | 4× Cortex-A76 @ 2.4GHz, ARMv8.2-A | **có `asimddp`** → INT8 tăng tốc thật |
| CPU LITTLE | 4× Cortex-A55 @ 1.8GHz, ARMv8.2-A | cũng có `asimddp`, nhưng chậm hơn nhiều |
| GPU | Mali-G610 MP4 | đường tính toán thứ ba (OpenCL) |
| NPU | RKNPU2, 6 TOPS, **3 nhân** | INT8 chuyên dụng, mạnh nhất, hạn chế op |
| RAM | 8GB | dư sức, swap không bao giờ chạm tới |

Bốn engine tính toán khác nhau trên một con chip. **Đó là toàn bộ đề tài mới
của bạn.** Chi tiết ở §3.

Chạy trên board để tự xác minh, đừng tin bảng trên:

```bash
make probe          # in ra cluster, tần số, cờ ISA, NPU
make affinity       # in ra topology + lệnh taskset
```

---

## 1. INT8: kết luận cũ đã đảo, và đó là bài học chính

Với Pi 4 (Cortex-A72, ARMv8.0-A) tôi đã kết luận INT8 **có thể làm chậm đi**,
vì A72 thiếu lệnh dot-product số nguyên SDOT/UDOT. Kết luận đó đúng — **với
con chip đó**.

RK3588S dùng Cortex-A76 và A55, cả hai đều là **ARMv8.2-A và đều có
`asimddp`**. Nên trên máy của bạn bây giờ:

| CPU | dot-product | INT8 vs FP32 |
|---|---|---|
| Cortex-A72 (Pi 4) | không | 1.0× hoặc chậm hơn |
| **Cortex-A76 (RK3588S, Pi 5)** | **có** | **1.83× nhanh hơn** |
| Cortex-A78AE (Orin) | có | 2.11× nhanh hơn |
| Cortex-A53 | không | 1.65× **chậm hơn** |

(arXiv:2609.16085)

**Quantization giờ là một bước tối ưu thật, không còn là giả thuyết.** Nhánh
INT8 trong kế hoạch của bạn hoạt động như dự định ban đầu.

### Bài học vẫn còn nguyên giá trị, và nên viết vào báo cáo

Điểm đáng nói không phải "INT8 nhanh" hay "INT8 chậm". Là thế này:

> Cùng một kỹ thuật tối ưu, cùng một model, cùng một runtime — kết quả đảo
> dấu khi đổi CPU, vì nó phụ thuộc vào **một cờ ISA** mà không tên kỹ thuật
> nào nhắc tới.

Đó là lý do `make probe` tồn tại: nó **đo** chứ không **giả định**, nên khi
phần cứng đổi thì công cụ vẫn đúng còn tài liệu thì sai. Nếu bạn trình bày
được đúng luận điểm này — kèm bảng A72 vs A76 ở trên — thì đó là một nhận
định phương pháp luận, không phải một con số benchmark. Hội đồng đánh giá
cao thứ đó hơn.

---

## 2. Pruning: phần này không đổi

- **Unstructured pruning** (đặt từng trọng số về 0): tensor giữ nguyên hình
  dạng → GEMM giữ nguyên kích thước → **chạy đúng tốc độ cũ**. Thứ thay đổi
  là dung lượng file sau nén.
- **Structured pruning** (xóa cả kênh/filter): tensor co lại thật →
  convolution làm ít việc hơn → nhanh lên thật.

Đúng trên A72, đúng trên A76, đúng trên NPU. `scripts/train/prune_structured.py`
**từ chối chạy** với `--unstructured` và in lý do.

Lưu ý riêng cho NPU: structured pruning có thể làm số kênh lệch khỏi bội số
mà NPU thích (thường là bội của 8 hoặc 16). Prune xuống 0.4 rồi làm tròn số
kênh lên bội số gần nhất — nếu không, NPU sẽ pad ngầm và bạn mất phần lớn
lợi ích mà không thấy lý do.

**⚠ Rủi ro đã kiểm chứng thật, 2026-09-26** (không phải giả định): chạy
`prune_structured.py` thật trên `yolo11n.pt` với `torch 2.14` +
`torch_pruning 1.6.1` cho ra **script chạy không lỗi, in bảng có vẻ hợp
lý, và cắt được đúng 0.0% — exit code 0**. Bộ dò dependency graph của
`torch_pruning` chỉ trace được 1 module trong số 88 lớp Conv2d, không có
cảnh báo nào. Đây là đúng loại lỗi "chạy xong, có số, số đó là giả" mà
toàn bộ dự án này tồn tại để chặn (giống hệt ca INT8 detect-0 của RKNN).

Đã thêm 2 lớp chặn cứng vào script (so số Conv2d được trace, so tham số
trước/sau mỗi bước) — giờ nó **exit 1 kèm chẩn đoán rõ ràng** thay vì báo
thành công giả. Nguyên nhân gốc **chưa xác minh xong** (nghi do lệch phiên
bản `torch`/`torch_pruning` — ultralytics tự pin `torch==2.4` cho môi
trường RKNN cô lập của họ, khác xa `torch 2.14` mà pip resolve mặc định).
Xem `docs/KNOWN_ISSUES.md` #12 trước khi tin bất kỳ con số pruning nào từ
script này. **Đừng bỏ qua bảng kết quả chỉ vì nó không báo lỗi** — verify
`p1 < p0` thật bằng mắt, không chỉ tin exit code 0.

---

## 3. Đóng góp mới: lập lịch đa nhiệm trên chip không đồng nhất

Đây là chỗ đề tài của bạn mạnh hơn hẳn phiên bản Pi 4.

Bạn có **5 tác vụ** với yêu cầu rất khác nhau, và **4 engine** để chạy chúng:

| Tác vụ | Tần suất cần | Đặc tính |
|---|---|---|
| A: person detection | mọi frame | nặng, liên tục, quyết định độ trễ hệ thống |
| B: PPE trên crop | ~0.5 Hz/người | nặng, thưa, chịu được độ trễ |
| C: fire/smoke | 1 Hz | trung bình, đều đặn, chịu được độ trễ |
| D: fall | mọi frame | hình học, gần như miễn phí |
| E: restricted zone | mọi frame | hình học, miễn phí |

| Engine | Mạnh | Yếu |
|---|---|---|
| NPU (3 nhân) | throughput INT8 cao nhất, tiết kiệm điện | chỉ INT8, giới hạn op, độ trễ khởi động mỗi lần gọi |
| CPU big (A76×4) | linh hoạt, mọi op | tốn điện, tranh chấp với pipeline |
| CPU LITTLE (A55×4) | rất tiết kiệm điện | chậm |
| GPU (Mali-G610) | song song tốt | driver OpenCL lắt léo, tranh chấp với hiển thị |

**Câu hỏi nghiên cứu mới (RQ-chính):**

> Trên một SoC biên không đồng nhất, việc ánh xạ các tác vụ an toàn có yêu
> cầu tần suất và độ trễ khác nhau lên các engine tính toán khác nhau
> (NPU / CPU-big / CPU-LITTLE / GPU) ảnh hưởng thế nào đến độ trễ đầu-cuối,
> năng lượng trên mỗi sự kiện, và độ ổn định p99?

Không ai trong 8 bài Related Work của bạn trả lời câu này — vì họ chạy trên
Pi 4, nơi chỉ có một loại engine nên câu hỏi không tồn tại.

Và `AdaptiveRouter` bạn đã có **chính là kiến trúc để trả lời nó**. Nó đã
tách tác vụ theo cadence rồi; giờ mỗi tác vụ chỉ cần thêm một thuộc tính:
chạy ở đâu.

### Giả thuyết đáng kiểm chứng

- Stage A lên NPU (chạy mọi frame, INT8, op đơn giản) — được lợi nhiều nhất.
- Stage B cũng NPU nhưng **thưa**: mỗi lần gọi NPU có chi phí khởi động, nên
  với tác vụ thưa, CPU-big có khi **thắng**. Đây là một kết quả phản trực giác
  đáng đo.
- Stage C lên CPU-LITTLE: 1 Hz, không ai vội, chạy ở đó gần như miễn phí về
  điện và không tranh chấp với stage A.
- D và E là hình học, luôn ở CPU-LITTLE.
- Chạy **tất cả** trên NPU có thể tệ hơn phân bổ hỗn hợp, vì 3 nhân NPU bị
  tranh chấp. Cũng đáng đo.

Một bảng "tác vụ × engine → latency, watt" là kết quả trung tâm của bài.

---

## 4. big.LITTLE: tối ưu miễn phí, và là điều kiện để số liệu hợp lệ

Linux scheduler tự quyết định thread inference chạy trên nhân nào, và nó tối
ưu cho năng lượng toàn hệ thống chứ không cho p99 của bạn. Nó **sẽ** đẩy
thread nóng sang nhân A55, và triệu chứng duy nhất là độ trễ dao động trông
như nhiễu.

```bash
python3 -m labsafe.optimize.affinity --benchmark   # đo big vs LITTLE vs không ghim
taskset -c 4,5,6,7 python3 -m labsafe.app --bench ...   # ghim vào cụm big
```

Báo cáo **cả hai** con số:
- **p50 gain** = tốc độ tăng được.
- **p99 gain** = lập luận về tính hợp lệ của phép đo. Benchmark không ghim là
  benchmark cho scheduler chen vào đuôi phân phối, và đó không phải lỗi model.

Đánh đổi phải nói rõ, không giấu: ghim vào nhân big làm tăng điện và nhiệt,
đẩy thời điểm throttle đến sớm hơn. Đo năng lượng trên mỗi suy luận, không
chỉ latency, và theo dõi nhiệt độ.

---

## 5. Cảnh báo thực tế về RKNN — đây là rủi ro lớn nhất còn lại

Đường NPU không cắm là chạy. Ba chỗ hay mất nhiều ngày:

1. **Hai gói khác nhau, phải khớp phiên bản.** `rknn-toolkit2` chạy trên máy
   x86 để convert ONNX → `.rknn`. `rknn-toolkit-lite2` chạy trên board để suy
   luận. Cả hai phải khớp với `librknnrt.so` có sẵn trong image. Lệch phiên
   bản thì lỗi khó hiểu.

2. **`yolo_decode()` chỉ nhận output đã decode, không nhận nhánh raw kiểu
   `rknn_model_zoo`.** Có hai cách cắt model khác nhau và chúng KHÔNG tương
   thích:
   - Đường chính thức của repo này: export bằng chính Ultralytics với
     `end2end=False`. Output vẫn là `(1, 4+nc, N)` đã qua DFL và đã giải mã
     theo stride, chỉ thiếu NMS. `yolo_decode()` trong
     `src/labsafe/models/postprocess.py` nhận đúng dạng này và làm NMS bằng
     numpy thuần trên CPU.
   - Đường `rknn_model_zoo` của Rockchip cắt sâu hơn: bỏ luôn phần DFL, xuất
     3-9 nhánh reg/cls thô theo từng stride. `yolo_decode()` **không** đọc
     được dạng này — nó sẽ không crash, chỉ transpose nhầm trục và ra box
     rác một cách im lặng (xem `docs/KNOWN_ISSUES.md` mục 5).
   Dùng đường Ultralytics (`end2end=False`), không dùng đường `model_zoo`,
   trừ khi ai đó viết lại `yolo_decode()` để nhận cả hai dạng và tự nhận diện
   được dạng nào đang được đưa vào.

3. **Op không map được sẽ rơi về CPU, âm thầm.** Một model trông như "chạy
   trên NPU" có thể đang tiêu phần lớn thời gian ở các fallback. **Luôn so với
   backend CPU**, đừng cho rằng NPU mặc nhiên thắng. Backend RKNN trong repo
   này dùng chung interface với NCNN/TFLite đúng để so sánh đó là một dòng config.

4. **Lần đầu thử format export mới (TFLite) trong venv đã verify RKNN có
   thể âm thầm đổi `torch` của cả venv.** Xác minh thật, 2026-09-26: thử
   `export_models.py --formats tflite` lần đầu khiến `ultralytics` tự
   AutoUpdate và **hạ `torch` từ 2.14.0 xuống 2.13.0** (tải 502 MB, không
   hỏi, không cảnh báo) để cài `litert-torch`. Lần chạy ĐÓ còn crash — dễ
   hiểu nhầm là "TFLite không tương thích" — nhưng chạy lại đúng lệnh ở
   tiến trình mới thì **thành công** (torch 2.13 + litert-torch không thực
   sự xung đột, chỉ là tiến trình đầu bị mâu thuẫn phiên bản giữa chừng lúc
   pip cài đè). Điều thật sự đáng lo không phải TFLite, mà là việc `torch`
   có thể đổi phiên bản bất cứ lúc nào một format mới được thử — đã re-test
   RKNN + 150 test sau sự cố này, vẫn ổn, nhưng luôn re-verify RKNN sau bất
   kỳ AutoUpdate nào. Xem `docs/KNOWN_ISSUES.md` #14.

**Phương án dự phòng:** nếu RKNN ngốn quá 1 tuần, dừng lại. Đường
NCNN trên 4 nhân A76 @2.4GHz đã nhanh hơn Pi 4 rất nhiều và đủ để có một hệ
thống chạy tốt. NPU khi đó thành một chương "hướng phát triển" có số liệu
một phần, thay vì một hố đen nuốt cả đồ án.

---

## 6. Tài liệu tham khảo

### 6.1. Quantization

| # | Công trình | Vì sao cần |
|---|---|---|
| Q1 | Jacob et al., **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**, CVPR 2018 (arXiv:1712.05877) | Nền tảng INT8 của TFLite và của NPU. |
| Q2 | Krishnamoorthi, **Quantizing deep convolutional networks for efficient inference: A whitepaper**, 2018 (arXiv:1806.08342) | PTQ vs QAT, per-tensor vs per-channel. |
| Q3 | Nagel et al., **A White Paper on Neural Network Quantization**, 2021 (arXiv:2106.08295) | Tổng quan hiện đại. |
| Q4 | Shin, **Is INT8 Portable? A Cross-Platform Measurement Study**, 2026 (arXiv:2609.16085) | Bảng A72/A53/A76/A78 ở §1 lấy từ đây. Chỗ dựa cho luận điểm "phụ thuộc ISA". |

### 6.2. Pruning

| # | Công trình | Vì sao cần |
|---|---|---|
| P1 | Han et al., **Deep Compression**, ICLR 2016 (arXiv:1510.00149) | Kinh điển, nhưng tối ưu **dung lượng**. |
| P2 | Li et al., **Pruning Filters for Efficient ConvNets**, ICLR 2017 (arXiv:1608.08710) | Bài cho **tốc độ**. Phương pháp trong `prune_structured.py`. |
| P3 | Frankle & Carbin, **The Lottery Ticket Hypothesis**, ICLR 2019 (arXiv:1803.03635) | Vì sao prune từng bước tốt hơn prune một phát. |
| P4 | Fang et al., **DepGraph: Towards Any Structural Pruning**, CVPR 2023 (arXiv:2301.12900) | Bài của `torch-pruning`; xử lý phụ thuộc residual/concat. |

### 6.3. Knowledge distillation

| # | Công trình | Vì sao cần | Trạng thái code |
|---|---|---|---|
| D1 | Hinton, Vinyals & Dean, **Distilling the Knowledge in a Neural Network**, 2015 (arXiv:1503.02531) | Bài gốc — soft-target qua loss. | **Đã áp dụng thật**, `src/labsafe/models/kd_loss.py` (`KDDetectionLoss`) + `scripts/train/distill_kd.py`. Thích nghi cho đầu multi-label của YOLO (BCE nhiệt-độ-hoá thay vì softmax+KL của bản gốc, vì YOLO không có phân phối softmax trên lớp) — xem docstring của `kd_loss.py` để trích dẫn đúng phạm vi, đừng nói "y hệt Hinton". Verify thật 2026-09-26 (`docs/KNOWN_ISSUES.md` #16), 9 unit test. |
| D2 | Chen et al., **Learning Efficient Object Detection Models with Knowledge Distillation**, NeurIPS 2017 | KD cho detection bằng feature-mimicking. | **Không áp dụng.** `KDDetectionLoss` chỉ KD nhánh classification, không khớp feature map trung gian như D2 — nói rõ trong docstring, đừng nhận đây là D2. |
| D3 | Cao et al., **Learning Lightweight Object Detectors via Multi-Teacher Progressive Distillation**, ICML 2023 | Nhiều teacher, theo giai đoạn. | **Không áp dụng.** Chỉ 1 teacher, không có lịch trình progressive. |

`scripts/train/distill_pseudolabel.py` là kỹ thuật KHÁC — pseudo-labeling (chuyển
tri thức qua DATASET, một lần, offline) — không phải D1/D2/D3 (chuyển qua
LOSS, mỗi bước, online). Hai cái bổ sung cho nhau, không thay thế nhau;
đừng gọi pseudo-labeling là "distillation" trước hội đồng nếu không nói rõ
đây là hình thức khác D1-D3.

### 6.4. Kiến trúc hiệu quả và lập lịch không đồng nhất

| # | Công trình | Vì sao cần |
|---|---|---|
| A1 | Howard et al., **MobileNets**, 2017 (arXiv:1704.04861) | Depthwise separable conv. |
| A3 | Howard et al., **Searching for MobileNetV3**, ICCV 2019 (arXiv:1905.02244) | Tối ưu theo **latency thực đo**, không theo FLOPs — đúng tinh thần đề tài. |
| A4 | Cai et al., **Once-for-All**, ICLR 2020 (arXiv:1908.09791) | Một lần train, nhiều cấu hình theo ràng buộc thiết bị. |
| A5 | Lin et al., **MCUNet: Tiny Deep Learning on IoT Devices**, NeurIPS 2020 (arXiv:2007.10319) | Đồng thiết kế model + runtime. |

> Tự kiểm tra lại số arXiv/DOI trước khi nộp. Tên và năm là chuẩn, nhưng sai
> một chữ số DOI thì hội đồng thấy ngay.

---

## 7. Ma trận thí nghiệm

| ID | Cấu hình | Giả thuyết | Đo gì |
|---|---|---|---|
| **O0** | `make probe` + `make roofline` | xác định nút thắt và năng lực ISA | cluster, asimddp, ridge point |
| **O1** | affinity: big / LITTLE / không ghim | 1.3–2× **miễn phí**; p99 hẹp lại rõ | p50, p90, p99, W, °C |
| **O2** | sweep thread × imgsz × backend | thêm 1.2–1.5× miễn phí | p50/p90/p99 mỗi cấu hình |
| **O3** | INT8 PTQ trên CPU A76 | **~1.8× nhanh hơn** (có asimddp) | latency, size, mAP theo slice |
| **O4** | INT8 QAT | phục hồi accuracy so với O3 | mAP theo slice vs O3 |
| **O5** | structured pruning 20/40/60% | giảm FLOPs; chú ý bội số kênh cho NPU | params, GFLOPs, latency, mAP slice |
| **O6** | KD teacher → student | bù accuracy mất ở O5 | mAP slice ở cùng latency |
| **O7** | NPU 1 nhân vs 3 nhân | 3 nhân nhanh hơn nhưng tốn điện hơn | latency, W, J/inference |
| **O8** | **ánh xạ tác vụ × engine** | phân bổ hỗn hợp thắng "tất cả lên NPU" | bảng tác vụ×engine: latency, W |
| **O9** | O8 + adaptive pipeline | cấu hình cuối | toàn bộ + J/event vs occupancy |

**O8 là đóng góp trung tâm.** O1 và O7 là hai kết quả phụ rẻ và đẹp.

---

## 8. Câu chuyện nên kể trước hội đồng

> Chúng tôi triển khai hệ thống giám sát an toàn đa nhiệm trên một SoC biên
> không đồng nhất có bốn engine tính toán. Thay vì coi thiết bị như một CPU,
> chúng tôi coi nó như một bài toán lập lịch: năm tác vụ an toàn có yêu cầu
> tần suất và độ trễ khác nhau được ánh xạ lên NPU, CPU-big, CPU-LITTLE và
> GPU. Chúng tôi chỉ ra rằng phân bổ hỗn hợp vượt trội so với dồn tất cả lên
> NPU, rằng việc ghim tiến trình vào cụm nhân lớn thu hẹp p99 **X%** mà không
> tốn accuracy nào, và rằng lợi ích của INT8 phụ thuộc vào một cờ tập lệnh
> chứ không vào bản thân kỹ thuật lượng tử hóa — điều chúng tôi xác minh bằng
> một công cụ probe chạy trên chính thiết bị đích.

Đủ ba kỹ thuật thể lệ nêu tên, cộng một trục tăng tốc phần cứng thật, cộng
một nhận định phương pháp luận. Đó là nghiên cứu.

---

## 9. Thứ tự chạy

```bash
# 0. Trên board, trước khi tối ưu bất cứ thứ gì
make probe
make affinity
make roofline ARGS="--gflops 1.2 --weights-mb 6.2"

# 1. Miễn phí trước: ghim nhân, rồi quét thread/imgsz
python3 -m labsafe.optimize.affinity --benchmark
taskset -c 4,5,6,7 python3 scripts/bench/sweep_runtime.py --backend ncnn \
        --weights <ncnn_model> --threads 1,2,3,4 --imgsz 256,320,416

# 2. Quantization — giờ là bước tối ưu thật
python3 scripts/export/export_models.py <best.pt> --imgsz 320 --formats tflite \
        --int8 --data <data.yaml>

# 3. Pruning + distillation
python3 scripts/train/prune_structured.py --weights <best.pt> --data <data.yaml> \
        --ratio 0.4 --steps 4
# 3a. pseudo-label thêm dữ liệu (qua DATASET, offline) -- KHÔNG phải D1-D3
python3 scripts/train/distill_pseudolabel.py --teacher yolov8m.pt \
        --videos data/raw/lab_footage --out data/processed/kd
# 3b. KD thật kiểu Hinton (qua LOSS, online) -- đây mới là D1
python3 scripts/train/distill_kd.py --teacher yolo26s.pt --student yolo26n.pt \
        --data <data.yaml> --epochs 50 --alpha 0.5 --temperature 2.0

# 4. NPU (đặt hạn 1 tuần, có phương án dừng)
#    convert trên x86: rknn-toolkit2, cắt head trước decode
python3 -m labsafe.app --bench --exp O7_npu3 --config configs/exp/O7_npu.yaml

# 5. Đo lại toàn bộ, mAP theo từng slice
bash scripts/bench/bench_matrix.sh
```
