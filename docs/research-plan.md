# Đánh giá hướng đề tài + kế hoạch nghiên cứu

> **Cập nhật 24/09/2026 — đổi phần cứng.** Thiết bị đích là **Orange Pi 5
> (RK3588S)**, không phải Raspberry Pi 4. Hệ quả: §2.5 và §2.6 dưới đây viết
> cho Pi 4 và phần kết luận INT8 **đã đảo dấu** — A76/A55 đều có `asimddp`.
> Phân tích hiện hành nằm ở `docs/optimization.md`. Ma trận §4 đã cập nhật.
> Các lập luận về slice, throttling và occupancy **không đổi**, chúng độc lập
> với chip.

> Tài liệu này là phần "đánh giá" đi kèm code. Đọc trước khi chạy `scripts/`.

## 0. Kết quả đo THẬT trên Orange Pi 5 (cập nhật 30/09/2026)

Mọi số dưới đây đo trên board thật (RK3588S, 8 GB, NPU 0.9.8, camera C270 640x480),
có bằng chứng trong `KNOWN_ISSUES.md` và `.bangiao/`. Chưa có số nào dùng power meter
(chưa mua) nên **chưa có kết luận về năng lượng/sự kiện** (RQ0/RQ2 phần năng lượng).
Model vẫn là YOLO26n COCO (`person`), chưa fine-tune cho lab.

| Cấu hình (stage A, người) | FPS thực | Suy luận/frame | Nhiệt độ SoC | Ghi chú |
|---|---|---|---|---|
| CPU ONNX FP32 (N0L), 120 s | 10,5 | 90,9 ms | **86,8 °C, chưa bão hòa** | vượt ngưỡng 75 °C sau ~2 phút |
| NPU RKNN FP16 (N1L), 900 s | 14,5 | 63,2 ms | 46,2 °C | đêm 28/09, phòng tối, camera 15 FPS |
| NPU RKNN FP16 (N1L), 60 s | 19,8 | 47,8 ms | — | sáng 29/09, camera 30 FPS |
| NPU RKNN INT8 (N2L), 60 s | 35,1 | 25,1 ms | — | pipeline chạm trần camera 30 FPS |

- Cùng model FP16 nhưng 63 ms và 47,8 ms ở hai lần đo. **Chưa rõ nguyên nhân**
  (nghi NPU đổi xung theo tải khi camera chỉ 15 FPS; chưa kiểm chứng). Vì vậy so sánh
  FP16 với INT8 chỉ tin được trong cùng một phiên đo (47,8 ms so với 25,1 ms, ~1,9x).
- Ở chế độ tải liên tục CPU không giữ được nhiệt trong giới hạn còn NPU thì có: đây là
  luận cứ định lượng cho việc dùng NPU, độc lập với FPS.
- INT8 qua cổng parity (22 ảnh: recall 1,0, precision 0,96, IoU 0,87; FP16 IoU 0,997)
  **chỉ sau khi** sửa lỗi tọa độ chuẩn hóa (#24). Cổng parity đo độ khớp khung, không
  chỉ đếm phát hiện — bản INT8 lỗi có 49 phát hiện, 0 khớp, mà FPS và liveness vẫn xanh.

### Hệ thống đầy đủ A+B+C trên board (01/10/2026, INT8, phát lại video có 1 người)

| Mỗi lần gọi một mình (1 nhân NPU) | FP16 | INT8 |
|---|---|---|
| A: người, 640 | 50,6 ms | 30,5 ms |
| B: kính/găng trên crop 320, mỗi người | 16,6 ms | 12,5 ms |
| C: lửa/khói, 416 | 25,3 ms | 17,8 ms |

Chạy nối tiếp độ trễ khung p99 là 61 ms; cho B và C chạy bất đồng bộ trên nhân NPU riêng (0/1/2)
còn 50 ms (-18%), FPS và độ trễ trung vị không đổi vì gating đã làm B+C chỉ tốn ~0,8 ms/khung
(`KNOWN_ISSUES.md` #30). Tối ưu giải mã chỉ-lớp-cần-giữ giảm bước giải mã của A từ 2,1 xuống
0,38 ms (FP16) và 2,6 xuống 0,90 ms (INT8). Chưa kiểm độ chính xác B/C ở lab, chưa đo năng lượng.

### Đóng góp có bằng chứng tới nay (đề xuất trình bày)

1. **Lỗi im lặng khi đưa model lên NPU và cách chặn:** end2end segfault (#2), INT8 không
   phát hiện gì (#3), chuẩn hóa hai lần (#4), tọa độ INT8 đã chia cho imgsz (#24), tất cả
   đều không báo lỗi và không làm FPS thay đổi. Cổng parity + model card là biện pháp.
2. **Domain gap PPE công trường và phòng lab, có số đo:** fine-tune trên PPE công khai
   đạt mAP50 0,80 trên tập cùng miền nhưng bắt người ở lab kém model gốc (16/20 so với
   20/20) (#25).
3. **Open-vocabulary zero-shot chưa đủ để gán nhãn tự động cho vật nhỏ:** AP50 kính
   0,13–0,20, găng 0,41–0,46 (YOLOE 11s/11l) so với fine-tune 0,73/0,80 (#26).
   Kết luận âm này chỉ áp dụng cho mô hình và bộ dữ liệu đã thử.

### Còn thiếu (đừng viết như đã làm)
- Chưa có dataset lab, nên chưa có mAP hợp lệ cho PPE/áo blouse/fire/smoke và chưa có
  đánh giá theo slice (RQ3).
- Chưa đo năng lượng (RQ0/RQ2), chưa chạy adaptive vs parallel trên board.
- Stage B (PPE) và C (fire/smoke) đang tắt; fall và zone chưa được kiểm trên video có
  nhãn sự kiện.
- **RAM tăng ~6 MB/phút trong lượt 50 phút (157 -> 543 MB), không bình nguyên: rò thật, chưa tìm ra nguồn** (`KNOWN_ISSUES.md` #27). Chưa được báo cáo là "chạy ổn định dài hạn".

## 1. Tên đề tài (giữ, có chỉnh)

**Cost-Optimized Multi-Task Edge AI System for Real-Time Laboratory Safety Monitoring**

Tên ổn. Nhưng "Cost-Optimized" hiện đang mơ hồ: rẻ tiền phần cứng, hay rẻ tính toán?
Chốt một nghĩa và bám nó suốt bài. Đề xuất: **cost = năng lượng tính toán trên mỗi sự
kiện an toàn được phát hiện đúng**, vì đó là thứ pipeline adaptive thực sự tối ưu, và nó
đo được bằng power meter + log của hệ thống.

## 2. Những chỗ hướng cũ sẽ bị hỏi và khó trả lời

### 2.1. "Điểm mới" adaptive routing là điểm mới yếu nếu trình bày như hiện tại
Gating theo sự hiện diện của người (có người mới chạy PPE/fall) là **cascade detection**,
một kỹ thuật cũ (Viola–Jones 2001 đã là cascade; ROI-gating trên edge có rất nhiều bản).
Nếu bạn nói "điểm mới của em là không chạy model khi không có người", hội đồng sẽ coi đó
là kỹ thuật kỹ thuật (engineering), không phải đóng góp khoa học.

**Cách cứu:** đóng góp không nằm ở việc *có* gating, mà ở việc **đo và mô hình hoá
trade-off của gating dưới điều kiện thực tế của phòng lab**, cụ thể:
- Gating làm giảm năng lượng bao nhiêu **như một hàm của occupancy rate** (tỉ lệ thời
  gian có người trong khung hình). Phòng lab trống 70% thời gian thì gating thắng lớn;
  phòng lab luôn đông thì gating gần như vô dụng — và **bạn phải tự nói ra điều đó**.
- Gating làm **tăng detection latency** bao nhiêu ở biên (sự kiện xảy ra ngay lúc stage
  A vừa bỏ lỡ frame). Đây là cái giá phải trả, và không ai trong các paper bạn liệt kê đo nó.

Một đường cong `energy-per-event vs occupancy` + một bảng `event-detection latency` là
đóng góp định lượng thật. "Em có adaptive pipeline" thì không.

### 2.2. Một detector 4–5 lớp cho person + PPE + fire + smoke là rủi ro kỹ thuật lớn nhất
Bốn nhóm lớp này đến từ bốn dataset khác nhau, khác domain, khác phân bố kích thước
object, khác điều kiện ánh sáng. Gộp vào một YOLO nano sẽ gặp:
- **Mất cân bằng lớp nghiêm trọng** (ảnh fire không có person, ảnh PPE không có smoke →
  mỗi ảnh là negative cho 3/4 số lớp; model học "an toàn nhất là đừng predict").
- **Background inconsistency**: ảnh PPE có người không đeo kính nhưng không được gán nhãn
  → model bị phạt khi phát hiện đúng.
- mAP tổng sẽ đẹp giả, vì nó trung bình trên các lớp dễ (fire to, rõ) và lớp khó
  (kính bảo hộ nhỏ, bị che).

**Phương án tôi build thay vào đó — kiến trúc 2 tầng, tách theo tần suất cần thiết:**

| Tầng | Nhiệm vụ | Model | Cadence | Dataset |
|---|---|---|---|---|
| A | person detection | detector nhẹ, 1 lớp, input nhỏ | mọi frame | COCO-person / CrowdHuman subset |
| B | PPE trên crop người | detector hoặc multi-label classifier | chỉ khi A thấy người | PPE dataset |
| C | fire/smoke | classifier toàn khung, ảnh downscale | 1–2 Hz, độc lập với A | fire/smoke dataset |
| D | fall | hình học + thời gian trên track của A | miễn phí | tự quay + fall dataset để validate |
| E | restricted zone | polygon + tọa độ chân người | miễn phí | không cần |

Lợi ích: mỗi tầng train riêng bằng dataset gốc của nó, không phải gộp nhãn; mỗi tầng có
metric riêng (đúng chuẩn multi-task); và **cadence khác nhau chính là cái tạo ra câu
chuyện tối ưu chi phí**, mạnh hơn nhiều so với gating nhị phân.

Vẫn giữ phiên bản "một YOLO đa lớp" — nhưng làm **baseline V1 để so sánh**, không phải
sản phẩm cuối. Đó chính là ablation hội đồng muốn thấy.

### 2.3. Fall detection bằng bbox người là điểm yếu nhất về mặt khoa học
Bài Sensors 2024 bạn trích đạt 94.8% accuracy — nhưng cần xem họ dùng đặc trưng gì. Nếu
bạn chỉ dùng `w/h` của bounding box thì:
- Người **ngồi xổm** nhặt đồ → bbox bè ra → false positive.
- Người ngã **về phía camera** → bbox vẫn cao → false negative.
- Camera C270 góc thấp làm cả hai tệ hơn.

**Cách xử lý trong code này:** fall không phải một ngưỡng, mà là **máy trạng thái thời
gian** — yêu cầu (1) aspect ratio đảo chiều nhanh, (2) tâm bbox tụt nhanh, (3) **giữ
nguyên tư thế đó ≥ N giây** (người ngồi xuống rồi đứng lên không phải ngã; người ngã thì
nằm yên). Điều kiện (3) là cái giết phần lớn false positive và gần như miễn phí.
Đây là điểm bạn nên viết rõ trong bài — nó là một đóng góp nhỏ nhưng thật.

### 2.4. Các con số FPS/mAP bạn trích từ paper không được dùng làm mục tiêu
Bạn đã tự nhận ra điều này với bài Scientific Reports (benchmark chạy trên RTX 4090).
Nguyên tắc cần áp dụng cho **tất cả** bảng so sánh trong bài:
- NanoDet 15.2 FPS / 94.1% mAP là trên **5 lớp biển báo** — object to, rõ nét, tương phản
  cao, hình dạng chuẩn hoá. Kính bảo hộ trên mặt người khó hơn nhiều bậc.
- Đừng bao giờ đặt bảng của bạn cạnh bảng của họ như thể so sánh được. Trong bài, ghi rõ:
  *"các số liệu dưới đây từ các công trình khác nhau, không so sánh trực tiếp được do khác
  dataset, input size và runtime; chúng chỉ dùng để định vị vùng hiệu năng khả thi trên
  Raspberry Pi 4."*

### 2.5. Hai lỗi đo lường sẽ làm hỏng toàn bộ phần thực nghiệm
Đây là phần tôi lo nhất, vì nó âm thầm.

**(a) Thermal throttling.** Pi 4 chạy AI liên tục sẽ nóng và bị hạ xung ở ~80°C. Đo FPS
trong 30 giây rồi báo cáo là **sai**. Phải chạy ≥ 15 phút và báo cáo FPS ổn định +
đường nhiệt độ. Rất nhiều bài bỏ qua chỗ này; bạn làm đúng thì đó là điểm cộng.

**(b) Benchmark thiếu điều kiện khó làm cho tối ưu hoá trông như miễn phí.**
Nếu tập test của bạn chỉ có người đứng gần, đủ sáng, không che — thì giảm input từ 640
xuống 320 sẽ gần như không rớt mAP, và bạn sẽ kết luận "quantization/giảm resolution
không mất gì". Thực tế trên phòng lab, đúng những ca khó (người ở xa, kính nhỏ, ánh sáng
đèn huỳnh quang nhấp nháy, che khuất bởi thiết bị) mới là nơi tối ưu hoá phá huỷ accuracy.
**Bắt buộc:** tập test phải được chia thành các *slice* (xa/gần, sáng/tối, có che/không
che) và báo cáo mAP **theo từng slice**, không chỉ tổng. Xem `docs/measurement-protocol.md`.

### 2.6. Về Coral TPU: đồng ý chưa mua, nhưng lý do nên đổi
Lý do "để chứng minh Pi 4 CPU-only làm được" là lý do yếu (nghe như tự giới hạn). Lý do
mạnh hơn: **Coral làm hỏng câu chuyện cost-optimization của chính bạn** — thêm ~1.3–1.8
triệu vào BOM để lấy FPS, trong khi luận điểm của bạn là tối ưu *phần mềm* mang lại
hiệu năng. Giữ Coral như **phần thảo luận** ("nếu thêm X đồng thì được thêm Y FPS —
đây là điểm mà tối ưu phần mềm hết dư địa"), đó là một kết luận đẹp và không tốn tiền.

## 3. Câu hỏi nghiên cứu (viết lại cho đo được)

> **RQ0 (mới, câu hỏi trung tâm).** Trên một SoC biên không đồng nhất
> (NPU + CPU-big + CPU-LITTLE + GPU), việc ánh xạ các tác vụ an toàn có yêu
> cầu tần suất và độ trễ khác nhau lên các engine khác nhau ảnh hưởng thế nào
> đến độ trễ đầu-cuối, năng lượng trên mỗi sự kiện, và độ ổn định p99?
>
> **RQ1.** Với Raspberry Pi 4 CPU-only, cấu hình nào (model × runtime × precision ×
> input size) đạt được biên Pareto tốt nhất giữa mAP và latency cho phát hiện người và
> PPE trong phòng thí nghiệm?
>
> **RQ2.** Kiến trúc đa nhiệm phân tầng theo cadence giảm được bao nhiêu năng lượng trên
> mỗi sự kiện an toàn so với kiến trúc chạy song song tất cả, **và cái giá phải trả về
> độ trễ phát hiện là bao nhiêu**, như một hàm của tỉ lệ occupancy?
>
> **RQ3.** Các phép tối ưu (INT8, giảm input size, đổi runtime) ảnh hưởng thế nào đến
> accuracy **trên từng slice điều kiện khó**, so với ảnh hưởng trên tập test tổng?

RQ3 là câu ít người làm nhất và dễ ghi điểm nhất.

## 4. Ma trận thí nghiệm

> Bản đầy đủ của nhánh tối ưu, kèm tài liệu tham khảo và lý do từng kỹ thuật,
> nằm ở `docs/optimization.md`. Đọc file đó trước khi chạy E3.

| ID | Cấu hình | Mục đích | Biến đo |
|---|---|---|---|
| O0 | `arm_probe` + `roofline` trên Pi 4 | xác định nút thắt TRƯỚC khi tối ưu | ridge point, có/không dot-product |
| E1 | YOLO-n FP32, PyTorch, 640, mọi tác vụ chạy song song | baseline "ngây thơ" | mAP, FPS, RAM, W, °C |
| E2 | E1 nhưng runtime NCNN | tác động của inference engine | như trên |
| O1 | sweep thread × imgsz × backend | tối ưu **miễn phí**, không mất accuracy | p50/p90/p99 từng cấu hình |
| E3 | + INT8 | **giả thuyết**, không phải bước tối ưu — A72 thiếu SDOT | latency, size, mAP **theo slice** |
| O4 | + structured pruning 20/40/60% | đòn chính nếu compute-bound | params, GFLOPs, latency thực, mAP theo slice |
| O5 | + knowledge distillation | bù accuracy đã mất ở O4 | mAP theo slice ở cùng latency |
| E4 | + input 416 / 320 | tác động của resolution | + mAP theo slice |
| E5 | + kiến trúc phân tầng (adaptive) | đóng góp kiến trúc | + energy/event, detect latency |
| E6 | E5 ở occupancy 10/30/50/80% | đường cong của RQ2 | energy/event vs occupancy |
| E7 | (thảo luận) E4 + Coral | biên của tối ưu phần mềm | FPS/đồng |

Hai thí nghiệm không ai trong danh sách paper của bạn làm: **E6** (đường cong
theo occupancy) và **E3 báo cáo như kết quả âm có giải thích cơ chế**. Đó là
hai chỗ bài của bạn khác biệt.

**RQ4 bổ sung:** *Trên một CPU không có lệnh dot-product số nguyên, các kỹ
thuật nén model (quantization, structured pruning, distillation) đóng góp
thế nào vào latency thực tế, và thứ tự ưu tiên giữa chúng nên là gì?*

## 5. Lộ trình build (code trong repo này bám theo)

- **Phase 1 — hạ tầng đo trước, model sau.** Pipeline chạy được end-to-end với detector
  giả (`stub`), bench harness ghi đủ latency/FPS/RAM/°C/W. *Lý do:* nếu hạ tầng đo lường
  chưa đúng thì mọi con số train ra sau đều phải đo lại. Đây là phần đã build xong.
- **Phase 2 — dataset.** Gom, chuẩn hoá, **verify per-class counts** (`labsafe.datasets.verify`),
  chia slice điều kiện khó.
- **Phase 3 — train baseline** trên laptop RTX 3050 (E1).
- **Phase 4 — export + đo trên Pi 4** (E2–E4).
- **Phase 5 — adaptive** (E5–E6) + demo.
