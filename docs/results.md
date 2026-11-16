# Kết quả đã đo (tổng hợp cho báo cáo)

Cập nhật 21/10/2026. Mọi số ở đây là **đo thật**; mỗi bảng ghi nguồn (mục `docs/KNOWN_ISSUES.md` hoặc log) để kiểm lại.
Phần cứng: Orange Pi 5 (RK3588S, 8 GB), NPU RKNPU2, camera Logitech C270 640x480. Model: YOLO26n.
Mục cuối liệt kê những gì **chưa** đo — đừng viết chúng như đã làm.

## 1. Đưa model lên NPU: tốc độ và nhiệt (Stage A, phát hiện người)

| Cấu hình | FPS thực | Suy luận/khung | Nhiệt độ SoC |
|---|---|---|---|
| CPU, ONNX FP32 | 10,5 | 90,9 ms | **86,8 °C sau ~2 phút, chưa bão hòa** |
| NPU, RKNN FP16 | 19,8 | 47,8 ms | — |
| NPU, RKNN INT8 | ~29-30 (trần camera) | 25,1 ms | 47-50 °C khi chạy dài |

- INT8 nhanh ~1,9 lần FP16 (cùng phiên đo). Hệ thống đã chạm **trần camera** (~30 FPS): tối ưu thêm phần suy luận không
  tăng FPS; NPU còn dư ~27% (tải nhân 0 khoảng 72%).
- Luận cứ độc lập với FPS: CPU không giữ được nhiệt khi chạy liên tục, NPU giữ được.
- Nguồn: `docs/research-plan.md` §0, KNOWN_ISSUES #24, #31, #37.

## 2. Tối ưu trong pipeline

| Kỹ thuật | Trước | Sau | Nguồn |
|---|---|---|---|
| Giải mã chỉ các lớp cần giữ (`class_subset`) | 2,1 ms (FP16) / 2,6 ms (INT8) | 0,38 / 0,90 ms | #30 |
| Stage B/C bất đồng bộ trên nhân NPU riêng | p99 61,2 ms, max 78,6 ms | p99 **49,9 ms**, max 58,3 ms | #30 |
| Camera chờ ảnh mới (`wait_new`) | 37 FPS "thổi phồng", 20% suy luận lại ảnh cũ | 29,6 FPS thật, 0% trùng | #31 |
| Cảnh báo lặp (cooldown 60 s, tracker nhớ 3 s) | 122 cảnh báo / 14,5 phút | 24 / 9,5 phút | #37 |

Gating (B mỗi 2 s/người, C mỗi 1 s) làm chi phí trung bình B+C chỉ ~0,8 ms/khung.

## 3. Cổng kiểm tra khi chuyển model sang NPU (parity)

Bốn lỗi **im lặng** (FPS và "có phát hiện" vẫn bình thường, nhưng kết quả sai) đã bị cổng parity bắt: end2end segfault
(#2), INT8 không phát hiện gì (#3), chuẩn hóa hai lần (#4), tọa độ INT8 đã chia cho imgsz (#24 — 49 phát hiện, 0 khớp).

| Model lab trên board (INT8 so với ONNX) | recall | precision | IoU khớp | Kết luận |
|---|---|---|---|---|
| `lab_v3`, 40 ảnh test, librknnrt 2.3.0 | 0,92 | 0,87 | 0,82 | PASS |
| `lab_v3`, cùng ảnh, librknnrt 2.3.2 | 0,92 | 0,87 | 0,82 | PASS (giống hệt) |
| `lab_v5`, 78 ảnh test (2 góc), librknnrt 2.3.2 | 0,98 | 0,96 | 0,84 | PASS (15/10) |
| `lab_v6` (model đang dùng), cùng 78 ảnh | 0,98 | 0,93 | 0,84 | PASS (15/10) |

## 4. Dữ liệu và học với ít nhãn

Lớp: `person`, `lab_coat` (đã bỏ kính và găng: không ai đeo kính; mọi khung găng là một đôi nằm trên bàn, #33).

| Bước | Kết quả đo | Nguồn |
|---|---|---|
| Model học trên PPE công trường, đưa vào lab | bắt người kém model COCO gốc (16/20 so với 20/20) | #25 |
| Model open-vocabulary làm "thầy" gán nhãn | AP50 kính 0,13-0,20, găng 0,41-0,46 — không dùng được | #26 |
| Pilot 146 ảnh, test CÙNG phiên | AP50 áo 0,91 nhưng gọi áo thun trắng là áo lab 18/20 | #34 |
| Model gợi ý nhãn, người chỉ sửa | dùng cho mọi lô (~1.700 ảnh đã gán) | #32 |
| **Chọn ảnh khó** (model lưỡng lự về áo), lô 1 | độ chính xác áo **0,52 -> 0,76** | #35 |
| Lô khó thứ 2, cùng phiên | không cải thiện đo được (bão hòa) | #35 |
| Train trên nhãn máy đoán, không sửa | không dùng: học lại chính lỗi của nó | — |

Bài học đo lường: tập test cùng phiên cho điểm cao giả (0,99 so với 0,45 trên người mới); **test phải là phiên khác**.

## 5. Độ chính xác trên tập test thật (phiên riêng, người model chưa thấy)

Luật "vùng nguy hiểm phải mặc áo lab", chấm theo **từng người**, ngưỡng áo 0,6, trung bình 3 lần train (seed 0/1/2).
`scripts/eval/eval_zone_rule.py`, log `runs/logs/eval_v5.txt`.

| | Góc 1: bắt người không áo | Góc 1: báo nhầm người có áo | Góc 2: phát hiện người AP50 | Góc 2: bắt không áo | Góc 2: báo nhầm |
|---|---|---|---|---|---|
| `lab_v3` (941 ảnh, 1 góc) | 75% (72-77) | 5% | 0,94 (0,93-0,95) | 87% | 9% |
| `lab_v5` (1.400 ảnh, 2 góc) | **78%** (75-82) | 4% | **0,96** (0,96-0,96) | **90%** | 10% |

- Chỉ đổi seed đã lệch **±4 điểm**: chênh nhỏ hơn mức đó không được kết luận.
- **Đổi góc camera:** `lab_v3` chưa thấy góc 2 vẫn đạt AP50 áo 0,94 ở đó; thêm ảnh góc 2 làm việc bắt người tốt hơn
  chắc chắn (+2, ba seed không chồng nhau) mà không kém đi ở góc 1.
- Ngưỡng áo là đánh đổi: (góc 1, `lab_v3`) 0,4: bắt 66% / nhầm 2%; 0,6: 77% / 4%; 0,7: 82% / 5%; 0,8: 89% / 15%.
- Model lớn hơn (YOLO26s): chỉ hơn ~2 điểm ở cùng mức báo nhầm, chạy chậm hơn 2-3 lần -> giữ YOLO26n.
- Áo thun trắng bị gọi là áo lab ở 19-20/20 ảnh với mọi model chỉ học ảnh lab. **Đã sửa ở `lab_v6`** bằng 238 ảnh COCO
  người mặc áo trắng làm âm tính: **0-4/20** (3 seed), điểm ở lab không đổi trong mức nhiễu (#40).

## 5b. So sánh kiến trúc detector (16/10): cùng dữ liệu, cùng công thức, 3 seed mỗi bên

Dữ liệu `lab_v6` (1.638 ảnh), 60 epoch, ảnh 640, batch 16, công thức mặc định; chấm bằng `scripts/eval/eval_ap.py` (một bộ chấm
chung, FP32/PyTorch) trên hai tập test khác phiên. Trung bình 3 seed, khoảng min-max trong ngoặc. Log `runs/cmp/`.

| Model | Tham số | Góc 1 mAP50 | Góc 1 mAP50-95 | Góc 2 mAP50 | Góc 2 mAP50-95 |
|---|---|---|---|---|---|
| **YOLO26n** (đang dùng) | 2,50 M | 0,869 (0,865-0,876) | **0,668** (0,665-0,671) | 0,973 (0,972-0,974) | **0,792** (0,791-0,794) |
| YOLOv8n | 3,01 M | 0,867 (0,861-0,871) | 0,656 (0,647-0,663) | 0,971 (0,970-0,972) | 0,786 (0,781-0,789) |
| YOLO11n | 2,59 M | 0,867 (0,861-0,878) | 0,665 (0,655-0,683) | 0,971 (0,969-0,973) | 0,783 (0,780-0,786) |
| YOLOv5nu | 2,51 M | 0,868 (0,861-0,871) | 0,655 (0,650-0,659) | 0,972 (0,971-0,973) | 0,783 (0,780-0,785) |

- **Độ chính xác phát hiện (mAP50) của cả bốn ngang nhau**: chênh <= 0,002, nhỏ hơn dao động giữa các seed.
- **mAP50-95** (độ khít của khung) YOLO26n cao nhất ở cả hai góc; hơn YOLOv8n và YOLOv5nu khoảng 0,01 và khoảng
  seed không chồng nhau ở góc 1; so với YOLO11n thì chồng nhau (chưa kết luận). Chênh nhỏ nhưng nhất quán.
- Vậy chọn model KHÔNG nên dựa vào độ chính xác trên tập này; quyết định nằm ở triển khai trên NPU (độ trễ INT8, mức tụt
  điểm khi lượng tử hóa, hỗ trợ xuất RKNN). **Chưa đo trên bo mạch** (offline lúc viết): cần độ trễ NPU và AP INT8 của
  từng model.

## 5c. Lỗi nằm ở đâu: điểm theo lát cắt, có khoảng tin cậy 95% (16/10)

`scripts/eval/eval_slices.py`, model `lab_v6` (seed 0, FP32), gộp hai tập test khác phiên (299 ảnh, 1.724 người, 1.145 áo).
Ngưỡng như khi triển khai: người 0,4, áo 0,6, IoU 0,5. Khoảng tin cậy lấy bằng bootstrap theo ẢNH (1.000 lần lấy mẫu lại).

| Lát cắt | Recall người | Recall áo | Precision người | Precision áo |
|---|---|---|---|---|
| Tất cả | 0,917 [0,903-0,930] | 0,893 [0,875-0,909] | 0,886 | 0,877 |
| Người **xa** (khung cao < 188 px) | **0,800** [0,766-0,831] | --- | --- | --- |
| Người cỡ vừa (188-278 px) | 0,971 [0,955-0,983] | --- | --- | --- |
| Người **gần** (>= 278 px) | 0,979 [0,967-0,991] | --- | --- | --- |
| Khung người không chồng người khác | 0,916 [0,900-0,930] | --- | --- | --- |
| Khung người chồng lên người khác | 0,924 [0,888-0,958] | --- | --- | --- |
| Ảnh thưa (<= 3 người) | 0,932 [0,897-0,963] | 0,940 | 0,798 | **0,733** [0,636-0,831] |
| Ảnh vừa (4-6 người) | 0,900 [0,876-0,922] | 0,909 | 0,873 | 0,782 |
| Ảnh đông (>= 7 người) | 0,926 [0,907-0,944] | 0,882 | 0,911 | 0,940 |
| Ảnh tối nhất (độ sáng < 127) | 0,914 [0,886-0,940] | 0,881 | 0,857 | 0,805 |
| Ảnh sáng nhất (>= 136) | 0,942 [0,925-0,959] | 0,894 | 0,921 | 0,943 |

- **Điểm yếu thật duy nhất có ý nghĩa thống kê: người xa.** Recall 0,80 so với 0,97-0,98 khi gần; khoảng tin cậy không
  chồng nhau. Đây là hướng cải thiện có căn cứ (độ phân giải đầu vào lớn hơn, cắt ô cho vùng xa).
- **Che khuất (theo độ chồng khung) và đông người không làm giảm recall** đáng kể (các khoảng chồng nhau).
- **Ảnh thưa có precision áo thấp hơn** (0,73): ít người nên mỗi khung áo sai chiếm tỉ lệ lớn.
- **Ánh sáng yếu CHƯA được kiểm tra**: độ sáng trung bình của mọi ảnh test chỉ nằm trong khoảng 100-153 (trên 255), phòng
  luôn đủ sáng; chênh giữa nhóm "tối nhất" và "sáng nhất" (0,914 so với 0,942) nằm gần ranh giới tin cậy và không phản ánh
  điều kiện thiếu sáng thật.
- Hiệu chuẩn INT8: bộ xuất RKNN dùng **toàn bộ 1.400 ảnh của tập train** (theo thứ tự tên tệp, không chọn lọc);
  không có ảnh COCO trong tập hiệu chuẩn.

## 5d. INT8 thật trên NPU: lỗi 19 điểm và cách sửa (16/10, KNOWN_ISSUES #42)

Mọi số ở §5, §5b, §5c là **FP32**. Bản INT8 xuất theo cách của Ultralytics (một tensor `[1,6,8400]` chứa cả tọa độ khung và điểm lớp,
dùng chung một thang 8 bit) mất ~19 điểm mAP50-95 vì mép khung bị lượng tử hóa ~2,7 px. **Đã sửa** bằng xuất đầu ra tách
(`scripts/export/export_split.py`: 6 tensor, mỗi scale một cặp khung/điểm lớp; giải mã trên CPU bằng `postprocess.assemble_split_outputs`).
Chấm bằng `scripts/eval/eval_ap.py` trên NPU (lõi 0), trung bình 3 seed ± độ lệch chuẩn, hiệu chuẩn 300 ảnh. Log `runs/split_eval/`.

| Cấu hình (YOLO26n, lab_v6) | Góc 1 mAP50 | Góc 1 mAP50-95 | Góc 2 mAP50 | Góc 2 mAP50-95 |
|---|---|---|---|---|
| FP32 (PyTorch, seed 0) | 0,864 | 0,661 | 0,976 | 0,791 |
| INT8 một tensor (Ultralytics), 640, seed 0 | 0,828 | 0,481 | 0,950 | 0,590 |
| **INT8 đầu ra tách, 640** (3 seed) | 0,869 ± 0,008 | 0,657 ± 0,004 | 0,973 ± 0,001 | 0,784 ± 0,004 |
| INT8 đầu ra tách, 512 (huấn luyện ở 512, 3 seed) | 0,873 ± 0,005 | 0,655 ± 0,007 | 0,971 ± 0,001 | 0,770 ± 0,005 |
| INT8 đầu ra tách, 416 (huấn luyện ở 416, 3 seed) | 0,846 ± 0,011 | 0,620 ± 0,014 | 0,967 ± 0,002 | 0,762 ± 0,001 |

- INT8 đầu ra tách **giữ lại gần hết** độ chính xác FP32 (mAP50-95 mất <= 1 điểm thay vì 17-20 điểm). Parity với ONNX tách:
  recall 0,983, precision 0,998, IoU khớp **0,967** (bản một tensor: 0,84). Báo cáo parity `runs/parity/lab_v6_split_int8` (99 ảnh).
- Luật vùng chấm bằng INT8 **thật** trên board (ngưỡng áo 0,6, 3 seed; `runs/logs/eval_v6_split_int8.txt`): góc 1 bắt người không áo
  81% / 75% / 86%, báo nhầm 8% / 7% / 9%; góc 2 bắt 86% / 90% / 89%, báo nhầm 10-11%; áo thun trắng bị gọi là áo lab 1-3/20.
  Nằm trong mức nhiễu so với FP32 (§5): INT8 không làm luật vùng kém đi.
- **416 kém rõ** (-2,7 điểm mAP50 ở góc 1). **512 ngang 640 về mAP50**, kém ~1,4 điểm mAP50-95 ở góc 2 (ngoài mức nhiễu).
- Độ trễ (cả lời gọi backend: tiền xử lý + NPU + giải mã), `scripts/bench/bench_backend_latency.py`, board rảnh (dịch vụ live tắt), xung mặc định, 3 vòng đan xen,
  `runs/lat_pinned/default.json`: **640 p50 25,9 ms (p95 38-44, p99 47-53); 512 p50 18,4 ms (p95 23-28, p99 35-36)**; NPU 1000 MHz, DDR 1068 MHz, CPU 2,35 GHz cả hai.
  512 nhanh hơn ~7,4 ms (29%), nhất quán. (Các lần đo trước dao động 5-7 ms vì còn tiến trình khác chạy trên board.)
- **Khóa toàn bộ xung** (CPU/DDR/NPU/GPU `performance`, `scripts/bench/bench_pinned.sh`, `runs/lat_pinned/pinned.json`, 3 vòng đan xen): **640 p50 23,4 ms, p95 24,3-24,6, p99 24,9-25,2; 512 p50 16,5 ms, p95 16,8-17,0, p99 16,8-17,9** (DDR 2112 MHz).
  So với xung mặc định: p99 của 640 từ 46-53 ms xuống 25 ms, p99 của 512 từ 35-41 xuống 17 ms; p50 chỉ giảm 2,6 ms (640) và 1,9-3,6 ms (512, tùy lần đo mặc định). Ở xung mặc định DDR nằm ở 1068 MHz và có vòng tụt xuống 528 MHz
  (512 vòng 3: p50 22,8 ms), NPU (1000 MHz) và CPU (2,35 GHz) đã ở trần. **Vậy nguyên nhân dao động độ trễ là xung DDR (`dmc_ondemand`), không phải NPU/CPU/nhiệt**: khóa DDR làm đuôi độ trễ (p95/p99) gần như phẳng.
  Chênh 512 so với 640 không đổi khi khóa xung (-6,9 ms, -30%), nên kết luận 512/640 ở trên vẫn đúng. Nhiệt cuối đo 40,7-46,2 °C (tăng dần theo từng vòng, chưa bão hòa; mỗi vòng ~1 phút).
  Chưa đo: ảnh hưởng lên FPS/p99 end-to-end của cả hệ thống, và chi phí điện/nhiệt khi giữ DDR ở 2112 MHz liên tục.
- Đổi lại ở INT8 (3 seed, `scripts/eval/eval_slices.py --backend rknn`, `runs/slices_int8/`): recall người xa (cao < 188 px) **0,795 -> 0,782** (-1,3 điểm),
  recall áo 0,877 -> 0,864, độ chính xác áo 0,898 -> 0,924, recall người chung 0,914 -> 0,911; mAP50-95 góc 2 kém 1,4 điểm.
  Camera giới hạn 30 FPS nên 512 **không** tăng FPS; lợi ích chỉ là tải NPU, nhiệt, p99. Quyết định: giữ **640** mặc định, 512 là tùy chọn tiết kiệm
  (đổi `weights`/`imgsz` trong config); chưa đo công suất/nhiệt dài hạn và bản khóa xung (`scripts/bench/bench_pinned.sh`).
- Đã thử mà KHÔNG giúp (một tensor): thuật toán KL, hybrid, auto-hybrid, hiệu chuẩn 300 so với 1.400 ảnh; FP16 trên NPU không mất gì nhưng chậm (52 ms).

**Bộ ảnh hiệu chuẩn (16/10, 1 model `lab_v6` seed 0, 640 đầu ra tách, đo trên NPU lõi 1; `runs/calib_eval/`):**

| Bộ hiệu chuẩn | Góc 1 mAP50 / mAP50-95 | Góc 2 mAP50 / mAP50-95 | AP50 áo (góc 1 / 2) | Áo thun trắng bị gọi là áo | Luật vùng @0,6 góc 1: bắt / báo nhầm | Parity: recall / precision / IoU | Thời gian build |
|---|---|---|---|---|---|---|---|
| 1.400 ảnh lab (toàn bộ train) | 0,855 / 0,646 | 0,975 / 0,781 | 0,872 / 0,974 | 2/20 | 80% / 8% | 0,983 / 0,995 / 0,964 | 311 s |
| **300 chọn đều (đang dùng)** | 0,862 / 0,652 | 0,975 / 0,779 | 0,888 / 0,973 | 1/20 | 81% / 8% | 0,983 / 0,998 / 0,967 | chưa ghi |
| 100 ngẫu nhiên | 0,858 / 0,648 | 0,972 / 0,778 | 0,877 / 0,968 | 1/20 | 80% / 7% | 0,979 / 0,996 / 0,967 | 26 s |
| 300 chọn đều + 100 ảnh COCO áo trắng | 0,857 / 0,645 | 0,974 / 0,776 | 0,875 / 0,969 | 3/20 | 79% / 8% | 0,975 / 0,993 / 0,961 | 91 s |

Cả bốn bộ khác nhau <= 0,7 điểm mAP, trong mức nhiễu; **cỡ và thành phần bộ hiệu chuẩn không quyết định độ chính xác INT8** (100 ảnh đã đủ, thêm ảnh âm tính COCO vào
hiệu chuẩn không giảm lỗi áo thun trắng). Giữ 300 ảnh chọn đều. Chỉ 1 model, 1 lần build mỗi bộ: không đủ để phân biệt chênh dưới 1 điểm. Ảnh COCO chỉ dùng để huấn luyện và hiệu chuẩn;
tập test là hai phiên lab khác phiên (`boot014test`, `boot018test`) và 20 khung áo thun trắng của nhóm (`runs/board/N1L_withperson`), không có ảnh COCO.

## 6. Chạy dài trên board

30 phút, hệ thống đầy đủ + trang web + một người xem: 29,0 FPS, tối đa 49,9 °C, RAM tăng **1,15 MB/phút** ở nửa sau
(~70 MB/giờ; trước khi sửa 6-13 MB/phút). Phần còn lại nằm trong heap native, chưa tìm ra nguồn; chạy cả ngày được,
khởi động lại app mỗi ngày (#27, #31, #37). Ngày thu 13/10: board chạy liên tục 5 giờ 21 phút với nguồn hạ áp.

## 6b. Adaptive so với parallel trên board (15/10, KNOWN_ISSUES #39)

Cùng video, hệ thống lab đầy đủ (A + kiểm tra crop + lửa/khói), mỗi lượt ~1 phút.

| | adaptive | parallel |
|---|---|---|
| Lần gọi stage B / C mỗi khung | 0,05 / 0,03 | 2,5 / 1,0 |
| Công NPU (ms/khung) | **24-32** | 61-67 |
| CPU tiến trình | **23-39%** | 58-80% |
| FPS, xung cố định tối đa | **42,0** | 40,1 |
| Trễ p99, xung cố định tối đa | **24,9 ms** | 25,9 ms |
| FPS, bộ điều xung mặc định | 31,6 | 37,0 |

**Đo dài (15 phút/lượt, 3 lượt mỗi chế độ, xung mặc định, video đúng 30 FPS):** adaptive 28,2 FPS, NPU 32 ms/khung,
CPU 35%, **41,5 °C** khi ổn định; parallel 29,8 FPS, NPU 74 ms/khung, CPU 73%, **54,3 °C**. Ba lượt mỗi bên gần như
trùng nhau. Chênh **12,8 °C** là bằng chứng gián tiếp về điện năng (chưa có số watt).

Adaptive làm ít việc hơn ~2,5 lần. Với bộ điều xung mặc định nó TRÔNG chậm hơn vì tải nhẹ làm CPU và bộ nhớ DDR tự
hạ xung (mỗi lần suy luận 30 ms thay vì 25 ms); cố định xung thì adaptive nhanh hơn. Năng lượng: chưa đo.

## 6c. Khóa riêng xung DDR, đo cả hệ thống (16/10)

Chế độ adaptive, model INT8 đầu ra tách 640, tầng B/C bất đồng bộ, video phát lại 30 FPS, 3 vòng đan xen x 240 s, chỉ khác governor `dmc`
(`scripts/bench/ddr_ab_board.sh`, `runs/ddr_ab-20261016-150903/`, tổng hợp `scripts/bench/ddr_ab_summary.py`). Các xung khác để mặc định.

| `dmc` | FPS bền | độ trễ vòng lặp p50 | p99 | max | CPU % | nhiệt cuối |
|---|---|---|---|---|---|---|
| `dmc_ondemand` (mặc định) | 26,84 (26,81-26,86) | 37,2 ms | 44,1 ms | 62,4 ms | 51,4 | 41,6 °C |
| `performance` (DDR 2112 MHz) | **29,02** (29,02-29,03) | 34,1 ms | 41,4 ms | 49,5 ms | 49,5 | 44,7 °C |

- Ba vòng của mỗi bên gần như trùng nhau (FPS lệch <= 0,05), chênh giữa hai bên rõ: **+2,2 FPS (+8%)**, p50 -3,1 ms, p99 -2,7 ms, max -13 ms.
  Ở xung mặc định hệ thống **không theo kịp camera 30 FPS** (26,8: rớt khoảng 11% khung); khóa DDR đưa nó lên sát trần camera (29,0).
- Cái giá: nhiệt cuối **+3,1 °C** (41,6 -> 44,7 °C, sau 4 phút chưa bão hòa). **Chưa đo công suất (W)**: DDR ở 2112 MHz liên tục chắc chắn tốn thêm điện.
- Đã đưa vào dịch vụ systemd (`ExecStartPre=+` khóa DDR, `ExecStopPost=+` trả về `dmc_ondemand`); cần cài lại dịch vụ (`make install-live-service`).
- Chưa đo: 15 phút trở lên (nhiệt bão hòa), công suất, và tác động lên bộ nhớ/độ ổn định lâu dài.

## 6d. Vẽ hình cho dashboard: `video_fps` 0 so với 15 (16/10, KNOWN_ISSUES #45)

Chế độ live (dashboard bật, một người xem kết nối luồng hình), cùng video phát lại 30 FPS, DDR khóa `performance` cả hai bên, 3 vòng đan xen x 120 s
(`scripts/bench/video_fps_ab_board.py`, `runs/video_fps_ab-20261016-155047/`; `video_fps` 0 = vẽ + nén JPEG mọi khung, 15 = tối đa 15 lần/giây):

| `dashboard.video_fps` | FPS bền (3 vòng) | CPU tiến trình |
|---|---|---|
| 0 (mọi khung) | 24,0 (24,2 / 23,7 / 24,1) | 62,8% |
| **15 (mặc định mới)** | **25,9** (25,9 / 25,9 / 26,0) | **56,4%** |

- Hai bên tách rời hẳn (các vòng không chồng nhau): **+1,9 FPS (+8%)**, CPU -6,4 điểm. Trạng thái và cảnh báo vẫn cập nhật mọi khung.
- **Vẫn chưa tới 29 FPS** của bench replay không dashboard (§6c): ở chế độ live còn ~3 FPS bị mất vào việc khác (tổng hợp trạng thái, luồng HTTP, ảnh bằng chứng...), chưa tìm ra phần nào.
  Chưa profile. 26 FPS vẫn đủ cho luật vùng (debounce 3 s), nhưng thấp hơn trần camera 30 FPS.

## 6e. RAM qua đêm, dịch vụ `labsafe-live` (16-17/10, `runs/ramlog/ram-boot001.csv`, `ram-boot002.csv`, tổng hợp `scripts/board/ram_summary.py`)

Board bật 19:22, chạy 10 giờ 14 phút liên tục, ghi mỗi 60 s (PSS mỗi 15 phút) bằng logger ngoài tiến trình. Camera ở phòng tối: FPS camera ~15 cả đêm (trung vị 14,9-15,0), nên FPS không phải thước đo ở đây.

| Đoạn (PID) | thời lượng | RSS @30 / 60 / 120 / 240 phút | RSS cuối | RSS lớn nhất (VmHWM) | dốc giờ đầu | dốc sau giờ đầu | nhiệt (đầu -> cuối, max) |
|---|---|---|---|---|---|---|---|
| 1190 | 360 phút (hết hạn `RuntimeMaxSec=6h`) | 186 / 181 / 211 / 245 MB | 233 MB | 257 (263) MB | +1,12 MB/phút | **+0,10 MB/phút** | 37,9 -> 45,3 °C (49,9) |
| 14304 | 253 phút (chưa hết) | 189 / 232 / 215 / 220 MB | 220 MB | 244 MB | +1,49 MB/phút | **+0,03 MB/phút** | 44,4 -> 50,8 °C (51,8) |

- **Không có rò rỉ không giới hạn:** sau giờ khởi động (tăng nhanh ~100-130 MB trong giờ đầu: bộ cấp phát, bộ bám, cache của thư viện) RSS đi ngang quanh 215-250 MB, có dao động lên xuống
  (245 -> 233 MB cuối đoạn 1), PSS anon cuối đoạn 2 là 159 MB. Mức cao nhất 263 MB, bằng 17% giới hạn 1,5 GB của dịch vụ. Số "1,1 MB/phút" cũ là độ dốc của GIỜ ĐẦU, không phải độ dốc lâu dài.
  Với số này, restart định kỳ 6 giờ không cần thiết để chống rò RAM.
- **Lần restart tự động ở mốc 6 giờ (01:21:43) hoạt động, nhưng `NRestarts` nhảy 0 -> 3 trong khoảng 30 s** (ứng dụng mới chạy được ở 01:22:13): hai lần khởi động đầu thất bại. Nghi camera chưa được trả lại hoặc trang web chưa
  nhả cổng khi tiến trình cũ đang tắt. **Không xác minh được**: nhật ký `journalctl` của board chỉ còn khoảng 40 dòng cuối (không bền, bị cắt sớm) nên mất bằng chứng. 3/5 lần là sát giới hạn `StartLimitBurst=5` trong 5 phút:
  thêm một lần thất bại nữa dịch vụ sẽ bỏ cuộc. Đã thêm thử lại mở camera 20 s trong `CameraSource` (không dùng ngân sách restart của systemd); **chưa kiểm trên board**.
- **Cảnh báo qua đêm không phải phép đo báo giả:** 266 ảnh bằng chứng trong 10 giờ (97 + 169), mẫu 12 ảnh cho thấy người nằm ngủ trên sàn trong vùng cấm, không mặc áo: là phát hiện đúng theo luật. ~400 mã track chỉ trong 4 giờ
  và cooldown tính theo mã track nên một người nằm yên bị báo lại mỗi phút. Có một cảnh báo té ngã `still_down` lúc 05:22 (người nằm): chưa kiểm là đúng hay sai.

## 6f. Cảnh báo theo SỰ KIỆN trên video quay có kịch bản (17/10, phòng nhà, `scripts/eval/eval_events.py`, `runs/events/home_grid.json`, `home_partial.json`)

5 cảnh (`EV_01,02,03,05,06`), 1 người (áo lab 2), 30 FPS, tổng 8 phút, đáp án là cue theo giây ghi cùng chỉ số khung. Detector chạy một lần, chuỗi quyết định phát lại cho từng cấu hình trên giờ mô phỏng.
Chỉ tính cảnh báo vùng áo lab (không đo té ngã). 7 vi phạm kỳ vọng (người không áo ở trong vùng >= 4 s). **Dữ liệu nhỏ: 7 sự kiện, 8 phút, một người, một phòng nhà; số báo giả mỗi giờ chỉ để so sánh giữa cấu hình, không phải số để công bố.**

| `grace_s` | partial | bắt được | bỏ sót | báo giả (8 phút) | trễ trung vị từ cue | báo lặp/sự kiện |
|---|---|---|---|---|---|---|
| 0 | tắt | 7/7 | 0 | 3 | 6,84 s | 0 |
| 0,5 / 1,0 / 1,5 | tắt | 7/7 | 0 | **5** | 5,11 s | 0-0,14 |
| 0,5 / 1,0 / 1,5 | **< 0,3** | 7/7 | 0 | **3** | 5,24 s | 0 |
| 0,5 / 1,0 / 1,5 | < 0,4 | 7/7 | 0 | 3 | 5,34 s | 0 |

- **Ngưỡng nhận áo (0,5 / 0,6 / 0,7) và thời gian nhớ áo (1 / 2 / 3 s) không đổi kết quả** trên tập này (36 cấu hình, cùng 7/7 và 3-6 báo giả): chưa có lý do đổi 0,6 và 2 s.
- **`grace_s` 0,5 / 1,0 / 1,5 cho kết quả giống nhau**; so với 0 nó giảm độ trễ trung vị 1,7 s (không bị đứt cửa sổ 3 s khi bộ bám chớp), nhưng nối các lần chớp ngắn thành báo giả khi có người lộ một phần.
- **Kiểm từng báo giả bằng ảnh** (khung tại thời điểm báo): (a) `EV_01` 89,7 s: người MẶC áo trắng rõ ràng vẫn bị báo vi phạm (mất khung áo ở khung đó): **lỗi thật của hệ thống**, 1 trong 7+ sự kiện; (b) `EV_02` 61,4 s: người băng qua vùng lâu hơn 2 s theo cue (lệch cue của người quay, hệ thống báo đúng);
  (c) `EV_06` 15,2 s: người ngồi ở mép trái nằm TRONG đa giác (lệch kịch bản, báo đúng theo hình học); (d) `EV_01` 117,3 s và `EV_03` 106,6 s (chỉ khi `grace_s` > 0): chỉ lộ cái đầu ở góc dưới khung (người điều khiển ngồi cạnh máy), hộp chạm mép dưới nên điểm chân bị ghim vào mép và nằm trong vùng: **lỗi thật của hệ thống** (#47).
- **Sửa:** `zones.partial_min_h` (mặc định 0 = tắt): hộp chạm mép dưới và thấp hơn tỉ lệ đó của chiều cao khung thì không xét vùng. Với 0,3 hai báo giả (d) biến mất, độ trễ vẫn ~5,2 s. Đã bật 0,3 trong `configs/live/dashboard_lab_board.yaml`.
- **Còn lại 3 báo giả = 1 lỗi hệ thống (mất áo) + 2 lệch kịch bản.** Chưa có hai người che nhau (cảnh 04), chưa có té ngã thật (cảnh 07).

## 6g. 24 giờ liên tục và lần restart tự động (17-18/10, `runs/ramlog/ram-boot002.csv`, `runs/labsafe-live.log`)

Dịch vụ `labsafe-live` (cấu hình mới: `RuntimeMaxSec=24h`, log riêng, mở camera có thử lại) chạy từ 11:23:24 ngày 17/10 đến lần restart tự động 11:23:24 ngày 18/10 (1.440 phút), phòng nhà có người đi lại.

| | Giá trị |
|---|---|
| RSS lúc 30 / 60 / 120 / 240 / 360 phút | 209 / 215 / 208 / 205 / 208 MB |
| RSS cuối (24 giờ) / lớn nhất (VmHWM) | 218 MB / 269 MB (18% giới hạn 1,5 GB) |
| dốc giờ đầu / sau giờ đầu | +0,74 MB/phút / **+0,024 MB/phút** (23 giờ) |
| PSS / anon / private dirty (đầu -> cuối) | 178 / 124 / 134 -> 211 / 156 / 166 MB |
| FPS | ~29 ban ngày, ~15 ban đêm (phòng tối, camera tự giảm); không có mẫu nào dưới 5 |
| nhiệt | 50,8 -> 53,6 °C, lớn nhất 54,5 °C |
| lỗi | 0 (`"errors": 0`), không có Traceback hay lần mở lại camera |

- **Lần restart tự động hết hạn 24 giờ chỉ tốn 1 lần khởi động** (`NRestarts=1`, trước đây 3, #46): 11:23:24 yêu cầu dừng, 11:23:25 đóng xong sạch, 11:23:31 tiến trình mới chạy, 11:23:33 mở được camera: **khoảng 9 giây không giám sát**, toàn bộ diễn biến có trong `runs/labsafe-live.log`. Tiến trình mới lặp lại cách khởi động quen thuộc (RSS 138 -> 214 MB trong 25 phút, rồi đi ngang).
- **RAM phẳng sau giờ khởi động trong 23 giờ liên tục**: số "1,1 MB/phút" cũ là độ dốc của giờ đầu. Kết luận này đo ở 24 giờ và một cấu hình, chưa phải nhiều ngày.
- **Cảnh báo cả ngày** (đến 07:11 ngày 18): 435 cảnh báo vùng, 43 cảnh báo té ngã, 322 ảnh bằng chứng, nhưng cảnh có người đi lại nên đây không phải phép đo báo giả. Thư mục `runs/` trên board nặng 587 MB sau 2 ngày (31 thư mục chạy), đĩa còn 51 GB: cần dọn khi chạy nhiều tuần.

## 6h. Chỉ phán áo khi thấy được quần áo (21/10, KNOWN_ISSUES #50)

`zones.visibility` = {min_h 0,22, tall_aspect 2, min_w_px 24, max_occlusion 0,4}. FP32 `lab_v6`, ngưỡng áo 0,6, `scripts/eval/eval_zone_rule.py --vis`, log `runs/vis/run.log`.

| Tập | quy tắc | người được chấm | bắt người không áo | người có áo bị báo nhầm |
|---|---|---|---|---|
| góc 1 | tắt | 644 | 292/376 (78%) | 12/172 (7%) |
| góc 1 | bật | 518 | 247/317 (78%) | 9/163 (6%) |
| góc 2 | tắt | 1.080 | 76/88 (86%) | 79/942 (8%) |
| góc 2 | bật | 709 | 12/18 (67%) | 16/684 (2%) |

Video demo lab đông người, cùng 170 s: 31 cảnh báo -> 6, cả 6 đúng. Chấm sự kiện ở nhà: không đổi. Hạn chế: người bị che không bị phán cho tới khi lộ thân; bộ nhãn
không phân biệt "không mặc áo" với "áo bị che" nên số bắt ở góc 2 chỉ còn 18 người được chấm, quá ít để kết luận.

## 7. Chưa đo (đừng viết như đã làm)

- **Năng lượng**: đã thử đo bằng đồng hồ USB-C (15/10) nhưng hai lần đo đều hỏng (người gõ số trễ; board khởi động lại giữa lượt parallel, #41) -> **chưa có số W/Wh hợp lệ**. Công cụ `make energy` đã sửa, chạy lại được bất cứ lúc nào.
- Adaptive so với parallel: đã đo tốc độ, công tính toán và nhiệt độ ổn định (6b); **chưa đo năng lượng (watt)**.
- **Luật vùng trên video liên tục:** đã có phép đo ban đầu (§6f), nhưng chỉ 7
  sự kiện, 8 phút, một người và quay ở nhà. Không dùng 7/7 như độ chính xác
  tổng quát; cần lặp với nhiều người và nhiều phiên trong lab.
- **Té ngã**: chưa thử trên video thật có người té.
- **Lửa/khói ở lab**: model Stage C có (mAP50 0,72 trên bộ công khai đã làm sạch, #29) nhưng chưa thử trong lab.
- **Tính tổng quát:** hai góc camera trong một phòng; test góc 2 có ít người
  không áo. Ngưỡng áo đã được khảo sát trên chính các tập test nên các số luật
  vùng hơi lạc quan; lần đánh giá tiếp theo cần một validation set riêng.
- **Ánh sáng yếu / ngược sáng**: tập test không có ca này (độ sáng 100-153), chưa đo.
- **Dashboard mới:** đã kiểm bằng trình duyệt với server giả, chưa chạy lại
  toàn bộ trên board thật và máy chiếu sau lần thiết kế giao diện 20/10.

## 6i. Camera ở góc mới tại lab, rồi đặt lại về góc huấn luyện (23/10, KNOWN_ISSUES #52)

Camera thật của board, qua đường hầm SSH. Chạy lại model trên khung lấy từ luồng ở ngưỡng phát hiện 0,1 (ONNX FP32; board dùng 0,4). Người "mặc áo" = người đứng giữa khung, một người, 8 giây liên tiếp (122 khung).

| Model | khung có áo >= 0,6 | khung có áo >= 0,3 | người khác bị gán áo (>= 0,6 / >= 0,3) |
|---|---|---|---|
| lab_v6 (đang dùng) | 0/122 (0%) | 0/122 (0%) | 0/160 / 0/160 |
| lab_v5 | 24/122 (20%) | 73/122 (60%) | 0/336 / 3/336 |
| lab_v3 | 20/121 (17%) | 61/121 (50%) | 0/412 / 15/412 |

Bản ONNX chuẩn của lab_v6 cũng không có khung áo nào (41 khung, ngưỡng 0,05); trên board `coat_score` = None liên tục. Mô tả góc: dữ liệu huấn luyện là người xa, thấy trọn thân, nền tủ gỗ;
góc mới camera rất gần, cắt mất đầu, nhìn từ trên xuống. Chưa tách riêng từng yếu tố; giả thuyết "mẫu âm tính COCO làm lab_v6 cứng hơn ở góc lạ" chưa kiểm.

Sau khi đặt camera về khung hình giống dữ liệu huấn luyện: bo mạch thật, cấu hình sản phẩm (lab_v6, 0,6, nhớ áo 2 s), 25 giây, thăm dò `/status` mỗi 0,4 s, 6 người.

| Nhóm | Người | Kết quả |
|---|---|---|
| mặc áo lab (4) | #130, #127, #92, #134 | khung áo >= 0,6 ở 92/94 lần thấy áo (98%), điểm 0,56-0,94; "đang mặc áo" 98-100% |
| không áo (3) | #102, #103, #125 | điểm áo tối đa 0,55, 0/38 lần >= 0,6 |
| áo thun/polo trắng (2) | #104, #131 | không có khung áo nào |
| mượn áo của người đứng sát | #103, #125 | "đang mặc áo" 48% và 32% số lần thăm dò |

Hạn chế: một phiên, 25 giây, sáu người, các lần thăm dò cách 0,4 s nên không độc lập; không có đáp án gán tay; KHÔNG phải độ chính xác. Dùng để kết luận lỗi nằm ở góc camera, không phải ngưỡng.

## 6j. Độ tin cậy của việc "đã vào vùng" (điểm chân), 24/10, `scripts/eval/eval_zone_foot.py`

Tập `boot014test` (phiên khác lúc huấn luyện, góc 1), vùng `configs/zones_lab.yaml` (vẽ trên góc boot014), model `lab_v6` FP32, ngưỡng người 0,4. Mỗi người có nhãn tay được ghép với khung dự đoán có IoU >= 0,5; so quyết định "chân trong vùng" của khung dự đoán với của khung nhãn tay.

| Chỉ số | Kết quả |
|---|---|
| Người có nhãn / ghép được | 644 / 550 (85%); 94 người (15%) không được phát hiện đủ tốt nên KHÔNG có quyết định |
| Sai số điểm chân (điểm ảnh, ảnh 640x480) | trung vị 6,6; p90 35,7; p95 62,4 |
| Cùng quyết định trong/ngoài vùng | 531/550 (97%); chân thấy rõ 97%, chân bị cắt ở đáy khung 96% |
| Theo khoảng cách chân tới mép vùng | 10-30 px: **87%** (82/94); 30-80 px: 99%; > 80 px: 100%. (Nhóm 0-10 px, 98%, phần lớn là người bị cắt ở đáy khung, được kéo vào trong nên không phản ánh vùng sát mép) |
| Người thật sự TRONG vùng (454) | hệ thống nói "trong": 452 (100%) |
| Người thật sự NGOÀI vùng (96) | hệ thống nói "trong" (báo vào vùng nhầm): 17 (18%), phần lớn chân cách mép dưới 30 px |

Hạn chế: chuẩn so sánh là đáy khung NHÃN TAY chứ không phải điểm chạm sàn thật, nên chân bị bàn/người che không bị phạt; mỗi ảnh xét riêng (không có chống rung, bộ bám); một vùng, một góc; tập chủ yếu là người trong vùng (454/550). Kết luận dùng được: quyết định ổn khi chân cách mép vùng hơn khoảng 30 px; sát mép thì có thể đổi qua lại (trong dịch vụ thật có chống rung 3 s + ân hạn 1 s làm dịu).

## 6k. Công suất điện bo mạch, 29/10, `scripts/bench/energy_session.py`, `docs/energy-2026-10-29/`

Đồng hồ USB-C KWS-1902C giữa bộ sạc và bo mạch; số mWh cộng dồn đọc ở đầu/cuối mỗi pha (người gõ). Kiểm đồng hồ trước (`scripts/bench/meter_check.py`): nghỉ 73 s = 3,54 W = V×A; tải CPU hết công suất: 3,2 → 8,3 W hiển thị. Số W hiển thị thấp hơn V×A khoảng 10--13%, số cộng dồn khớp V×A, nên sai số tuyệt đối khoảng ±10%, so sánh tương đối đáng tin hơn.

| Pha | Thời gian | Công suất TB | Trên mức nghỉ | FPS (capture_fps) | mJ/khung |
|---|---|---|---|---|---|
| Nghỉ (đo lại, 5 phút) | 5,3 phút, 75 mWh | 0,86 W | --- | --- | --- |
| Sản phẩm (`dashboard_lab_board.yaml`, tầng A, replay) | 10,2 phút, 375 mWh | 2,20 W | +1,34 | 27,3 | 81 |
| Thích nghi (L3, B/C bất đồng bộ) | 16,2 phút, 537 mWh | 1,98 W | +1,12 | 28,9 | 69 |
| Chạy hết (A+B+C mỗi khung) | 15,2 phút, 847 mWh | 3,34 W | +2,48 | 29,7 | 113 |

- Thích nghi thấp hơn chạy hết 1,36 W (−41%), năng lượng/khung −39% ở cùng ~29 FPS. Vượt sai số đồng hồ nhưng chỉ có một cặp.
- Sản phẩm (2,20 W) và thích nghi (1,98 W) chênh trong sai số: không kết luận. Sản phẩm có thêm dashboard/vẽ khung/LED/OLED.
- Số W giữa chừng đọc trên đồng hồ: 2,03 / 2,00 / 3,50 W, lệch tối đa 8% so với số cộng dồn.
- Hạn chế: video phát lại (không camera thật; thêm khoảng 0,3--0,5 W theo ước lượng, chưa đo riêng), mỗi pha một lần, nhiệt độ cuối pha chạy hết 59 °C.
- Lần đo đầu của pha nghỉ (`readings_main.csv`) hỏng do gõ số W vào ô mWh (ra −6,96 W) và đã đo lại (`readings_idle_redo.csv`); ước lượng gián tiếp từ khoảng cách giữa các lần đọc (0,8--1,0 W) khớp.
- Đo watt khi chạy dịch vụ thật có camera: 3,54 W (phép thử đồng hồ, 73 s).

