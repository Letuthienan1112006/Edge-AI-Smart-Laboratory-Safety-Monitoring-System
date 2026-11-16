# Thuật toán, luồng xử lý, model hiện tại — và những chỗ có thể cải thiện

Tài liệu cho người muốn nghiên cứu cải thiện. Mỗi mục ghi **file chứa code** để đọc. Số liệu ở `docs/results.md`.

## 1. Model

| | |
|---|---|
| Kiến trúc | YOLO26n (Ultralytics), ~2,4 triệu tham số, ~5,3 GFLOPs. Detector một tầng, không anchor |
| Lớp (tầng A) | 2 lớp: `person`, `lab_coat` (`configs/lab_classes.yaml`) |
| Khởi tạo | trọng số COCO `yolo26n.pt`, tinh chỉnh toàn bộ |
| Huấn luyện | 60 epoch, ảnh 640, batch 16, công thức mặc định của Ultralytics (mosaic, HSV, lật ngang, co giãn...), không chỉnh tay. 3 seed. `scripts/train_baseline.py` |
| Dữ liệu (lab_v6) | ~1.400 ảnh lab (3 buổi, 2 góc camera) + 238 ảnh COCO người mặc áo trắng làm mẫu âm tính cho `lab_coat` |
| Xuất ra NPU | ONNX -> RKNN-Toolkit2 2.3.2, **INT8 sau huấn luyện (PTQ)**, hiệu chuẩn bằng ảnh tập train. `scripts/export_rknn.py` |
| Đầu ra khi chạy | KHÔNG dùng đầu "không NMS" (end2end): nó làm sập NPU RK3588 (#2). Dùng đầu thường + NMS bằng numpy tự viết |
| Tiền xử lý | letterbox 640x640 (đệm xám), NHWC uint8; chuẩn hóa (chia 255) nằm TRONG model RKNN. `src/labsafe/models/postprocess.py` |
| Giải mã | đọc 4 kênh khung + điểm lớp, bỏ khung dưới ngưỡng (người 0,4), **chỉ giải mã lớp cần giữ** (`class_subset`: 2,1 -> 0,38 ms), NMS tham lam theo lớp (IoU 0,5) |
| Model phụ | Tầng B: YOLO26n trên ảnh cắt người 320 (kính/găng, huấn luyện trên ảnh công trường; chỉ làm khối lượng đo, không thuộc sản phẩm). Tầng C: YOLO26n 416 lửa/khói (mAP50 0,72 bộ công khai) |

## 2. Luồng xử lý một khung (`src/labsafe/pipeline/router.py`, `app.py`)

1. **Camera** (`io/camera.py`): luồng riêng đọc MJPG 640x480; `read(wait_new=True)` chờ ảnh MỚI để không suy luận lại ảnh cũ.
2. **Tầng A** (mỗi khung, NPU lõi 0): ra khung `person` và `lab_coat` kèm điểm.
3. **Tách loại:** khung `person` thành người; khung `lab_coat` là "thuộc tính mặc".
4. **Bộ bám** (`pipeline/tracker.py`): ghép tham lam theo IoU (ngưỡng 0,3); chưa khớp thì thử khoảng cách tâm (<=1,2 đường chéo khung); track sống tối đa 3 s không thấy.
5. **Gán áo** (`router._attach_attire`): mỗi khung áo gán cho **một** người có phần diện tích áo nằm trong khung người lớn nhất (>=50%). Chỉ nhận áo có điểm >= `attire_min_conf` (0,6). Người "đang mặc" nếu thấy áo trong 2 s gần nhất.
6. **Luật vùng** (`pipeline/zones.py`): điểm chân = giữa cạnh dưới khung; thuật toán tia xét có nằm trong đa giác (tọa độ chuẩn hóa 0..1). Vùng `requires: [lab_coat]`: ai vào mà không mặc -> `zone_ppe_violation`.
7. **Té ngã** (`pipeline/fall.py`): máy trạng thái 3 điều kiện: khung rộng hơn cao (tỉ lệ >=1,15) VÀ tâm tụt nhanh (18% trong 1 s) VÀ nằm yên >=2,5 s.
8. **Tầng B** (bất đồng bộ, NPU lõi 1): cắt ảnh từng người, 2 s/lần/người. **Tầng C** (bất đồng bộ, lõi 2): cả khung, 1 Hz. Hàng đợi có giới hạn, quá tải thì bỏ (`pipeline/async_stage.py`).
9. **Chống rung / chống lặp** (`pipeline/decision.py`): sự kiện phải kéo dài 3 s mới báo; cùng một người im 60 s sau lần báo.
10. **Đầu ra:** cảnh báo + ảnh bằng chứng (`io/evidence.py`), trang web (`io/dashboard.py`), còi/đèn (`io/alerts.py`).

Chế độ **thích nghi** (mặc định): tầng B chỉ chạy khi có người đến hạn, tầng C theo chu kỳ. Chế độ **parallel**: B và C mỗi khung (để so sánh).

## 3. Quy trình dữ liệu và đánh giá

- Thu: bo mạch chụp 1 ảnh/4 s, bỏ khung tĩnh/tối (`scripts/capture_interval.py`).
- Chọn ảnh gán: điểm xa nhất trên ảnh nhỏ 48x36 + ảnh model lưỡng lự về áo + bỏ khung không phải góc camera (`scripts/make_batch.py`).
- Nhãn nháp bằng model, người sửa trên Roboflow; nhập theo **phiên** (`scripts/import_roboflow.py`), chia train/test theo phiên (`scripts/build_lab_dataset.py`).
- Chấm: AP một bộ chấm chung cho mọi loại chính xác (`scripts/eval_ap.py`); luật vùng chấm theo từng người (`scripts/eval_zone_rule.py`); mỗi cấu hình 3 seed.

## 4. Những chỗ có thể cải thiện (xếp theo đáng làm)

| # | Ý tưởng | Vì sao đáng làm / hiện đang yếu ở đâu | Cách đo | Công sức |
|---|---|---|---|---|
| 1 | **Điểm áo theo từng người, làm mượt theo thời gian** (EMA + trễ hai ngưỡng) thay cho "thấy áo trong 2 s" | Hiện một khung áo sai (lúc người xoay, bị che) cho người vi phạm lọt, hoặc ngược lại | số cảnh báo sai/giây và độ trễ trên video thật | thấp |
| 2 | **Bộ bám tốt hơn**: kiểu ByteTrack (dùng cả khung điểm thấp) + bộ lọc Kalman | Bám tham lam đổi ID khi che nhau: soak đầu có 24 ID cho 2-4 người, buộc phải nới nhớ 3 s | số ID mỗi người, IDF1 trên video có nhãn | trung bình |
| 3 | **Tầng B thành bộ phân loại "có áo / không áo" trên ảnh cắt người** (MobileNet nhỏ), chỉ chạy khi tầng A lưỡng lự (điểm 0,3-0,8) | Tầng B hiện chỉ là khối lượng đo. Gating đã có sẵn, hợp với đề tài. Kiểm chứng chéo giảm báo nhầm | mức bắt/báo nhầm của luật vùng, chi phí NPU thêm | trung bình |
| 4 | **Học bán giám sát với ảnh chưa gán** (khoảng 3.500 ảnh `boot014`/`boot018`): nhãn giả chỉ giữ khi tự tin cao, hoặc mean-teacher | Đây là "AI tự học" đúng nghĩa; đã thử nhãn giả thô thì học lại lỗi, nên cần lọc | AP ở tập test phiên khác | trung bình |
| 5 | **Chưng cất từ model lớn** (YOLO26s/l làm thầy cho 26n) | Model lớn hơn chỉ hơn ~2 điểm nhưng chậm 2-3 lần; chưng cất mang lại phần đó mà không tốn thời gian chạy | AP của 26n sau chưng cất | trung bình |
| 6 | **Giảm tụt điểm INT8**: hiệu chuẩn bằng ảnh cả hai góc, thuật toán `mmse`, lượng tử hóa hỗn hợp, QAT | Chưa đo mức tụt so với đáp án thật (chỉ có parity). Cần board | `eval_ap.py` FP32 so với INT8 | thấp |
| 7 | **Cho tầng A tự "ngủ" khi phòng trống** (so khung hình, bỏ suy luận khi không có chuyển động) | Mới gating tầng B/C; trong dữ liệu mỗi khung gần như luôn có người nên tầng A không tiết kiệm; ban đêm có thể tiết kiệm lớn | công NPU và nhiệt độ ở mức chiếm chỗ thấp | thấp |
| 8 | **Té ngã bằng tư thế khung xương** (YOLO-pose/RTMPose) | Luật hiện dựa tỉ lệ khung, dễ nhầm với cúi/ngồi xổm; chưa kiểm trên video thật | độ chính xác/độ nhạy trên video dàn dựng | cao |
| 9 | **Vị trí chân bền hơn**: khi chân bị cắt ở mép khung, ước lượng từ chiều cao; hoặc mặt phẳng sàn (homography) | Người gần camera thường bị cắt chân (đã vá một lỗi liên quan #35) | số lần xét sai vùng ở rìa dưới | thấp |
| 10 | **Vòng tự học trên bo mạch:** ghi lại khung model lưỡng lự, gom thành lô gán nhãn tự động | Nối "thu -> chọn -> gán" thành một vòng liền mạch, giảm công người | số ảnh khó/giờ, độ chính xác sau mỗi vòng | trung bình |
| 11 | **Độ phân giải/cắt ô cho người xa** (tầng A 800 hoặc cắt ô vùng xa) | Người xa nhỏ, bắt kém; đánh đổi tốc độ | AP theo cỡ người | thấp-trung bình |
| 12 | **Mở rộng đánh giá:** khoảng tin cậy (bootstrap), tách theo lát cắt (xa/gần, đông/thưa, sáng/tối), thêm phòng/giờ khác | Tập test nhỏ (vài chục người không áo), dao động ±4 điểm; hội đồng sẽ hỏi độ tin cậy | khoảng 95% cho mỗi số | thấp |

Nguyên tắc đã rút ra (đừng bỏ qua): tập test phải **khác phiên**; mỗi cấu hình **3 seed**, chênh <~4 điểm thì không kết luận; so tốc độ giữa hai chế độ phải **khóa xung**; sửa lỗi bằng **dữ liệu đúng loại** có tác dụng hơn đổi model lớn hơn.
