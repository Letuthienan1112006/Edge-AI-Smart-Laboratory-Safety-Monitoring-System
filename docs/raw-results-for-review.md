# Số đo thô cho người đánh giá (16/10) — chỉ số liệu, không code

Mọi số đến từ file trong `runs/` (đường dẫn ghi ở từng mục). Chỗ nào CHƯA đo thì ghi "chưa đo".
Lưu ý: "độ trễ" ở mục 1 là **cả lời gọi backend** (tiền xử lý + NPU + giải mã), không phải riêng lệnh `invoke` của NPU.

## 1. 512 và 640, xung mặc định và khóa xung (`runs/lat_pinned/{default,pinned}.json`)
Board rảnh (dịch vụ live tắt), 300 lần mỗi vòng sau 150 lần khởi động, 3 vòng đan xen, nhân CPU lớn 4-7. Khóa = CPU/DDR/NPU/GPU `performance`.

| res | governor | vòng | NPU MHz | DDR MHz | CPU4 MHz | p50 ms | p95 ms | p99 ms | mean ms | nhiệt cuối °C |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | mặc định | 1 | 1000 | 1068 | 2352 | 18,38 | 28,28 | 36,20 | 19,39 | 40,7 |
| 512 | mặc định | 2 | 1000 | 1068 | 2352 | 19,06 | 34,43 | 40,85 | 21,72 | 41,6 |
| 512 | mặc định | 3 | 1000 | 528 | 2352 | 22,81 | 33,84 | 41,54 | 23,48 | 41,6 |
| 512 | khóa | 1 | 1000 | 2112 | 2352 | 16,55 | 16,98 | 17,87 | 15,93 | 44,4 |
| 512 | khóa | 2 | 1000 | 2112 | 2352 | 16,50 | 16,77 | 16,89 | 16,25 | 45,3 |
| 512 | khóa | 3 | 1000 | 2112 | 2352 | 16,50 | 16,75 | 16,83 | 16,26 | 46,2 |
| 640 | mặc định | 1 | 1000 | 1068 | 2352 | 25,76 | 41,94 | 45,84 | 27,35 | 40,7 |
| 640 | mặc định | 2 | 1000 | 1068 | 2352 | 25,98 | 37,86 | 45,91 | 27,06 | 41,6 |
| 640 | mặc định | 3 | 1000 | 1068 | 2352 | 26,14 | 44,74 | 47,09 | 30,25 | 41,6 |
| 640 | khóa | 1 | 1000 | 2112 | 2352 | 23,36 | 24,64 | 25,09 | 23,30 | 43,5 |
| 640 | khóa | 2 | 1000 | 2112 | 2352 | 23,41 | 24,32 | 24,85 | 23,20 | 44,4 |
| 640 | khóa | 3 | 1000 | 2112 | 2352 | 23,49 | 24,33 | 25,19 | 23,44 | 45,3 |

(CPU0 khóa ở 1800 MHz; nhiệt đầu chạy không được ghi riêng, mỗi vòng ~1 phút.)

Độ chính xác INT8 đầu ra tách, tập test 2 góc (3 seed, trung bình ± độ lệch chuẩn; `runs/split_eval/`, `runs/slices_int8/`):

| res | góc 1 mAP50 / 50-95 | góc 2 mAP50 / 50-95 | recall người xa (<188 px) | recall người gần (>=278 px) | recall người chung | recall áo | precision áo |
|---|---|---|---|---|---|---|---|
| 640 | 0,869±0,008 / 0,657±0,004 | 0,973±0,001 / 0,784±0,004 | 0,795±0,005 | 0,980±0,001 | 0,914±0,001 | 0,877±0,005 | 0,898±0,014 |
| 512 | 0,873±0,005 / 0,655±0,007 | 0,971±0,001 / 0,770±0,005 | 0,782±0,011 | 0,979±0,001 | 0,911±0,003 | 0,864±0,004 | 0,924±0,005 |

FPS toàn hệ thống (adaptive, 640 tách, video phát lại 30 FPS, 3 vòng x 240 s; `runs/ddr_ab-20261016-150903/`): chỉ khóa DDR —

| DDR | FPS | p50 vòng lặp ms | p99 ms | max ms | CPU % | nhiệt cuối °C |
|---|---|---|---|---|---|---|
| mặc định | 26,84 (26,81-26,86) | 37,18 | 44,08 | 62,36 | 51,4 | 41,6 |
| khóa 2112 MHz | 29,02 (29,02-29,03) | 34,12 | 41,37 | 49,49 | 49,5 | 44,7 |

FPS toàn hệ thống với **512**: **chưa đo**. FPS/độ trễ ở trên đo cho 640.

## 2. Bốn bộ ảnh hiệu chuẩn (1 model lab_v6 seed 0, 640 tách; `runs/calib_eval/`, `runs/slices_calib/`)
Ngưỡng người 0,40, áo 0,60, IoU 0,5; KTC 95% bootstrap theo ảnh trong ngoặc cho recall/precision áo (gộp 2 tập test).

| Bộ | mAP50 / 50-95 góc 1 | mAP50 / 50-95 góc 2 | recall áo | precision áo | áo thun trắng bị gọi là áo | parity recall / precision / IoU | build | kích thước |
|---|---|---|---|---|---|---|---|---|
| 1.400 ảnh lab | 0,855 / 0,646 | 0,975 / 0,781 | 0,889 (0,871-0,909) | 0,885 (0,859-0,910) | 2/20 | 0,983 / 0,995 / 0,964 | 311 s | 4,24 MB |
| 100 ngẫu nhiên | 0,858 / 0,648 | 0,972 / 0,778 | 0,892 (0,874-0,911) | 0,882 (0,856-0,907) | 1/20 | 0,979 / 0,996 / 0,967 | 26 s | 4,24 MB |
| 300 chọn đều | 0,862 / 0,652 | 0,975 / 0,779 | 0,882 (0,865-0,901) | 0,886 (0,859-0,908) | 1/20 | 0,983 / 0,998 / 0,967 | chưa ghi | 4,24 MB |
| 300 + 100 COCO áo trắng | 0,857 / 0,645 | 0,974 / 0,776 | 0,882 (0,863-0,902) | 0,889 (0,864-0,911) | 3/20 | 0,975 / 0,993 / 0,961 | 91 s | 4,24 MB |

Luật vùng @0,6 (người không áo bị bắt / người có áo bị báo nhầm): C1 góc 1 299/374 và 13/172, góc 2 74/85 và 89/942; C2 303/374 và 14/172, 72/84 và 92/942;
C3 298/373 và 12/172, 77/87 và 83/940; C4 293/371 và 13/172, 71/82 và 90/939. Mỗi bộ chỉ 1 lần build: chênh <1 điểm không phân biệt được. Ảnh COCO không nằm trong tập test.

## 3. Sự kiện theo video và `grace_s`: **chưa đo**
Chưa có video 30 FPS có đáp án. Đã có: kịch bản `configs/collection_plan_events.yaml` (7 cảnh, ~12 phút, cue theo giây ghi cùng chỉ số khung), bộ chống rung có `grace_s` (hiện đặt 1,0 để sửa lỗi #43, KHÔNG phải giá trị đã tối ưu).
Số duy nhất liên quan: trên video phát lại ngắn lặp, `grace_s`=0 thì cảnh báo KHÔNG BAO GIỜ nổ (0 cảnh báo trong 60-80 s có người vi phạm liên tục), `grace_s`=1,0 thì nổ sau 3 s, có ảnh bằng chứng. Chưa có recall/FA mỗi giờ/độ trễ/báo lặp.
Quan sát chưa định lượng: trong phòng đông người, dịch vụ live ghi nhiều cảnh báo cách nhau vài giây với mã track khác nhau (cooldown theo mã track).

## 4. RAM qua đêm: **chưa đo**
Chưa có phiên chạy dài nào với model đầu ra tách. Số cũ (model trước, app chạy tay): rò ~1,1 MB/phút sau khi nâng librknnrt 2.3.2 (KNOWN_ISSUES #27/#31), chưa bão hòa, nguyên nhân chưa rõ.
Hiện tại: dịch vụ chạy từ 16:06, RSS 169 MB sau 15 phút, `MemoryCurrent` 130,7 MB, cgroup giới hạn 1,5 GB.

## 5. systemd (lúc 16:21, board bật 5 giờ 7 phút)
`NRestarts=0`, `Result=success`, `journalctl` không có Traceback/kill/OOM trong 3 giờ qua. Các lần Start/Stop trong nhật ký đều do tôi/người dùng chủ động (cài đặt, A/B DDR, A/B video_fps).
Dịch vụ mới chạy 15 phút liên tục: **chưa có số liệu 24 giờ**, chưa kiểm crash-loop, chưa kiểm `RuntimeMaxSec=6h` hay `MemoryMax` thực sự kích hoạt. Tên dịch vụ là `labsafe-live`.
Đã kiểm bằng test (không bằng rút cáp thật): camera dừng giao khung >3 s thì tự mở lại thiết bị (`tests/test_camera_stall.py`).
