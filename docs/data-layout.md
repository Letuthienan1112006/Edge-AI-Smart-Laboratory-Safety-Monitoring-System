# Dữ liệu

## Bố cục

```
data/raw/          dataset tải về, KHÔNG sửa
data/interim/      trung gian (crops kiểm tra nhãn, ảnh đã lọc)
data/processed/    dataset cuối theo format YOLO + data.yaml
```

## Nguyên tắc bắt buộc

1. **Không train trước khi chạy `scripts/data/verify_dataset.py`.** Tên lớp trong
   `data.yaml` chỉ là một *lời khai* về ý nghĩa của các số nguyên trong file
   nhãn; không có gì đảm bảo nó đúng sau một lần convert. Xem crop thật.
2. **Ghi nguồn và giấy phép của từng dataset** vào bảng dưới. Cuộc thi và bài
   báo đều cần, và tìm lại sau thì rất mất thời gian.
3. **Tập test phải có tag slice.** `data/processed/test/slices.csv`:
   `image_id,distance,light,occlusion`. Lý do: `docs/measurement-protocol.md` §4.
4. **Không để ảnh cùng một phiên quay nằm cả ở train và test.** Frame liền kề
   gần như trùng nhau; chia ngẫu nhiên theo ảnh sẽ làm mAP cao giả. Chia theo
   **phiên quay/video**, không theo frame.

## Nguồn dữ liệu

| Stage | Dataset | Nguồn | Giấy phép | #ảnh | Ghi chú |
|---|---|---|---|---|---|
| A person | | | | | |
| B PPE | | | | | |
| C fire/smoke | | | | | |
| D fall (validate) | | | | | |

## Video benchmark

| File | Thời lượng | Occupancy | Nội dung | Ghi chú |
|---|---|---|---|---|
| `bench_clip.mp4` | | | | dùng cho toàn bộ ma trận E1–E5 |

Toàn bộ ma trận thí nghiệm phải chạy trên **cùng một video**. Đổi video giữa
các cấu hình là đổi hai biến một lúc, và bảng kết quả mất ý nghĩa.

## Ground truth sự kiện

`data/processed/events.csv`: `video,frame_id,event_type` — thời điểm sự kiện
*bắt đầu xảy ra*, để đo detection latency (`docs/measurement-protocol.md` §6).

## Gán nhãn bán tự động cho ảnh lab

```bash
python3 scripts/data/extract_frames.py data/raw/lab_sessions/<phien>            # video -> ảnh (bỏ ảnh trùng)
python3 scripts/data/prelabel.py data/interim/lab_frames/<phien> --out data/interim/prelabel/<phien> \
        --lab-coat --copy-images                                           # nhãn gợi ý, ngưỡng thấp
```
Nhập `images/` + `labels/` + `obj.names` vào CVAT hoặc Label Studio (định dạng YOLO), mở `review.csv`
để sửa ảnh kém tin cậy trước. Lớp: `configs/lab_classes.yaml` (0 person, 1 lab_coat; goggles và gloves đã bỏ, #33). Chia train/val/test
THEO PHIÊN QUAY, không chia theo khung. Chi tiết và số đo: `docs/KNOWN_ISSUES.md` #32.
