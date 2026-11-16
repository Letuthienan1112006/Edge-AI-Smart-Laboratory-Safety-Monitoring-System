# Thu dữ liệu lab → gán nhãn → train (hướng dẫn chạy)

Mọi bước dưới đây đã được kiểm thử bằng video giả (camera laptop lúc kiểm thử chỉ trả ảnh đen); phần quay
bằng camera thật trên board **chưa chạy thử** vì board offline lúc viết.

## 0. Chuẩn bị (một lần)
- Cắm board, nối cáp LAN, `ssh orangepi-lan` vào được.
- Chép code lên board: `rsync -a scripts configs src orangepi-lan:AI_Challenge/` (board không dùng git).
- Đặt camera cố định ở **đúng vị trí demo**. Kiểm tra khung hình: `make peek` (mở ảnh vừa chụp).
- Đặt lại giờ board nếu lệch (không bắt buộc; tên phiên không phụ thuộc giờ).

### Kiểm tra góc camera (làm TRƯỚC khi quay hay để chụp lâu)
```bash
make camera        # laptop: mở trình duyệt xem camera board trực tiếp, Ctrl+C để tắt và nhả camera
```
Trang hiện hình có lưới chia 3 phần, khung quanh người nhận diện được (kèm % chiều cao người trên khung) và
cảnh báo: **ngược sáng** (% điểm ảnh cháy sáng), **tối**, **người quá nhỏ** (<25% khung: kính/găng chỉ còn vài
điểm ảnh) hoặc **quá sát / bị cắt mép** (>95%). Nút "Lưu ảnh này" lưu ảnh để so sánh các vị trí. Ngưỡng là quy
tắc kinh nghiệm để so sánh vị trí đặt, không phải giới hạn đã đo. Luồng hình chỉ đi qua đường hầm SSH (không mở
ra mạng). Lệnh này tự dừng dịch vụ tự chụp để lấy camera.

### Bắt đầu chụp tự nhiên (sau khi góc camera ổn)
```bash
make start-capture     # chụp 1 ảnh/4 s, tối đa 12 giờ; chạy như dịch vụ trên board -> RÚT CÁP được, KHÔNG rút nguồn
make capture-status    # (khi cắm lại cáp) dịch vụ có chạy không, mỗi phiên bao nhiêu ảnh, ảnh cuối cách bao lâu
make stop-capture      # dừng sớm
make fetch-sessions    # kéo ảnh về laptop để xử lý
```

## 1. Quay (chạy TRÊN BOARD, có hướng dẫn từng cảnh)
```bash
ssh -t orangepi-lan 'cd ~/AI_Challenge && python3 scripts/collect_guided.py'
```
16 cảnh, ~12 phút quay (kịch bản: `configs/collection_plan.yaml`). Mỗi cảnh in việc cần làm, chờ Enter,
đếm ngược 5 giây rồi quay. Ctrl+C dừng; chạy lại sẽ **tiếp tục** từ cảnh chưa quay (cảnh hỏng được quay
lại). Xem tiến độ: `python3 scripts/collect_guided.py --list`. Chỉ quay vài cảnh: `--only 06,08`.
Cảnh 15, 16 cần **người khác** (không phải bạn) để làm tập test.

## 2. Kéo về, cắt ảnh, gợi ý nhãn (chạy trên LAPTOP)
```bash
make fetch-sessions        # HOST=orangepi FPS=1 COAT=0 make fetch-sessions để đổi tuỳ chọn
```
Mỗi phiên → `data/interim/lab_frames/<phiên>/` (ảnh) và `data/interim/prelabel/<phiên>/` (ảnh + nhãn gợi ý +
`review.csv` ưu tiên ảnh kém tin cậy). Chạy lại an toàn (phiên đã làm sẽ bỏ qua).

## 3. Sửa nhãn (thủ công, ~2-4 giờ cho ~1000 ảnh)
Nhập `images/` + `labels/` + `obj.names` của từng phiên vào CVAT hoặc Label Studio (định dạng YOLO), sửa
khung gợi ý (xóa khung sai, vẽ khung thiếu), xuất lại YOLO vào `data/labeled/<phiên>/{images,labels}`.
Lớp: `configs/lab_classes.yaml` = 0 person, 1 lab_coat (đã bỏ `goggles` và `gloves` ngày 03/10, xem KNOWN_ISSUES #33). Ảnh không có gì: giữ file `.txt` RỖNG (thiếu file = lỗi).
Không cần sửa hết 1500 ảnh: ưu tiên mỗi phiên khoảng 40-80 ảnh đại diện, nhưng phiên test (15, 16) sửa kỹ.
Ghi lại thời gian sửa một lô để đo xem gợi ý tiết kiệm bao nhiêu (đo lại #32).

### 3b. Nếu gán nhãn bằng Roboflow
Xuất: **Versions → Generate → (không chọn augmentation) → Export Dataset → YOLOv8** (khung chữ nhật, không phải polygon/segmentation), tải .zip. Roboflow đổi thứ tự lớp và tên file và tự chia train/valid/test ngẫu nhiên: **đừng dùng trực tiếp**, chạy:
```bash
make import-roboflow ZIP=~/Downloads/<ten>.zip        # thêm MERGE=1 để gộp các mảnh áo blouse thành một khung
```
Lệnh này đổi lớp theo TÊN về thứ tự của dự án, khôi phục tên file gốc và tên phiên, bỏ chia tập của Roboflow, chuyển polygon thành khung, loại bản augment trùng, rồi ghi vào `data/labeled/<phiên>/`. Nó dừng nếu gặp tên lớp lạ.

### 3c. Lô gán nhãn tiếp theo (đã có model): `scripts/make_batch.py`
```bash
.venv/bin/python scripts/make_batch.py --session boot006 --n 250 \
    --weights /mnt/big/AGV_ANLEE/weights/lab_v1_pilot_best.pt --out data/interim/roboflow_upload/boot006_b2
```
Chọn các ảnh KHÁC NHAU NHẤT trong số chưa gán (không dựa vào đầu ra model; lô đầu ưu tiên "ảnh có găng" nên lệch,
#33), cho model gợi ý nhãn ngưỡng thấp, xuất `<out>.zip` để tải lên Roboflow. Model phải được train đúng các lớp của
`configs/lab_classes.yaml` (tool từ chối nếu khác). Model học từ một phiên hay báo dư áo trắng là áo blouse (#34).

## 4. Ghép bộ dữ liệu, chia THEO PHIÊN
```bash
make build-lab-dataset     # -> data/processed/lab_v1, từ chối nếu thiếu nhãn / sai mã lớp / test rỗng
# CHỈ có một phiên (thử nghiệm sơ bộ)? Khối cuối của phiên làm test (số đo LẠC QUAN, không báo cáo như khả năng tổng quát):
.venv/bin/python scripts/build_lab_dataset.py --pilot-time-holdout 0.2
.venv/bin/python scripts/verify_dataset.py data/processed/lab_v1/data.yaml --crops 8   # XEM crop bằng mắt
```
## 5. Train rồi đo trên tập test của lab
```bash
.venv/bin/python scripts/train_baseline.py --data data/processed/lab_v1/data.yaml \
    --model /mnt/big/AGV_ANLEE/weights/yolo26n.pt --imgsz 640 --epochs 80 --name lab_v1
```
Báo cáo **AP theo từng khoảng cách (cảnh 02/07/10 so với 06/08/11)** và theo từng lớp; so với model gốc
trên cùng tập test. Xuất RKNN INT8 chỉ sau khi qua cổng parity (`scripts/export_rknn.py`, rồi
`scripts/validate_conversion.py` trên board).
