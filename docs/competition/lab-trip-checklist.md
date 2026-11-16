# Danh sách việc khi lên lab (đầu tuần 26/10/2026)

Mục tiêu: lấy **số năng lượng** (yêu cầu bắt buộc của thể lệ), video demo gọn và kiểm camera trước vòng sơ loại **05/11** (poster A2 + sản phẩm phần cứng demo).

## 0. Trước khi đi

- Mang: board + nguồn, camera C270 + kẹp, cáp LAN, đồng hồ USB-C KWS-1902C, laptop + sạc, 2 áo lab, 1 áo thun trắng, điện thoại.
- Laptop có Wi-Fi/hotspot để board ra mạng (Zalo).

## 1. Khởi động (10 phút)

1. Cắm camera trước, rồi điện board, rồi cáp LAN; chờ khoảng 1 phút.
2. `make lab-preflight OPEN=1` → phải ra READY (tự sửa đồng hồ board).
3. `make board-internet MODE=on` (sudo) → để tin Zalo gửi được.
4. Nếu đổi chỗ đặt camera: vẽ lại vùng trên trang web (mục 2 của `docs/competition/lab-day-runbook.md`).

## 2. Đo năng lượng (khoảng 1 giờ 15 phút) — ƯU TIÊN SỐ 1

Đồng hồ trước đó hiển thị 0,4-0,67 mà người dùng nghi là sai. Kiểm trước:

1. Chụp/đọc ba số **V, A, W** lúc board chạy bình thường (đồng hồ nối giữa nguồn và board, KHÔNG đi đường khác).
2. Nhờ Claude chạy tải CPU 8 nhân 60 giây trên board: dòng điện phải **tăng rõ**. Không tăng = đồng hồ nối sai chỗ.
3. Bộ đếm mAh/Wh phải tăng đều (3 W ≈ 50 mWh/phút).
4. Đạt cả 3 thì `make energy` (≈50 phút, đo nghỉ / thích nghi / chạy hết; gõ số đọc khi script hỏi, đừng gõ trễ).
5. Gửi số cho Claude để đưa vào poster (`poster/poster.tex`, tìm ENERGY) và báo cáo.

## 3. Nhận áo lab (10 phút)

`python3 scripts/eval/coat_check.py --labels "khong ao" "ao 1" "ao 2" "ao 1 mo khuy" --seconds 20` (đứng một mình giữa khung). Mức đạt: áo >= 90%, không áo <= 5%. Không đạt → dừng, chụp màn hình, nhắn Claude (đừng hạ ngưỡng).

## 4. Video demo gọn (20 phút)

`make record-demo NAME=lan2`, bấm vào cửa sổ trình duyệt (chế độ P, **bật âm báo**: bấm vào trang hoặc phím M). Kịch bản 5 bước ở `docs/competition/presentation-script.md` mục 4; mỗi lần MỘT người vào vùng, không đứng sát người khác. Quay thêm bằng điện thoại cảnh người vào + màn hình + đèn/OLED.

## 5. Ghi lại cho vòng sơ loại

- Chụp ảnh vị trí đặt camera.
- Hỏi ban tổ chức: phòng thi sơ loại 05/11 là phòng nào (D102?), có vào thử trước được không, bàn/điện/mạng có sẵn không.
- Chụp ảnh sản phẩm phần cứng (board + LED + OLED) rõ nét hơn ảnh trong poster.

## 6. Về nhà

`make fetch-violations`; chép video/ảnh vào repo hoặc cho Claude đường dẫn.
