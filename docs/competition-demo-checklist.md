# Checklist demo cuộc thi

Cập nhật 21/10/2026 (thứ tự làm + lệnh cho buổi lab: `docs/lab-day-runbook.md`). Checklist này tách rõ việc làm được ở nhà và việc bắt
buộc phải làm tại phòng lab. Không dùng video ở nhà để kết luận độ chính xác
trong lab.

## Ở nhà: chỉ kiểm phần mềm và vận hành

- [x] Board khởi động được, `labsafe-live` và `labsafe-ramlog` active.
- [x] Đồng hồ board được đồng bộ trước khi ghi log.
- [x] Code, config và hai model RKNN trên board khớp SHA-256 laptop.
- [x] Dashboard, `/status` và MJPEG hoạt động; tốc độ quan sát khoảng 29--30
  FPS, frame không cũ.
- [ ] Kiểm trang trên màn hình/máy chiếu sẽ dùng khi thi.
- [ ] Chuẩn bị một video demo dự phòng được quay từ cấu hình cuối tại lab.

Khung hình ở nhà hiện nhìn vào bàn và không thấy trọn người/sàn. Mọi cảnh báo
áo, vùng hoặc té ngã sinh ra trong bố trí này chỉ dùng để tìm bug phần mềm,
không đưa vào báo cáo độ chính xác.

## Tại lab: đặt camera trước khi chạy test

- [ ] Cố định camera ở đúng vị trí dự kiến khi thi.
- [ ] Thấy trọn đầu, thân và chân của người ở toàn bộ lối đi cần giám sát.
- [ ] Vùng sàn không bị bàn/tủ che phần lớn và người không bị cắt ở mép dưới.
- [ ] Kiểm ảnh khi phòng sáng bình thường; ghi lại vị trí camera để có thể đặt
  lại giống nhau.
- [ ] Vẽ lại polygon trên dashboard, lưu, tải lại trang và xác nhận polygon
  vẫn đúng. Thử `Khôi phục vùng trước` và `Esc` hủy chỉnh sửa.

Nếu không đạt các mục trên thì dừng: sửa camera/vùng trước, không chỉnh ngưỡng
model để che lỗi bố trí.

## Ma trận kiểm thử end-to-end

Mỗi tình huống chạy ít nhất 3 lần. Ghi `PASS/FAIL`, thời gian bắt đầu cảnh và
thời gian banner đổi trạng thái.

| ID | Tình huống | Kết quả bắt buộc |
|---|---|---|
| D1 | Phòng trống, camera có hình | `AN TOÀN`, số người 0, FPS có số |
| D2 | Người không áo đứng ngoài vùng | `CHÚ Ý`, không phát cảnh báo vùng |
| D3 | Người không áo bước vào vùng và đứng yên ít nhất 8 s | `BÁO ĐỘNG`; có ảnh người và ảnh toàn khung |
| D4 | Người mặc áo bước vào cùng đường đi | Không phát cảnh báo thiếu áo |
| D5 | Người ra khỏi vùng sau cảnh báo | Banner không tiếp tục mô tả vi phạm đang diễn ra |
| D6 | Hai người, một người có áo và một người không áo | Không gán nhầm áo của người này cho người kia |
| D7 | Người chỉ lộ đầu/vai ở mép dưới | Không tính là đứng trong vùng (`partial_min_h=0.3`) |
| D8 | Rút camera | Trong vài giây phải hiện `MẤT TÍN HIỆU`, số liệu thành `–`; không bao giờ hiện `AN TOÀN` |
| D9 | Cắm lại camera | Hình và trạng thái tự phục hồi, service không cần restart thủ công |
| D10 | Ngắt SSH tunnel/HTTP rồi nối lại | Trang giãn nhịp request và tự cập nhật lại sau khi kết nối |

Không dùng D3 đạt một lần để gọi là accuracy 100%. Đây là smoke test sản phẩm,
khác với tập đánh giá có ground truth.

## Bằng chứng cần mang về

- [ ] Một video quay cả người thực hiện và màn hình dashboard cho D1--D5.
- [ ] Ảnh chụp màn hình `AN TOÀN`, `CHÚ Ý`, `BÁO ĐỘNG`, `MẤT TÍN HIỆU`.
- [ ] Một bundle log/run của cấu hình cuối; ghi commit, model SHA-256 và config.
- [ ] Ảnh camera/vị trí lắp đặt và file polygon cuối.
- [ ] Bảng PASS/FAIL cho D1--D10, không xóa dòng FAIL.
- [ ] Video dự phòng ngắn, không phụ thuộc Internet.

## Điều kiện chốt demo

Chỉ gọi bản demo sẵn sàng khi D1--D5, D8 và D9 đều đạt 3/3 lần; dashboard đọc
được từ vị trí giám khảo; video dự phòng mở được trên laptop; và mọi số sử dụng
trong thuyết trình khớp `docs/competition-claims.md`.
