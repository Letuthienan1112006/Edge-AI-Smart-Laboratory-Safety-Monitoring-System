# Thể lệ cuộc thi EDGE AI DESIGN CHALLENGE 2026 và đối chiếu với dự án

Nguồn: Kế hoạch số 38/KH-ĐĐT ngày 16/9/2026, Khoa Điện - Điện tử, ĐH Công nghệ Kỹ thuật TP.HCM (người dùng đưa PDF ngày 24/10/2026).

## Lịch

Kế hoạch số 38 ghi vòng 2 ngày 30/10 và chung kết 15/11; thông báo mới hơn của ban tổ chức (người dùng chụp màn hình 24/10) ghi:

- **Vòng sơ loại 05/11/2026:** trình bày dựa trên sản phẩm gần hoàn thiện hoặc đã hoàn thiện 100%. Mỗi nhóm chuẩn bị **poster khổ A2** (tóm tắt ý tưởng, đóng góp chính, thiết kế, chức năng, kết quả đánh giá) và **sản phẩm phần cứng demo**.
- **Vòng chung kết 14/11/2026:** hoàn thiện đề tài và **báo cáo**. (Phần sau của thông báo bị cắt; poster/demo chung kết chưa rõ.)

Địa điểm theo kế hoạch số 38: phòng chuyên đề **D102**, khu D, khoa Điện-Điện tử. Cần xác nhận phòng thi sơ loại 05/11 có phải D102 không.

Ngôn ngữ poster: tiếng Việt (người dùng xác nhận).

## Yêu cầu kỹ thuật (bắt buộc)

1. Suy luận trực tiếp trên thiết bị biên, không suy luận hoàn toàn trên cloud.
2. Mô tả kiến trúc model, quá trình huấn luyện/tối ưu, phương pháp triển khai lên phần cứng.
3. Đánh giá định lượng: **độ chính xác, độ trễ, FPS, tài nguyên (RAM, năng lượng tiêu thụ)**.
4. Giao diện hiển thị/giám sát (màn hình, web hoặc app); được phép gửi cảnh báo lên cloud.

Khuyến khích: quantization, pruning, knowledge distillation, kèm đánh giá độ trễ, độ chính xác, năng lượng.

## Tiêu chí chấm (mỗi tiêu chí 25%)

1. Hiệu năng model trên thiết bị biên (độ chính xác, độ trễ, tài nguyên).
2. Chất lượng thiết kế hệ thống phần cứng - phần mềm và giao diện giám sát/điều khiển.
3. Mức độ tối ưu hóa/tăng tốc triển khai (quantization, pruning, FPGA...).
4. Báo cáo và trình bày.

## Đối chiếu (24/10/2026)

| Yêu cầu / tiêu chí | Hiện có | Còn thiếu |
|---|---|---|
| Suy luận tại biên | NPU RK3588S, INT8, không cần mạng | - |
| Mô tả kiến trúc, huấn luyện, triển khai | `report/bao_cao.pdf` mục 2-4 | - |
| Độ chính xác | mAP50/50-95 hai tập kiểm tra tách theo phiên, luật vùng theo từng người, 3 seed | số đo theo sự kiện tại lab (video lab chưa chấm); độ nhạy góc camera (#52) |
| Độ trễ, FPS | p50 25,9 ms (23,4 khóa xung), 29,02 FPS, dashboard 25,9 FPS | - |
| RAM | 24 giờ, RSS 205-218 MB | - |
| **Năng lượng** | KHÔNG CÓ số watt hợp lệ (2 lần đo hỏng, #41) | **đo bằng `make energy` (đồng hồ KWS-1902C, ~50 phút)**: yêu cầu bắt buộc VÀ nằm trong tiêu chí 1, 3 |
| Giao diện giám sát | web 4 trạng thái, trình chiếu, đèn LED + OLED, cảnh báo Zalo (cloud) | - |
| Tối ưu hóa (tiêu chí 3) | INT8 đầu ra tách (mất tối đa ~1 điểm thay vì 17-20), cổng parity, pipeline thích nghi, khóa xung DDR | pruning, knowledge distillation chưa làm (chỉ là ý tưởng #5 trong `docs/algorithms-and-ideas.md`) |
| Báo cáo | `report/bao_cao.pdf` 17 trang | chốt định dạng/độ dài theo yêu cầu nộp |
| **Poster A2** (sơ loại 05/11) | `poster/poster.pdf` (bản đầu, 1 trang A2, tiếng Việt, `make poster-pdf`) | thêm dòng năng lượng sau khi đo; in thử một bản |
| Slide, demo, phản biện | kịch bản `docs/competition/presentation-script.md`, `docs/competition/judge-questions.md`; chưa có slide | slide, video demo gọn |

## Cập nhật 03/11/2026: hồ sơ nộp sơ loại (hạn hết thứ Bảy 07/11/2026)

Mỗi nhóm nộp **2 tệp** tóm tắt đề tài và sản phẩm:
1. **Poster khổ A2** (`poster/poster_A2_in.pdf`, nguồn `poster/poster.tex`, `make poster-pdf`).
2. **Bài báo ngắn PDF 2 cột, 6--8 trang**, trình bày đầy đủ nội dung đề tài theo chuẩn một bài báo, ngắn gọn, tập trung vào đóng góp chính, sản phẩm, đánh giá (`report/bai_bao_ngan.pdf`, nguồn `report/bai_bao_ngan.tex`, `make paper-pdf`).

**Cả hai tệp phải ghi rõ họ tên và MSSV của từng thành viên.** MSSV của bài báo điền ở các macro `\mssvAn ... \mssvThanh` đầu `report/bai_bao_ngan.tex`; poster ghi ở dòng nhóm trưởng/thành viên trong tiêu đề.
