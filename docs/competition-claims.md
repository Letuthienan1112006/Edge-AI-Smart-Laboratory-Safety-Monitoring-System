# Số liệu và phạm vi công bố khi thi

Cập nhật 23/10/2026. File này là nguồn ngắn gọn để làm slide, poster và trả
lời giám khảo. Nếu một con số không có ở đây, phải kiểm lại `docs/results.md`
và log gốc trước khi dùng.

## Thông điệp chính

Đây là hệ thống **hỗ trợ giám sát** chạy offline trên Orange Pi 5. Tác vụ được
kiểm chứng đầy đủ nhất là phát hiện người không mặc áo lab trong vùng nguy
hiểm, tạo cảnh báo và lưu ảnh bằng chứng để người phụ trách xác nhận. Không mô
tả hệ thống như một cơ chế bảo đảm an toàn hay tự động thay thế con người.

Đóng góp kỹ thuật chính không phải chỉ là chạy YOLO trên NPU:

1. Cổng parity bắt lỗi chuyển đổi model im lặng trước khi deploy.
2. Tách tensor đầu ra khi lượng tử hóa để giảm mức mất mAP50-95 từ khoảng
   17--20 điểm xuống tối đa khoảng 1 điểm.
3. Pipeline thích nghi giảm số phép tính NPU và nhiệt so với chạy mọi tầng ở
   mọi khung.
4. Đánh giá theo phiên riêng, nhiều seed, chạy dài 24 giờ và lưu bằng chứng
   cho từng cảnh báo.

## Các số được phép dùng

| Nội dung | Số công bố | Điều kiện và giới hạn |
|---|---:|---|
| Thiết bị | Orange Pi 5, RK3588S, 8 GB; Logitech C270 640x480 | Toàn bộ suy luận chạy tại chỗ, không cần cloud |
| CPU ONNX FP32 | 10,5 FPS; 90,9 ms suy luận; 86,8 °C sau khoảng 2 phút | Stage A; nhiệt chưa bão hòa |
| NPU RKNN INT8 (model đang dùng, đầu ra tách 640) | lời gọi đầy đủ p50 25,9 ms (xung mặc định), 23,4 ms (khóa xung) | Stage A riêng, gồm tiền xử lý + NPU + giải mã; số 25,1 ms cũ là model COCO một tensor (29/09), không dùng nữa |
| Toàn pipeline, DDR khóa | 29,02 FPS; p99 41,4 ms | Video phát lại 30 FPS, không phải giao diện live |
| Dashboard bật | 25,9 FPS | Video phát lại 30 FPS (không phải camera), DDR khóa, một người xem, vẽ/nén video tối đa 15 lần/giây |
| INT8 đầu ra tách, góc 1 | mAP50 0,869 ± 0,008; mAP50-95 0,657 ± 0,004 | 3 seed, NPU thật |
| INT8 đầu ra tách, góc 2 | mAP50 0,973 ± 0,001; mAP50-95 0,784 ± 0,004 | 3 seed, NPU thật |
| Luật vùng, góc 1 | bắt 75--86% người không áo; báo nhầm 7--9% | 3 seed; còn bỏ sót đáng kể |
| Luật vùng, góc 2 | bắt 86--90%; báo nhầm 10--11% | Tập góc 2 ít người không áo |
| Event-level ban đầu | 7/7 sự kiện; trễ trung vị khoảng 5,24 s | Chỉ 7 sự kiện, 1 người, 8 phút, quay ở nhà; không dùng như tỷ lệ tổng quát |
| Adaptive so với parallel | công NPU 32 so với 74 ms/khung; nhiệt ổn định 41,5 so với 54,3 °C (thấp hơn 12,8 °C); đo dài 3 lượt × 15 phút | Số watt: xem hai dòng dưới |
| Công suất bo mạch (29/10), đồng hồ USB-C KWS-1902C, số mWh cộng dồn | nghỉ 0,86 W; sản phẩm 2,20 W; thích nghi 1,98 W; chạy hết 3,34 W (sai số tuyệt đối khoảng ±10%) | Video phát lại, không camera thật; 1 lần đo mỗi pha (5--16 phút); `docs/energy-2026-10-29/` |
| Thích nghi so với chạy hết, điện | thấp hơn 1,36 W (−41%); năng lượng/khung ≈ 69 so với 113 mJ (−39%) | Cùng tốc độ khung 29 FPS; chênh vượt sai số đồng hồ, nhưng chỉ 1 cặp; sản phẩm 2,20 W KHÔNG khác thích nghi 1,98 W ngoài sai số, không nói sản phẩm tốn hơn/ít hơn |
| Chạy dài | 24 giờ; RSS cuối 218 MB, lớn nhất 269 MB; 0 lỗi | Một cấu hình, một lần chạy; restart có kiểm soát mất khoảng 9 s |
| Quan sát tại lab, góc camera mới (23/10) | model đang dùng nhận áo 0/122 khung; lab_v5 20% ở ngưỡng 0,6 | 1 người mặc áo, 8 giây liên tiếp, FP32; chỉ để minh họa nguyên nhân (góc camera), không phải độ chính xác |
| Quan sát tại lab, camera về góc huấn luyện (23/10) | 4 người mặc áo: có khung áo ở 92/94 lần thăm dò (98%); 3 người không áo: 0/38 lần đạt 0,6 | Bo mạch thật, cấu hình sản phẩm, 25 giây, 6 người, thăm dò 0,4 s (không độc lập); chưa phải độ chính xác |
| Mượn áo giữa người đứng sát (23/10) | 2 người không áo bị tính "đang mặc áo" 48% và 32% số lần thăm dò | Cùng phiên 25 giây; hạn chế #51 |
| Test phần mềm | 345 passed (23/10/2026) | Số cũ 296 passed (21/10) đã lỗi thời; nay có thêm test cho đèn/OLED và tin Zalo |

## Không được tuyên bố

- Không nói hệ thống đã chứng minh đầy đủ năm tác vụ an toàn. Kính/găng không
  nằm trong cấu hình live; té ngã và lửa/khói chưa có đánh giá thực địa đủ.
- Không nói dashboard chạy 30 FPS. Con số đã đo với dashboard là 25,9 FPS;
  29--30 FPS thuộc benchmark khác.
- Không nói đã đo mức tiết kiệm điện. Chưa có phép đo W/Wh hợp lệ.
- Không dùng mAP hoặc AP trên tập cùng phiên làm kết quả chính.
- Không gọi 7/7 sự kiện là độ chính xác 100%; tập này quá nhỏ và chỉ có một
  người.
- Không nói hệ thống "chịu được mọi góc camera": ở góc rất khác dữ liệu huấn luyện model không nhận áo (KNOWN_ISSUES #52).
- Không dùng 98% hay 0/122 ở lab như độ chính xác; chúng là phép đo một phiên ngắn dùng để chỉ ra nguyên nhân.
- Không nói tin Zalo gửi ảnh (chỉ gửi chữ) và không nói "không cần mạng" cho phần Zalo: nhận diện, đèn và OLED không cần mạng, tin Zalo cần Internet.
- Không gọi đây là hệ thống safety-critical hay cơ chế thay thế người giám sát.

## Luồng demo nên dùng

1. **An toàn:** có hình camera và không có vi phạm; giải thích đây là trạng
   thái hiện tại, không phải lời bảo đảm tuyệt đối.
2. **Vi phạm:** người không áo lab bước vào vùng; chờ chống rung, chỉ ra banner,
   ảnh crop và ảnh toàn cảnh làm bằng chứng.
3. **Mất tín hiệu:** rút camera hoặc dùng kịch bản tương đương; giao diện phải
   hiện `MẤT TÍN HIỆU`, không được hiện `AN TOÀN`.

Đèn LED (xanh/vàng/đỏ) và màn OLED trên bo mạch hiển thị cùng bốn trạng thái với trang web; tin Zalo (chữ) là phần tùy chọn duy nhất cần Internet.

Phát hiện té ngã và lửa/khói ĐÃ TẮT trong mọi cấu hình demo/live từ 21/10 (quyết định của chủ dự án:
sản phẩm chỉ giám sát áo lab trong vùng nguy hiểm). Mã nguồn vẫn còn trong repo cho phần nghiên cứu
adaptive/parallel; nếu nhắc tới thì chỉ ở slide kiến trúc như hướng mở rộng.

## Nguồn kiểm tra

- `docs/results.md`: bảng số và giới hạn đầy đủ.
- `KNOWN_ISSUES.md`: nguyên nhân các lỗi chuyển đổi, dữ liệu và vận hành.
- `runs/split_eval/`, `runs/events/`, `runs/ramlog/`: log gốc tương ứng.
- `configs/live/dashboard_lab_board.yaml`: cấu hình live hiện tại.
