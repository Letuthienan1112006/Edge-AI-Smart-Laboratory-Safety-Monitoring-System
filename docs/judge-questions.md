# Câu hỏi phản biện dự kiến

Cập nhật 23/10/2026. Câu trả lời dưới đây cố ý ngắn và chỉ dùng bằng chứng đã
có. Khi chưa đo, trả lời “chưa đo” rồi nêu kế hoạch; không lấp bằng suy đoán.

## “Điểm mới là gì? Chạy YOLO trên Orange Pi đã có nhiều người làm.”

Điểm chính không phải việc chạy YOLO. Đóng góp là quy trình deploy đáng tin:
cổng parity phát hiện lỗi chuyển đổi im lặng, sửa lượng tử hóa bằng đầu ra tách,
pipeline thích nghi giảm tải các tầng phụ, và đánh giá từ accuracy đến vận hành
24 giờ trên thiết bị thật.

## “Tại sao gọi là đa tác vụ khi kính/găng đang tắt?”

Kiến trúc hỗ trợ nhiều tầng và dùng Stage B/C để nghiên cứu cách lập lịch trên
ba lõi NPU. Sản phẩm hiện được xác thực đầy đủ nhất cho người, áo lab và vùng
nguy hiểm. Kính/găng không có dữ liệu lab nên đã chủ động loại khỏi cấu hình
live; fall và fire/smoke là module tích hợp nhưng chưa đủ đánh giá thực địa.

## “Hệ thống có đạt 30 FPS không?”

Stage A INT8 đạt khoảng 29--30 FPS. Toàn pipeline trên video 30 FPS đạt 29,02
FPS khi khóa DDR. Khi phục vụ dashboard thật, kết quả đã đo là 25,9 FPS. Không
dùng ba con số này thay thế lẫn nhau.

## “Vì sao INT8 mặc định bị giảm accuracy, và sửa thế nào?”

Đầu ra mặc định gộp tọa độ khung và điểm lớp vào một tensor dùng chung thang
8-bit, làm mép khung bị lượng tử hóa thô. Tách box và class của từng scale thành
sáu tensor giúp mỗi loại có thang riêng; mAP50-95 chỉ còn mất tối đa khoảng một
điểm thay vì 17--20 điểm.

## “Accuracy có đủ cho một hệ thống an toàn không?”

Chưa đủ để thay thế con người. Ở hai góc đã thử, luật vùng bắt khoảng 75--90%
người không áo tùy seed/góc và vẫn có bỏ sót. Vì vậy đây là hệ thống hỗ trợ
giám sát: mỗi cảnh báo có ảnh bằng chứng để người phụ trách xác nhận.

## “7/7 sự kiện có phải accuracy 100% không?”

Không. Đó chỉ là 7 sự kiện, một người, 8 phút và quay ở nhà. Nó chứng minh luồng
end-to-end chạy được và giúp tìm bug hình học; không đủ để ước lượng accuracy
hay số báo giả mỗi giờ trong lab.

## “Tại sao chọn YOLO26n thay vì model lớn hơn?”

Bốn kiến trúc nano có mAP50 gần như ngang nhau trên cùng dữ liệu. YOLO26n có
mAP50-95 nhỉnh hơn và triển khai RKNN đã được kiểm chứng. YOLO26s chỉ hơn khoảng
hai điểm ở luật vùng nhưng chậm hơn 2--3 lần; nút thắt chính là dữ liệu, đặc biệt
người xa, không phải kích thước model.

## “Các bạn tránh data leakage thế nào?”

Train và test được tách theo phiên quay, không chia ngẫu nhiên các frame gần
nhau. Bộ fire/smoke công khai từng có ảnh gốc trùng giữa các split nên đã được
nhóm lại theo ảnh gốc trước khi chấm. Kết quả test cùng phiên không được dùng
làm số chính.

## “Ngưỡng 0,6 có được chọn trên test không?”

Có khảo sát ngưỡng trên các tập test hiện tại, nên số luật vùng hơi lạc quan.
Đây là hạn chế đã ghi trong báo cáo. Vòng đánh giá tiếp theo phải có validation
set riêng để chọn ngưỡng và một test set khóa kín để báo cáo cuối.

## “Có thật sự tiết kiệm năng lượng không?”

Có đo bằng đồng hồ USB-C đặt giữa bộ sạc và bo mạch (29/10, số mWh cộng dồn, sai số
tuyệt đối khoảng ±10%): bo mạch nghỉ 0,86 W, cấu hình sản phẩm 2,20 W, chế độ thích
nghi 1,98 W, chạy hết mọi tầng 3,34 W. Cùng tốc độ khung 29 FPS, thích nghi dùng ít
hơn chạy hết 1,36 W, năng lượng mỗi khung ≈ 69 so với 113 mJ. Giới hạn: video phát
lại (không có camera thật), mỗi pha đo một lần. Sản phẩm 2,20 W và thích nghi 1,98 W
chênh nhau trong sai số đồng hồ, nên không khẳng định cái nào tốn hơn.

## “Nếu camera bị rút hoặc đứng hình thì sao?”

Sau khoảng 3 giây không có frame mới, hệ thống ngừng phân tích frame cũ, thử mở
lại camera và dashboard chuyển sang `MẤT TÍN HIỆU`; nó không được phép hiện
`AN TOÀN` khi không có hình. Logic đã có test thiết bị giả; rút/cắm camera thật
tại vị trí demo là bước bắt buộc trước cuộc thi.

## “Nếu mất mạng thì hệ thống có chạy không?”

Suy luận, luật cảnh báo và lưu ảnh đều chạy trên board, không cần Internet.
Laptop chỉ dùng SSH tunnel để xem dashboard. Mất tunnel làm mất màn hình giám
sát từ xa nhưng không dừng pipeline trên board.

## “Hệ thống chạy lâu có rò RAM không?”

Phép đo 24 giờ cho thấy RSS đi ngang sau giờ khởi động: RSS cuối 218 MB, mức cao
nhất 269 MB, không có lỗi. Đây là một lần chạy với một cấu hình, chưa phải bằng
chứng cho nhiều ngày hoặc mọi cấu hình.

## “Bước tiếp theo có giá trị nhất là gì?”

Mở rộng test theo sự kiện trong lab với nhiều người và validation/test tách
riêng; tăng dữ liệu người xa, thiếu sáng/ngược sáng; đo W/Wh; sau đó mới chấm
fall và fire/smoke thực địa. Không ưu tiên tăng kích thước model khi dữ liệu vẫn
là nút thắt.

## "Nếu đổi chỗ đặt camera thì hệ thống còn đúng không?"

Chưa chắc, và chúng tôi đã gặp lỗi này thật. Ở một góc rất khác dữ liệu huấn luyện (camera rất gần, nhìn từ trên xuống, cắt mất đầu người), model đang dùng
không nhận được áo blouse ở khung nào trong 122 khung đo; hạ ngưỡng không cứu được. Đặt camera về khung hình giống dữ liệu huấn luyện thì áo được nhận ở 98% số lần
thăm dò trên bốn người mặc áo. Đó là phép đo một phiên 25 giây, chưa phải độ chính xác. Cách xử lý: cố định camera, kiểm bằng một người mặc áo đứng một mình
trước khi dùng, và nếu bắt buộc đổi góc thì thu dữ liệu tại vị trí đó rồi huấn luyện lại. Chi tiết: KNOWN_ISSUES #52, báo cáo mục 5.2.

## "Vì sao hệ thống không báo vi phạm cho người chỉ lộ mặt hoặc bị che?"

"Không thấy áo" khác "không mặc áo". Người chỉ lộ mặt, bị che sau người đứng gần hơn hoặc quá nhỏ được xếp là "chưa thấy rõ áo" (khung xám) và không bị tính vi phạm.
Trên một video lab đông người dài 170 giây, số cảnh báo giảm từ 31 xuống 6 và cả 6 đều đúng. Cái giá là người bị che không bị phán cho tới khi lộ thân.

## "Tin nhắn Zalo có cần Internet không? Sao không gửi ảnh?"

Chỉ tin Zalo cần Internet; nhận diện, luật cảnh báo, đèn LED, màn OLED và ảnh bằng chứng đều chạy không cần mạng. Việc gửi chạy ở luồng riêng, mất mạng thì bỏ tin đó
và không làm chậm nhận diện. Chúng tôi chủ động chỉ gửi chữ: Zalo chỉ nhận ảnh qua đường dẫn công khai nên phải đưa ảnh người lên máy chủ của bên thứ ba. Ảnh bằng chứng
nằm trên bo mạch và xem qua trang web. Độ trễ từ lúc báo đỏ đến lúc điện thoại nhận tin chưa đo.

## "Hai người đứng sát nhau thì có nhầm áo không?"

Có thể, và chúng tôi đã đo: người đứng sát người mặc áo có thể "mượn" áo trong tối đa 2 giây. Ở lab, hai người không áo đứng sát được tính "đang mặc áo" 48% và 32%
số lần thăm dò trong 25 giây. Đã giảm bằng cách xóa trí nhớ áo khi bộ bám ghép theo khoảng cách tâm; chưa loại hẳn. Vì vậy khi demo, từng người vào vùng riêng.
