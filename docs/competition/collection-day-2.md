# Ngày thu dữ liệu thứ 2 (batch L2)

Dữ liệu `boot014` đã **bão hòa**: thêm 250 ảnh khó cùng phiên không cải thiện gì đo được (KNOWN_ISSUES #35, 3 seed).
Ngày này chỉ thu thứ model **chưa có**. Ba mục tiêu, theo thứ tự quan trọng:

1. **Âm tính áo trắng:** người mặc áo thun / sơ mi / đồng phục trắng, KHÔNG phải blouse. Model hiện gọi 19-20/20 ảnh loại này là áo lab.
2. **Người mới, ngày khác:** tập TEST thật (phiên `L2_13`…`L2_16`), không bao giờ dùng để train.
3. **Video 30 FPS liên tục:** để đo luật vùng cấm (cần bám một người ≥ 3 s). Ảnh chụp 4 s/khung (như `boot014`) không đo được chuyện đó.

## Cần chuẩn bị (hôm trước)
- **8-10 người**, trong đó **ít nhất 4 người CHƯA TỪNG** xuất hiện trong ảnh `boot006`/`boot014` (nhóm test). Mỗi người ký giấy đồng ý được quay.
- Mỗi người mang **cả hai bộ**: blouse và 1-2 áo trắng/sáng không phải blouse (thun trắng trơn, sơ mi trắng, đồng phục).
- Giữ camera **đúng góc cũ** (vùng `configs/zones_lab.yaml` vẽ theo góc đó). Nếu phải dời: ghi lại và vẽ lại vùng.
- Board cắm điện, cáp mạng nối laptop. Ổ đĩa còn dư (hiện ~50 GB).

## Trình tự trong ngày
1. `make camera` — kiểm tra góc, đủ sáng, thấy cả sàn trắng. Ctrl+C khi xong.
2. `make collect-day2` — quay lần lượt 16 cảnh (~17 phút video, nhưng cần ~45-60 phút vì chờ người đổi áo). Mỗi cảnh in việc cần làm, nhấn Enter khi sẵn sàng. Làm được tới đâu dừng ở đó (Ctrl+C), chạy lại sẽ tiếp tục. **Làm các cảnh 01-12 trước, test 13-16 sau cùng với người mới.**
3. Cuối `collect-day2` hỏi có chạy chụp cả ngày không → `y`. Sau đó **rút cáp mạng, KHÔNG rút nguồn**; board tự chụp 1 ảnh/4 s trong 12 giờ (cho ảnh tự nhiên, nhiều người).
4. Chiều tối nối lại cáp: `make capture-status`, `make fetch-sessions`.

## Phải GHI LẠI bằng tay (đồng hồ board sai)
- Giờ điện thoại lúc bắt đầu mỗi lượt của cảnh `L2_16` (ai bước vào / ra, mặc gì). Dùng để đối chiếu cảnh báo.
- Giờ điện thoại lúc rút cáp mạng.
- Ai (tên/mã) xuất hiện ở cảnh test nào, và có từng xuất hiện trước đây không.

## Sau khi kéo về (tôi làm)
1. Trích khung từ video (~2 khung/giây), chọn lô khác nhau nhất, nhãn gợi ý bằng `lab_v3` → zip cho Roboflow.
2. **Test trước:** phiên `L2_13`…`L2_16` gán nhãn đầy đủ trước, rồi mới train. Tạo tập test thứ hai độc lập với `boot014test`.
3. Chạy `lab_v3` trên các video vùng (`L2_11`, `L2_12`, `L2_16`) bằng đúng hệ thống live (bám người, debounce 3 s) để đo: bắt được bao nhiêu % lượt người không áo bước vào, báo nhầm bao nhiêu lượt người có áo.
4. Train lại với ≥3 seed (chênh dưới ~4 điểm không kết luận).

## Quy tắc gán nhãn (như cũ, nhắc lại)
- Chỉ vẽ `lab_coat` khi **chắc chắn là blouse đang mặc**. Áo trắng khác / áo cầm trên tay / áo treo → KHÔNG vẽ, vẫn giữ khung `person`.
- Áo mở: 2 mảnh, kéo tới cổ tay. Xuất YOLO26, không Resize, không Augmentation.

## Kết quả ngày thu 13/10 (ghi lại)
- Board chạy liên tục 5 giờ 21 phút (không mất điện); chụp tự động `boot018`: **4.011 ảnh** (10:41-15:48, 573 khung tĩnh bị bỏ).
- 99% khung có người, 92% có từ 3 người trở lên, đông nhất 13 người. Khoảng 11:41-12:41 số áo lab thấp hơn số người
  (TB 1,4-2,0 áo trên 2,7-3,9 người) -> có nhiều người không mặc áo lab: dữ liệu âm tính.
- **Góc camera KHÁC `boot014`** (nhìn rộng sang phía bàn kính hiển vi, sàn trắng lớn): `configs/zones_lab.yaml` không khớp;
  vẽ vùng mới bằng nút "Vẽ lại vùng" trên trang web. Phiên này đồng thời là phép thử đổi góc.
- Chỉ quay được 1 cảnh có hướng dẫn (`L2_01` phòng trống 20 s, 602 khung): chưa có video 30 FPS cho luật vùng.
- Lô gán nhãn: `boot018_TEST` (150 ảnh, đoạn 1800-2900, phiên riêng `boot018test`) + `boot018_b1` (250 ảnh, 120 ảnh khó),
  mỗi lô có bản `_person` (chỉ khung người).
