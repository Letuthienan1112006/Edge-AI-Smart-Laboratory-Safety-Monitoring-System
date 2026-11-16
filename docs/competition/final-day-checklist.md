# Chung kết 14/11/2026, 8h sáng, phòng C404B: làm gì, theo thứ tự

Thể lệ: 1 file PPT (4–5 slide) + sản phẩm HW/SW demo; **5 phút trình bày + 5 phút demo + hỏi đáp**.
File PPT: `slides/chung_khao_edge_ai_lab_safety.pptx` (ghi chú người nói có mốc thời gian trong phần Notes).

## Mang theo
- [ ] Board trong vỏ + camera C270 + dây LAN (board ↔ laptop) + **nguồn 5 V / 4 A** (hoặc sạc dự phòng đủ dòng 5 V / 3 A trở lên, cáp USB-C tốt)
- [ ] Laptop (đã `make demo-day` thử ở nhà) + sạc laptop + file PPT + `poster_A2_in.pdf`
- [ ] Điện thoại có Zalo (nhận tin) + cáp (chia sẻ mạng USB nếu cần)
- [ ] Video demo dự phòng quay ở lab (file trên máy/điện thoại), mở được không cần mạng
- [ ] Một áo blouse thật + một áo tối màu (đóng vai "không áo"); không mặc áo thun trắng khi đóng vai không áo
- [ ] `secrets/board.env` có mật khẩu board (`BOARD_SUDO_PASS=...`), file `secrets/zalo.json` đã ở trên board

## Nếu có board ở nhà tối nay (board cắm điện)
1. `make demo-day ARGS="--net"` (cần laptop có Wi-Fi/hotspot; bỏ `--net` nếu không thử Zalo). Lệnh này: đợi board khởi động, chỉnh đồng hồ, chép mã mới (**có bảng "Hiệu năng hệ thống"**), khởi động lại dịch vụ, kiểm tra, mở trang ở chế độ trình chiếu.
2. Xem trang: góc phải có 6 thanh (FPS, độ trễ, NPU, CPU, RAM, nhiệt). Nếu số nào lạ, chụp màn hình gửi lại.
3. Rút điện, cắm lại: xem đèn LED nháy thử 1,5 s rồi xanh; sau khoảng 1 phút `make demo-open` mở lại trang. Nếu vậy thì **chỉ cần cắm điện là tự chạy**.

## Sáng mai, tại phòng C404B
1. Đặt camera **đúng góc huấn luyện**: người đứng xa, thấy trọn thân, nền tủ như ảnh poster. Chụp ảnh vị trí.
2. Cắm điện board (và camera trước khi cắm điện nếu được). Cắm dây LAN vào laptop.
3. **Một lệnh:** `make demo-day ARGS="--no-deploy"` (board đã có mã mới từ tối qua) hoặc `make demo-open`. Trang tự mở ở chế độ trình chiếu tại http://localhost:8000/#present.
4. Phép thử góc camera (10 giây): một người mặc áo blouse đứng **một mình** trước camera → phải thấy khung **xanh** "Có áo lab". Không xanh = sai góc, chỉnh camera cao/thấp/xa.
5. Bấm **V** vẽ lại vùng nguy hiểm theo hình camera hiện tại, lưu.
6. Zalo: `make board-internet MODE=on` (laptop có mạng) rồi gửi thử; không có mạng thì bỏ qua, demo vẫn chạy.
7. Bấm vào trang một lần để có âm bíp; phím **P** trình chiếu, **F** toàn màn hình.

## Khi trình bày (5 phút) và demo (5 phút)
- Slide 1–5 theo ghi chú. Kịch bản demo 5 bước ở slide 5 (và `docs/competition/presentation-script.md` §4).
- Nói trước: "Hệ thống chờ 3 giây để chắc chắn nên báo động hiện sau khoảng 3–5 giây."
- Một người vào vùng mỗi lần; không đứng sát người mặc áo.

## Tối nay (không có board): làm gì được
1. **Tập nói** theo ghi chú slide: tổng khoảng 4 phút, còn dư thời gian để nói chậm. Bấm giờ một lần.
2. **Tập demo bằng laptop:** `make dashboard-demo` chạy cùng trang web trên video lab (khoảng 25 giây để lên), tự mở ở chế độ trình chiếu. Đây cũng là **phương án dự phòng thứ hai** nếu board hỏng sáng mai: nói rõ "đây là hệ thống chạy trên laptop với video quay tại lab". Dừng bằng Ctrl+C trước khi chạy `make demo-day`, vì cùng cổng 8000.
3. **Sao chép file** sang điện thoại hoặc USB: `slides/chung_khao_edge_ai_lab_safety.pptx`, `poster/poster_A2_in.pdf`, `report/bai_bao_ngan.pdf`, video dự phòng.
4. **Laptop:** sạc đầy, đã đăng nhập Wi-Fi hoặc hotspot để thử `--net`; mở thử file PPT và kiểm tra ghi chú người nói.
5. Đến lab **sớm 30–45 phút** để chạy `make demo-day` lần đầu (chép mã mới có bảng hiệu năng và khởi động lại dịch vụ mất khoảng 3–4 phút) và đặt camera.

## Chạy như hộp rời với màn hình HDMI (không cần laptop)
Cài một lần khi board đang nối laptop: `make kiosk-setup`. Lệnh kiểm tra board có giao diện đồ họa và trình duyệt chưa, rồi bật tự đăng nhập và tự mở trang web toàn màn hình.
Sau đó: **cắm màn hình/máy chiếu HDMI vào board trước, rồi cắm điện** → khoảng 1–2 phút trang giám sát tự hiện. Tắt: `make kiosk-off`.
Lưu ý: không có laptop thì không đổi được cấu hình bằng lệnh; vẽ lại vùng cần chuột cắm vào board (phím V). Đồng hồ board sai giờ nếu chưa nối laptop (chỉ ảnh hưởng giờ trên ảnh/tin nhắn).

## Demo bằng VIDEO lab thật (khi camera không ở đúng góc huấn luyện)
Chạy HỆ THỐNG THẬT (NPU, đèn LED, OLED, web, Zalo) nhưng nguồn ảnh là video quay ở lab thay cho camera:
```
make demo-video                          # video mặc định: data/interim/demo/boot014_demo.mp4 (60 s, lặp lại)
make demo-video VIDEO=/đường/dẫn/video.mp4   # video khác (chỉ hợp với video tua nhanh như boot014)
make demo-day                            # quay về camera thật
```
Board nhớ chế độ này (file `live.env`), nên **cắm điện là tự chạy video**; chỉ cần `make demo-open` để mở trang. Vùng nguy hiểm của chế độ video là vùng đã vẽ sẵn cho video này, không đụng tới vùng vẽ cho camera thật.
Nói rõ với hội đồng: "đây là video quay tại phòng thí nghiệm, hệ thống xử lý thật trên NPU".

## Sự cố thường gặp
| Hiện tượng | Xử lý |
|---|---|
| `make demo-day` báo board không tới được | kiểm dây LAN, nguồn board, đợi 1 phút sau khi cắm điện, chạy lại |
| Trang hiện MẤT TÍN HIỆU | đúng thiết kế khi mất hình: cắm lại camera, đợi 10 s |
| Camera nhận không ra áo (khung cam/đỏ cho người mặc áo) | sai góc: chỉnh camera về góc huấn luyện; hoặc `make live-config CONFIG=demo` (ngưỡng áo 0,5) |
| Không có tin Zalo | board chưa có mạng; bỏ qua, nói "tin Zalo cần Internet", mọi thứ khác vẫn chạy |
| Laptop không mở được trang | `ssh -N -L 8000:127.0.0.1:8000 orangepi-lan` rồi mở http://localhost:8000/#present |
| Hỏng hết | bật video dự phòng, nói rõ "đây là video quay tại vị trí camera cuối" |

## Câu hỏi hội đồng chung kết có thể hỏi thêm (trả lời bằng số đã đo)
1. **Đóng góp mới của nhóm là gì?** Không phải chạy YOLO, mà: INT8 đầu ra tách (mất 0,4 điểm thay vì 18), kiểm tra tương đương NPU–CPU (bắt 4 lỗi chuyển đổi), chế độ thích nghi (1,98 W so với 3,34 W), luật "chỉ kết luận khi thấy rõ trang phục" (31 → 6 cảnh báo, cả 6 đúng).
2. **Độ chính xác bao nhiêu?** Theo từng người trên ảnh kiểm tra độc lập: phát hiện 75–86% (góc 1), báo nhầm 7–9%; còn bỏ sót, nên là công cụ hỗ trợ, mỗi cảnh báo có ảnh để người xác nhận. Chưa phải tỉ lệ phát hiện sự kiện liên tục.
3. **Có tiết kiệm điện không?** Đo bằng đồng hồ USB-C: nghỉ 0,86 W, chạy 2,2 W; thích nghi thấp hơn chế độ song song 1,36 W (41%) ở cùng tốc độ khung. Sai số ±10%, đo trên video phát lại, chưa gồm camera.
4. **Đổi góc camera thì sao?** Đã gặp thật: góc rất khác thì mô hình không nhận áo (0/122 khung). Cách xử lý: cố định camera, kiểm tra trước bằng một người mặc áo, thu dữ liệu tại vị trí đó nếu buộc đổi.
5. **Sao báo động trễ 3–5 giây?** Cố ý: bước xác nhận 3 giây để một khung nhiễu không bật còi.
6. **Hình người có bị gửi đi đâu không?** Không. Xử lý trên bo mạch; Zalo chỉ nhận tin chữ; ảnh bằng chứng nằm trên bo mạch và xem qua trang web.
7. **Sau này mở rộng gì?** Dữ liệu đúng vị trí lắp đặt, tập validation riêng để chọn ngưỡng, đo công suất với camera thật, đánh giá té ngã và lửa/khói.
Chi tiết hơn: `docs/competition/judge-questions.md`, `docs/competition/competition-claims.md`.
