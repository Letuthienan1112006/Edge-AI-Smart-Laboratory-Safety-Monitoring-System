# Sổ tay buổi lab: thử nghiệm + quay video demo

Viết 23/10/2026. Làm theo đúng thứ tự từ trên xuống, tổng khoảng 2 giờ. Mọi lệnh chạy trên **laptop**, trong thư mục `AI_Challenge`.
Phạm vi: chỉ luật **người không mặc áo lab trong vùng cấm**; té ngã và lửa/khói đã tắt.
Bảng PASS/FAIL đầy đủ D1–D10 nằm trong `docs/competition-demo-checklist.md`. Ở đây chỉ ghi thứ tự làm và lệnh cần chạy.

## 0. Mang theo

- Board + nguồn, camera Logitech C270 + kẹp/chân cố định, cáp LAN, laptop + sạc, ổ cắm kéo dài.
- **2 áo lab**, áo trong **màu tối** (người làm "không áo"), **1 áo thun/sơ mi trắng thường** (cảnh 05).
- Băng keo màu để dán ranh giới vùng trên sàn (hội đồng nhìn thấy vùng, mình đứng đúng chỗ).
- Điện thoại + giá đỡ: quay cảnh người đi vào **cùng lúc** với màn hình (video dự phòng thuyết phục hơn khi thấy cả hai).
- Một người thứ hai cho cảnh hai người (nếu có).

## 0b. Báo về Zalo (làm trước ở nhà, 5 phút; không bắt buộc)

Khi hệ thống báo động, điện thoại nhận tin Zalo, kể cả khi đang khóa màn hình. Chỉ gửi chữ; ảnh bằng chứng xem trên trang web.

1. Trong Zalo, tìm OA **Zalo Bot Manager**, chọn **Tạo bot**, đặt tên bắt đầu bằng "Bot" (ví dụ "Bot An Toan Lab").
2. Zalo gửi **Bot Token** vào tin nhắn. Chép token vào file `~/zalo_token.txt` trên laptop. **Không dán token vào chat, không commit.**
3. `make zalo-setup TOKEN=~/zalo_token.txt`. Trong lúc lệnh chờ, mở bot trong Zalo và nhắn "hi". Bạn sẽ nhận tin thử "✅ Đã kết nối…".
4. Khi board đã cắm, chạy `make zalo-push`. Lệnh chép khóa lên board rồi gửi tin thử **từ board**.
   - Board cần có internet: cắm điện thoại vào cổng USB của board, bật **Điểm phát sóng → Chia sẻ Internet qua USB**.
   - Dịch vụ tự nhận khóa, không cần khởi động lại.

Mỗi tin cách nhau ít nhất 10 s. Nhiều người vi phạm cùng lúc thì gộp thành một tin. Mất mạng thì tin đó bị bỏ qua, nhận diện vẫn chạy bình thường.
Ở bước D3 của mục 4, kiểm xem tin Zalo có tới không và tới sau bao nhiêu giây.

## 1. Đặt camera (15 phút)

1. Cắm camera vào board **trước**, rồi cắm điện board, nối cáp LAN vào laptop. Chờ khoảng 60 giây.
2. `make install-live-service` (gõ mật khẩu sudo của board). Lệnh này chép code + model mới nhất lên board,
   cài lại dịch vụ (bản 23/10 đổi được config, xem bước 4) và khởi động lại. Chỉ cần làm **một lần**.
3. `make lab-preflight OPEN=1`: kiểm mọi thứ và mở trang web. Phải ra **READY**. Mục nào có `!!` thì làm theo gợi ý trên chính dòng đó.
4. Nhìn hình trên trang web, chỉnh camera sao cho:
   - thấy **trọn người từ đầu tới chân** ở toàn bộ lối đi vào vùng;
   - người đứng trong vùng **không bị cắt ở mép dưới** khung hình;
   - không ngược sáng (cửa sổ không nằm sau lưng người).
5. Chụp ảnh vị trí camera bằng điện thoại để lần sau (và lúc thi) đặt lại giống vậy.

## 2. Vẽ vùng (5 phút)

Vùng đang lưu trên board là **vùng vẽ ở nhà**, nên phải vẽ lại:

1. Dán băng keo lên sàn.
2. Trên trang web, thoát chế độ trình chiếu (phím **P**) nếu đang bật.
3. Bấm **"Vẽ lại vùng nguy hiểm"** rồi bấm từng góc theo đường băng keo.
   - Vẽ trên **sàn**, vì hệ thống xét điểm ở **chân** người.
   - Bấm **Lưu**, chọn **xóa vùng cũ**.
4. Tải lại trang (F5) để chắc vùng vẫn còn.

## 3. Thử áo lab thật (10 phút)

Đứng **một mình** trước camera, ở chỗ sẽ đứng khi demo. Chạy:

```
python3 scripts/coat_check.py --labels "khong ao" "ao 1" "ao 2" "ao 1 mo khuy" --seconds 20
```

| Kết quả | Làm gì |
|---|---|
| "ao 1"/"ao 2" ≥ 90 % **và** "khong ao" ≤ 5 % | Giữ config chuẩn |
| Áo nào đó < 80 % | `make live-config CONFIG=demo` (ngưỡng 0,5, nhớ áo 3 s), chạy lại `coat_check` |
| "khong ao" > 10 % | **Dừng lại.** Ghi lại tình huống và báo Claude; đừng hạ ngưỡng thêm |

Dùng áo nào đạt điểm cao hơn để đi demo trước hội đồng.

## 4. Diễn tập (20 phút): D1–D5 mỗi cảnh 3 lần

Trang web: bấm **P** (trình chiếu). Lần bấm phím đầu tiên cũng mở khóa **âm báo**.

- **D1:** phòng trống → **AN TOÀN**.
- **D2:** không áo, đứng **ngoài** vùng → **CHÚ Ý**, khung cam, không báo.
- **D3:** không áo, vào vùng, đứng yên 8 s → khung đỏ "VI PHẠM", **BÁO ĐỘNG** (khoảng 3 s sau khi vào), tiếng bíp, ảnh bằng chứng.
- **D4:** mặc áo, vào cùng đường → khung xanh "Có áo lab", **không báo**.
- **D5:** ra khỏi vùng sau khi bị báo → hết báo.

Ghi PASS/FAIL vào bảng ở `docs/competition-demo-checklist.md`. Chụp màn hình 3 trạng thái AN TOÀN / CHÚ Ý / BÁO ĐỘNG.

## 5. Quay video demo dự phòng (20 phút, 2–3 lần quay)

1. Trang web ở chế độ **P**, cửa sổ trình duyệt to vừa màn hình. Bấm vào trang một lần để bật âm.
2. Bật quay bằng `make record-demo NAME=lan1`, rồi **bấm vào cửa sổ trình duyệt** khi được hỏi.
   - Lệnh ghi 2 file vào `runs/demo_video/`:
     - `*_stream.mp4`: hình camera có khung nhận diện, đúng thứ AI thấy;
     - `*_screen.mp4`: cả trang web, có tiếng bíp.
   - Dừng bằng **Ctrl+C**.
3. Đồng thời quay bằng điện thoại cảnh người đi vào (lấy được cả màn hình laptop thì càng tốt).

**Kịch bản khoảng 2 phút** (một người làm được, trừ cảnh 5):

| # | Giây | Làm gì | Hội đồng phải thấy |
|---|---|---|---|
| 1 | 10 | Phòng trống | AN TOÀN |
| 2 | 20 | Mặc áo lab, đi vào vùng, đứng 5 s, đi ra | Khung **xanh** "Có áo lab 0,9x", vẫn AN TOÀN |
| 3 | 25 | Cởi áo (ngoài khung), đi vào vùng, đứng 8 s | Khung **đỏ** "VI PHẠM", **BÁO ĐỘNG**, bíp, ảnh bằng chứng |
| 4 | 25 | Vẫn trong vùng, **mặc áo vào** tại chỗ, đứng 8 s | Đỏ → xanh, hết báo |
| 5 | 20 | (2 người) một có áo, một không, cùng vào | Chỉ người không áo bị đỏ |
| 6 | 10 | Không áo, đứng **ngoài** vùng | Khung cam "Không áo lab", CHÚ Ý, không báo |
| 7 | 10 | Đứng sát camera chỉ thấy mặt | Khung **xám** "Chưa thấy rõ áo", không báo |

Quay xong, mở thử file trên laptop. Lần quay tốt nhất là video dự phòng.

## 6. Quay dữ liệu chấm điểm (20 phút)

Phần này quay video **thô** có mốc thời gian, để chấm nhận diện theo sự kiện ở nhà. Phải dừng dịch vụ vì camera chỉ dùng được cho một chương trình:

```
ssh -t orangepi-lan 'sudo systemctl stop labsafe-live'
ssh -t orangepi-lan 'cd AI_Challenge && python3 scripts/collect_guided.py --plan configs/collection_plan_demo.yaml'
ssh -t orangepi-lan 'sudo systemctl start labsafe-live'
```

- Có 8 cảnh, khoảng 7 phút quay. Mỗi cảnh nhấn Enter để bắt đầu, rồi làm đúng câu nhắc hiện trên màn hình (có tiếng bíp).
- Cảnh 07 cần 2 người; nếu chỉ có một người thì Ctrl+C và bỏ qua.
- Bị ngắt giữa chừng thì chạy lại lệnh thứ hai: những cảnh đã quay xong sẽ tự bỏ qua.

## 7. Kiểm mất camera (5 phút)

- **D8:** rút camera → trong vài giây trang phải hiện **MẤT TÍN HIỆU**, không bao giờ hiện AN TOÀN.
- **D9:** cắm lại → tự phục hồi trong khoảng 10–20 s, không cần gõ lệnh.

## 8. Trước khi về: mang dữ liệu về laptop

```
make fetch-sessions                     # video thô của bước 6
make fetch-violations                   # ảnh bằng chứng các lần báo
scp orangepi-lan:AI_Challenge/data/zones_live.yaml configs/zones_lab_final.yaml   # vùng vừa vẽ
make lab-preflight                      # ghi lại trạng thái cuối
```

Để board **chạy tiếp** nếu được; nếu phải rút điện thì cứ rút, không cần tắt trước.

## Sau buổi lab (ở nhà, Claude làm)

- Chấm sự kiện:
  `PYTHONPATH=src python3 scripts/eval_events.py --sessions 'data/raw/lab_sessions/DM_*' --zones configs/zones_lab_final.yaml --grace 0 0.5 1 1.5 --attire-min-conf 0.5 0.6 --memory 2 3`
  Kết quả này quyết định giữ config chuẩn hay config demo.
- Cắt và ghép video dự phòng; cập nhật `docs/competition-claims.md` bằng số đo ở lab.
- Cảnh 08 (áo lab thứ 2) và ảnh khó: gán nhãn trên Roboflow để fine-tune nếu `coat_check` yếu.
