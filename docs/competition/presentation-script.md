# Kịch bản thuyết trình trước hội đồng

Cập nhật 23/10/2026. Người nghe: hội đồng chấm phần **nhận diện AI**. Mọi con số trong file này lấy từ `docs/competition/competition-claims.md`
và `report/bao_cao.pdf`; con số nào không có ở đó thì không dùng. Thời lượng gợi ý: 10 phút nói + 3 phút demo trực tiếp (cắt bớt mục 6 và 7 nếu chỉ có 8 phút).

## 0. Một câu để hội đồng nhớ

> Một bo mạch Orange Pi 5 giá rẻ, không cần mạng, nhìn camera và báo động khi có người **không mặc áo blouse bước vào vùng nguy hiểm**;
> điểm mới không phải là chạy YOLO, mà là cách làm cho nó **chạy đúng, đo trung thực và biết nói rõ giới hạn của mình**.

## 1. Luồng logic: hệ thống quyết định như thế nào (nói trong 90 giây, chỉ vào hình 1 của báo cáo)

1. **Camera** gửi ảnh 640x480, 30 khung/giây. Hệ thống luôn lấy ảnh *mới nhất*, không suy luận lại ảnh cũ.
2. **Mô hình (tầng A, trên NPU)** tìm hai thứ trong mỗi khung: `người` và `áo blouse`, mỗi khung kèm một điểm tin cậy.
3. **Bộ bám** giữ cho mỗi người một số định danh qua các khung (#48 vẫn là #48 khi họ đi lại).
4. **Gán áo:** một khung áo thuộc về *một* người, là người chứa phần lớn diện tích khung áo (từ 50% trở lên) và áo phải có điểm từ 0,6. Người được coi là **đang mặc áo**
   nếu thấy áo trong 2 giây gần nhất (để một khung model bỏ sót không làm đổi kết luận).
5. **Vùng nguy hiểm:** lấy **điểm chân** của người (giữa cạnh dưới khung) và hỏi "chân có nằm trong đa giác vùng không". Dùng chân thay vì tâm khung vì người đứng sau vùng có tâm khung trôi vào trong vùng.
6. **Quyết định vi phạm:** *trong vùng* VÀ *không mặc áo* VÀ *nhìn thấy được quần áo*. Nếu chỉ lộ mặt, bị che, hoặc quá nhỏ thì xếp là "chưa thấy rõ áo" (khung xám), **không** tính vi phạm.
7. **Chống rung, chống lặp:** sự kiện phải kéo dài 3 giây mới báo động; cùng một người chỉ báo lại sau 60 giây.
8. **Đầu ra:** trang web (4 trạng thái), ảnh bằng chứng, đèn LED + màn OLED trên bo mạch, tin Zalo tới điện thoại (tùy chọn).

Câu chốt của mục này: *"Hệ thống có thể nói 'tôi chưa chắc' thay vì đoán, và không bao giờ báo AN TOÀN khi mất hình."*

### Bốn trạng thái (một bảng, dùng chung cho trang web, đèn và OLED)

| Trạng thái | Điều kiện | Đèn | OLED |
|---|---|---|---|
| AN TOÀN | camera có hình, không ai vi phạm | xanh sáng | "AN TOÀN", số người |
| CHÚ Ý | người không áo ngoài vùng, hoặc vi phạm đang được xác nhận, hoặc vừa có cảnh báo (giữ 10 giây) | vàng sáng | "CHÚ Ý" |
| BÁO ĐỘNG | người không áo trong vùng, đã qua 3 giây chống rung | **đỏ nháy** | nháy đảo màu, số người vi phạm |
| MẤT TÍN HIỆU | không có khung mới quá 3 giây | vàng nháy chậm | "MẤT TÍN HIỆU" |

## 2. Thuật toán và vì sao chọn như vậy (mỗi ý một câu "vì sao")

| Chủ đề | Nói gì | Vì sao | Số / bằng chứng |
|---|---|---|---|
| Model | YOLO26n, 2 lớp `người` và `áo blouse`, tinh chỉnh từ COCO, 3 seed | Bốn kiến trúc nano ngang nhau; model lớn hơn chỉ hơn khoảng 2 điểm nhưng chậm 2-3 lần. Nút thắt là dữ liệu | báo cáo mục 5 |
| INT8 đầu ra tách | Tách tọa độ khung và điểm lớp thành 6 tensor, mỗi cái một thang 8-bit | INT8 mặc định gộp chung một thang làm mép khung bị lượng tử hóa thô | mAP50-95 mất ~17-20 điểm → tối đa ~1 điểm |
| Cổng parity | So kết quả NPU với CPU trên cùng ảnh trước khi triển khai | Lỗi chuyển đổi sang NPU là lỗi *im lặng*: tốc độ vẫn đẹp nhưng kết quả sai | bắt được 4 lỗi mà số tốc độ không thấy |
| Điểm chân trong đa giác | Thuật toán tia, đa giác chuẩn hóa 0..1 | Không phụ thuộc độ phân giải, vẽ lại được trên web khi đổi chỗ camera | báo cáo mục 2.3 |
| Một áo, một người | Áo gán cho người chứa nhiều diện tích áo nhất | Hai người đứng sát không "mượn" áo của nhau | còn một hạn chế đo được, xem mục 5 |
| Chỉ phán khi thấy quần áo | Che/nhỏ/chỉ lộ mặt → "chưa thấy rõ áo" | "Không thấy áo" khác "không mặc áo". Trên video lab đông người, 31 cảnh báo → 6, cả 6 đúng | báo cáo mục 2.3 |
| Chống rung có "ân hạn" 1 giây | Một khung vắng sự kiện < 1 giây không xóa đồng hồ 3 giây | Một khung model bỏ sót áo là đồng hồ khởi động lại và cảnh báo không bao giờ phát | KNOWN_ISSUES #43 |
| Chế độ thích nghi | Model phụ chạy bất đồng bộ trên nhân NPU riêng, chỉ khi cần | Giảm công NPU ~2,3 lần, nhiệt thấp hơn 12,8 °C, điện 1,98 so với 3,34 W | **chưa đo watt**, không gọi là tiết kiệm điện |

Số hiệu năng (dùng đúng nhãn, đừng đổi chỗ cho nhau): Stage A INT8 khoảng 29-30 FPS (trần camera); toàn pipeline khi phát video 30 FPS đạt 29,02 FPS; dashboard thật đạt 25,9 FPS.

## 3. Cấu trúc slide gợi ý (12 slide, ~10 phút)

| # | Slide | Thông điệp | Phút |
|---|---|---|---|
| 1 | Bài toán | Áo blouse là lớp bảo vệ đầu tiên; nhắc thủ công không theo dõi liên tục được; đưa video lên cloud lộ hình người | 0,5 |
| 2 | Yêu cầu | Chạy tại chỗ, thời gian thực, đúng tác vụ, số liệu trung thực | 0,5 |
| 3 | Kiến trúc | Hình 1: camera → tầng A (NPU) → bộ bám → gán áo → luật vùng → chống rung → đầu ra | 1 |
| 4 | Luồng logic | Bảng 4 trạng thái + 3 điều kiện vi phạm (mục 1 ở trên) | 1 |
| 5 | Vì sao khó | INT8 làm mất 17-20 điểm mAP; lỗi chuyển đổi im lặng | 1 |
| 6 | Cách sửa | Đầu ra tách + cổng parity: mất tối đa ~1 điểm | 1 |
| 7 | Dữ liệu ít nhãn | ~1.700 ảnh, 2 góc; chọn ảnh khó (độ chính xác áo 0,52→0,76); thêm mẫu âm tính COCO (áo thun trắng 20/20 → 0-4/20); test tách theo phiên | 1 |
| 8 | Kết quả | Luật vùng: bắt 75-86% người không áo, báo nhầm 7-9% (góc 1); còn bỏ sót đáng kể | 1 |
| 9 | Chạy liên tục | 24 giờ, RSS đi ngang ~218 MB, 0 lỗi; tự khởi động lại 9 giây | 0,5 |
| 10 | Bài học ở phòng thật | **Góc camera** là một phần của dữ liệu huấn luyện: ở góc lạ model không nhận áo (0/122 khung); đặt về góc cũ thì nhận ~98% (4 người, 25 giây) | 1 |
| 11 | Giới hạn | Công suất đo trên video phát lại (chưa có camera thật); té ngã, lửa/khói đã tắt; test tại lab chưa có đáp án; tin Zalo cần Internet | 0,5 |
| 12 | Hướng tiếp | Dữ liệu tại đúng vị trí camera + validation riêng; đo W/Wh; mở rộng nhiều người | 0,5 |

Slide 10 là chỗ **đáng khoe nhất** về mặt khoa học: bạn phát hiện, đo và giải thích một lỗi thật thay vì giấu nó.

## 4. Demo trực tiếp (3 phút) và video dự phòng

Chuẩn bị: `make lab-preflight OPEN=1` ra READY; trang ở chế độ trình chiếu (phím P), đã bấm vào trang một lần để có tiếng; `make board-internet MODE=on` nếu muốn tin Zalo.

| Bước | Làm gì | Hội đồng thấy |
|---|---|---|
| 1 | Không ai trong khung | AN TOÀN, đèn xanh |
| 2 | Người **mặc áo blouse** bước vào vùng, đứng 5 giây | Khung xanh "Có áo lab 0,9x", vẫn AN TOÀN |
| 3 | Người **không áo** (áo tối) bước vào vùng, đứng 8 giây | Khung đỏ "VI PHẠM", BÁO ĐỘNG, bíp, đèn đỏ nháy, OLED nháy, ảnh bằng chứng, tin Zalo |
| 4 | Người đó **mặc áo vào** tại chỗ | Đỏ chuyển xanh, hết báo |
| 5 | Một người không áo đứng **ngoài** vùng | Khung cam, CHÚ Ý, không báo |

- Nói trước khi bước vào: "Hệ thống chờ 3 giây để chắc chắn, nên báo động hiện sau khoảng 3-5 giây chứ không tức thì." (Trễ trung vị đo được: 5,24 giây tính từ mốc kịch bản, gồm 3 giây chống rung và thời gian người diễn phản ứng, khoảng 1-2 giây.)
- Chỉ **một người vào vùng tại một thời điểm** và không đứng sát người mặc áo (xem hạn chế "mượn áo" ở mục 5).
- Áo bên trong màu tối, không mặc áo thun trắng khi diễn vai "không áo": mô hình vẫn có thể nhầm áo trắng thường là áo blouse.
- **Video dự phòng:** nếu camera, mạng hay board có sự cố thì bật video đã quay ở lab; nói rõ "đây là video quay tại vị trí camera cuối".
- Nếu trang hiện MẤT TÍN HIỆU, đừng giấu: nói "đây chính là hành vi thiết kế: mất hình thì không bao giờ báo an toàn" rồi cắm lại camera.

## 5. Hạn chế đã biết của chính luồng logic (nói trước khi bị hỏi)

| Hạn chế | Hệ quả | Đã đo / đã xử lý |
|---|---|---|
| "Mượn áo": người đứng sát người mặc áo được coi là có áo trong tối đa 2 giây | Một người không áo đứng sát có thể chưa bị báo kịp | Đo ở lab: 48% và 32% số lần thăm dò cho hai người đứng sát; giảm bằng cách xóa trí nhớ áo khi bộ bám ghép theo khoảng cách tâm |
| Nhầm áo trắng thường với áo blouse | Người mặc áo thun/polo trắng có thể bị coi là có áo | Đã giảm: áo thun trắng 20/20 → 0-4/20 sau khi thêm mẫu âm tính; ở lab hai người áo trắng không có khung áo nào; vẫn nên kiểm cảnh này |
| Đổi mã định danh khi người che nhau | Có thể sinh một cảnh báo thêm cho cùng một người | Nới giữ mã lên 3 giây (từ 24 mã cho 2-4 người) |
| Đổi góc camera mạnh | Model không nhận áo (0/122 khung ở góc lạ) | Đã đo, ghi KNOWN_ISSUES #52; cách xử lý: cố định camera, kiểm trước khi dùng, thu dữ liệu đúng vị trí nếu bắt buộc đổi |
| Trễ báo động 3-5 giây | Không phải phản ứng tức thời | Cố ý, để một khung nhiễu không bật còi; trễ trung vị 5,24 giây trên 7 sự kiện ở nhà |
| Zalo cần Internet và chỉ gửi chữ | Mất mạng thì không có tin nhắn (nhận diện, đèn, trang vẫn chạy) | Không gửi ảnh người lên máy chủ bên thứ ba vì lý do riêng tư |

## 6. Năm câu hỏi khó nhất, trả lời trong 20 giây (chi tiết ở `docs/competition/judge-questions.md`)

1. **"Điểm mới là gì, YOLO trên Orange Pi có nhiều người làm rồi."** → Quy trình deploy đáng tin: cổng parity, đầu ra tách, thích nghi, đánh giá tách theo phiên và chạy 24 giờ.
2. **"Accuracy có đủ cho hệ thống an toàn không?"** → Chưa đủ để thay con người: bắt khoảng 75-90% người không áo, vẫn bỏ sót. Vì vậy là hệ thống *hỗ trợ giám sát*, mỗi cảnh báo có ảnh để người xác nhận.
3. **"7/7 sự kiện có phải 100%?"** → Không: 7 sự kiện, một người, 8 phút, quay ở nhà. Chỉ chứng minh luồng end-to-end chạy.
4. **"Nếu đổi chỗ đặt camera thì sao?"** → Chúng tôi đã gặp thật: ở góc rất khác model không nhận áo. Phát hiện nhờ số đo theo từng người; cách xử lý là cố định camera và kiểm trước, hoặc thu dữ liệu ở vị trí đó.
5. **"Có tiết kiệm điện không?"** → Có đo bằng đồng hồ USB-C: nghỉ 0,86 W, sản phẩm 2,20 W, thích nghi 1,98 W, chạy hết 3,34 W (±10%). Thích nghi thấp hơn chạy hết 1,36 W ở cùng 29 FPS. Hạn chế: video phát lại, mỗi pha đo một lần.

## 7. Không được nói

- "Hệ thống an toàn tuyệt đối" hoặc "thay thế người giám sát".
- "Chịu được mọi góc camera" (mục 5 cho thấy điều ngược lại).
- "Dashboard chạy 30 FPS" (đã đo 25,9 FPS).
- "7/7 là 100% chính xác"; "tiết kiệm X% điện năng".
- Số 98% hay 0/122 ở lab như một *độ chính xác*: đó là phép đo một phiên ngắn, chỉ dùng để minh họa nguyên nhân.
- "Gửi ảnh qua Zalo": hệ thống chỉ gửi chữ.

## 8. Checklist trước giờ thuyết trình

- [ ] Board bật, `make lab-preflight OPEN=1` ra READY (đồng hồ board đúng, vùng vẽ đúng vị trí camera hiện tại).
- [ ] Camera ở đúng vị trí đã quay video; chụp ảnh vị trí để đặt lại được.
- [ ] Một người mặc áo đứng một mình trước camera: khung xanh. (Đây là phép thử 10 giây phát hiện sự cố góc camera.)
- [ ] Máy chiếu: trang đọc được từ chỗ ngồi hội đồng; âm thanh bíp nghe được.
- [ ] Video dự phòng mở được trên laptop, không cần Internet.
- [ ] Tin Zalo: đã `make board-internet MODE=on`, điện thoại nhận được tin thử.
- [ ] In hoặc mở sẵn `docs/competition/judge-questions.md` và `docs/competition/competition-claims.md`.
