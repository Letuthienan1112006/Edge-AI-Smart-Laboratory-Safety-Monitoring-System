# Giao thức đo lường

Mục tiêu: mọi con số trong bài NCKH phải tái lập được và không bị nhiễu bởi những thứ
người đọc không nhìn thấy. Làm đúng file này thì phần thực nghiệm của bạn mạnh hơn phần
lớn paper cùng chủ đề.

## 1. Quy tắc bất biến

1. **Không báo cáo FPS trung bình đơn lẻ.** Luôn báo `mean / p50 / p90 / p99` của
   latency mỗi stage. FPS trung bình che mất các frame chậm — mà an toàn lao động thì
   frame chậm mới là frame quan trọng.
2. **Không báo cáo burst.** Mỗi lần đo chạy ≥ 15 phút. Ghi nhiệt độ CPU theo thời gian.
   Báo cáo *sustained* FPS = trung bình của 5 phút cuối.
3. **Không so sánh xuyên thiết bị.** Mọi bảng đều có cột "thiết bị đo".
4. **Không so sánh xuyên dataset.** Số của paper khác chỉ vào phần Related Work kèm ghi
   chú không-so-sánh-trực-tiếp.
5. **Mỗi cấu hình chạy 3 lần**, báo cáo trung bình ± độ lệch chuẩn. Pi 4 nhiễu nhiều
   (SD card, throttle, tiến trình nền).

## 2. Chuẩn bị Pi 4 trước mỗi lần đo

```bash
sudo systemctl stop bluetooth           # tắt thứ không dùng
vcgencmd measure_temp                   # phải < 50°C trước khi bắt đầu
vcgencmd get_throttled                  # phải là throttled=0x0
```
Sau khi đo xong, chạy lại `get_throttled`. Nếu khác `0x0` → **kết quả đó bỏ**, ghi rõ
trong log là đã bị throttle. Bench harness (`labsafe.bench`) tự động đọc và ghi hai giá
trị này.

## 3. Đo công suất

USB-C power meter đo **toàn bộ board**, không phải riêng inference. Nên:
- Đo `P_idle`: pipeline đứng yên, camera bật, không inference — ≥ 5 phút.
- Đo `P_run`: chạy cấu hình cần đo — ≥ 15 phút.
- Báo cáo **cả hai** và `ΔP = P_run − P_idle` (đây mới là chi phí của AI).
- **Energy per event** = `ΔP × thời gian chạy / số sự kiện phát hiện đúng`. Đây là metric
  trung tâm của đề tài; nó chỉ có nghĩa khi đi kèm occupancy rate của đoạn video đo.

Ghi tay vào `runs/<exp_id>/power.csv` theo format `timestamp_s,volts,amps` (đọc từ màn
hình power meter mỗi 30s là đủ), hoặc dùng `--power-csv` của bench runner để merge.

## 4. Chia slice cho tập test (bắt buộc — xem research-plan §2.5)

Tập test phải gán tag cho mỗi ảnh, tối thiểu:

| Chiều | Giá trị | Cách gán |
|---|---|---|
| distance | `near` (<2m) / `far` (≥2m) | ước lượng khi chụp, hoặc theo chiều cao bbox person (px) |
| light | `bright` / `dim` | đèn phòng bật / chỉ đèn bàn |
| occlusion | `clear` / `occluded` | PPE bị che một phần bởi thiết bị, tóc, tay |

Báo cáo mAP cho: tổng, và **từng slice**. Một tối ưu hoá làm rớt 1% mAP tổng nhưng rớt
12% trên slice `far+occluded` là một tối ưu hoá **không** được coi là miễn phí — và
chính đó là phát hiện đáng viết trong bài.

Tag lưu ở `data/processed/test/slices.csv`: `image_id,distance,light,occlusion`.

## 5. Occupancy rate (cho RQ2)

Với mỗi video đo, occupancy = tỉ lệ frame có ít nhất một người. Bench harness ghi sẵn
vào `metrics.json`. Bạn cần ≥ 4 video ở các mức occupancy khác nhau (~10/30/50/80%) để
vẽ được đường cong E6. Có thể ghép video thủ công từ các đoạn quay thật.

## 6. Detection latency (cái giá của gating — cho RQ2)

Với mỗi sự kiện an toàn trong video test, ghi tay `frame_id` lúc sự kiện **bắt đầu xảy
ra** (người bước vào vùng cấm, người ngã chạm đất, người bỏ kính). Latency = thời điểm
hệ thống báo động − thời điểm đó. So sánh giữa E1 (chạy song song) và E5 (adaptive).

Ground truth sự kiện lưu ở `data/processed/events.csv`: `video,frame_id,event_type`.

## 7. Bảng kết quả cuối cùng nên có dạng

| Exp | Runtime | Prec | Input | mAP@.5 | mAP far+occl | p50 ms | p99 ms | Sustained FPS | RAM MB | ΔP (W) | J/event |
|---|---|---|---|---|---|---|---|---|---|---|---|

Cột `mAP far+occl` và `J/event` là hai cột làm bài của bạn khác với các bài khác.
