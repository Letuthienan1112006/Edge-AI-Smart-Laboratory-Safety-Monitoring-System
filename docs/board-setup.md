# Setup Orange Pi 5 — từ lúc mở hộp tới `make probe` chạy được

Đọc theo thứ tự. Mỗi bước có lệnh xác minh — đừng bước tiếp nếu bước
xác minh không ra kết quả như mô tả. Phần lớn lỗi trên board là hậu quả
của việc bỏ qua một dấu tích ở đây rồi hai tuần sau mới lộ ra thành một
con số benchmark vô nghĩa.

---

## 0. Trước khi cắm điện

- [ ] Nguồn **USB-C PD 5V/4A trở lên**. Củ sạc điện thoại thường (5V/2A,
      5V/3A) sẽ gây undervoltage khi NPU+CPU cùng tải — triệu chứng là
      tụt xung âm thầm, không phải crash.
- [ ] Case có **quạt chủ động**. RK3588S nóng hơn Pi 4 đáng kể khi chạy
      NPU + 4 nhân A76 cùng lúc.
- [ ] Thẻ SD hoặc SSD NVMe (nếu board có khe M.2) — Class A2 trở lên nếu
      dùng SD.
- [ ] Cáp mạng (khuyến nghị dùng LAN khi benchmark — tắt WiFi loại bỏ
      một nguồn jitter khỏi p99 latency).

## 1. Flash OS

Dùng **Ubuntu 22.04 Jammy Server (aarch64) chính thức của Orange Pi**,
KHÔNG dùng bản desktop (desktop ăn RAM + có tiến trình nền chen vào phép
đo).

**⚠ Xác minh thật, 2026-09-28 — CHỈ dùng image chính thức, đừng dùng
image cộng đồng cho project này.** Đã thử image "Ubuntu Rockchip"
(Joshua-Riek, `ubuntu-24.10-preinstalled-server-arm64-orangepi-5.img.xz`)
vì tưởng "sạch hơn, ổn định hơn" — **kernel của nó không có module
`rknpu` ở đâu cả** (`modprobe rknpu` báo "Module not found", không phải
"chưa load"). Project đó đã bị **archive (ngừng phát triển)**, và yêu
cầu thêm RKNPU đã bị từ chối chính thức ("not planned",
[issue #1093](https://github.com/Joshua-Riek/ubuntu-rockchip/issues/1093)).
Không có NPU = không dùng được tính năng cốt lõi của cả đề tài này.

Tải đúng image chính thức tại mục "Ubuntu Image" trên trang Orange Pi:
https://drive.google.com/drive/folders/1i5zQOg1GIA4_VNGikFl2nPM0Y2MBw2M0
— chọn file **`Orangepi5_x.x.x_ubuntu_jammy_server_linux6.1.99.7z`**
(chữ `jammy` + `server`, không phải `desktop`; số `6.1.99` là dòng
kernel BSP của Rockchip, có driver RKNPU sẵn — verify thật: kernel này
load `rknpu 0.9.8`, bản mới nhất tại thời điểm viết, không cần build lại
kernel). File nén `.7z` — giải nén bằng `7z` hoặc `pip install py7zr`
nếu máy không có `p7zip`. **Luôn so khớp SHA256** với file `.sha` đi kèm
trước khi ghi thẻ.

```bash
# trên máy laptop, xác định đúng thiết bị trước khi dd — dd sai đích sẽ
# xoá ổ cứng máy bạn, không phải thẻ SD
lsblk
sha256sum Orangepi5_*_jammy_server_linux6.1.99.img   # so với file .sha đi kèm
sudo dd if=Orangepi5_*_jammy_server_linux6.1.99.img of=/dev/sdX bs=4M status=progress conv=fsync
sync
```

**Xác minh:** cắm thẻ vào board, cấp nguồn, đèn LED trạng thái nhấp
nháy trong ~30-60s (boot lần đầu resize partition).

## 2. Boot lần đầu + mạng

```bash
# tìm IP board trên mạng LAN (từ laptop, cùng subnet) -- board mới boot
# lần đầu tự resize partition, đợi ~60-90s trước khi quét
nmap -sn 192.168.1.0/24
# hoặc lọc thiết bị có cổng SSH mở nếu không biết dải IP nào là board mới
nmap -p 22 --open 192.168.1.0/24
ssh orangepi@<ip-board>
```

**User/pass mặc định xác minh thật, 2026-09-28: `orangepi` / `orangepi`**
— image chính thức KHÔNG bắt đổi mật khẩu ngay như một số image
cloud-init khác, đăng nhập được thẳng lần đầu.

Việc đầu tiên khi vào được máy — đổi mật khẩu mặc định:

```bash
passwd
sudo timedatectl set-timezone Asia/Ho_Chi_Minh
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

**Xác minh sau reboot:**
```bash
uname -m        # phải ra: aarch64  (không phải armv7l)
cat /etc/os-release | grep VERSION
```

## 3. Xác minh phần cứng đúng như kỳ vọng — TRƯỚC KHI cài bất cứ gì

Đây là bước quan trọng nhất trong toàn bộ tài liệu này. Nó biến "tôi
nghĩ đây là RK3588S" thành một sự thật đã đo, và nó chạy được ngay cả
khi chưa cài Python package nào của dự án.

```bash
cat /proc/cpuinfo | grep -E "CPU part|processor" | head -20
cat /proc/device-tree/model; echo
free -h
cat /sys/class/thermal/thermal_zone0/temp
ls /dev/dri /dev/mali* /dev/rknpu* 2>&1
dmesg | grep -i rknpu | head -5
```

**Kỳ vọng:** 8 dòng `CPU part` với 2 giá trị khác nhau (big.LITTLE —
`0xd0b` Cortex-A76 và `0xd05` Cortex-A55), model chứa "RK3588", RAM
~7.x GiB khả dụng (8GB danh nghĩa), nhiệt độ idle dưới 50°C.

**Kiểm tra NPU, xác minh thật 2026-09-28 — ĐỪNG chỉ tìm `/dev/rknpu*`.**
Trên kernel BSP hiện tại (`6.1.99-rockchip-rk3588`), RKNPU đăng ký như
một **DRM device** (`[drm] Initialized rknpu 0.9.8 ... on minor 1`
trong `dmesg`), không tạo ra node `/dev/rknpu*` kiểu cũ — nó xuất hiện
dưới `/dev/dri/renderD1xx` (thường là node render thứ hai, sau Mali
GPU). `ls /dev/rknpu*` báo "No such file" là **bình thường**, không phải
lỗi. Cách xác minh đúng: `dmesg | grep -i rknpu` phải có dòng
`[drm] Initialized rknpu <version>` — đó mới là bằng chứng driver đã
load, không phải sự tồn tại của một tên file cụ thể.

Chỉ khi cả `dmesg` không có dòng `rknpu` nào (không phải chỉ thiếu
`/dev/rknpu*`) mới là driver thật sự chưa load — lúc đó mới cần lo, và
việc đầu tiên cần kiểm tra là image có đúng dùng kernel BSP của Rockchip
hay không (xem cảnh báo ở §1 về image cộng đồng thiếu hẳn driver này).

## 4. Cài gói hệ thống

```bash
sudo apt install -y \
    python3-pip python3-venv python3-opencv python3-numpy python3-yaml \
    python3-psutil python3-libgpiod gpiod git build-essential
```

`python3-libgpiod` + `gpiod` là bắt buộc — `src/labsafe/io/alerts.py`
dùng libgpiod cho buzzer/LED, không dùng `RPi.GPIO` (thư viện đó không
tồn tại trên Orange Pi và số chân BCM của Pi vô nghĩa ở đây).

## 5. Lấy code lên board

```bash
git clone <url-repo-của-bạn> ~/AI_Challenge
cd ~/AI_Challenge
pip install --break-system-packages -r requirements/edge.txt
```

Nếu `rknn-toolkit-lite2` trong `requirements/edge.txt` cài lỗi vì
`librknnrt.so` chưa khớp phiên bản — xem `docs/KNOWN_ISSUES.md` mục về hai
gói `rknn-toolkit2`/`rknn-toolkit-lite2` khác nhau, và
`docs/rknn-pipeline.md` §0.

## 6. Ba lệnh xác nhận hệ thống sẵn sàng — chạy theo đúng thứ tự này

```bash
cd ~/AI_Challenge
export PYTHONPATH=src

# (a) CPU này có gì — big.LITTLE, NPU, cờ ISA
python3 -m labsafe.optimize.arm_probe
```

**Kỳ vọng:** in ra 2 cluster (`Cortex-A76 x4 (big)`, `Cortex-A55 x4
(LITTLE)`), cả hai đều có `asimddp` (INT8 dot-product — khác hẳn Pi 4,
xem `docs/optimization.md` §1), và một dòng NPU accelerator được phát
hiện.

```bash
# (b) topology + lệnh taskset thật để ghim vào cụm big
python3 -m labsafe.optimize.affinity
```

**Kỳ vọng:** in ra `taskset -c 4,5,6,7` (hoặc mask tương đương thật của
board bạn — có thể không phải đúng 4,5,6,7, số CPU id phụ thuộc cách
kernel enumerate).

```bash
# (c) máy này compute-bound hay memory-bound
python3 -m labsafe.optimize.roofline
```

**Kỳ vọng:** in ra băng thông bộ nhớ đo được (GB/s) và GFLOP/s FP32 đo
được — hai con số thật của chính board bạn, không phải số tra bảng.

Nếu cả ba lệnh chạy xong không lỗi và in ra như trên, **board đã sẵn
sàng cho Phase 2** (dataset). Gửi output của cả 3 lệnh lại — đó là dữ
liệu O0 quyết định phần còn lại của ma trận thí nghiệm.

## 7. Kiểm tra buzzer/LED (nếu đã đấu breadboard)

**⚠ Nếu chưa xác nhận đã đấu dây buzzer/LED và chưa tra offset thật bằng
`gpioinfo`:** `configs/pipeline.yaml` mặc định `alerts.gpio: auto` với số
chân BCM của **Raspberry Pi** (`buzzer_pin: 17, led_r_pin: 27,
led_g_pin: 22`). Nếu `python3-libgpiod` đã cài (mục 4), `AlertSink` sẽ
**request và drive thật** line offset 17/27/22 trên `gpiochip0` của Orange
Pi — có thể trúng nhầm chân không liên quan gì tới buzzer/LED. Trong
trường hợp này:
- chỉ chạy các lệnh **read-only** bên dưới (`gpiodetect`, `gpioinfo`),
  KHÔNG chạy đoạn test `AlertSink(... gpio: 'on' ...)`;
- MỌI lệnh `python3 -m labsafe.app` phải kèm `--gpio off` cho tới khi có
  offset thật đã xác nhận.

```bash
gpioinfo   # liệt kê TẤT CẢ line offset thật của board — KHÔNG phải số BCM của Pi
```

Tìm 3 offset tương ứng chân bạn đã nối buzzer/LED đỏ/LED xanh, điền vào
`configs/pipeline.yaml` mục `alerts` (`gpio_chip`, `buzzer_pin`,
`led_r_pin`, `led_g_pin`). Copy nguyên số chân từ hướng dẫn Pi 4 sang
đây sẽ **drive nhầm chân, không báo lỗi** — trông y hệt buzzer chết.

```bash
python3 -c "
import sys; sys.path.insert(0,'src')
from labsafe.io.alerts import AlertSink
a = AlertSink({'gpio': 'on', 'gpio_chip': 'gpiochip0',
               'buzzer_pin': <offset>, 'led_r_pin': <offset>, 'led_g_pin': <offset>})
a.alert('critical', 'test'); import time; time.sleep(2); a.clear(); a.close()
"
```

**Kỳ vọng:** buzzer kêu, LED đỏ sáng ~2s rồi tắt, LED xanh sáng lại.
`gpio: 'on'` (không phải `'auto'`) ép báo lỗi ngay nếu libgpiod không
dùng được, thay vì âm thầm rơi về chế độ chỉ-log.

## 8. Chạy webcam sống đầu tiên (chưa cần model thật)

```bash
python3 -m labsafe.app --source 0 --no-display --gpio off
```

Backend mặc định là `stub` (detector giả) — mục đích là xác nhận toàn
bộ pipeline camera → tracker → zone → fall → debounce → alert chạy
được trên chính phần cứng thật, trước khi model thật vào cuộc. Ctrl+C
sau vài giây, xem JSON thống kê in ra cuối (nếu không thấy JSON in ra sau
Ctrl+C, đó là bug — `main()` phải bắt `KeyboardInterrupt` quanh vòng lặp
live trước khi in summary).

`--gpio off` ở đây vì `configs/pipeline.yaml` mặc định `gpio: auto` với số
chân BCM của Pi (xem cảnh báo ở mục 7) — không liên quan gì tới việc dùng
`stub` hay model thật.

## 9. Chạy RKNN live đầu tiên

### Cách chạy hằng ngày (tự lưu đầy đủ bằng chứng)

Trên Orange Pi, dùng đúng một lệnh:

```bash
cd ~/AI_Challenge
make board-live
```

Nhấn `Ctrl+C` để kết thúc sạch. Mỗi lần chạy tự tạo
`runs/board-live-YYYYmmdd-HHMMSS/` gồm:

- `annotated.mp4`: video đã vẽ bbox, vùng cấm, cảnh báo và FPS;
- `run.log`: log khởi động/backend/lỗi;
- `summary.json`: thống kê pipeline, camera và bộ ghi video;
- `command.json`: chính xác config và tham số đã chạy;
- `frames/*.jpg` + `frames/*.json`: ảnh mẫu và detection/event tương ứng.

Video được encode ở thread riêng. Nếu thẻ nhớ không ghi kịp, hệ thống bỏ
frame video và ghi số lượng vào `video.json`, thay vì làm chậm inference.
Vì việc vẽ overlay vẫn có một ít chi phí, phép benchmark dùng cho bảng NCKH
không tự bật video; hãy chạy một evidence run riêng ngay trước/sau benchmark.

Trên laptop, kéo toàn bộ phiên mới nhất về bằng:

```bash
cd /home/an-lee/AI_Challenge
make fetch-board
```

Kết quả nằm tại `runs/board/board-live-.../`. Mở file `annotated.mp4` bằng
trình phát video, không gõ đường dẫn MP4 trực tiếp như một lệnh shell.

Sau khi có model đã qua cổng parity (`scripts/export/validate_conversion.py`,
`docs/rknn-pipeline.md`) và `runs/export/yolo26n_fp16_640.rknn` tồn tại.

**Model checkpoint hiện có là COCO-80, chưa fine-tune cho taxonomy của
project (`person` + PPE)** — chạy thẳng với `configs/classes.yaml` (chỉ có
`[person]`) sẽ crash lúc warmup vì model trả về 80 kênh lớp trong khi
config chỉ khai 1 (`docs/KNOWN_ISSUES.md` #20). Dùng
`configs/classes_coco80.yaml` + config live riêng (`N1L`), có
`stage_a.keep_classes: [person]` để router chỉ theo dõi người, không phải
"bất kỳ vật gì trong 80 lớp COCO".

Smoke 60 giây, có lưu ảnh để xem detection bằng mắt:

```bash
PYTHONPATH=src python3 -m labsafe.app --bench --exp N1L_smoke \
    --config configs/exp/N1L_live_rknn_fp16_640.yaml \
    --classes configs/classes_coco80.yaml \
    --source 0 --duration 60 --warmup 5 --no-display --gpio off \
    --save-every 3 --save-max 20
```

Đo sustained 15 phút thật (số liệu chính, không phải smoke test):

```bash
PYTHONPATH=src python3 -m labsafe.app --bench --exp N1L_sustained \
    --config configs/exp/N1L_live_rknn_fp16_640.yaml \
    --classes configs/classes_coco80.yaml \
    --source 0 --duration 900 --warmup 30 --no-display --gpio off
```

Đối chứng CPU (ONNX FP32) — bắt buộc để biết NPU có thật sự nhanh hơn,
không giả định:

```bash
PYTHONPATH=src python3 -m labsafe.app --bench --exp N0L_cpu \
    --config configs/exp/N0L_live_cpu_onnx_640.yaml \
    --classes configs/classes_coco80.yaml \
    --source 0 --duration 120 --warmup 10 --no-display --gpio off
```

Đo nhiệt độ/tần số/NPU load song song (chạy trước app ~15s để có baseline
idle, cần root):

```bash
sudo bash scripts/board/board_sampler.sh runs/samples/N1L_sustained.csv 990 1
```

Xem kết quả: `metrics.json` có khối `router.stage_a_breakdown_ms`
(pre/infer/post) và `camera` (resolution, capture_fps, duplicate_ratio);
`frames/` chứa ảnh đã lưu để xem detection bằng mắt (`viz.draw()` giờ in
kèm score trên nhãn track).

---

## Việc còn lại sau khi board xanh hết mục trên

Đây là lúc bắt đầu Phase 2 thật:

1. Quay dữ liệu PPE bằng Logitech C270 trong phòng lab thật (`docs/research-plan.md`
   §Phase 2, `data/README.md`).
2. Convert model YOLO26 sang RKNN — làm trên **laptop x86**, không phải
   trên board (`docs/rknn-pipeline.md`).
3. Chạy `bash scripts/bench/bench_matrix_npu.sh` trên board sau khi có model
   đã qua cổng parity.

## Nếu một mục xác minh KHÔNG ra kết quả như mô tả

Đừng đoán và đi tiếp. Việc phổ biến nhất bị bỏ qua ở bước này khiến
mọi số liệu đo sau đó vô giá trị mà không ai biết. Ghi lại chính xác
lệnh nào, output thật là gì, và dừng ở đó — gửi lại để chẩn đoán trước
khi làm gì tiếp theo.
