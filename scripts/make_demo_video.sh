#!/usr/bin/env bash
# Turn the raw side-by-side recording (screen 1920x1080 | phone view of the device 1920x1080, 3840x1080) into ONE clean 1920x1080 demo video:
# the dashboard (browser chrome and bookmarks cropped out), the device (LEDs + OLED), the Zalo message, a title, panel labels and timed captions.
#   bash scripts/make_demo_video.sh [input.mp4] [output.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-1024(1).mp4}"; OUT="${2:-runs/demo_video/demo_lab_professional.mp4}"
F=src/labsafe/assets/DejaVuSans-Bold.ttf
BG=0x0F2238; PANEL=0x16324F; TEAL=0x0E7C86
mkdir -p "$(dirname "$OUT")"
# geometry (canvas 1920x1080): header 0-96 | dashboard (30,130) 1040x827 | device (1100,130) 790x527 | Zalo (1100,700) 470x320 + caption
FC="
color=c=${BG}:s=1920x1080:r=30:d=33[bg];
[0:v]split=3[a][b][c];
[a]crop=1170:850:735:150,scale=1040:-2[dash];
[b]crop=1080:720:2360:100,scale=790:527[hw];
[c]crop=545:262:185:668[zalo];
[bg][dash]overlay=30:150[t1];
[t1][hw]overlay=1100:150[t2];
[t2][zalo]overlay=1100:720[t3];
[t3]
drawbox=x=0:y=0:w=1920:h=100:color=${PANEL}:t=fill,
drawtext=fontfile=${F}:text='Edge AI Smart Laboratory Safety Monitoring System':fontcolor=white:fontsize=40:x=40:y=18,
drawtext=fontfile=${F}:text='Demo · người không mặc áo lab bước vào vùng nguy hiểm':fontcolor=0xBFD8E6:fontsize=24:x=40:y=64,
drawtext=fontfile=${F}:text='1  Trang web giám sát':fontcolor=white:fontsize=26:x=30:y=112:box=1:boxcolor=${TEAL}:boxborderw=6,
drawtext=fontfile=${F}:text='2  Thiết bị · đèn LED và màn OLED':fontcolor=white:fontsize=26:x=1100:y=112:box=1:boxcolor=${TEAL}:boxborderw=6,
drawtext=fontfile=${F}:text='3  Tin nhắn Zalo':fontcolor=white:fontsize=26:x=1100:y=682:box=1:boxcolor=${TEAL}:boxborderw=6,
drawtext=fontfile=${F}:text='tin gửi tới':fontcolor=0xBFD8E6:fontsize=24:x=1670:y=800,
drawtext=fontfile=${F}:text='điện thoại':fontcolor=0xBFD8E6:fontsize=24:x=1670:y=832,
drawtext=fontfile=${F}:text='người phụ trách':fontcolor=0xBFD8E6:fontsize=24:x=1670:y=864,
drawbox=x=0:y=990:w=1920:h=90:color=${PANEL}:t=fill,
drawtext=fontfile=${F}:text='CHÚ Ý · vừa có cảnh báo, đã dừng':fontcolor=0xFFD978:fontsize=34:x=40:y=1017:enable='lt(t,3.1)',
drawtext=fontfile=${F}:text='BÁO ĐỘNG · người không mặc áo lab trong vùng nguy hiểm  →  web đỏ, LED, OLED, Zalo':fontcolor=0xFFB4AE:fontsize=34:x=40:y=1017:enable='gte(t,3.1)',
fade=t=in:st=0:d=0.6,fade=t=out:st=32.2:d=0.8
[v]"
ffmpeg -v error -y -i "$IN" -filter_complex "$FC" -map "[v]" -t 31.8 -an -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p -movflags +faststart "$OUT" 2>&1 | head -5 || true
