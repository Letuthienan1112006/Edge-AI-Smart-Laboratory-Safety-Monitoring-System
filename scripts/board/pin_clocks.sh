#!/usr/bin/env bash
# Board side, needs root:  sudo bash scripts/board/pin_clocks.sh pin|restore|show
# pin     -> CPU, DDR (dmc), NPU and GPU governors to `performance` (clocks stay at max: removes DVFS from a measurement)
# restore -> the governors the board boots with (ondemand / dmc_ondemand / rknpu_ondemand / simple_ondemand)
set -euo pipefail
case "${1:-show}" in
  pin)     cpu=performance; dmc=performance; npu=performance; gpu=performance ;;
  restore) cpu=ondemand;    dmc=dmc_ondemand; npu=rknpu_ondemand; gpu=simple_ondemand ;;
  show)    ;;
  *) echo "usage: $0 pin|restore|show"; exit 2 ;;
esac
if [ "${1:-show}" != show ]; then
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$cpu" > "$g"; done
  echo "$dmc" > /sys/class/devfreq/dmc/governor
  echo "$npu" > /sys/class/devfreq/fdab0000.npu/governor
  echo "$gpu" > /sys/class/devfreq/fb000000.gpu/governor
fi
echo "cpu0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz | cpu4 $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor) $(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq) kHz"
for d in dmc fdab0000.npu fb000000.gpu; do echo "$d $(cat /sys/class/devfreq/$d/governor) $(cat /sys/class/devfreq/$d/cur_freq) Hz"; done
