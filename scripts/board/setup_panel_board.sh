#!/usr/bin/env bash
# Laptop side, ONCE: prepare the board for the status LEDs + OLED. Asks for the board's sudo password (ssh -t).
#   1. udev rule: user orangepi may open /dev/gpiochip* and /dev/i2c-* (they are root-only by default)
#   2. device-tree overlay i2c5-m3: turns on the I2C5 bus on header pins 3 (SDA) and 5 (SCL); /boot/orangepiEnv.txt is backed up first
# The overlay takes effect after a reboot: `sudo reboot` is printed at the end, not run. Safe to run twice.
set -euo pipefail
HOST="${HOST:-orangepi-lan}"
cd "$(dirname "$0")/../.."
rsync -aq deploy/99-labsafe-io.rules "$HOST":AI_Challenge/deploy/
ssh -t "$HOST" 'set -e
  test -f /boot/dtb/rockchip/overlay/rk3588-i2c5-m3.dtbo || { echo "overlay rk3588-i2c5-m3.dtbo not found on this image"; exit 1; }
  sudo install -m 644 ~/AI_Challenge/deploy/99-labsafe-io.rules /etc/udev/rules.d/99-labsafe-io.rules
  sudo udevadm control --reload
  sudo udevadm trigger --subsystem-match=gpio --subsystem-match=i2c-dev
  if grep -qE "^overlays=.*\bi2c5-m3\b" /boot/orangepiEnv.txt; then echo "overlay i2c5-m3 already enabled"
  else
    sudo cp -n /boot/orangepiEnv.txt /boot/orangepiEnv.txt.bak-labsafe
    if grep -q "^overlays=" /boot/orangepiEnv.txt; then sudo sed -i "/^overlays=/ s/\$/ i2c5-m3/" /boot/orangepiEnv.txt
    else echo "overlays=i2c5-m3" | sudo tee -a /boot/orangepiEnv.txt >/dev/null; fi
    echo "overlay i2c5-m3 added (backup: /boot/orangepiEnv.txt.bak-labsafe)"
  fi
  grep -E "^overlay" /boot/orangepiEnv.txt
  ls -l /dev/gpiochip0 /dev/gpiochip4
  echo; echo "Done. Reboot the board once so the I2C5 bus appears:   sudo reboot"'
