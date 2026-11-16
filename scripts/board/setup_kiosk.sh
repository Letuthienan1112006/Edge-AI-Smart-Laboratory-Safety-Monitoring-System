#!/usr/bin/env bash
# Laptop side, ONCE: make the board show the monitoring page full screen on an HDMI monitor / projector by itself after power-on.
#   bash scripts/board/setup_kiosk.sh          (make kiosk-setup)   check + install
#   bash scripts/board/setup_kiosk.sh off      (make kiosk-off)     remove the autostart entry (desktop and autologin left as they are)
# Needs: the board's desktop image (a display manager + a browser). The script checks both and says what is missing; installing a
# browser needs internet on the board (make board-internet MODE=on). Asks the board's sudo password once (ssh -t) for the autologin.
set -uo pipefail
HOST="${HOST:-orangepi-lan}"
cd "$(dirname "$0")/../.."
ssh -o ConnectTimeout=8 "$HOST" true || { echo "Cannot reach $HOST: LAN cable + power, wait a minute after power-on."; exit 1; }
if [ "${1:-}" = off ]; then
  ssh "$HOST" 'rm -f ~/.config/autostart/labsafe-kiosk.desktop; pkill -f labsafe-kiosk.sh; echo "kiosk autostart removed (takes effect at next boot)"'
  exit 0
fi
rsync -a deploy/labsafe-kiosk.sh deploy/labsafe-kiosk.desktop "$HOST":AI_Challenge/deploy/
echo "== what the board has"
ssh "$HOST" 'echo "  default target : $(systemctl get-default)"
  dm=$(basename "$(readlink -f /etc/systemd/system/display-manager.service 2>/dev/null)" .service); echo "  display manager: ${dm:-none}"
  for b in chromium chromium-browser google-chrome firefox firefox-esr; do command -v $b >/dev/null && { echo "  browser        : $b"; break; }; done
  for c in /sys/class/drm/card*-HDMI-A-*/status; do [ -f "$c" ] && echo "  $(basename $(dirname $c)): $(cat $c)"; done'
info=$(ssh "$HOST" 'dm=$(basename "$(readlink -f /etc/systemd/system/display-manager.service 2>/dev/null)" .service); b=""; for x in chromium chromium-browser google-chrome firefox firefox-esr; do command -v $x >/dev/null && { b=$x; break; }; done; echo "$dm|$b"')
dm=${info%%|*}; br=${info##*|}
if [ -z "$dm" ] || [ "$dm" = "." ]; then
  echo "!! no desktop on this board image (server image): the page cannot be shown on HDMI without one."
  echo "   Options: flash the Orange Pi 5 *desktop* image, or tell Claude to build a framebuffer viewer (no desktop needed)."
  exit 1
fi
if [ -z "$br" ]; then
  echo "!! no browser on the board. Give it internet (make board-internet MODE=on), then:"
  echo "     ssh -t $HOST 'sudo apt update && sudo apt install -y chromium-browser unclutter'"
  echo "   and run make kiosk-setup again."
  exit 1
fi
echo "== installing: autostart entry + desktop autologin for user orangepi ($dm)"
ssh "$HOST" 'mkdir -p ~/.config/autostart && cp ~/AI_Challenge/deploy/labsafe-kiosk.desktop ~/.config/autostart/ && chmod +x ~/AI_Challenge/deploy/labsafe-kiosk.sh && echo "  autostart entry installed"'
case "$dm" in
  lightdm) ssh -t "$HOST" 'sudo mkdir -p /etc/lightdm/lightdm.conf.d && printf "[Seat:*]\nautologin-user=orangepi\nautologin-user-timeout=0\n" | sudo tee /etc/lightdm/lightdm.conf.d/30-labsafe-autologin.conf >/dev/null && echo "  lightdm autologin on"';;
  gdm|gdm3) ssh -t "$HOST" 'f=/etc/gdm3/custom.conf; sudo cp -n $f $f.labsafe.bak; sudo sed -i "/^\[daemon\]/a AutomaticLoginEnable=true\nAutomaticLogin=orangepi" $f && echo "  gdm autologin on (backup $f.labsafe.bak)"';;
  *) echo "  display manager $dm: set autologin for user orangepi by hand (the autostart entry is installed)";;
esac
echo
echo "DONE. Plug an HDMI monitor/projector into the board BEFORE power-on, then power on: after about 1-2 minutes the page opens full screen."
echo "Sound: plug speakers/headphones into the board or use the monitor's speakers. Logs: ~/AI_Challenge/runs/kiosk.log on the board."
