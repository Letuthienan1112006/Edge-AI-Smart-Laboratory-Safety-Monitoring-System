#!/usr/bin/env bash
# Runs ON THE BOARD inside the desktop session (started by ~/.config/autostart/labsafe-kiosk.desktop): shows the monitoring page
# full screen on the HDMI monitor / projector. Waits until the live service answers (the page would otherwise show a browser
# error and never retry), keeps the screen awake, and restarts the browser if it closes or crashes.
URL="${LABSAFE_KIOSK_URL:-http://127.0.0.1:8000/#present}"
LOG="$HOME/AI_Challenge/runs/kiosk.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "$(date '+%F %T') kiosk start, url $URL"
# screen must never blank during a demo
xset s off 2>/dev/null; xset -dpms 2>/dev/null; xset s noblank 2>/dev/null
command -v unclutter >/dev/null && (unclutter -idle 1 -root &)
BROWSER=""
for b in chromium chromium-browser google-chrome firefox firefox-esr; do command -v "$b" >/dev/null && { BROWSER="$b"; break; }; done
[ -z "$BROWSER" ] && { echo "no browser installed (chromium / firefox): sudo apt install chromium-browser"; exit 1; }
while true; do
  until curl -s -m 2 -o /dev/null http://127.0.0.1:8000/status; do sleep 2; done
  echo "$(date '+%F %T') service up, starting $BROWSER"
  case "$BROWSER" in
    firefox*) "$BROWSER" --kiosk --private-window "$URL" ;;
    *) "$BROWSER" --kiosk --incognito --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
         --autoplay-policy=no-user-gesture-required --check-for-update-interval=31536000 --no-first-run "$URL" ;;
  esac
  echo "$(date '+%F %T') browser exited ($?), restarting in 3 s"; sleep 3
done
