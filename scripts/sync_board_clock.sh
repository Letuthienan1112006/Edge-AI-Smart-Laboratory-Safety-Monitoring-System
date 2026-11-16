#!/usr/bin/env bash
# Laptop side: set the board clock from the laptop clock. The Orange Pi has no RTC battery and no internet here, so
# after every power loss it boots with a stale date (seen: 10 days behind) and every timestamp it writes is wrong
# (capture sessions, violation photos, logs). Called by the other board scripts; harmless to run any time.
#   bash scripts/sync_board_clock.sh [host]
HOST="${1:-orangepi-lan}"
. "$(dirname "$0")/_board_env.sh"
before=$(ssh -o ConnectTimeout=8 "$HOST" 'date +%s' 2>/dev/null) || { echo "clock sync: $HOST unreachable"; exit 0; }
now=$(date +%s)
if (( now - before > 5 || before - now > 5 )); then
  ssh "$HOST" "echo '$BOARD_SUDO_PASS' | sudo -S date -s @$(date +%s) >/dev/null 2>&1" \
    && echo "board clock corrected by $(( now - before )) s -> $(ssh "$HOST" date)" \
    || echo "clock sync: could not set the board clock (sudo)"
fi
