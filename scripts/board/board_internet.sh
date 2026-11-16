#!/usr/bin/env bash
# Laptop side: give the BOARD internet through the LAN cable it already has to this laptop (the laptop's own Wi-Fi / phone hotspot is the way out).
# No cable to a phone, no USB Wi-Fi. Needed only for the Zalo messages (make zalo-push, alarms to the phone).
#   bash scripts/board/board_internet.sh on       # board gets 192.168.2.2, default route + DNS through the laptop (asks the board's sudo password)
#   bash scripts/board/board_internet.sh off      # undo (back to the lab-network settings the board has: 192.168.196.199 via 192.168.196.1)
#   bash scripts/board/board_internet.sh status   # can the board reach Zalo?
# Requires the laptop's cable connection in NetworkManager "Shared to other computers" mode (it is: enp8s0 = 192.168.2.1/24) and the laptop online.
# Changes are runtime only: a board reboot forgets them (run `on` again). The SSH link (IPv6 link-local on enp8s0) is not touched.
set -uo pipefail
HOST="${HOST:-orangepi-lan}"
LAP=192.168.2.1; BOARD=192.168.2.2
test_net() {
  ssh -o ConnectTimeout=8 "$HOST" 'timeout 8 curl -s -m 6 -o /dev/null -w "board -> Zalo API: HTTP %{http_code}\n" https://bot-api.zaloplatforms.com || echo "board -> Zalo API: NO internet"'
}
case "${1:-status}" in
  on)
    nmcli -g ipv4.method con show "$(nmcli -g NAME,DEVICE con show --active | awk -F: '$2=="enp8s0"{print $1}')" | grep -q shared \
      || { echo "the laptop's enp8s0 is not in 'shared' mode: nmcli con modify <connection> ipv4.method shared; nmcli con up <connection>"; exit 1; }
    curl -s -m 5 -o /dev/null https://bot-api.zaloplatforms.com || { echo "the laptop itself has no internet: connect it to Wi-Fi / the phone hotspot first"; exit 1; }
    ssh -t "$HOST" "sudo ip addr replace $BOARD/24 dev eth0 &&
      sudo ip route replace default via $LAP dev eth0 metric 50 &&
      (grep -q '^nameserver $LAP\$' /etc/resolv.conf || sudo sed -i '1i nameserver $LAP' /etc/resolv.conf) && echo 'board: internet through the laptop ON'"
    sleep 2; test_net;;
  off)
    ssh -t "$HOST" "sudo ip route del default via $LAP dev eth0 2>/dev/null; sudo ip addr del $BOARD/24 dev eth0 2>/dev/null;
      sudo sed -i '/^nameserver $LAP\$/d' /etc/resolv.conf; echo 'board: back to its own network settings'; ip route | head -3";;
  status) test_net; ssh "$HOST" 'ip -4 -br addr show eth0';;
  *) echo "usage: $0 [on|off|status]"; exit 2;;
esac
