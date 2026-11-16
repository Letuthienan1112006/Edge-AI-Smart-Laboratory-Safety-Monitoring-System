#!/usr/bin/env bash
# Pull one complete evidence bundle from Orange Pi to this laptop.
set -euo pipefail

board="${1:-${LABSAFE_BOARD:-orangepi@fe80::9d0b:b98c:abc1:2b9a%enp8s0}}"
remote_root="${2:-~/AI_Challenge/runs}"
local_root="${3:-runs/board}"

ssh_opts=()
rsync_board="$board"
if [[ "$board" == *:* ]]; then
  ssh_opts=(-6)
  # rsync's host:path parser needs brackets around a literal IPv6 address;
  # plain ssh needs the same address without brackets.
  user_part="${board%%@*}"
  host_part="${board#*@}"
  rsync_board="${user_part}@[${host_part}]"
fi

latest="$(ssh "${ssh_opts[@]}" "$board" "find $remote_root -mindepth 1 -maxdepth 1 -type d -name 'board-live-*' -printf '%f\n' | sort | tail -1")"
if [[ -z "$latest" ]]; then
  echo "No board-live-* run found on ${board}:${remote_root}" >&2
  exit 1
fi

mkdir -p "$local_root"
rsync -av --partial -e "ssh ${ssh_opts[*]}" \
  "$rsync_board:$remote_root/$latest/" "$local_root/$latest/"
echo
echo "Downloaded: $local_root/$latest"
echo "Video:     $local_root/$latest/annotated.mp4"
echo "Log:       $local_root/$latest/run.log"
echo "Summary:   $local_root/$latest/summary.json"
