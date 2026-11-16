# Sourced by the laptop-side board scripts. Loads BOARD_SUDO_PASS from the git-ignored secrets/board.env (one line:
#   BOARD_SUDO_PASS=<the board's sudo password>
# ). The password is never stored in this repository. Without it the `sudo` steps (stop the capture service, set the
# board clock) fail quietly; run them by hand with `ssh -t <host> 'sudo ...'`.
_board_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# a malformed file (e.g. an unquoted placeholder) must not abort the calling script (set -e): ignore it
if [ -f "$_board_root/secrets/board.env" ]; then . "$_board_root/secrets/board.env" 2>/dev/null || BOARD_SUDO_PASS=""; fi
: "${BOARD_SUDO_PASS:=}"
export BOARD_SUDO_PASS
