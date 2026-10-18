#!/usr/bin/env bash
# Can other machines on the network reach the image server? Run it on the server Mac.
#
# A request to this Mac's own network address passes through the macOS firewall the
# same way one from another machine does, so this catches a blocked Python. (The
# firewall's --getappblocked reports "permitted" for programs it has never been
# asked about, even while it holds their connections.) Prints no secrets.
set -uo pipefail
cd "$(dirname "$0")/.."
env_get() { sed -n "s/^$1=//p" .env 2>/dev/null | tail -n 1 | tr -d "\"'"; }

# The address of the interface the default route uses (Wi-Fi, Ethernet, USB or
# Thunderbolt adapters), not a VPN tunnel; en0/en1 as a fallback.
lan_ip() {
  local iface
  iface=$(route -n get default 2>/dev/null | awk '/interface:/ {print $2}')
  case "$iface" in utun* | ipsec* | ppp* | "") iface="" ;; esac
  { [[ -n "$iface" ]] && ipconfig getifaddr "$iface"; } 2>/dev/null ||
    ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true
}

port=$(env_get IMAGE_SERVER_PORT)
port=${port:-7865}
if [[ "$(env_get IMAGE_SERVER_HOST)" != "0.0.0.0" ]]; then
  echo "LAN access is off (loopback only); scripts/enable_lan.sh turns it on"
  exit 0
fi
if ! curl -sf -o /dev/null --max-time 4 "http://127.0.0.1:$port/health"; then
  echo "the image server isn't running (see ~/Library/Logs/teacher-image-harness/image-server.log," \
    "or start it with scripts/install_image_server.sh)"
  exit 1
fi
listeners=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR > 1 {print $9}' | sort -u)
if [[ -n "$listeners" && -z "$(grep -v '^127\.0\.0\.1:' <<<"$listeners")" ]]; then
  echo "the server still listens on this Mac only: restart it to apply LAN mode (scripts/install_image_server.sh)"
  exit 1
fi
ip=$(lan_ip)
if [[ -z "$ip" ]]; then
  echo "this Mac has no IPv4 address on its active network connection"
  exit 1
fi

probe() { # host-header -> "<http code> <tcp connect seconds>"
  curl -s -o /dev/null -w '%{http_code} %{time_connect}' --max-time 4 -H "Host: $1" "http://$ip:$port/health"
}
read -r code connected <<<"$(probe "$ip:$port")"
case "$code" in
  200) ;;
  400)
    echo "the server doesn't accept the address $ip (it may have changed): run scripts/enable_lan.sh," \
      "then scripts/install_image_server.sh"
    exit 1 ;;
  000)
    if [[ "${connected:-0}" == 0 || "${connected:-0}" == 0.000000 ]]; then
      echo "nothing answers on http://$ip:$port (the server may not be listening on this network yet):" \
        "restart it with scripts/install_image_server.sh"
      exit 1
    fi
    fw=/usr/libexec/ApplicationFirewall/socketfilterfw
    python_bin=$(cd image-server && uv run --frozen python -c \
      'import os, sys; print(os.path.realpath(sys.executable))' 2>/dev/null || echo "<server python>")
    echo "blocked: http://$ip:$port accepts the connection but never answers, so the macOS firewall is"
    echo "holding connections to the server's Python. Click Allow if macOS asks about python3.12, or run:"
    echo "  sudo $fw --add \"$python_bin\""
    echo "  sudo $fw --unblockapp \"$python_bin\""
    echo "Then run this check again. (The approval is per Python build, so redo it after uv upgrades Python.)"
    exit 1 ;;
  *)
    echo "unexpected answer (HTTP $code) from http://$ip:$port/health"
    exit 1 ;;
esac

# The name clients are told to use: does the server still accept it? (Tested through the
# IP, so it doesn't depend on name lookup.)
name="$(scutil --get LocalHostName 2>/dev/null | tr '[:upper:]' '[:lower:]').local"
read -r code _ <<<"$(probe "$name:$port")"
if [[ "$code" != 200 ]]; then
  echo "reachable at http://$ip:$port, but the server doesn't accept the Mac's name $name (it may have" \
    "been renamed): run scripts/enable_lan.sh, then scripts/install_image_server.sh, and update the" \
    "address on the computers that use it"
  exit 1
fi
echo "reachable on the network at http://$name:$port and http://$ip:$port"
