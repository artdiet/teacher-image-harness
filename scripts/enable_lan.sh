#!/usr/bin/env bash
# Let other machines on your home network reach the image server (LAN only:
# no tunnels or port forwarding). Writes only to the gitignored .env.
#
#   scripts/enable_lan.sh        # listen on all interfaces; allow this Mac's LAN names
#   scripts/enable_lan.sh off    # back to loopback-only
#
# Then restart the server: scripts/install_image_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] || { echo "missing .env; run scripts/setup_env.sh first" >&2; exit 1; }

mode=$(tr '[:upper:]' '[:lower:]' <<<"${1:-on}")
case "$mode" in
  on | off) ;;
  *) echo "usage: $0 [on|off]" >&2; exit 2 ;;
esac

# The address of the interface the default route uses (Wi-Fi, Ethernet, USB or
# Thunderbolt adapters), not a VPN tunnel; en0/en1 as a fallback.
lan_ip() {
  local iface
  iface=$(route -n get default 2>/dev/null | awk '/interface:/ {print $2}')
  case "$iface" in utun* | ipsec* | ppp* | "") iface="" ;; esac
  { [[ -n "$iface" ]] && ipconfig getifaddr "$iface"; } 2>/dev/null ||
    ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true
}
ip=$(lan_ip)
# Lowercase: host names are case-insensitive and clients send them lowercased.
name="$(scutil --get LocalHostName 2>/dev/null || hostname -s).local"
name=$(tr '[:upper:]' '[:lower:]' <<<"$name")

python3 - "$mode" "$ip" "$name" <<'EOF'
import pathlib, re, sys
mode, ip, name = sys.argv[1:4]
env = pathlib.Path(".env")
text = env.read_text()

def put(text, key, value):
    line = f"{key}={value}"
    if re.search(rf"(?m)^{key}=", text):
        return re.sub(rf"(?m)^{key}=.*$", line, text)
    return text.rstrip("\n") + "\n" + line + "\n"

local = ["127.0.0.1", "localhost", "host.docker.internal"]
if mode == "off":
    settings = {"IMAGE_SERVER_HOST": "127.0.0.1", "IMAGE_SERVER_ALLOW_REMOTE": "false",
                "IMAGE_SERVER_ALLOWED_HOSTS": ",".join(local)}
else:
    hosts = local + [name] + ([ip] if ip else [])
    settings = {"IMAGE_SERVER_HOST": "0.0.0.0", "IMAGE_SERVER_ALLOW_REMOTE": "true",
                "IMAGE_SERVER_ALLOWED_HOSTS": ",".join(hosts)}
for key, value in settings.items():
    text = put(text, key, value)
env.write_text(text)
EOF
chmod 600 .env

port=$(sed -n 's/^IMAGE_SERVER_PORT=//p' .env | tail -n 1 | tr -d "\"'")
port=${port:-7865}
if [[ "$mode" == "off" ]]; then
  echo "LAN access off (loopback only). Restart: scripts/install_image_server.sh"
  exit 0
fi
echo "LAN access on. Clients use: http://$name:$port  (or http://${ip:-THIS-MAC-IP}:$port)"
echo "Allowed Host names: localhost, $name${ip:+, $ip}. If the IP changes (DHCP), re-run this script,"
echo "or reserve the address for this Mac in your router."

# The macOS application firewall filters incoming connections per program.
fw=/usr/libexec/ApplicationFirewall/socketfilterfw
python_bin=$(cd image-server && uv run --frozen python -c 'import os, sys; print(os.path.realpath(sys.executable))' 2>/dev/null || true)
if [[ "$($fw --getglobalstate 2>/dev/null)" == *"enabled"* && -n "$python_bin" ]]; then
  echo
  echo "The macOS firewall is on. If other machines can't connect, allow the server's Python once:"
  echo "  sudo $fw --add \"$python_bin\""
  echo "  sudo $fw --unblockapp \"$python_bin\""
fi
echo
echo "Restart the server to apply: scripts/install_image_server.sh"
echo "(it then checks the network address with scripts/check_lan.sh, which spots a firewall block)"
