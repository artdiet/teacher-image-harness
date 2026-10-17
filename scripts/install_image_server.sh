#!/usr/bin/env bash
# Install, restart or remove the image server as a per-user launchd agent: it
# starts at login, restarts if it crashes, and restarts lean after idling.
# Re-run it after changing .env (the server reads .env only at startup).
#
#   scripts/install_image_server.sh            # install / restart
#   scripts/install_image_server.sh uninstall  # stop and remove
set -euo pipefail

LABEL=com.teacher-image-harness.image-server
REPO=$(cd "$(dirname "$0")/.." && pwd)
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGS="$HOME/Library/Logs/teacher-image-harness"
DOMAIN="gui/$(id -u)"

stop_agent() {
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  for _ in $(seq 1 30); do # bootout can return before the old process has exited
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || return 0
    sleep 1
  done
  echo "warning: $LABEL is still shutting down" >&2
}

if [[ "${1:-}" == "uninstall" ]]; then
  stop_agent
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

# Preflight, before touching the running server.
UV=$(command -v uv) || { echo "uv not found (brew install uv)" >&2; exit 1; }
[[ -f "$REPO/.env" ]] || { echo "missing $REPO/.env; run scripts/setup_env.sh first" >&2; exit 1; }
chmod 600 "$REPO/.env"
(cd "$REPO/image-server" && "$UV" sync --frozen --quiet)
if ! (cd "$REPO" && "$UV" run --frozen --quiet --project image-server python - <<'EOF'
import sys
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))
from image_server.config import Settings
try:
    problems = Settings.from_env().validate()
except ValueError as exc:
    problems = [str(exc)]
for problem in problems:
    print("config error:", problem, file=sys.stderr)
sys.exit(1 if problems else 0)
EOF
); then
  echo "fix .env, then re-run" >&2
  exit 1
fi

mkdir -p "$LOGS" "$(dirname "$PLIST")"
sed -e "s|__UV__|$UV|g" -e "s|__REPO__|$REPO|g" -e "s|__LOGS__|$LOGS|g" \
  "$REPO/deploy/launchd/$LABEL.plist.template" >"$PLIST.tmp"
plutil -lint "$PLIST.tmp" >/dev/null
stop_agent
mv "$PLIST.tmp" "$PLIST"
launchctl bootstrap "$DOMAIN" "$PLIST"

port=$(sed -n 's/^IMAGE_SERVER_PORT=//p' "$REPO/.env" | tail -n 1 | tr -d "\"'")
for _ in $(seq 1 30); do
  if curl -sf --max-time 2 "http://127.0.0.1:${port:-7865}/health" >/dev/null; then
    echo "started $LABEL on 127.0.0.1:${port:-7865} (logs: $LOGS/image-server.log)"
    exit 0
  fi
  sleep 1
done
echo "$LABEL did not answer within 30 s; see $LOGS/image-server.log" >&2
exit 1
