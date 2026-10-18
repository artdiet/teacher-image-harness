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
  # bootout returns before the old process has exited. A stop can take up to the plist's
  # ExitTimeOut (60 s): running pictures get time to finish. launchd kills it after that.
  for i in $(seq 1 75); do
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || return 0
    [[ $i == 5 ]] && echo "waiting for the old server to stop (it lets running pictures finish, up to a minute)…"
    sleep 1
  done
  echo "the old server didn't stop; forcing it" >&2
  launchctl kill SIGKILL "$DOMAIN/$LABEL" 2>/dev/null || true
  for _ in $(seq 1 10); do
    launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || return 0
    sleep 1
  done
  return 1
}

if [[ "${1:-}" == "uninstall" ]]; then
  stop_agent || echo "warning: $LABEL may still be stopping" >&2
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
if ! stop_agent; then
  rm -f "$PLIST.tmp"
  echo "$LABEL is still registered with launchd; try again in a minute" >&2
  exit 1
fi
mv "$PLIST.tmp" "$PLIST"
for attempt in 1 2 3 4 5; do # launchd can briefly refuse right after a stop
  launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null && break
  [[ $attempt == 5 ]] && { launchctl bootstrap "$DOMAIN" "$PLIST"; exit 1; }
  sleep 2
done

port=$(sed -n 's/^IMAGE_SERVER_PORT=//p' "$REPO/.env" | tail -n 1 | tr -d "\"'")
host=$(sed -n 's/^IMAGE_SERVER_HOST=//p' "$REPO/.env" | tail -n 1 | tr -d "\"'")
for _ in $(seq 1 30); do
  if curl -sf --max-time 2 "http://127.0.0.1:${port:-7865}/health" >/dev/null; then
    echo "started $LABEL on ${host:-127.0.0.1}:${port:-7865} (logs: $LOGS/image-server.log)"
    if [[ "$host" == "0.0.0.0" ]]; then
      "$REPO/scripts/check_lan.sh" || true
    fi
    exit 0
  fi
  sleep 1
done
echo "$LABEL did not answer within 30 s; see $LOGS/image-server.log" >&2
exit 1
