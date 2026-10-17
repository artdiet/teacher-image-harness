#!/usr/bin/env bash
# Install or update the Teach Image model in the running Open WebUI container.
#
#   scripts/install_openwebui.sh
#
# Reads from the repo's .env: IMAGE_SERVER_TOKEN, IMAGE_SERVER_PORT,
# OWUI_MODEL_NAME (the name shown in the model list) and OWUI_PUBLIC
# (blank = keep access as set in Open WebUI, 1 = every account incl. future ones,
# 0 = remove that).
# Backs up Open WebUI's database first, to
# ~/Library/Application Support/teacher-image-harness/backups (outside the
# repo; the newest 3 are kept, since they hold password hashes and chats).
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=${OWUI_CONTAINER:-open-webui}
BACKUPS="$HOME/Library/Application Support/teacher-image-harness/backups"
KEEP_BACKUPS=3

[[ -f .env ]] || { echo "missing .env; run scripts/setup_env.sh first" >&2; exit 1; }
env_get() { # value of KEY in .env, without surrounding quotes (the file isn't shell syntax)
  sed -n "s/^$1=//p" .env | tail -n 1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/"
}
IMAGE_SERVER_TOKEN=$(env_get IMAGE_SERVER_TOKEN)
port=$(env_get IMAGE_SERVER_PORT)
IMAGE_SERVER_URL=${IMAGE_SERVER_URL:-http://host.docker.internal:${port:-7865}}
OWUI_MODEL_NAME=${OWUI_MODEL_NAME:-$(env_get OWUI_MODEL_NAME)}
OWUI_PUBLIC=${OWUI_PUBLIC:-$(env_get OWUI_PUBLIC)}
export IMAGE_SERVER_TOKEN IMAGE_SERVER_URL OWUI_MODEL_NAME OWUI_PUBLIC
[[ -n "$IMAGE_SERVER_TOKEN" ]] || { echo "IMAGE_SERVER_TOKEN is empty in .env" >&2; exit 1; }
docker inspect "$CONTAINER" >/dev/null 2>&1 || { echo "container '$CONTAINER' not found" >&2; exit 1; }

stamp=$(date +%Y%m%d-%H%M%S)
inner="/app/backend/data/webui-backup-$stamp.db"
cleanup() {
  docker exec "$CONTAINER" rm -f "$inner" /tmp/teach_image_pipe.py /tmp/teach_image_avatar.png \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 1. Back up the database (sqlite backup API: safe while Open WebUI is running).
mkdir -p "$BACKUPS" && chmod 700 "$BACKUPS"
docker exec -e OUT="$inner" "$CONTAINER" python3 -c '
import os, sqlite3
src = sqlite3.connect("file:/app/backend/data/webui.db?mode=ro", uri=True)
dst = sqlite3.connect(os.environ["OUT"])
src.backup(dst)
ok = dst.execute("pragma integrity_check").fetchone()[0]
dst.close(); src.close()
raise SystemExit(0 if ok == "ok" else "backup integrity check failed: " + ok)'
docker cp "$CONTAINER:$inner" "$BACKUPS/" >/dev/null
chmod 600 "$BACKUPS/webui-backup-$stamp.db"
echo "backed up Open WebUI database to $BACKUPS/webui-backup-$stamp.db"
shopt -s nullglob
backups=("$BACKUPS"/webui-backup-*.db) # names sort by time
if (( ${#backups[@]} > KEEP_BACKUPS )); then
  rm -f -- "${backups[@]:0:${#backups[@]}-KEEP_BACKUPS}"
fi

# 2. Install the pipe and its model entry with Open WebUI's own data layer.
docker cp openwebui/teach_image_pipe.py "$CONTAINER:/tmp/teach_image_pipe.py"
avatar_arg=""
if [[ -f openwebui/avatar.png ]]; then
  docker cp openwebui/avatar.png "$CONTAINER:/tmp/teach_image_avatar.png"
  avatar_arg=/tmp/teach_image_avatar.png
fi
if ! output=$(docker exec -i -e IMAGE_SERVER_TOKEN -e IMAGE_SERVER_URL -e OWUI_MODEL_NAME \
    -e OWUI_PUBLIC -e OWUI_GRANT_ALL "$CONTAINER" sh -c '
    cd /app/backend &&
    WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY:-$(cat "${WEBUI_SECRET_KEY_FILE:-/app/backend/.webui_secret_key}")}" \
    ENABLE_DB_MIGRATIONS=False PYTHONDONTWRITEBYTECODE=1 \
    python3 - /tmp/teach_image_pipe.py '"$avatar_arg" <scripts/owui_install.py 2>&1); then
  echo "$output" >&2
  echo "install failed (the database backup above is untouched)" >&2
  exit 1
fi
printf '%s\n' "$output" | grep -vE '^(INFO|DEBUG|WARNING)|^[[:space:]]*$|WEBUI_SECRET_KEY|USER_AGENT' || true
echo "done: reload Open WebUI in the browser to see the model"
