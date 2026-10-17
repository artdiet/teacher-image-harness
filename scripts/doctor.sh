#!/usr/bin/env bash
# Check every link in the chain: launchd -> image server -> Open WebUI.
# Prints no secrets. Exit code is non-zero if anything is wrong.
set -uo pipefail

LABEL=com.teacher-image-harness.image-server
CONTAINER=${OWUI_CONTAINER:-open-webui}
REPO=$(cd "$(dirname "$0")/.." && pwd)
env_port=$(sed -n 's/^IMAGE_SERVER_PORT=//p' "$REPO/.env" 2>/dev/null | tail -n 1 | tr -d "\"'")
PORT=${IMAGE_SERVER_PORT:-${env_port:-7865}}
failed=0
ok() { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; failed=1; }

echo "image server"
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then ok "launchd agent loaded"
else bad "launchd agent not loaded: run scripts/install_image_server.sh"; fi
if curl -sf --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; then ok "answering on 127.0.0.1:$PORT"
else bad "not answering on 127.0.0.1:$PORT (see ~/Library/Logs/teacher-image-harness/image-server.log)"; fi
listeners=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | awk 'NR > 1 {print $9}' | sort -u)
if [[ -n "$listeners" && -z "$(grep -v '^127\.0\.0\.1:' <<<"$listeners")" ]]; then ok "listening on loopback only"
elif [[ -n "$listeners" ]]; then bad "listening beyond loopback: $listeners"; fi

echo "ollama"
ollama_listen=$(lsof -nP -iTCP:11434 -sTCP:LISTEN 2>/dev/null | awk 'NR > 1 {print $9}' | sort -u)
if [[ -z "$ollama_listen" ]]; then
  bad "Ollama is not running (the director needs it; open the Ollama app)"
elif grep -qE '^\*:11434$|^0\.0\.0\.0:11434$' <<<"$ollama_listen"; then
  bad "Ollama listens on all interfaces (turn off 'Expose Ollama to the network')"
else ok "Ollama running, not exposed to the network"; fi

echo "open webui"
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" == "true" ]]; then
  ok "container '$CONTAINER' running"
  # From inside the container, call the image server with the token stored in
  # the pipe's valves: proves network path, Host check and token all line up.
  result=$(docker exec -i "$CONTAINER" python3 - <<'EOF' 2>&1
import json, sqlite3, urllib.request, urllib.error
db = sqlite3.connect("file:/app/backend/data/webui.db?mode=ro", uri=True)
row = db.execute("select valves, is_active from function where id = 'teach_image'").fetchone()
if not row:
    print("missing: function teach_image not installed"); raise SystemExit
valves, active = json.loads(row[0] or "{}"), row[1]
if not db.execute("select 1 from model where id = 'teach_image' and is_active = 1").fetchone():
    print("missing: model entry teach_image"); raise SystemExit
url = valves.get("IMAGE_SERVER_URL", "http://host.docker.internal:7865").rstrip("/") + "/v1/models"
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + valves.get("IMAGE_SERVER_TOKEN", "")})
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        models = [m["id"] for m in json.load(r)["data"]]
    print(f"ok active={active} models={','.join(models)}")
except urllib.error.HTTPError as e:
    print(f"http {e.code}")
except Exception as e:
    print(f"unreachable {type(e).__name__}")
# The director: Ollama reachable from the container and the model pulled (loads nothing).
if valves.get("USE_DIRECTOR", True):
    name = valves.get("DIRECTOR_MODEL", "gemma4:12b")
    tags = valves.get("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags, timeout=5) as r:
            pulled = {m["name"] for m in json.load(r).get("models", [])}
        found = name in pulled or (":" not in name and f"{name}:latest" in pulled)
        print(f"director {'ok' if found else 'missing'} {name}")
    except Exception as e:
        print(f"director unreachable {type(e).__name__}")
else:
    print("director off")
EOF
)
  while IFS= read -r line; do
    case "$line" in
      ok*) ok "pipe installed and reaches the image server (${line#ok })" ;;
      "http 401") bad "token mismatch: restart the server (scripts/install_image_server.sh), then re-run scripts/install_openwebui.sh" ;;
      "director ok"*) ok "director model ${line#director ok } available to the pipe" ;;
      "director missing"*) bad "director model not pulled: ollama pull ${line#director missing }" ;;
      "director unreachable"*) bad "the pipe can't reach Ollama (${line#director unreachable })" ;;
      "director off") ok "director turned off in the pipe settings" ;;
      "") ;;
      *) bad "pipe check: $line" ;;
    esac
  done <<<"$result"
else
  bad "container '$CONTAINER' not running (start Docker Desktop)"
fi

exit "$failed"
