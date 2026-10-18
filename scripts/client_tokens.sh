#!/usr/bin/env bash
# Per-device tokens for Claude machines that use the image server (one per device,
# so one can be revoked without touching the others). Only SHA-256 hashes are stored,
# in <data dir>/client_tokens.json (mode 600); the server picks up changes without a restart.
#
#   scripts/client_tokens.sh add <device-name>     # prints the new token once
#   scripts/client_tokens.sh list
#   scripts/client_tokens.sh revoke <device-name>
set -euo pipefail
cd "$(dirname "$0")/.."

action=${1:-list}
label=${2:-}
[[ "$action" == "list" || -n "$label" ]] || { echo "usage: $0 add|revoke <device-name> | list" >&2; exit 1; }

uv run --frozen --quiet --project image-server python - "$action" "$label" <<'EOF'
import json, os, re, secrets, sys, time
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))
from image_server.config import Settings
from image_server.security import hash_token

action, label = sys.argv[1], sys.argv[2]
path = Settings.from_env().tokens_file
data = {"tokens": []}
if path.exists():
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        sys.exit(f"{path} is damaged ({exc}). The server refuses all device tokens until it is fixed: "
                 "repair it by hand, or delete it and add the devices again.")
    if not isinstance(data, dict) or not isinstance(data.get("tokens", []), list):
        sys.exit(f'{path} is damaged (expected {{"tokens": [...]}}): repair or delete it')
tokens = data.setdefault("tokens", [])
label_of = lambda t: t.get("label") if isinstance(t, dict) else None  # noqa: E731

def save():
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)

if action == "add":
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", label):
        sys.exit("device name: letters, digits, dot, dash or underscore (max 40)")
    if any(label_of(t) == label for t in tokens):
        sys.exit(f"'{label}' already has a token; revoke it first")
    token = secrets.token_urlsafe(32)
    tokens.append({"label": label, "sha256": hash_token(token), "created_at": time.time()})
    save()
    print(f"token for '{label}' (shown once; paste it into that machine's Teach Image extension):")
    print(token)
elif action == "revoke":
    before = len(tokens)
    data["tokens"] = [t for t in tokens if label_of(t) != label]  # malformed entries are kept
    if len(data["tokens"]) == before:
        sys.exit(f"no token named '{label}'")
    save()
    print(f"revoked '{label}' (takes effect within a few seconds)")
elif action == "list":
    if not tokens:
        print("no device tokens yet (scripts/client_tokens.sh add <device-name>)")
    for t in tokens:
        if not label_of(t):
            print("(malformed entry: the server ignores it)")
            continue
        added = time.strftime('%Y-%m-%d %H:%M', time.localtime(t.get('created_at', 0)))
        print(f"{label_of(t):<24} added {added}")
else:
    sys.exit(f"unknown action {action!r}")
EOF
