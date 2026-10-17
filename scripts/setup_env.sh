#!/usr/bin/env bash
# Create the gitignored .env from .env.example and fill in a random token.
# Re-running keeps an existing token. The token is never printed.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "created .env from .env.example"
fi
chmod 600 .env

if grep -qE '^IMAGE_SERVER_TOKEN=.+' .env; then
  echo "IMAGE_SERVER_TOKEN already set; leaving it alone"
else
  python3 - <<'EOF'
import pathlib, re, secrets
env = pathlib.Path(".env")
text = env.read_text()
line = "IMAGE_SERVER_TOKEN=" + secrets.token_urlsafe(32)
text, found = re.subn(r"(?m)^IMAGE_SERVER_TOKEN=.*$", line, text, count=1)
env.write_text(text if found else text.rstrip("\n") + "\n" + line + "\n")
EOF
  echo "generated IMAGE_SERVER_TOKEN"
fi
