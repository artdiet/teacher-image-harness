"""Write docs/openapi.json from the server's actual routes.

    uv run --project image-server python scripts/export_openapi.py

The running server doesn't serve its schema (openapi_url is off); this keeps
the published contract in sync with the code instead.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from image_server.app import create_app
from image_server.config import Settings
from image_server.engine import Engine


class _NoBackend:
    def __init__(self, spec):  # never called: no jobs run during export
        raise RuntimeError("export only")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(token="x" * 32, data_dir=Path(tmp))
        app = create_app(settings, Engine(_NoBackend, release_memory=lambda: None))
        schema = app.openapi()
    schema["components"].setdefault("securitySchemes", {})["bearer"] = {
        "type": "http", "scheme": "bearer",
        "description": "A device token from scripts/client_tokens.sh, or IMAGE_SERVER_TOKEN from the server's .env"}
    for path, ops in schema["paths"].items():
        for op in ops.values():
            # The auth dependency reads the Authorization header; document it as the security scheme.
            params = [p for p in op.get("parameters", [])
                      if not (p["in"] == "header" and p["name"].lower() == "authorization")]
            if params:
                op["parameters"] = params
            else:
                op.pop("parameters", None)
            if path != "/health":
                op["security"] = [{"bearer": []}]
    schema["servers"] = [{"url": "http://{host}:{port}", "variables": {
        "host": {"default": "127.0.0.1", "description": "the Mac's LAN address or name"},
        "port": {"default": "7865"}}}]
    out = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
    out.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {out} ({len(schema['paths'])} paths)")


if __name__ == "__main__":
    main()
