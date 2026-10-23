"""Entry point: `uv run image-server` (or `python -m image_server`)."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def fingerprint(env_path: str) -> str:
    """Hash of everything a restart would re-read: .env, the lock file and the sources."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in [Path(env_path) if env_path else None, package.parents[1] / "uv.lock",
                 package.parents[1] / "pyproject.toml"]:
        if path and path.is_file():
            digest.update(path.read_bytes())
    for source in sorted(package.rglob("*.py")):
        stat = source.stat()
        digest.update(f"{source.relative_to(package)}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


def uvicorn_options() -> dict:
    """How the server runs under uvicorn (tests/test_shutdown.py starts it the same way)."""
    from .connections import GuardedH11Protocol
    from .jobs import SHUTDOWN_GRACE_SECONDS

    return {
        "log_level": "info",
        "access_log": False,
        "server_header": False,
        # Per-client connection caps and stall deadlines instead of uvicorn's global
        # limit_concurrency, which any LAN device could fill with idle sockets.
        "http": GuardedH11Protocol,
        # No proxy in front: the peer address is the client (the private-network filter
        # and the loopback-only primary token rely on it).
        "proxy_headers": False,
        # On stop, uvicorn waits for open requests before the app's own shutdown (which
        # finishes or records background jobs). Without a limit one stalled client keeps a
        # stopping server alive forever with its listener already closed. Open requests
        # (Open WebUI's pictures) get the same grace as background jobs, on one clock started
        # by serve(); grace + stopping + unloading stays under launchd's ExitTimeOut (60 s).
        "timeout_graceful_shutdown": int(SHUTDOWN_GRACE_SECONDS),
    }


def serve(app, host: str, port: int, *, on_stop, **overrides) -> None:
    """Run `app` like uvicorn.run(), calling on_stop() as soon as a stop is requested."""
    import uvicorn

    class DrainingServer(uvicorn.Server):
        def handle_exit(self, sig, frame) -> None:
            if not self.should_exit:
                on_stop()  # in the signal handler: before uvicorn even closes its listener
            super().handle_exit(sig, frame)

    server = DrainingServer(uvicorn.Config(app, host=host, port=port, **{**uvicorn_options(), **overrides}))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    if not server.started:
        sys.exit(3)  # like uvicorn.run: a failed start (port in use) exits non-zero for launchd


def main() -> None:
    # The repo-root .env (gitignored) holds the token. Real env vars take precedence.
    # Loaded before anything imports huggingface_hub, so HF_HOME set there applies too.
    env_path = os.environ.get("IMAGE_SERVER_ENV_FILE") or find_dotenv(usecwd=True)
    load_dotenv(env_path)
    startup_fingerprint = fingerprint(env_path)

    from .config import Settings

    try:
        settings = Settings.from_env()
        problems = settings.validate()
    except ValueError as exc:
        problems = [str(exc)]
    if problems:
        for problem in problems:
            print(f"config error: {problem}", file=sys.stderr)
        sys.exit(2)
    if settings.offline:
        # Must be set before huggingface_hub is imported (it happens lazily, on first model load).
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TQDM_DISABLE", "1")  # no progress bars in the log; jobs are timed here

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    for folder in (settings.data_dir, settings.data_dir / "outputs"):
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(0o700)  # prompts and pictures are private

    from .app import create_app
    from .backends import create_backend, init_mlx_thread
    from .engine import Engine
    from .ollama import free_ollama_memory

    async def before_job() -> None:
        await free_ollama_memory(settings.ollama_url)

    engine = Engine(
        create_backend,
        before_job=before_job if settings.free_ollama else None,
        thread_initializer=init_mlx_thread,
    )
    app = create_app(settings, engine, restart_allowed=lambda: fingerprint(env_path) == startup_fingerprint)
    serve(app, settings.host, settings.port, on_stop=app.state.jobs.begin_shutdown)


if __name__ == "__main__":
    main()
