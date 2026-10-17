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

    import uvicorn

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
    uvicorn.run(
        create_app(settings, engine, restart_allowed=lambda: fingerprint(env_path) == startup_fingerprint),
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
        server_header=False,
        limit_concurrency=16,
    )


if __name__ == "__main__":
    main()
