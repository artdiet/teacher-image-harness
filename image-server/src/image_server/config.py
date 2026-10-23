"""Settings (from environment variables / .env) and the model registry."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
DEFAULT_DATA_DIR = "~/Library/Application Support/teacher-image-harness"


@dataclass(frozen=True)
class ModelSpec:
    """A pre-quantized MLX model the server knows how to run."""

    id: str
    family: str  # backend key, see backends.py
    repo: str  # Hugging Face repo with the MLX weights
    revision: str  # pinned commit (scripts/download_models.sh fetches the same)
    license: str
    supports_edit: bool
    max_reference_images: int = 0
    default_steps: int | None = None
    default_guidance: float | None = None
    research_only: bool = False  # license forbids production use


MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="flux2-klein-4b",
            family="flux2-klein",
            repo="mflux-community/flux2-klein-4b-mflux-q8",
            revision="ca49409e4ce21c6717ee87caadd9488e0b15208f",
            license="Apache-2.0",
            supports_edit=True,
            max_reference_images=4,
            default_steps=4,  # distilled
        ),
        ModelSpec(
            id="z-image-turbo",
            family="z-image",
            repo="mflux-community/z-image-turbo-mflux-q8",
            revision="4ccff28917346aa9daae49cd3c477cf5260a242c",
            license="Apache-2.0",
            supports_edit=False,
            default_steps=9,
        ),
        ModelSpec(
            id="qwen-image-2.1-turbo",
            family="qwen-image-2.1",
            repo="mflux-community/qwen-image-2-1-turbo-mflux-q4",
            revision="bda8951ec6b8cc288a961432fe15c22433521a1d",
            license="Qwen Research License (research or evaluation only)",
            supports_edit=True,
            max_reference_images=10,
            default_steps=8,  # Turbo
            research_only=True,
        ),
    )
}


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"IMAGE_SERVER_{name} must be a whole number, got {value!r}") from None


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 7865
    token: str = ""
    allow_remote: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "host.docker.internal")
    data_dir: Path = Path(DEFAULT_DATA_DIR).expanduser()
    save_outputs: bool = True
    default_model: str = "flux2-klein-4b"
    edit_model: str = "flux2-klein-4b"
    enable_research_models: bool = False
    offline: bool = True  # never reach Hugging Face at runtime
    idle_unload_seconds: int = 600
    restart_when_idle: bool = True  # after an idle unload, exit so launchd restarts a lean process
    free_ollama: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    max_n: int = 4
    max_pixels: int = 1536 * 1024
    max_upload_bytes: int = 20 * 1024 * 1024
    private_clients_only: bool = True  # refuse clients outside private/loopback ranges
    client_tokens_file: Path | None = None  # default: <data_dir>/client_tokens.json

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        def get(name: str) -> str | None:
            return env.get(f"IMAGE_SERVER_{name}")

        defaults = cls()
        hosts = get("ALLOWED_HOSTS")
        return cls(
            host=get("HOST") or defaults.host,
            port=_int("PORT", get("PORT"), defaults.port),
            token=(get("TOKEN") or "").strip(),
            allow_remote=_bool(get("ALLOW_REMOTE"), defaults.allow_remote),
            allowed_hosts=tuple(h.strip().lower() for h in hosts.split(",") if h.strip())
            if hosts
            else defaults.allowed_hosts,
            data_dir=Path(get("DATA_DIR") or DEFAULT_DATA_DIR).expanduser(),
            save_outputs=_bool(get("SAVE_OUTPUTS"), defaults.save_outputs),
            default_model=get("DEFAULT_MODEL") or defaults.default_model,
            edit_model=get("EDIT_MODEL") or defaults.edit_model,
            enable_research_models=_bool(
                get("ENABLE_RESEARCH_MODELS"), defaults.enable_research_models
            ),
            offline=_bool(get("OFFLINE"), defaults.offline),
            idle_unload_seconds=_int("IDLE_UNLOAD_SECONDS", get("IDLE_UNLOAD_SECONDS"), defaults.idle_unload_seconds),
            restart_when_idle=_bool(get("RESTART_WHEN_IDLE"), defaults.restart_when_idle),
            free_ollama=_bool(get("FREE_OLLAMA"), defaults.free_ollama),
            ollama_url=get("OLLAMA_URL") or defaults.ollama_url,
            max_n=_int("MAX_N", get("MAX_N"), defaults.max_n),
            max_pixels=_int("MAX_PIXELS", get("MAX_PIXELS"), defaults.max_pixels),
            max_upload_bytes=_int("MAX_UPLOAD_BYTES", get("MAX_UPLOAD_BYTES"), defaults.max_upload_bytes),
            private_clients_only=_bool(get("PRIVATE_CLIENTS_ONLY"), defaults.private_clients_only),
            client_tokens_file=Path(get("CLIENT_TOKENS_FILE")).expanduser()
            if get("CLIENT_TOKENS_FILE") else None,
        )

    @property
    def tokens_file(self) -> Path:
        return self.client_tokens_file or self.data_dir / "client_tokens.json"

    def enabled_models(self) -> dict[str, ModelSpec]:
        return {
            mid: spec
            for mid, spec in MODELS.items()
            if self.enable_research_models or not spec.research_only
        }

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty when OK)."""
        problems = []
        if not self.token:
            problems.append(
                "IMAGE_SERVER_TOKEN is empty. Generate one with: "
                'python3 -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        elif len(self.token) < 24:
            problems.append("IMAGE_SERVER_TOKEN is too short (use at least 24 characters).")
        if self.host not in LOOPBACK_HOSTS and not self.allow_remote:
            problems.append(
                f"IMAGE_SERVER_HOST={self.host} is not a loopback address. "
                "Keep it on 127.0.0.1 (Docker reaches it via host.docker.internal), "
                "or set IMAGE_SERVER_ALLOW_REMOTE=1 if you really mean it."
            )
        models = self.enabled_models()
        for name, mid in (("DEFAULT_MODEL", self.default_model), ("EDIT_MODEL", self.edit_model)):
            if mid not in models:
                problems.append(f"IMAGE_SERVER_{name}={mid} is unknown or not enabled.")
        if self.edit_model in models and not models[self.edit_model].supports_edit:
            problems.append(f"IMAGE_SERVER_EDIT_MODEL={self.edit_model} cannot edit images.")
        if self.max_n < 1:
            problems.append("IMAGE_SERVER_MAX_N must be at least 1.")
        if not 1 <= self.port <= 65535:
            problems.append("IMAGE_SERVER_PORT must be between 1 and 65535.")
        if self.max_pixels < 256 * 256:
            problems.append("IMAGE_SERVER_MAX_PIXELS must be at least 65536.")
        if self.max_upload_bytes < 1024 * 1024:
            problems.append("IMAGE_SERVER_MAX_UPLOAD_BYTES must be at least 1048576 (1 MB).")
        if self.idle_unload_seconds < 0:
            problems.append("IMAGE_SERVER_IDLE_UNLOAD_SECONDS cannot be negative.")
        return problems
