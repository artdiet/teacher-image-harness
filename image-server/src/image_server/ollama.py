"""Ask the local Ollama to unload its models before an image job.

On a 24 GB Mac the GPU can address ~17.8 GB. An image model (~9-11 GB while
generating) and a chat LLM that Ollama keeps warm can't both fit, so we free
Ollama's memory first. The chat model simply reloads on its next use.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


async def free_ollama_memory(base_url: str, timeout: float = 5.0) -> list[str]:
    """Unload every model Ollama has in memory. Returns their names; never raises."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            response = await client.get("/api/ps")
            response.raise_for_status()
            names = [m.get("name") or m.get("model") for m in response.json().get("models", [])]
            names = [n for n in names if n]
            for name in names:
                await client.post("/api/generate", json={"model": name, "keep_alive": 0})
    except (httpx.HTTPError, ValueError) as exc:
        log.debug("could not free Ollama memory: %s", exc)
        return []
    if names:
        log.info("unloaded Ollama models to make room: %s", ", ".join(names))
    return names
