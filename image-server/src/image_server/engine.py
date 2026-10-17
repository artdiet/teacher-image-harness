"""Runs image jobs one at a time on a dedicated MLX thread, keeping one model loaded."""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from .config import ModelSpec

log = logging.getLogger(__name__)


class Backend(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int | None,
        guidance: float | None,
        references: list[Image.Image],
        should_stop: Callable[[], bool] | None = None,
    ) -> Image.Image: ...

    def close(self) -> None: ...


BackendFactory = Callable[[ModelSpec], Backend]


class EngineBusy(RuntimeError):
    """Too many jobs are already waiting."""


class JobCancelled(RuntimeError):
    """The client went away; the job was dropped or stopped between steps."""


@dataclass
class Job:
    model: ModelSpec
    prompt: str
    width: int
    height: int
    seed: int
    n: int = 1
    steps: int | None = None
    guidance: float | None = None
    references: list[Image.Image] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)


@dataclass
class JobResult:
    image: Image.Image
    seed: int
    seconds: float


def release_mlx_memory() -> None:
    """Return MLX's cached GPU buffers to the system."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear is not None:
        clear()


class Engine:
    def __init__(
        self,
        factory: BackendFactory,
        *,
        before_job: Callable[[], Awaitable[object]] | None = None,
        release_memory: Callable[[], None] = release_mlx_memory,
        thread_initializer: Callable[[], None] | None = None,
        max_pending: int = 8,
    ) -> None:
        self._factory = factory
        self._before_job = before_job
        self._release_memory = release_memory
        self._max_pending = max_pending
        # All MLX work (load, generate, unload) happens on this single thread.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx", initializer=thread_initializer
        )
        self._turn = asyncio.Lock()  # FIFO: jobs take the GPU in arrival order
        self._backend: Backend | None = None
        self._loaded: ModelSpec | None = None
        self.pending = 0
        self.last_used = time.monotonic()
        self.load_seconds: float | None = None
        self.used = False  # ML libraries get imported on first load and stay in memory

    @property
    def loaded_model(self) -> str | None:
        return self._loaded.id if self._loaded else None

    async def run(self, job: Job) -> list[JobResult]:
        if self.pending >= self._max_pending:
            raise EngineBusy(f"{self.pending} jobs already waiting")
        self.pending += 1
        try:
            async with self._turn:
                if job.cancel.is_set():
                    raise JobCancelled("client left while the job was queued")
                if self._before_job is not None:
                    await self._before_job()  # right before this job, not when it was queued
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(self._executor, self._run_sync, job)
        finally:
            self.pending -= 1
            self.last_used = time.monotonic()

    async def unload(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._unload_sync)

    async def unload_if_idle(self, idle_seconds: float) -> bool:
        """Unload after idle_seconds without jobs. Returns True if something was unloaded."""
        idle = time.monotonic() - self.last_used
        if self._loaded is not None and self.pending == 0 and idle >= idle_seconds:
            log.info("idle for %.0fs, unloading %s", idle, self._loaded.id)
            await self.unload()
            return True
        return False

    def should_restart(self, idle_seconds: float) -> bool:
        """True when a fresh process would be leaner: libraries loaded, no model, idle."""
        idle = time.monotonic() - self.last_used
        return self.used and self._loaded is None and self.pending == 0 and idle >= idle_seconds

    def shutdown(self) -> None:
        self._executor.submit(self._unload_sync)
        self._executor.shutdown(wait=True)

    # --- everything below runs on the MLX thread ---

    def _run_sync(self, job: Job) -> list[JobResult]:
        if job.cancel.is_set():
            raise JobCancelled("client left before the job started")
        backend = self._ensure_loaded(job.model)
        results = []
        try:
            for i in range(job.n):
                if job.cancel.is_set():
                    raise JobCancelled("client left")
                seed = job.seed + i
                started = time.perf_counter()
                image = backend.generate(
                    prompt=job.prompt,
                    seed=seed,
                    width=job.width,
                    height=job.height,
                    steps=job.steps,
                    guidance=job.guidance,
                    references=job.references,
                    should_stop=job.cancel.is_set,
                )
                results.append(JobResult(image, seed, time.perf_counter() - started))
        except (JobCancelled, ValueError):
            # A cancel or input the model rejected: the loaded model itself is fine.
            gc.collect()
            self._release_memory()
            raise
        except Exception as exc:
            # Most other failures are memory pressure; start clean next time. Clear the
            # traceback first: its frames still reference the model and its buffers.
            traceback.clear_frames(exc.__traceback__)
            self._unload_sync()
            raise
        gc.collect()
        self._release_memory()  # hand activation buffers back between jobs
        return results

    def _ensure_loaded(self, spec: ModelSpec) -> Backend:
        if self._backend is not None and self._loaded is not None and self._loaded.id == spec.id:
            return self._backend
        self._unload_sync()
        log.info("loading %s", spec.id)
        started = time.perf_counter()
        self.used = True
        try:
            self._backend = self._factory(spec)
        except Exception as exc:
            traceback.clear_frames(exc.__traceback__)
            gc.collect()
            self._release_memory()
            raise
        self._loaded = spec
        self.load_seconds = time.perf_counter() - started
        log.info("loaded %s in %.1fs", spec.id, self.load_seconds)
        return self._backend

    def _unload_sync(self) -> None:
        if self._backend is None:
            return
        log.info("unloading %s", self._loaded.id if self._loaded else "model")
        try:
            self._backend.close()
        finally:
            self._backend = None
            self._loaded = None
            gc.collect()
            self._release_memory()
