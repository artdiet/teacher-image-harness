"""Runs image jobs one at a time on a dedicated MLX thread, keeping one model loaded."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import sys
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
        on_step: Callable[[int, int], None] | None = None,
    ) -> Image.Image: ...

    def close(self) -> None: ...


BackendFactory = Callable[[ModelSpec], Backend]


class EngineBusy(RuntimeError):
    """Too many jobs are already waiting."""


class EngineClosed(EngineBusy):
    """The server is shutting down and takes no new work."""


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
    on_start: Callable[[], None] | None = None  # called on the MLX thread when the job starts
    on_progress: Callable[[float], None] | None = None  # 0.0 .. 1.0 across all n images
    # True while the job waits for its turn: only then may its asyncio task be cancelled
    # (once it runs on the MLX thread, a cancel must go through `cancel` instead).
    waiting: bool = False


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


def mlx_memory_stats(reset_peak: bool = False) -> dict:
    """GPU memory numbers in GB, only if MLX is already loaded (never imports it)."""
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return {}
    gb = 1e9
    stats = {"active_gb": round(mx.get_active_memory() / gb, 2),
             "peak_gb": round(mx.get_peak_memory() / gb, 2),
             "cache_gb": round(mx.get_cache_memory() / gb, 2)}
    try:
        stats["budget_gb"] = round(mx.device_info()["max_recommended_working_set_size"] / gb, 2)
    except Exception:
        pass
    if reset_peak:
        mx.reset_peak_memory()
    return stats


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
        self._closed = False
        self.draining = False  # stopping: run what's queued, but don't swap models for it
        self.memory: dict = {}  # GPU memory after the last job (peak = that job's peak)

    @property
    def loaded_model(self) -> str | None:
        return self._loaded.id if self._loaded else None

    @property
    def is_full(self) -> bool:
        return self.pending >= self._max_pending

    @property
    def closed(self) -> bool:
        return self._closed

    def reserve(self) -> None:
        """Take a queue slot now, synchronously, so a burst of submissions can't all pass a
        capacity check and then fail later. Pass reserved=True to run() to use it."""
        if self._closed:
            raise EngineClosed("the image server is shutting down")
        if self.pending >= self._max_pending:
            raise EngineBusy(f"{self.pending} jobs already waiting")
        self.pending += 1

    def release(self) -> None:
        """Give back a slot taken with reserve() for a job that never reached run()."""
        self.pending = max(0, self.pending - 1)

    async def run(self, job: Job, *, reserved: bool = False) -> list[JobResult]:
        if not reserved:
            self.reserve()
        try:
            if job.cancel.is_set():  # cancelled before it even queued
                raise JobCancelled("client left while the job was queued")
            job.waiting = True
            try:
                await self._turn.acquire()  # cancelling the task here is safe: asyncio.Lock passes the turn on
            finally:
                job.waiting = False
            try:
                if job.cancel.is_set():
                    raise JobCancelled("client left while the job was queued")
                if self._closed or (self.draining and self.loaded_model not in (None, job.model.id)):
                    raise EngineClosed("the image server is shutting down")
                if self._before_job is not None:
                    await self._before_job()  # right before this job, not when it was queued
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(self._executor, self._run_sync, job)
                try:
                    return await asyncio.shield(future)
                except asyncio.CancelledError:
                    # The request went away (or shutdown cancelled it): stop the picture at its
                    # next step, and keep the GPU turn until the MLX thread has really let go.
                    job.cancel.set()
                    with contextlib.suppress(BaseException):
                        await future
                    raise
            finally:
                self._turn.release()
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
        if self._closed:
            return
        self._closed = True
        self._executor.submit(self._unload_sync)
        self._executor.shutdown(wait=True)

    # --- everything below runs on the MLX thread ---

    def _run_sync(self, job: Job) -> list[JobResult]:
        if job.cancel.is_set():
            raise JobCancelled("client left before the job started")
        if job.on_start is not None:
            job.on_start()
        backend = self._ensure_loaded(job.model)
        mlx_memory_stats(reset_peak=True)  # so the next reading is this job's peak
        results = []
        try:
            for i in range(job.n):
                if job.cancel.is_set():
                    raise JobCancelled("client left")
                seed = (job.seed + i) % 2**32  # stays a seed the API accepts back

                def step(done: int, total: int, i: int = i) -> None:
                    if job.on_progress is not None and total:
                        job.on_progress(min(1.0, (i + done / total) / job.n))

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
                    on_step=step,
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
        self.memory = mlx_memory_stats()
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
