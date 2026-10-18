"""Background image jobs: submit, poll, cancel.

Jobs run one at a time on the shared Engine (single MLX thread, FIFO), the same
queue the OpenAI-style endpoints use. Results go to the ImageStore with a JSON
sidecar. A job record is written to disk when the job is accepted and again when
it ends, so a job_id handed to a client always resolves: across the server's idle
restart, a reinstall, or a crash (an unfinished job then reads as interrupted).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from PIL import Image
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import ModelSpec, Settings
from .engine import Engine, EngineBusy, EngineClosed, Job, JobCancelled
from .imaging import ImageInputError, load_reference_image, parse_size, remove_white_background, size_like, to_png
from .prompts import build_prompt
from .store import ImageStore, ReferenceStore, read_json, write_private

log = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"^job_[0-9a-f]{16}$")
KEEP_IN_MEMORY = 500
INTERRUPTED = "interrupted by a server restart; please try again"
SHUTDOWN_GRACE_SECONDS = 25.0  # let running pictures finish before a restart...
STOP_SECONDS = 5.0  # ...then stop the rest (a running job stops at its next denoising step)


class JobRequest(BaseModel):
    """POST /jobs body. Unknown fields are rejected so typos don't pass silently."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    style: Literal["icon", "coloring", "storybook", "none"] = Field(
        default="none", description="Wrap the prompt in a classroom style template (see docs/API.md)")
    negative_prompt: str | None = Field(default=None, max_length=2000)
    model: str | None = None
    reference_ids: list[str] = Field(default_factory=list, max_length=10)
    source_image_id: str | None = None
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    size: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0, le=20)
    n: int = Field(default=1, ge=1, le=8)
    transparent: bool = False


class JobError(ValueError):
    """The request can't be run as asked (HTTP 400)."""


class JobNotRecorded(RuntimeError):
    """The job couldn't be written down, so its id couldn't be promised to resolve (HTTP 503)."""


@dataclass
class JobRecord:
    id: str
    params: dict
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    error: str | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    cancel_requested: bool = False  # by a client (DELETE), as opposed to a shutdown
    job: Job | None = None  # while queued or running
    task: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def public(self) -> dict:
        out = {"job_id": self.id, "status": self.status, "progress": round(self.progress, 3),
               "params": self.params, "created_at": self.created_at,
               "started_at": self.started_at, "finished_at": self.finished_at}
        if self.error:
            out["error"] = self.error
        if self.result:
            out["result"] = self.result
        return out


class JobManager:
    def __init__(self, settings: Settings, engine: Engine, images: ImageStore,
                 references: ReferenceStore) -> None:
        self.settings = settings
        self.engine = engine
        self.images = images
        self.references = references
        self.models = settings.enabled_models()
        self._jobs: dict[str, JobRecord] = {}
        self._tasks: set[asyncio.Task] = set()
        self._dir = settings.data_dir / "jobs"
        self._active_dir = self._dir / "active"  # one empty marker per unfinished job
        self._stop_deadline: float | None = None  # set when a stop is requested
        self.closing = False  # set before the process exits: no new jobs

    # --- validation -------------------------------------------------------------------

    def _resolve(self, req: JobRequest) -> tuple[ModelSpec, list[Image.Image], tuple[int, int]]:
        editing = bool(req.source_image_id or req.reference_ids)
        if req.model:
            spec = self.models.get(req.model)
            if spec is None:
                raise JobError(f"unknown model {req.model!r}; available: {', '.join(self.models)}")
            if editing and not spec.supports_edit:
                raise JobError(f"{spec.id} can't use reference or source images; "
                               f"use {self.settings.edit_model} or leave model empty")
        else:
            spec = self.models[self.settings.edit_model if editing else self.settings.default_model]
        if req.n > self.settings.max_n:
            raise JobError(f"n must be at most {self.settings.max_n}")
        if len(req.reference_ids) + bool(req.source_image_id) > spec.max_reference_images:
            raise JobError(f"{spec.id} accepts at most {spec.max_reference_images} pictures "
                           "(source image + references)")
        if self.engine.is_full:  # fail fast, before decoding pictures on the event loop
            raise EngineBusy("too many jobs waiting")

        pictures: list[Image.Image] = []
        if req.source_image_id:
            data = self.images.full(req.source_image_id)
            if data is None:
                raise JobError(f"no image {req.source_image_id!r}")
            try:
                pictures.append(load_reference_image(data))
            except ImageInputError as exc:
                raise JobError(str(exc)) from None
        for reference_id in req.reference_ids:
            image = self.references.image(reference_id)
            if image is None:
                raise JobError(f"no reference {reference_id!r}")
            pictures.append(image)

        if req.width or req.height:
            if not (req.width and req.height):
                raise JobError("give both width and height, or neither")
            size = f"{req.width}x{req.height}"
        else:
            size = req.size
        default = size_like(pictures[0], 1024 * 1024) if pictures else (1024, 1024)
        try:
            dims = parse_size(size, default, self.settings.max_pixels)
        except ImageInputError as exc:
            raise JobError(str(exc)) from None
        return spec, pictures, dims

    # --- lifecycle --------------------------------------------------------------------

    def submit(self, req: JobRequest) -> JobRecord:
        if self.closing:
            raise EngineClosed("the image server is restarting")
        spec, pictures, (width, height) = self._resolve(req)
        seed = req.seed if req.seed is not None else secrets.randbelow(2**31)
        full_prompt = build_prompt(req.style, req.prompt)
        params = {
            "prompt": req.prompt, "style": req.style, "full_prompt": full_prompt,
            "negative_prompt": req.negative_prompt, "model": spec.id,
            "reference_ids": req.reference_ids, "source_image_id": req.source_image_id,
            "width": width, "height": height, "seed": seed,
            "steps": req.steps or spec.default_steps,
            "guidance": req.guidance if req.guidance is not None else spec.default_guidance,
            "n": req.n, "transparent": req.transparent,
        }
        record = JobRecord(id="job_" + secrets.token_hex(8), params=params)
        record.job = Job(model=spec, prompt=full_prompt, width=width, height=height, seed=seed, n=req.n,
                         steps=params["steps"], guidance=params["guidance"], references=pictures,
                         cancel=record.cancel)
        self.engine.reserve()  # take the queue slot now (EngineBusy -> 503), not when the task starts
        # The id goes to the client only once the job is on disk: it must resolve after any restart.
        if not (self._persist(record) and self._mark_active(record.id, True)):
            self.engine.release()
            (self._dir / f"{record.id}.json").unlink(missing_ok=True)
            raise JobNotRecorded("the image server couldn't record the job (is its disk full?)")
        record.task = asyncio.create_task(self._run(record, record.job, spec))
        self._tasks.add(record.task)
        record.task.add_done_callback(self._tasks.discard)
        self._jobs[record.id] = record
        self._trim()
        return record

    async def _run(self, record: JobRecord, job: Job, spec: ModelSpec) -> None:
        def started() -> None:
            record.status, record.started_at = "running", time.time()

        def progress(value: float) -> None:
            record.progress = value

        job.on_start, job.on_progress = started, progress
        try:
            results = await self.engine.run(job, reserved=True)
        except (JobCancelled, asyncio.CancelledError) as exc:
            if record.cancel_requested:
                record.status = "cancelled"
            else:  # stopped by a shutdown, not by the client
                record.status, record.error = "failed", INTERRUPTED
            if isinstance(exc, asyncio.CancelledError):
                self._finish(record)
                raise
        except EngineClosed:
            record.status, record.error = "failed", INTERRUPTED
        except EngineBusy:
            record.status, record.error = "failed", "the image server is busy; try again shortly"
        except ValueError as exc:  # the model rejected the input
            record.status, record.error = "failed", str(exc)
        except Exception:
            log.exception("job %s failed", record.id)
            record.status, record.error = "failed", "image generation failed; see the image server log"
        else:
            try:
                record.result = await self._save(record, spec, results)
                record.status, record.progress = "done", 1.0
            except Exception:
                log.exception("job %s: saving results failed", record.id)
                record.status, record.error = "failed", "could not save the result"
        self._finish(record)

    def _finish(self, record: JobRecord) -> None:
        record.finished_at = time.time()
        record.job = None  # let go of the reference pictures
        # Without the final record on disk, the marker stays, so the next start reports the job
        # as interrupted instead of leaving it "queued" forever.
        if self._persist(record):
            self._mark_active(record.id, False)

    def _mark_active(self, job_id: str, active: bool) -> bool:
        """Markers let restore() find unfinished jobs without reading every record."""
        marker = self._active_dir / job_id
        try:
            if active:
                self._active_dir.mkdir(parents=True, exist_ok=True)
                os.chmod(self._active_dir, 0o700)
                marker.touch(mode=0o600)
            else:
                marker.unlink(missing_ok=True)
            return True
        except OSError as exc:
            log.warning("could not update the marker for job %s: %s", job_id, exc)
            return False

    async def _save(self, record: JobRecord, spec: ModelSpec, results) -> dict:
        params = record.params
        images = []
        for result in results:
            picture = remove_white_background(result.image) if params["transparent"] else result.image
            png = to_png(picture, {"model": spec.id, "seed": result.seed, "steps": params["steps"],
                                   "width": picture.width, "height": picture.height})
            sidecar = {**params, "seed": result.seed, "width": picture.width, "height": picture.height,
                       "n": 1, "job_id": record.id, "seconds": round(result.seconds, 2),
                       "license": spec.license}
            image_id = await asyncio.to_thread(self.images.save, png, sidecar)
            images.append({"image_id": image_id, "seed": result.seed, "width": picture.width,
                           "height": picture.height, "seconds": round(result.seconds, 2)})
        result = {"image_id": images[0]["image_id"], "seed": images[0]["seed"], "images": images,
                  "model": spec.id, "params": params}
        if params.get("negative_prompt"):
            result["ignored"] = ["negative_prompt"]  # none of the current models use one
        return result

    def get(self, job_id: str) -> dict | None:
        record = self._jobs.get(job_id)
        if record is not None:
            return record.public()
        if not JOB_ID_RE.fullmatch(job_id or ""):
            return None
        return read_json(self._dir / f"{job_id}.json")

    def cancel(self, job_id: str) -> dict | None:
        record = self._jobs.get(job_id)
        if record is None:
            return self.get(job_id)  # already finished (or unknown)
        if record.active:
            record.cancel_requested = True
            record.cancel.set()  # a running job stops at its next denoising step
            if record.job is not None and record.job.waiting and record.task is not None:
                # Still waiting for its turn: drop it now so it frees its queue slot at once.
                record.task.cancel()
                record.status = "cancelled"
        return record.public()

    async def list(self, limit: int = 20) -> list[dict]:
        """Newest first: jobs of this process plus earlier ones read back from disk."""
        records = {r.id: r.public() for r in self._jobs.values()}
        for data in await asyncio.to_thread(self._recent_on_disk, limit):
            records.setdefault(data.get("job_id"), data)
        return sorted(records.values(), key=lambda r: r.get("created_at", 0), reverse=True)[:limit]

    def _recent_on_disk(self, limit: int) -> list[dict]:
        if not self._dir.is_dir():
            return []
        paths = sorted(self._dir.glob("job_*.json"), key=_mtime, reverse=True)[:limit]
        return [data for data in map(read_json, paths) if data]

    @property
    def active(self) -> int:
        return sum(1 for r in self._jobs.values() if r.active)

    def restore(self) -> int:
        """At startup: jobs an earlier process accepted but never finished (crash, forced
        stop) are recorded as interrupted, so their clients get an answer. Only the jobs
        that still have an active marker are read, however long the server was off."""
        if not self._active_dir.is_dir():
            return 0
        interrupted = 0
        for marker in self._active_dir.iterdir():
            if not JOB_ID_RE.fullmatch(marker.name):
                continue
            path = self._dir / f"{marker.name}.json"
            data = read_json(path)
            if data and data.get("status") in ("queued", "running"):
                data.update(status="failed", error=INTERRUPTED, finished_at=time.time())
                try:
                    write_private(path, json.dumps(data, indent=2).encode())
                    interrupted += 1
                except OSError as exc:
                    log.warning("could not update job record %s: %s", path.name, exc)
                    continue
            marker.unlink(missing_ok=True)
        if interrupted:
            log.warning("%d job(s) were interrupted when the server last stopped", interrupted)
        return interrupted

    def begin_shutdown(self) -> None:
        """A stop was requested (SIGTERM, also the idle restart): take no new jobs, load no
        other model, and start the grace clock now. Open requests (Open WebUI's pictures) and
        background jobs share it: uvicorn waits up to the same grace for requests before the
        app's own shutdown runs."""
        if self._stop_deadline is None:
            self._stop_deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
        self.closing = True
        self.engine.draining = True

    async def shutdown(self, grace: float | None = None) -> None:
        """Before the process exits: let queued and running jobs finish within what is left of
        the grace period, then stop the rest and record them as interrupted."""
        self.begin_shutdown()
        if grace is None:
            grace = max(0.0, self._stop_deadline - time.monotonic())
        pending = {t for t in self._tasks if not t.done()}
        if pending:
            log.info("letting %d job(s) finish (up to %.0fs) before stopping", len(pending), grace)
            _, pending = await asyncio.wait(pending, timeout=grace)
        if pending:
            for record in self._jobs.values():
                if record.active:
                    record.cancel.set()
                    if record.job is not None and record.job.waiting and record.task is not None:
                        record.task.cancel()
            _, pending = await asyncio.wait(pending, timeout=STOP_SECONDS)
        for record in self._jobs.values():
            if record.active:  # still not stopped: record it now; the process is about to exit
                record.status, record.error = "failed", INTERRUPTED
                self._finish(record)

    def _persist(self, record: JobRecord) -> bool:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self._dir, 0o700)
            write_private(self._dir / f"{record.id}.json", json.dumps(record.public(), indent=2).encode())
            return True
        except OSError as exc:
            log.warning("could not save job record %s: %s", record.id, exc)
            return False

    def _trim(self) -> None:
        if len(self._jobs) <= KEEP_IN_MEMORY:
            return
        finished = sorted((r for r in self._jobs.values() if not r.active), key=lambda r: r.created_at)
        for record in finished[: len(self._jobs) - KEEP_IN_MEMORY]:
            del self._jobs[record.id]  # still readable from disk


def _mtime(path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:  # removed meanwhile
        return 0.0
