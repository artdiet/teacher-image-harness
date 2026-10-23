"""FastAPI app: the image server's HTTP API.

All routes except /health need "Authorization: Bearer <token>".

Job API (asynchronous, for MCP clients and the book harness; see docs/API.md):
  POST   /jobs                    queue a generation or edit -> {job_id}
  GET    /jobs/{job_id}           status, progress, result
  DELETE /jobs/{job_id}           cancel
  GET    /jobs                    recent jobs
  GET    /images                  recent images (sidecar metadata)
  GET    /images/{id}?size=       preview (WebP, ~768 px; ?budget= bytes) or full (PNG)
  GET    /images/{id}/meta        sidecar: prompt, seed, model, params, references
  POST   /references              add a reference picture (upload, or from an image id)
  GET    /references              list (optional ?tag=)
  GET    /references/{id}         metadata; /references/{id}/image for the picture
  DELETE /references/{id}
  GET    /models                  models and what they support
  GET    /health                  liveness, loaded model, queue depth, GPU memory (no auth)

OpenAI-compatible (synchronous, used by the Open WebUI pipe; unchanged):
  POST /v1/images/generations, POST /v1/images/edits, GET /v1/models, GET /v1/status,
  POST /v1/unload

Routes never declare body parameters: FastAPI would read the body before the
token check runs. They read it themselves, with size caps, after auth.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import secrets
import signal
import time
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from . import __version__
from .config import ModelSpec, Settings
from .engine import Engine, EngineBusy, EngineClosed, Job, JobCancelled
from .imaging import (
    ImageInputError,
    load_reference_image,
    parse_size,
    remove_white_background,
    size_like,
    to_png,
)
from .jobs import JobError, JobManager, JobNotRecorded, JobRequest
from .schemas import (
    Health, ImageList, ImageMeta, ImageUploaded, JobCreated, JobList, JobStatus, ModelList,
    ReferenceFromImage, ReferenceList, ReferenceRecord,
)
from .security import PrivateNetworkOnly, TokenSet, TrustedHosts, bearer_token_dependency
from .store import MIN_PREVIEW_BUDGET, PREVIEW_BUDGET, ImageStore, ReferenceStore, write_private

log = logging.getLogger(__name__)

DEFAULT_SIZE = (1024, 1024)
MAX_PROMPT_CHARS = 4000
MAX_JSON_BODY = 64 * 1024  # a 4000-character prompt is well under this
MAX_FILES = 16
DECODES_AT_ONCE = 2  # a 50 MP upload briefly needs a few hundred MB while it is decoded
STARTED = time.time()


class GenerationRequest(BaseModel):
    # Ignore OpenAI fields we don't use (quality, style, user, ...).
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str | None = None
    response_format: str | None = "b64_json"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0, le=20)
    background: Literal["transparent", "opaque", "auto"] | None = None


def _validation_error(exc: ValidationError) -> HTTPException:
    return HTTPException(422, exc.errors(include_url=False, include_context=False, include_input=False))


def _json_body(model: type[BaseModel]) -> dict:
    """OpenAPI requestBody for routes that read their JSON body themselves."""
    return {"requestBody": {"required": True, "content": {
        "application/json": {"schema": model.model_json_schema(ref_template="#/components/schemas/{model}")}}}}


async def read_json(request: Request, model: type[BaseModel]) -> BaseModel:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_JSON_BODY:
            raise HTTPException(413, "request body too large")
    try:
        return model.model_validate_json(bytes(body))
    except ValidationError as exc:
        raise _validation_error(exc) from None


async def read_form(request: Request, limit: int, max_files: int = MAX_FILES):
    """Multipart form with a cap on bytes actually received (chunked bodies too)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "request body too large")
    received = 0

    async def capped_receive():
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "request body too large")
        return message

    return await Request(request.scope, capped_receive).form(max_files=max_files, max_fields=32)


def create_app(
    settings: Settings,
    engine: Engine,
    restart_allowed: Callable[[], bool] = lambda: True,
) -> FastAPI:
    """restart_allowed() says whether a restart would come back with the same .env and
    code; if not, an idle restart is skipped so a bad edit can't become a crash loop."""
    models = settings.enabled_models()
    require_token = bearer_token_dependency(TokenSet(settings.token, settings.tokens_file))
    images = ImageStore(settings.data_dir)
    references = ReferenceStore(settings.data_dir)
    jobs = JobManager(settings, engine, images, references)

    async def unload_when_idle() -> None:
        warned = False
        while True:
            await asyncio.sleep(30)
            try:
                await engine.unload_if_idle(settings.idle_unload_seconds)
            except Exception:
                log.exception("idle unload failed")
                continue
            if (settings.restart_when_idle and jobs.active == 0
                    and engine.should_restart(settings.idle_unload_seconds)):
                if restart_allowed():
                    # Imported ML libraries keep ~0.4 GB even with no model loaded; a fresh
                    # process idles at a fraction of that. launchd (KeepAlive) restarts us.
                    # Refuse new jobs from here on: one whose request is still arriving would
                    # otherwise be accepted during the graceful shutdown and then cut short.
                    jobs.closing = True
                    log.info("restarting to release memory while idle")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                if not warned:
                    log.warning(".env or server code changed since startup; not restarting "
                                "automatically. Run scripts/install_image_server.sh to apply.")
                    warned = True

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        await asyncio.to_thread(jobs.restore)
        task = asyncio.create_task(unload_when_idle())
        try:
            yield
        finally:
            task.cancel()
            await jobs.shutdown()  # finish or record every accepted job before the engine stops
            engine.shutdown()

    app = FastAPI(
        title="teacher-image-harness image server",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,  # the contract is exported to docs/openapi.json instead
    )
    app.add_middleware(TrustedHosts, allowed_hosts=list(settings.allowed_hosts))
    if settings.private_clients_only:
        app.add_middleware(PrivateNetworkOnly)
    auth = [Depends(require_token)]
    busy = {"Retry-After": "30"}
    decode_slots = asyncio.Semaphore(DECODES_AT_ONCE)

    async def decode(data: bytes) -> Image.Image:
        async with decode_slots:
            return await asyncio.to_thread(load_reference_image, data)

    def save_output(png: bytes, record: dict) -> None:
        """Open WebUI's pictures, archived as in 1.0: by date, with the prompt, outside the
        job API's image store (so device tokens can't list other people's prompts)."""
        now = datetime.now()
        folder = settings.data_dir / "outputs" / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        stem = f"{now:%H%M%S}-{record['model']}-{record['seed']}-{secrets.token_hex(2)}"
        write_private(folder / f"{stem}.png", png)
        write_private(folder / f"{stem}.json", json.dumps(record, indent=2).encode())

    # --- OpenAI-compatible endpoints (synchronous; the Open WebUI pipe uses these) ------

    def resolve_model(requested: str | None, *, editing: bool) -> ModelSpec:
        # Unknown names (e.g. "dall-e-2" from Open WebUI's built-in engine) fall back to defaults.
        spec = models.get(requested) if requested else None
        if spec is None:
            spec = models[settings.edit_model if editing else settings.default_model]
        if editing and not spec.supports_edit:
            spec = models[settings.edit_model]
        return spec

    async def watch_disconnect(request: Request, job: Job) -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(1)
        job.cancel.set()

    async def generate(req: GenerationRequest, pictures: list[Image.Image], request: Request) -> dict:
        if req.response_format not in (None, "b64_json"):
            raise HTTPException(400, "only response_format=b64_json is supported")
        if req.n > settings.max_n:
            raise HTTPException(400, f"n must be at most {settings.max_n}")
        spec = resolve_model(req.model, editing=bool(pictures))
        if len(pictures) > spec.max_reference_images:
            raise HTTPException(
                400, f"{spec.id} accepts at most {spec.max_reference_images} reference images"
            )
        default = DEFAULT_SIZE
        if pictures:  # like OpenAI edits: keep the reference's shape unless told otherwise
            default = size_like(pictures[0], DEFAULT_SIZE[0] * DEFAULT_SIZE[1])
        try:
            width, height = parse_size(req.size, default, settings.max_pixels)
        except ImageInputError as exc:
            raise HTTPException(400, str(exc)) from None

        job = Job(
            model=spec,
            prompt=req.prompt,
            width=width,
            height=height,
            seed=req.seed if req.seed is not None else secrets.randbelow(2**31),
            n=req.n,
            steps=req.steps or spec.default_steps,
            guidance=req.guidance if req.guidance is not None else spec.default_guidance,
            references=pictures,
        )
        started = time.perf_counter()
        watcher = asyncio.create_task(watch_disconnect(request, job))
        try:
            results = await engine.run(job)
        except JobCancelled:
            log.info("client went away; dropped %s job", spec.id)
            raise HTTPException(499, "client closed the request") from None
        except EngineBusy:
            raise HTTPException(503, "the image server is busy; try again in a minute", headers=busy) from None
        except ValueError as exc:  # the model rejected the input
            raise HTTPException(400, str(exc)) from None
        except Exception:
            log.exception("generation failed (model=%s)", spec.id)
            raise HTTPException(500, "image generation failed; see the image server log") from None
        finally:
            watcher.cancel()
        log.info(
            "%s %dx%d n=%d refs=%d in %.1fs",
            spec.id, width, height, req.n, len(pictures), time.perf_counter() - started,
        )

        data = []
        for result in results:
            image = result.image
            if req.background == "transparent":
                image = remove_white_background(image)
            meta = {"model": spec.id, "seed": result.seed, "steps": job.steps,
                    "width": image.width, "height": image.height}
            png = to_png(image, meta)
            if settings.save_outputs:
                record = {**meta, "prompt": req.prompt, "references": len(pictures),
                          "transparent": req.background == "transparent",
                          "seconds": round(result.seconds, 2), "created_at": time.time()}
                try:  # the archive is a convenience; still return the picture
                    await asyncio.to_thread(save_output, png, record)
                except OSError as exc:
                    log.warning("could not archive output: %s", exc)
            data.append({"b64_json": base64.b64encode(png).decode(), "seed": result.seed,
                         "seconds": round(result.seconds, 2), "width": image.width, "height": image.height})
        return {"created": int(time.time()), "model": spec.id, "data": data}

    @app.post("/v1/images/generations", dependencies=auth, tags=["openai"],
              openapi_extra=_json_body(GenerationRequest))
    async def generations(request: Request) -> dict:
        """Text-to-image, waits for the result (OpenAI Images API shape)."""
        req = await read_json(request, GenerationRequest)
        return await generate(req, [], request)

    @app.post("/v1/images/edits", dependencies=auth, tags=["openai"],
              openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                  "type": "object", "required": ["prompt", "image"], "properties": {
                      "image": {"type": "string", "format": "binary",
                                "description": "one picture; send several as image[]"},
                      "image[]": {"type": "array", "items": {"type": "string", "format": "binary"}},
                      "prompt": {"type": "string"}, "model": {"type": "string"}, "n": {"type": "integer"},
                      "size": {"type": "string"}, "seed": {"type": "integer"}, "steps": {"type": "integer"},
                      "guidance": {"type": "number"}, "background": {"type": "string",
                                                                     "enum": ["transparent", "opaque", "auto"]},
                      "response_format": {"type": "string", "enum": ["b64_json"]}}}}}}})
    async def edits(request: Request) -> dict:
        """Edit with 1..N uploaded pictures ('image' / 'image[]'), waits for the result."""
        form = await read_form(request, settings.max_upload_bytes * MAX_FILES + 1024 * 1024)
        uploads = [u for u in form.getlist("image[]") + form.getlist("image")
                   if isinstance(u, UploadFile)]
        if not uploads:
            raise HTTPException(400, "send at least one image as 'image' or 'image[]'")
        fields = {k: v for k, v in form.items() if isinstance(v, str) and v != ""}
        try:
            req = GenerationRequest.model_validate(fields)
        except ValidationError as exc:
            raise _validation_error(exc) from None
        spec = resolve_model(req.model, editing=True)
        if len(uploads) > spec.max_reference_images:  # check before decoding anything
            raise HTTPException(
                400, f"{spec.id} accepts at most {spec.max_reference_images} reference images"
            )
        pictures = []
        for upload in uploads:
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "an uploaded image is too large")
            try:
                pictures.append(await decode(data))
            except ImageInputError as exc:
                raise HTTPException(400, str(exc)) from None
        return await generate(req, pictures, request)

    @app.get("/v1/models", dependencies=auth, tags=["openai"])
    async def list_models_openai() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": s.id, "object": "model", "owned_by": "local", "license": s.license,
                 "supports_edit": s.supports_edit,
                 "max_reference_images": s.max_reference_images,
                 "research_only": s.research_only,
                 "default": s.id == settings.default_model}
                for s in models.values()
            ],
        }

    @app.get("/v1/status", dependencies=auth, tags=["service"])
    async def status() -> dict:
        return {"loaded_model": engine.loaded_model, "pending": engine.pending,
                "last_load_seconds": engine.load_seconds, "pid": os.getpid()}

    @app.post("/v1/unload", dependencies=auth, tags=["service"])
    async def unload() -> dict:
        """Free the model's memory now (it reloads on the next job)."""
        was = engine.loaded_model
        await engine.unload()
        return {"unloaded": was}

    # --- job API -----------------------------------------------------------------------

    @app.post("/jobs", status_code=202, dependencies=auth, tags=["jobs"],
              openapi_extra=_json_body(JobRequest), responses={202: {"model": JobCreated}})
    async def create_job(request: Request) -> dict:
        """Queue a generation (or an edit, with source_image_id / reference_ids)."""
        req = await read_json(request, JobRequest)
        try:
            record = jobs.submit(req)
        except JobError as exc:
            raise HTTPException(400, str(exc)) from None
        except EngineClosed:
            raise HTTPException(503, "the image server is restarting; try again in a few seconds",
                                headers={"Retry-After": "5"}) from None
        except EngineBusy:
            raise HTTPException(503, "the image server is busy; try again in a minute", headers=busy) from None
        except JobNotRecorded as exc:
            raise HTTPException(503, str(exc), headers=busy) from None
        # pending already counts this job (its slot was reserved): report the ones ahead of it.
        return {"job_id": record.id, "status": record.status, "queue_depth": max(0, engine.pending - 1)}

    @app.get("/jobs", dependencies=auth, tags=["jobs"], responses={200: {"model": JobList}})
    async def list_jobs(limit: int = Query(20, ge=1, le=200)) -> dict:
        return {"jobs": await jobs.list(limit)}

    @app.get("/jobs/{job_id}", dependencies=auth, tags=["jobs"], responses={200: {"model": JobStatus}})
    async def get_job(job_id: str) -> dict:
        record = jobs.get(job_id)
        if record is None:
            raise HTTPException(404, "no such job")
        return record

    @app.delete("/jobs/{job_id}", dependencies=auth, tags=["jobs"], responses={200: {"model": JobStatus}})
    async def cancel_job(job_id: str) -> dict:
        """Cancel a queued or running job (it stops between denoising steps)."""
        record = jobs.cancel(job_id)
        if record is None:
            raise HTTPException(404, "no such job")
        return record

    # --- images ------------------------------------------------------------------------

    @app.post("/images", status_code=201, dependencies=auth, tags=["images"],
              responses={201: {"model": ImageUploaded}},
              openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {
                  "schema": {"type": "object", "required": ["file"], "properties": {
                      "file": {"type": "string", "format": "binary"}}}}}}})
    async def upload_image(request: Request) -> JSONResponse:
        """Store a picture to edit (use its image_id as a job's source_image_id)."""
        form = await read_form(request, settings.max_upload_bytes + 64 * 1024, max_files=1)
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(400, "send the picture as multipart field 'file'")
        data = await upload.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "the picture is too large")
        try:
            picture = await decode(data)
        except ImageInputError as exc:
            raise HTTPException(400, str(exc)) from None
        record = await asyncio.to_thread(images.save_upload, picture, upload.filename)
        return JSONResponse({"image_id": record["id"], "width": record["width"],
                             "height": record["height"]}, status_code=201)

    @app.get("/images", dependencies=auth, tags=["images"], responses={200: {"model": ImageList}})
    async def list_images(limit: int = Query(20, ge=1, le=200), before: float | None = None) -> dict:
        """Newest first; pass the last created_at as `before` to page."""
        return {"images": await asyncio.to_thread(images.list, limit, before)}

    @app.get("/images/{image_id}", dependencies=auth, tags=["images"],
             responses={200: {"content": {"image/png": {}, "image/webp": {}}}})
    async def get_image(
        image_id: str,
        size: Literal["preview", "full"] = "full",
        budget: int = Query(PREVIEW_BUDGET, ge=MIN_PREVIEW_BUDGET, le=PREVIEW_BUDGET,
                            description="preview only: largest size in bytes (smaller pictures for tight budgets)"),
    ) -> Response:
        """`full`: the original PNG. `preview`: WebP, ~768 px on the long edge (smaller to fit `budget`)."""
        if size == "full":
            data, media, ext = images.full(image_id), "image/png", "png"
        else:
            data, media, ext = await asyncio.to_thread(images.preview, image_id, budget), "image/webp", "webp"
        if data is None:
            raise HTTPException(404, "no such image")
        return Response(data, media_type=media, headers={
            "Cache-Control": "private, max-age=86400",
            "Content-Disposition": f'inline; filename="{image_id}.{ext}"'})

    @app.get("/images/{image_id}/meta", dependencies=auth, tags=["images"],
             responses={200: {"model": ImageMeta}})
    async def get_image_meta(image_id: str) -> dict:
        meta = images.meta(image_id)
        if meta is None:
            raise HTTPException(404, "no such image")
        return meta

    # --- references ----------------------------------------------------------------------

    @app.post("/references", status_code=201, dependencies=auth, tags=["references"],
              responses={201: {"model": ReferenceRecord}},
              openapi_extra={"requestBody": {"required": True, "content": {
                  "multipart/form-data": {"schema": {"type": "object", "required": ["file"], "properties": {
                      "file": {"type": "string", "format": "binary"},
                      "name": {"type": "string"},
                      "tags": {"type": "string", "description": "comma-separated"}}}},
                  "application/json": {"schema": ReferenceFromImage.model_json_schema()}}}})
    async def add_reference(request: Request) -> JSONResponse:
        """Add a character sheet / style plate: upload a file, or promote a generated image."""
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await read_json(request, ReferenceFromImage)
            data = images.full(body.from_image_id)
            if data is None:
                raise HTTPException(404, f"no image {body.from_image_id!r}")
            name, tags, source = body.name, body.tags, f"image:{body.from_image_id}"
        else:
            form = await read_form(request, settings.max_upload_bytes + 64 * 1024, max_files=1)
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "send the picture as multipart field 'file'")
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "the picture is too large")
            name = str(form.get("name") or upload.filename or "reference")
            tags, source = str(form.get("tags") or ""), "upload"
        try:
            picture = await decode(data)
        except ImageInputError as exc:
            raise HTTPException(400, str(exc)) from None
        record = await asyncio.to_thread(references.save, picture, name, tags, source)
        return JSONResponse(record, status_code=201)

    @app.get("/references", dependencies=auth, tags=["references"], responses={200: {"model": ReferenceList}})
    async def list_references(tag: str | None = None) -> dict:
        return {"references": await asyncio.to_thread(references.list, tag)}

    @app.get("/references/{reference_id}", dependencies=auth, tags=["references"],
             responses={200: {"model": ReferenceRecord}})
    async def get_reference(reference_id: str) -> dict:
        meta = references.meta(reference_id)
        if meta is None:
            raise HTTPException(404, "no such reference")
        return meta

    @app.get("/references/{reference_id}/image", dependencies=auth, tags=["references"],
             responses={200: {"content": {"image/png": {}}}})
    async def get_reference_image(reference_id: str) -> Response:
        path = references.path(reference_id)
        if path is None:
            raise HTTPException(404, "no such reference")
        return Response(path.read_bytes(), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=86400"})

    @app.delete("/references/{reference_id}", status_code=204, dependencies=auth, tags=["references"])
    async def delete_reference(reference_id: str) -> Response:
        if not references.delete(reference_id):
            raise HTTPException(404, "no such reference")
        return Response(status_code=204)

    # --- service ---------------------------------------------------------------------------

    @app.get("/models", dependencies=auth, tags=["service"], responses={200: {"model": ModelList}})
    async def list_models() -> dict:
        return {"models": [
            {"id": s.id, "license": s.license, "supports_edit": s.supports_edit,
             "max_reference_images": s.max_reference_images, "default_steps": s.default_steps,
             "research_only": s.research_only, "default": s.id == settings.default_model,
             "edit_default": s.id == settings.edit_model, "supports_negative_prompt": False,
             "supports_transparent": True}
            for s in models.values()
        ]}

    @app.get("/health", tags=["service"], responses={200: {"model": Health}})
    async def health() -> dict:
        """Liveness and load. No auth (shows no prompts, images or secrets)."""
        return {"status": "ok", "version": __version__, "loaded_model": engine.loaded_model,
                "queue_depth": engine.pending, "jobs_active": jobs.active, "memory": engine.memory,
                "uptime_s": round(time.time() - STARTED, 1)}

    app.state.jobs, app.state.images, app.state.references = jobs, images, references
    return app
