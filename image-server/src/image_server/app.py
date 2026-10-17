"""FastAPI app: a local, OpenAI-compatible slice of the Images API.

Routes (all but /health need the bearer token):
  GET  /health                  liveness, no details
  GET  /v1/models               models this server can run
  GET  /v1/status               loaded model, queue depth
  POST /v1/unload               free the model's memory now
  POST /v1/images/generations   JSON body, text-to-image
  POST /v1/images/edits         multipart, prompt + 1..N reference images ("image" or "image[]")

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

from fastapi import Depends, FastAPI, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import ModelSpec, Settings
from .engine import Engine, EngineBusy, Job, JobCancelled
from .imaging import (
    ImageInputError,
    load_reference_image,
    parse_size,
    remove_white_background,
    size_like,
    to_png,
)
from .security import bearer_token_dependency

log = logging.getLogger(__name__)

DEFAULT_SIZE = (1024, 1024)
MAX_PROMPT_CHARS = 4000
MAX_JSON_BODY = 64 * 1024  # a 4000-character prompt is well under this
MAX_FILES = 16


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


def create_app(
    settings: Settings,
    engine: Engine,
    restart_allowed: Callable[[], bool] = lambda: True,
) -> FastAPI:
    """restart_allowed() says whether a restart would come back with the same .env and
    code; if not, an idle restart is skipped so a bad edit can't become a crash loop."""
    models = settings.enabled_models()
    require_token = bearer_token_dependency(settings.token)

    async def unload_when_idle() -> None:
        warned = False
        while True:
            await asyncio.sleep(30)
            try:
                await engine.unload_if_idle(settings.idle_unload_seconds)
            except Exception:
                log.exception("idle unload failed")
                continue
            if settings.restart_when_idle and engine.should_restart(settings.idle_unload_seconds):
                if restart_allowed():
                    # Imported ML libraries keep ~0.4 GB even with no model loaded; a fresh
                    # process idles at a fraction of that. launchd (KeepAlive) restarts us.
                    log.info("restarting to release memory while idle")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                if not warned:
                    log.warning(".env or server code changed since startup; not restarting "
                                "automatically. Run scripts/install_image_server.sh to apply.")
                    warned = True

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(unload_when_idle())
        try:
            yield
        finally:
            task.cancel()
            engine.shutdown()

    app = FastAPI(
        title="teacher-image-harness image server",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

    def resolve_model(requested: str | None, *, editing: bool) -> ModelSpec:
        # Unknown names (e.g. "dall-e-2" from Open WebUI's built-in engine) fall back to defaults.
        spec = models.get(requested) if requested else None
        if spec is None:
            spec = models[settings.edit_model if editing else settings.default_model]
        if editing and not spec.supports_edit:
            spec = models[settings.edit_model]
        return spec

    def save_output(png: bytes, record: dict) -> None:
        now = datetime.now()
        folder = settings.data_dir / "outputs" / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        stem = f"{now:%H%M%S}-{record['model']}-{record['seed']}"
        (folder / f"{stem}.png").write_bytes(png)
        (folder / f"{stem}.json").write_text(json.dumps(record, indent=2))

    async def watch_disconnect(request: Request, job: Job) -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(1)
        job.cancel.set()

    async def generate(req: GenerationRequest, references: list[Image.Image], request: Request) -> dict:
        if req.response_format not in (None, "b64_json"):
            raise HTTPException(400, "only response_format=b64_json is supported")
        if req.n > settings.max_n:
            raise HTTPException(400, f"n must be at most {settings.max_n}")
        spec = resolve_model(req.model, editing=bool(references))
        if len(references) > spec.max_reference_images:
            raise HTTPException(
                400, f"{spec.id} accepts at most {spec.max_reference_images} reference images"
            )
        default = DEFAULT_SIZE
        if references:  # like OpenAI edits: keep the reference's shape unless told otherwise
            default = size_like(references[0], DEFAULT_SIZE[0] * DEFAULT_SIZE[1])
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
            references=references,
        )
        started = time.perf_counter()
        watcher = asyncio.create_task(watch_disconnect(request, job))
        try:
            results = await engine.run(job)
        except JobCancelled:
            log.info("client went away; dropped %s job", spec.id)
            raise HTTPException(499, "client closed the request") from None
        except EngineBusy:
            raise HTTPException(503, "the image server is busy; try again in a minute") from None
        except ValueError as exc:  # the model rejected the input
            raise HTTPException(400, str(exc)) from None
        except Exception:
            log.exception("generation failed (model=%s)", spec.id)
            raise HTTPException(500, "image generation failed; see the image server log") from None
        finally:
            watcher.cancel()
        log.info(
            "%s %dx%d n=%d refs=%d in %.1fs",
            spec.id, width, height, req.n, len(references), time.perf_counter() - started,
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
                record = {**meta, "prompt": req.prompt, "references": len(references),
                          "seconds": round(result.seconds, 2)}
                try:
                    await asyncio.to_thread(save_output, png, record)
                except OSError as exc:  # the archive is a convenience; still return the picture
                    log.warning("could not archive output: %s", exc)
            data.append({"b64_json": base64.b64encode(png).decode(), "seed": result.seed,
                         "seconds": round(result.seconds, 2),
                         "width": image.width, "height": image.height})
        return {"created": int(time.time()), "model": spec.id, "data": data}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(require_token)])
    async def list_models() -> dict:
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

    @app.get("/v1/status", dependencies=[Depends(require_token)])
    async def status() -> dict:
        return {"loaded_model": engine.loaded_model, "pending": engine.pending,
                "last_load_seconds": engine.load_seconds, "pid": os.getpid()}

    @app.post("/v1/unload", dependencies=[Depends(require_token)])
    async def unload() -> dict:
        was = engine.loaded_model
        await engine.unload()
        return {"unloaded": was}

    @app.post("/v1/images/generations", dependencies=[Depends(require_token)])
    async def generations(request: Request) -> dict:
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_JSON_BODY:
                raise HTTPException(413, "request body too large")
        try:
            req = GenerationRequest.model_validate_json(bytes(body))
        except ValidationError as exc:
            raise _validation_error(exc) from None
        return await generate(req, [], request)

    @app.post("/v1/images/edits", dependencies=[Depends(require_token)])
    async def edits(request: Request) -> dict:
        limit = settings.max_upload_bytes * MAX_FILES + 1024 * 1024
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise HTTPException(413, "request body too large")
        received = 0

        async def capped_receive():  # also caps chunked bodies, which have no Content-Length
            nonlocal received
            message = await request.receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, "request body too large")
            return message

        form = await Request(request.scope, capped_receive).form(max_files=MAX_FILES, max_fields=32)
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
        references = []
        for upload in uploads:
            data = await upload.read(settings.max_upload_bytes + 1)
            if len(data) > settings.max_upload_bytes:
                raise HTTPException(413, "an uploaded image is too large")
            try:
                references.append(await asyncio.to_thread(load_reference_image, data))
            except ImageInputError as exc:
                raise HTTPException(400, str(exc)) from None
        return await generate(req, references, request)

    return app
