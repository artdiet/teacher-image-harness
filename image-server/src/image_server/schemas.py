"""Response shapes for the job API, used to document the contract (docs/openapi.json).

Routes return plain dicts; these models describe them for the OpenAPI export.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JobParams(BaseModel):
    prompt: str = Field(description="As given by the client")
    style: Literal["icon", "coloring", "storybook", "none"]
    full_prompt: str = Field(description="What the model received (prompt wrapped in the style template)")
    negative_prompt: str | None
    model: str
    reference_ids: list[str]
    source_image_id: str | None
    width: int
    height: int
    seed: int = Field(description="Seed of the first image; image i uses seed + i")
    steps: int | None
    guidance: float | None
    n: int
    transparent: bool


class JobImage(BaseModel):
    image_id: str
    seed: int
    width: int
    height: int
    seconds: float


class JobResult(BaseModel):
    image_id: str = Field(description="First (or only) image")
    seed: int
    images: list[JobImage]
    model: str
    params: JobParams
    ignored: list[str] | None = Field(default=None, description="Requested options the model doesn't use")


class JobCreated(BaseModel):
    job_id: str
    status: Literal["queued"]
    queue_depth: int


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float = Field(description="0.0 to 1.0 across all requested images")
    params: JobParams
    created_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None = None
    result: JobResult | None = None


class JobList(BaseModel):
    jobs: list[JobStatus]


class ImageMeta(BaseModel):
    """Sidecar of a stored image. Generated images carry the job params; uploads carry source/filename."""

    id: str
    created_at: float
    source: str | None = Field(default=None, description="'upload' for pictures sent by a client")
    filename: str | None = None
    prompt: str | None = None
    style: str | None = None
    full_prompt: str | None = None
    negative_prompt: str | None = None
    model: str | None = None
    reference_ids: list[str] | None = None
    source_image_id: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    transparent: bool | None = None
    job_id: str | None = None
    seconds: float | None = None
    license: str | None = None


class ImageUploaded(BaseModel):
    image_id: str
    width: int
    height: int


class ImageList(BaseModel):
    images: list[ImageMeta]


class ReferenceRecord(BaseModel):
    id: str
    name: str
    tags: list[str]
    width: int
    height: int
    source: str = Field(description="'upload' or 'image:<image_id>'")
    created_at: float


class ReferenceList(BaseModel):
    references: list[ReferenceRecord]


class ReferenceFromImage(BaseModel):
    """JSON body for POST /references: promote a generated image to a reference."""

    from_image_id: str
    name: str = Field(min_length=1, max_length=120)
    tags: list[str] | str = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    license: str
    supports_edit: bool
    max_reference_images: int
    default_steps: int | None
    research_only: bool
    default: bool = Field(description="Used for text-to-image when no model is given")
    edit_default: bool = Field(description="Used when references or a source image are given")
    supports_negative_prompt: bool
    supports_transparent: bool


class ModelList(BaseModel):
    models: list[ModelInfo]


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    loaded_model: str | None
    queue_depth: int
    jobs_active: int
    memory: dict = Field(description="GPU memory in GB after the last job, once a model has run")
    uptime_s: float
