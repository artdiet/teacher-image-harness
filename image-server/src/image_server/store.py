"""On-disk stores: generated images (with JSON sidecars) and reference pictures.

Layout under the data dir (outside the repo, mode 0700):
  images/img_<16 hex>.png        the picture as generated
  images/img_<16 hex>.json       sidecar: prompt, seed, model, params, references, timestamps
  images/img_<16 hex>.preview.webp   cached downscaled preview
  references/ref_<16 hex>.png    reference picture, normalized to what the model sees
  references/ref_<16 hex>.json   name, tags, size, source

Ids are random and validated against a strict pattern before touching the
filesystem, so a request can never name a path outside these folders.
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path

from PIL import Image

IMAGE_ID_RE = re.compile(r"^img_[0-9a-f]{16}$")
REFERENCE_ID_RE = re.compile(r"^ref_[0-9a-f]{16}$")
MAX_TAGS = 20
MAX_NAME = 120
PREVIEW_BUDGET = 70_000  # bytes; ~93k base64 chars, well under Claude Desktop's ~150k-char tool-result cap
MIN_PREVIEW_BUDGET = 8_000
PREVIEW_STEPS = ((768, 82), (768, 70), (640, 70), (512, 65), (384, 60), (320, 55), (256, 50))


def write_private(path: Path, data: bytes) -> None:
    """Atomic write, mode 600: readers never see a half-written file.

    The temporary file has a unique name, so two threads writing the same path (two
    requests for the same uncached preview) can't trip over each other's temp file.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # mode 600
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def clean_tags(tags) -> list[str]:
    if isinstance(tags, str):
        tags = tags.split(",")
    out = []
    for tag in tags or []:
        tag = " ".join(str(tag).split())[:40]
        if tag and tag not in out:
            out.append(tag)
    return out[:MAX_TAGS]


class ImageStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "images"

    def _ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def path(self, image_id: str) -> Path | None:
        if not IMAGE_ID_RE.fullmatch(image_id or ""):
            return None
        path = self.root / f"{image_id}.png"
        return path if path.is_file() else None

    def save(self, png: bytes, meta: dict) -> str:
        self._ensure()
        image_id = "img_" + secrets.token_hex(8)
        record = {"id": image_id, "created_at": time.time(), **meta}
        write_private(self.root / f"{image_id}.png", png)
        write_private(self.root / f"{image_id}.json", json.dumps(record, indent=2).encode())
        return image_id

    def save_upload(self, image: Image.Image, filename: str | None) -> dict:
        """Store a picture sent by a client (to edit or use as a source)."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        name = os.path.basename(filename or "")[:120] or None
        meta = {"source": "upload", "filename": name, "width": image.width, "height": image.height}
        image_id = self.save(buf.getvalue(), meta)
        return {"id": image_id, **meta}

    def meta(self, image_id: str) -> dict | None:
        if self.path(image_id) is None:
            return None
        return read_json(self.root / f"{image_id}.json")

    def full(self, image_id: str) -> bytes | None:
        path = self.path(image_id)
        return path.read_bytes() if path else None

    def preview(self, image_id: str, budget: int = PREVIEW_BUDGET) -> bytes | None:
        """~768 px long edge WebP, smaller if needed to fit `budget` bytes.

        The default-budget preview is cached next to the original; smaller budgets
        (several previews in one tool result) are made on request."""
        path = self.path(image_id)
        if path is None:
            return None
        if budget != PREVIEW_BUDGET:
            return make_preview(path.read_bytes(), budget)
        cached = self.root / f"{image_id}.preview.webp"
        if cached.is_file():
            return cached.read_bytes()
        data = make_preview(path.read_bytes())
        write_private(cached, data)
        return data

    def list(self, limit: int = 20, before: float | None = None) -> list[dict]:
        if not self.root.is_dir():
            return []
        records = []
        for sidecar in self.root.glob("img_*.json"):
            meta = read_json(sidecar)
            if meta and (before is None or meta.get("created_at", 0) < before):
                records.append(meta)
        records.sort(key=lambda m: m.get("created_at", 0), reverse=True)
        return records[:limit]


class ReferenceStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "references"

    def _ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def path(self, reference_id: str) -> Path | None:
        if not REFERENCE_ID_RE.fullmatch(reference_id or ""):
            return None
        path = self.root / f"{reference_id}.png"
        return path if path.is_file() else None

    def save(self, image: Image.Image, name: str, tags, source: str) -> dict:
        self._ensure()
        reference_id = "ref_" + secrets.token_hex(8)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        record = {
            "id": reference_id,
            "name": (" ".join(str(name).split())[:MAX_NAME]) or reference_id,
            "tags": clean_tags(tags),
            "width": image.width,
            "height": image.height,
            "source": source,
            "created_at": time.time(),
        }
        write_private(self.root / f"{reference_id}.png", buf.getvalue())
        write_private(self.root / f"{reference_id}.json", json.dumps(record, indent=2).encode())
        return record

    def meta(self, reference_id: str) -> dict | None:
        if self.path(reference_id) is None:
            return None
        return read_json(self.root / f"{reference_id}.json")

    def image(self, reference_id: str) -> Image.Image | None:
        path = self.path(reference_id)
        if path is None:
            return None
        with Image.open(path) as img:
            return img.convert("RGB")

    def delete(self, reference_id: str) -> bool:
        path = self.path(reference_id)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        (self.root / f"{reference_id}.json").unlink(missing_ok=True)
        return True

    def list(self, tag: str | None = None) -> list[dict]:
        if not self.root.is_dir():
            return []
        records = [m for m in (read_json(p) for p in self.root.glob("ref_*.json")) if m]
        if tag:
            records = [m for m in records if tag in m.get("tags", [])]
        records.sort(key=lambda m: m.get("created_at", 0), reverse=True)
        return records



def make_preview(png: bytes, budget: int = PREVIEW_BUDGET) -> bytes:
    """Downscaled WebP (keeps transparency), stepping size and quality down to fit the budget."""
    with Image.open(io.BytesIO(png)) as original:
        original.load()
        if original.mode not in ("RGB", "RGBA"):
            original = original.convert("RGBA" if "A" in original.getbands() else "RGB")
        data = b""
        for long_edge, quality in PREVIEW_STEPS:
            img = original
            scale = long_edge / max(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)),
                                 Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=quality, method=4)
            data = buf.getvalue()
            if len(data) <= budget:
                break
        return data
