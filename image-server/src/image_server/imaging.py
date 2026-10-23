"""Image helpers: size parsing, reference-image decoding, PNG encoding."""

from __future__ import annotations

import io
import json
import math
import warnings
from collections.abc import Mapping

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

SIZE_MULTIPLE = 16
MIN_SIDE = 256
MAX_SIDE = 2048
REFERENCE_MAX_PIXELS = 1024 * 1024  # bigger references cost memory without helping
INPUT_MAX_PIXELS = 50_000_000  # refuse to decode anything bigger (48 MP phone photos fit)
INPUT_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP")
HIGH_BIT_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


class ImageInputError(ValueError):
    """A request contained an unusable size or image."""


def _snap(value: float, *, down: bool = False) -> int:
    steps = value // SIZE_MULTIPLE if down else math.floor(value / SIZE_MULTIPLE + 0.5)
    return max(SIZE_MULTIPLE, int(steps) * SIZE_MULTIPLE)


def parse_size(size: str | None, default: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    """Turn "1024x1536" into (width, height), snapped to 16 px and capped at max_pixels."""
    if size is None or size.strip().lower() in {"", "auto"}:
        width, height = default
    else:
        w, sep, h = size.strip().lower().partition("x")
        try:
            if not sep:
                raise ValueError
            width, height = int(w), int(h)
        except ValueError:
            raise ImageInputError(f"size must look like 1024x1024, got {size!r}") from None
    if min(width, height) < MIN_SIDE or max(width, height) > MAX_SIDE:
        raise ImageInputError(f"each side must be between {MIN_SIDE} and {MAX_SIDE} px")
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        return _snap(width * scale, down=True), _snap(height * scale, down=True)
    return _snap(width), _snap(height)


def size_like(image: Image.Image, area: int) -> tuple[int, int]:
    """A width x height with the image's aspect ratio and roughly the given pixel area."""
    scale = (area / (image.width * image.height)) ** 0.5
    width = min(MAX_SIDE, max(MIN_SIDE, _snap(image.width * scale)))
    height = min(MAX_SIDE, max(MIN_SIDE, _snap(image.height * scale)))
    return width, height


def load_reference_image(data: bytes) -> Image.Image:
    """Decode an uploaded image into an RGB PIL image of at most ~1 MP."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=INPUT_FORMATS) as probe:
                pixels = probe.width * probe.height
                probe.verify()
            if pixels > INPUT_MAX_PIXELS:
                raise ImageInputError("an uploaded picture has too many pixels")
            image = Image.open(io.BytesIO(data), formats=INPUT_FORMATS)
            if image.format == "JPEG":
                # libjpeg can decode at 1/2, 1/4 or 1/8 scale: a 48 MP phone photo becomes a
                # few megapixels instead of ~150 MB of pixels (the result is ~1 MP anyway).
                image.draft("RGB", (1024, 1024))
            image.load()
    except ImageInputError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageInputError("an uploaded picture has too many pixels") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageInputError("an uploaded file is not a readable PNG, JPEG, WebP or GIF") from exc

    image = _shrink_early(image)  # before the conversions and exif_transpose, which copy it all
    if image.mode in HIGH_BIT_MODES:  # 16-bit / float grayscale
        values = np.asarray(image, dtype=np.float32)
        if image.mode.startswith("I;16") or values.max() > 255:
            values = values * (255.0 / 65535.0)
        image = _keep_info(Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)), image)
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # malformed EXIF (e.g. "not a TIFF file"): keep the stored orientation
        pass
    if image.has_transparency_data:  # alpha channel or a tRNS colour key: flatten onto white
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    if image.width * image.height > REFERENCE_MAX_PIXELS:
        scale = (REFERENCE_MAX_PIXELS / (image.width * image.height)) ** 0.5
        image = image.resize(
            (_snap(image.width * scale, down=True), _snap(image.height * scale, down=True)),
            Image.Resampling.LANCZOS,
        )
    if min(image.size) < SIZE_MULTIPLE:
        raise ImageInputError(f"each picture must be at least {SIZE_MULTIPLE} px on every side")
    return image


def _shrink_early(image: Image.Image) -> Image.Image:
    """Cheaply cut a very large picture down (integer box reduction) before the colour
    conversions below, which each copy the whole image: a 50 MP RGBA upload otherwise
    peaks around 1 GB. Keeps at least twice the final pixel count for the LANCZOS resize."""
    factor = int((image.width * image.height / (2 * REFERENCE_MAX_PIXELS)) ** 0.5)
    if factor < 2:
        return image
    original = image
    if image.mode in HIGH_BIT_MODES:  # keep the bit depth: converting first would clip it
        size = (image.width // factor, image.height // factor)
        return _keep_info(image.resize(size, Image.Resampling.BOX), original)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA" if image.has_transparency_data else "RGB")
    return _keep_info(image.reduce(factor), original)


def _keep_info(image: Image.Image, source: Image.Image) -> Image.Image:
    """Carry the metadata (EXIF orientation) over to a converted copy."""
    image.info = {**source.info, **image.info}
    return image


def remove_white_background(image: Image.Image, min_level: int = 235) -> Image.Image:
    """Make near-white background that touches the border transparent.

    Meant for icons generated on a plain white background. Interior white
    (eyes, highlights) stays opaque because only border-connected pixels go.
    """
    import cv2  # imported lazily: keeps the idle server small

    rgb = np.array(image.convert("RGB"))
    near_white = np.all(rgb >= min_level, axis=-1).astype(np.uint8)
    _, labels = cv2.connectedComponents(near_white, connectivity=4)
    edges = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    edge_labels = np.unique(edges[edges > 0])  # label 0 = not near-white
    background = np.isin(labels, edge_labels)
    alpha = np.where(background, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def to_png(image: Image.Image, metadata: Mapping[str, object]) -> bytes:
    """Encode as PNG with generation settings (never the prompt) in a text chunk."""
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "teacher-image-harness")
    info.add_text("parameters", json.dumps(dict(metadata), sort_keys=True))
    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()
