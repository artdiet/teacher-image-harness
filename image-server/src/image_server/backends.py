"""mflux backends, one per model family.

Everything here runs on the engine's single MLX thread. mflux and MLX are
imported lazily so the web app (and the tests) can start without them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import ModelSpec
from .engine import Backend, JobCancelled

log = logging.getLogger(__name__)

MLX_CACHE_LIMIT_BYTES = 2 * 1024**3


def init_mlx_thread() -> None:
    """Runs once on the MLX thread: cap MLX's cache of freed GPU buffers."""
    import mlx.core as mx

    mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)


class StopHook:
    """mflux step callback: reports progress and stops a cancelled job between steps.

    mflux has no unregister, so each loaded model gets one hook for its whole
    life and the current job's check is swapped in and out around each call.
    """

    def __init__(self) -> None:
        self.should_stop = None
        self.on_step = None

    def call_in_loop(self, **kwargs: object) -> None:
        if self.should_stop is not None and self.should_stop():
            raise JobCancelled("stopped between steps")
        total = getattr(kwargs.get("config"), "num_inference_steps", None)
        t = kwargs.get("t")
        if self.on_step is not None and isinstance(t, int) and total:
            self.on_step(t + 1, total)

    def attach(self, model) -> StopHook:
        model.callbacks.register(self)
        return self


def snapshot_dir(spec: ModelSpec) -> Path:
    """Local folder holding the pinned revision (fetched by scripts/download_models.sh)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(HF_HUB_CACHE) / f"models--{spec.repo.replace('/', '--')}" / "snapshots" / spec.revision
    if not folder.is_dir():
        raise RuntimeError(
            f"{spec.id}: weights for {spec.repo}@{spec.revision[:8]} are not downloaded; "
            "run scripts/download_models.sh"
        )
    return folder


class Flux2KleinBackend:
    """FLUX.2 [klein] 4B: text-to-image, plus editing with reference images.

    mflux has separate classes for the two (Flux2Klein / Flux2KleinEdit) built
    by the same initializer. Loading both would hold two ~8 GB copies, so the
    edit pipeline is a view that shares the loaded model's state.
    """

    default_steps = 4

    def __init__(self, spec: ModelSpec) -> None:
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein

        self._t2i = Flux2Klein(
            model_path=str(snapshot_dir(spec)),
            model_config=ModelConfig.flux2_klein_4b(),
            quantize=None,  # weights are pre-quantized (8-bit); the stored level is used
        )
        self._stop = StopHook().attach(self._t2i)  # the edit view shares this registry
        self._edit = None

    def _edit_view(self):
        if self._edit is None:
            from mflux.models.flux2.variants import Flux2KleinEdit
            from mlx import nn

            edit = Flux2KleinEdit.__new__(Flux2KleinEdit)
            nn.Module.__init__(edit)
            # nn.Module is a dict: copy its entries (sub-modules, dict attributes) without
            # going through Module.update(), which expects a parameter tree.
            dict.update(edit, self._t2i)
            edit.__dict__.update(self._t2i.__dict__)  # plain attributes (config, tokenizers, ...)
            self._edit = edit
        return self._edit

    def generate(self, *, prompt, seed, width, height, steps, guidance, references, should_stop=None, on_step=None):
        common = dict(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps or self.default_steps,
            width=width,
            height=height,
            guidance=1.0,  # distilled checkpoint: guidance is fixed
        )
        self._stop.should_stop, self._stop.on_step = should_stop, on_step
        try:
            if references:
                result = self._edit_view().generate_image(image_paths=list(references), **common)
            else:
                result = self._t2i.generate_image(**common)
        finally:
            self._stop.should_stop = self._stop.on_step = None
        return result.image

    def close(self) -> None:
        self._edit = None
        self._t2i = None


class ZImageTurboBackend:
    """Z-Image-Turbo: text-to-image only."""

    default_steps = 9

    def __init__(self, spec: ModelSpec) -> None:
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image import ZImageTurbo

        self._model = ZImageTurbo(
            model_path=str(snapshot_dir(spec)),
            model_config=ModelConfig.z_image_turbo(),
            quantize=None,
        )
        self._stop = StopHook().attach(self._model)

    def generate(self, *, prompt, seed, width, height, steps, guidance, references, should_stop=None, on_step=None):
        if references:
            raise ValueError("z-image-turbo cannot use reference images")
        self._stop.should_stop, self._stop.on_step = should_stop, on_step
        try:
            result = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps or self.default_steps,
                width=width,
                height=height,
            )
        finally:
            self._stop.should_stop = self._stop.on_step = None
        return result.image

    def close(self) -> None:
        self._model = None


class QwenImage21TurboBackend:
    """Qwen-Image-2.1-Turbo: text-to-image and edits with up to 10 references.

    QwenImage21Edit is the one class that covers both (and returns RGBA). The
    Turbo config must be passed explicitly or mflux runs the base schedule.
    """

    default_steps = 8
    reference_budget = 512  # px budget per reference; 1024 can exceed 24 GB

    def __init__(self, spec: ModelSpec) -> None:
        from mflux.models.common.config import ModelConfig
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

        self._model = QwenImage21Edit(
            model_path=str(snapshot_dir(spec)),
            model_config=ModelConfig.qwen_image_21_turbo(),
            quantize=None,
        )
        self._model.tiling_config = TilingConfig(vae_decode_tile_size=512)
        self._stop = StopHook().attach(self._model)

    def generate(self, *, prompt, seed, width, height, steps, guidance, references, should_stop=None, on_step=None):
        self._stop.should_stop, self._stop.on_step = should_stop, on_step
        try:
            result = self._model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps or self.default_steps,
                width=width // 32 * 32,  # Qwen needs multiples of 32
                height=height // 32 * 32,
                guidance=1.0,  # Turbo rejects anything else
                image_paths=list(references) or None,
                output_resolution=self.reference_budget,
            )
        finally:
            self._stop.should_stop = self._stop.on_step = None
            cache = getattr(self._model, "prompt_cache", None)
            if cache:
                cache.clear()  # grows without bound otherwise
        return result.image

    def close(self) -> None:
        self._model = None


BACKENDS = {
    "flux2-klein": Flux2KleinBackend,
    "z-image": ZImageTurboBackend,
    "qwen-image-2.1": QwenImage21TurboBackend,
}


def create_backend(spec: ModelSpec) -> Backend:
    return BACKENDS[spec.family](spec)
