from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image

from conftest import FakeBackend, png_bytes
from image_server.config import MODELS, Settings
from image_server.engine import Engine, EngineBusy, Job
from image_server.imaging import ImageInputError, load_reference_image, parse_size


def job(model_id: str, **kw) -> Job:
    return Job(model=MODELS[model_id], prompt="x", width=256, height=256, seed=1, **kw)


def test_engine_keeps_one_model_and_swaps():
    released = []
    engine = Engine(FakeBackend, release_memory=lambda: released.append(1))

    async def scenario():
        await engine.run(job("flux2-klein-4b"))
        await engine.run(job("flux2-klein-4b"))
        await engine.run(job("z-image-turbo"))

    asyncio.run(scenario())
    first, second = FakeBackend.instances
    assert first.closed and not second.closed
    assert len(first.calls) == 2 and engine.loaded_model == "z-image-turbo"
    assert released == [1, 1, 1, 1]  # after each job + on swap
    engine.shutdown()
    assert second.closed


def test_engine_unloads_after_failure():
    class Boom(FakeBackend):
        def generate(self, **kwargs):
            raise RuntimeError("[METAL] out of memory")

    engine = Engine(Boom, release_memory=lambda: None)
    with pytest.raises(RuntimeError):
        asyncio.run(engine.run(job("flux2-klein-4b")))
    assert engine.loaded_model is None
    engine.shutdown()


def test_engine_idle_unload_and_busy():
    engine = Engine(FakeBackend, release_memory=lambda: None, max_pending=0)
    with pytest.raises(EngineBusy):
        asyncio.run(engine.run(job("flux2-klein-4b")))
    engine = Engine(FakeBackend, release_memory=lambda: None)

    async def scenario():
        await engine.run(job("flux2-klein-4b"))
        await engine.unload_if_idle(3600)
        assert engine.loaded_model == "flux2-klein-4b"
        await engine.unload_if_idle(0)
        assert engine.loaded_model is None

    asyncio.run(scenario())
    engine.shutdown()


def test_engine_runs_before_job_hook():
    calls = []

    async def hook():
        calls.append("freed")

    engine = Engine(FakeBackend, before_job=hook, release_memory=lambda: None)
    asyncio.run(engine.run(job("flux2-klein-4b")))
    assert calls == ["freed"]
    engine.shutdown()


@pytest.mark.parametrize(
    ("size", "expected"),
    [(None, (1024, 1024)), ("auto", (1024, 1024)), ("1000x1000", (1008, 1008)),
     ("768X1344", (768, 1344))],
)
def test_parse_size(size, expected):
    assert parse_size(size, (1024, 1024), 1536 * 1024) == expected


def test_parse_size_caps_pixels_and_rejects_junk():
    w, h = parse_size("2048x1024", (1024, 1024), 1024 * 1024)
    assert w * h <= 1024 * 1024 and w % 16 == 0 and h % 16 == 0 and w > h
    for bad in ("big", "1024", "x", "100x100", "4096x4096"):
        with pytest.raises(ImageInputError):
            parse_size(bad, (1024, 1024), 1536 * 1024)


def test_reference_images_are_rgb_and_downscaled():
    big = load_reference_image(png_bytes(size=(3000, 2000)))
    assert big.mode == "RGB" and big.width * big.height <= 1024 * 1024
    buf = io.BytesIO()
    Image.new("RGBA", (50, 50), (0, 0, 0, 0)).save(buf, format="PNG")
    flat = load_reference_image(buf.getvalue())
    assert flat.mode == "RGB" and flat.getpixel((0, 0)) == (255, 255, 255)


def test_settings_validation():
    assert Settings(token="").validate()
    assert any("too short" in p for p in Settings(token="short").validate())
    remote = Settings(token="x" * 32, host="0.0.0.0").validate()
    assert any("loopback" in p for p in remote)
    assert not Settings(token="x" * 32, host="0.0.0.0", allow_remote=True).validate()
    assert Settings(token="x" * 32, default_model="qwen-image-2.1-turbo").validate()
    assert not Settings(
        token="x" * 32, default_model="qwen-image-2.1-turbo", enable_research_models=True
    ).validate()
    assert Settings(token="x" * 32, edit_model="z-image-turbo").validate()


def test_settings_from_env():
    s = Settings.from_env({
        "IMAGE_SERVER_TOKEN": " abc ",
        "IMAGE_SERVER_PORT": "9000",
        "IMAGE_SERVER_FREE_OLLAMA": "no",
        "IMAGE_SERVER_ALLOWED_HOSTS": "127.0.0.1, host.docker.internal",
        "IMAGE_SERVER_DATA_DIR": "~/x",
    })
    assert s.token == "abc" and s.port == 9000 and s.free_ollama is False
    assert s.allowed_hosts == ("127.0.0.1", "host.docker.internal")
    assert str(s.data_dir).endswith("/x") and "~" not in str(s.data_dir)
