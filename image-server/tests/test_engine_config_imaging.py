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


def test_large_uploads_are_shrunk_before_conversion_and_keep_orientation():
    # 25 MP RGBA with a transparent border: still flattened onto white after the early shrink.
    image = Image.new("RGBA", (5000, 5000), (0, 0, 0, 0))
    image.paste((20, 120, 200, 255), (1000, 1000, 4000, 4000))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    out = load_reference_image(buf.getvalue())
    assert out.mode == "RGB" and out.width * out.height <= 1024 * 1024
    assert out.getpixel((2, 2)) == (255, 255, 255) and out.getpixel((out.width // 2,) * 2) == (20, 120, 200)
    # A big phone photo stored landscape with EXIF "rotate 90": comes out portrait.
    exif = Image.Exif()
    exif[0x0112] = 6
    for fmt in ("JPEG", "PNG"):
        buf = io.BytesIO()
        Image.new("RGB", (6000, 4500), (200, 30, 30)).save(buf, format=fmt, exif=exif.tobytes())
        out = load_reference_image(buf.getvalue())
        assert out.height > out.width, fmt


def test_cancelled_request_stops_its_picture_and_keeps_the_gpu_turn_until_then():
    """When a request task is cancelled mid-picture (client gone, server stopping), the
    job must stop at its next step, and the next job must not start before it has."""
    import threading
    import time

    from image_server.engine import JobCancelled

    log = []

    class Stoppable(FakeBackend):
        def generate(self, **kwargs):
            log.append(("start", kwargs["prompt"]))
            if kwargs["prompt"] == "long":
                while not kwargs["should_stop"]():
                    time.sleep(0.01)
                time.sleep(0.2)  # the last step finishes before the hook notices
                log.append(("stopped", "long"))
                raise JobCancelled("stopped")
            return super().generate(**kwargs)

    async def scenario():
        engine = Engine(Stoppable, release_memory=lambda: None)
        spec = MODELS["flux2-klein-4b"]
        long_job = Job(model=spec, prompt="long", width=256, height=256, seed=1)
        first = asyncio.create_task(engine.run(long_job))
        while not log:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.run(Job(model=spec, prompt="next", width=256, height=256, seed=2)))
        await asyncio.sleep(0.05)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert long_job.cancel.is_set()
        await second
        engine.shutdown()

    asyncio.run(scenario())
    assert log == [("start", "long"), ("stopped", "long"), ("start", "next")]


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
