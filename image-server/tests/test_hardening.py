"""Behaviors added after the code review: auth before body, cancellation, input checks."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading

import numpy as np
import pytest
from PIL import Image

from conftest import TOKEN, FakeBackend, png_bytes
from image_server.app import create_app
from image_server.config import MODELS, Settings
from image_server.engine import Engine, Job, JobCancelled
from image_server.imaging import ImageInputError, load_reference_image


def decode(item: dict) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))


def test_generations_rejects_bad_token_before_reading_the_body(settings):
    app = create_app(settings, Engine(FakeBackend, release_memory=lambda: None))
    chunks_read = 0
    sent = []

    async def receive():
        nonlocal chunks_read
        chunks_read += 1
        return {"type": "http.request", "body": b"x" * 1024 * 1024, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": "/v1/images/generations", "raw_path": b"/v1/images/generations",
        "query_string": b"", "root_path": "", "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 7865),
        "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json")],
    }
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 401
    assert chunks_read == 0


def test_generations_body_is_capped(client, auth):
    r = client.post("/v1/images/generations", content=b"{" + b" " * 70_000 + b"}",
                    headers={**auth, "content-type": "application/json"})
    assert r.status_code == 413


def test_queued_job_is_dropped_when_client_leaves():
    engine = Engine(FakeBackend, release_memory=lambda: None)
    job = Job(model=MODELS["flux2-klein-4b"], prompt="x", width=256, height=256, seed=1)
    job.cancel.set()
    with pytest.raises(JobCancelled):
        asyncio.run(engine.run(job))
    assert FakeBackend.instances == []  # never even loaded
    engine.shutdown()


def test_cancel_between_images_keeps_model_loaded():
    class StopsAfterFirst(FakeBackend):
        def generate(self, **kwargs):
            image = super().generate(**kwargs)
            job.cancel.set()  # client leaves while the first image is drawn
            return image

    engine = Engine(StopsAfterFirst, release_memory=lambda: None)
    job = Job(model=MODELS["flux2-klein-4b"], prompt="x", width=256, height=256, seed=1, n=3)
    with pytest.raises(JobCancelled):
        asyncio.run(engine.run(job))
    assert len(StopsAfterFirst.instances[0].calls) == 1
    assert engine.loaded_model == "flux2-klein-4b"
    engine.shutdown()


def test_backend_receives_a_stop_check():
    engine = Engine(FakeBackend, release_memory=lambda: None)
    job = Job(model=MODELS["flux2-klein-4b"], prompt="x", width=256, height=256, seed=1)
    asyncio.run(engine.run(job))
    should_stop = FakeBackend.instances[0].calls[0]["should_stop"]
    assert should_stop() is False
    job.cancel.set()
    assert should_stop() is True
    engine.shutdown()


def test_input_the_model_rejects_is_a_400_and_keeps_the_model(make_client, auth, settings):
    class Picky(FakeBackend):
        def generate(self, **kwargs):
            raise ValueError("Reference image too small to condition on")

    engine = Engine(Picky, release_memory=lambda: None)
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings, engine), base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        r = client.post("/v1/images/generations", json={"prompt": "x"}, headers=auth)
        assert r.status_code == 400 and "too small" in r.text
        assert engine.loaded_model == "flux2-klein-4b"


def test_failed_archive_still_returns_the_picture(make_client, auth, tmp_path, settings):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    broken = Settings(**{**settings.__dict__, "data_dir": blocker})
    r = make_client(broken).post("/v1/images/generations", json={"prompt": "x"}, headers=auth)
    assert r.status_code == 200 and decode(r.json()["data"][0]).size == (1024, 1024)


def test_metadata_records_steps_and_real_size(client, auth):
    body = client.post("/v1/images/generations", json={"prompt": "x"}, headers=auth).json()
    params = json.loads(decode(body["data"][0]).info["parameters"])
    assert params["steps"] == 4 and (params["width"], params["height"]) == (1024, 1024)
    assert (body["data"][0]["width"], body["data"][0]["height"]) == (1024, 1024)


def test_edit_output_follows_reference_shape_unless_size_given(client, auth):
    buf = io.BytesIO()
    Image.new("RGB", (900, 1200), "white").save(buf, format="PNG")
    files = [("image", ("p.png", buf.getvalue(), "image/png"))]
    r = client.post("/v1/images/edits", data={"prompt": "x"}, files=files, headers=auth)
    w, h = decode(r.json()["data"][0]).size
    assert h > w and w % 16 == 0 and abs(w / h - 0.75) < 0.02
    r = client.post("/v1/images/edits", data={"prompt": "x", "size": "1024x1024"}, files=files, headers=auth)
    assert decode(r.json()["data"][0]).size == (1024, 1024)


def test_unload_endpoint(client, auth):
    client.post("/v1/images/generations", json={"prompt": "x"}, headers=auth)
    assert client.post("/v1/unload", headers=auth).json() == {"unloaded": "flux2-klein-4b"}
    assert client.get("/v1/status", headers=auth).json()["loaded_model"] is None
    assert client.post("/v1/unload").status_code == 401


def png_with(image: Image.Image, **save_kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", **save_kwargs)
    return buf.getvalue()


def test_colour_key_transparency_becomes_white():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    img.paste((255, 0, 0), (16, 16, 48, 48))
    out = load_reference_image(png_with(img, transparency=(0, 0, 0)))
    assert out.getpixel((2, 2)) == (255, 255, 255) and out.getpixel((32, 32)) == (255, 0, 0)


def test_sixteen_bit_grayscale_keeps_its_tones():
    arr = np.full((64, 64), 32768, dtype=np.uint16)  # mid-gray in 16-bit
    out = load_reference_image(png_with(Image.fromarray(arr)))
    assert 120 <= out.getpixel((10, 10))[0] <= 135


def test_reference_limits():
    with pytest.raises(ImageInputError, match="at least 16"):
        load_reference_image(png_with(Image.new("RGB", (1000, 8), "white")))
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="TIFF")
    with pytest.raises(ImageInputError, match="readable"):
        load_reference_image(buf.getvalue())
    bomb = png_with(Image.new("1", (8000, 7000)))  # 56 MP, tiny file
    with pytest.raises(ImageInputError, match="too many pixels"):
        load_reference_image(bomb)


def test_settings_errors_are_friendly():
    with pytest.raises(ValueError, match="IMAGE_SERVER_MAX_PIXELS must be a whole number"):
        Settings.from_env({"IMAGE_SERVER_TOKEN": "x" * 32, "IMAGE_SERVER_MAX_PIXELS": "1536*1024"})
    problems = Settings(token="x" * 32, port=0, max_pixels=10).validate()
    assert any("PORT" in p for p in problems) and any("MAX_PIXELS" in p for p in problems)


def test_engine_queue_frees_ollama_right_before_each_job():
    order = []

    class Slow(FakeBackend):
        def generate(self, **kwargs):
            order.append(f"draw {kwargs['prompt']}")
            threading.Event().wait(0.05)
            return super().generate(**kwargs)

    async def hook():
        order.append("free")

    engine = Engine(Slow, before_job=hook, release_memory=lambda: None)

    async def scenario():
        job = lambda p: Job(model=MODELS["flux2-klein-4b"], prompt=p, width=256, height=256, seed=1)  # noqa: E731
        await asyncio.gather(engine.run(job("a")), engine.run(job("b")))

    asyncio.run(scenario())
    assert order == ["free", "draw a", "free", "draw b"]
    engine.shutdown()


def test_reference_count_checked_before_decoding(client, auth):
    junk = [("image[]", (f"{i}.png", b"not an image", "image/png")) for i in range(5)]
    r = client.post("/v1/images/edits", data={"prompt": "x"}, files=junk, headers=auth)
    assert r.status_code == 400 and "at most 4" in r.text  # not "not a readable image"


def test_png_bytes_helper_still_works():
    assert load_reference_image(png_bytes()).size == (64, 64)


def test_garbage_exif_is_ignored():
    for fmt in ("PNG", "WEBP"):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 200, 10)).save(buf, format=fmt, exif=b"Exif\x00\x00GARBAGEGARBAGE")
        assert load_reference_image(buf.getvalue()).size == (64, 48)


def test_chunked_edit_upload_is_capped(make_client, auth, settings):
    small = Settings(**{**settings.__dict__, "max_upload_bytes": 1024 * 1024})
    client = make_client(small)
    boundary = "xyz"

    def body():
        yield f'--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nx\r\n'.encode()
        yield (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="a.png"\r\n'
               f"Content-Type: image/png\r\n\r\n").encode()
        for _ in range(20):  # ~20 MB, well over 16 x 1 MB + 1 MB
            yield b"\0" * (1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    r = client.post("/v1/images/edits", content=body(),
                    headers={**auth, "content-type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 413


def test_restart_rule_needs_a_used_idle_process():
    engine = Engine(FakeBackend, release_memory=lambda: None)
    assert not engine.should_restart(0)  # fresh process: nothing to gain
    asyncio.run(engine.run(Job(model=MODELS["flux2-klein-4b"], prompt="x", width=256, height=256, seed=1)))
    assert not engine.should_restart(0)  # model still loaded
    asyncio.run(engine.unload())
    assert engine.should_restart(0) and not engine.should_restart(3600)
    engine.shutdown()


def test_fingerprint_changes_with_env_file(tmp_path):
    from image_server.__main__ import fingerprint

    env = tmp_path / ".env"
    env.write_text("IMAGE_SERVER_TOKEN=a\n")
    first = fingerprint(str(env))
    assert fingerprint(str(env)) == first
    env.write_text("IMAGE_SERVER_TOKEN=b\n")
    assert fingerprint(str(env)) != first
