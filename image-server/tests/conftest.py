from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from image_server.app import create_app
from image_server.config import ModelSpec, Settings
from image_server.engine import Engine

TOKEN = "test-token-0123456789abcdefghij"


class FakeBackend:
    """Draws a black square on white so tests can check size and transparency."""

    instances: list[FakeBackend] = []

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.calls: list[dict] = []
        self.closed = False
        FakeBackend.instances.append(self)

    def generate(self, **kwargs) -> Image.Image:
        self.calls.append(kwargs)
        for step in (1, 2):  # report progress like the mflux step hook
            if kwargs.get("on_step"):
                kwargs["on_step"](step, 2)
        w, h = kwargs["width"], kwargs["height"]
        image = Image.new("RGB", (w, h), (255, 255, 255))
        ImageDraw.Draw(image).rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(0, 0, 0))
        return image

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_backends():
    FakeBackend.instances.clear()
    yield
    FakeBackend.instances.clear()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(token=TOKEN, data_dir=tmp_path / "data", free_ollama=False)


@pytest.fixture
def make_client(settings):
    clients = []

    def _make(settings: Settings = settings, base_url: str = "http://127.0.0.1") -> TestClient:
        engine = Engine(FakeBackend, release_memory=lambda: None)
        client = TestClient(create_app(settings, engine), base_url=base_url, client=("127.0.0.1", 50000))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def png_bytes(size=(64, 64), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()
