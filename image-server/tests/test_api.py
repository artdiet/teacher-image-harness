from __future__ import annotations

import base64
import io
import json

from PIL import Image

from conftest import TOKEN, FakeBackend, png_bytes
from image_server.config import Settings


def decode(item: dict) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))


def test_health_needs_no_token(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_api_requires_token(client):
    body = {"prompt": "a frog"}
    assert client.post("/v1/images/generations", json=body).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.post("/v1/images/generations", json=body, headers=bad).status_code == 401
    assert client.get("/v1/models").status_code == 401


def test_untrusted_host_is_rejected(make_client, auth):
    evil = make_client(base_url="http://attacker.example")
    assert evil.post("/v1/images/generations", json={"prompt": "x"}, headers=auth).status_code == 400


def test_no_cors_headers(client, auth):
    r = client.options(
        "/v1/images/generations",
        headers={"Origin": "http://attacker.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


def test_generation_returns_png_with_settings_but_not_prompt(client, auth):
    r = client.post(
        "/v1/images/generations", json={"prompt": "secret student name", "seed": 7}, headers=auth
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "flux2-klein-4b"
    image = decode(body["data"][0])
    assert image.format == "PNG" and image.size == (1024, 1024)
    params = json.loads(image.info["parameters"])
    assert params["seed"] == 7 and params["model"] == "flux2-klein-4b"
    assert "secret" not in image.info["parameters"]


def test_unknown_model_falls_back_to_default(client, auth):
    r = client.post("/v1/images/generations", json={"prompt": "x", "model": "dall-e-2"}, headers=auth)
    assert r.json()["model"] == "flux2-klein-4b"


def test_explicit_model_and_n_give_consecutive_seeds(client, auth):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "x", "model": "z-image-turbo", "n": 3, "seed": 100},
        headers=auth,
    )
    body = r.json()
    assert body["model"] == "z-image-turbo"
    assert [d["seed"] for d in body["data"]] == [100, 101, 102]


def test_limits_and_unsupported_options(client, auth):
    post = lambda body: client.post("/v1/images/generations", json=body, headers=auth)  # noqa: E731
    assert post({"prompt": "x", "n": 5}).status_code == 400
    assert post({"prompt": "x", "response_format": "url"}).status_code == 400
    assert post({"prompt": "x", "size": "huge"}).status_code == 400
    assert post({"prompt": "x", "size": "64x64"}).status_code == 400
    assert post({"prompt": ""}).status_code == 422


def test_size_is_respected_and_capped(client, auth):
    r = client.post("/v1/images/generations", json={"prompt": "x", "size": "1024x1536"}, headers=auth)
    assert decode(r.json()["data"][0]).size == (1024, 1536)
    r = client.post("/v1/images/generations", json={"prompt": "x", "size": "2048x2048"}, headers=auth)
    w, h = decode(r.json()["data"][0]).size
    assert w * h <= 1536 * 1024 and w % 16 == 0 and h % 16 == 0


def test_transparent_background(client, auth):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "icon", "background": "transparent", "size": "512x512"},
        headers=auth,
    )
    image = decode(r.json()["data"][0])
    assert image.mode == "RGBA"
    assert image.getpixel((5, 5))[3] == 0  # white border removed
    assert image.getpixel((256, 256))[3] == 255  # subject kept


def test_edits_accept_image_and_image_array(client, auth):
    files = [
        ("image", ("a.png", png_bytes(), "image/png")),
        ("image[]", ("b.png", png_bytes(), "image/png")),
    ]
    r = client.post(
        "/v1/images/edits", data={"prompt": "make it blue", "n": "1"}, files=files, headers=auth
    )
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "flux2-klein-4b"
    assert len(FakeBackend.instances[-1].calls[-1]["references"]) == 2


def test_edit_with_text_only_model_uses_edit_model(client, auth):
    files = [("image", ("a.png", png_bytes(), "image/png"))]
    r = client.post(
        "/v1/images/edits", data={"prompt": "x", "model": "z-image-turbo"}, files=files, headers=auth
    )
    assert r.json()["model"] == "flux2-klein-4b"


def test_edit_validation(client, auth):
    assert client.post("/v1/images/edits", data={"prompt": "x"}, headers=auth).status_code == 400
    junk = [("image", ("a.png", b"not an image", "image/png"))]
    r = client.post("/v1/images/edits", data={"prompt": "x"}, files=junk, headers=auth)
    assert r.status_code == 400
    too_many = [("image[]", (f"{i}.png", png_bytes(), "image/png")) for i in range(5)]
    r = client.post("/v1/images/edits", data={"prompt": "x"}, files=too_many, headers=auth)
    assert r.status_code == 400 and "at most 4" in r.text


def test_edit_requires_token_before_reading_body(client):
    files = [("image", ("a.png", png_bytes(), "image/png"))]
    assert client.post("/v1/images/edits", data={"prompt": "x"}, files=files).status_code == 401


def test_outputs_are_archived_with_prompt(client, auth, settings):
    client.post("/v1/images/generations", json={"prompt": "owl icon", "seed": 3}, headers=auth)
    pngs = list((settings.data_dir / "outputs").rglob("*.png"))
    records = list((settings.data_dir / "outputs").rglob("*.json"))
    assert len(pngs) == 1 and len(records) == 1
    assert json.loads(records[0].read_text())["prompt"] == "owl icon"


def test_research_models_hidden_unless_enabled(make_client, auth, settings):
    ids = {m["id"] for m in make_client().get("/v1/models", headers=auth).json()["data"]}
    assert "qwen-image-2.1-turbo" not in ids
    enabled = Settings(**{**settings.__dict__, "enable_research_models": True})
    ids = {m["id"] for m in make_client(enabled).get("/v1/models", headers=auth).json()["data"]}
    assert "qwen-image-2.1-turbo" in ids


def test_token_constant_is_long_enough():
    assert not Settings(token=TOKEN).validate()
