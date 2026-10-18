"""The asynchronous job API: jobs, images, references, models, health."""

from __future__ import annotations

import io
import threading
import time

from PIL import Image

from conftest import FakeBackend, png_bytes
from image_server.config import Settings


def wait(client, auth, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=auth).json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {body}")


def submit(client, auth, **body) -> str:
    r = client.post("/jobs", json=body, headers=auth)
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "queued"
    return r.json()["job_id"]


def test_job_lifecycle_and_image_endpoints(client, auth):
    job_id = submit(client, auth, prompt="a cute owl icon", seed=7)
    body = wait(client, auth, job_id)
    assert body["status"] == "done" and body["progress"] == 1.0
    result = body["result"]
    assert result["seed"] == 7 and result["model"] == "flux2-klein-4b"
    assert result["params"]["width"] == 1024 and result["params"]["steps"] == 4
    image_id = result["image_id"]

    full = client.get(f"/images/{image_id}?size=full", headers=auth)
    assert full.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(full.content)).size == (1024, 1024)
    preview = client.get(f"/images/{image_id}?size=preview", headers=auth)
    assert preview.headers["content-type"] == "image/webp"
    assert max(Image.open(io.BytesIO(preview.content)).size) == 768

    meta = client.get(f"/images/{image_id}/meta", headers=auth).json()
    for key in ("prompt", "seed", "model", "width", "height", "steps", "reference_ids",
                "source_image_id", "created_at", "job_id"):
        assert key in meta
    assert meta["job_id"] == job_id and meta["prompt"] == "a cute owl icon"
    listed = client.get("/images", headers=auth).json()["images"]
    assert listed[0]["id"] == image_id
    assert client.get("/jobs", headers=auth).json()["jobs"][0]["job_id"] == job_id


def test_job_api_requires_token(client):
    assert client.post("/jobs", json={"prompt": "x"}).status_code == 401
    assert client.get("/jobs/job_0000000000000000").status_code == 401
    assert client.get("/images").status_code == 401
    assert client.get("/references").status_code == 401
    assert client.get("/models").status_code == 401


def test_strict_validation(client, auth):
    post = lambda body: client.post("/jobs", json=body, headers=auth)  # noqa: E731
    assert post({"prompt": "x", "colour": "red"}).status_code == 422  # unknown field
    assert post({"prompt": ""}).status_code == 422
    r = post({"prompt": "x", "model": "dall-e-3"})
    assert r.status_code == 400 and "available" in r.text
    r = post({"prompt": "x", "model": "z-image-turbo", "reference_ids": ["ref_0000000000000000"]})
    assert r.status_code == 400 and "can't use reference" in r.text
    assert "no reference" in post({"prompt": "x", "reference_ids": ["ref_0123456789abcdef"]}).text
    assert "no image" in post({"prompt": "x", "source_image_id": "img_0123456789abcdef"}).text
    assert "both width and height" in post({"prompt": "x", "width": 512}).text
    assert post({"prompt": "x", "n": 5}).status_code == 400  # max_n is 4


def test_edit_job_uses_source_image_and_keeps_its_shape(client, auth):
    first = wait(client, auth, submit(client, auth, prompt="owl", width=768, height=1024))
    source = first["result"]["image_id"]
    body = wait(client, auth, submit(client, auth, prompt="give it a hat", source_image_id=source))
    call = FakeBackend.instances[-1].calls[-1]
    assert len(call["references"]) == 1
    assert call["height"] > call["width"]  # portrait source, portrait result
    meta = client.get(f"/images/{body['result']['image_id']}/meta", headers=auth).json()
    assert meta["source_image_id"] == source


def test_references_upload_promote_list_delete(client, auth):
    r = client.post("/references", files={"file": ("bear.png", png_bytes((300, 200)), "image/png")},
                    data={"name": "Bear, front view", "tags": "bear, character"}, headers=auth)
    assert r.status_code == 201, r.text
    bear = r.json()
    assert bear["name"] == "Bear, front view" and bear["tags"] == ["bear", "character"]
    assert bear["source"] == "upload" and (bear["width"], bear["height"]) == (300, 200)

    image_id = wait(client, auth, submit(client, auth, prompt="a fox"))["result"]["image_id"]
    r = client.post("/references", json={"from_image_id": image_id, "name": "Fox", "tags": ["fox"]},
                    headers=auth)
    assert r.status_code == 201 and r.json()["source"] == f"image:{image_id}"
    fox = r.json()

    assert {x["id"] for x in client.get("/references", headers=auth).json()["references"]} == {bear["id"], fox["id"]}
    assert [x["id"] for x in client.get("/references?tag=bear", headers=auth).json()["references"]] == [bear["id"]]
    assert client.get(f"/references/{bear['id']}", headers=auth).json()["name"] == "Bear, front view"
    picture = client.get(f"/references/{bear['id']}/image", headers=auth)
    assert picture.headers["content-type"] == "image/png"

    body = wait(client, auth, submit(client, auth, prompt="the bear and the fox", reference_ids=[bear["id"], fox["id"]]))
    assert body["status"] == "done"
    assert len(FakeBackend.instances[-1].calls[-1]["references"]) == 2

    assert client.delete(f"/references/{bear['id']}", headers=auth).status_code == 204
    assert client.get(f"/references/{bear['id']}", headers=auth).status_code == 404
    junk = {"file": ("x.png", b"not an image", "image/png")}
    assert client.post("/references", files=junk, headers=auth).status_code == 400


def test_cancel_a_queued_job(make_client, auth, settings):
    gate = threading.Event()

    class Slow(FakeBackend):
        def generate(self, **kwargs):
            gate.wait(5)
            return super().generate(**kwargs)

    from fastapi.testclient import TestClient

    from image_server.app import create_app
    from image_server.engine import Engine

    with TestClient(create_app(settings, Engine(Slow, release_memory=lambda: None)),
                    base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        first = submit(client, auth, prompt="slow one")
        second = submit(client, auth, prompt="cancel me")
        time.sleep(0.1)  # the first job holds the GPU; the second waits for its turn
        # A queued job is dropped at once: it doesn't wait for the jobs ahead of it, and it
        # frees its queue slot right away.
        assert client.delete(f"/jobs/{second}", headers=auth).json()["status"] == "cancelled"
        assert client.get("/health").json()["jobs_active"] == 1
        assert client.get(f"/jobs/{second}", headers=auth).json()["status"] == "cancelled"
        gate.set()
        assert wait(client, auth, first)["status"] == "done"
        assert wait(client, auth, second)["status"] == "cancelled"


def test_finished_jobs_survive_a_restart(make_client, auth, settings):
    client = make_client()
    job_id = submit(client, auth, prompt="persist me")
    wait(client, auth, job_id)
    fresh = make_client()  # new app, empty memory, same data dir
    assert fresh.get(f"/jobs/{job_id}", headers=auth).json()["status"] == "done"
    assert fresh.get("/jobs/job_zzzz", headers=auth).status_code == 404


def test_ids_cannot_escape_the_store(client, auth):
    for bad in ("..%2F..%2Fetc%2Fpasswd", "img_../../x", "ref_0000", "IMG_0123456789ABCDEF"):
        assert client.get(f"/images/{bad}", headers=auth).status_code == 404
        assert client.get(f"/references/{bad}", headers=auth).status_code == 404


def test_models_and_health(client, auth):
    models = client.get("/models", headers=auth).json()["models"]
    klein = next(m for m in models if m["id"] == "flux2-klein-4b")
    assert klein["default"] and klein["edit_default"] and klein["supports_edit"]
    assert all(not m["research_only"] for m in models)  # research models off by default
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["jobs_active"] == 0


def test_n_images_and_transparency(client, auth):
    body = wait(client, auth, submit(client, auth, prompt="icons", n=2, seed=10, transparent=True))
    images = body["result"]["images"]
    assert [i["seed"] for i in images] == [10, 11]
    full = client.get(f"/images/{images[1]['image_id']}", headers=auth)
    assert Image.open(io.BytesIO(full.content)).mode == "RGBA"


def test_negative_prompt_is_recorded_and_flagged(client, auth):
    body = wait(client, auth, submit(client, auth, prompt="a cat", negative_prompt="blurry"))
    assert body["result"]["ignored"] == ["negative_prompt"]
    meta = client.get(f"/images/{body['result']['image_id']}/meta", headers=auth).json()
    assert meta["negative_prompt"] == "blurry"


def test_save_outputs_off_still_stores_job_results(make_client, auth, settings):
    client = make_client(Settings(**{**settings.__dict__, "save_outputs": False}))
    body = wait(client, auth, submit(client, auth, prompt="x"))
    assert client.get(f"/images/{body['result']['image_id']}", headers=auth).status_code == 200
    r = client.post("/v1/images/generations", json={"prompt": "x"}, headers=auth)
    assert "image_id" not in r.json()["data"][0]  # the OpenAI endpoint honors save_outputs


def test_seeds_wrap_instead_of_overflowing(client, auth):
    body = wait(client, auth, submit(client, auth, prompt="x", n=2, seed=2**32 - 1))
    assert [i["seed"] for i in body["result"]["images"]] == [2**32 - 1, 0]
