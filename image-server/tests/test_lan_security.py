"""LAN-facing behavior: per-device tokens, private-network filter, uploads, styles, previews."""

from __future__ import annotations

import io
import json
import os
import threading
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from conftest import TOKEN, FakeBackend, png_bytes
from image_server.app import create_app
from image_server.config import Settings
from image_server.engine import Engine
from image_server.security import TokenSet, hash_token, is_private_client
from image_server.store import PREVIEW_BUDGET, PREVIEW_STEPS, make_preview, write_private


def write_tokens(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tokens": [{"label": l, "sha256": hash_token(t)} for l, t in entries]}))


def test_device_tokens_work_and_revoke_without_restart(make_client, settings):
    device = "device-token-abcdefghijklmnopqrstuvwxyz0123"
    write_tokens(settings.tokens_file, [("kitchen-laptop", device)])
    client = make_client()
    headers = {"Authorization": f"Bearer {device}"}
    assert client.get("/models", headers=headers).status_code == 200
    assert client.get("/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    write_tokens(settings.tokens_file, [])  # revoke
    future = time.time() + 5
    os.utime(settings.tokens_file, (future, future))  # make sure the mtime changes
    time.sleep(2.1)  # reload interval
    assert client.get("/models", headers=headers).status_code == 401


def test_token_set_labels_and_bad_files(tmp_path):
    path = tmp_path / "tokens.json"
    tokens = TokenSet("primary-token-0123456789abcdef", path)
    assert tokens.check("primary-token-0123456789abcdef") == "primary"
    assert tokens.check("nope") is None
    write_tokens(path, [("a", "token-a"), ("b", "token-b")])
    tokens._checked = 0
    assert tokens.check("token-b") == "b"

    def rewrite(text, bump):
        path.write_text(text)
        os.utime(path, (time.time() + bump, time.time() + bump))
        tokens._checked = 0

    # A damaged file fails closed: a revoked token must not keep working until a restart.
    rewrite("{not json", 10)
    assert tokens.check("token-b") is None
    assert tokens.check("primary-token-0123456789abcdef") == "primary"  # Open WebUI keeps working
    rewrite("[]", 20)  # valid JSON, wrong shape
    assert tokens.check("token-b") is None
    # One malformed entry doesn't take the others down.
    good = {"label": "b", "sha256": hash_token("token-b")}
    rewrite(json.dumps({"tokens": [{"label": "x"}, {"label": "y", "sha256": "abc"}, "junk", good]}), 30)
    assert tokens.check("token-b") == "b"


def test_private_network_filter(settings):
    def app():
        return create_app(settings, Engine(FakeBackend, release_memory=lambda: None))

    with TestClient(app(), base_url="http://127.0.0.1", client=("8.8.8.8", 4000)) as public:
        assert public.get("/health").status_code == 403
    with TestClient(app(), base_url="http://127.0.0.1", client=("192.168.1.20", 4000)) as lan:
        assert lan.get("/health").status_code == 200
    for host, ok in [("10.0.0.5", True), ("172.20.1.1", True), ("100.101.102.103", True),
                     ("::1", True), ("fe80::1", True), ("::ffff:192.168.1.2", True),
                     ("1.1.1.1", False), ("2606:4700::1111", False), ("testclient", False), (None, False)]:
        assert is_private_client(host) is ok, host


def test_upload_then_edit_by_image_id(client, auth):
    buf = io.BytesIO()
    Image.new("RGB", (900, 600), (20, 120, 200)).save(buf, format="JPEG")
    r = client.post("/images", files={"file": ("my photo.jpg", buf.getvalue(), "image/jpeg")}, headers=auth)
    assert r.status_code == 201, r.text
    upload = r.json()
    assert (upload["width"], upload["height"]) == (900, 600)
    meta = client.get(f"/images/{upload['image_id']}/meta", headers=auth).json()
    assert meta["source"] == "upload" and meta["filename"] == "my photo.jpg"
    job = client.post("/jobs", json={"prompt": "make it a coloring page", "style": "coloring",
                                     "source_image_id": upload["image_id"]}, headers=auth).json()
    for _ in range(200):
        body = client.get(f"/jobs/{job['job_id']}", headers=auth).json()
        if body["status"] == "done":
            break
        time.sleep(0.02)
    call = FakeBackend.instances[-1].calls[-1]
    assert len(call["references"]) == 1 and call["width"] > call["height"]
    assert call["prompt"].startswith("make it a coloring page, coloring book page for kids")
    params = body["result"]["params"]
    assert params["prompt"] == "make it a coloring page" and params["style"] == "coloring"
    assert params["full_prompt"] == call["prompt"]


def test_upload_rejects_non_images(client, auth):
    r = client.post("/images", files={"file": ("x.png", b"%PDF-1.7 not an image", "image/png")}, headers=auth)
    assert r.status_code == 400
    assert client.post("/images", files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 401


def test_preview_fits_the_budget():
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1024, 1024, 3), dtype=np.uint8))
    buf = io.BytesIO()
    noisy.save(buf, format="PNG")
    smallest = PREVIEW_STEPS[-1][0]
    for budget in (PREVIEW_BUDGET, 20_000):
        preview = make_preview(buf.getvalue(), budget)
        assert len(preview) <= budget or max(Image.open(io.BytesIO(preview)).size) == smallest
    flat = make_preview(png_bytes((1024, 1024)))
    assert max(Image.open(io.BytesIO(flat)).size) == 768  # simple images keep full preview size


def test_preview_budget_parameter(client, auth):
    job = client.post("/jobs", json={"prompt": "x"}, headers=auth).json()
    for _ in range(200):
        body = client.get(f"/jobs/{job['job_id']}", headers=auth).json()
        if body["status"] == "done":
            break
        time.sleep(0.02)
    image_id = body["result"]["image_id"]
    small = client.get(f"/images/{image_id}?size=preview&budget=8000", headers=auth)
    assert small.status_code == 200 and len(small.content) <= 8000
    assert client.get(f"/images/{image_id}?size=preview&budget=100", headers=auth).status_code == 422
    assert client.get(f"/images/{image_id}?size=preview&budget=900000", headers=auth).status_code == 422


def test_busy_server_sends_retry_after(settings, auth):
    engine = Engine(FakeBackend, release_memory=lambda: None, max_pending=0)
    with TestClient(create_app(settings, engine), base_url="http://127.0.0.1",
                    client=("127.0.0.1", 50000)) as client:
        r = client.post("/jobs", json={"prompt": "x"}, headers=auth)
        assert r.status_code == 503 and r.headers["retry-after"] == "30"


def test_primary_token_only_from_this_mac(settings):
    device = "device-token-for-the-classroom-pc-0123456789"
    write_tokens(settings.tokens_file, [("classroom-pc", device)])

    def client_at(address):
        engine = Engine(FakeBackend, release_memory=lambda: None)
        return TestClient(create_app(settings, engine), base_url="http://127.0.0.1", client=(address, 4000))

    with client_at("192.168.1.20") as lan:
        # Open WebUI's token is useless from another computer; that computer's own works.
        assert lan.get("/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 401
        assert lan.get("/models", headers={"Authorization": f"Bearer {device}"}).status_code == 200
    with client_at("127.0.0.1") as local:  # Open WebUI (Docker) arrives as loopback
        assert local.get("/models", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200


def test_host_check_ignores_case(settings):
    mixed = Settings(**{**settings.__dict__, "allowed_hosts": ("127.0.0.1", "Teachers-MacBook-Pro.local")})
    engine = Engine(FakeBackend, release_memory=lambda: None)
    with TestClient(create_app(mixed, engine), client=("127.0.0.1", 4000)) as client:
        for host in ("teachers-macbook-pro.local", "TEACHERS-MACBOOK-PRO.LOCAL:7865"):
            assert client.get("/health", headers={"Host": host}).status_code == 200, host
        assert client.get("/health", headers={"Host": "evil.local"}).status_code == 400
    env = {"IMAGE_SERVER_ALLOWED_HOSTS": "Teachers-MacBook-Pro.local, 192.168.1.20"}
    assert Settings.from_env(env).allowed_hosts == ("teachers-macbook-pro.local", "192.168.1.20")


def test_concurrent_writes_to_one_path_never_fail(tmp_path):
    target = tmp_path / "img_0123456789abcdef.preview.webp"
    errors = []

    def writer(n):
        try:
            for i in range(50):
                write_private(target, f"{n}-{i}".encode() * 100)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert target.stat().st_mode & 0o777 == 0o600
    assert [p.name for p in tmp_path.iterdir()] == [target.name]  # no temp files left behind


def test_unreadable_token_file_fails_closed(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "client_tokens.json"
    write_tokens(path, [("laptop", "token-laptop")])
    tokens = TokenSet("primary-token-0123456789abcdef", path)
    assert tokens.check("token-laptop") == "laptop"
    folder.chmod(0)  # stat() now fails with PermissionError
    try:
        tokens._checked = 0
        assert tokens.check("token-laptop") is None
        assert tokens.check("primary-token-0123456789abcdef") == "primary"
    finally:
        folder.chmod(0o700)
    tokens._checked = 0
    assert tokens.check("token-laptop") == "laptop"  # readable again: back
    # The file itself unreadable (stat works, read fails): fixing its mode changes no mtime.
    write_tokens(path, [("laptop", "token-laptop"), ("tablet", "token-tablet")])
    path.chmod(0)
    try:
        tokens._checked = 0
        assert tokens.check("token-tablet") is None
    finally:
        path.chmod(0o600)
    tokens._checked = 0
    assert tokens.check("token-tablet") == "tablet"
