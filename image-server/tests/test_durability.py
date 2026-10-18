"""A job_id handed to a client always resolves: across restarts, shutdowns and bursts."""

from __future__ import annotations

import json
import threading
import time

from fastapi.testclient import TestClient

from conftest import TOKEN, FakeBackend
from image_server import jobs as jobs_module
from image_server.app import create_app
from image_server.engine import Engine, JobCancelled

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def open_client(settings, backend=FakeBackend, **engine_args) -> TestClient:
    engine = Engine(backend, release_memory=lambda: None, **engine_args)
    client = TestClient(create_app(settings, engine), base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    client.__enter__()
    return client


def submit(client, **body) -> str:
    r = client.post("/jobs", json={"prompt": "a fox", **body}, headers=AUTH)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def wait(client, job_id, statuses=("done", "failed", "cancelled"), timeout=10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"{job_id} still {body['status']}")


def record_on_disk(settings, job_id) -> dict:
    return json.loads((settings.data_dir / "jobs" / f"{job_id}.json").read_text())


class Gated(FakeBackend):
    """Holds each picture until the test opens the gate."""

    gate = threading.Event()

    def generate(self, **kwargs):
        Gated.gate.wait(10)
        return super().generate(**kwargs)


class UntilStopped(FakeBackend):
    """Runs until asked to stop, like a long mflux job (stops between steps)."""

    def generate(self, **kwargs):
        deadline = time.monotonic() + 10
        while not kwargs["should_stop"]():
            if time.monotonic() > deadline:
                raise AssertionError("never asked to stop")
            time.sleep(0.01)
        raise JobCancelled("stopped")


def test_record_is_written_when_the_job_is_accepted(settings):
    Gated.gate.clear()
    client = open_client(settings, Gated)
    try:
        job_id = submit(client)
        assert record_on_disk(settings, job_id)["status"] in ("queued", "running")
    finally:
        Gated.gate.set()
        client.__exit__(None, None, None)
    assert record_on_disk(settings, job_id)["status"] == "done"  # shutdown let it finish


def test_unfinished_records_from_a_crash_read_as_interrupted(settings):
    # What a crash leaves behind: the record written at submit and its active marker,
    # however long ago (the server may have been off for weeks).
    folder = settings.data_dir / "jobs"
    (folder / "active").mkdir(parents=True)
    old = time.time() - 30 * 24 * 3600
    for job_id, status in (("job_00000000000000aa", "running"), ("job_00000000000000bb", "done")):
        record = {"job_id": job_id, "status": status, "progress": 0.5, "params": {},
                  "created_at": old, "started_at": None, "finished_at": None}
        (folder / f"{job_id}.json").write_text(json.dumps(record))
        (folder / "active" / job_id).touch()
    client = open_client(settings)  # startup restores
    try:
        body = client.get("/jobs/job_00000000000000aa", headers=AUTH).json()
        assert body["status"] == "failed" and "interrupted" in body["error"]
        assert client.get("/jobs/job_00000000000000bb", headers=AUTH).json()["status"] == "done"
        assert list((folder / "active").iterdir()) == []  # markers cleared
    finally:
        client.__exit__(None, None, None)


def test_finished_jobs_leave_no_marker(settings):
    client = open_client(settings)
    try:
        wait(client, submit(client))
    finally:
        client.__exit__(None, None, None)
    assert list((settings.data_dir / "jobs" / "active").iterdir()) == []


def test_shutdown_stops_what_it_cannot_wait_for_and_records_it(settings, monkeypatch):
    monkeypatch.setattr(jobs_module, "SHUTDOWN_GRACE_SECONDS", 0.2)
    client = open_client(settings, UntilStopped)
    running = submit(client)
    queued = submit(client)
    wait(client, running, statuses=("running",))
    client.__exit__(None, None, None)  # lifespan shutdown
    for job_id in (running, queued):
        record = record_on_disk(settings, job_id)
        assert record["status"] == "failed" and "interrupted" in record["error"], record
    fresh = open_client(settings)
    try:  # and the next process answers for them
        assert fresh.get(f"/jobs/{running}", headers=AUTH).json()["status"] == "failed"
    finally:
        fresh.__exit__(None, None, None)


def test_new_jobs_are_refused_while_restarting(settings):
    client = open_client(settings)
    try:
        client.app.state.jobs.closing = True  # what the idle restart sets before exiting
        r = client.post("/jobs", json={"prompt": "late"}, headers=AUTH)
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
        assert "restarting" in r.json()["detail"]
    finally:
        client.__exit__(None, None, None)


def test_a_burst_beyond_the_queue_is_refused_up_front(settings):
    Gated.gate.clear()
    client = open_client(settings, Gated, max_pending=2)
    try:
        codes = [client.post("/jobs", json={"prompt": f"p{i}"}, headers=AUTH).status_code for i in range(4)]
        assert codes == [202, 202, 503, 503]  # never 202 now and "busy" later
    finally:
        Gated.gate.set()
        client.__exit__(None, None, None)


def test_job_list_includes_jobs_from_before_a_restart(settings):
    client = open_client(settings)
    job_id = submit(client)
    wait(client, job_id)
    client.__exit__(None, None, None)
    fresh = open_client(settings)
    try:
        listed = fresh.get("/jobs", headers=AUTH).json()["jobs"]
        assert [j["job_id"] for j in listed] == [job_id]
    finally:
        fresh.__exit__(None, None, None)


def test_draining_doesnt_load_another_model(settings):
    import asyncio

    from image_server.config import MODELS
    from image_server.engine import EngineClosed, Job

    async def scenario():
        engine = Engine(FakeBackend, release_memory=lambda: None)
        klein, zimage = MODELS["flux2-klein-4b"], MODELS["z-image-turbo"]
        await engine.run(Job(model=klein, prompt="a", width=256, height=256, seed=1))
        engine.draining = True
        await engine.run(Job(model=klein, prompt="same model: fine", width=256, height=256, seed=2))
        try:
            await engine.run(Job(model=zimage, prompt="needs a swap", width=256, height=256, seed=3))
        except EngineClosed:
            return "refused"
        finally:
            engine.shutdown()
        return "ran"

    assert asyncio.run(scenario()) == "refused"


def test_picture_count_is_checked_before_decoding(settings, monkeypatch):
    from image_server import jobs as jobs_mod

    decoded = []
    monkeypatch.setattr(jobs_mod, "load_reference_image", lambda data: decoded.append(1))
    client = open_client(settings)
    try:
        refs = [f"ref_{i:016x}" for i in range(5)]
        r = client.post("/jobs", json={"prompt": "x", "reference_ids": refs}, headers=AUTH)
        assert r.status_code == 400 and "at most 4" in r.text
        assert decoded == []
    finally:
        client.__exit__(None, None, None)


def test_queue_depth_counts_the_jobs_ahead(settings):
    Gated.gate.clear()
    client = open_client(settings, Gated)
    try:
        first = client.post("/jobs", json={"prompt": "a"}, headers=AUTH).json()
        second = client.post("/jobs", json={"prompt": "b"}, headers=AUTH).json()
        assert (first["queue_depth"], second["queue_depth"]) == (0, 1)
    finally:
        Gated.gate.set()
        client.__exit__(None, None, None)


def test_a_job_that_cannot_be_recorded_is_refused(settings, monkeypatch):
    client = open_client(settings)
    try:
        monkeypatch.setattr(client.app.state.jobs, "_persist", lambda record: False)
        r = client.post("/jobs", json={"prompt": "x"}, headers=AUTH)
        assert r.status_code == 503 and "couldn't record" in r.json()["detail"]
        assert client.get("/health").json()["queue_depth"] == 0  # its slot was given back
    finally:
        client.__exit__(None, None, None)


def test_marker_stays_when_the_final_record_fails(settings, monkeypatch):
    client = open_client(settings)
    jobs = client.app.state.jobs
    real = jobs._persist
    calls = []

    def flaky(record):  # the record at submit works, the final one doesn't
        calls.append(record.status)
        return real(record) if len(calls) == 1 else False

    monkeypatch.setattr(jobs, "_persist", flaky)
    try:
        job_id = submit(client)
        wait(client, job_id)
        assert (settings.data_dir / "jobs" / "active" / job_id).exists()  # next start: interrupted
    finally:
        client.__exit__(None, None, None)
