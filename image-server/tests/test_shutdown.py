"""The server process exits on SIGTERM even while a client is stalled mid-request.

uvicorn closes its listener on SIGTERM and then waits for open requests before
the app's shutdown runs; without a limit, one stalled client (a laptop that slept
mid-upload) kept a stopping server alive forever and launchd never restarted it.
This runs the server in its own process with the options __main__ uses.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from conftest import TOKEN

CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[3])
    from pathlib import Path
    from conftest import FakeBackend
    from image_server import jobs as jobs_module
    from image_server.__main__ import serve
    from image_server.app import create_app
    from image_server.config import Settings
    from image_server.engine import Engine

    GRACE = float(sys.argv[5])
    jobs_module.SHUTDOWN_GRACE_SECONDS = GRACE

    class Slow(FakeBackend):  # each picture takes 4 s, and stops between steps when asked
        def generate(self, **kwargs):
            for _ in range(8):
                if kwargs["should_stop"]():
                    from image_server.engine import JobCancelled
                    raise JobCancelled("stopped")
                time.sleep(0.5)
            return super().generate(**kwargs)

    settings = Settings(token=sys.argv[2], data_dir=Path(sys.argv[4]), free_ollama=False)
    inner = create_app(settings, Engine(Slow, release_memory=lambda: None))

    async def app(scope, receive, send):  # report when the app's own shutdown has finished
        async def reporting_send(message):
            if message["type"] == "lifespan.shutdown.complete":
                print("app shutdown complete", flush=True)
            await send(message)
        await inner(scope, receive, reporting_send if scope["type"] == "lifespan" else send)

    serve(app, "127.0.0.1", int(sys.argv[1]), on_stop=inner.state.jobs.begin_shutdown,
          timeout_graceful_shutdown=int(GRACE), log_level="critical")
""")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_child(tmp_path, grace):
    port = free_port()
    tests_dir = str(Path(__file__).parent)
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(port), TOKEN, tests_dir, str(tmp_path / "data"),
                              str(grace)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 15
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return child, port
        except OSError:
            assert time.monotonic() < deadline, "server didn't start"
            time.sleep(0.1)


def test_sigterm_exits_while_a_client_is_stalled_mid_upload(tmp_path):
    child, port = start_child(tmp_path, grace=2)
    try:
        stalled = socket.create_connection(("127.0.0.1", port))
        stalled.sendall(  # an authorized upload whose body stops arriving
            f"POST /images HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {TOKEN}\r\n"
            "Content-Type: multipart/form-data; boundary=x\r\nContent-Length: 100000\r\n\r\n--x\r\n".encode())
        time.sleep(0.5)
        started = time.monotonic()
        os.kill(child.pid, signal.SIGTERM)
        output, _ = child.communicate(timeout=20)
        assert time.monotonic() - started < 10, "shutdown waited on the stalled client"
        # The jobs' shutdown ran (uvicorn then re-raises SIGTERM, so the exit code is -15).
        assert "app shutdown complete" in output, output
        assert child.returncode in (0, -signal.SIGTERM)
        stalled.close()
    finally:
        if child.poll() is None:
            child.kill()


def test_open_webui_picture_in_progress_finishes_during_the_grace(tmp_path):
    """A stop gives an open /v1 request the same grace as background jobs: a 4 s picture
    started before SIGTERM still gets its answer."""
    import http.client
    import json
    import threading

    child, port = start_child(tmp_path, grace=8)
    answer = {}

    def draw():
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
        conn.request("POST", "/v1/images/generations", body=json.dumps({"prompt": "a fox", "size": "256x256"}),
                     headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
        response = conn.getresponse()
        answer["status"] = response.status
        answer["body"] = response.read()

    try:
        thread = threading.Thread(target=draw)
        thread.start()
        time.sleep(1)  # the picture is being drawn
        os.kill(child.pid, signal.SIGTERM)
        thread.join(20)
        child.communicate(timeout=30)
        assert answer.get("status") == 200, answer
        assert b"b64_json" in answer["body"]
    finally:
        if child.poll() is None:
            child.kill()
