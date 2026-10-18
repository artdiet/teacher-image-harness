"""GuardedH11Protocol under a real uvicorn: per-client caps and the request-head deadline.

These run uvicorn's actual h11 protocol, so a uvicorn upgrade that changes the
internals the guard relies on fails here.
"""

from __future__ import annotations

import asyncio
import collections
import socket
import threading
import time

import pytest
import uvicorn

from image_server.connections import GuardedH11Protocol


BIG = b"x" * (4 * 1024 * 1024)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    if scope["method"] == "POST":  # read the whole body, like the image routes do
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            more = message.get("more_body", False)
    if scope["path"] == "/slow":
        await asyncio.sleep(1.0)  # longer than the head deadline below
    body = BIG if scope["path"] == "/big" else b"ok"
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


@pytest.fixture
def serve():
    servers = []

    def start(protocol):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server = uvicorn.Server(uvicorn.Config(app, http=protocol, lifespan="off", log_level="error"))
        thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        servers.append((server, thread))
        return sock.getsockname()[1]

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(5)


def guarded(**attrs):
    """A protocol class with test limits and its own connection counts."""
    return type("TestProtocol", (GuardedH11Protocol,), {"_open": collections.Counter(), **attrs})


def connect(port: int) -> socket.socket:
    s = socket.create_connection(("127.0.0.1", port), timeout=3)
    s.settimeout(3)
    return s


def closed_by_server(s: socket.socket, within: float) -> bool:
    s.settimeout(within)
    try:
        return s.recv(1024) == b""
    except TimeoutError:
        return False
    except ConnectionResetError:
        return True


def request(s: socket.socket, path: str = "/") -> bytes:
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
    data = b""
    while not (data.endswith(b"ok") or data.endswith(b"}")):
        try:
            chunk = s.recv(1024)
        except ConnectionResetError:
            break
        if not chunk:
            break
        data += chunk
    return data


def test_silent_and_trickling_connections_are_closed(serve):
    port = serve(guarded(head_timeout=0.4))
    silent = connect(port)
    trickle = connect(port)
    trickle.sendall(b"GET / HTTP/1.1\r\n")  # headers never finish
    assert closed_by_server(silent, within=2)
    assert closed_by_server(trickle, within=2)


def test_keep_alive_and_slow_requests_still_work(serve):
    port = serve(guarded(head_timeout=0.4))
    s = connect(port)
    assert request(s).startswith(b"HTTP/1.1 200")
    assert request(s).startswith(b"HTTP/1.1 200")  # same connection, second request
    assert request(s, "/slow").startswith(b"HTTP/1.1 200")  # the app takes longer than the deadline
    assert closed_by_server(s, within=2)  # idle after the last response: closed again


def test_connections_per_client_are_capped(serve):
    protocol = guarded(max_per_client=3, exempt_loopback=False, head_timeout=30)
    port = serve(protocol)
    held = [connect(port) for _ in range(3)]
    extra = connect(port)  # the 4th from the same address is refused, with a real answer
    answer = request(extra)
    assert answer.startswith(b"HTTP/1.1 503") and b"too many open connections" in answer
    assert closed_by_server(extra, within=3)
    extra.close()  # a client closes its end after reading the answer
    held.pop().close()
    time.sleep(0.3)
    again = connect(port)
    assert request(again).startswith(b"HTTP/1.1 200")  # a slot freed up
    for s in held:
        s.close()


def test_loopback_is_exempt_from_the_cap(serve):
    port = serve(guarded(max_per_client=2, head_timeout=30))
    held = [connect(port) for _ in range(4)]  # this Mac (and Docker) arrive as loopback
    assert all(request(s).startswith(b"HTTP/1.1 200") for s in held)


def test_a_stalled_request_body_is_closed(serve):
    port = serve(guarded(body_idle_timeout=0.5, watch_interval=0.1))
    s = connect(port)
    s.sendall(b"POST /upload HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 1000\r\n\r\n" + b"a" * 10)
    assert closed_by_server(s, within=3)  # the rest of the body never came


def test_a_slow_but_moving_body_is_not_closed(serve):
    port = serve(guarded(body_idle_timeout=0.5, watch_interval=0.1))
    s = connect(port)
    s.sendall(b"POST /upload HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 6\r\n\r\n")
    for chunk in (b"ab", b"cd", b"ef"):  # 1.2 s in total, but never 0.5 s without progress
        time.sleep(0.3)
        s.sendall(chunk)
    s.settimeout(3)
    assert s.recv(1024).startswith(b"HTTP/1.1 200")


def test_a_client_that_stops_reading_is_dropped(serve):
    port = serve(guarded(write_stall_timeout=0.5, watch_interval=0.1, head_timeout=30))
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # fills up fast
    s.connect(("127.0.0.1", port))
    # Pipelined requests and no reading: the server's next reply waits on a full buffer.
    s.sendall(b"GET /big HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n" * 20)
    time.sleep(2)
    s.settimeout(3)
    received = 0
    try:
        while chunk := s.recv(1 << 20):
            received += len(chunk)
    except (ConnectionResetError, TimeoutError):
        pass
    assert received < 20 * len(BIG)  # dropped, not served in full
    assert received < 3 * len(BIG)


def test_a_single_big_response_nobody_reads_is_dropped(serve):
    # One ordinary request (no pipelining): uvicorn hands the whole body to the socket at
    # once and calls the response complete; the client then never reads it.
    port = serve(guarded(write_stall_timeout=0.5, watch_interval=0.1, head_timeout=30))
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.connect(("127.0.0.1", port))
    s.sendall(b"GET /big HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
    time.sleep(2)
    s.settimeout(3)
    received = 0
    try:
        while chunk := s.recv(1 << 20):
            received += len(chunk)
    except (ConnectionResetError, TimeoutError):
        pass
    assert received < len(BIG)


def test_a_slow_but_steady_reader_gets_the_whole_response(serve):
    port = serve(guarded(write_stall_timeout=0.5, watch_interval=0.1, head_timeout=30))
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    s.connect(("127.0.0.1", port))
    s.sendall(b"GET /big HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    s.settimeout(3)
    received = 0
    while chunk := s.recv(65536):  # about 1.6 MB/s: slow, but never 0.5 s without progress
        received += len(chunk)
        time.sleep(0.04)
    assert received > len(BIG)  # headers + the whole body
