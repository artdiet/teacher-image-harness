"""HTTP/1.1 connection limits that uvicorn doesn't have.

uvicorn starts no timer when a connection opens and only caps connections globally
(limit_concurrency), so in LAN mode any device could hold all of the server's
connections open without sending a byte or a token. GuardedH11Protocol adds:

- a cap on open connections per client address (this Mac, including Docker's
  host.docker.internal, is exempt); one over the cap gets a 503 instead of service;
- a deadline for each request's headers: a connection that hasn't sent a complete
  request head within HEAD_TIMEOUT seconds of opening, or of its previous response,
  is closed;
- a watchdog while a request is in progress or a response is still being sent: a body
  that stops arriving for BODY_IDLE_TIMEOUT, or a response the client stops reading
  for WRITE_STALL_TIMEOUT, gets the connection dropped. Slow requests (a picture being
  drawn) and slow-but-moving transfers are not affected;
- TCP keepalive, so a laptop that went to sleep or left the network mid-request is
  noticed within about 90 s instead of never.

It subclasses uvicorn's h11 protocol (the version is pinned in uv.lock);
tests/test_connections.py runs it under a real uvicorn to catch upgrades that change
those internals.
"""

from __future__ import annotations

import collections
import ipaddress
import socket

from uvicorn.protocols.http.h11_impl import H11Protocol

MAX_PER_CLIENT = 16
HEAD_TIMEOUT = 10.0
BODY_IDLE_TIMEOUT = 30.0
WRITE_STALL_TIMEOUT = 60.0
WATCH_INTERVAL = 5.0
KEEPALIVE = ((getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None), 60),
             (getattr(socket, "TCP_KEEPINTVL", None), 10), (getattr(socket, "TCP_KEEPCNT", None), 3))
REFUSE_LINGER = 1.0  # keep reading a refused connection briefly, so its 503 isn't lost to a reset
_TOO_MANY = b'{"detail": "too many open connections from this computer; try again in a moment"}'
TOO_MANY_RESPONSE = (b"HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\nretry-after: 5\r\n"
                     b"connection: close\r\ncontent-length: " + str(len(_TOO_MANY)).encode() + b"\r\n\r\n" + _TOO_MANY)


def _is_loopback(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _enable_keepalive(transport) -> None:
    sock = transport.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for option, value in KEEPALIVE:
            if option is not None:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
    except OSError:  # not a TCP socket (tests), or the option isn't supported
        pass


class GuardedH11Protocol(H11Protocol):
    max_per_client = MAX_PER_CLIENT
    head_timeout = HEAD_TIMEOUT
    body_idle_timeout = BODY_IDLE_TIMEOUT
    write_stall_timeout = WRITE_STALL_TIMEOUT
    watch_interval = WATCH_INTERVAL
    exempt_loopback = True
    _open: collections.Counter = collections.Counter()  # per client address; one event loop, no locking

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._head_timer = None
        self._watch_timer = None
        self._counted: str | None = None
        self._refused = False
        self._last_progress = 0.0  # last byte of request body received
        self._buffered = 0  # response bytes waiting to be sent at the last check
        self._last_drained = 0.0  # last time that backlog shrank (or was empty)

    def connection_made(self, transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        _enable_keepalive(transport)
        host = self.client[0] if self.client else None
        if host and not (self.exempt_loopback and _is_loopback(host)):
            self._counted = host
            self._open[host] += 1
            if self._open[host] > self.max_per_client:
                self.logger.warning("refused a connection from %s: %d already open", host, self.max_per_client)
                self._refuse()
                return
        self._arm_head_timer()

    def _refuse(self) -> None:
        # A real answer, so the client reports "too many connections", not a vanished server.
        # Half-close and keep draining what it sends for a moment: closing with its request
        # unread would turn into a reset that can wipe the answer before it is read.
        self._refused = True
        self.transport.write(TOO_MANY_RESPONSE)
        if self.transport.can_write_eof():
            self.transport.write_eof()
        self.loop.call_later(REFUSE_LINGER, self.transport.close)  # connection_lost() undoes the count

    def connection_lost(self, exc: Exception | None) -> None:
        self._cancel_head_timer()
        if self._watch_timer is not None:
            self._watch_timer.cancel()
            self._watch_timer = None
        if self._counted is not None:
            self._open[self._counted] -= 1
            if self._open[self._counted] <= 0:
                del self._open[self._counted]
            self._counted = None
        super().connection_lost(exc)

    def data_received(self, data: bytes) -> None:
        if self._refused:
            return  # read and dropped: this connection gets no service
        self._last_progress = self.loop.time()
        super().data_received(data)

    def handle_events(self) -> None:
        super().handle_events()
        if self._request_in_progress():
            self._cancel_head_timer()  # the head arrived; the watchdog takes over
            self._last_progress = self.loop.time()
            self._start_watch()

    def _start_watch(self) -> None:
        if self._watch_timer is None:
            self._buffered, self._last_drained = 0, self.loop.time()
            self._watch_timer = self.loop.call_later(self.watch_interval, self._watch)

    def _watch(self) -> None:
        """Runs while a request is in progress or response bytes wait to be sent (also after
        uvicorn has marked the response complete, or asked to close once it is flushed)."""
        self._watch_timer = None
        busy = self._request_in_progress() and not self.transport.is_closing()
        buffered = self.transport.get_write_buffer_size()
        if not busy and not buffered:
            return  # idle: the head deadline takes over
        now = self.loop.time()
        if busy and self.flow.read_paused:  # we stopped reading (the app hasn't caught up), not the client
            self._last_progress = now
        if buffered == 0 or buffered < self._buffered or self._buffered == 0:
            self._last_drained = now  # the client is reading (or there was nothing to read)
        self._buffered = buffered
        body_stalled = busy and self.cycle.more_body and now - self._last_progress > self.body_idle_timeout
        reader_stalled = buffered and now - self._last_drained > self.write_stall_timeout
        if body_stalled or reader_stalled:
            self.logger.warning("dropping a stalled connection from %s (%s)", self.client[0] if self.client else "?",
                                "request body stopped" if body_stalled else "response not being read")
            self.transport.abort()  # close() would wait forever for the unread bytes to flush
            return
        self._watch_timer = self.loop.call_later(self.watch_interval, self._watch)

    def on_response_complete(self) -> None:
        super().on_response_complete()
        if self.transport.get_write_buffer_size():
            self._start_watch()  # the whole response may still be waiting to be read
        # uvicorn's own keep-alive timer is cancelled by any incoming byte, so a client
        # trickling the next request's headers would never time out without this.
        if not self.transport.is_closing() and not self._request_in_progress():
            self._arm_head_timer()

    def _request_in_progress(self) -> bool:
        return self.cycle is not None and not self.cycle.response_complete

    def _arm_head_timer(self) -> None:
        self._cancel_head_timer()
        self._head_timer = self.loop.call_later(self.head_timeout, self._head_timed_out)

    def _cancel_head_timer(self) -> None:
        if self._head_timer is not None:
            self._head_timer.cancel()
            self._head_timer = None

    def _head_timed_out(self) -> None:
        self._head_timer = None
        if self._request_in_progress() or self.transport.is_closing():
            return
        if self.transport.get_write_buffer_size():  # still sending the last response: the watchdog decides
            self._start_watch()
            self._arm_head_timer()
            return
        self.transport.close()
