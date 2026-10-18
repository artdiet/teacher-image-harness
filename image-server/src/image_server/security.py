"""Bearer tokens and the private-network client filter.

The server listens on 127.0.0.1, or on the LAN when enabled (scripts/enable_lan.sh).
Every route except /health needs "Authorization: Bearer <token>":

- the primary token (IMAGE_SERVER_TOKEN in .env), used by the Open WebUI pipe. It is
  accepted only from this Mac (Docker's host.docker.internal arrives as loopback), so a
  leaked copy is useless elsewhere on the network;
- per-device client tokens (one per Claude machine), stored only as SHA-256 hashes in
  client_tokens.json and managed with scripts/client_tokens.sh. Edits to that file
  are picked up without a restart, so a lost laptop can be revoked on its own.

TrustedHosts blocks DNS-rebinding tricks, no CORS headers are sent, and
PrivateNetworkOnly refuses clients outside private, loopback, link-local and CGNAT
(e.g. Tailscale) ranges. Connection limits live in connections.py.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import time
from pathlib import Path

from fastapi import Header, HTTPException, Request, status
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse

log = logging.getLogger(__name__)

CGNAT = ipaddress.ip_network("100.64.0.0/10")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenSet:
    """The primary token plus hashed per-device tokens from a JSON file."""

    def __init__(self, primary: str, clients_file: Path | None = None) -> None:
        self.primary = primary.encode()
        self.clients_file = clients_file
        self._hashes: list[tuple[str, bytes]] = []
        self._mtime: int | None = None
        self._checked = 0.0
        self._failing = False

    def _reload(self) -> None:
        if self.clients_file is None or time.monotonic() - self._checked < 2:
            return
        self._checked = time.monotonic()
        try:
            mtime = self.clients_file.stat().st_mtime_ns
        except FileNotFoundError:
            self._hashes, self._mtime = [], None
            return
        except OSError as exc:  # can't even look at it: fail closed, try again next time
            self._refuse(f"could not read {self.clients_file} ({exc}): device tokens are refused until it is readable")
            self._mtime = None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self.clients_file.read_text())
        except OSError as exc:
            # Unreadable (permissions): fixing that changes no mtime, so don't remember this one.
            self._refuse(f"could not read {self.clients_file} ({exc}): device tokens are refused until it is readable")
            self._mtime = None
            return
        except ValueError as exc:  # damaged content stays damaged until the file is written again
            self._refuse(f"{self.clients_file} is damaged ({exc}): device tokens are refused until it is fixed")
            self._mtime = mtime
            return
        self._mtime = mtime
        entries = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            self._refuse(f'{self.clients_file} is damaged (expected {{"tokens": [...]}}): device tokens are refused')
            return
        hashes = []
        for entry in entries:
            try:
                digest = bytes.fromhex(entry["sha256"])
                if len(digest) != hashlib.sha256().digest_size:
                    raise ValueError("not a SHA-256 hex digest")
                hashes.append((str(entry["label"]), digest))
            except (KeyError, TypeError, ValueError, AttributeError):
                log.warning("ignoring a malformed entry in %s", self.clients_file)
        self._hashes = hashes
        self._failing = False
        log.info("loaded %d client token(s)", len(hashes))

    def _refuse(self, why: str) -> None:
        """Fail closed: a damaged or unreadable file must not keep a revoked token working."""
        self._hashes = []
        if not self._failing:  # once per problem, not every 2 s
            log.error(why)
            self._failing = True

    def check(self, presented: str) -> str | None:
        """The token's label ('primary' or a device name), or None if it isn't valid."""
        token = presented.encode()
        if self.primary and hmac.compare_digest(token, self.primary):
            return "primary"
        self._reload()
        digest = hashlib.sha256(token).digest()
        match = None
        for label, stored in self._hashes:  # no early exit: constant work per check
            if hmac.compare_digest(digest, stored) and match is None:
                match = label
        return match


def bearer_token_dependency(tokens: TokenSet | str):
    if isinstance(tokens, str):
        tokens = TokenSet(tokens)

    async def require_token(request: Request, authorization: str | None = Header(default=None)) -> None:
        scheme, _, presented = (authorization or "").partition(" ")
        presented = presented.strip()
        label = tokens.check(presented) if scheme.lower() == "bearer" and presented else None
        if label == "primary" and not is_loopback_client(request.client.host if request.client else None):
            label = None  # other computers use their own device token
        if label is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        request.state.client = label

    return require_token


def _client_ip(host: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return None
    return ip.ipv4_mapped if ip.version == 6 and ip.ipv4_mapped else ip


def is_private_client(host: str | None) -> bool:
    ip = _client_ip(host)
    if ip is None:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or (ip.version == 4 and ip in CGNAT)


def is_loopback_client(host: str | None) -> bool:
    ip = _client_ip(host)
    return ip is not None and ip.is_loopback


class TrustedHosts(TrustedHostMiddleware):
    """Starlette's Host check (against DNS rebinding), case-insensitive as host names are.

    macOS LocalHostNames are mixed case ("Teachers-MacBook-Pro") and clients such as
    Node's fetch send them lowercased, which an exact match would refuse."""

    def __init__(self, app, allowed_hosts: list[str]) -> None:
        super().__init__(app, allowed_hosts=[h.lower() for h in allowed_hosts], www_redirect=False)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = {**scope, "headers": [(k, v.lower()) if k == b"host" else (k, v)
                                          for k, v in scope["headers"]]}
        await super().__call__(scope, receive, send)


class PrivateNetworkOnly:
    """ASGI middleware: 403 for clients outside private networks."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client = scope.get("client")
            if not is_private_client(client[0] if client else None):
                await PlainTextResponse("forbidden: private network clients only", status_code=403)(
                    scope, receive, send)
                return
        await self.app(scope, receive, send)
