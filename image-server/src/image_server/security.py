"""Bearer-token check for every API route.

The server binds to 127.0.0.1 and is only reached by the Open WebUI container
(via host.docker.internal). The token stops other local software, and web pages
open in a browser, from using it; the Host check (TrustedHostMiddleware in
app.py) blocks DNS-rebinding tricks. No CORS headers are ever sent.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status


def bearer_token_dependency(token: str):
    expected = token.encode()

    async def require_token(authorization: str | None = Header(default=None)) -> None:
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            presented.strip().encode(), expected
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return require_token
