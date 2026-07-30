"""
auth.py
-------
API Key and Bearer token authentication middleware for InferenceOS Server.
"""
from __future__ import annotations

from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from server.config import ServerConfig


async def auth_middleware(request: Request, call_next: Callable, config: ServerConfig) -> Response:
    """Validate Bearer API token against configured keys if authentication is enabled."""
    if not config.auth_api_keys:
        return await call_next(request)

    # Bypass public health, version, and docs routes
    path = request.url.path
    if path in ("/health", "/version", "/metrics", "/docs", "/openapi.json"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("api-key")

    token: Optional[str] = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    elif api_key_header:
        token = api_key_header.strip()

    if not token or token not in config.auth_api_keys:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "message": "Incorrect or missing API key.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )

    return await call_next(request)
