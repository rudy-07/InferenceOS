"""
logging.py
----------
Structured JSON request & response logging middleware for InferenceOS Server.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable
from fastapi import Request, Response


logger = logging.getLogger("InferenceOSServer")


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Structured request logging middleware."""
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    log_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(elapsed_ms, 2),
        "client_ip": request.client.host if request.client else "unknown",
    }

    if response.status_code >= 400:
        logger.error(json.dumps(log_payload))
    else:
        logger.info(json.dumps(log_payload))

    return response
