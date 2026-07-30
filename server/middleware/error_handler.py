"""
error_handler.py
----------------
Global exception handling middleware converting runtime errors to OpenAI format.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from server.metrics.telemetry import get_telemetry_collector


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map python exceptions to OpenAI API error structure."""
    telemetry = get_telemetry_collector()
    
    if isinstance(exc, FileNotFoundError):
        telemetry.record_request_error("model_not_found")
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    elif isinstance(exc, ValueError):
        telemetry.record_request_error("invalid_parameter")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_parameter",
                }
            },
        )
    elif isinstance(exc, MemoryError) or "OOM" in str(exc).upper():
        telemetry.record_request_error("out_of_memory")
        return JSONResponse(
            status_code=507,
            content={
                "error": {
                    "message": f"Inference engine Out Of Memory: {exc}",
                    "type": "server_error",
                    "param": None,
                    "code": "out_of_memory",
                }
            },
        )

    telemetry.record_request_error(type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "api_error",
                "param": None,
                "code": "internal_error",
            }
        },
    )
