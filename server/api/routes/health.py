"""
health.py
---------
Health check and version endpoints for InferenceOS Server.
"""
from __future__ import annotations

import sys
import time
from fastapi import APIRouter
from server.runtime_adapter.adapter import get_runtime_adapter

router = APIRouter()


@router.get("/health")
def get_health():
    adapter = get_runtime_adapter()
    return {
        "status": "ok",
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "loaded_models": adapter.get_loaded_models_count(),
    }


@router.get("/version")
def get_version():
    return {
        "server": "InferenceOS Server",
        "version": "1.0.0",
        "python": sys.version,
        "api_compatibility": "OpenAI v1",
    }
