"""
models.py
---------
OpenAI-compatible GET /v1/models route handler.
"""
from __future__ import annotations

from fastapi import APIRouter
from server.models.openai import ModelListResponse, ModelObject
from server.runtime_adapter.adapter import get_runtime_adapter

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
def list_models():
    adapter = get_runtime_adapter()
    reg_models = adapter.get_registered_models()
    
    model_objects = []
    for m in reg_models:
        model_objects.append(
            ModelObject(
                id=m.get("nickname", "unknown"),
                status=m.get("status", "registered"),
                size_bytes=m.get("size_bytes", 0),
                backend=m.get("backend", "auto"),
                context_length=m.get("context", 4096),
            )
        )
        
    return ModelListResponse(data=model_objects)
