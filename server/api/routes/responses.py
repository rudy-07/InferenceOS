"""
responses.py
------------
POST /v1/responses optional placeholder endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict

router = APIRouter()


class ResponseRequest(BaseModel):
    model: str
    input: Any


@router.post("/v1/responses")
def create_response(body: ResponseRequest):
    return {
        "id": "resp-inferenceos-placeholder",
        "object": "response",
        "model": body.model,
        "status": "completed",
        "output": "InferenceOS Response API endpoint ready.",
    }
