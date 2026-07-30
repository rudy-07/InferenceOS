"""
embeddings.py
-------------
OpenAI-compatible POST /v1/embeddings placeholder route handler.
"""
from __future__ import annotations

from fastapi import APIRouter
from server.models.openai import EmbeddingRequest, EmbeddingResponse, EmbeddingObject, UsageInfo

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embedding(body: EmbeddingRequest):
    # Standard placeholder embedding vector (128 dims)
    inputs = [body.input] if isinstance(body.input, str) else body.input
    objects = []
    for idx, inp in enumerate(inputs):
        objects.append(
            EmbeddingObject(
                embedding=[0.01 * (i + 1) for i in range(128)],
                index=idx,
            )
        )
    return EmbeddingResponse(
        data=objects,
        model=body.model,
        usage=UsageInfo(prompt_tokens=10, completion_tokens=0, total_tokens=10),
    )
