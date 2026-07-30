"""
sse.py
------
Server-Sent Events (SSE) stream generator with client disconnect detection
and instant unbuffered token delivery.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi import Request
from server.models.openai import ChatCompletionChunk, ChatCompletionChunkChoice, DeltaMessage


async def stream_chat_completion_sse(
    request: Request,
    model: str,
    token_queue: asyncio.Queue[Optional[str]],
    completion_id: Optional[str] = None,
    session_id: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding OpenAI SSE formatted text/event-stream chunks.
    Detects client disconnects and triggers cancel_event if client leaves.
    """
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # Send initial role chunk
    initial_chunk = ChatCompletionChunk(
        id=cid,
        created=created_ts,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
        session_id=session_id,
    )
    yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n"

    try:
        while True:
            # Check client disconnect
            if await request.is_disconnected():
                if cancel_event:
                    cancel_event.set()
                break

            try:
                # Wait for token with timeout to allow disconnect checking
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # None is sentinel for end-of-stream
            if token is None:
                token_queue.task_done()
                break

            chunk = ChatCompletionChunk(
                id=cid,
                created=created_ts,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=token),
                        finish_reason=None,
                    )
                ],
                session_id=session_id,
            )
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            token_queue.task_done()

        # Send final stop chunk
        final_chunk = ChatCompletionChunk(
            id=cid,
            created=created_ts,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
            session_id=session_id,
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"

    except asyncio.CancelledError:
        if cancel_event:
            cancel_event.set()
        raise
