"""
chat.py
-------
OpenAI-compatible POST /v1/chat/completions route handler with session binding
and instant SSE token streaming.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from server.models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)
from server.sessions.session_manager import get_session_manager
from server.runtime_adapter.adapter import get_runtime_adapter
from server.queue.request_queue import get_queue_manager
from server.streaming.sse import stream_chat_completion_sse
from server.metrics.telemetry import get_telemetry_collector

router = APIRouter()


@router.post("/v1/chat/completions")
async def create_chat_completion(request: Request, body: ChatCompletionRequest):
    session_mgr = get_session_manager()
    adapter = get_runtime_adapter()
    queue_mgr = get_queue_manager()
    telemetry = get_telemetry_collector()

    start_perf = telemetry.record_request_start()
    session = None

    # Resolve session binding
    if body.session_id:
        session = session_mgr.get_session(body.session_id)
        if not session:
            session = session_mgr.create_session(model=body.model, session_id=body.session_id)
        # Append incoming messages to session history
        if body.messages:
            for msg in body.messages:
                session.add_message(msg)

    # Request parameter overrides dict
    request_overrides = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
        "seed": body.seed,
        "max_tokens": body.get_max_tokens(),
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if body.stream:
        # SSE Streaming Path
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        cancel_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def sync_on_token(token: str) -> None:
            if not cancel_event.is_set():
                loop.call_soon_threadsafe(token_queue.put_nowait, token)

        async def run_async_inference():
            try:
                # Run execution in threadpool to avoid blocking async loop
                result = await loop.run_in_executor(
                    None,
                    lambda: adapter.execute_chat(
                        model_query=body.model,
                        messages=body.messages,
                        request_overrides=request_overrides,
                        session=session,
                        on_token=sync_on_token,
                    )
                )
                if session and result.generated_text:
                    session.add_assistant_response(result.generated_text)
                
                stats = result.stats
                telemetry.record_request_success(
                    start_time_perf=start_perf,
                    prompt_tokens=getattr(stats, "prompt_tokens", 0),
                    gen_tokens=getattr(stats, "tokens_generated", 0),
                    ttft_ms=getattr(stats, "prompt_eval_ms", 0.0),
                    prompt_tps=getattr(stats, "prompt_eval_tps", 0.0),
                    gen_tps=getattr(stats, "eval_tps", 0.0),
                )
            except Exception as e:
                telemetry.record_request_error(type(e).__name__)
                raise
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        # Launch inference job via Queue Manager
        asyncio.create_task(
            queue_mgr.enqueue_and_execute(
                model_name=body.model,
                execution_func=run_async_inference,
                session_id=body.session_id,
            )
        )

        return StreamingResponse(
            stream_chat_completion_sse(
                request=request,
                model=body.model,
                token_queue=token_queue,
                completion_id=completion_id,
                session_id=body.session_id,
                cancel_event=cancel_event,
            ),
            media_type="text/event-stream",
        )

    else:
        # Non-streaming Path
        loop = asyncio.get_running_loop()

        async def run_sync_job():
            return await loop.run_in_executor(
                None,
                lambda: adapter.execute_chat(
                    model_query=body.model,
                    messages=body.messages,
                    request_overrides=request_overrides,
                    session=session,
                    on_token=None,
                )
            )

        result = await queue_mgr.enqueue_and_execute(
            model_name=body.model,
            execution_func=run_sync_job,
            session_id=body.session_id,
        )

        gen_text = result.generated_text or ""
        if session and gen_text:
            session.add_assistant_response(gen_text)

        stats = result.stats
        prompt_tokens = getattr(stats, "prompt_tokens", 0)
        gen_tokens = getattr(stats, "tokens_generated", 0)
        total_tokens = prompt_tokens + gen_tokens

        telemetry.record_request_success(
            start_time_perf=start_perf,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            ttft_ms=getattr(stats, "prompt_eval_ms", 0.0),
            prompt_tps=getattr(stats, "prompt_eval_tps", 0.0),
            gen_tps=getattr(stats, "eval_tps", 0.0),
        )

        return ChatCompletionResponse(
            id=completion_id,
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=gen_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=gen_tokens,
                total_tokens=total_tokens,
            ),
            session_id=body.session_id,
        )
