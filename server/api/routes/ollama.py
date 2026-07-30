"""
ollama.py
---------
Ollama native API endpoints (/api/tags, /api/version, /api/chat, /api/generate, /api/show)
providing exact Ollama NDJSON streaming format for Chatbox and Open-WebUI.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from server.runtime_adapter.adapter import get_runtime_adapter
from server.queue.request_queue import get_queue_manager
from server.models.openai import ChatMessage
from server.metrics.telemetry import get_telemetry_collector

router = APIRouter()


@router.get("/api/tags")
def list_ollama_tags():
    """Ollama-compatible GET /api/tags model discovery route."""
    adapter = get_runtime_adapter()
    reg_models = adapter.get_registered_models()

    models_list = []
    for m in reg_models:
        name = m.get("nickname", "unknown")
        models_list.append({
            "name": name,
            "model": name,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size": m.get("size_bytes", 0),
            "digest": f"sha256:{hash(name) & 0xffffffffffffffff:016x}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        })

    return {"models": models_list}


@router.get("/api/version")
def get_ollama_version():
    return {"version": "0.1.48"}


class OllamaChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


async def stream_ollama_chat_ndjson(
    request: Request,
    model: str,
    token_queue: asyncio.Queue[Optional[str]],
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[str, None]:
    """Async generator producing newline-delimited JSON (NDJSON) for Ollama clients."""
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        while True:
            if await request.is_disconnected():
                if cancel_event:
                    cancel_event.set()
                break

            try:
                token = await asyncio.wait_for(token_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if token is None:
                token_queue.task_done()
                break

            chunk = {
                "model": model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": token},
                "done": False,
            }
            yield json.dumps(chunk) + "\n"
            token_queue.task_done()

        # Send final completion chunk
        final_chunk = {
            "model": model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
        yield json.dumps(final_chunk) + "\n"

    except asyncio.CancelledError:
        if cancel_event:
            cancel_event.set()
        raise


@router.post("/api/chat")
async def ollama_chat(request: Request, body: OllamaChatRequest):
    """Native Ollama /api/chat route handling both NDJSON streaming and JSON responses."""
    adapter = get_runtime_adapter()
    queue_mgr = get_queue_manager()
    telemetry = get_telemetry_collector()

    start_perf = telemetry.record_request_start()
    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    
    is_stream = body.stream if body.stream is not None else True

    if is_stream:
        token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        cancel_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def sync_on_token(token: str) -> None:
            if not cancel_event.is_set():
                loop.call_soon_threadsafe(token_queue.put_nowait, token)

        async def run_async_inference():
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: adapter.execute_chat(
                        model_query=body.model,
                        messages=chat_messages,
                        request_overrides=body.options or {},
                        session=None,
                        on_token=sync_on_token,
                    )
                )
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

        asyncio.create_task(
            queue_mgr.enqueue_and_execute(
                model_name=body.model,
                execution_func=run_async_inference,
            )
        )

        return StreamingResponse(
            stream_ollama_chat_ndjson(
                request=request,
                model=body.model,
                token_queue=token_queue,
                cancel_event=cancel_event,
            ),
            media_type="application/x-ndjson",
        )

    else:
        loop = asyncio.get_running_loop()

        async def run_sync_job():
            return await loop.run_in_executor(
                None,
                lambda: adapter.execute_chat(
                    model_query=body.model,
                    messages=chat_messages,
                    request_overrides=body.options or {},
                    session=None,
                    on_token=None,
                )
            )

        result = await queue_mgr.enqueue_and_execute(
            model_name=body.model,
            execution_func=run_sync_job,
        )

        gen_text = result.generated_text or ""
        stats = result.stats

        telemetry.record_request_success(
            start_time_perf=start_perf,
            prompt_tokens=getattr(stats, "prompt_tokens", 0),
            gen_tokens=getattr(stats, "tokens_generated", 0),
            ttft_ms=getattr(stats, "prompt_eval_ms", 0.0),
            prompt_tps=getattr(stats, "prompt_eval_tps", 0.0),
            gen_tps=getattr(stats, "eval_tps", 0.0),
        )

        return {
            "model": body.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": gen_text},
            "done": True,
        }
