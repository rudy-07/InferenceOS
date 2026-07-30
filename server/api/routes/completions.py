"""
completions.py
--------------
OpenAI-compatible POST /v1/completions route handler.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, List, Optional
from fastapi import APIRouter, Request
from server.models.openai import CompletionRequest, CompletionResponse, CompletionChoice, ChatMessage, UsageInfo
from server.runtime_adapter.adapter import get_runtime_adapter
from server.queue.request_queue import get_queue_manager
from server.metrics.telemetry import get_telemetry_collector

router = APIRouter()


@router.post("/v1/completions")
async def create_completion(request: Request, body: CompletionRequest):
    adapter = get_runtime_adapter()
    queue_mgr = get_queue_manager()
    telemetry = get_telemetry_collector()

    start_perf = telemetry.record_request_start()
    prompt_str = body.prompt if isinstance(body.prompt, str) else "\n".join(body.prompt)
    messages = [ChatMessage(role="user", content=prompt_str)]

    request_overrides = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "seed": body.seed,
        "max_tokens": body.max_tokens or 512,
    }

    loop = asyncio.get_running_loop()

    async def run_sync_job():
        return await loop.run_in_executor(
            None,
            lambda: adapter.execute_chat(
                model_query=body.model,
                messages=messages,
                request_overrides=request_overrides,
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
    prompt_tokens = getattr(stats, "prompt_tokens", 0)
    gen_tokens = getattr(stats, "tokens_generated", 0)

    telemetry.record_request_success(
        start_time_perf=start_perf,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        ttft_ms=getattr(stats, "prompt_eval_ms", 0.0),
        prompt_tps=getattr(stats, "prompt_eval_tps", 0.0),
        gen_tps=getattr(stats, "eval_tps", 0.0),
    )

    cid = f"cmpl-{uuid.uuid4().hex[:12]}"
    return CompletionResponse(
        id=cid,
        model=body.model,
        choices=[
            CompletionChoice(
                index=0,
                text=gen_text,
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
        ),
    )
