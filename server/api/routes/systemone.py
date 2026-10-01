"""
systemone.py
------------
Native /v1/systemone and /v1/decision API route handlers.
Drop-in replacement for TypeSafe Jev and jev-ultrafast System 1 models.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from server.runtime_adapter.adapter import get_runtime_adapter

router = APIRouter(tags=["System 1 Decision Engine"])


class SystemOneRequest(BaseModel):
    model: str = Field(..., description="Model nickname or path (e.g. 'laya:v14s')")
    state: Union[Dict[str, Any], str] = Field(default="", description="Task state, document, page DOM, or context text")
    questions: Dict[str, Any] = Field(default_factory=dict, description="Typed questions dict (choice, score, noul)")
    max_options: Optional[int] = Field(default=60, description="Option chunking threshold for coarse-to-fine evaluation")
    escalate: Optional[bool] = Field(default=False, description="Enable automatic System 2 teacher escalation on low confidence")
    escalate_tau: Optional[float] = Field(default=0.70, description="Confidence threshold below which to escalate to System 2")
    system2_model: Optional[str] = Field(default=None, description="System 2 teacher model nickname or path")
    system2_base_url: Optional[str] = Field(default=None, description="Optional remote OpenAI-compatible base URL for System 2")
    dagger_log_path: Optional[str] = Field(default=None, description="Optional path to append DAgger distillation cases")


@router.get("/v1/systemone")
async def systemone_status():
    """Health and status check for System 1 endpoint."""
    adapter = get_runtime_adapter()
    s1_models = [m for m in adapter.get_registered_models() if m.get("format") == "system1"]
    return {
        "status": "active",
        "engine": "InferenceOS System 1 Runtime",
        "registered_system1_models": [m["nickname"] for m in s1_models],
        "loaded_models_count": adapter.get_loaded_models_count(),
    }


@router.post("/v1/systemone")
@router.post("/v1/decision")
async def execute_systemone_decision(request: Request, body: SystemOneRequest):
    """
    Execute single forward-pass decision inference.
    Returns calibrated probabilities, chosen candidates, and latency.
    """
    adapter = get_runtime_adapter()

    # Allow HTTP headers to override escalation parameters
    escalate = body.escalate
    header_esc = request.headers.get("x-system1-escalate")
    if header_esc is not None:
        escalate = header_esc.strip().lower() in ("1", "true", "yes")

    escalate_tau = body.escalate_tau or 0.70
    header_tau = request.headers.get("x-system1-tau")
    if header_tau is not None:
        try:
            escalate_tau = float(header_tau)
        except ValueError:
            pass

    try:
        result = adapter.execute_systemone(
            model_query=body.model,
            state=body.state,
            questions=body.questions,
            max_options=body.max_options or 60,
            escalate=escalate,
            escalate_tau=escalate_tau,
            system2_model=body.system2_model,
            system2_base_url=body.system2_base_url,
            dagger_log_path=body.dagger_log_path,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"System 1 execution error: {e}")


class SystemOneToolRequest(BaseModel):
    model: str = Field(..., description="System 1 model nickname or path")
    state: Union[Dict[str, Any], str] = Field(default="", description="Task context or DOM state")
    options: Union[Dict[str, str], list] = Field(..., description="Options to evaluate and select from")
    goal: Optional[str] = Field(default="Select optimal action", description="Decision goal or instruction")


@router.post("/v1/systemone/tool")
async def execute_systemone_tool_endpoint(body: SystemOneToolRequest):
    """
    Direction B: System 2 -> System 1 Decision Tool.
    Fast reflex evaluation across wide action/candidate spaces in ~20ms.
    """
    adapter = get_runtime_adapter()
    try:
        return adapter.execute_systemone_tool(
            model_query=body.model,
            state=body.state,
            options=body.options,
            goal=body.goal or "Select optimal action",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"System 1 tool error: {e}")


class SystemOneGuardrailRequest(BaseModel):
    model: str = Field(..., description="System 1 model nickname or path")
    proposed_action: str = Field(..., description="Proposed action or command to evaluate")
    context: Optional[str] = Field(default=None, description="Optional task environment or context")
    safe_threshold: Optional[float] = Field(default=0.85, description="Safety threshold [0.0 - 1.0]")


@router.post("/v1/systemone/guardrail")
async def execute_systemone_guardrail_endpoint(body: SystemOneGuardrailRequest):
    """
    Direction B: System 2 -> System 1 Safety Guardrail.
    Fast 15ms noul probability check verifying actions before execution.
    """
    adapter = get_runtime_adapter()
    try:
        return adapter.execute_systemone_guardrail(
            model_query=body.model,
            proposed_action=body.proposed_action,
            context=body.context,
            safe_threshold=body.safe_threshold or 0.85,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"System 1 guardrail error: {e}")
