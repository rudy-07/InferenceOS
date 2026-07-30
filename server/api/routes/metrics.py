"""
metrics.py
----------
GET /metrics route handler exposing JSON and Prometheus formatted metrics.
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Request, Response
from server.metrics.telemetry import get_telemetry_collector
from server.sessions.session_manager import get_session_manager
from server.runtime_adapter.adapter import get_runtime_adapter

router = APIRouter()


@router.get("/metrics")
def get_metrics(request: Request, format: Optional[str] = None):
    telemetry = get_telemetry_collector()
    session_mgr = get_session_manager()
    adapter = get_runtime_adapter()

    active_sessions = len(session_mgr.list_sessions())
    loaded_models = adapter.get_loaded_models_count()

    accept = request.headers.get("accept", "")
    if format == "prometheus" or "text/plain" in accept:
        prom_text = telemetry.format_prometheus_metrics(
            active_sessions=active_sessions,
            loaded_models=loaded_models,
        )
        return Response(content=prom_text, media_type="text/plain; version=0.0.4")

    return telemetry.get_metrics_summary(
        active_sessions=active_sessions,
        loaded_models=loaded_models,
    )
