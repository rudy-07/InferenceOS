"""
sessions.py
-----------
InferenceOS Session Management REST API endpoints.
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from server.models.session_models import (
    CreateSessionRequest,
    SessionResponse,
    SessionDetailResponse,
    SessionListResponse,
    LoadSessionRequest,
    SaveSessionResponse,
)
from server.sessions.session_manager import get_session_manager

router = APIRouter()


@router.post("/v1/sessions", response_model=SessionResponse)
def create_session(body: CreateSessionRequest):
    mgr = get_session_manager()
    session = mgr.create_session(
        model=body.model,
        session_id=body.session_id,
        system_prompt=body.system_prompt,
        config_overrides=body.config_overrides,
        initial_messages=body.initial_messages,
    )
    return SessionResponse(
        session_id=session.session_id,
        model=session.model,
        status="active",
        created_at=session.creation_time,
        last_accessed_at=session.last_access_time,
    )


@router.get("/v1/sessions", response_model=SessionListResponse)
def list_sessions():
    mgr = get_session_manager()
    sessions = mgr.list_sessions()
    data = [
        SessionResponse(
            session_id=s.session_id,
            model=s.model,
            status="active",
            created_at=s.creation_time,
            last_accessed_at=s.last_access_time,
        )
        for s in sessions
    ]
    return SessionListResponse(data=data, total=len(data))


@router.get("/v1/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session_info(session_id: str):
    mgr = get_session_manager()
    session = mgr.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return SessionDetailResponse(
        session_id=session.session_id,
        model=session.model,
        messages=session.messages,
        token_count=session.token_count,
        kv_cache_ref=session.kv_cache_ref,
        placement_info=session.placement_info,
        runtime_stats=session.runtime_stats,
        config_overrides=session.config_overrides,
        creation_time=session.creation_time,
        last_access_time=session.last_access_time,
        prompt_history=session.prompt_history,
        reasoning_stats=session.reasoning_stats,
        memory_usage=session.memory_usage,
    )


@router.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str):
    mgr = get_session_manager()
    success = mgr.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return {"session_id": session_id, "status": "deleted"}


@router.post("/v1/sessions/{session_id}/reset")
def reset_session(session_id: str, system_prompt: Optional[str] = None):
    mgr = get_session_manager()
    session = mgr.reset_session(session_id, system_prompt=system_prompt)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return {"session_id": session_id, "status": "reset"}


@router.post("/v1/sessions/{session_id}/save", response_model=SaveSessionResponse)
def save_session(session_id: str, file_path: Optional[str] = None):
    mgr = get_session_manager()
    saved_path = mgr.save_session(session_id, file_path=file_path)
    if not saved_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to save session '{session_id}'.",
        )
    return SaveSessionResponse(
        session_id=session_id,
        status="saved",
        saved_path=str(saved_path),
    )


@router.post("/v1/sessions/load", response_model=SessionResponse)
def load_session(body: LoadSessionRequest):
    mgr = get_session_manager()
    target = body.file_path or body.session_id
    if not target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify file_path or session_id to load.",
        )
    session = mgr.load_session(target)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not load session from '{target}'.",
        )
    return SessionResponse(
        session_id=session.session_id,
        model=session.model,
        status="loaded",
        created_at=session.creation_time,
        last_accessed_at=session.last_access_time,
    )
