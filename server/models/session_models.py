"""
session_models.py
------------------
Session Management API Pydantic models for InferenceOS.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from server.models.openai import ChatMessage


class CreateSessionRequest(BaseModel):
    model: str
    session_id: Optional[str] = None  # Optional custom ID
    system_prompt: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    initial_messages: Optional[List[ChatMessage]] = None


class SessionResponse(BaseModel):
    session_id: str
    model: str
    status: str = "active"
    created_at: float
    last_accessed_at: float


class SessionDetailResponse(BaseModel):
    session_id: str
    model: str
    messages: List[ChatMessage]
    token_count: int
    kv_cache_ref: Optional[str]
    placement_info: Dict[str, Any]
    runtime_stats: Dict[str, Any]
    config_overrides: Dict[str, Any]
    creation_time: float
    last_access_time: float
    prompt_history: List[str]
    reasoning_stats: Dict[str, Any]
    memory_usage: Dict[str, Any]


class SessionListResponse(BaseModel):
    object: str = "list"
    data: List[SessionResponse]
    total: int


class LoadSessionRequest(BaseModel):
    file_path: Optional[str] = None
    session_id: Optional[str] = None


class SaveSessionResponse(BaseModel):
    session_id: str
    status: str = "saved"
    saved_path: str
