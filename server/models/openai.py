"""
openai.py
---------
OpenAI-compatible request and response Pydantic models for InferenceOS Server.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common / Chat Message Objects
# ---------------------------------------------------------------------------

class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def text_content(self) -> str:
        """Helper to extract raw text content cleanly."""
        if isinstance(self.content, str):
            return self.content or ""
        elif isinstance(self.content, list):
            parts = []
            for part in self.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return ""


# ---------------------------------------------------------------------------
# Response Format / Tool Definitions
# ---------------------------------------------------------------------------

class ResponseFormat(BaseModel):
    type: str = "text"  # "text" or "json_object"


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    session_id: Optional[str] = None  # InferenceOS session tracking extension
    
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # OpenAI alternate param name
    
    # Reasoning & system overrides
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    user: Optional[str] = None

    def get_max_tokens(self) -> int:
        return self.max_tokens or self.max_completion_tokens or 1024


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = "fp_inferenceos_v1"
    session_id: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = "fp_inferenceos_v1"
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Text Completions (Legacy)
# ---------------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    session_id: Optional[str] = None
    
    stream: bool = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = 512
    echo: bool = False
    logprobs: Optional[int] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Embeddings (Placeholder)
# ---------------------------------------------------------------------------

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Models List Endpoint
# ---------------------------------------------------------------------------

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "inferenceos"
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "registered"  # "loaded" or "registered"
    size_bytes: int = 0
    backend: str = "auto"
    context_length: int = 4096


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]
