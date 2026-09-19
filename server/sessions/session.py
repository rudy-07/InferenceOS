"""
session.py
----------
Stateful conversation session abstraction for InferenceOS Server.
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from server.models.openai import ChatMessage


@dataclass
class Session:
    """
    Stateful session maintaining chat context, runtime overrides, statistics,
    and placement details across multiple turns without model reloads.
    """
    session_id: str
    model: str
    messages: List[ChatMessage] = field(default_factory=list)
    token_count: int = 0
    kv_cache_ref: Optional[str] = None
    placement_info: Dict[str, Any] = field(default_factory=dict)
    runtime_stats: Dict[str, Any] = field(default_factory=dict)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    prompt_history: List[str] = field(default_factory=list)
    reasoning_stats: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_access_time = time.time()

    def add_message(self, message: ChatMessage) -> None:
        """Append message to chat history."""
        self.messages.append(message)
        self.touch()

    def add_user_prompt(self, text: str) -> None:
        """Track user prompt in prompt history."""
        self.prompt_history.append(text)
        self.add_message(ChatMessage(role="user", content=text))

    def add_assistant_response(self, text: str) -> None:
        """Track assistant response in history."""
        self.add_message(ChatMessage(role="assistant", content=text))

    def reset_history(self, system_prompt: Optional[str] = None) -> None:
        """Clear conversation history, optionally preserving system prompt."""
        self.messages.clear()
        self.prompt_history.clear()
        self.token_count = 0
        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))
        self.touch()

    def update_stats(self, stats_dict: Dict[str, Any], new_tokens: int = 0) -> None:
        """Update aggregate session runtime statistics."""
        self.runtime_stats.update(stats_dict)
        self.token_count += new_tokens
        if "eval_tps" in stats_dict:
            self.reasoning_stats["last_eval_tps"] = stats_dict["eval_tps"]
        if "prompt_eval_tps" in stats_dict:
            self.reasoning_stats["last_prompt_tps"] = stats_dict["prompt_eval_tps"]
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "messages": [m.model_dump() for m in self.messages],
            "token_count": self.token_count,
            "kv_cache_ref": self.kv_cache_ref,
            "placement_info": self.placement_info,
            "runtime_stats": self.runtime_stats,
            "config_overrides": self.config_overrides,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "prompt_history": self.prompt_history,
            "reasoning_stats": self.reasoning_stats,
            "memory_usage": self.memory_usage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        """Deserialize session from dictionary."""
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            model=data["model"],
            messages=messages,
            token_count=data.get("token_count", 0),
            kv_cache_ref=data.get("kv_cache_ref"),
            placement_info=data.get("placement_info", {}),
            runtime_stats=data.get("runtime_stats", {}),
            config_overrides=data.get("config_overrides", {}),
            creation_time=data.get("creation_time", time.time()),
            last_access_time=data.get("last_access_time", time.time()),
            prompt_history=data.get("prompt_history", []),
            reasoning_stats=data.get("reasoning_stats", {}),
            memory_usage=data.get("memory_usage", {}),
        )
