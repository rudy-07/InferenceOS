"""
config.py
---------
Server Configuration dataclass and settings loader for InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class ServerConfig:
    """
    Configuration options for InferenceOS Server.
    """
    host: str = "0.0.0.0"
    port: int = 11434
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None
    auth_api_keys: List[str] = field(default_factory=list)
    
    max_sessions: int = 100
    max_concurrent_requests: int = 16
    max_models: int = 10
    
    idle_timeout_sec: float = 3600.0       # Auto-cleanup idle sessions after 1 hour
    session_timeout_sec: float = 86400.0   # Hard session expiration 24 hours
    auto_unload_minutes: float = 30.0      # Auto unload model weights if idle
    
    vram_budget_mb: Optional[float] = None
    ram_budget_mb: Optional[float] = None
    
    backend: Optional[str] = None  # None = AUTO, or vulkan, cuda, cpu
    log_level: str = "INFO"
    enable_json_logging: bool = True
    enable_telemetry: bool = True
    
    default_context_length: int = 4096
    default_threads: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "tls_cert": self.tls_cert,
            "tls_key": self.tls_key,
            "auth_enabled": len(self.auth_api_keys) > 0,
            "max_sessions": self.max_sessions,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_models": self.max_models,
            "idle_timeout_sec": self.idle_timeout_sec,
            "session_timeout_sec": self.session_timeout_sec,
            "backend": self.backend or "AUTO",
            "log_level": self.log_level,
            "enable_telemetry": self.enable_telemetry,
        }
