"""
interfaces.py
-------------
Core data structures, decisions, and interface definitions for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KVState:
    """Current state of KV cache memory and configuration."""
    current_kv_mb: float = 0.0
    max_kv_mb: float = 0.0
    token_count: int = 0
    compression_mode: str = "adaptive"
    eviction_policy: str = "adaptive"
    pressure_level: str = "Low"
    is_compressed: bool = False
    memory_saved_mb: float = 0.0


@dataclass
class KVDecision:
    """
    Result of an Intelligent KV Manager scheduling decision.

    Attributes
    ----------
    current_kv_gb : float
        Current KV cache memory footprint in GB.
    pressure_level : str
        Memory pressure level ("Idle", "Low", "Medium", "High", "Critical").
    compression_strategy : str
        Selected compression mode ("Disabled", "Lossless", "Balanced", "Aggressive", "Adaptive").
    memory_saved_mb : float
        Total memory saved through KV compression and eviction (MB).
    eviction_action : str
        Summary of eviction action taken ("None", "LRU Eviction (128 tokens)", etc.).
    policy_name : str
        Active policy name ("Adaptive Policy", "Balanced Policy", etc.).
    reasoning : List[str]
        Human-readable decision explanation.
    timestamp : float
        Epoch timestamp when decision was computed.
    """

    current_kv_gb: float
    pressure_level: str
    compression_strategy: str
    memory_saved_mb: float
    eviction_action: str
    policy_name: str
    reasoning: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to JSON-serializable dictionary."""
        return {
            "current_kv_gb": round(self.current_kv_gb, 3),
            "pressure_level": self.pressure_level,
            "compression_strategy": self.compression_strategy,
            "memory_saved_mb": round(self.memory_saved_mb, 2),
            "eviction_action": self.eviction_action,
            "policy_name": self.policy_name,
            "reasoning": list(self.reasoning),
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """
        Format decision details into the clean CLI representation required by InferenceOS.
        """
        saved_str = f"{int(round(self.memory_saved_mb))} MB" if self.memory_saved_mb > 0 else "0 MB"
        lines = [
            "Intelligent KV Manager",
            "Current KV",
            f"  {self.current_kv_gb:.1f} GB",
            "Pressure",
            f"  {self.pressure_level}",
            "Compression",
            f"  {self.compression_strategy}",
            "Memory Saved",
            f"  {saved_str}",
            "Eviction",
            f"  {self.eviction_action}",
            "Policy",
            f"  {self.policy_name}",
            "Reason",
        ]
        if self.reasoning:
            for r in self.reasoning:
                lines.append(f"  {r}")
        else:
            lines.append("  Memory pressure within normal operational boundaries.")

        return "\n".join(lines)


@dataclass
class KVPolicyConfig:
    """Configuration options for KV Manager policies."""
    enabled: bool = True
    compression_enabled: bool = True
    eviction_enabled: bool = True
    compression_mode: str = "adaptive"  # disabled | lossless | balanced | aggressive | adaptive
    eviction_policy: str = "adaptive"    # lru | fifo | lfu | adaptive
    compression_threshold_pct: float = 80.0
    eviction_threshold_pct: float = 95.0
    max_kv_mb: float = 0.0  # 0 = dynamic
    verbose: bool = False
