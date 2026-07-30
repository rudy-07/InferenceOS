"""
statistics.py
-------------
KV Cache statistics metrics for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class KVStatistics:
    """Summary metrics of KV Cache operations."""
    total_evaluations: int = 0
    total_memory_saved_mb: float = 0.0
    total_evictions: int = 0
    avg_compression_ratio: float = 1.0
    active_policy: str = "Adaptive"
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "total_memory_saved_mb": round(self.total_memory_saved_mb, 2),
            "total_evictions": self.total_evictions,
            "avg_compression_ratio": round(self.avg_compression_ratio, 2),
            "active_policy": self.active_policy,
            "last_update": self.last_update,
        }
