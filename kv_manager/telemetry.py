"""
telemetry.py
------------
KV Cache telemetry collection for Intelligent KV Manager in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class KVTelemetrySnapshot:
    kv_size_mb: float
    growth_rate_mb_per_sec: float
    compression_ratio: float
    compression_time_ms: float
    eviction_count: int
    memory_saved_mb: float
    restore_events: int
    pressure_level: str
    policy_name: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kv_size_mb": round(self.kv_size_mb, 2),
            "growth_rate_mb_per_sec": round(self.growth_rate_mb_per_sec, 2),
            "compression_ratio": round(self.compression_ratio, 2),
            "compression_time_ms": round(self.compression_time_ms, 2),
            "eviction_count": self.eviction_count,
            "memory_saved_mb": round(self.memory_saved_mb, 2),
            "restore_events": self.restore_events,
            "pressure_level": self.pressure_level,
            "policy_name": self.policy_name,
            "timestamp": self.timestamp,
        }


class KVTelemetryCollector:
    """
    Collects real-time telemetry metrics for KV Cache behavior.
    """

    def __init__(self, max_snapshots: int = 1000) -> None:
        self.max_snapshots = max_snapshots
        self._snapshots: List[KVTelemetrySnapshot] = []

    def record_snapshot(self, snapshot: KVTelemetrySnapshot) -> None:
        """Add a telemetry snapshot to history."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def get_latest(self) -> Optional[KVTelemetrySnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def get_history(self) -> List[KVTelemetrySnapshot]:
        return list(self._snapshots)

    def clear(self) -> None:
        self._snapshots.clear()
