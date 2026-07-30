"""
history.py
----------
Telemetry execution history recorder and store for Runtime Learning in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionRecord:
    """Detailed record of a single inference execution pass."""
    record_id: str
    hardware_fingerprint: str
    model_fingerprint: str
    model_name: str
    gpu_name: str
    backend: str
    n_gpu_layers: int
    n_cpu_layers: int
    n_igpu_layers: int
    microbatch_size: int
    context_length: int
    thread_count: int
    vram_used_mb: float
    ram_used_mb: float
    prompt_tps: float
    eval_tps: float
    ttft_ms: float
    total_latency_ms: float
    gpu_utilization_pct: float
    cpu_utilization_pct: float
    memory_pressure_level: str
    scheduler_decisions: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    duration_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "gpu_name": self.gpu_name,
            "backend": self.backend,
            "n_gpu_layers": self.n_gpu_layers,
            "n_cpu_layers": self.n_cpu_layers,
            "n_igpu_layers": self.n_igpu_layers,
            "microbatch_size": self.microbatch_size,
            "context_length": self.context_length,
            "thread_count": self.thread_count,
            "vram_used_mb": round(self.vram_used_mb, 2),
            "ram_used_mb": round(self.ram_used_mb, 2),
            "prompt_tps": round(self.prompt_tps, 2),
            "eval_tps": round(self.eval_tps, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "cpu_utilization_pct": round(self.cpu_utilization_pct, 1),
            "memory_pressure_level": self.memory_pressure_level,
            "scheduler_decisions": self.scheduler_decisions,
            "warnings": self.warnings,
            "success": self.success,
            "duration_sec": round(self.duration_sec, 3),
            "timestamp": self.timestamp,
        }


class ExecutionHistoryStore:
    """In-memory and query manager for historical execution records."""

    def __init__(self, max_records: int = 5000) -> None:
        self.max_records = max_records
        self._records: List[ExecutionRecord] = []

    def add_record(self, record: ExecutionRecord) -> None:
        """Add execution record to store."""
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records.pop(0)

    def query(
        self,
        hardware_fingerprint: Optional[str] = None,
        model_fingerprint: Optional[str] = None,
        min_context: Optional[int] = None,
        max_context: Optional[int] = None,
        limit: int = 100,
    ) -> List[ExecutionRecord]:
        """Query execution records matching filters."""
        results: List[ExecutionRecord] = []
        for r in reversed(self._records):
            if hardware_fingerprint and r.hardware_fingerprint != hardware_fingerprint:
                continue
            if model_fingerprint and r.model_fingerprint != model_fingerprint:
                continue
            if min_context and r.context_length < min_context:
                continue
            if max_context and r.context_length > max_context:
                continue
            results.append(r)
            if len(results) >= limit:
                break
        return results

    def clear(self) -> None:
        self._records.clear()

    def count(self) -> int:
        return len(self._records)
