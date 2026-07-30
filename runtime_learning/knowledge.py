"""
knowledge.py
------------
Structured knowledge base data models for hardware, model, and workload intelligence in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HardwareKnowledge:
    """Hardware-specific learned knowledge."""
    hardware_fingerprint: str
    gpu_name: str
    total_vram_mb: float
    safe_vram_mb: float
    optimal_backend: str
    optimal_microbatch: int
    optimal_safety_margin_mb: float
    max_stable_context: int
    execution_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_fingerprint": self.hardware_fingerprint,
            "gpu_name": self.gpu_name,
            "total_vram_mb": self.total_vram_mb,
            "safe_vram_mb": self.safe_vram_mb,
            "optimal_backend": self.optimal_backend,
            "optimal_microbatch": self.optimal_microbatch,
            "optimal_safety_margin_mb": self.optimal_safety_margin_mb,
            "max_stable_context": self.max_stable_context,
            "execution_count": self.execution_count,
            "last_updated": self.last_updated,
        }


@dataclass
class ModelKnowledge:
    """Model-specific learned knowledge."""
    model_fingerprint: str
    model_name: str
    architecture: str
    optimal_gpu_layers: int
    avg_prompt_tps: float
    avg_eval_tps: float
    avg_ttft_ms: float
    execution_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "optimal_gpu_layers": self.optimal_gpu_layers,
            "avg_prompt_tps": round(self.avg_prompt_tps, 2),
            "avg_eval_tps": round(self.avg_eval_tps, 2),
            "avg_ttft_ms": round(self.avg_ttft_ms, 2),
            "execution_count": self.execution_count,
            "last_updated": self.last_updated,
        }


@dataclass
class WorkloadKnowledge:
    """Workload-specific combined knowledge (Hardware x Model x Context Bucket)."""
    workload_key: str
    hardware_fingerprint: str
    model_fingerprint: str
    context_bucket: int
    optimal_microbatch: int
    optimal_gpu_layers: int
    optimal_context: int
    expected_prompt_tps: float
    expected_eval_tps: float
    expected_ttft_ms: float
    expected_vram_mb: float
    expected_ram_mb: float
    confidence_score: float
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "hardware_fingerprint": self.hardware_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "context_bucket": self.context_bucket,
            "optimal_microbatch": self.optimal_microbatch,
            "optimal_gpu_layers": self.optimal_gpu_layers,
            "optimal_context": self.optimal_context,
            "expected_prompt_tps": round(self.expected_prompt_tps, 2),
            "expected_eval_tps": round(self.expected_eval_tps, 2),
            "expected_ttft_ms": round(self.expected_ttft_ms, 2),
            "expected_vram_mb": round(self.expected_vram_mb, 2),
            "expected_ram_mb": round(self.expected_ram_mb, 2),
            "confidence_score": round(self.confidence_score, 3),
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_updated": self.last_updated,
        }
