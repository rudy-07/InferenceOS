"""
interfaces.py
-------------
Data structures and interface definitions for Runtime Knowledge Base (RKB) in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HardwareFingerprint:
    gpu_name: str = "GPU"
    vram_gb: float = 8.0
    ram_gb: float = 16.0
    cpu_name: str = "CPU"
    backend: str = "vulkan"
    vendor: str = "AMD/NVIDIA"
    architecture: str = "x86_64"
    driver_version: str = "latest"
    os_name: str = "Windows"

    def to_string(self) -> str:
        return f"{self.gpu_name}_{int(self.vram_gb)}GB_{self.backend}"


@dataclass
class ModelFingerprint:
    model_name: str = "Qwen3-4B"
    param_count_b: float = 4.0
    quantization: str = "Q4_K_M"
    architecture: str = "llama"
    max_context: int = 32768

    def to_string(self) -> str:
        return f"{self.model_name}_{self.quantization}"


@dataclass
class ExecutionRecord:
    record_id: str
    hardware_fp: HardwareFingerprint
    model_fp: ModelFingerprint
    backend: str
    gpu_layers: int
    microbatch_size: int
    context_length: int
    memory_used_mb: float
    eval_tps: float
    ttft_ms: float
    latency_ms: float
    gpu_utilization_pct: float
    health_status: str
    scheduler_decisions: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "hardware_fp": self.hardware_fp.to_string(),
            "model_fp": self.model_fp.to_string(),
            "backend": self.backend,
            "gpu_layers": self.gpu_layers,
            "microbatch_size": self.microbatch_size,
            "context_length": self.context_length,
            "memory_used_mb": round(self.memory_used_mb, 1),
            "eval_tps": round(self.eval_tps, 2),
            "ttft_ms": round(self.ttft_ms, 1),
            "latency_ms": round(self.latency_ms, 1),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "health_status": self.health_status,
            "success": self.success,
            "timestamp": self.timestamp,
        }


@dataclass
class KnowledgeRecommendation:
    item_name: str
    recommended_value: Any
    confidence_pct: float
    execution_count: int
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_name": self.item_name,
            "recommended_value": self.recommended_value,
            "confidence_pct": round(self.confidence_pct, 1),
            "execution_count": self.execution_count,
            "reasoning": self.reasoning,
        }


@dataclass
class RKBConfig:
    enabled: bool = True
    storage_dir: str = "~/.inferenceos/knowledge"
    verbose: bool = False
