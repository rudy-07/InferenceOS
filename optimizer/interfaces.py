"""
interfaces.py
-------------
Data structures and interface definitions for Automatic Performance Optimizer (APO) in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelFingerprintData:
    sha256_hash: str
    model_name: str = "Qwen3-4B"
    param_count_b: float = 4.0
    quantization: str = "Q4_K_M"
    n_layers: int = 32
    hidden_dim: int = 4096
    vocab_size: int = 152064
    n_heads: int = 32
    context_capability: int = 32768
    gguf_version: int = 3

    def get_fingerprint_hash(self) -> str:
        """Returns short stable hash prefix."""
        return self.sha256_hash[:12] if self.sha256_hash else f"{self.model_name}_{self.quantization}"


@dataclass
class HardwareFingerprintData:
    gpu_name: str = "GPU"
    vram_gb: float = 8.0
    cpu_name: str = "CPU"
    ram_gb: float = 16.0
    backend: str = "vulkan"
    driver_version: str = "latest"
    os_name: str = "Windows"
    inferenceos_version: str = "1.0.0"

    def get_fingerprint_hash(self) -> str:
        return f"{self.gpu_name}_{int(self.vram_gb)}GB_{self.backend}"


@dataclass
class OptimizationProfileKey:
    model_hash: str
    gpu_name: str


@dataclass
class CandidateConfig:
    gpu_layers: int = 32
    microbatch_size: int = 512
    context_length: int = 4096
    memory_strategy: str = "balanced"
    thread_count: int = 8
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_layers": self.gpu_layers,
            "microbatch_size": self.microbatch_size,
            "context_length": self.context_length,
            "memory_strategy": self.memory_strategy,
            "thread_count": self.thread_count,
            "score": round(self.score, 2),
        }


@dataclass
class OptimizationProfile:
    model_name: str
    model_hash: str
    gpu_name: str
    best_candidate: CandidateConfig
    expected_tps: float
    expected_ttft_ms: float
    expected_latency_ms: float
    expected_memory_mb: float
    confidence_pct: float = 95.0
    goal: str = "Balanced"
    version: int = 1
    is_valid: bool = True
    reasoning: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "gpu_name": self.gpu_name,
            "best_candidate": self.best_candidate.to_dict(),
            "expected_tps": round(self.expected_tps, 1),
            "expected_ttft_ms": round(self.expected_ttft_ms, 1),
            "expected_latency_ms": round(self.expected_latency_ms, 1),
            "expected_memory_mb": round(self.expected_memory_mb, 1),
            "confidence_pct": round(self.confidence_pct, 1),
            "goal": self.goal,
            "version": self.version,
            "is_valid": self.is_valid,
            "reasoning": list(self.reasoning),
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """Format details into exact verbose CLI representation required by InferenceOS."""
        lines = [
            "Automatic Performance Optimizer",
            "Model",
            f"  {self.model_name}",
            "Fingerprint",
            f"  {self.model_hash[:8]}...",
            "Hardware",
            f"  {self.gpu_name}",
            "Optimization Goal",
            f"  {self.goal}",
            "Testing",
            "Placement",
            f"  {self.best_candidate.gpu_layers} GPU",
            "Microbatch",
            f"  {self.best_candidate.microbatch_size}",
            "Threads",
            f"  {self.best_candidate.thread_count}",
            "Result",
            f"  {self.expected_tps:.1f} TPS",
            "Validation",
            "  Passed",
            "Confidence",
            f"  {int(round(self.confidence_pct))}%",
            "Optimization Profile Saved",
        ]
        return "\n".join(lines)


@dataclass
class APOConfig:
    enabled: bool = True
    goal: str = "Balanced"  # Balanced | Max Throughput | Lowest Latency | Lowest Memory | Max Stability
    force: bool = False
    storage_dir: str = "~/.inferenceos/optimization"
    verbose: bool = False
