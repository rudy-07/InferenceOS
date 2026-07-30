"""
metrics.py
----------
Metric data models for Runtime Health Monitor in InferenceOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GPUMetrics:
    gpu_name: str = "GPU"
    utilization_pct: float = 0.0
    idle_time_pct: float = 100.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 8192.0
    temp_c: float = 45.0
    clock_mhz: int = 1500
    pressure_level: str = "Low"
    is_throttling: bool = False


@dataclass
class CPUMetrics:
    utilization_pct: float = 0.0
    thread_utilization_pct: float = 0.0
    clock_mhz: int = 3200
    num_cores: int = 8


@dataclass
class RAMMetrics:
    ram_used_gb: float = 4.0
    ram_total_gb: float = 16.0
    swap_used_gb: float = 0.0
    ram_pressure_level: str = "Low"


@dataclass
class InferenceMetrics:
    prompt_tps: float = 0.0
    eval_tps: float = 0.0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    context_usage_pct: float = 0.0
    microbatch_utilization: float = 0.0
    kv_size_mb: float = 0.0
