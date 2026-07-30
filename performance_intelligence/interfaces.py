"""
interfaces.py
-------------
Data structures and interface definitions for Performance Intelligence Engine (PIE) in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceBaseline:
    avg_prompt_tps: float = 180.0
    avg_eval_tps: float = 50.0
    avg_ttft_ms: float = 330.0
    avg_latency_ms: float = 1200.0
    avg_gpu_utilization: float = 90.0
    avg_vram_mb: float = 4500.0
    sample_count: int = 0


@dataclass
class PerformanceScore:
    overall_score: int = 94
    tps_score: float = 95.0
    latency_score: float = 92.0
    memory_score: float = 94.0
    stability_score: float = 95.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "tps_score": round(self.tps_score, 1),
            "latency_score": round(self.latency_score, 1),
            "memory_score": round(self.memory_score, 1),
            "stability_score": round(self.stability_score, 1),
        }


@dataclass
class RegressionReport:
    is_regression: bool = False
    metric_name: str = "Generation TPS"
    old_value: float = 51.4
    new_value: float = 51.4
    change_pct: float = 0.0
    confidence_pct: float = 95.0
    reasoning: List[str] = field(default_factory=list)


@dataclass
class RootCauseAnalysis:
    regression_detected: bool = False
    probable_cause: str = "None"
    explanation: str = "Performance is operating within baseline boundaries."
    severity: str = "INFO"


@dataclass
class PIERecommendation:
    action_name: str = "Maintain Strategy"
    reasoning: str = "Current optimization remains optimal."
    confidence_pct: float = 95.0


@dataclass
class PIEConfig:
    enabled: bool = True
    storage_dir: str = "~/.inferenceos/performance"
    verbose: bool = False
