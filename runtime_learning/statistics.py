"""
statistics.py
-------------
Learning statistics tracking for Runtime Learning in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LearningStatistics:
    """System-wide summary of runtime learning performance metrics."""
    total_executions: int = 0
    total_successful: int = 0
    total_failed: int = 0
    unique_models_learned: int = 0
    unique_hardware_learned: int = 0
    avg_throughput_gain_pct: float = 0.0
    regressions_detected: int = 0
    database_size_bytes: int = 0
    last_learning_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_executions": self.total_executions,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "unique_models_learned": self.unique_models_learned,
            "unique_hardware_learned": self.unique_hardware_learned,
            "avg_throughput_gain_pct": round(self.avg_throughput_gain_pct, 2),
            "regressions_detected": self.regressions_detected,
            "database_size_bytes": self.database_size_bytes,
            "last_learning_update": self.last_learning_update,
        }
