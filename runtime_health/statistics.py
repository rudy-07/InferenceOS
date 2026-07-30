"""
statistics.py
-------------
Health status transition statistics for Runtime Health Monitor in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HealthStatistics:
    total_observations: int = 0
    excellent_count: int = 0
    good_count: int = 0
    warning_count: int = 0
    critical_count: int = 0
    failed_count: int = 0
    last_status: str = "Good"
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_observations": self.total_observations,
            "excellent_count": self.excellent_count,
            "good_count": self.good_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "failed_count": self.failed_count,
            "last_status": self.last_status,
            "last_update": self.last_update,
        }
