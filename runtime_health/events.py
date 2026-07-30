"""
events.py
---------
Health event models for Runtime Health Monitor in InferenceOS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HealthEvent:
    event_type: str = "GenericHealthEvent"
    severity: str = "INFO"  # "INFO", "WARNING", "CRITICAL"
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
        }


@dataclass
class HighMemoryPressureEvent(HealthEvent):
    event_type: str = "HighMemoryPressure"
    severity: str = "WARNING"
    vram_used_pct: float = 0.0


@dataclass
class ThermalThrottlingEvent(HealthEvent):
    event_type: str = "ThermalThrottling"
    severity: str = "CRITICAL"
    temperature_c: float = 0.0


@dataclass
class PerformanceRegressionEvent(HealthEvent):
    event_type: str = "PerformanceRegression"
    severity: str = "WARNING"
    drop_pct: float = 0.0
