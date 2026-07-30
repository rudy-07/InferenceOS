"""
monitor.py
----------
Runtime Health Monitor facade and observation engine for InferenceOS.

Continuously observes runtime behavior, GPU utilization, temperature, memory pressure, and stability.
Exposes structured health metrics without directly modifying runtime state.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .alerts import AlertEngine
from .events import HealthEvent
from .metrics import CPUMetrics, GPUMetrics, InferenceMetrics, RAMMetrics
from .sensors import HealthSensorSuite
from .statistics import HealthStatistics

logger = logging.getLogger("InferenceOS.RuntimeHealth.Monitor")


class HealthStatus(str, Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    WARNING = "Warning"
    CRITICAL = "Critical"
    FAILED = "Failed"

    def __str__(self) -> str:
        return self.value


@dataclass
class RuntimeHealth:
    """
    Structured observation of current runtime health and performance metrics.

    Attributes
    ----------
    status : str
        Health status ("Excellent", "Good", "Warning", "Critical", "Failed").
    gpu_utilization_pct : float
        GPU utilization percentage.
    gpu_memory_gb : float
        GPU VRAM consumed in GB.
    temperature_c : float
        GPU temperature in Celsius.
    memory_pressure : str
        Memory pressure level ("Low", "Medium", "High", "Critical").
    performance_stability : str
        Performance state ("Stable", "Slight Degradation", "Unstable").
    warnings : List[str]
        List of active warning messages.
    timestamp : float
        Epoch timestamp when health snapshot was taken.
    """

    status: str = "Good"
    gpu_utilization_pct: float = 0.0
    gpu_memory_gb: float = 0.0
    temperature_c: float = 45.0
    memory_pressure: str = "Low"
    performance_stability: str = "Stable"
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert health snapshot to a JSON-serializable dictionary."""
        return {
            "status": self.status,
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "gpu_memory_gb": round(self.gpu_memory_gb, 2),
            "temperature_c": round(self.temperature_c, 1),
            "memory_pressure": self.memory_pressure,
            "performance_stability": self.performance_stability,
            "warnings": list(self.warnings),
            "timestamp": self.timestamp,
        }

    def format_cli_output(self) -> str:
        """Format health metrics into the exact CLI representation required by InferenceOS."""
        lines = [
            "Runtime Health",
            "GPU Utilization",
            f"  {int(round(self.gpu_utilization_pct))}%",
            "RAM / VRAM",
            f"  {self.gpu_memory_gb:.1f} GB",
            "Pressure",
            f"  {self.memory_pressure}",
            "Temperature",
            f"  {int(round(self.temperature_c))}°C",
            "Status",
            f"  {self.status}",
        ]
        return "\n".join(lines)


class RuntimeHealthMonitor:
    """
    Main facade class for observing system health without modifying runtime state.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.sensors = HealthSensorSuite()
        self.alert_engine = AlertEngine()
        self.stats = HealthStatistics()

    def observe_health(
        self,
        hw_profile: Optional[Dict[str, Any]] = None,
        stats: Optional[Any] = None,
    ) -> RuntimeHealth:
        """
        Sample sensors and return a structured RuntimeHealth object.
        """
        gpu = self.sensors.sample_gpu(hw_profile)
        cpu = self.sensors.sample_cpu(hw_profile)
        ram = self.sensors.sample_ram(hw_profile)
        inf = self.sensors.sample_inference(stats)

        alerts = self.alert_engine.evaluate_alerts(gpu, cpu, ram)
        warnings = [a.message for a in alerts]

        # Determine overall status
        if gpu.is_throttling or gpu.pressure_level == "Critical":
            status = HealthStatus.CRITICAL
        elif gpu.pressure_level == "High" or warnings:
            status = HealthStatus.WARNING
        elif gpu.utilization_pct > 80.0 and gpu.temp_c < 75.0:
            status = HealthStatus.EXCELLENT
        else:
            status = HealthStatus.GOOD

        # Update Statistics
        self.stats.total_observations += 1
        if status == HealthStatus.EXCELLENT:
            self.stats.excellent_count += 1
        elif status == HealthStatus.GOOD:
            self.stats.good_count += 1
        elif status == HealthStatus.WARNING:
            self.stats.warning_count += 1
        elif status == HealthStatus.CRITICAL:
            self.stats.critical_count += 1
        self.stats.last_status = status.value
        self.stats.last_update = time.time()

        health = RuntimeHealth(
            status=status.value,
            gpu_utilization_pct=gpu.utilization_pct,
            gpu_memory_gb=gpu.vram_used_mb / 1024.0,
            temperature_c=gpu.temp_c,
            memory_pressure=gpu.pressure_level,
            performance_stability="Stable" if not warnings else "Slight Degradation",
            warnings=warnings,
        )

        if self.verbose:
            print(health.format_cli_output())

        return health
