"""
alerts.py
---------
Alert engine for Runtime Health Monitor in InferenceOS.
"""
from __future__ import annotations

from typing import List
from .events import HealthEvent, HighMemoryPressureEvent, ThermalThrottlingEvent
from .metrics import CPUMetrics, GPUMetrics, RAMMetrics


class AlertEngine:
    """
    Evaluates hardware and system metrics to generate structured health events and alerts.
    """

    def evaluate_alerts(
        self,
        gpu: GPUMetrics,
        cpu: CPUMetrics,
        ram: RAMMetrics,
    ) -> List[HealthEvent]:
        """Generate health events for metric threshold violations."""
        events: List[HealthEvent] = []

        if gpu.is_throttling or gpu.temp_c > 85.0:
            events.append(
                ThermalThrottlingEvent(
                    event_type="ThermalThrottling",
                    severity="CRITICAL",
                    message=f"GPU thermal throttling detected ({gpu.temp_c:.1f}°C > 85.0°C).",
                    temperature_c=gpu.temp_c,
                )
            )

        if gpu.pressure_level in ("High", "Critical"):
            v_pct = (gpu.vram_used_mb / max(1.0, gpu.vram_total_mb)) * 100.0
            events.append(
                HighMemoryPressureEvent(
                    event_type="HighMemoryPressure",
                    severity="WARNING" if gpu.pressure_level == "High" else "CRITICAL",
                    message=f"Elevated GPU VRAM pressure ({v_pct:.1f}% used).",
                    vram_used_pct=v_pct,
                )
            )

        return events
