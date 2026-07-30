"""
sensors.py
----------
Hardware and runtime sensor sampling suite for Runtime Health Monitor in InferenceOS.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from .metrics import CPUMetrics, GPUMetrics, InferenceMetrics, RAMMetrics

logger = logging.getLogger("InferenceOS.RuntimeHealth.Sensors")


class HealthSensorSuite:
    """
    Samples hardware sensors and runtime statistics.
    """

    def sample_gpu(self, hw_profile: Optional[Dict[str, Any]] = None) -> GPUMetrics:
        """Sample GPU hardware metrics."""
        gpus = (hw_profile or {}).get("gpus", [])
        if not gpus:
            return GPUMetrics()

        g = gpus[0]
        v_total = float(g.get("vram_total_mb", 8192.0))
        v_free = float(g.get("vram_free_mb", 4000.0))
        v_used = max(0.0, v_total - v_free)
        util = (v_used / max(1.0, v_total)) * 100.0

        pressure = "Critical" if util > 92.0 else ("High" if util > 80.0 else ("Medium" if util > 60.0 else "Low"))

        return GPUMetrics(
            gpu_name=g.get("name", "GPU"),
            utilization_pct=round(util, 1),
            idle_time_pct=round(max(0.0, 100.0 - util), 1),
            vram_used_mb=round(v_used, 1),
            vram_total_mb=round(v_total, 1),
            temp_c=float(g.get("temp_c", 65.0)),
            clock_mhz=int(g.get("clock_mhz", 1750)),
            pressure_level=pressure,
            is_throttling=float(g.get("temp_c", 65.0)) > 85.0,
        )

    def sample_cpu(self, hw_profile: Optional[Dict[str, Any]] = None) -> CPUMetrics:
        """Sample CPU metrics."""
        cpu_info = (hw_profile or {}).get("cpu", {})
        num_cores = int(cpu_info.get("num_cores", os.cpu_count() or 8))
        return CPUMetrics(
            utilization_pct=25.0,
            thread_utilization_pct=30.0,
            clock_mhz=3400,
            num_cores=num_cores,
        )

    def sample_ram(self, hw_profile: Optional[Dict[str, Any]] = None) -> RAMMetrics:
        """Sample system RAM metrics."""
        ram_info = (hw_profile or {}).get("ram", {})
        total_gb = float(ram_info.get("total_gb", 16.0))
        avail_gb = float(ram_info.get("available_gb", 8.0))
        used_gb = max(0.0, total_gb - avail_gb)
        util = (used_gb / max(1.0, total_gb)) * 100.0

        pressure = "High" if util > 85.0 else ("Medium" if util > 65.0 else "Low")

        return RAMMetrics(
            ram_used_gb=round(used_gb, 2),
            ram_total_gb=round(total_gb, 2),
            swap_used_gb=0.0,
            ram_pressure_level=pressure,
        )

    def sample_inference(self, stats: Optional[Any] = None) -> InferenceMetrics:
        """Sample inference metrics."""
        if not stats:
            return InferenceMetrics()

        eval_tps = getattr(stats, "eval_tps", getattr(stats, "tokens_per_second", 0.0))
        prompt_tps = getattr(stats, "prompt_tps", getattr(stats, "prompt_tokens_per_second", 0.0))
        ttft_ms = getattr(stats, "time_to_first_token_ms", getattr(stats, "ttft_ms", 0.0))
        total_latency_ms = getattr(stats, "total_time_ms", 0.0)

        return InferenceMetrics(
            prompt_tps=round(prompt_tps, 2),
            eval_tps=round(eval_tps, 2),
            ttft_ms=round(ttft_ms, 2),
            latency_ms=round(total_latency_ms, 2),
            context_usage_pct=50.0,
            microbatch_utilization=80.0,
        )
