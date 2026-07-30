"""
baseline.py
-----------
Baseline engine for Performance Intelligence Engine in InferenceOS.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .interfaces import PerformanceBaseline


class BaselineEngine:
    """
    Computes statistically meaningful performance baselines with outlier rejection.
    """

    def compute_baseline(self, history_records: List[Dict[str, Any]]) -> PerformanceBaseline:
        if not history_records:
            return PerformanceBaseline()

        # Reject outliers (records with eval_tps <= 0)
        valid = [r for r in history_records if r.get("eval_tps", 0.0) > 0.0]
        if not valid:
            return PerformanceBaseline()

        n = len(valid)
        prompt_tps = sum(r.get("prompt_tps", 180.0) for r in valid) / float(n)
        eval_tps = sum(r.get("eval_tps", 50.0) for r in valid) / float(n)
        ttft = sum(r.get("ttft_ms", 330.0) for r in valid) / float(n)
        lat = sum(r.get("latency_ms", 1200.0) for r in valid) / float(n)
        gpu_util = sum(r.get("gpu_utilization", 90.0) for r in valid) / float(n)
        vram = sum(r.get("vram_used_mb", 4500.0) for r in valid) / float(n)

        return PerformanceBaseline(
            avg_prompt_tps=round(prompt_tps, 1),
            avg_eval_tps=round(eval_tps, 1),
            avg_ttft_ms=round(ttft, 1),
            avg_latency_ms=round(lat, 1),
            avg_gpu_utilization=round(gpu_util, 1),
            avg_vram_mb=round(vram, 1),
            sample_count=n,
        )
